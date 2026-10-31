"""
backtest/sim_challenge.py — Simulation de challenge FTMO à partir d'une liste de trades.

Entrée : DataFrame de trades avec colonnes :
  entry_time, exit_time, r, pnl (si disponible), side
  ou une equity curve (Series indexée par datetime UTC)

Méthode :
  - On rejoue les trades (ou l'equity) en faisant varier la date de départ (glissement).
  - Pour chaque départ, on simule jour par jour selon les règles FTMO.
  - On compte : réussite, échec, abandon (durée max dépassée), durée médiane.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

# ---------------------------------------------------------------- Constantes FTMO
FTMO_RULES = {
    "1phase": {
        "phases": [{"target_pct": 10.0, "dd_daily_pct": 5.0, "dd_total_pct": 10.0, "min_days": 0}],
    },
    "2phases": {
        "phases": [
            {"target_pct": 10.0, "dd_daily_pct": 5.0, "dd_total_pct": 10.0, "min_days": 4},
            {"target_pct":  5.0, "dd_daily_pct": 5.0, "dd_total_pct": 10.0, "min_days": 4},
        ],
    },
}

FTMO_DAY_CHANGE_HOUR_UTC = 22  # le "jour FTMO" change à 22h UTC


# ---------------------------------------------------------------- Construction de l'equity intra-bougie (pire cas)
def build_equity_curve(
    trades: pd.DataFrame,
    capital: float,
    risk_pct: float,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """
    Reconstruit une courbe d'équité bougie par bougie à partir des trades.
    Pour le pire cas intra-bougie : pendant qu'un trade est ouvert, on suppose que le SL est touché
    (équité = capital_avant_trade - 1R) jusqu'à la sortie effective.

    trades colonnes requises : entry_time, exit_time, r, side
    timestamps : index de toutes les bougies 5m (UTC)
    capital : capital de départ
    risk_pct : risque par trade en % du capital (pour calculer 1R en $)
    """
    eq = pd.Series(capital, index=timestamps, dtype=float)
    bal = capital
    for _, t in trades.iterrows():
        mask_open = (timestamps >= t["entry_time"]) & (timestamps < t["exit_time"])
        risk_cash = bal * risk_pct / 100.0
        # Pire cas : on suppose que l'équité chute de 1R pendant que le trade est ouvert
        eq[mask_open] = bal - risk_cash
        bal = bal + t["r"] * risk_cash
        eq[t["exit_time"] <= timestamps] = bal  # sera écrasé par le prochain trade
    return eq


def _ftmo_day(ts: pd.Timestamp) -> pd.Timestamp:
    """Retourne le début du 'jour FTMO' d'un timestamp (22h UTC la veille)."""
    t = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if t.hour < FTMO_DAY_CHANGE_HOUR_UTC:
        base = (t - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=FTMO_DAY_CHANGE_HOUR_UTC)
    else:
        base = t.normalize() + pd.Timedelta(hours=FTMO_DAY_CHANGE_HOUR_UTC)
    return base


# ---------------------------------------------------------------- Simulation d'une tentative (avec support superposition)
def _run_phases_on_df(
    df: pd.DataFrame,
    capital: float,
    phase_rules: List[Dict],
    start_ts: pd.Timestamp,
    idx0: int,
    max_days: int = 365,
    worst_case: bool = True,
) -> Tuple[str, int, int]:
    """
    Simule une tentative de challenge FTMO (1 ou 2 phases) à partir de `start_ts` (premier trade `idx0`).
    Supporte la superposition de trades (plusieurs trades ouverts simultanément sur différentes paires/stratégies) :
      - Les sorties des trades ouverts sont traitées par ordre chronologique exact (`exit_time`).
      - En mode `worst_case=True`, l'équité flottante pire cas est `balance - somme(risk_cash des trades simultanément ouverts)`.
      - La durée retournée est en vrais jours calendaires écoulés depuis `start_ts`.
    """
    total_cal_days = 0
    n = len(df)
    phase_start_ts = start_ts

    for phase_idx, rules in enumerate(phase_rules):
        target = rules["target_pct"] / 100.0
        dd_daily = rules["dd_daily_pct"] / 100.0
        dd_total = rules["dd_total_pct"] / 100.0
        min_days = rules["min_days"]

        # Sur FTMO, chaque phase démarre avec le capital initial du compte
        phase_capital = capital
        balance = phase_capital
        day_start_balance = phase_capital
        days_traded = set()
        current_day = _ftmo_day(phase_start_ts)
        open_trades = []  # liste de [exit_time, pnl_cash, risk_cash]
        phase_done = False

        def advance_day_if_needed(ts: pd.Timestamp):
            nonlocal current_day, day_start_balance
            fday = _ftmo_day(ts)
            if fday != current_day:
                current_day = fday
                day_start_balance = balance

        def close_until(ts_limit: pd.Timestamp):
            nonlocal balance, open_trades
            open_trades.sort(key=lambda x: x[0])
            while open_trades and open_trades[0][0] <= ts_limit:
                xt, pnl_c, _ = open_trades.pop(0)
                advance_day_if_needed(xt)
                cal_d = max(1, int(np.ceil((xt - phase_start_ts).total_seconds() / 86400.0)))
                if cal_d > max_days:
                    return "timeout", total_cal_days + max_days, xt
                balance += pnl_c
                if balance < phase_capital * (1 - dd_total):
                    return "breach_total", total_cal_days + cal_d, xt
                if balance < day_start_balance * (1 - dd_daily):
                    return "breach_daily", total_cal_days + cal_d, xt
                if balance >= phase_capital * (1 + target) and len(days_traded) >= min_days:
                    return "phase_success", cal_d, xt
            return None, 0, ts_limit

        idx = idx0
        while idx < n:
            row = df.iloc[idx]
            entry_t = row["entry_time"]
            exit_t = row["exit_time"]

            # 1) Clôturer tous les trades ouverts qui se terminent avant ou à entry_t
            status, d_used, ev_ts = close_until(entry_t)
            if status in ("breach_total", "breach_daily", "timeout"):
                return status, d_used, phase_idx
            if status == "phase_success":
                total_cal_days += d_used
                phase_start_ts = ev_ts
                idx0 = idx
                phase_done = True
                break

            # 2) Vérifier le délai calendaire à l'ouverture du nouveau trade
            cal_d = max(1, int(np.ceil((entry_t - phase_start_ts).total_seconds() / 86400.0)))
            if cal_d > max_days:
                return "timeout", total_cal_days + max_days, phase_idx

            advance_day_if_needed(entry_t)
            days_traded.add(current_day)

            risk_pct_row = float(row["_risk"])
            risk_cash = balance * risk_pct_row / 100.0
            pnl_cash = float(row["r"]) * risk_cash
            open_trades.append([exit_t, pnl_cash, risk_cash])

            # 3) Pire cas intra-bougie : somme des risques de tous les trades simultanément ouverts
            if worst_case:
                tot_open_risk = sum(t[2] for t in open_trades)
                floated = balance - tot_open_risk
                if floated < phase_capital * (1 - dd_total):
                    return "breach_total", total_cal_days + cal_d, phase_idx
                if floated < day_start_balance * (1 - dd_daily):
                    return "breach_daily", total_cal_days + cal_d, phase_idx

            idx += 1

        if not phase_done:
            # Clôturer les derniers trades encore ouverts en fin d'historique
            status, d_used, ev_ts = close_until(pd.Timestamp.max.tz_localize("UTC"))
            if status in ("breach_total", "breach_daily", "timeout"):
                return status, d_used, phase_idx
            if status == "phase_success":
                total_cal_days += d_used
                phase_start_ts = ev_ts
                idx0 = n
                phase_done = True
            else:
                cal_end = max(1, int(np.ceil((df.iloc[-1]["exit_time"] - phase_start_ts).total_seconds() / 86400.0)))
                return "timeout", total_cal_days + min(cal_end, max_days), phase_idx

    return "success", total_cal_days, len(phase_rules)


def simulate_attempt(
    trades: pd.DataFrame,
    capital: float,
    risk_pct: float,
    phase_rules: List[Dict],
    start_idx: int = 0,
    max_days: int = 365,
    worst_case: bool = True,
    start_ts: Optional[pd.Timestamp] = None,
) -> Tuple[str, int, int]:
    """Simule une tentative de challenge FTMO à partir de `start_idx`."""
    df = trades.copy()
    if "_risk" not in df.columns:
        df["_risk"] = risk_pct
    for col in ("entry_time", "exit_time"):
        df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ns, UTC]")
    st_ts = pd.Timestamp(start_ts).tz_convert("UTC") if start_ts is not None else df.iloc[start_idx]["entry_time"]
    return _run_phases_on_df(df, capital, phase_rules, st_ts, start_idx, max_days, worst_case)


# ---------------------------------------------------------------- Simulation Monte-Carlo avec départs glissants
def simulate_challenge(
    trades: pd.DataFrame,
    capital: float = 10000.0,
    risk_pct: float = 0.75,
    mode: str = "2phases",
    sliding_days: int = 3,
    max_days_per_phase: int = 9999,
    worst_case: bool = True,
    n_min_trades: int = 5,
) -> Dict:
    """
    Lance la simulation avec départs glissants (tous les `sliding_days` jours à partir du premier trade).
    """
    if trades.empty:
        return {"success_pct": 0, "breach_daily_pct": 0, "breach_total_pct": 0,
                "timeout_pct": 0, "median_days": 0, "mean_days": 0, "n_simulations": 0, "details": []}

    rules = FTMO_RULES[mode]["phases"]
    df = trades.copy()
    if "_risk" not in df.columns:
        df["_risk"] = risk_pct
    for col in ("entry_time", "exit_time"):
        df[col] = pd.to_datetime(df[col], utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values("entry_time").reset_index(drop=True)

    min_t = df["entry_time"].min()
    max_t = df["entry_time"].max()
    starts = pd.date_range(min_t, max_t, freq=f"{sliding_days}D", tz="UTC")
    entry_ns = df["entry_time"].values.astype("int64")

    results = []
    for start_ts in starts:
        idx0 = int(np.searchsorted(entry_ns, int(pd.Timestamp(start_ts).value)))
        if idx0 >= len(df) - n_min_trades:
            continue
        res, days, phase = _run_phases_on_df(
            df, capital, rules, start_ts, idx0, max_days_per_phase, worst_case
        )
        results.append({"result": res, "days": days, "phase_reached": phase})

    if not results:
        return {"success_pct": 0, "breach_daily_pct": 0, "breach_total_pct": 0,
                "timeout_pct": 0, "median_days": 0, "mean_days": 0, "n_simulations": 0, "details": []}

    df_r = pd.DataFrame(results)
    n = len(df_r)
    success_mask = df_r["result"] == "success"
    days_success = df_r.loc[success_mask, "days"]

    return {
        "success_pct": float(success_mask.mean() * 100),
        "breach_daily_pct": float((df_r["result"] == "breach_daily").mean() * 100),
        "breach_total_pct": float((df_r["result"] == "breach_total").mean() * 100),
        "timeout_pct": float((df_r["result"] == "timeout").mean() * 100),
        "median_days": float(days_success.median()) if len(days_success) else float("nan"),
        "mean_days": float(days_success.mean()) if len(days_success) else float("nan"),
        "n_simulations": n,
        "details": results,
    }


# ---------------------------------------------------------------- Simulation combinée A+B (superposition activée)
def simulate_combined(
    trades_ote: pd.DataFrame,
    trades_fvg: pd.DataFrame,
    capital: float = 10000.0,
    risk_ote: float = 0.5,
    risk_fvg: float = 0.75,
    mode: str = "2phases",
    sliding_days: int = 3,
    max_days_per_phase: int = 9999,
    allow_overlap: bool = True,
) -> Dict:
    """
    Combine les trades OTE (risque risk_ote) et FVG (risque risk_fvg) et simule le challenge FTMO.
    Par défaut (`allow_overlap=True`), les deux stratégies et les différentes paires peuvent superposer
    leurs trades simultanément (avec cumul du risque flottant en pire cas intra-bougie).
    """
    if trades_ote.empty and trades_fvg.empty:
        return simulate_challenge(pd.DataFrame(), capital, risk_ote, mode, sliding_days, max_days_per_phase)

    dfs = []
    if not trades_ote.empty:
        t = trades_ote.copy()
        t["_risk"] = risk_ote
        t["_strat"] = "OTE"
        dfs.append(t)
    if not trades_fvg.empty:
        t = trades_fvg.copy()
        t["_risk"] = risk_fvg
        t["_strat"] = "FVG"
        dfs.append(t)

    combined = pd.concat(dfs, ignore_index=True)
    for col in ("entry_time", "exit_time"):
        combined[col] = pd.to_datetime(combined[col], utc=True).astype("datetime64[ns, UTC]")
    combined = combined.sort_values("entry_time").reset_index(drop=True)

    if not allow_overlap:
        filtered = []
        busy_until = pd.Timestamp.min.tz_localize("UTC")
        for _, row in combined.iterrows():
            if row["entry_time"] >= busy_until:
                busy_until = row["exit_time"]
                filtered.append(row)
        if not filtered:
            return simulate_challenge(pd.DataFrame(), capital, risk_ote, mode, sliding_days, max_days_per_phase)
        combined = pd.DataFrame(filtered).reset_index(drop=True)

    return simulate_challenge(
        combined,
        capital=capital,
        risk_pct=risk_ote,
        mode=mode,
        sliding_days=sliding_days,
        max_days_per_phase=max_days_per_phase,
        worst_case=True,
    )


def format_sim_result(result: Dict, mode: str, risk: float) -> str:
    """Formate le résultat de simulation en texte lisible."""
    lines = [
        f"📊 Simulation FTMO {mode} — risque {risk}%",
        f"  N simulations : {result['n_simulations']}",
        f"  ✅ Réussite    : {result['success_pct']:.1f}%",
        f"  ❌ Breach DD quotidien : {result['breach_daily_pct']:.1f}%",
        f"  ❌ Breach DD total     : {result['breach_total_pct']:.1f}%",
        f"  ⏰ Timeout     : {result['timeout_pct']:.1f}%",
        f"  📅 Durée médiane (succès) : {result['median_days']:.0f} jours" if not np.isnan(result.get('median_days', float('nan'))) else "  📅 Aucun succès",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Test rapide
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades_ote", default="", help="CSV de trades OTE")
    ap.add_argument("--trades_fvg", default="", help="CSV de trades FVG")
    ap.add_argument("--capital",    type=float, default=10000.0)
    ap.add_argument("--risk_ote",   type=float, default=0.5)
    ap.add_argument("--risk_fvg",   type=float, default=0.75)
    ap.add_argument("--mode",       default="2phases", choices=["1phase", "2phases"])
    ap.add_argument("--sliding_days", type=int, default=3)
    a = ap.parse_args()

    trades_ote = pd.read_csv(a.trades_ote, parse_dates=["entry_time", "exit_time"]) if a.trades_ote else pd.DataFrame()
    trades_fvg = pd.read_csv(a.trades_fvg, parse_dates=["entry_time", "exit_time"]) if a.trades_fvg else pd.DataFrame()

    if not trades_ote.empty or not trades_fvg.empty:
        result = simulate_combined(trades_ote, trades_fvg, a.capital, a.risk_ote, a.risk_fvg, a.mode, a.sliding_days)
        print(format_sim_result(result, a.mode, f"OTE={a.risk_ote}%+FVG={a.risk_fvg}%"))
    else:
        print("Fournir au moins un CSV de trades avec --trades_ote ou --trades_fvg")
