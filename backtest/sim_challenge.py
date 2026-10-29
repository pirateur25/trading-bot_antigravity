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


# ---------------------------------------------------------------- Simulation d'une tentative
def simulate_attempt(
    trades: pd.DataFrame,
    capital: float,
    risk_pct: float,
    phase_rules: List[Dict],
    start_idx: int = 0,
    max_days: int = 365,
    worst_case: bool = True,
) -> Tuple[str, int, int]:
    """
    Simule une tentative de challenge FTMO.

    trades : DataFrame trié par entry_time, colonnes [entry_time, exit_time, r, side]
    capital : capital initial de la phase
    risk_pct : risque par trade (%)
    phase_rules : liste de dicts {"target_pct", "dd_daily_pct", "dd_total_pct", "min_days"}
    start_idx : indice du premier trade à inclure dans cette tentative
    max_days : plafond de durée par phase
    worst_case : True = équité flottante = balance - 1R pendant qu'un trade est ouvert

    Retourne : (résultat, nb_jours_total, phase_atteinte)
    résultat = "success" | "breach_daily" | "breach_total" | "timeout"
    """
    current_capital = capital
    total_days = 0

    for phase_idx, rules in enumerate(phase_rules):
        target = rules["target_pct"] / 100.0
        dd_daily = rules["dd_daily_pct"] / 100.0
        dd_total = rules["dd_total_pct"] / 100.0
        min_days = rules["min_days"]

        phase_capital = current_capital
        balance = phase_capital
        day_start_balance = phase_capital  # équité de début du jour FTMO courant
        days_traded = set()
        phase_days = 0
        current_day = None

        idx = start_idx
        n = len(trades)

        while idx < n:
            trade = trades.iloc[idx]
            entry_t = trade["entry_time"]
            exit_t = trade["exit_time"]

            # Changement de jour FTMO ?
            ftmo_day = _ftmo_day(entry_t)
            if current_day is None:
                current_day = ftmo_day
            elif ftmo_day != current_day:
                current_day = ftmo_day
                day_start_balance = balance  # reset quotidien
                phase_days += 1
                if phase_days >= max_days:
                    return "timeout", total_days + phase_days, phase_idx

            days_traded.add(ftmo_day)

            risk_cash = balance * risk_pct / 100.0

            # Vérification pire cas intra-bougie
            if worst_case:
                floated_eq = balance - risk_cash  # pire cas : le SL est touché
                if floated_eq < phase_capital * (1 - dd_total):
                    return "breach_total", total_days + phase_days, phase_idx
                if floated_eq < day_start_balance * (1 - dd_daily):
                    return "breach_daily", total_days + phase_days, phase_idx

            # Application du trade
            pnl_r = float(trade["r"])
            balance += pnl_r * risk_cash

            # Vérification après sortie
            if balance < phase_capital * (1 - dd_total):
                return "breach_total", total_days + phase_days, phase_idx
            if balance < day_start_balance * (1 - dd_daily):
                return "breach_daily", total_days + phase_days, phase_idx

            # Phase validée ?
            if balance >= phase_capital * (1 + target) and len(days_traded) >= min_days:
                current_capital = balance
                total_days += phase_days
                start_idx = idx + 1
                break

            idx += 1
        else:
            # Plus de trades : timeout
            return "timeout", total_days + phase_days, phase_idx

    return "success", total_days, len(phase_rules)


# ---------------------------------------------------------------- Simulation Monte-Carlo avec départs glissants
def simulate_challenge(
    trades: pd.DataFrame,
    capital: float = 10000.0,
    risk_pct: float = 0.75,
    mode: str = "2phases",
    sliding_days: int = 3,
    max_days_per_phase: int = 365,
    worst_case: bool = True,
    n_min_trades: int = 5,
) -> Dict:
    """
    Lance la simulation avec départs glissants (tous les `sliding_days` jours à partir du premier trade).

    Retourne un dict avec :
      success_pct, breach_daily_pct, breach_total_pct, timeout_pct,
      median_days, mean_days, n_simulations, details (liste des résultats)
    """
    if trades.empty:
        return {"success_pct": 0, "breach_daily_pct": 0, "breach_total_pct": 0,
                "timeout_pct": 0, "median_days": 0, "mean_days": 0, "n_simulations": 0, "details": []}

    rules = FTMO_RULES[mode]["phases"]
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    # Génère des points de départ (tous les `sliding_days` jours)
    min_t = trades["entry_time"].min()
    max_t = trades["entry_time"].max()
    starts = pd.date_range(min_t, max_t, freq=f"{sliding_days}D", tz="UTC" if min_t.tzinfo else None)

    results = []
    for start_ts in starts:
        # Trouve le premier trade >= start_ts
        idx0 = int(np.searchsorted(trades["entry_time"].values.astype("int64"),
                                    int(pd.Timestamp(start_ts).value)))
        if idx0 >= len(trades) - n_min_trades:
            continue
        res, days, phase = simulate_attempt(
            trades, capital, risk_pct, rules, idx0, max_days_per_phase, worst_case
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


# ---------------------------------------------------------------- Simulation combinée A+B
def simulate_combined(
    trades_ote: pd.DataFrame,
    trades_fvg: pd.DataFrame,
    capital: float = 10000.0,
    risk_ote: float = 0.5,
    risk_fvg: float = 0.75,
    mode: str = "2phases",
    sliding_days: int = 3,
    max_days_per_phase: int = 365,
) -> Dict:
    """
    Combine les trades OTE (risque risk_ote) et FVG (risque risk_fvg) et simule le challenge.
    Les deux stratégies tournent en parallèle ; un seul trade à la fois au global
    (si les deux signaux arrivent au même moment, OTE a priorité).
    """
    if trades_ote.empty and trades_fvg.empty:
        return simulate_challenge(pd.DataFrame(), capital, risk_ote, mode, sliding_days, max_days_per_phase)

    # Marque le risque de chaque trade
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

    combined = pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    # Exclut les chevauchements (un seul trade à la fois)
    filtered = []
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    for _, row in combined.iterrows():
        et = row["entry_time"]
        if hasattr(et, "tzinfo") and et.tzinfo is None:
            et = et.tz_localize("UTC")
        if et >= busy_until:
            xt = row["exit_time"]
            if hasattr(xt, "tzinfo") and xt.tzinfo is None:
                xt = xt.tz_localize("UTC")
            busy_until = xt
            filtered.append(row)

    if not filtered:
        return simulate_challenge(pd.DataFrame(), capital, risk_ote, mode, sliding_days, max_days_per_phase)

    combined_filtered = pd.DataFrame(filtered).reset_index(drop=True)

    # Simule avec risque variable par trade
    rules = FTMO_RULES[mode]["phases"]
    starts = pd.date_range(
        combined_filtered["entry_time"].min(),
        combined_filtered["entry_time"].max(),
        freq=f"{sliding_days}D",
        tz="UTC",
    )

    results = []
    for start_ts in starts:
        idx0 = int(np.searchsorted(
            combined_filtered["entry_time"].values.astype("int64"),
            int(pd.Timestamp(start_ts).value)
        ))
        if idx0 >= len(combined_filtered) - 5:
            continue

        # Simulation avec risque variable
        current_capital = capital
        total_days = 0
        final_result = "timeout"
        final_days = 0

        for phase_idx, prules in enumerate(rules):
            target = prules["target_pct"] / 100.0
            dd_daily = prules["dd_daily_pct"] / 100.0
            dd_total = prules["dd_total_pct"] / 100.0
            min_days = prules["min_days"]

            phase_capital = current_capital
            balance = phase_capital
            day_start_balance = phase_capital
            days_traded = set()
            phase_days = 0
            current_day = None
            breached = False
            success = False

            for i in range(idx0, len(combined_filtered)):
                row = combined_filtered.iloc[i]
                entry_t = row["entry_time"]
                if hasattr(entry_t, "tzinfo") and entry_t.tzinfo is None:
                    entry_t = entry_t.tz_localize("UTC")

                ftmo_day = _ftmo_day(entry_t)
                if current_day is None:
                    current_day = ftmo_day
                elif ftmo_day != current_day:
                    current_day = ftmo_day
                    day_start_balance = balance
                    phase_days += 1
                    if phase_days >= max_days_per_phase:
                        final_result = "timeout"
                        final_days = total_days + phase_days
                        breached = True
                        break

                days_traded.add(ftmo_day)
                risk_pct_row = float(row["_risk"])
                risk_cash = balance * risk_pct_row / 100.0

                # Pire cas intra-bougie
                floated = balance - risk_cash
                if floated < phase_capital * (1 - dd_total):
                    final_result = "breach_total"
                    final_days = total_days + phase_days
                    breached = True
                    break
                if floated < day_start_balance * (1 - dd_daily):
                    final_result = "breach_daily"
                    final_days = total_days + phase_days
                    breached = True
                    break

                balance += float(row["r"]) * risk_cash

                if balance < phase_capital * (1 - dd_total):
                    final_result = "breach_total"
                    final_days = total_days + phase_days
                    breached = True
                    break
                if balance < day_start_balance * (1 - dd_daily):
                    final_result = "breach_daily"
                    final_days = total_days + phase_days
                    breached = True
                    break

                if balance >= phase_capital * (1 + target) and len(days_traded) >= min_days:
                    current_capital = balance
                    total_days += phase_days
                    idx0 = i + 1
                    success = True
                    break

            if breached:
                break
            if not success:
                final_result = "timeout"
                final_days = total_days + phase_days
                break
            if phase_idx == len(rules) - 1:
                final_result = "success"
                final_days = total_days

        results.append({"result": final_result, "days": final_days})

    if not results:
        return {"success_pct": 0, "breach_daily_pct": 0, "breach_total_pct": 0,
                "timeout_pct": 0, "median_days": 0, "mean_days": 0, "n_simulations": 0}

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
    }


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
