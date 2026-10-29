"""
backtest/optimize_ote.py — Optimisation Optuna (TPE) de la stratégie OTE (ZigZag + confirmation MSS/Bulle).

Protocole :
  - TRAIN : 2021-2022 (h1) + 2023 (h2). Score = min(score_h1, score_h2).
  - TEST  : 2024-2025 (lu une seule fois à la fin).
  - Minimum de trades : min_trades par paire par moitié sinon score = -3.
  - Sauvegarde intermédiaire + reprise sqlite.
  - Limite de temps wall_time (minutes).

NOTE sur les signatures réelles de strategy_ote :
  - prepare(df)  — lit le module-global PARAMS (pas de paramètre p).
  - signal(d, i) — idem.
  L'adaptateur _OTEStrategyAdapter remplace temporairement strategy_ote.PARAMS
  avant d'appeler prepare(), ce qui est le seul moyen d'injecter les
  hyperparamètres sans modifier strategy_ote.py.
"""
import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data
import engine_ote as engine_mod
import strategy_ote as strategy

SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "XAUUSD"]

# Espace de recherche OTE (section 3 du prompt)
SPACE = {
    "side":               ("cat", ["short", "long", "both"]),
    "confirms":           ("cat", ["bubble", "mss", "bubble,mss"]),
    "conf_tfs":           ("cat", ["5m", "15m", "5m,15m", "5m,15m,1h", "15m,1h", "1h"]),
    "trend_filter":       ("int", 0, 1),
    "trend_mode":         ("cat", ["dow", "ma", "both"]),
    "zz_atr_trend":       ("float", 1.5, 8.0),
    "ma_len":             ("cat", [20, 50, 100, 200]),
    "range_tfs":          ("cat", ["4h", "1h", "4h,1h"]),
    "zz_atr_4h":          ("float", 1.5, 6.0),
    "zz_atr_1h":          ("float", 1.5, 6.0),
    "range_origin":       ("cat", ["leg", "pivot"]),
    "leg_lookback":       ("cat", [100, 300, 600]),
    "ote_lo":             ("float", 0.50, 0.75),
    "ote_hi":             ("float", 0.79, 1.00),
    "ote_react":          ("int", 0, 1),
    "ote_react_level":    ("float", 0.3, 0.6),
    "mss_fractal":        ("int", 2, 12),
    "mss_speed_ratio":    ("float", 0.5, 4.0),
    "mss_min_sweep_pct":  ("float", 0.0, 0.05),
    "mss_trough_pivot":   ("int", 0, 1),
    "bubble_min_pct_5m":  ("float", 0.0, 0.06),
    "bubble_min_pct_15m": ("float", 0.0, 0.10),
    "bubble_min_pct_1h":  ("float", 0.0, 0.20),
    "tp_fib":             ("float", 0.5, 3.5),
    "be_rr":              ("cat", [1000.0, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "sl_buf_atr":         ("float", 0.0, 2.0),
    "min_sl_pct":         ("float", 0.0, 0.4),
    "min_rr":             ("float", 0.0, 3.0),
    # session_start / session_end sont gérés via SESSION_PAIRS ci-dessous
}

# Sessions valides (start, end) — on couple start/end ensemble pour éviter
# des combinaisons incohérentes (ex. start=12, end=7).
SESSION_PAIRS = [(0, 24), (7, 12), (12, 17), (7, 17), (0, 7), (7, 22)]


def suggest(trial, space):
    p = dict(strategy.PARAMS)
    # Session : on tire un index dans SESSION_PAIRS pour garantir la cohérence
    sess_idx = trial.suggest_categorical("session_idx", list(range(len(SESSION_PAIRS))))
    p["session_start"], p["session_end"] = SESSION_PAIRS[sess_idx]

    for k, spec in space.items():
        if spec[0] == "cat":
            p[k] = trial.suggest_categorical(k, spec[1])
        elif spec[0] == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2])
        elif spec[0] == "int":
            if spec[1] == spec[2]:
                p[k] = spec[1]
            else:
                p[k] = trial.suggest_int(k, spec[1], spec[2])
    return p


class _OTEStrategyAdapter:
    """
    Adapte les hyperparamètres p pour engine_ote.run_backtest().

    strategy_ote.prepare(df) et strategy_ote.signal(d, i) lisent toutes deux
    le dictionnaire MODULE-GLOBAL strategy_ote.PARAMS. Il n'y a pas de
    paramètre p dans leurs signatures. La seule façon d'injecter un jeu de
    paramètres différent à chaque trial Optuna est de remplacer temporairement
    strategy.PARAMS avant l'appel, puis de le restaurer dans le bloc finally.

    Cette approche est thread-safe tant qu'Optuna n'utilise qu'un seul worker
    (n_jobs=1, comportement par défaut). Pour un usage multi-processus, il
    faudrait lancer chaque trial dans un sous-processus isolé.
    """
    def __init__(self, params: dict):
        self.p = params

    def prepare(self, df):
        _orig = dict(strategy.PARAMS)
        try:
            strategy.PARAMS.update(self.p)
            return strategy.prepare(df)
        finally:
            # Restaure le dictionnaire original quoi qu'il arrive
            strategy.PARAMS.clear()
            strategy.PARAMS.update(_orig)

    def signal(self, d, i):
        return strategy.signal(d, i)


def _build_df_from_pairdata(pair_data, i0: int, i1: int) -> pd.DataFrame:
    """
    Reconstruit un DataFrame 5m standard (colonnes time/open/high/low/close)
    à partir d'un PairData de data_forex, pour la tranche [i0, i1).
    """
    base = pair_data.base
    times_ns = base["open_ns"][i0:i1]
    times = pd.to_datetime(times_ns, utc=True, unit="ns")
    return pd.DataFrame({
        "time":  times,
        "open":  base["o"][i0:i1],
        "high":  base["h"][i0:i1],
        "low":   base["l"][i0:i1],
        "close": base["c"][i0:i1],
    })


def _run_ote(sym, pair_data, p, start_ts, end_ts, risk, fee_pct, lev):
    """
    Lance le backtest OTE sur la période [start_ts, end_ts).

    Paramètres
    ----------
    fee_pct : float — commission en % (ex. 0.005 → 0.005 %)
              engine_ote.run_backtest attend fee_pct en % (il divise par 100
              en interne, ligne: fee = fee_pct / 100.0).
    """
    i0 = pair_data.idx_at(start_ts)
    i1 = pair_data.idx_at(end_ts)
    if i1 <= i0:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0}

    df = _build_df_from_pairdata(pair_data, i0, i1)
    strat = _OTEStrategyAdapter(p)
    try:
        _, _, stats = engine_mod.run_backtest(
            df, strat,
            capital=10000.0,
            risk_pct=risk,
            fee_pct=fee_pct,
            max_leverage=lev,
            allow_short=True,
            be_rr=p.get("be_rr", 1000.0),
        )
        return {
            "return_pct":  stats["return_pct"],
            "max_dd_pct":  stats["max_drawdown_pct"],
            "trades":      stats["trades"],
        }
    except Exception:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0}


class EvaluatorOTE:
    def __init__(self, symbols, datadir, split, fee, risk, lev,
                 start="2021-01-04", mid="2023-01-01", min_trades=10):
        self.symbols = symbols
        # fee est fourni en % (ex. 0.005) ; on le passe tel quel à _run_ote
        # qui le transmet à engine_ote.run_backtest(fee_pct=...) (engine divise par 100).
        self.fee = fee
        self.risk = risk
        self.lev = lev
        self.min_trades = min_trades
        self.split = split
        self.start = start
        self.mid = mid
        self.P = {}
        for s in symbols:
            self.P[s] = data.PairData(s, datadir)
            print(f"  {s}: {self.P[s].n} bougies chargées", flush=True)

    def half_score(self, rows):
        scores = []
        for st in rows.values():
            if st["trades"] < self.min_trades:
                scores.append(-3.0)
            else:
                s = float(np.clip(st["return_pct"] / max(abs(st["max_dd_pct"]), 2.0), -3, 3))
                scores.append(s)
        return float(np.mean(scores))

    def objective(self, p):
        r1 = {s: _run_ote(s, self.P[s], p, self.start, self.mid,  self.risk, self.fee, self.lev)
              for s in self.symbols}
        r2 = {s: _run_ote(s, self.P[s], p, self.mid,   self.split, self.risk, self.fee, self.lev)
              for s in self.symbols}
        return min(self.half_score(r1), self.half_score(r2))

    def evaluate_test(self, p, end="2026-01-01"):
        return {s: _run_ote(s, self.P[s], p, self.split, end, self.risk, self.fee, self.lev)
                for s in self.symbols}


def fmt(rows):
    return " | ".join(
        f"{s}: {st['return_pct']:+.1f}% DD {st['max_dd_pct']:.1f}% {st['trades']}t"
        for s, st in rows.items()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",    default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--datadir",    default="data")
    ap.add_argument("--split",      default="2024-01-01")
    ap.add_argument("--n_trials",   type=int,   default=200)
    ap.add_argument("--fee",        type=float, default=0.005,
                    help="Commission en %% (ex. 0.005 = 0.005%%)")
    ap.add_argument("--risk",       type=float, default=1.0)
    ap.add_argument("--leverage",   type=float, default=30.0)
    ap.add_argument("--outdir",     default="results_opt")
    ap.add_argument("--save_every", type=int,   default=25)
    ap.add_argument("--wall_time",  type=float, default=300.0,
                    help="Durée max en minutes (GitHub Actions: 340)")
    ap.add_argument("--min_trades", type=int,   default=10)
    ap.add_argument("--study_name", default="ote_optuna")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    symbols = [s.strip().upper() for s in a.symbols.split(",")]
    ev = EvaluatorOTE(
        symbols, a.datadir, a.split,
        fee=a.fee, risk=a.risk, lev=a.leverage,
        min_trades=a.min_trades,
    )

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{os.path.join(a.outdir, 'study_ote.db')}"
    try:
        study = optuna.create_study(
            study_name=a.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
    except Exception:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

    best_score = float("-inf")
    best_params = None
    t_start = time.time()
    wall_sec = a.wall_time * 60

    def objective_fn(trial):
        nonlocal best_score, best_params
        if time.time() - t_start > wall_sec - 120:
            trial.study.stop()
            return float("-inf")
        p = suggest(trial, SPACE)
        score = ev.objective(p)
        if score > best_score:
            best_score = score
            best_params = dict(p)
            json.dump(
                best_params,
                open(os.path.join(a.outdir, "best_params_ote.json"), "w"),
                indent=1,
            )
        return score

    def save_checkpoint(study, trial):
        if trial.number % a.save_every == 0 and study.best_trial is not None:
            json.dump(
                study.best_trial.params,
                open(os.path.join(a.outdir, f"ote_checkpoint_{trial.number}.json"), "w"),
                indent=1,
            )
            print(f"  [trial {trial.number}] meilleur score OTE: {study.best_value:.3f}", flush=True)

    study.optimize(
        objective_fn,
        n_trials=a.n_trials,
        callbacks=[save_checkpoint],
        catch=(Exception,),
        show_progress_bar=False,
    )

    if best_params is None and len(study.trials) > 0:
        best_params = {**dict(strategy.PARAMS), **study.best_trial.params}
        best_score = study.best_value

    if best_params is None:
        print("Aucun trial valide.", flush=True)
        return

    print(f"\nMeilleure config OTE (score train {best_score:.3f}) :")
    print(json.dumps(best_params, indent=1))
    json.dump(
        best_params,
        open(os.path.join(a.outdir, "best_params_ote.json"), "w"),
        indent=1,
    )

    test_rows = ev.evaluate_test(best_params)
    print(f"\n[test] " + fmt(test_rows))

    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(a.outdir, "optuna_log_ote.csv"), index=False)
    except Exception as e:
        print(f"Erreur export trials: {e}")


if __name__ == "__main__":
    main()
