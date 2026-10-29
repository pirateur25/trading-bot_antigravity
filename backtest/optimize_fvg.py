"""
backtest/optimize_fvg.py — Optimisation Optuna (TPE) de la stratégie FVG HTF + MSS.

Corrections par rapport à l'ancienne version :
  - mkdir outdir avant tout (bug du tee : results_opt n'existait pas)
  - Minimum de trades imposé (sinon score = -3 = pénalité max)
  - Sauvegarde intermédiaire toutes les N=50 trials (le job ne perd pas 30 min de calcul)
  - Reprise possible (sqlite storage : si le job est relancé, il repart de là où il s'est arrêté)
  - Limite de temps (wall_time) respectant les 6h GitHub Actions
  - Walk-forward annuel optionnel (--wf)
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
import engine_fvg as engine
import strategy_fvg as strategy

SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "XAUUSD"]

SPACE = {
    "side":          ("cat", ["short", "long", "both"]),
    "conf_tfs":      ("cat", ["5m", "15m", "5m,15m"]),
    "htf":           ("cat", ["1h", "4h", "1d", "1h,4h", "4h,1d", "1h,4h,1d"]),
    "mit_tf":        ("cat", ["1h", "4h"]),
    "mit_depth":     ("float", 0.0, 1.0),
    "mss_k":         ("int", 2, 16),
    "speed_ratio":   ("float", 0.5, 4.0),
    "speed_ref":     ("int", 0, 1),
    "trough_pivot":  ("int", 0, 1),
    "min_sweep_pct": ("float", 0.0, 0.05),
    "min_corr_pct":  ("float", 0.0, 0.5),
    "min_corr_atr":  ("float", 0.0, 5.0),
    "max_back":      ("int", 1, 10),
    "zone_pad_atr":  ("float", 0.0, 1.5),
    "entry_lvl":     ("float", 0.0, 1.0),
    "fvg_min_atr":   ("float", 0.0, 1.0),
    "sl_buf_atr":    ("float", 0.0, 2.5),
    "tp_mode":       ("cat", ["rr", "fvg"]),
    "rr":            ("float", 1.0, 6.0),
    "min_rr":        ("float", 0.5, 3.0),
    "max_rr":        ("float", 3.0, 10.0),
    "tp_edge":       ("float", 0.0, 1.0),
    "be_rr":         ("float", 0.0, 4.0),
    "ttl":           ("cat", [0, 12, 36, 96, 288]),
    "min_sl_pct":    ("float", 0.0, 0.3),
    "trend":         ("cat", ["none", "4h:200", "4h:50", "1d:50", "1d:200"]),
    "session_start": ("int", 0, 0),  # fixe à 0 (filtre session = option avancée)
    "session_end":   ("int", 24, 24),
}


def suggest(trial, space):
    p = dict(strategy.PARAMS)
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
    # Cohérence : max_rr >= min_rr
    if p.get("tp_mode") == "fvg" and p.get("max_rr", 10) < p.get("min_rr", 0.5):
        p["max_rr"] = p["min_rr"]
    return p


class Evaluator:
    def __init__(self, symbols, datadir, split, fee, risk, lev, start="2021-01-04", mid="2023-01-01", min_trades=15):
        self.symbols = symbols
        self.fee = fee
        self.risk = risk
        self.lev = lev
        self.min_trades = min_trades
        self.P = {}
        for s in symbols:
            self.P[s] = data.PairData(s, datadir)
            print(f"  {s}: {self.P[s].n} bougies 5m chargées", flush=True)
        self.split = {s: self.P[s].idx_at(split) for s in symbols}
        self.start = {s: self.P[s].idx_at(start) for s in symbols}
        self.mid   = {s: self.P[s].idx_at(mid) for s in symbols}

    def run_pair(self, s, p, a, b):
        P = self.P[s]
        sigs = strategy.signals(P, p, a, b)
        _, _, st = engine.run(P, sigs, 10000.0, self.risk, self.fee, self.lev, p["be_rr"], p["ttl"], a, b)
        return st

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
        r1 = {s: self.run_pair(s, p, self.start[s], self.mid[s]) for s in self.symbols}
        r2 = {s: self.run_pair(s, p, self.mid[s], self.split[s]) for s in self.symbols}
        return min(self.half_score(r1), self.half_score(r2))

    def evaluate_all(self, p, period="test"):
        rows = {}
        for s in self.symbols:
            P = self.P[s]
            if period == "test":
                a, b = self.split[s], P.n
            elif period == "train":
                a, b = self.start[s], self.split[s]
            else:
                a, b = self.start[s], P.n
            rows[s] = self.run_pair(s, p, a, b)
        return rows


def fmt(rows):
    return " | ".join(f"{s}: {st['return_pct']:+.1f}% DD {st['max_dd_pct']:.1f}% {st['trades']}t"
                      for s, st in rows.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--split",   default="2024-01-01")
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--fee",     type=float, default=0.005)
    ap.add_argument("--risk",    type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=30.0)
    ap.add_argument("--outdir",  default="results_opt")
    ap.add_argument("--save_every", type=int, default=50, help="Sauvegarde intermédiaire toutes les N trials")
    ap.add_argument("--wall_time", type=float, default=300.0, help="Durée max en minutes (GitHub: 340)")
    ap.add_argument("--min_trades", type=int, default=15)
    ap.add_argument("--study_name", default="fvg_optuna")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)  # FIX: mkdir avant tout (ancienne erreur tee)

    symbols = [s.strip().upper() for s in a.symbols.split(",")]
    ev = Evaluator(symbols, a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{os.path.join(a.outdir, 'study.db')}"
    try:
        study = optuna.create_study(
            study_name=a.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,  # reprise si le job a planté
            sampler=optuna.samplers.TPESampler(seed=42),
        )
    except Exception:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    best_score = float("-inf")
    best_params = None
    t_start = time.time()
    wall_ns = a.wall_time * 60

    def objective_fn(trial):
        nonlocal best_score, best_params
        elapsed = time.time() - t_start
        if elapsed > wall_ns - 120:  # 2 min de marge avant la limite
            trial.study.stop()
            return float("-inf")
        p = suggest(trial, SPACE)
        score = ev.objective(p)
        if score > best_score:
            best_score = score
            best_params = dict(p)
            # Sauvegarde intermédiaire
            json.dump(best_params, open(os.path.join(a.outdir, "best_params_fvg.json"), "w"), indent=1)
        return score

    def save_checkpoint(study, trial):
        if trial.number % a.save_every == 0 and study.best_trial is not None:
            json.dump(study.best_trial.params, open(os.path.join(a.outdir, f"checkpoint_{trial.number}.json"), "w"), indent=1)
            print(f"  [trial {trial.number}] meilleur score: {study.best_value:.3f}", flush=True)

    study.optimize(objective_fn, n_trials=a.n_trials, callbacks=[save_checkpoint],
                   catch=(Exception,), show_progress_bar=False)

    if best_params is None and len(study.trials) > 0:
        bt = study.best_trial
        best_score = bt.value
        best_params = {**dict(strategy.PARAMS), **bt.params}

    if best_params is None:
        print("Aucun trial valide.", flush=True)
        return

    print(f"\nMeilleure configuration (score train {best_score:.3f}) :")
    print(json.dumps(best_params, indent=1))
    json.dump(best_params, open(os.path.join(a.outdir, "best_params_fvg.json"), "w"), indent=1)

    for period in ("train", "test", "all"):
        rows = ev.evaluate_all(best_params, period)
        print(f"\n[{period}] " + fmt(rows))

    # Sauvegarde du log Optuna
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(a.outdir, "optuna_log_fvg.csv"), index=False)
    except Exception as e:
        print(f"Erreur export trials: {e}")


if __name__ == "__main__":
    main()
