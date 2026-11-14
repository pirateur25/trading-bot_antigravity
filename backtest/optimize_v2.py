"""
backtest/optimize_v2.py — Optimiseur Walk-Forward Multi-Paliers (3 Paliers x 4 Folds) pour OTE 2.0, FVG 2.0 et OPR 2.0 (OTR 2.0).

Architecture anti-sur-apprentissage inspirée de CRYPTO V4 :
  - 733 000 à 883 000 bougies M5 par actif (2015-2026) -> découpage du TRAIN en 4 Folds chronologiques + 1 Holdout TEST final.
  - Optimisation séquentielle en 3 PALIERS (via `v2_common.run_staged_study`) :
      * PALIER 1 : Biais d'efficacité directionnelle H4 + Momentum M15 + Fenêtre de volatilité M15/ATR_H1 + Filtres cœur
      * PALIER 2 : Trailing Stop-Loss multi-paliers en R (`step_1r`, `step_05r`, `step_buffered`, `hybrid_30m`) + Break-Even + Cibles TP
      * PALIER 3 : Protection post-SL (`sl_cooldown_hours`), Anti-Repeat POI/Zone cassée (`block_same_poi`) & Annulation 30m sur flip
  - À chaque palier :
      * Essai #0 = réglages du palier précédent (garantit de ne jamais dégrader la référence 1.0)
      * Sélection du CENTRE DU PLATEAU de stabilité (`oc.plateau_center`)
  - Sélection automatique des actifs les plus rentables sur TRAIN uniquement (zéro fuite du TEST) et sauvegarde dans
    `results/best_params_ote2.json`, `results/best_params_fvg2.json`, `results/best_params_opr2.json` (et `best_params_otr2.json`).
"""
import argparse
import json
import os
import shutil
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data  # noqa: E402
import engine_fvg  # noqa: E402
import engine_ote  # noqa: E402
import opt_common as oc  # noqa: E402
import realism  # noqa: E402
import strategy_fvg  # noqa: E402
import strategy_opr  # noqa: E402
import strategy_ote  # noqa: E402
import v2_common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

DEFAULT_SYMBOLS = {
    "ote2": ["GBPUSD", "USDJPY", "EURUSD"],
    "fvg2": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
    "opr2": ["USA500", "US30", "NAS100"],
}

# --- Espaces de recherche discrets (Catégoriels = plateaux stables) ---
V2_COMMON_SPACE = {
    "h4_bias_mode": ("cat", ["none", "contra_block", "crypto_v4", "strict"]),
    "h4_lookback": ("cat", [6, 8, 10, 12]),
    "h4_thresh": ("cat", [0.25, 0.30, 0.35, 0.40, 0.45]),
    "m15_min_atr": ("cat", [0.0, 0.5, 0.7, 0.9, 1.1]),
    "m15_max_atr": ("cat", [0.0, 2.8, 3.4, 4.0, 4.8]),
    "trail_mode": ("cat", ["be_only", "step_1r", "step_05r", "step_buffered"]),
    "trail_check": ("cat", ["bar", "close_30m", "hybrid_30m"]),
    "sl_cooldown_hours": ("cat", [0.0, 1.0, 2.0, 3.0, 4.0]),
    "block_same_poi": ("cat", [0, 1]),
    "same_poi_tol_pct": ("cat", [0.08, 0.12, 0.15, 0.25]),
    "cancel_on_flip": ("cat", [0, 1]),
}

SPACE_OTE2 = {
    **V2_COMMON_SPACE,
    "side": ("cat", ["both", "long"]),
    "trend_filter": ("cat", [0, 1]),
    "ote_hi": ("cat", [0.706, 0.75, 0.786]),
    "be_rr": ("cat", [1.0, 1.25, 1.5, 2.0, 1000.0]),
    "tp_fib": ("cat", [1.618, 2.0, 2.5, 3.0, 3.5]),
    "sl_buf_atr": ("cat", [0.15, 0.25, 0.3407, 0.5]),
    "min_rr": ("cat", [0.1101, 0.8, 1.2, 1.5]),
    "ttl": ("cat", [48, 72, 96, 144]),
}

STAGES_OTE2 = [
    {"name": "Biais H4 + Volatilité M15 + Zone OTE", "weight": 0.42,
     "keys": ["h4_bias_mode", "h4_lookback", "h4_thresh", "m15_min_atr", "m15_max_atr", "side", "trend_filter", "ote_hi"]},
    {"name": "Trailing Multi-Paliers R + BE + TP", "weight": 0.35,
     "keys": ["trail_mode", "trail_check", "be_rr", "tp_fib", "sl_buf_atr", "min_rr"]},
    {"name": "Cooldown SL + Anti-Repeat POI + Flip 30m", "weight": 0.23,
     "keys": ["sl_cooldown_hours", "block_same_poi", "same_poi_tol_pct", "cancel_on_flip", "ttl"]},
]

SPACE_FVG2 = {
    **V2_COMMON_SPACE,
    "trend": ("cat", ["none", "4h:50", "4h:100", "4h:200", "1d:50"]),
    "htf": ("cat", ["4h", "1h,4h", "4h,1d"]),
    "conf_tfs": ("cat", ["5m", "15m", "5m,15m"]),
    "be_rr": ("cat", [0.0, 1.0, 1.25, 1.5, 2.0]),
    "tp_mode": ("cat", ["rr", "fvg"]),
    "rr": ("cat", [2.0, 2.5, 3.0, 3.5, 4.0]),
    "min_rr": ("cat", [1.5, 1.8, 2.2]),
    "sl_buf_atr": ("cat", [0.0, 0.3, 0.5, 1.0]),
    "entry_lvl": ("cat", [0.0, 0.25, 0.5]),
    "ttl": ("cat", [24, 48, 72, 96]),
}

STAGES_FVG2 = [
    {"name": "Biais H4 + Volatilité M15 + Zones HTF", "weight": 0.42,
     "keys": ["h4_bias_mode", "h4_lookback", "h4_thresh", "m15_min_atr", "m15_max_atr", "trend", "htf", "conf_tfs"]},
    {"name": "Trailing Multi-Paliers R + BE + TP", "weight": 0.35,
     "keys": ["trail_mode", "trail_check", "be_rr", "tp_mode", "rr", "min_rr", "sl_buf_atr", "entry_lvl"]},
    {"name": "Cooldown SL + Anti-Repeat Zone/POI + Flip 30m", "weight": 0.23,
     "keys": ["sl_cooldown_hours", "block_same_poi", "same_poi_tol_pct", "cancel_on_flip", "ttl"]},
]

SPACE_OPR2 = {
    **V2_COMMON_SPACE,
    "min_vix": ("cat", [12.0, 14.0, 15.0]),
    "min_opr_em": ("cat", [0.15, 0.20, 0.25]),
    "max_opr_em": ("cat", [0.50, 0.55, 0.65]),
    "min_mm_spread_atr": ("cat", [0.10, 0.15, 0.20]),
    "be_rr": ("cat", [1.0, 1.25, 1.5]),
    "rr_break": ("cat", [2.25, 2.5, 2.75, 3.0, 3.25]),
    "max_rr_tp": ("cat", [3.2, 3.6, 3.8, 4.2]),
    "rv3_tp_mode": ("cat", ["rv3_dyn", "cap_c", "ext_res"]),
    "max_fill_bars": ("cat", [4, 6, 8, 10]),
}

STAGES_OPR2 = [
    {"name": "Biais H4 + Volatilité M15 + Filtres OPR/VIX", "weight": 0.42,
     "keys": ["h4_bias_mode", "h4_lookback", "h4_thresh", "m15_min_atr", "m15_max_atr", "min_vix", "min_opr_em", "max_opr_em", "min_mm_spread_atr"]},
    {"name": "Trailing Multi-Paliers R + BE + Cibles RV3", "weight": 0.35,
     "keys": ["trail_mode", "trail_check", "be_rr", "rr_break", "max_rr_tp", "rv3_tp_mode"]},
    {"name": "Cooldown SL + Anti-Repeat POI + Flip 30m", "weight": 0.23,
     "keys": ["sl_cooldown_hours", "block_same_poi", "same_poi_tol_pct", "cancel_on_flip", "max_fill_bars"]},
]


def _load_baseline_1_0(strat_key, mod):
    fname = {"ote2": "best_params_ote.json", "fvg2": "best_params_fvg.json", "opr2": "best_params_opr.json"}[strat_key]
    p = dict(mod.PARAMS)
    p.update(v2_common.V2_DEFAULT_PARAMS)
    path = os.path.join(RESULTS, fname)
    if os.path.exists(path):
        try:
            saved = json.load(open(path))
            src = saved.get("params", saved) if isinstance(saved, dict) else {}
            for k, v in src.items():
                if k in p:
                    p[k] = v
        except Exception:
            pass
    return p


class _OTEWrap:
    def __init__(self, p):
        self.p = p

    def prepare(self, df):
        orig = dict(strategy_ote.PARAMS)
        try:
            strategy_ote.PARAMS.update(self.p)
            strategy_ote.PARAMS["_quiet"] = True
            strategy_ote.DIAG.clear()
            return strategy_ote.prepare(df)
        finally:
            strategy_ote.PARAMS.clear()
            strategy_ote.PARAMS.update(orig)

    def signal(self, d, i):
        return strategy_ote.signal(d, i)


class EvaluatorV2:
    def __init__(self, strat_key, symbols, datadir, split, fee, risk, lev, min_trades=60, n_folds=4):
        self.strat_key = strat_key
        self.symbols = symbols
        self.risk = risk
        self.min_trades = min_trades
        self.n_folds = n_folds
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {}
        self.df_ote = {}
        self.t = {}
        for s in symbols:
            oc.ensure_data(s, datadir)
            P = data.PairData(s, datadir)
            self.P[s] = P
            if strat_key == "opr2":
                strategy_opr._precompute_daily_sessions(P, "paris")
            elif strat_key == "ote2":
                b = P.base
                self.df_ote[s] = pd.DataFrame({
                    "time": pd.to_datetime(b["open_ns"], utc=True),
                    "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"],
                })
            self.t[s] = v2_common.compute_multi_fold_bounds(P.base["open_ns"], split, n_folds=n_folds, warmup_days=120)
            print(f"  [{strat_key.upper()}] {s}: {P.n:,} bougies 5m | {n_folds} folds TRAIN -> {split} | frais {self.fee[s]:.4f}%", flush=True)

    def run_pair_range(self, s, p, a, b):
        P = self.P[s]
        if self.strat_key == "fvg2":
            sigs = strategy_fvg.signals(P, p, a, b)
            tr, _, st = engine_fvg.run(P, sigs, 10000.0, self.risk, self.fee[s], self.lev[s], p["be_rr"], p["ttl"], a, b, p=p)
            st["r_long"], st["r_short"] = oc.side_r(tr)
            return st
        elif self.strat_key == "opr2":
            tr, eq = strategy_opr.run(P, capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s], leverage=self.lev[s], p=p, i0=a, i1=b)
            st = engine_fvg.stats(tr, eq, 10000.0)
            st["r_long"], st["r_short"] = oc.side_r(tr)
            return st
        else:
            df_full = self.df_ote[s].iloc[:b]
            w = _OTEWrap(p)
            df_prep = w.prepare(df_full).iloc[a:b]
            tr, eq, st_ote = engine_ote.run_backtest(
                df_prep, w, capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s],
                slippage_pct=0.0, max_leverage=self.lev[s], allow_short=True,
                be_rr=p.get("be_rr", 1000.0), pre_prepared=True, p=p,
            )
            st = {
                "return_pct": st_ote.get("return_pct", 0.0),
                "max_dd_pct": abs(st_ote.get("max_drawdown_pct", 0.0)),
                "trades": st_ote.get("trades", 0),
                "win_pct": st_ote.get("win_rate_pct", 0.0),
                "avg_r": st_ote.get("avg_r", 0.0),
                "pf": st_ote.get("profit_factor", 0.0),
            }
            st["r_long"], st["r_short"] = oc.side_r(tr)
            return st

    def evaluate(self, p, period="train"):
        out = {}
        for s in self.symbols:
            bounds = self.t[s]
            a, b = (bounds["start"], bounds["split"]) if period == "train" else (bounds["split"], bounds["end"])
            out[s] = self.run_pair_range(s, p, a, b)
        return out

    def objective(self, p):
        f_full = self.evaluate(p, "train")
        n_sym = max(1, len(self.symbols))
        pair_target = float(self.min_trades) / n_sym
        min_tot = max(20, int(self.min_trades * 0.7))
        tot_tr = sum(r.get("trades", 0) for r in f_full.values())
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(max(1, min_tot))) * 5.0

        fold_rows_list = []
        for k in range(self.n_folds):
            fr = {}
            for s in self.symbols:
                fa, fb = self.t[s]["folds"][k]
                fr[s] = self.run_pair_range(s, p, fa, fb)
            fold_rows_list.append(fr)

        return v2_common.score_multi_fold(f_full, fold_rows_list, pair_target=pair_target, min_tot=min_tot, dd_limit=16.0)


def main():
    ap = argparse.ArgumentParser(description="Optimiseur Multi-Paliers + Walk-Forward pour OTE 2.0, FVG 2.0 et OPR 2.0 (OTR 2.0)")
    ap.add_argument("--strategy", default="all", help="ote2 | fvg2 | opr2 | otr2 | all")
    ap.add_argument("--pairs", default="", help="Actifs séparés par virgule (vide = actifs candidats par défaut)")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--trials", type=int, default=600, help="Nombre d'essais Optuna par stratégie (répartis sur les 3 paliers)")
    ap.add_argument("--split", default="2024-07-01", help="Début de la période Holdout TEST")
    ap.add_argument("--n_folds", type=int, default=4, help="Nombre de plis temporels Walk-Forward sur le TRAIN")
    ap.add_argument("--max_hours", type=float, default=4.5, help="Budget temps max total en heures")
    ap.add_argument("--fee", type=float, default=-1.0)
    ap.add_argument("--risk", type=float, default=0.75)
    ap.add_argument("--leverage", type=float, default=0.0)
    ap.add_argument("--min_trades", type=int, default=75)
    ap.add_argument("--outdir", default="results_opt_v2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    strat_arg = args.strategy.strip().lower().replace("otr2", "opr2")
    strat_list = ["ote2", "fvg2", "opr2"] if strat_arg == "all" else [
        s.strip().lower().replace("otr2", "opr2") for s in strat_arg.split(",") if s.strip()
    ]

    per_strat_min = (float(args.max_hours) * 60.0) / max(1, len(strat_list))

    for sk in strat_list:
        if sk not in ("ote2", "fvg2", "opr2"):
            raise ValueError(f"Stratégie inconnue '{sk}' (attendu : ote2, fvg2, opr2, otr2 ou all)")

        syms = [x.strip().upper() for x in args.pairs.split(",") if x.strip()] if args.pairs else DEFAULT_SYMBOLS[sk]
        mod = {"ote2": strategy_ote, "fvg2": strategy_fvg, "opr2": strategy_opr}[sk]
        space = {"ote2": SPACE_OTE2, "fvg2": SPACE_FVG2, "opr2": SPACE_OPR2}[sk]
        stages = {"ote2": STAGES_OTE2, "fvg2": STAGES_FVG2, "opr2": STAGES_OPR2}[sk]
        base_v1 = _load_baseline_1_0(sk, mod)
        min_tr = args.min_trades if sk != "ote2" else max(40, args.min_trades // 2)

        ev = EvaluatorV2(sk, syms, args.datadir, args.split, args.fee, args.risk, args.leverage, min_trades=min_tr, n_folds=args.n_folds)

        sub_args = argparse.Namespace(
            outdir=args.outdir,
            n_trials=args.trials,
            wall_time=per_strat_min,
            split=args.split,
            save_every=25,
        )
        best_fname = f"best_params_{sk}.json"
        sum_fname = f"summary_{sk}.txt"

        best_p = v2_common.run_staged_study(
            label=sk,
            ev=ev,
            stages=stages,
            full_space=space,
            initial_frozen={},
            baseline_v1=base_v1,
            a=sub_args,
            best_file=best_fname,
            summary_file=sum_fname,
        )

        # Copie directe dans results/best_params_{ote2,fvg2,opr2}.json pour utilisation immédiate par run_backtest.py
        dst = os.path.join(RESULTS, best_fname)
        shutil.copyfile(os.path.join(args.outdir, best_fname), dst)
        print(f"  💾 Mis à jour -> {dst}")
        if sk == "opr2":
            dst_alias = os.path.join(RESULTS, "best_params_otr2.json")
            shutil.copyfile(os.path.join(args.outdir, best_fname), dst_alias)
            print(f"  💾 Alias OTR 2.0 mis à jour -> {dst_alias}")


if __name__ == "__main__":
    main()
