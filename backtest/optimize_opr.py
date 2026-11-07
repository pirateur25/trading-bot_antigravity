"""
backtest/optimize_opr.py — Optimisation Optuna (TPE) de la stratégie OPR + RV3 avec Walk-Forward (WFA).

Piliers figés (non négociables) :
  - Zone OPR M15 15h30 -> 15h45 heure française (`tz_mode="paris"`).
  - Exécution M5 jusqu'à 19h30 FR maximum (1 trade / jour max).
  - Setups d'entrée : Cassure + Réintégration (`setups="Cassure,Reintegration"`).
  - Filtre de tendance obligatoire : `mm20_50` en H1 (`trend_tf="1h"`, `trend_mode="mm20_50"`).
  - Formules RV3 sur 60 séances US (EM brut = Open*(VIX/100)/16, ligne B = moyenne calibrée 60j, ligne C = enveloppe +1 sigma).
  - Planchers SL 15 <= VIX <= 20 : 80 pts US30, 70 pts US100, 50 pts US500.
  - Annulation de l'ordre limite non déclenché si +1 R atteint (`cancel_rr=1.0`).

Leviers optimisés par Optuna (sur TRAIN 2015 -> split en 3 plis Walk-Forward + validation TEST hors-échantillon) :
  - be_rr                : Break-Even à X R (0.75, 1.0, 1.25, 1.5 ; 1.0 en référence)
  - em_noise_pct         : Seuil de bruit EM en fraction de l'EM brut (0.15, 0.20, 0.25)
  - sl_atr_mult          : Multiplicateur ATR(14) H1 quand VIX > 20 (0.5, 0.75, 1.0, 1.25, 1.5)
  - sl_vix20_mode        : Mode SL quand VIX > 20 ("max_atr" = max(SL_M5, mult*ATR_H1) | "add_atr" = SL_M5 + mult*ATR_H1)
  - rr_break             : Objectif TP du setup Cassure en R (1.5, 2.0, 2.5, 3.0, 3.5)
  - min_rr_reint         : RR minimum exigé sur le setup Réintégration (1.5, 2.0, 2.5)
  - res_be_mode          : Filtre d'obstacle avant d'atteindre le BE ("none", "rv3_b", "rv3_c", "sr_fvg", "all")
  - rv3_tp_mode          : Gestion du TP par rapport aux bornes RV3 B et C ("none", "reject_c", "reject_b", "cap_c", "cap_b", "ext_res")
  - fvg_h1_lookback_days : Fenêtre de validité des FVG H1 non mitigées (10, 20, 30 jours)
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data  # noqa: E402
import opt_common as oc  # noqa: E402
import realism  # noqa: E402
import strategy_opr as strategy  # noqa: E402

SYMBOLS_DEFAULT = ["NAS100", "USA500", "US30"]

SPACE = {
    # --- Piliers figés par défaut (libérables uniquement via --unfreeze) ---
    "side": ("cat", ["both", "long", "short"]),
    "setups": ("cat", ["Cassure,Reintegration", "Cassure", "Reintegration"]),
    "trend_tf": ("cat", ["1h"]),
    "trend_mode": ("cat", ["mm20_50"]),
    "cancel_rr": ("cat", [1.0]),
    "sl_floor_mode": ("cat", ["pct", "fixed_pts"]),
    "true_breakout_only": ("cat", [1, 0]),
    # --- Leviers optimisés : SL, BE, TP et Obstacles RV3 ---
    "be_rr": ("cat", [0.75, 1.0, 1.25, 1.5]),
    "em_noise_pct": ("cat", [0.15, 0.20, 0.25]),
    "sl_vix15_mode": ("cat", ["m5_only", "half_noise", "full_noise"]),
    "sl_atr_mult": ("cat", [0.5, 0.75, 1.0, 1.25, 1.5]),
    "sl_vix20_mode": ("cat", ["max_atr", "add_atr"]),
    "rr_break": ("cat", [1.5, 2.0, 2.5, 3.0, 3.5]),
    "min_rr_reint": ("cat", [1.5, 2.0, 2.5]),
    "res_be_mode": ("cat", ["none", "rv3_b", "rv3_c", "sr_fvg", "all"]),
    "rv3_tp_mode": ("cat", ["none", "reject_c", "reject_b", "cap_c", "cap_b", "ext_res"]),
    "fvg_h1_lookback_days": ("cat", [10, 20, 30]),
    # --- Leviers Anti-Range & Fenêtre d'exécution ---
    "min_vix": ("cat", [0.0, 13.0, 14.0, 15.0, 16.0]),
    "min_opr_em": ("cat", [0.0, 0.15, 0.20, 0.25]),
    "max_opr_em": ("cat", [0.45, 0.55, 0.70, 99.0]),
    "min_mm_spread_atr": ("cat", [0.0, 0.15, 0.30, 0.45]),
    "end_hour": ("cat", [17, 18, 19]),
    "max_fill_bars": ("cat", [3, 6, 12, 99]),
}

FROZEN_DEFAULT = {
    "side": "both",
    "setups": "Cassure,Reintegration",
    "trend_tf": "1h",
    "trend_mode": "mm20_50",
    "cancel_rr": 1.0,
    "sl_floor_mode": "pct",
    "true_breakout_only": 1,
}

REFERENCE = {
    "side": "both",
    "setups": "Cassure,Reintegration",
    "trend_tf": "1h",
    "trend_mode": "mm20_50",
    "tz_mode": "paris",
    "end_hour": 18,
    "end_min": 30,
    "cancel_rr": 1.0,
    "be_rr": 1.25,
    "em_noise_pct": 0.20,
    "sl_vix15_mode": "m5_only",
    "sl_floor_mode": "pct",
    "sl_atr_mult": 1.0,
    "sl_vix20_mode": "max_atr",
    "rr_break": 2.5,
    "min_rr_reint": 2.0,
    "res_be_mode": "rv3_b",
    "rv3_tp_mode": "cap_c",
    "max_rr_tp": 5.0,
    "fvg_h1_lookback_days": 15,
    "min_vix": 14.0,
    "min_opr_em": 0.20,
    "max_opr_em": 0.55,
    "min_mm_spread_atr": 0.15,
    "true_breakout_only": 1,
    "max_fill_bars": 6,
}


def normalize_sym(sym: str) -> str:
    s = sym.strip().upper()
    if s == "US100":
        return "NAS100"
    if s == "US500":
        return "USA500"
    return s


def suggest_params(trial, frozen, base):
    p = dict(base)
    p.update(frozen)

    def pick(*keys):
        nonlocal p
        p = oc.apply_space(trial, {k: SPACE[k] for k in keys}, frozen, p)

    pick(
        "side",
        "setups",
        "trend_tf",
        "trend_mode",
        "cancel_rr",
        "sl_floor_mode",
        "true_breakout_only",
        "be_rr",
        "em_noise_pct",
        "sl_vix15_mode",
        "sl_atr_mult",
        "sl_vix20_mode",
        "rr_break",
        "min_rr_reint",
        "res_be_mode",
        "rv3_tp_mode",
        "min_vix",
        "min_opr_em",
        "max_opr_em",
        "min_mm_spread_atr",
        "end_hour",
        "max_fill_bars",
    )
    if p["res_be_mode"] in ("sr_fvg", "all") or p["rv3_tp_mode"] == "ext_res":
        pick("fvg_h1_lookback_days")
    return p


class EvaluatorOPR:
    def __init__(self, symbols, datadir, split, fee, risk, lev, min_trades=180):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            # Pré-chauffe les caches (sessions quotidiennes RV3, tendance H1, FVG H1) une seule fois avant les essais
            strategy._precompute_daily_sessions(self.P[s], "paris")
            strategy._trend_h1_mm20_50(self.P[s], "1h", "mm20_50")
            strategy._h1_atr_and_fvgs(self.P[s])
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        # Les 6 premiers mois servent de préchauffage (warmup 60 séances RV3 + MM50 H1),
        # puis la période TRAIN [start, split] est découpée en 3 plis Walk-Forward équilibrés (f1, f2, f3).
        self.t = {}
        for s in symbols:
            open_ns = self.P[s].base["open_ns"]
            i_start = int(len(open_ns) and (open_ns < open_ns[0] + 180 * 86400 * 10**9).sum())
            i_split = self.P[s].idx_at(split)
            span = max(3, i_split - i_start)
            self.t[s] = dict(
                start=i_start,
                w1=i_start + span // 3,
                w2=i_start + (2 * span) // 3,
                split=i_split,
                end=self.P[s].n,
            )

    def run_pair(self, s, p, a, b):
        if b - a < 1000:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}
        tr, eq = strategy.run(self.P[s], capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s], leverage=self.lev[s], p=p, i0=a, i1=b)
        if len(tr) == 0 or len(eq) == 0:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}
        eq_arr = np.asarray(eq, dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        ret_pct = float((eq_arr[-1] / 10000.0 - 1.0) * 100.0)
        dd_pct = float(((eq_arr - peak) / peak).min() * 100.0)
        rl, rs = oc.side_r(tr)
        return {"return_pct": ret_pct, "max_dd_pct": dd_pct, "trades": int(len(tr)), "r_long": rl, "r_short": rs}

    def rows(self, p, a_key, b_key):
        return {s: self.run_pair(s, p, self.t[s][a_key], self.t[s][b_key]) for s in self.symbols}

    def objective(self, p):
        if float(p.get("be_rr", 1.0)) >= float(p.get("rr_break", 2.5)):
            return -9.0
        f_full = self.rows(p, "start", "split")
        tot_tr = sum(r["trades"] for r in f_full.values())
        n_sym = max(1, len(self.symbols))
        pair_target = (self.min_trades / float(n_sym)) if self.min_trades >= 90 else float(self.min_trades)
        min_tot = max(60, int(pair_target * n_sym * 0.60))
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(min_tot)) * 5.0
        fold_min = max(10.0, pair_target / 3.0)
        fold_min_tot = max(20, min_tot // 3)
        f1 = self.rows(p, "start", "w1")
        f2 = self.rows(p, "w1", "w2")
        f3 = self.rows(p, "w2", "split")
        s_full = oc.half_score(f_full, pair_target, min_total_trades=min_tot, dd_limit=16.0)
        s1 = oc.half_score(f1, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s2 = oc.half_score(f2, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s3 = oc.half_score(f3, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s_wfa = 0.5 * min(s1, s2, s3) + 0.5 * ((s1 + s2 + s3) / 3.0)
        return 0.5 * s_full + 0.5 * s_wfa

    def evaluate(self, p, period):
        return self.rows(p, *{"train": ("start", "split"), "test": ("split", "end")}[period])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--n_trials", type=int, default=800)
    ap.add_argument("--fee", type=float, default=-1.0, help="% par ordre ; -1 = frais FTMO réels par actif")
    ap.add_argument("--risk", type=float, default=0.5)
    ap.add_argument("--leverage", type=float, default=0.0, help="0 = levier FTMO par actif")
    ap.add_argument("--outdir", default="results_opt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--wall_time", type=float, default=300.0)
    ap.add_argument("--min_trades", type=int, default=180, help="trades visés sur les 9 ans de TRAIN (180 = ~60/indice)")
    ap.add_argument("--freeze", default="", help="'k=v;k2=v2' leviers figés en plus (ex: be_rr=1.0)")
    ap.add_argument("--unfreeze", default="", help="leviers à libérer, séparés par des virgules")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frozen = dict(FROZEN_DEFAULT)
    frozen.update(oc.parse_kv(a.freeze))
    for k in [x.strip() for x in a.unfreeze.split(",") if x.strip()]:
        frozen.pop(k, None)
    syms = [normalize_sym(s) for s in a.symbols.split(",") if s.strip()]
    ev = EvaluatorOPR(syms, a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)
    base = dict(strategy.PARAMS)
    base.update(REFERENCE)
    base.update(frozen)
    oc.run_study("opr", ev, lambda tr: suggest_params(tr, frozen, base), base, SPACE, frozen, a, "best_params_opr.json", "summary_opr.txt")
    # Copie automatique vers results/best_params_opr.json pour que run_backtest.py l'utilise directement
    src_json = os.path.join(a.outdir, "best_params_opr.json")
    dst_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    if os.path.exists(src_json):
        import shutil
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src_json, os.path.join(dst_dir, "best_params_opr.json"))


if __name__ == "__main__":
    main()

