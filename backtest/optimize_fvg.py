"""
backtest/optimize_fvg.py — Optimisation Optuna (TPE) de la stratégie FVG HTF + MSS avec Walk-Forward (WFA).

Anti-suroptimisation intégré :
  1. Hybride Walk-Forward 3 plis (2021, 2022, 2023) + Courbe continue 3 ans (capture le vrai Max Drawdown cumulé).
  2. Panier multi-paires agrégé + pénalité asymétrique (une perte + gros DD est doublement sanctionnée).
  3. Paliers discrets (catégoriels) au lieu de flottants continus pour forcer la recherche de zones plates.
  4. Détection de zones plates (médiane/mode du top 15%) + garde-fou Référence + export graphique (slices_fvg.png).
  5. Non-mitigation stricte avant le sommet H1 (mit_depth figé à 0.0 par défaut = moindre mèche invalide la FVG).
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data  # noqa: E402
import engine_fvg as engine  # noqa: E402
import opt_common as oc  # noqa: E402
import realism  # noqa: E402
import strategy_fvg as strategy  # noqa: E402

# Panier FVG par défaut : paires directionnelles liquides (USDCHF et les Cryptos sont exclues par défaut car en retour à la moyenne / bruit 24/7 sur FVG)
SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

SPACE = {
    "side": ("cat", ["short", "long", "both"]),
    "conf_tfs": ("cat", ["5m", "15m", "5m,15m"]),
    "htf": ("cat", ["1h", "4h", "1d", "1h,4h", "4h,1d", "1h,4h,1d"]),
    "mit_tf": ("cat", ["1h"]),
    "mit_depth": ("cat", [0.0]),
    "mss_k": ("cat", [2, 3, 4, 6, 8]),
    "speed_ratio": ("cat", [1.0, 1.3, 1.6, 2.0, 3.0]),
    "speed_ref": ("int", 0, 1),
    "trough_pivot": ("int", 0, 1),
    "min_sweep_pct": ("cat", [0.0, 0.01, 0.03]),
    "min_corr_pct": ("cat", [0.0, 0.1, 0.2, 0.3]),
    "min_corr_atr": ("cat", [0.0, 1.0, 2.0, 3.0]),
    "max_back": ("int", 1, 8),
    "zone_pad_atr": ("cat", [0.0]),
    "entry_lvl": ("cat", [0.0, 0.25, 0.5]),
    "fvg_min_atr": ("cat", [0.0, 0.25, 0.5]),
    "sl_buf_atr": ("cat", [0.0, 0.5, 1.0, 1.5, 2.0]),
    "tp_mode": ("cat", ["fvg", "rr"]),
    "rr": ("cat", [2.0, 2.5, 3.0, 3.5, 4.0]),
    "min_rr": ("cat", [1.5, 2.0, 2.5]),
    "max_rr": ("cat", [3.0, 4.0, 5.0, 6.0, 8.0]),
    "tp_edge": ("cat", [0.0, 0.5, 1.0]),
    "be_rr": ("cat", [0.0, 0.7, 1.0, 1.5, 2.0]),
    "ttl": ("cat", [0, 24, 96, 288]),
    "min_sl_pct": ("cat", [0.0, 0.05, 0.1, 0.2]),
    "trend": ("cat", ["none", "4h:200", "4h:50", "1d:50", "1d:200"]),
    "session": ("cat", ["0-24", "7-17", "7-22"]),
}
FROZEN_DEFAULT = {
    "side": "both",
    "mit_tf": "1h",
    "mit_depth": 0.0,
    "zone_pad_atr": 0.0,
    "min_sl_pct": 0.0,
    "min_sweep_pct": 0.0,
    "tp_edge": 0.0,
    "speed_ref": 0,
    "trough_pivot": 0,
    "min_corr_pct": 0.0,
    "min_corr_atr": 0.0,
    "max_back": 5,
}
REFERENCE = {
    "conf_tfs": "15m", "mss_k": 2, "speed_ratio": 1.0, "speed_ref": 0, "trough_pivot": 0, "min_sweep_pct": 0.0,
    "htf": "1h,4h", "mit_tf": "1h", "mit_depth": 0.0, "zone_pad_atr": 0.0, "entry_lvl": 0.0, "fvg_min_atr": 0.25,
    "sl_buf_atr": 2.0, "tp_mode": "rr", "rr": 4.0, "min_rr": 1.5, "max_rr": 5.0, "tp_edge": 0.0, "be_rr": 0.7,
    "ttl": 0, "min_sl_pct": 0.0, "side": "both", "trend": "1d:50", "session": "0-24",
    "session_start": 0, "session_end": 24,
}


class Evaluator:
    def __init__(self, symbols, datadir, split, fee, risk, lev, min_trades=25):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        # Les 6 premiers mois de 2021 (2021-01 -> 2021-07) servent de tampon de préchauffage (warmup)
        # pour les EMA Daily (1d:50 / 1d:200) et le réservoir de zones FVG HTF (1h/4h/1d).
        self.t = {
            s: dict(
                start=self.P[s].idx_at("2021-07-01"),
                w1=self.P[s].idx_at("2022-10-01"),
                split=self.P[s].idx_at(split),
                end=self.P[s].n,
            )
            for s in symbols
        }

    def run_pair(self, s, p, a, b):
        sigs = strategy.signals(self.P[s], p, a, b)
        tr, _, st = engine.run(self.P[s], sigs, 10000.0, self.risk, self.fee[s], self.lev[s], p["be_rr"], p["ttl"], a, b)
        st["r_long"], st["r_short"] = oc.side_r(tr)
        return st

    def rows(self, p, a_key, b_key):
        return {s: self.run_pair(s, p, self.t[s][a_key], self.t[s][b_key]) for s in self.symbols}

    def objective(self, p):
        # min_trades = objectif par paire sur le TRAIN (ex: 25 => ~12 par pli Walk-Forward de 15 mois)
        f_full = self.rows(p, "start", "split")
        tot_tr = sum(r["trades"] for r in f_full.values())
        min_tot = max(24, int(self.min_trades * len(self.symbols) * 0.55))
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(min_tot)) * 5.0
        fold_min = max(6.0, self.min_trades / 2.0)
        fold_min_tot = max(8, min_tot // 3)
        f1 = self.rows(p, "start", "w1")
        f2 = self.rows(p, "w1", "split")
        s_full = oc.half_score(f_full, self.min_trades, min_total_trades=min_tot, dd_limit=15.0)
        s1 = oc.half_score(f1, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s2 = oc.half_score(f2, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s_wfa = 0.6 * min(s1, s2) + 0.4 * ((s1 + s2) / 2.0)
        return 0.5 * s_full + 0.5 * s_wfa

    def evaluate(self, p, period):
        return self.rows(p, *{"train": ("start", "split"), "test": ("split", "end")}[period])


def suggest_params(trial, frozen, base):
    p = dict(base)
    p.update(frozen)

    def pick(*keys):
        nonlocal p
        p = oc.apply_space(trial, {k: SPACE[k] for k in keys}, frozen, p)

    pick("side", "conf_tfs", "htf", "mit_tf", "mit_depth", "mss_k", "speed_ratio", "speed_ref", "trough_pivot",
         "min_sweep_pct", "min_corr_pct", "min_corr_atr", "max_back", "zone_pad_atr", "entry_lvl", "fvg_min_atr",
         "sl_buf_atr", "tp_mode", "be_rr", "ttl", "min_sl_pct", "trend", "session")
    if p["tp_mode"] == "rr":
        pick("rr")
    else:
        pick("min_rr", "max_rr", "tp_edge")
        if p["max_rr"] < p["min_rr"]:
            p["max_rr"] = p["min_rr"]
    s0, s1 = p.get("session", "0-24").split("-")
    p["session_start"], p["session_end"] = int(s0), int(s1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--fee", type=float, default=-1.0, help="% par ordre ; -1 = frais FTMO réels par actif")
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=0.0, help="0 = levier FTMO par actif")
    ap.add_argument("--outdir", default="results_opt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--wall_time", type=float, default=300.0)
    ap.add_argument("--min_trades", type=int, default=25, help="trades visés par paire sur les 3 ans de TRAIN")
    ap.add_argument("--freeze", default="", help="'k=v;k2=v2' leviers figés en plus")
    ap.add_argument("--unfreeze", default="", help="leviers à libérer, séparés par des virgules")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frozen = dict(FROZEN_DEFAULT)
    frozen.update(oc.parse_kv(a.freeze))
    for k in [x.strip() for x in a.unfreeze.split(",") if x.strip()]:
        frozen.pop(k, None)
    ev = Evaluator([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)
    base = dict(strategy.PARAMS)
    base.update(REFERENCE)
    base.update(frozen)
    oc.run_study("fvg", ev, lambda tr: suggest_params(tr, frozen, base), base, SPACE, frozen, a, "best_params_fvg.json", "summary_fvg.txt")


if __name__ == "__main__":
    main()
