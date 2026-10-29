"""
backtest/optimize_fvg.py — Optimisation Optuna (TPE) de la stratégie FVG HTF + MSS.

Protocole : TRAIN = 2021-01 -> --split (2 moitiés : avant/après 2023-01, score = la PIRE des deux), TEST = après --split (lu à la fin seulement).
Correctifs : dossier créé avant tout ; score "doux" (pas de plateau à -3) ; réglages de référence testés en premier et comparés ;
sauvegardes + reprise (sqlite) ; arrêt avant la limite de temps ; frais et levier réels FTMO par actif.
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

SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "XAUUSD"]

SPACE = {
    "side": ("cat", ["short", "long", "both"]),
    "conf_tfs": ("cat", ["5m", "15m", "5m,15m"]),
    "htf": ("cat", ["1h", "4h", "1d", "1h,4h", "4h,1d", "1h,4h,1d"]),
    "mit_tf": ("cat", ["1h", "4h"]),
    "mit_depth": ("float", 0.0, 1.0),
    "mss_k": ("int", 2, 16),
    "speed_ratio": ("float", 0.5, 4.0),
    "speed_ref": ("int", 0, 1),
    "trough_pivot": ("int", 0, 1),
    "min_sweep_pct": ("float", 0.0, 0.05),
    "min_corr_pct": ("float", 0.0, 0.5),
    "min_corr_atr": ("float", 0.0, 5.0),
    "max_back": ("int", 1, 10),
    "zone_pad_atr": ("float", 0.0, 1.5),
    "entry_lvl": ("float", 0.0, 1.0),
    "fvg_min_atr": ("float", 0.0, 1.0),
    "sl_buf_atr": ("float", 0.0, 2.5),
    "tp_mode": ("cat", ["rr", "fvg"]),
    "rr": ("float", 1.0, 6.0),
    "min_rr": ("float", 0.5, 3.0),
    "max_rr": ("float", 3.0, 10.0),
    "tp_edge": ("float", 0.0, 1.0),
    "be_rr": ("float", 0.0, 4.0),
    "ttl": ("cat", [0, 12, 36, 96, 288]),
    "min_sl_pct": ("float", 0.0, 0.3),
    "trend": ("cat", ["none", "4h:200", "4h:50", "1d:50", "1d:200"]),
}
# Figés par défaut : le sens (long/short seul = on parie sur la dérive haussière d'une période) et la session (pas d'effet voulu).
# --freeze "k=v;k2=v2" ajoute des figés ; --unfreeze "k1,k2" en libère.
FROZEN_DEFAULT = {"side": "both"}
# Réglages de référence (testés en premier) : ceux de la config d'origine
REFERENCE = {"conf_tfs": "15m", "mss_k": 2, "speed_ratio": 1.0, "speed_ref": 0, "trough_pivot": 0, "min_sweep_pct": 0.03, "htf": "1h",
             "mit_tf": "4h", "mit_depth": 1.0, "zone_pad_atr": 0.5, "entry_lvl": 0.0, "fvg_min_atr": 0.25, "sl_buf_atr": 2.0,
             "tp_mode": "rr", "rr": 4.0, "min_rr": 1.5, "max_rr": 5.0, "tp_edge": 1.0, "be_rr": 0.7, "ttl": 0, "min_sl_pct": 0.2,
             "side": "both", "trend": "1d:50"}


class Evaluator:
    def __init__(self, symbols, datadir, split, fee, risk, lev, start="2021-01-04", mid="2023-01-01", min_trades=15):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        self.t = {s: dict(start=self.P[s].idx_at(start), mid=self.P[s].idx_at(mid), split=self.P[s].idx_at(split), end=self.P[s].n) for s in symbols}

    def run_pair(self, s, p, a, b):
        sigs = strategy.signals(self.P[s], p, a, b)
        tr, _, st = engine.run(self.P[s], sigs, 10000.0, self.risk, self.fee[s], self.lev[s], p["be_rr"], p["ttl"], a, b)
        st["r_long"], st["r_short"] = oc.side_r(tr)
        return st

    def rows(self, p, a_key, b_key):
        return {s: self.run_pair(s, p, self.t[s][a_key], self.t[s][b_key]) for s in self.symbols}

    def objective(self, p):
        return min(oc.half_score(self.rows(p, "start", "mid"), self.min_trades), oc.half_score(self.rows(p, "mid", "split"), self.min_trades))

    def evaluate(self, p, period):
        return self.rows(p, *{"train": ("start", "split"), "test": ("split", "end")}[period])


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
    ap.add_argument("--min_trades", type=int, default=25, help="trades visés par paire et par moitié (en dessous, le score est réduit proportionnellement)")
    ap.add_argument("--freeze", default="", help="'k=v;k2=v2' leviers figés en plus")
    ap.add_argument("--unfreeze", default="", help="leviers à libérer, séparés par des virgules (ex: side)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frozen = dict(FROZEN_DEFAULT)
    frozen.update(oc.parse_kv(a.freeze))
    for k in [x.strip() for x in a.unfreeze.split(",") if x.strip()]:
        frozen.pop(k, None)
    ev = Evaluator([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)
    base = dict(strategy.PARAMS)
    base.update(REFERENCE)

    def suggest(trial):
        p = oc.apply_space(trial, SPACE, frozen, base)
        if p["tp_mode"] == "fvg" and p["max_rr"] < p["min_rr"]:
            p["max_rr"] = p["min_rr"]
        return p

    oc.run_study("fvg", ev, suggest, base, SPACE, frozen, a, "best_params_fvg.json", "summary_fvg.txt")


if __name__ == "__main__":
    main()
