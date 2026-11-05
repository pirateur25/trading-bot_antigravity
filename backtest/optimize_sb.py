"""
backtest/optimize_sb.py — Optimisation Optuna (TPE) de la stratégie Silver Bullet (SB) avec Walk-Forward (WFA).

Anti-suroptimisation intégré (identique OTE / FVG) :
  1. Hybride Walk-Forward 3 plis sur 2015 -> split (après 6 mois de préchauffage EMA/ZigZag) + Courbe continue 9 ans.
  2. Panier multi-actifs agrégé + pénalité asymétrique (perte + gros DD doublement sanctionnés, équilibre Long/Short).
  3. Paliers discrets (catégoriels) pour favoriser la détection de zones plates robustes.
  4. Détection du centre de plateau (médiane/mode du top 15%) + garde-fou Référence + export graphique (slices_sb.png).

Leviers optimisés :
  - be_rr        : passage à Break-Even à partir de X R (0.0 = désactivé, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
  - rr           : Take Profit en R (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
  - trend_filter : 0 = pas d'outil de tendance | 1 = uniquement dans le sens de la tendance HTF
  - trend_tf     : unité de temps de la tendance ("1h" ou "4h")
  - trend_mode   : type d'outil de tendance :
      * "mm20_50"     : SMA20 > SMA50 et close > SMA50 (haussier) ; inverse (baissier)
      * "ema_mm20_50" : EMA20 > EMA50 et close > EMA50 (haussier) ; inverse (baissier)
      * "ema"         : close vs EMA(ma_len) (outil OTE/FVG)
      * "ma"          : close vs EMA(ma_len) + pente EMA sur 5 bougies (outil OTE)
      * "dow"         : structure ZigZag HH+HL / LH+LL (outil OTE)
      * "both"        : accord simultané "dow" ET "ma" (outil OTE)
  - ma_len       : période EMA pour "ema"/"ma"/"both" (20, 50, 100, 200)
  - zz_atr_trend : multiplicateur ATR du ZigZag pour "dow"/"both" (3.0, 4.0, 5.0, 6.0)
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
import strategy_sb as strategy  # noqa: E402

SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDJPY", "NAS100", "USA500"]

SPACE = {
    "side": ("cat", ["both", "long", "short"]),
    "sessions": ("cat", [
        "London,NY AM,NY PM",
        "London,NY AM",
        "NY AM,NY PM",
        "London,NY PM",
        "London",
        "NY AM",
        "NY PM",
    ]),
    "rr": ("cat", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    "be_rr": ("cat", [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
    "trend_filter": ("cat", [0, 1]),
    "trend_tf": ("cat", ["1h", "4h"]),
    "trend_mode": ("cat", ["mm20_50", "ema_mm20_50", "ema", "ma", "dow", "both"]),
    "ma_len": ("cat", [20, 50, 100, 200]),
    "zz_atr_trend": ("cat", [3.0, 4.0, 5.0, 6.0]),
}

FROZEN_DEFAULT = {
    "side": "both",
    "sessions": "London,NY AM,NY PM",
}

REFERENCE = {
    "rr": 2.0,
    "be_rr": 0.0,
    "sessions": "London,NY AM,NY PM",
    "side": "both",
    "trend_filter": 0,
    "trend_tf": "1h",
    "trend_mode": "mm20_50",
    "ma_len": 200,
    "ma_slope_bars": 5,
    "zz_atr_trend": 4.0,
}


def suggest_params(trial, frozen, base):
    p = dict(base)
    p.update(frozen)

    def pick(*keys):
        nonlocal p
        p = oc.apply_space(trial, {k: SPACE[k] for k in keys}, frozen, p)

    pick("side", "sessions", "rr", "be_rr", "trend_filter")
    if p.get("trend_filter"):
        pick("trend_tf", "trend_mode")
        if p["trend_mode"] in ("ema", "ma", "both"):
            pick("ma_len")
        if p["trend_mode"] in ("dow", "both"):
            pick("zz_atr_trend")
    return p


class EvaluatorSB:
    def __init__(self, symbols, datadir, split, fee, risk, lev, min_trades=180):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        # Même découpage que OTE / FVG : 6 premiers mois = préchauffage (warmup EMA 200 / ZigZag),
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
        # Sur une configuration illogique où le BE >= TP, le BE ne peut jamais se déclencher avant le TP
        if float(p.get("be_rr", 0.0)) >= float(p.get("rr", 2.0)):
            return -9.0
        f_full = self.rows(p, "start", "split")
        tot_tr = sum(r["trades"] for r in f_full.values())
        n_sym = max(1, len(self.symbols))
        pair_target = (self.min_trades / float(n_sym)) if self.min_trades >= 90 else float(self.min_trades)
        min_tot = max(75, int(pair_target * n_sym * 0.65))
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(min_tot)) * 5.0
        fold_min = max(10.0, pair_target / 3.0)
        fold_min_tot = max(25, min_tot // 3)
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
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--fee", type=float, default=-1.0, help="% par ordre ; -1 = frais FTMO réels par actif")
    ap.add_argument("--risk", type=float, default=0.5)
    ap.add_argument("--leverage", type=float, default=0.0, help="0 = levier FTMO par actif")
    ap.add_argument("--outdir", default="results_opt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--wall_time", type=float, default=300.0)
    ap.add_argument("--min_trades", type=int, default=180, help="trades visés sur les 9 ans de TRAIN")
    ap.add_argument("--freeze", default="", help="'k=v;k2=v2' leviers figés en plus")
    ap.add_argument("--unfreeze", default="", help="leviers à libérer, séparés par des virgules (ex: sessions,side)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frozen = dict(FROZEN_DEFAULT)
    frozen.update(oc.parse_kv(a.freeze))
    for k in [x.strip() for x in a.unfreeze.split(",") if x.strip()]:
        frozen.pop(k, None)
    ev = EvaluatorSB([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)
    base = dict(strategy.PARAMS)
    base.update(REFERENCE)
    base.update(frozen)
    oc.run_study("sb", ev, lambda tr: suggest_params(tr, frozen, base), base, SPACE, frozen, a, "best_params_sb.json", "summary_sb.txt")


if __name__ == "__main__":
    main()

