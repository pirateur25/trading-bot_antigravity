"""
backtest/optimize_ote.py — Optimisation Optuna (TPE) de la stratégie OTE avec Walk-Forward (WFA).

Anti-suroptimisation intégré :
  1. Walk-Forward glissant sur 3 plis (2021, 2022, 2023) : score = 0.6 * min(plis) + 0.4 * mean(plis) (Calmar).
  2. Panier multi-paires agrégé + pénalité dure (-10) si le panier fait moins de min_total_trades.
  3. Paliers discrets (ratios Fibonacci réels 1.0, 1.272, 1.618, 2.0, 2.618, paliers ATR nets) pour favoriser les zones plates.
  4. Détection de zones plates (médiane du top 15%) + export graphique des tranches (slices_ote.png).
  5. Suivi de tendance type FVG (trend_mode="ema" ou "dow" sur 4h/1d) et filtre de session (0-24, 7-17, 7-22).
  6. Zone OTE théorique stricte figée à 0.618–0.786 par défaut ; L2 = vrai plus bas mèche comprise (mss_trough_pivot=0).
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data  # noqa: E402
import engine_ote as engine_mod  # noqa: E402
import opt_common as oc  # noqa: E402
import realism  # noqa: E402
import strategy_ote as strategy  # noqa: E402

SYMBOLS_DEFAULT = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY", "XAUUSD"]

SPACE = {
    "side": ("cat", ["short", "both"]),
    "confirms": ("cat", ["mss", "bubble", "bubble,mss"]),
    "conf_tfs": ("cat", ["5m", "15m", "5m,15m", "15m,1h", "1h"]),
    "trend_filter": ("cat", [0, 1]),
    "trend_tf": ("cat", ["4h", "1d"]),
    "trend_mode": ("cat", ["ema", "dow"]),
    "zz_atr_trend": ("cat", [3.0, 4.0, 5.0, 6.0]),
    "ma_len": ("cat", [20, 50, 100, 200]),
    "range_tfs": ("cat", ["4h", "1h", "4h,1h"]),
    "zz_atr_4h": ("cat", [2.0, 3.0, 4.0, 5.0]),
    "zz_atr_1h": ("cat", [1.5, 2.5, 3.5, 4.5]),
    "range_origin": ("cat", ["leg", "pivot"]),
    "leg_lookback": ("cat", [100, 300, 600]),
    "ote_lo": ("cat", [0.60, 0.618, 0.65, 0.70]),
    "ote_hi": ("cat", [0.75, 0.786, 0.80, 0.85]),
    "ote_react": ("cat", [0, 1]),
    "ote_react_level": ("cat", [0.382, 0.5]),
    "mss_fractal": ("cat", [2, 3, 4, 6]),
    "mss_speed_ratio": ("cat", [1.0, 1.5, 2.0, 3.0]),
    "mss_min_sweep_pct": ("cat", [0.0, 0.01, 0.03]),
    "mss_trough_pivot": ("cat", [0, 1]),
    "bubble_min_pct_5m": ("cat", [0.0, 0.02, 0.04]),
    "bubble_min_pct_15m": ("cat", [0.0, 0.04, 0.08]),
    "bubble_min_pct_1h": ("cat", [0.0, 0.08, 0.15]),
    "tp_fib": ("cat", [1.0, 1.272, 1.618, 2.0, 2.618]),
    "be_rr": ("cat", [1000.0, 1.0, 1.5, 2.0, 3.0]),
    "sl_buf_atr": ("cat", [0.0, 0.5, 1.0, 1.5, 2.0]),
    "min_sl_pct": ("cat", [0.0, 0.1, 0.2]),
    "min_rr": ("cat", [0.5, 1.0, 1.5, 2.0]),
    "session": ("cat", ["0-24", "7-17", "7-22"]),
}
# Figés par défaut :
# - Zone OTE fixée à 0.618 - 0.786
# - Confirmation fixée à "mss" (une seule confirmation par étude)
# - L2 = vrai plus bas mèche comprise (mss_trough_pivot=0)
# - Pas de seuils % fixes qui déséquilibrent EURUSD vs XAUUSD (min_sl_pct=0.0, mss_min_sweep_pct=0.0)
# - range_origin="leg" (leg_lookback=300), ote_react=0, side="both"
FROZEN_DEFAULT = {
    "ote_lo": 0.618,
    "ote_hi": 0.786,
    "confirms": "mss",
    "mss_trough_pivot": 0,
    "min_sl_pct": 0.0,
    "mss_min_sweep_pct": 0.0,
    "range_origin": "leg",
    "leg_lookback": 300,
    "ote_react": 0,
    "side": "both",
}
REFERENCE = {
    "side": "short", "confirms": "mss", "conf_tfs": "5m", "trend_filter": 0, "trend_tf": "4h", "trend_mode": "ema",
    "zz_atr_trend": 5.0, "ma_len": 200, "range_tfs": "1h", "zz_atr_4h": 3.5, "zz_atr_1h": 1.5,
    "range_origin": "leg", "leg_lookback": 300, "ote_lo": 0.618, "ote_hi": 0.786, "ote_react": 0,
    "mss_fractal": 2, "mss_speed_ratio": 3.0, "mss_min_sweep_pct": 0.0, "mss_trough_pivot": 0,
    "tp_fib": 1.618, "be_rr": 2.0, "sl_buf_atr": 1.5, "min_sl_pct": 0.0, "min_rr": 1.0,
    "session": "0-24", "session_start": 0, "session_end": 24,
}


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def suggest_params(trial, frozen, base):
    p = dict(base)
    p.update(frozen)

    def pick(*keys):
        nonlocal p
        p = oc.apply_space(trial, {k: SPACE[k] for k in keys}, frozen, p)

    pick("side", "confirms", "conf_tfs", "trend_filter", "range_tfs", "range_origin", "ote_lo", "ote_hi", "ote_react",
         "tp_fib", "be_rr", "sl_buf_atr", "min_sl_pct", "min_rr", "session")
    rt = _csv(p["range_tfs"])
    if "4h" in rt:
        pick("zz_atr_4h")
    if "1h" in rt:
        pick("zz_atr_1h")
    if p["range_origin"] == "leg":
        pick("leg_lookback")
    if p["trend_filter"]:
        pick("trend_tf", "trend_mode")
        if p["trend_mode"] in ("dow", "both"):
            pick("zz_atr_trend")
        if p["trend_mode"] in ("ma", "both", "ema"):
            pick("ma_len")
    if p["ote_react"]:
        pick("ote_react_level")
    if "mss" in _csv(p["confirms"]):
        pick("mss_fractal", "mss_speed_ratio", "mss_min_sweep_pct", "mss_trough_pivot")
    if "bubble" in _csv(p["confirms"]):
        pick(*[f"bubble_min_pct_{tf}" for tf in _csv(p["conf_tfs"])])
    s0, s1 = p.get("session", "0-24").split("-")
    p["session_start"], p["session_end"] = int(s0), int(s1)
    return p


class _OTE:
    """Adaptateur : injecte p dans strategy_ote.PARAMS le temps de prepare()."""
    def __init__(self, p):
        self.p = p

    def prepare(self, df):
        orig = dict(strategy.PARAMS)
        try:
            strategy.PARAMS.update(self.p)
            strategy.DIAG.clear()
            return strategy.prepare(df)
        finally:
            strategy.PARAMS.clear()
            strategy.PARAMS.update(orig)

    def signal(self, d, i):
        return strategy.signal(d, i)


class EvaluatorOTE:
    def __init__(self, symbols, datadir, split, fee, risk, lev, min_trades=10):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        self.t = {
            s: dict(
                start=self.P[s].idx_at("2021-01-04"),
                w1=self.P[s].idx_at("2022-01-01"),
                w2=self.P[s].idx_at("2023-01-01"),
                split=self.P[s].idx_at(split),
                end=self.P[s].n,
            )
            for s in symbols
        }

    def run_pair(self, s, p, i0, i1):
        if i1 - i0 < 1000:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}
        b = self.P[s].base
        df = pd.DataFrame({"time": pd.to_datetime(b["open_ns"][i0:i1], utc=True), "open": b["o"][i0:i1], "high": b["h"][i0:i1],
                           "low": b["l"][i0:i1], "close": b["c"][i0:i1]})
        try:
            tr, _, st = engine_mod.run_backtest(df, _OTE(p), capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s], slippage_pct=0.0,
                                                max_leverage=self.lev[s], allow_short=True, be_rr=p.get("be_rr", 1000.0))
            rl, rs = oc.side_r(tr)
            return {"return_pct": st["return_pct"], "max_dd_pct": st["max_drawdown_pct"], "trades": st["trades"], "r_long": rl, "r_short": rs}
        except Exception:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}

    def rows(self, p, a_key, b_key):
        return {s: self.run_pair(s, p, self.t[s][a_key], self.t[s][b_key]) for s in self.symbols}

    def objective(self, p):
        f1 = self.rows(p, "start", "w1")
        f2 = self.rows(p, "w1", "w2")
        f3 = self.rows(p, "w2", "split")
        tot_tr = sum(r["trades"] for f in (f1, f2, f3) for r in f.values())
        min_tot = max(40, self.min_trades * len(self.symbols) * 2)
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(min_tot)) * 5.0
        s1 = oc.half_score(f1, self.min_trades, min_total_trades=8)
        s2 = oc.half_score(f2, self.min_trades, min_total_trades=8)
        s3 = oc.half_score(f3, self.min_trades, min_total_trades=8)
        return 0.6 * min(s1, s2, s3) + 0.4 * ((s1 + s2 + s3) / 3.0)

    def evaluate(self, p, period):
        return self.rows(p, *{"train": ("start", "split"), "test": ("split", "end")}[period])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--fee", type=float, default=-1.0, help="% par ordre ; -1 = frais FTMO réels par actif")
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=0.0)
    ap.add_argument("--outdir", default="results_opt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--wall_time", type=float, default=300.0)
    ap.add_argument("--min_trades", type=int, default=10)
    ap.add_argument("--freeze", default="")
    ap.add_argument("--unfreeze", default="", help="ex: ote_lo,ote_hi,confirms")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    frozen = dict(FROZEN_DEFAULT)
    frozen.update(oc.parse_kv(a.freeze))
    for k in [x.strip() for x in a.unfreeze.split(",") if x.strip()]:
        frozen.pop(k, None)
    ev = EvaluatorOTE([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.split, a.fee, a.risk, a.leverage, min_trades=a.min_trades)
    base = dict(strategy.PARAMS)
    base.update(REFERENCE)
    base.update(frozen)
    oc.run_study("ote", ev, lambda tr: suggest_params(tr, frozen, base), base, SPACE, frozen, a, "best_params_ote.json", "summary_ote.txt")


if __name__ == "__main__":
    main()
