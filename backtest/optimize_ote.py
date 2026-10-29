"""
backtest/optimize_ote.py — Optimisation Optuna (TPE) de la stratégie OTE.

Protocole : TRAIN = 2021-01 -> --split (2 moitiés avant/après 2023-01, score = la PIRE des deux), TEST = après --split.
Leviers CONDITIONNELS : un réglage n'est tiré que s'il sert (ex. seuils de bulle seulement si la confirmation 'bubble' est active),
ce qui réduit fortement le bruit. Zone OTE figée à 0,618–0,786 par défaut (--unfreeze ote_lo,ote_hi pour les libérer).
NOTE : strategy_ote.prepare(df) lit le dict global PARAMS ; l'adaptateur le remplace temporairement à chaque essai.
Les réglages de session n'existent pas dans strategy_ote : ils ne sont donc PAS optimisés (ils étaient sans effet auparavant).
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
    "side": ("cat", ["short", "both"]),          # 'long' seul retiré : il exploite la dérive haussière (or, USDJPY...) et ne tient pas
    "confirms": ("cat", ["bubble", "mss", "bubble,mss"]),
    "conf_tfs": ("cat", ["5m", "15m", "5m,15m", "5m,15m,1h", "15m,1h", "1h"]),
    "trend_filter": ("int", 0, 1),
    "trend_mode": ("cat", ["dow", "ma", "both"]),
    "zz_atr_trend": ("float", 1.5, 8.0),
    "ma_len": ("cat", [20, 50, 100, 200]),
    "range_tfs": ("cat", ["4h", "1h", "4h,1h"]),
    "zz_atr_4h": ("float", 1.5, 6.0),
    "zz_atr_1h": ("float", 1.5, 6.0),
    "range_origin": ("cat", ["leg", "pivot"]),
    "leg_lookback": ("cat", [100, 300, 600]),
    "ote_lo": ("float", 0.50, 0.75),
    "ote_hi": ("float", 0.79, 1.00),
    "ote_react": ("int", 0, 1),
    "ote_react_level": ("float", 0.3, 0.6),
    "mss_fractal": ("int", 2, 12),
    "mss_speed_ratio": ("float", 0.5, 4.0),
    "mss_min_sweep_pct": ("float", 0.0, 0.05),
    "mss_trough_pivot": ("int", 0, 1),
    "bubble_min_pct_5m": ("float", 0.0, 0.06),
    "bubble_min_pct_15m": ("float", 0.0, 0.10),
    "bubble_min_pct_1h": ("float", 0.0, 0.20),
    "tp_fib": ("float", 0.5, 3.5),
    "be_rr": ("cat", [1000.0, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "sl_buf_atr": ("float", 0.0, 2.0),
    "min_sl_pct": ("float", 0.0, 0.4),
    "min_rr": ("float", 0.0, 3.0),
}
FROZEN_DEFAULT = {"ote_lo": 0.618, "ote_hi": 0.786}
REFERENCE = {"side": "short", "confirms": "mss", "conf_tfs": "5m", "trend_filter": 0, "trend_mode": "both", "zz_atr_trend": 5.5628,
             "ma_len": 200, "range_tfs": "1h", "zz_atr_4h": 3.5947, "zz_atr_1h": 1.672, "range_origin": "leg", "leg_lookback": 600,
             "ote_lo": 0.5237, "ote_hi": 0.9864, "ote_react": 0, "mss_fractal": 2, "mss_speed_ratio": 3.0635, "mss_min_sweep_pct": 0.0498,
             "mss_trough_pivot": 1, "tp_fib": 1.5233, "be_rr": 2.0, "sl_buf_atr": 1.8949, "min_sl_pct": 0.2657, "min_rr": 0.0143}


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def suggest_params(trial, frozen, base):
    p = dict(base)
    p.update(frozen)

    def pick(*keys):
        nonlocal p
        p = oc.apply_space(trial, {k: SPACE[k] for k in keys}, frozen, p)

    pick("side", "confirms", "conf_tfs", "trend_filter", "range_tfs", "range_origin", "ote_lo", "ote_hi", "ote_react",
         "tp_fib", "be_rr", "sl_buf_atr", "min_sl_pct", "min_rr")
    rt = _csv(p["range_tfs"])
    if "4h" in rt:
        pick("zz_atr_4h")
    if "1h" in rt:
        pick("zz_atr_1h")
    if p["range_origin"] == "leg":
        pick("leg_lookback")
    if p["trend_filter"]:
        pick("trend_mode")
        if p["trend_mode"] in ("dow", "both"):
            pick("zz_atr_trend")
        if p["trend_mode"] in ("ma", "both"):
            pick("ma_len")
    if p["ote_react"]:
        pick("ote_react_level")
    if "mss" in _csv(p["confirms"]):
        pick("mss_fractal", "mss_speed_ratio", "mss_min_sweep_pct", "mss_trough_pivot")
    if "bubble" in _csv(p["confirms"]):
        pick(*[f"bubble_min_pct_{tf}" for tf in _csv(p["conf_tfs"])])
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
    def __init__(self, symbols, datadir, split, fee, risk, lev, start="2021-01-04", mid="2023-01-01", min_trades=15):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        for s in symbols:
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        self.t = {s: dict(start=self.P[s].idx_at(start), mid=self.P[s].idx_at(mid), split=self.P[s].idx_at(split), end=self.P[s].n) for s in symbols}

    def run_pair(self, s, p, i0, i1):
        if i1 - i0 < 1000:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0}
        b = self.P[s].base
        df = pd.DataFrame({"time": pd.to_datetime(b["open_ns"][i0:i1], utc=True), "open": b["o"][i0:i1], "high": b["h"][i0:i1],
                           "low": b["l"][i0:i1], "close": b["c"][i0:i1]})
        try:
            tr, _, st = engine_mod.run_backtest(df, _OTE(p), capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s], slippage_pct=0.0,
                                               max_leverage=self.lev[s], allow_short=True, be_rr=p.get("be_rr", 1000.0))
            rl, rs = oc.side_r(tr)
            return {"return_pct": st["return_pct"], "max_dd_pct": st["max_drawdown_pct"], "trades": st["trades"], "r_long": rl, "r_short": rs}
        except Exception:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0}

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
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--fee", type=float, default=-1.0, help="% par ordre ; -1 = frais FTMO réels par actif")
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=0.0)
    ap.add_argument("--outdir", default="results_opt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--wall_time", type=float, default=300.0)
    ap.add_argument("--min_trades", type=int, default=25)
    ap.add_argument("--freeze", default="")
    ap.add_argument("--unfreeze", default="", help="ex: ote_lo,ote_hi pour libérer la zone OTE")
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
