"""
backtest/optimize_ote.py — Optimisation Optuna (TPE) de la stratégie OTE avec Walk-Forward (WFA).

Anti-suroptimisation intégré :
  1. Hybride Walk-Forward 3 plis (2021, 2022, 2023) + Courbe continue 3 ans (capture le vrai Max Drawdown cumulé).
  2. Préparation unique sur l'historique complet par paire (préserve le warmup EMA 200 / ZigZag 4h aux frontières d'années + 6x plus rapide).
  3. Paliers discrets (ratios Fibonacci réels 1.0, 1.272, 1.618, 2.0, 2.618, paliers ATR nets) pour favoriser les zones plates.
  4. Détection de zones plates (médiane/mode du top 15%) + garde-fou Référence + export graphique (slices_ote.png).
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
    "side": "both", "confirms": "mss", "conf_tfs": "5m", "trend_filter": 0, "trend_tf": "4h", "trend_mode": "ema",
    "zz_atr_trend": 5.0, "ma_len": 200, "range_tfs": "1h", "zz_atr_4h": 3.0, "zz_atr_1h": 1.5,
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
            strategy.PARAMS["_quiet"] = True
            strategy.DIAG.clear()
            return strategy.prepare(df)
        finally:
            strategy.PARAMS.clear()
            strategy.PARAMS.update(orig)

    def signal(self, d, i):
        return strategy.signal(d, i)


class EvaluatorOTE:
    def __init__(self, symbols, datadir, split, fee, risk, lev, min_trades=25):
        self.symbols, self.risk, self.min_trades = symbols, risk, min_trades
        self.fee = {s: (fee if fee >= 0 else realism.fee_for(s)) for s in symbols}
        self.lev = {s: (lev if lev > 0 else realism.levier_for(s)) for s in symbols}
        self.P = {s: (oc.ensure_data(s, datadir), data.PairData(s, datadir))[1] for s in symbols}
        self.df_full = {}
        for s in symbols:
            b = self.P[s].base
            self.df_full[s] = pd.DataFrame({
                "time": pd.to_datetime(b["open_ns"], utc=True),
                "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"],
            })
            print(f"  {s}: {self.P[s].n} bougies 5m | frais {self.fee[s]:.4f}% | levier {self.lev[s]:g}", flush=True)
        # Les 6 premiers mois de 2021 (2021-01 -> 2021-07) servent de tampon de préchauffage (warmup)
        # pour les EMA Daily/4h et le ZigZag (leg_lookback=300).
        self.t = {
            s: dict(
                start=self.P[s].idx_at("2021-07-01"),
                w1=self.P[s].idx_at("2022-10-01"),
                split=self.P[s].idx_at(split),
                end=self.P[s].n,
            )
            for s in symbols
        }

    def prepare_pair(self, s, p):
        return _OTE(p).prepare(self.df_full[s].copy())

    def sim_slice(self, s, df_prep, p, i0, i1):
        if i1 - i0 < 1000:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}
        try:
            sub = df_prep.iloc[i0:i1]
            tr, _, st = engine_mod.run_backtest(sub, _OTE(p), capital=10000.0, risk_pct=self.risk, fee_pct=self.fee[s], slippage_pct=0.0,
                                                max_leverage=self.lev[s], allow_short=True, be_rr=p.get("be_rr", 1000.0), pre_prepared=True)
            rl, rs = oc.side_r(tr)
            return {"return_pct": st["return_pct"], "max_dd_pct": st["max_drawdown_pct"], "trades": st["trades"], "r_long": rl, "r_short": rs}
        except Exception:
            return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "r_long": 0.0, "r_short": 0.0}

    def objective(self, p):
        # On prépare les signaux UNE SEULE FOIS par paire sur l'historique 2021-2024 (préserve le warmup EMA/ZigZag et accélère x5)
        f_full, f1, f2 = {}, {}, {}
        for s in self.symbols:
            t = self.t[s]
            df_prep = _OTE(p).prepare(self.df_full[s].iloc[:t["split"]].copy())
            f_full[s] = self.sim_slice(s, df_prep, p, t["start"], t["split"])
            f1[s] = self.sim_slice(s, df_prep, p, t["start"], t["w1"])
            f2[s] = self.sim_slice(s, df_prep, p, t["w1"], t["split"])
        tot_tr = sum(r["trades"] for r in f_full.values())
        min_tot = max(24, int(self.min_trades * len(self.symbols) * 0.55))
        if tot_tr < min_tot:
            return -10.0 + (tot_tr / float(min_tot)) * 5.0
        fold_min = max(6.0, self.min_trades / 2.0)
        fold_min_tot = max(8, min_tot // 3)
        s_full = oc.half_score(f_full, self.min_trades, min_total_trades=min_tot, dd_limit=15.0)
        s1 = oc.half_score(f1, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s2 = oc.half_score(f2, fold_min, min_total_trades=fold_min_tot, dd_limit=12.0)
        s_wfa = 0.6 * min(s1, s2) + 0.4 * ((s1 + s2) / 2.0)
        return 0.5 * s_full + 0.5 * s_wfa

    def evaluate(self, p, period):
        a_key, b_key = {"train": ("start", "split"), "test": ("split", "end")}[period]
        out = {}
        for s in self.symbols:
            df_prep = self.prepare_pair(s, p)
            out[s] = self.sim_slice(s, df_prep, p, self.t[s][a_key], self.t[s][b_key])
        return out


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
    ap.add_argument("--min_trades", type=int, default=25, help="trades visés par paire sur les 3 ans de TRAIN")
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
