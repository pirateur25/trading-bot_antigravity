"""
backtest/run_backtest.py — point d'entrée : stratégies OTE (A) et/ou FVG (B) sur forex + cryptos FTMO,
rapport Telegram (résumé, courbe, CSV détaillé des trades) et simulation de challenge FTMO.

Exemple :
  python backtest/run_backtest.py --symbols EURUSD,USDJPY,XAUUSD --crypto BTCUSD,ETHUSD --mode both
Les cryptos doivent d'abord exister dans data/ (workflow « fetch_crypto » ou téléchargement automatique ci-dessous).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data_fx  # noqa: E402
import engine_fvg  # noqa: E402
import engine_ote  # noqa: E402
import realism  # noqa: E402
import sim_challenge as sim_ch  # noqa: E402
import strategy_fvg  # noqa: E402
import strategy_ote  # noqa: E402
import viewer_export as ve  # noqa: E402

CAPITAL = 10000.0


def tg(text=None, photo=None, document=None):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN/CHAT_ID manquant -> ignoré")
        return
    try:
        if text:
            for i in range(0, len(text), 4000):
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": text[i:i + 4000]}, timeout=20)
        if photo and os.path.exists(photo):
            with open(photo, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data={"chat_id": chat}, files={"photo": f}, timeout=30)
        if document and os.path.exists(document):
            with open(document, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat}, files={"document": f}, timeout=60)
    except requests.RequestException as e:
        print("TELEGRAM erreur:", e)


def ensure_data(sym, datadir, start):
    """Crypto absente de data/ : téléchargement auto (Binance Vision -> KuCoin -> Bybit) puis sauvegarde."""
    if any(os.path.exists(os.path.join(datadir, sym + e)) for e in (".csv.gz", ".csv")):
        return
    import data_crypto as dc
    df, src = dc.load(dc.ftmo_to_binance(sym), "5m", start, source="auto", use_cache=False)
    os.makedirs(datadir, exist_ok=True)
    df[["time", "open", "high", "low", "close", "volume"]].to_csv(os.path.join(datadir, sym + ".csv.gz"), index=False)
    print(f"{sym}: {len(df)} bougies 5m ({src})")


def stats_of(tr, eq):
    if len(tr) == 0:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "avg_r": 0.0, "win_pct": 0.0, "pf": 0.0, "longs": 0, "shorts": 0, "by_year": {}, "by_side": {}}
    eq = np.asarray(eq, float)
    eq = eq[~np.isnan(eq)]
    peak = np.maximum.accumulate(eq)
    r = tr["r"]
    wins, loss = r[r > 0].sum(), -r[r <= 0].sum()
    yr = pd.to_datetime(tr["entry_time"], utc=True).dt.year
    return {"return_pct": (eq[-1] / CAPITAL - 1) * 100, "max_dd_pct": float(((eq - peak) / peak).min() * 100), "trades": len(tr),
            "avg_r": float(r.mean()), "win_pct": float((r > 0).mean() * 100), "pf": float(wins / loss) if loss > 0 else float("inf"),
            "longs": int((tr["side"] == "long").sum()), "shorts": int((tr["side"] == "short").sum()),
            "by_year": {str(y): (len(g), float(g["r"].mean())) for y, g in tr.groupby(yr)},
            "by_side": {s: (len(g), float(g["r"].mean())) for s, g in tr.groupby("side")}}


def summary(sym, strat, st, risk, fee, extra=""):
    t = [f"📈 {sym} — {strat} | risque {risk}% | frais {fee:.4f}%/ordre",
         f"Rendement {st['return_pct']:+.2f}% | DD max {st['max_dd_pct']:.2f}% | {st['trades']} trades | gagnants {st['win_pct']:.1f}% | "
         f"espérance {st['avg_r']:+.3f} R | PF {st['pf']:.2f} | L/S {st['longs']}/{st['shorts']}"]
    if st["by_year"]:
        t.append("Par année : " + " | ".join(f"{y}: {n}t {r:+.2f}R" for y, (n, r) in st["by_year"].items()))
    if st["by_side"]:
        t.append("Par sens : " + " | ".join(f"{s}: {n}t {r:+.2f}R" for s, (n, r) in st["by_side"].items()))
    return "\n".join(t + ([extra] if extra else []))


def load_params(mod, best, extra):
    p = dict(mod.PARAMS)
    if best and os.path.exists(best):
        p.update(json.load(open(best)))
    return mod.parse_params(extra, p) if extra else p


def run_fvg(P, sym, i0, i1, risk, fee, lev, p):
    sigs = strategy_fvg.signals(P, p, i0, i1)
    tr, eq, _ = engine_fvg.run(P, sigs, CAPITAL, risk, fee, lev, p["be_rr"], p["ttl"], i0, i1)
    return tr, eq


class _OTE:
    def __init__(self, p):
        self.p = p

    def prepare(self, df):
        orig = dict(strategy_ote.PARAMS)
        try:
            strategy_ote.PARAMS.update(self.p)
            strategy_ote.DIAG.clear()
            return strategy_ote.prepare(df)
        finally:
            strategy_ote.PARAMS.clear()
            strategy_ote.PARAMS.update(orig)

    def signal(self, d, i):
        return strategy_ote.signal(d, i)


def run_ote(P, sym, i0, i1, risk, fee, lev, p):
    b = P.base
    df_full = pd.DataFrame({"time": pd.to_datetime(b["open_ns"][:i1], utc=True), "open": b["o"][:i1], "high": b["h"][:i1],
                            "low": b["l"][:i1], "close": b["c"][:i1]})
    df_prep = _OTE(p).prepare(df_full).iloc[i0:i1]
    be = p.get("be_rr", 1000.0)
    tr, eq, _ = engine_ote.run_backtest(df_prep, _OTE(p), capital=CAPITAL, risk_pct=risk, fee_pct=fee, slippage_pct=0.0,
                                        max_leverage=lev, allow_short=True, be_rr=be, pre_prepared=True)
    return tr, eq.ffill().to_numpy()


def plot(sym, strat, eq, st, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.asarray(eq, float) / CAPITAL * 100, linewidth=1.2)
    ax.set_title(f"{sym} {strat} — {st['return_pct']:+.1f}% | DD {st['max_dd_pct']:.1f}% | {st['trades']} trades")
    ax.set_ylabel("Capital (base 100)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDCHF,USDJPY,XAUUSD")
    ap.add_argument("--crypto", default="")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--start", default="2021-01-04")
    ap.add_argument("--end", default="")
    ap.add_argument("--risk_ote", type=float, default=0.5)
    ap.add_argument("--risk_fvg", type=float, default=0.75)
    ap.add_argument("--leverage", type=float, default=0.0, help="0 = levier FTMO par actif (realism.py)")
    ap.add_argument("--mode", default="both", choices=["both", "ote_only", "fvg_only"])
    ap.add_argument("--sim_challenge", default="oui")
    ap.add_argument("--ftmo_mode", default="2phases", choices=["1phase", "2phases"])
    ap.add_argument("--use_best_ote", default="oui")
    ap.add_argument("--use_best_fvg", default="oui")
    ap.add_argument("--params_ote", default="")
    ap.add_argument("--params_fvg", default="")
    ap.add_argument("--fvg_skip", default="USDCHF,BTCUSD,ETHUSD,XRPUSD", help="Actifs ignorés par FVG en mode both (vide = aucun)")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    forex = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    crypto = [s.strip().upper() for s in a.crypto.split(",") if s.strip()]
    fvg_skip = {s.strip().upper() for s in a.fvg_skip.split(",") if s.strip()} if a.mode == "both" else set()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p_ote = load_params(strategy_ote, os.path.join(here, "results", "best_params_ote.json") if a.use_best_ote == "oui" else "", a.params_ote)
    p_fvg = load_params(strategy_fvg, os.path.join(here, "results", "best_params_fvg.json") if a.use_best_fvg == "oui" else "", a.params_fvg)
    tg(f"🚀 Backtest {a.mode} : {', '.join(forex + crypto)} depuis {a.start}\nOTE risque {a.risk_ote}% | FVG risque {a.risk_fvg}%")

    todo = [("FVG", a.risk_fvg, run_fvg, p_fvg)] if a.mode == "fvg_only" else [("OTE", a.risk_ote, run_ote, p_ote)] if a.mode == "ote_only" \
        else [("FVG", a.risk_fvg, run_fvg, p_fvg), ("OTE", a.risk_ote, run_ote, p_ote)]
    allt = {"FVG": [], "OTE": []}
    viewer_trades = []
    recap = []
    dl_start = min(a.start, "2021-01-01") if a.start else "2021-01-01"
    for sym in forex + crypto:
        try:
            if sym in crypto:
                ensure_data(sym, a.datadir, dl_start)
            P = data_fx.PairData(sym, a.datadir)
        except Exception as e:
            tg(f"❌ {sym} données : {e}")
            continue
        i0 = P.idx_at(a.start)
        i1 = P.idx_at(pd.Timestamp(a.end) + pd.Timedelta(days=1)) if a.end else P.n
        fee = realism.fee_for(sym)
        lev = a.leverage or realism.levier_for(sym)
        for name, risk, fn, p in todo:
            if name == "FVG" and (sym in fvg_skip or sym in crypto and "CRYPTO" in fvg_skip):
                continue
            try:
                tr, eq = fn(P, sym, i0, i1, risk, fee, lev, p)
                st = stats_of(tr, eq)
                txt = summary(sym, name, st, risk, fee)
                print(txt, flush=True)
                recap.append(f"{sym} {name}: {st['return_pct']:+.1f}% | {st['trades']}t | {st['avg_r']:+.2f}R | DD {st['max_dd_pct']:.1f}%")
                if len(tr):
                    tr = tr.copy()
                    tr.insert(0, "symbol", sym)
                    tr.insert(1, "strategy", name)
                    csv = os.path.join(a.outdir, f"{sym}_{name}_trades.csv")
                    png = os.path.join(a.outdir, f"{sym}_{name}_equity.png")
                    tr.to_csv(csv, index=False)
                    plot(sym, name, eq, st, png)
                    tg(txt, photo=png)
                    tg(document=csv)
                    allt[name].append(tr)
                    viewer_trades += ve.trades_to_json(tr, name, sym, p.get("be_rr", 0))
                else:
                    tg(txt + "\n⚠️ Aucun trade.")
            except Exception as e:
                msg = f"❌ {sym} {name} : {type(e).__name__}: {e}"
                print(msg)
                tg(msg)
                recap.append(f"{sym} {name}: erreur")

    if viewer_trades:
        vp = ve.write_trades_json(viewer_trades, os.path.join(a.outdir, "viewer_trades.json"))
        tg("🗺️ viewer_trades.json : ouvre ton visualiseur (viewer/viewer.html) > « Charger trades » (ou glisse le fichier). Seuls les trades sont dans ce fichier, les bougies sont déjà dans le visualiseur.", document=vp)
    tg("📋 RÉCAP\n" + "\n".join(recap))
    print("\n".join(recap))
    dfs = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in allt.items()}
    for k, d in dfs.items():
        if len(d):
            d.to_csv(os.path.join(a.outdir, f"ALL_{k}_trades.csv"), index=False)

    if a.sim_challenge == "oui" and any(len(d) for d in dfs.values()):
        lines = [f"🎯 Simulation FTMO {a.ftmo_mode} (départs tous les 3 j, pire cas intra-trade)"]
        combos = [(a.risk_ote, a.risk_fvg), (a.risk_ote * 0.75, a.risk_fvg * 0.75), (a.risk_ote * 1.5, a.risk_fvg * 1.5)]
        for ro, rf in combos:
            try:
                if a.mode == "both":
                    res = sim_ch.simulate_combined(dfs["OTE"], dfs["FVG"], CAPITAL, ro, rf, a.ftmo_mode, 3)
                    label = f"OTE {ro:g}% + FVG {rf:g}%"
                else:
                    k, r = ("FVG", rf) if a.mode == "fvg_only" else ("OTE", ro)
                    res = sim_ch.simulate_challenge(dfs[k], CAPITAL, r, a.ftmo_mode, 3)
                    label = f"{k} {r:g}%"
                lines.append(f"[{label}] réussite {res['success_pct']:.0f}% | échec DD jour {res['breach_daily_pct']:.0f}% / DD total "
                             f"{res['breach_total_pct']:.0f}% | non finis {res['timeout_pct']:.0f}% | durée médiane {res['median_days']:.0f} j | N={res['n_simulations']}")
            except Exception as e:
                lines.append(f"erreur simulation : {e}")
        txt = "\n".join(lines)
        print(txt)
        open(os.path.join(a.outdir, "simulation_ftmo.txt"), "w", encoding="utf-8").write(txt)
        tg(txt)


if __name__ == "__main__":
    main()


