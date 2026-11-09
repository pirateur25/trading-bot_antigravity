"""
backtest/run_backtest.py — point d'entrée : stratégies OTE (A), FVG (B) et/ou OPR sur forex + indices (NAS100, USA500, US30) + cryptos FTMO,
rapport Telegram (résumé global et par setup OPR Cassure / Réintégration, courbes, CSV détaillé des trades) et simulation FTMO.

Exemple :
  python backtest/run_backtest.py --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30 --mode all
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
import strategy_opr  # noqa: E402
import strategy_ote  # noqa: E402
import viewer_export as ve  # noqa: E402

CAPITAL = 10000.0

INDICES_OPR = {"NAS100", "US100", "USA500", "US500", "US30", "GER40"}


def normalize_sym(sym: str) -> str:
    s = sym.strip().upper()
    if s == "US100":
        return "NAS100"
    if s == "US500":
        return "USA500"
    return s


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


def stats_of(tr, eq, risk_pct=0.5):
    if len(tr) == 0:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "trades": 0, "avg_r": 0.0, "win_pct": 0.0, "pf": 0.0,
                "longs": 0, "shorts": 0, "by_year": {}, "by_side": {}, "by_setup": {}}
    eq = np.asarray(eq, float)
    eq = eq[~np.isnan(eq)]
    peak = np.maximum.accumulate(eq)
    r = tr["r"]
    wins, loss = r[r > 0].sum(), -r[r <= 0].sum()
    yr = pd.to_datetime(tr["entry_time"], utc=True).dt.year
    by_setup = strategy_opr.setup_stats(tr, risk_pct, CAPITAL) if "session" in tr.columns else {}
    return {
        "return_pct": (eq[-1] / CAPITAL - 1) * 100,
        "max_dd_pct": float(((eq - peak) / peak).min() * 100),
        "trades": len(tr),
        "avg_r": float(r.mean()),
        "win_pct": float((r > 0).mean() * 100),
        "pf": float(wins / loss) if loss > 0 else float("inf"),
        "longs": int((tr["side"] == "long").sum()),
        "shorts": int((tr["side"] == "short").sum()),
        "by_year": {str(y): (len(g), float(g["r"].mean())) for y, g in tr.groupby(yr)},
        "by_side": {s: (len(g), float(g["r"].mean())) for s, g in tr.groupby("side")},
        "by_setup": by_setup,
    }


def summary(sym, strat, st, risk, fee, extra=""):
    t = [f"📈 {sym} — {strat} | risque {risk}% | frais {fee:.4f}%/ordre",
         f"Rendement {st['return_pct']:+.2f}% | DD max {st['max_dd_pct']:.2f}% | {st['trades']} trades | gagnants {st['win_pct']:.1f}% | "
         f"espérance {st['avg_r']:+.3f} R | PF {st['pf']:.2f} | L/S {st['longs']}/{st['shorts']}"]
    if st.get("by_setup"):
        t.append("Par setup OPR :")
        for sname in strategy_opr.SETUP_ORDER:
            sd = st["by_setup"].get(sname, {})
            t.append(
                f"  • {sname:13s} : {sd.get('return_pct', 0.0):+6.2f}% | DD {sd.get('max_dd_pct', 0.0):6.2f}% | "
                f"{sd.get('trades', 0):3d}t | WR {sd.get('win_pct', 0.0):4.1f}% | {sd.get('avg_r', 0.0):+.2f}R (cumul {sd.get('sum_r', 0.0):+.1f}R)"
            )
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


def run_opr(P, sym, i0, i1, risk, fee, lev, p):
    return strategy_opr.run(P, capital=CAPITAL, risk_pct=risk, fee_pct=fee, leverage=lev, p=p, i0=i0, i1=i1)


def plot(sym, strat, eq, st, path, tr=None, risk=0.5):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.asarray(eq, float) / CAPITAL * 100, linewidth=1.3, label=f"Total {strat} ({st['return_pct']:+.1f}%)")
    ax.set_title(f"{sym} {strat} — {st['return_pct']:+.1f}% | DD {st['max_dd_pct']:.1f}% | {st['trades']} trades")
    ax.set_ylabel("Capital (base 100)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30")
    ap.add_argument("--crypto", default="")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--risk_ote", type=float, default=0.5)
    ap.add_argument("--risk_fvg", type=float, default=0.75)
    ap.add_argument("--risk_opr", type=float, default=0.5)
    ap.add_argument("--leverage", type=float, default=0.0, help="0 = levier FTMO par actif (realism.py)")
    ap.add_argument("--mode", default="all", choices=["all", "both", "opr_only", "ote_only", "fvg_only", "crypto_v4"])
    ap.add_argument("--sim_challenge", default="oui")
    ap.add_argument("--ftmo_mode", default="1phase", choices=["1phase", "2phases"])
    ap.add_argument("--use_best_ote", default="oui")
    ap.add_argument("--use_best_fvg", default="oui")
    ap.add_argument("--use_best_opr", default="oui")
    ap.add_argument("--params_ote", default="")
    ap.add_argument("--params_fvg", default="")
    ap.add_argument("--params_opr", default="")
    ap.add_argument("--fvg_skip", default="USDCHF,BTCUSD,ETHUSD,XRPUSD,NAS100,USA500,US30,US100,US500", help="Actifs ignorés par FVG en mode all/both")
    ap.add_argument("--ote_skip", default="NAS100,USA500,US30,US100,US500", help="Actifs ignorés par OTE en mode all/both")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.mode == "crypto_v4":
        import run_crypto_v4 as rcv4
        sys.argv = [
            sys.argv[0],
            "--symbols", a.crypto if a.crypto.strip() else ",".join(rcv4.scv4.CRYPTO_V4_SYMBOLS),
            "--datadir", a.datadir,
            "--start", a.start if a.start != "2015-01-01" else "2022-01-01",
            "--end", a.end,
            "--outdir", a.outdir,
        ]
        rcv4.main()
        return

    forex = [normalize_sym(s) for s in a.symbols.split(",") if s.strip()]
    crypto = [normalize_sym(s) for s in a.crypto.split(",") if s.strip()]
    fvg_skip = {normalize_sym(s) for s in a.fvg_skip.split(",") if s.strip()} if a.mode in ("both", "all") else set()
    ote_skip = {normalize_sym(s) for s in a.ote_skip.split(",") if s.strip()} if a.mode in ("both", "all") else set()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p_ote = load_params(strategy_ote, os.path.join(here, "results", "best_params_ote.json") if a.use_best_ote == "oui" else "", a.params_ote)
    p_fvg = load_params(strategy_fvg, os.path.join(here, "results", "best_params_fvg.json") if a.use_best_fvg == "oui" else "", a.params_fvg)
    p_opr = load_params(strategy_opr, os.path.join(here, "results", "best_params_opr.json") if a.use_best_opr == "oui" else "", a.params_opr)

    # Si on lance uniquement OPR et que seuls des forex étaient passés par erreur, s'assurer que les indices US sont inclus
    if a.mode == "opr_only" and not any(s in INDICES_OPR for s in forex):
        forex = ["NAS100", "USA500", "US30"]

    tg(f"🚀 Backtest {a.mode} : {', '.join(forex + crypto)} depuis {a.start}\nOTE {a.risk_ote}% | FVG {a.risk_fvg}% | OPR {a.risk_opr}%")

    if a.mode == "fvg_only":
        todo = [("FVG", a.risk_fvg, run_fvg, p_fvg)]
    elif a.mode == "ote_only":
        todo = [("OTE", a.risk_ote, run_ote, p_ote)]
    elif a.mode == "opr_only":
        todo = [("OPR", a.risk_opr, run_opr, p_opr)]
    elif a.mode == "both":
        todo = [("FVG", a.risk_fvg, run_fvg, p_fvg), ("OTE", a.risk_ote, run_ote, p_ote)]
    else:
        todo = [("FVG", a.risk_fvg, run_fvg, p_fvg), ("OTE", a.risk_ote, run_ote, p_ote), ("OPR", a.risk_opr, run_opr, p_opr)]

    allt = {"FVG": [], "OTE": [], "OPR": []}
    viewer_trades = []
    recap = []
    opr_setup_recap = []
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
            if name == "FVG" and (sym in fvg_skip or (sym in crypto and "CRYPTO" in fvg_skip)):
                continue
            if name == "OTE" and sym in ote_skip:
                continue
            if name == "OPR" and sym not in INDICES_OPR and a.mode in ("all", "opr_only"):
                continue
            try:
                tr, eq = fn(P, sym, i0, i1, risk, fee, lev, p)
                st = stats_of(tr, eq, risk_pct=risk)
                txt = summary(sym, name, st, risk, fee)
                print(txt, flush=True)
                recap.append(f"{sym} {name}: {st['return_pct']:+.1f}% | {st['trades']}t | {st['avg_r']:+.2f}R | DD {st['max_dd_pct']:.1f}%")
                if name == "OPR" and st.get("by_setup"):
                    setup_parts = []
                    for sname in strategy_opr.SETUP_ORDER:
                        sd = st["by_setup"][sname]
                        setup_parts.append(
                            f"  • {sname:13s}: {sd['return_pct']:+6.1f}% | DD {sd['max_dd_pct']:5.1f}% | {sd['trades']:3d}t | WR {sd['win_pct']:4.1f}% | {sd['avg_r']:+.2f}R ({sd['sum_r']:+.1f}R)"
                        )
                    opr_setup_recap.append(f"{sym} OPR par setup (risque {risk}%) :\n" + "\n".join(setup_parts))
                if len(tr):
                    tr = tr.copy()
                    tr.insert(0, "symbol", sym)
                    tr.insert(1, "strategy", name)
                    csv = os.path.join(a.outdir, f"{sym}_{name}_trades.csv")
                    png = os.path.join(a.outdir, f"{sym}_{name}_equity.png")
                    tr.to_csv(csv, index=False)
                    plot(sym, name, eq, st, png, tr=tr, risk=risk)
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
        tg("🗺️ viewer_trades.json : ouvre ton visualiseur (viewer/viewer.html) > « Charger trades » (ou glisse le fichier).", document=vp)

    dfs = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in allt.items()}
    for k, d in dfs.items():
        if len(d):
            d.to_csv(os.path.join(a.outdir, f"ALL_{k}_trades.csv"), index=False)

    # Bilan global par setup pour OPR (tous indices confondus)
    if len(dfs["OPR"]) > 0:
        global_setup = strategy_opr.setup_stats(dfs["OPR"], a.risk_opr, CAPITAL)
        g_lines = [f"🌍 TOTAL OPR PAR SETUP (tous indices, risque {a.risk_opr}%) :"]
        for sname in strategy_opr.SETUP_ORDER:
            sd = global_setup[sname]
            g_lines.append(
                f"  • {sname:13s}: {sd['return_pct']:+6.1f}% | DD {sd['max_dd_pct']:5.1f}% | {sd['trades']:4d}t | WR {sd['win_pct']:4.1f}% | {sd['avg_r']:+.2f}R (cumul {sd['sum_r']:+.1f}R)"
            )
        opr_setup_recap.append("\n".join(g_lines))

    full_recap = "📋 RÉCAP\n" + "\n".join(recap)
    if opr_setup_recap:
        full_recap += "\n\n🕒 BILAN OPR PAR SETUP (Cassure RR>=2.25 dyn RV3 | Réintégration RR>=2.0, 15h30-19h30 FR)\n" + "\n\n".join(opr_setup_recap)
    tg(full_recap)
    print(full_recap)

    if a.sim_challenge == "oui" and any(len(d) for d in dfs.values()):
        lines = []
        for fmode in ("1phase", "2phases"):
            lines.append(f"🎯 Simulation FTMO {fmode} (départs tous les 3 j, pire cas intra-trade, superposition active)")
            # 1) OPR seule (0.5%, 0.75%, 1.0% car DD max OPR ~5% à 0.5%)
            if len(dfs["OPR"]) > 0:
                for ropr in (a.risk_opr, a.risk_opr * 1.5, a.risk_opr * 2.0):
                    res = sim_ch.simulate_challenge(dfs["OPR"], CAPITAL, ropr, fmode, 3)
                    lines.append(
                        f"[OPR seule {ropr:g}%] réussite {res['success_pct']:.0f}% | échec DD jour {res['breach_daily_pct']:.0f}% / DD total "
                        f"{res['breach_total_pct']:.0f}% | non finis {res['timeout_pct']:.0f}% | durée médiane {res['median_days']:.0f} j | N={res['n_simulations']}"
                    )
            # 2) FVG + OTE + OPR (si tous présents)
            if len(dfs["OTE"]) > 0 and len(dfs["FVG"]) > 0 and len(dfs["OPR"]) > 0:
                for mult in (1.0, 0.75, 1.5):
                    ro, rf, ropr = a.risk_ote * mult, a.risk_fvg * mult, a.risk_opr * mult
                    res = sim_ch.simulate_combined(dfs["OTE"], dfs["FVG"], CAPITAL, ro, rf, fmode, 3, trades_sb=dfs["OPR"], risk_sb=ropr)
                    lines.append(
                        f"[OTE {ro:g}% + FVG {rf:g}% + OPR {ropr:g}%] réussite {res['success_pct']:.0f}% | échec DD jour {res['breach_daily_pct']:.0f}% / DD total "
                        f"{res['breach_total_pct']:.0f}% | non finis {res['timeout_pct']:.0f}% | durée médiane {res['median_days']:.0f} j | N={res['n_simulations']}"
                    )
            # 3) FVG + OTE seuls (référence sans OPR)
            if len(dfs["OTE"]) > 0 and len(dfs["FVG"]) > 0:
                for mult in (1.0, 0.75, 1.5):
                    ro, rf = a.risk_ote * mult, a.risk_fvg * mult
                    res = sim_ch.simulate_combined(dfs["OTE"], dfs["FVG"], CAPITAL, ro, rf, fmode, 3)
                    lines.append(
                        f"[OTE {ro:g}% + FVG {rf:g}%] réussite {res['success_pct']:.0f}% | échec DD jour {res['breach_daily_pct']:.0f}% / DD total "
                        f"{res['breach_total_pct']:.0f}% | non finis {res['timeout_pct']:.0f}% | durée médiane {res['median_days']:.0f} j | N={res['n_simulations']}"
                    )
            lines.append("")
        txt = "\n".join(lines).strip()
        print(txt)
        open(os.path.join(a.outdir, "simulation_ftmo.txt"), "w", encoding="utf-8").write(txt)
        tg(txt)


if __name__ == "__main__":
    main()
