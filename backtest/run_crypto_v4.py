"""
backtest/run_crypto_v4.py — Backtest complet de la stratégie CRYPTO V4 (A2-R — Stratégie SMC — Crypto Futures)
sur les 9 cryptos du PDF : BTC, ETH, SOL, XRP, AVAX, ZEC, INJ, PUMP, HYPE.

Génère :
  - Le rapport complet (Bilan par crypto, RR moyen, Gains/RR et Drawdown par Année, Gains/RR et Drawdown par Mois)
  - Les CSV : `ALL_CRYPTO_V4_trades.csv`, `CRYPTO_V4_monthly.csv`, `CRYPTO_V4_yearly.csv`, `<SYM>_CRYPTO_V4_trades.csv`
  - Les graphiques : `PORTFOLIO_CRYPTO_V4.png` (courbe d'équité + drawdown + histogramme mensuel) et `<SYM>_CRYPTO_V4_equity.png`
  - Le fichier `viewer_trades.json` pour visualiser les trades dans `viewer/viewer.html`
  - Envoi automatique sur Telegram (si BOT_TOKEN et CHAT_ID sont définis)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_crypto as dc  # noqa: E402
import data_forex as data_fx  # noqa: E402
import strategy_crypto_v4 as scv4  # noqa: E402
import viewer_export as ve  # noqa: E402


def normalize_crypto_sym(sym: str) -> str:
    s = sym.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if not s:
        return ""
    if s.endswith("USDT"):
        return s[:-1]  # BTCUSDT -> BTCUSD
    if s.endswith("USD"):
        return s
    return s + "USD"   # BTC -> BTCUSD, HYPE -> HYPEUSD


def tg(text=None, photo=None, document=None):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        print("TELEGRAM: BOT_TOKEN/CHAT_ID manquant -> ignoré")
        return
    try:
        if text:
            for i in range(0, len(text), 4000):
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat, "text": text[i:i + 4000]},
                    timeout=20,
                )
        if photo and os.path.exists(photo):
            with open(photo, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat},
                    files={"photo": f},
                    timeout=30,
                )
        if document and os.path.exists(document):
            with open(document, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendDocument",
                    data={"chat_id": chat},
                    files={"document": f},
                    timeout=60,
                )
    except requests.RequestException as e:
        print("TELEGRAM erreur:", e)


def ensure_crypto_data(sym_usd: str, datadir: str, start_date: str):
    os.makedirs(datadir, exist_ok=True)
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(datadir, sym_usd + ext)
        if os.path.exists(p):
            return p
    binance_sym = dc.ftmo_to_binance(sym_usd)
    dl_start = min(start_date, "2022-01-01") if start_date else "2022-01-01"
    print(f"[data] Téléchargement {sym_usd} ({binance_sym}) en 5m depuis {dl_start}...", flush=True)
    df, src = dc.load(binance_sym, "5m", dl_start, source="auto", use_cache=False)
    out = os.path.join(datadir, sym_usd + ".csv.gz")
    df[["time", "open", "high", "low", "close", "volume"]].to_csv(out, index=False, compression="gzip")
    print(f"  ✅ {sym_usd}: {len(df)} bougies 5m ({src}) -> {out}", flush=True)
    return out


def plot_symbol_equity(sym: str, tr: pd.DataFrame, start_cap: float, path: str):
    if len(tr) == 0:
        return
    df = tr.sort_values("exit_time").reset_index(drop=True)
    pnl = df["pnl_usdt"].to_numpy(dtype=float)
    r_arr = df["r"].to_numpy(dtype=float)
    eq_u = np.concatenate([[start_cap], start_cap + np.cumsum(pnl)])
    peak_u = np.maximum.accumulate(eq_u)
    dd_u = eq_u - peak_u

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eq_u, linewidth=1.4, color="#1f77b4", label=f"{sym} ({pnl.sum():+,.0f} USDT | {r_arr.sum():+.1f}R)")
    ax.axhline(start_cap, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        f"{sym} CRYPTO V4 — Gain {pnl.sum():+,.0f} USDT ({r_arr.sum():+.1f}R) | "
        f"RR moy {r_arr.mean():+.3f}R | DD max {dd_u.min():+,.0f} USDT | {len(df)} trades"
    )
    ax.set_ylabel("Capital (USDT)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_portfolio_overview(all_trades: pd.DataFrame, mo_df: pd.DataFrame, start_cap: float, path: str):
    if all_trades is None or len(all_trades) == 0:
        return
    df = all_trades.copy()
    df["_exit_dt"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values("_exit_dt").reset_index(drop=True)

    times = pd.concat([pd.Series([df["_exit_dt"].iloc[0]]), df["_exit_dt"]]).reset_index(drop=True)
    pnl = df["pnl_usdt"].to_numpy(dtype=float)
    r_arr = df["r"].to_numpy(dtype=float)
    eq_u = np.concatenate([[start_cap], start_cap + np.cumsum(pnl)])
    peak_u = np.maximum.accumulate(eq_u)
    dd_u = eq_u - peak_u

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2.2, 1.2, 1.6]})

    # 1) Courbe d'équité portefeuille
    ax1.plot(times, eq_u, color="#10b981", linewidth=1.6, label=f"Capital Portefeuille ({pnl.sum():+,.0f} USDT | {r_arr.sum():+.1f}R)")
    ax1.plot(times, peak_u, color="#94a3b8", linestyle="--", linewidth=0.9, label="Plus haut historique (High-Water Mark)")
    ax1.set_title(
        f"PORTEFEUILLE CRYPTO V4 ({df['symbol'].nunique()} cryptos) — "
        f"Gain {pnl.sum():+,.0f} USDT ({r_arr.sum():+.1f}R) | RR moy {r_arr.mean():+.3f}R | "
        f"DD max {dd_u.min():+,.0f} USDT | {len(df)} trades"
    )
    ax1.set_ylabel("Capital (USDT)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")

    # 2) Courbe de Drawdown
    ax2.fill_between(times, dd_u, 0, color="#ef4444", alpha=0.35, label="Drawdown (USDT)")
    ax2.plot(times, dd_u, color="#dc2626", linewidth=1.0)
    ax2.set_ylabel("Drawdown (USDT)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower left")

    # 3) Gains en R par mois + Drawdown mensuel
    if len(mo_df) > 0:
        x = np.arange(len(mo_df))
        vals_r = mo_df["sum_r"].to_numpy(dtype=float)
        dd_r = mo_df["max_dd_r"].to_numpy(dtype=float)
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals_r]
        ax3.bar(x, vals_r, color=colors, alpha=0.85, label="Gain net mensuel (R)")
        ax3.plot(x, dd_r, color="#f59e0b", marker="o", markersize=3.5, linewidth=1.2, label="Drawdown max du mois (R)")
        ax3.axhline(0.0, color="gray", linewidth=0.8)
        step = max(1, len(x) // 18)
        ax3.set_xticks(x[::step])
        ax3.set_xticklabels(mo_df["period"].iloc[::step], rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Mensuel (en R)")
        ax3.grid(alpha=0.3)
        ax3.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Backtest CRYPTO V4 (A2-R — Stratégie SMC — Crypto Futures)")
    ap.add_argument(
        "--symbols",
        default=",".join(scv4.CRYPTO_V4_SYMBOLS),
        help="Cryptos à backtester (défaut: BTC,ETH,SOL,XRP,AVAX,ZEC,INJ,PUMP,HYPE)",
    )
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--capital", type=float, default=1800.0, help="Capital initial portefeuille USDT (défaut: 9 x 200 = 1800 USDT)")
    ap.add_argument("--fee_pct", type=float, default=0.02, help="Frais par ordre en %% du notionnel (0.02 = 0.02%%, 0 = sans frais)")
    ap.add_argument("--trail_mode", default="check_30m", choices=["check_30m", "intra"])
    ap.add_argument("--params", default="", help="Réglages supplémentaires k=v;k2=v2")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    syms = []
    for raw_s in a.symbols.split(","):
        ns = normalize_crypto_sym(raw_s)
        if ns and ns not in syms:
            syms.append(ns)

    p = scv4.parse_params(a.params)
    p["fee_pct"] = float(a.fee_pct)
    p["trail_mode"] = a.trail_mode
    margin_usdt = float(p.get("margin_usdt", 200.0))
    one_r_usdt = margin_usdt * (float(p.get("sl_roi_pct", 50.0)) / 100.0)
    start_cap = float(a.capital) if a.capital > 0 else float(len(syms) * margin_usdt)

    tg(
        f"🚀 Backtest CRYPTO V4 (SMC Futures x{p['leverage']:g}, marge {margin_usdt:.0f} USDT)\n"
        f"Actifs ({len(syms)}) : {', '.join(syms)}\n"
        f"Période : depuis {a.start} | Frais : {a.fee_pct:.3f}%/ordre | Trail : {a.trail_mode}"
    )

    all_trades_list = []
    per_sym_stats = []
    viewer_trades = []

    for sym in syms:
        try:
            ensure_crypto_data(sym, a.datadir, a.start)
            P = data_fx.PairData(sym, a.datadir)
        except Exception as e:
            msg = f"❌ {sym} données indisponibles : {e}"
            print(msg, flush=True)
            tg(msg)
            continue

        i0 = P.idx_at(a.start) if a.start else 0
        i1 = P.idx_at(pd.Timestamp(a.end) + pd.Timedelta(days=1)) if a.end else P.n

        try:
            tr, eq = scv4.run(P, capital=start_cap, risk_pct=0.0, fee_pct=a.fee_pct, leverage=p["leverage"], p=p, i0=i0, i1=i1)
        except Exception as e:
            msg = f"❌ {sym} erreur exécution : {type(e).__name__}: {e}"
            print(msg, flush=True)
            tg(msg)
            continue

        n_t = len(tr)
        if n_t > 0:
            tr = tr.copy()
            tr.insert(0, "symbol", sym)
            tr.insert(1, "strategy", "CRYPTO_V4")

            r_arr = tr["r"].to_numpy(dtype=float)
            pnl_arr = tr["pnl_usdt"].to_numpy(dtype=float)
            wins = int((r_arr > 0.05).sum())
            be = int((np.abs(r_arr) <= 0.05).sum())
            eq_r = np.concatenate([[0.0], np.cumsum(r_arr)])
            dd_r = float((eq_r - np.maximum.accumulate(eq_r)).min())
            eq_u = np.concatenate([[start_cap], start_cap + np.cumsum(pnl_arr)])
            peak_u = np.maximum.accumulate(eq_u)
            dd_u = float((eq_u - peak_u).min())
            dd_pct = float(((eq_u - peak_u) / np.maximum(peak_u, 1.0)).min() * 100.0)

            st_sym = {
                "symbol": sym,
                "trades": n_t,
                "win_pct": float(wins / n_t * 100.0),
                "be_pct": float(be / n_t * 100.0),
                "avg_r": float(r_arr.mean()),
                "sum_r": float(r_arr.sum()),
                "pnl_usdt": float(pnl_arr.sum()),
                "max_dd_r": dd_r,
                "max_dd_usdt": dd_u,
                "max_dd_pct": dd_pct,
            }
            per_sym_stats.append(st_sym)
            all_trades_list.append(tr)

            csv_sym = os.path.join(a.outdir, f"{sym}_CRYPTO_V4_trades.csv")
            png_sym = os.path.join(a.outdir, f"{sym}_CRYPTO_V4_equity.png")
            tr.to_csv(csv_sym, index=False)
            plot_symbol_equity(sym, tr, start_cap, png_sym)
            viewer_trades += ve.trades_to_json(tr, "CRYPTO_V4", sym, 1.0)

            print(
                f"✅ {sym:8s} : {n_t:4d} trades | WR {st_sym['win_pct']:4.1f}% (BE {st_sym['be_pct']:4.1f}%) | "
                f"RR moy {st_sym['avg_r']:+.3f}R | Total {st_sym['sum_r']:+7.1f}R ({st_sym['pnl_usdt']:+8.0f} USDT) | "
                f"DD {st_sym['max_dd_r']:6.1f}R ({st_sym['max_dd_usdt']:+7.0f} USDT)",
                flush=True,
            )
        else:
            per_sym_stats.append({
                "symbol": sym, "trades": 0, "win_pct": 0.0, "be_pct": 0.0,
                "avg_r": 0.0, "sum_r": 0.0, "pnl_usdt": 0.0,
                "max_dd_r": 0.0, "max_dd_usdt": 0.0, "max_dd_pct": 0.0,
            })
            print(f"⚠️ {sym:8s} : 0 trade", flush=True)

    all_trades = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    if len(all_trades) > 0:
        all_trades["_sort_ts"] = pd.to_datetime(all_trades["exit_time"], utc=True)
        all_trades = all_trades.sort_values("_sort_ts").drop(columns=["_sort_ts"]).reset_index(drop=True)

        all_csv = os.path.join(a.outdir, "ALL_CRYPTO_V4_trades.csv")
        mo_csv = os.path.join(a.outdir, "CRYPTO_V4_monthly.csv")
        yr_csv = os.path.join(a.outdir, "CRYPTO_V4_yearly.csv")
        port_png = os.path.join(a.outdir, "PORTFOLIO_CRYPTO_V4.png")

        all_trades.to_csv(all_csv, index=False)
        mo_df = scv4.compute_period_table(all_trades, period="M", start_capital=start_cap, one_r_usdt=one_r_usdt)
        yr_df = scv4.compute_period_table(all_trades, period="Y", start_capital=start_cap, one_r_usdt=one_r_usdt)
        mo_df.to_csv(mo_csv, index=False)
        yr_df.to_csv(yr_csv, index=False)
        plot_portfolio_overview(all_trades, mo_df, start_cap, port_png)

        if viewer_trades:
            ve.write_trades_json(viewer_trades, os.path.join(a.outdir, "viewer_trades.json"))

        report_txt = scv4.format_crypto_v4_report(all_trades, per_sym_stats, start_capital=start_cap, p=p)
        rep_path = os.path.join(a.outdir, "CRYPTO_V4_report.txt")
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report_txt)

        print("\n" + report_txt, flush=True)
        tg(report_txt, photo=port_png)
        tg(document=mo_csv)
        tg(document=yr_csv)
        tg(document=all_csv)
    else:
        msg = "⚠️ Aucun trade CRYPTO V4 généré sur la période."
        print(msg, flush=True)
        tg(msg)


if __name__ == "__main__":
    main()

