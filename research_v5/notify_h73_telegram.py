"""
research_v5/notify_h73_telegram.py — Notificateur Telegram quotidien pour la stratégie validée `H7.3`
(1/3 Indices + 1/3 Or + 1/3 BTC, Donchian {20, 60, 120}j Long-Only, validée en Holdout).

Fonctionnalités :
  - Télécharge les dernières séances quotidiennes Yahoo Finance (200+ jours) pour les 7 actifs H7.3.
  - Calcule les signaux exacts Donchian {20, 60, 120}j, la vol 20j et les poids cibles H7.3.
  - Applique la zone morte (deadband = 0.05) par rapport à l'état sauvegardé (`--state_path`).
  - Affiche le bandeau FEU VERT / ATTENDRE (règle causale ≥ 2/3 piliers actifs pour lancer un challenge).
  - Envoie le rapport clair sur Telegram via les variables d'environnement `BOT_TOKEN` et `CHAT_ID`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

EQUITY_SYMS = ["US500", "US100", "US30", "GER40", "JP225"]
H73_SYMS = EQUITY_SYMS + ["XAUUSD", "BTCUSD"]
TICKERS = {
    "US500": "^GSPC",
    "US100": "^NDX",
    "US30": "^DJI",
    "GER40": "^GDAXI",
    "JP225": "^N225",
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
}


def fetch_yahoo_d1(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    now_ts = int(time.time()) + 86400
    start_ts = now_ts - lookback_days * 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?period1={start_ts}&period2={now_ts}&interval=1d&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).normalize(),
        "open": q["open"],
        "high": q["high"],
        "low": q["low"],
        "close": q["close"],
    }).dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df.set_index("date")


def compute_long_only_donchian(cl: pd.DataFrame, hi: pd.DataFrame, lo: pd.DataFrame) -> pd.DataFrame:
    s_tot = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
    for L in (20, 60, 120):
        rh = hi.shift(1).rolling(L, min_periods=L).max()
        rl = lo.shift(1).rolling(L, min_periods=L).min()
        sig = pd.DataFrame(np.nan, index=cl.index, columns=cl.columns)
        sig[cl > rh] = 1.0
        sig[cl < rl] = 0.0
        sig = sig.ffill().fillna(0.0)
        s_tot += sig / 3.0
    return s_tot


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            raise RuntimeError(f"Erreur Telegram API : {body}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Signal quotidien Telegram H7.3 (Prop Firm FTMO / The5ers)")
    ap.add_argument("--capital", type=float, default=10000.0, help="Capital du compte Prop Firm en USD (défaut: 10000)")
    ap.add_argument("--risk_mult", type=float, default=1.0, help="Multiplicateur de risque (1.0 = standard, 1.25 = rapide The5ers)")
    ap.add_argument("--state_path", type=str, default="research_v5/state_h73_signal.json", help="Fichier d'état pour la zone morte (deadband)")
    ap.add_argument("--dry_run", action="store_true", help="Affiche le message sans envoyer sur Telegram ni écraser l'état")
    args = ap.parse_args()

    raw = {}
    for sym, tk in TICKERS.items():
        raw[sym] = fetch_yahoo_d1(tk, lookback_days=365)
        time.sleep(0.2)

    # Index unifié sur les séances US500 + dernière séance BTC si week-end
    idx = raw["US500"].index
    cl = pd.DataFrame({s: raw[s]["close"].reindex(idx).ffill() for s in H73_SYMS})
    hi = pd.DataFrame({s: raw[s]["high"].reindex(idx).ffill() for s in H73_SYMS})
    lo = pd.DataFrame({s: raw[s]["low"].reindex(idx).ffill() for s in H73_SYMS})

    signals = compute_long_only_donchian(cl, hi, lo)
    vol20 = cl.pct_change(fill_method=None).rolling(20, min_periods=10).std().clip(lower=0.004)

    last_dt = cl.index[-1]
    last_date_str = str(last_dt.date())

    budgets = {s: 1.0 / 15.0 for s in EQUITY_SYMS}
    budgets["XAUUSD"] = 1.0 / 3.0
    budgets["BTCUSD"] = 1.0 / 3.0

    caps = {s: 1.5 * args.risk_mult for s in EQUITY_SYMS}
    caps["XAUUSD"] = 1.5 * args.risk_mult
    caps["BTCUSD"] = 0.5 * args.risk_mult

    sigma_target = 0.012 * args.risk_mult
    deadband = 0.05 * args.risk_mult
    kappa = math.sqrt(3.0)

    state_file = Path(args.state_path)
    prev_w = {s: 0.0 for s in H73_SYMS}
    if state_file.exists():
        try:
            saved = json.loads(state_file.read_text(encoding="utf-8"))
            prev_w.update({s: float(saved.get("weights", {}).get(s, 0.0)) for s in H73_SYMS})
        except Exception:
            pass

    target_w, new_w, sig_now, vol_now, px_now, actions = {}, {}, {}, {}, {}, {}
    for s in H73_SYMS:
        sv = float(signals.loc[last_dt, s])
        vv = float(vol20.loc[last_dt, s])
        px = float(cl.loc[last_dt, s])
        w_raw = budgets[s] * sv * (sigma_target / vv) * kappa
        tw = float(np.clip(w_raw, 0.0, caps[s]))
        nw = tw if abs(tw - prev_w[s]) >= deadband else prev_w[s]

        sig_now[s], vol_now[s], px_now[s] = sv, vv, px
        target_w[s], new_w[s] = tw, nw

        if nw == 0.0 and prev_w[s] == 0.0:
            actions[s] = "CASH"
        elif nw == 0.0 and prev_w[s] > 0.0:
            actions[s] = "🔴 FERMER"
        elif prev_w[s] == 0.0 and nw > 0.0:
            actions[s] = "🟢 OUVRIR LONG"
        elif nw > prev_w[s]:
            actions[s] = "🔼 RENFORCER"
        elif nw < prev_w[s]:
            actions[s] = "🔽 ALLÉGER"
        else:
            actions[s] = "⏸ GARDER"

    p_eq = float(np.mean([sig_now[s] for s in EQUITY_SYMS]))
    p_gd = sig_now["XAUUSD"]
    p_bt = sig_now["BTCUSD"]
    p_sum = p_eq + p_gd + p_bt

    if p_sum >= 2.0:
        launch_banner = f"🟢 <b>FEU VERT CHALLENGE</b> ({p_sum:.2f} / 3 piliers actifs ≥ 2,0)"
    else:
        launch_banner = f"🔴 <b>ATTENDRE AVANT D'ACHETER</b> ({p_sum:.2f} / 3 piliers actifs &lt; 2,0)"

    tot_notional = sum(new_w.values()) * args.capital
    tot_lev = sum(new_w.values())
    stop_usd = args.capital * 0.038

    lines = [
        f"📊 <b>SIGNAL QUOTIDIEN H7.3</b> — <code>{last_date_str}</code>",
        f"Compte : <b>{args.capital:,.0f} $</b> | Risque : <b>{args.risk_mult:.2f}x</b>",
        "",
        launch_banner,
        f"• Indices : <code>{p_eq*100:.0f}%</code> | Or : <code>{p_gd*100:.0f}%</code> | BTC : <code>{p_bt*100:.0f}%</code>",
        "",
        "<b>Positions cibles (à exécuter à l'ouverture) :</b>",
    ]
    for s in H73_SYMS:
        usd = new_w[s] * args.capital
        delta_usd = (new_w[s] - prev_w[s]) * args.capital
        d_str = f" ({delta_usd:+,.0f} $)" if abs(delta_usd) >= 1.0 else ""
        lines.append(
            f"• <b>{s}</b> [{sig_now[s]*3:.0f}/3] : <b>{usd:,.0f} $</b>{d_str} — {actions[s]} <i>(px {px_now[s]:,.2f})</i>"
        )

    lines.extend([
        "",
        f"💼 Exposition totale : <b>{tot_notional:,.0f} $</b> (<code>{tot_lev:.2f}x</code>)",
        f"🛡 Stop journalier sécurité (-3,8 %) : <b>-{stop_usd:,.0f} $</b>",
        f"🕒 Calculé le {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ])
    msg = "\n".join(lines)
    print(msg)

    if not args.dry_run:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({
                "date": last_date_str,
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "capital": args.capital,
                "risk_mult": args.risk_mult,
                "pillars_sum": round(p_sum, 4),
                "weights": {s: round(new_w[s], 6) for s in H73_SYMS},
            }, indent=2),
            encoding="utf-8",
        )
        token = os.environ.get("BOT_TOKEN", "").strip()
        chat_id = os.environ.get("CHAT_ID", "").strip()
        if token and chat_id:
            send_telegram(token, chat_id, msg)
            print("[Telegram] Notification envoyée avec succès.")
        else:
            print("[Telegram] BOT_TOKEN ou CHAT_ID manquant — envoi ignoré.")


if __name__ == "__main__":
    main()
