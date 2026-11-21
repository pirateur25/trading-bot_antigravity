"""
research_v5/live_h73_ctrader.py — Bot de trading automatique `H7.3` pour **cTrader Open API** (100 % officiel Spotware,
0 €/mois sur GitHub Actions + cron-job.org). Fonctionne sur compte Démo gratuit (IC Markets / Pepperstone / Spotware)
ET sur The5ers High Stakes cTrader simplement en changeant `CTRADER_ACCOUNT_ID` et `--env demo|live`.

Deux modes d'exécution :
  1. `--mode daily` (1×/jour après clôture US, ex. 22:15 UTC) :
     - Calcule les signaux exacts H7.3 (1/3 Indices + 1/3 Or + 1/3 BTC, Donchian {20,60,120}j Long-Only).
     - Se connecte à cTrader Open API, récupère solde, positions ouvertes et spécifications exactes des symboles.
     - Applique la zone morte (deadband = 0.05) et envoie les ordres d'ajustement avec Stop-Loss natif serveur.
     - Envoie le résumé complet sur Telegram (`BOT_TOKEN`, `CHAT_ID`).
  2. `--mode guard` (toutes les 5 minutes pendant la journée, dépôt public = illimité) :
     - Se connecte à cTrader Open API, calcule l'équité flottante en temps réel (`balance + netUnrealizedPnL`).
     - Si perte journalière <= `-3,8 %` (vs solde début de journée UTC) OU drawdown total <= `-8,5 %` (vs capital initial) :
       ferme IMMÉDIATEMENT toutes les positions ouvertes, verrouille la journée UTC et envoie une alerte 🚨 Telegram.
     - Si tout va bien, quitte silencieusement en ~3–5 secondes (zéro spam Telegram, zéro commit inutile).

Variables d'environnement requises (GitHub Secrets) :
  CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID
  BOT_TOKEN, CHAT_ID (optionnels pour Telegram)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

EQUITY_SYMS = ["US500", "US100", "US30", "GER40", "JP225"]
H73_SYMS = EQUITY_SYMS + ["XAUUSD", "BTCUSD"]

YAHOO_TICKERS = {
    "US500": "^GSPC",
    "US100": "^NDX",
    "US30": "^DJI",
    "GER40": "^GDAXI",
    "JP225": "^N225",
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
}

# Alias automatiques pour compatibilité IC Markets, Pepperstone, FTMO, The5ers cTrader
BROKER_ALIASES = {
    "US500": ["US500", "US500.cash", "SPX500", "SP500", "US500USD", "USA500"],
    "US100": ["US100", "US100.cash", "USTEC", "NAS100", "NDX100", "US100USD", "USATEC"],
    "US30": ["US30", "US30.cash", "DJ30", "WS30", "US30USD", "USA30"],
    "GER40": ["GER40", "GER40.cash", "DE40", "DAX40", "DE30", "GER30", "DE40EUR"],
    "JP225": ["JP225", "JP225.cash", "JPN225", "NIK225", "JP225USD"],
    "XAUUSD": ["XAUUSD", "GOLD", "XAUUSD.cash"],
    "BTCUSD": ["BTCUSD", "BTC/USD", "BITCOIN", "BTCUSDT"],
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
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
    }).dropna(subset=["open", "high", "low", "close"])
    return df.drop_duplicates(subset=["date"], keep="last").sort_values("date").set_index("date")


def compute_h73_targets(risk_mult: float = 1.0) -> dict:
    raw = {}
    for sym, tk in YAHOO_TICKERS.items():
        raw[sym] = fetch_yahoo_d1(tk, lookback_days=365)
        time.sleep(0.2)

    idx = raw["US500"].index
    cl = pd.DataFrame({s: raw[s]["close"].reindex(idx).ffill() for s in H73_SYMS})
    hi = pd.DataFrame({s: raw[s]["high"].reindex(idx).ffill() for s in H73_SYMS})
    lo = pd.DataFrame({s: raw[s]["low"].reindex(idx).ffill() for s in H73_SYMS})

    s_tot = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
    for L in (20, 60, 120):
        rh = hi.shift(1).rolling(L, min_periods=L).max()
        rl = lo.shift(1).rolling(L, min_periods=L).min()
        sig = pd.DataFrame(np.nan, index=cl.index, columns=cl.columns)
        sig[cl > rh] = 1.0
        sig[cl < rl] = 0.0
        s_tot += sig.ffill().fillna(0.0) / 3.0

    vol20 = cl.pct_change(fill_method=None).rolling(20, min_periods=10).std().clip(lower=0.004)
    last_dt = cl.index[-1]

    budgets = {s: 1.0 / 15.0 for s in EQUITY_SYMS}
    budgets["XAUUSD"] = 1.0 / 3.0
    budgets["BTCUSD"] = 1.0 / 3.0

    caps = {s: 1.5 * risk_mult for s in EQUITY_SYMS}
    caps["XAUUSD"] = 1.5 * risk_mult
    caps["BTCUSD"] = 0.5 * risk_mult

    sigma_target = 0.012 * risk_mult
    kappa = math.sqrt(3.0)

    weights, sigs, prices = {}, {}, {}
    for s in H73_SYMS:
        sv = float(s_tot.loc[last_dt, s])
        vv = float(vol20.loc[last_dt, s])
        px = float(cl.loc[last_dt, s])
        weights[s] = float(np.clip(budgets[s] * sv * (sigma_target / vv) * kappa, 0.0, caps[s]))
        sigs[s] = sv
        prices[s] = px

    p_sum = float(np.mean([sigs[s] for s in EQUITY_SYMS])) + sigs["XAUUSD"] + sigs["BTCUSD"]
    return {"date": str(last_dt.date()), "weights": weights, "signals": sigs, "prices": prices, "pillars_sum": p_sum}


def send_telegram(text: str) -> None:
    token = os.environ.get("BOT_TOKEN", "").strip()
    chat_id = os.environ.get("CHAT_ID", "").strip()
    if not token or not chat_id:
        print("[Telegram] BOT_TOKEN/CHAT_ID non configurés — message affiché en console uniquement.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            print(f"[Telegram] Erreur API : {body}")


def load_state(path: Path, initial_capital: float) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "initial_capital": initial_capital,
        "day_utc": "",
        "day_start_balance": initial_capital,
        "halted_utc_date": "",
        "weights": {s: 0.0 for s in H73_SYMS},
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def match_broker_symbols(light_symbols: list) -> dict[str, tuple[int, str]]:
    """Associe chaque actif H7.3 au symbolId exact du broker cTrader."""
    by_upper = {}
    for sym in light_symbols:
        name = getattr(sym, "symbolName", "")
        if name:
            by_upper[name.upper()] = (int(sym.symbolId), name)

    matched = {}
    for h_sym, aliases in BROKER_ALIASES.items():
        for al in aliases:
            if al.upper() in by_upper:
                matched[h_sym] = by_upper[al.upper()]
                break
        if h_sym not in matched:
            # Fallback recherche par préfixe
            for k, v in by_upper.items():
                if any(k.startswith(al.upper()) for al in aliases):
                    matched[h_sym] = v
                    break
    return matched


def round_volume(raw_vol: float, min_vol: int, step_vol: int, max_vol: int) -> int:
    if raw_vol < min_vol * 0.5:
        return 0
    stepped = int(round(raw_vol / max(1, step_vol))) * max(1, step_vol)
    if stepped < min_vol:
        stepped = min_vol
    return min(stepped, max_vol) if max_vol > 0 else stepped


def run_ctrader_session(args: argparse.Namespace) -> int:
    from twisted.internet import reactor
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoOAPayloadType
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
        ProtoOATraderReq,
        ProtoOAReconcileReq,
        ProtoOAGetPositionUnrealizedPnLReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolByIdReq,
        ProtoOANewOrderReq,
        ProtoOAClosePositionReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAOrderType, ProtoOATradeSide

    client_id = os.environ.get("CTRADER_CLIENT_ID", "").strip()
    client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "").strip()
    access_token = os.environ.get("CTRADER_ACCESS_TOKEN", "").strip()
    account_id_str = os.environ.get("CTRADER_ACCOUNT_ID", "").strip()
    if not all([client_id, client_secret, access_token, account_id_str]):
        raise RuntimeError(
            "Secrets manquants : CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN et CTRADER_ACCOUNT_ID requis."
        )
    account_id = int(account_id_str)

    state_path = Path(args.state_path)
    state = load_state(state_path, args.initial_capital)
    today_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    # En mode daily, on pré-calcule H7.3 avant d'ouvrir la socket cTrader
    h73_data = compute_h73_targets(risk_mult=args.risk_mult) if args.mode == "daily" else None

    host = EndPoints.PROTOBUF_LIVE_HOST if args.env == "live" else EndPoints.PROTOBUF_DEMO_HOST
    port = EndPoints.PROTOBUF_PORT
    client = Client(host, port, TcpProtocol)

    ctx = {
        "balance": None,
        "money_digits": 2,
        "positions": [],
        "unrealized_pnl": 0.0,
        "sym_map": {},       # h_sym -> (symbolId, brokerName)
        "sym_details": {},   # symbolId -> ProtoOASymbol
        "pending_orders": 0,
        "done": False,
        "exit_code": 0,
    }

    def finish(code: int = 0):
        if ctx["done"]:
            return
        ctx["done"] = True
        ctx["exit_code"] = code
        try:
            client.stopService()
        except Exception:
            pass
        if reactor.running:
            reactor.callLater(0.5, reactor.stop)

    # Garde-fou timeout global 45s (pour ne jamais bloquer un job GitHub Actions)
    reactor.callLater(45.0, lambda: (print("[Timeout] Fin forcée après 45s."), finish(1)))

    def on_error(failure):
        print(f"[cTrader Error] {failure}")
        finish(1)

    def send_req(req):
        d = client.send(req)
        d.addErrback(on_error)

    def connected(_c):
        print(f"[cTrader] Connecté ({args.env} -> {host}:{port}). Authentification app...")
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        send_req(req)

    def execute_guard_logic():
        bal = ctx["balance"]
        eq = bal + ctx["unrealized_pnl"]
        state_changed = False

        if state.get("day_utc") != today_utc:
            state["day_utc"] = today_utc
            state["day_start_balance"] = bal
            state_changed = True

        day_start = float(state.get("day_start_balance", bal))
        init_cap = float(state.get("initial_capital", args.initial_capital))
        daily_pnl_pct = (eq - day_start) / max(1.0, day_start) * 100.0
        total_dd_pct = (eq - init_cap) / max(1.0, init_cap) * 100.0

        print(
            f"[Guard {today_utc}] Balance={bal:,.2f}$ | Equity={eq:,.2f}$ | "
            f"Jour={daily_pnl_pct:+.2f}% (seuil -{args.daily_stop_pct:.1f}%) | "
            f"Total={total_dd_pct:+.2f}% (seuil -{args.max_dd_stop_pct:.1f}%) | Pos={len(ctx['positions'])}"
        )

        breach_daily = daily_pnl_pct <= -args.daily_stop_pct
        breach_total = total_dd_pct <= -args.max_dd_stop_pct

        if (breach_daily or breach_total) and len(ctx["positions"]) > 0:
            reason = "PERTE JOURNALIÈRE" if breach_daily else "DRAWDOWN TOTAL"
            msg = (
                f"🚨 <b>GARDE-FOU H7.3 DÉCLENCHÉ ({reason})</b>\n"
                f"Env : <code>{args.env}</code> | Compte : <code>{account_id}</code>\n"
                f"• Équité : <b>{eq:,.2f} $</b> (Jour : <b>{daily_pnl_pct:+.2f}%</b> | Total : <b>{total_dd_pct:+.2f}%</b>)\n"
                f"• Action : {'[DRY-RUN] ' if args.dry_run else ''}Fermeture immédiate de <b>{len(ctx['positions'])}</b> position(s)."
            )
            print(msg)
            send_telegram(msg)

            state["halted_utc_date"] = today_utc
            state["weights"] = {s: 0.0 for s in H73_SYMS}
            if not args.dry_run:
                save_state(state_path, state)
                ctx["pending_orders"] = len(ctx["positions"])
                for pos in ctx["positions"]:
                    creq = ProtoOAClosePositionReq()
                    creq.ctidTraderAccountId = account_id
                    creq.positionId = int(pos.positionId)
                    creq.volume = int(pos.tradeData.volume)
                    send_req(creq)
                reactor.callLater(5.0, lambda: finish(0))
                return
        elif state_changed and not args.dry_run:
            save_state(state_path, state)

        finish(0)

    def execute_daily_logic():
        bal = ctx["balance"]
        eq = bal + ctx["unrealized_pnl"]
        if state.get("day_utc") != today_utc:
            state["day_utc"] = today_utc
            state["day_start_balance"] = bal

        if state.get("halted_utc_date") == today_utc:
            msg = f"🛑 <b>H7.3 Daily ignoré ({today_utc})</b> : le garde-fou journalier a déjà coupé les positions aujourd'hui. Reprise demain UTC."
            print(msg)
            send_telegram(msg)
            finish(0)
            return

        capital_ref = float(state.get("initial_capital", args.initial_capital))
        deadband = 0.05 * args.risk_mult
        prev_w = state.get("weights", {s: 0.0 for s in H73_SYMS})
        target_w = h73_data["weights"]
        sigs = h73_data["signals"]
        prices = h73_data["prices"]

        # Poids retenus après zone morte
        new_w = {}
        for s in H73_SYMS:
            pw = float(prev_w.get(s, 0.0))
            tw = float(target_w[s])
            new_w[s] = tw if abs(tw - pw) >= deadband else pw

        # Stop-loss natif par position : d_SL = min(8 %, 3.8 % / max(0.5, total_notional))
        tot_not_mult = sum(new_w.values())
        sl_pct = min(0.08, (args.daily_stop_pct / 100.0) / max(0.5, tot_not_mult))

        # Positions ouvertes par symbolId
        pos_by_symid: dict[int, list] = {}
        for p in ctx["positions"]:
            sid = int(p.tradeData.symbolId)
            pos_by_symid.setdefault(sid, []).append(p)

        lines = [
            f"🤖 <b>H7.3 cTrader ({args.env.upper()}{' — DRY-RUN' if args.dry_run else ''})</b> — <code>{h73_data['date']}</code>",
            f"Solde : <b>{bal:,.2f} $</b> | Équité : <b>{eq:,.2f} $</b> | Risque : <b>{args.risk_mult:.2f}x</b>",
            f"Piliers actifs : <b>{h73_data['pillars_sum']:.2f} / 3</b> "
            + ("🟢 FEU VERT" if h73_data["pillars_sum"] >= 2.0 else "🔴 PRUDENCE"),
            "",
        ]

        orders_sent = 0
        for s in H73_SYMS:
            if s not in ctx["sym_map"]:
                lines.append(f"• <b>{s}</b> : ⚠️ symbole introuvable chez le broker")
                continue

            sid, bname = ctx["sym_map"][s]
            sdet = ctx["sym_details"].get(sid)
            px = prices[s]
            target_usd = new_w[s] * capital_ref

            # Conversion USD -> volume cTrader (en centièmes d'unité : 1 lot = lotSize en centièmes)
            lot_size = int(getattr(sdet, "lotSize", 100)) if sdet else 100
            min_vol = int(getattr(sdet, "minVolume", 100)) if sdet else 100
            step_vol = int(getattr(sdet, "stepVolume", 100)) if sdet else 100
            max_vol = int(getattr(sdet, "maxVolume", 10_000_000)) if sdet else 10_000_000

            # En cTrader, 1 unité de volume API = 0.01 unité de l'actif de base (ex: 100 = 1.00 contrat/oz/BTC)
            # Valeur notionnelle d'1 unité API = 0.01 * px (en USD pour actifs cotés en USD)
            raw_vol = (target_usd / max(1e-6, px)) * 100.0
            target_vol = round_volume(raw_vol, min_vol, step_vol, max_vol)

            open_list = pos_by_symid.get(sid, [])
            cur_vol = sum(int(p.tradeData.volume) for p in open_list if p.tradeData.tradeSide == ProtoOATradeSide.BUY)
            delta_vol = target_vol - cur_vol

            action_lbl = "⏸ GARDER"
            if abs(delta_vol) >= min_vol:
                if delta_vol > 0:
                    action_lbl = f"🟢 BUY +{delta_vol/100:.2f}u (SL -{sl_pct*100:.1f}%)"
                    if not args.dry_run:
                        oreq = ProtoOANewOrderReq()
                        oreq.ctidTraderAccountId = account_id
                        oreq.symbolId = sid
                        oreq.orderType = ProtoOAOrderType.MARKET
                        oreq.tradeSide = ProtoOATradeSide.BUY
                        oreq.volume = int(delta_vol)
                        # relativeStopLoss en 1/100 000e d'unité de prix
                        oreq.relativeStopLoss = max(100, int(round(px * sl_pct * 100000)))
                        oreq.label = "H73"
                        send_req(oreq)
                        orders_sent += 1
                else:
                    to_close = abs(delta_vol)
                    action_lbl = f"🔴 CLOSE -{to_close/100:.2f}u"
                    if not args.dry_run:
                        rem = to_close
                        for p in open_list:
                            if rem <= 0:
                                break
                            pvol = int(p.tradeData.volume)
                            cvol = min(pvol, rem)
                            creq = ProtoOAClosePositionReq()
                            creq.ctidTraderAccountId = account_id
                            creq.positionId = int(p.positionId)
                            creq.volume = int(cvol)
                            send_req(creq)
                            orders_sent += 1
                            rem -= cvol
            elif target_vol == 0 and cur_vol == 0:
                action_lbl = "CASH"

            lines.append(f"• <b>{s}</b> (<code>{bname}</code>) [{sigs[s]*3:.0f}/3] : <b>{target_usd:,.0f}$</b> — {action_lbl}")

        lines.append(f"\n💼 Exposition cible : <b>{tot_not_mult*capital_ref:,.0f} $</b> (<code>{tot_not_mult:.2f}x</code>) | SL serveur : <code>-{sl_pct*100:.1f}%</code>")
        msg = "\n".join(lines)
        print(msg)
        send_telegram(msg)

        if not args.dry_run:
            state["weights"] = {s: round(new_w[s], 6) for s in H73_SYMS}
            state["last_daily_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save_state(state_path, state)

        reactor.callLater(4.0 if orders_sent > 0 else 0.5, lambda: finish(0))

    def on_message(_c, message):
        parsed = Protobuf.extract(message)
        ptype = message.payloadType

        if ptype == ProtoOAPayloadType.PROTO_OA_APPLICATION_AUTH_RES:
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = account_id
            req.accessToken = access_token
            send_req(req)

        elif ptype == ProtoOAPayloadType.PROTO_OA_ACCOUNT_AUTH_RES:
            treq = ProtoOATraderReq()
            treq.ctidTraderAccountId = account_id
            send_req(treq)

        elif ptype == ProtoOAPayloadType.PROTO_OA_TRADER_RES:
            mdig = int(getattr(parsed.trader, "moneyDigits", 2))
            ctx["money_digits"] = mdig
            ctx["balance"] = float(parsed.trader.balance) / (10 ** mdig)
            rreq = ProtoOAReconcileReq()
            rreq.ctidTraderAccountId = account_id
            send_req(rreq)

        elif ptype == ProtoOAPayloadType.PROTO_OA_RECONCILE_RES:
            ctx["positions"] = list(parsed.position)
            preq = ProtoOAGetPositionUnrealizedPnLReq()
            preq.ctidTraderAccountId = account_id
            send_req(preq)

        elif ptype == ProtoOAPayloadType.PROTO_OA_GET_POSITION_UNREALIZED_PNL_RES:
            mdig = int(getattr(parsed, "moneyDigits", ctx["money_digits"]))
            ctx["unrealized_pnl"] = sum(
                float(x.netUnrealizedPnL) / (10 ** mdig) for x in parsed.positionUnrealizedPnL
            )
            if args.mode == "guard":
                execute_guard_logic()
            else:
                sreq = ProtoOASymbolsListReq()
                sreq.ctidTraderAccountId = account_id
                send_req(sreq)

        elif ptype == ProtoOAPayloadType.PROTO_OA_SYMBOLS_LIST_RES:
            ctx["sym_map"] = match_broker_symbols(list(parsed.symbol))
            if not ctx["sym_map"]:
                print("[cTrader] Aucun symbole H7.3 trouvé chez le broker !")
                finish(1)
                return
            dreq = ProtoOASymbolByIdReq()
            dreq.ctidTraderAccountId = account_id
            for sid, _bn in ctx["sym_map"].values():
                dreq.symbolId.append(sid)
            send_req(dreq)

        elif ptype == ProtoOAPayloadType.PROTO_OA_SYMBOL_BY_ID_RES:
            for sdet in parsed.symbol:
                ctx["sym_details"][int(sdet.symbolId)] = sdet
            execute_daily_logic()

        elif ptype == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
            print(f"[cTrader API Error] {parsed.errorCode}: {parsed.description}")
            finish(1)

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(lambda _c, r: print(f"[cTrader] Déconnecté : {r}"))
    client.setMessageReceivedCallback(on_message)
    client.startService()
    reactor.run()
    return ctx["exit_code"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bot Live H7.3 pour cTrader Open API (Démo gratuit & The5ers)")
    ap.add_argument("--mode", choices=["daily", "guard"], default="daily", help="daily = rééquilibrage 1x/j | guard = surveillance équité 5m")
    ap.add_argument("--env", choices=["demo", "live"], default="demo", help="demo = compte démo | live = serveur live")
    ap.add_argument("--initial_capital", type=float, default=10000.0, help="Capital initial du challenge (USD)")
    ap.add_argument("--risk_mult", type=float, default=1.0, help="1.0 = standard | 1.25 = rapide The5ers")
    ap.add_argument("--daily_stop_pct", type=float, default=3.8, help="Stop journalier sécurité en % (défaut: 3.8)")
    ap.add_argument("--max_dd_stop_pct", type=float, default=8.5, help="Stop DD total sécurité en % (défaut: 8.5)")
    ap.add_argument("--state_path", type=str, default="research_v5/state_h73_ctrader.json")
    ap.add_argument("--dry_run", action="store_true", help="Simulation sans envoyer d'ordres réels sur cTrader")
    ap.add_argument("--offline_test", action="store_true", help="Teste uniquement le calcul H7.3 sans connexion cTrader")
    args = ap.parse_args()

    if args.offline_test:
        d = compute_h73_targets(args.risk_mult)
        print(json.dumps(d, indent=2))
        return

    sys.exit(run_ctrader_session(args))


if __name__ == "__main__":
    main()

