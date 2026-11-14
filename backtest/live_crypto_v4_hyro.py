"""
backtest/live_crypto_v4_hyro.py — Exécuteur Live autonome CRYPTO V4 pour **HyroTrader One-Step (Bybit API v5 / Mode Démo / Dry-Run)**.

Intègre nativement :
  1. La stratégie **CRYPTO V4 Optimisée Walk-Forward** sur `BTCUSDT`, `ETHUSDT`, `AVAXUSDT`, `INJUSDT` :
     - Biais H4 (`h4_lookback=8`, `h4_thresh=0.35`)
     - Fenêtre M15 anti-krach (`m15_min_range_pct=0.35%`, `m15_max_range_pct=1.40%` sur 16 bougies 15m)
     - Cooldown après SL (`sl_cooldown_hours=3.0h`) + Anti-répétition du même POI cassé (`same_poi_tol_pct=0.22%`)
     - Levier x30 isolé, SL initial `-50%` ROI (`-1R`), TP `+300%` ROI (`+6R`)
     - Trailing SL 5 paliers `tight_high` :
       * `+50% (+1R)`  -> `0% (BE)`
       * `+100% (+2R)` -> `+50% (+1.0R)`
       * `+150% (+3R)` -> `+105% (+2.1R)`
       * `+200% (+4R)` -> `+160% (+3.2R)`
       * `+250% (+5R)` -> `+215% (+4.3R)`
  2. Les **5 Règles HyroTrader One-Step + Hard Daily Profit Lock (`+3.85%`)** :
     - **Règle 1 (`hyro_risk_pct = 0.68%`)** : `1R = 68 USDT` sur un challenge `10 000 USDT` (TP `+6R` net = `+3.98% <= +4.00%`).
     - **Règle 2 (`hyro_daily_stop_pct = 2.20%` + Garde-fou simultané)** : Disjoncteur journalier qui coupe les positions et bloque toute nouvelle entrée dès `-2.20%` sur la journée UTC (marge de sécurité de `1.80%` avant la limite éliminatoire de `-4.0%`).
     - **Règle 3 (`hyro_pause_at_cap_pct = 3.60%` + `hyro_hard_day_tp_pct = 3.85%` + Écrêtage 40%)** :
       * Dès que le gain du jour UTC (réalisé + latent) atteint `+3.85%`, clôture immédiatement toutes les positions ouvertes au marché, annule les ordres limites et verrouille la journée jusqu'à `00:00 UTC` (respect absolu de la règle de consistance des 40% = `+4.0%/jour` max).
       * Dès `+3.60%` réalisé, n'ouvre plus de nouveau trade sur la journée UTC.
     - **Règle 4 (`hyro_dd_throttle_pct = 2.80%` -> `x0.55`)** : Si le drawdown depuis le capital initial ou le HWM atteint `-2.80%`, divise le risque par ~2 (`x0.55`).
     - **Règle 5 (`hyro_lock_at_pct = 6.50%` -> `x0.55` + `max 4 positions` + `notionnel <= 2.0x`)** : Réduit le risque à l'approche des `+10%`, plafonne l'exposition notionnelle totale à `2.0x` le solde, et verrouille définitivement le compte dès que `+10.05%` (et `>= 3` jours tradés) sont atteints.

Environnements supportés (`--env`) :
  - `demo`    : Bybit Demo Trading (`https://api-demo.bybit.com`) — utilisé par les challenges HyroTrader Bybit !
  - `mainnet` : Bybit Mainnet (`https://api.bybit.com`) — sous-compte financé Bybit
  - `testnet` : Bybit Testnet (`https://api-testnet.bybit.com`)
  - `dry_run` : Simulation papier complète sans clé API (utilise les vraies bougies Binance/Bybit en direct et sauvegarde l'état dans `data/live_hyro_state.json`).
"""
import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_crypto as dc  # noqa: E402
import strategy_crypto_v4 as scv4  # noqa: E402

STATE_PATH_DEFAULT = os.path.join("data", "live_hyro_state.json")

BYBIT_ENDPOINTS = {
    "demo": "https://api-demo.bybit.com",
    "mainnet": "https://api.bybit.com",
    "testnet": "https://api-testnet.bybit.com",
}


def tg_send(text: str):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat or not text:
        return
    try:
        for i in range(0, len(text), 4000):
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text[i:i + 4000]},
                timeout=20,
            )
    except requests.RequestException as e:
        print(f"[Telegram] Erreur d'envoi : {e}", flush=True)


def to_bybit_symbol(sym: str) -> str:
    s = sym.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"


def round_step(val: float, step: float, mode: str = "floor") -> float:
    if step <= 0:
        return float(val)
    ratio = val / step
    if mode == "floor":
        n = math.floor(ratio + 1e-9)
    elif mode == "ceil":
        n = math.ceil(ratio - 1e-9)
    else:
        n = round(ratio)
    decimals = max(0, int(round(-math.log10(step)))) if step < 1.0 else 0
    return round(n * step, decimals)


class BybitV5Client:
    """Client léger et robuste pour Bybit API v5 (Linear USDT Perpetual — Demo, Mainnet et Testnet)."""

    def __init__(self, api_key: str, api_secret: str, env: str = "demo", timeout: int = 20):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        self.base_url = BYBIT_ENDPOINTS.get(env, BYBIT_ENDPOINTS["demo"])
        self.recv_window = "10000"
        self.timeout = timeout
        self.session = requests.Session()
        self.instruments_cache = {}

    def _sign(self, ts_str: str, payload_str: str) -> str:
        raw = f"{ts_str}{self.api_key}{self.recv_window}{payload_str}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def request(self, method: str, path: str, params: dict = None, auth: bool = True) -> dict:
        params = params or {}
        url = f"{self.base_url}{path}"
        ts_str = str(int(time.time() * 1000))

        if method.upper() == "GET":
            query_str = urlencode(sorted(params.items())) if params else ""
            if query_str:
                url = f"{url}?{query_str}"
            payload_str = query_str
            body = None
        else:
            payload_str = json.dumps(params, separators=(",", ":"))
            body = payload_str.encode("utf-8")

        headers = {"Content-Type": "application/json", "User-Agent": "Antigravity-HyroOneStep/1.0"}
        if auth:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("Clés BYBIT_API_KEY / BYBIT_API_SECRET manquantes pour l'appel authentifié.")
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": ts_str,
                "X-BAPI-RECV-WINDOW": self.recv_window,
                "X-BAPI-SIGN": self._sign(ts_str, payload_str),
            })

        resp = self.session.request(method.upper(), url, headers=headers, data=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Erreur HTTP {resp.status_code} Bybit ({path}) : {resp.text[:300]}")
        data = resp.json()
        ret_code = data.get("retCode", 0)
        # 110043 = leverage not modified (déjà à x30) -> non bloquant
        if ret_code not in (0, 110043):
            raise RuntimeError(f"Erreur API Bybit ({path}, code={ret_code}) : {data.get('retMsg')}")
        return data.get("result", {})

    def get_instrument_spec(self, symbol: str) -> dict:
        if symbol not in self.instruments_cache:
            res = self.request("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": symbol}, auth=False)
            lst = res.get("list", [])
            if not lst:
                self.instruments_cache[symbol] = {"qty_step": 0.001, "min_qty": 0.001, "tick_size": 0.01}
            else:
                item = lst[0]
                lot = item.get("lotSizeFilter", {})
                pf = item.get("priceFilter", {})
                self.instruments_cache[symbol] = {
                    "qty_step": float(lot.get("qtyStep", 0.001)),
                    "min_qty": float(lot.get("minOrderQty", 0.001)),
                    "tick_size": float(pf.get("tickSize", 0.01)),
                }
        return self.instruments_cache[symbol]

    def get_account_overview(self) -> dict:
        res = self.request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        lst = res.get("list", [])
        if not lst:
            raise RuntimeError("Aucun compte UNIFIED retourné par Bybit.")
        acc = lst[0]
        equity = float(acc.get("totalEquity") or acc.get("totalWalletBalance") or 0.0)
        wallet = float(acc.get("totalWalletBalance") or equity)
        avail = float(acc.get("totalAvailableBalance") or wallet)
        return {"equity": equity, "wallet_balance": wallet, "available_balance": avail}

    def get_positions(self) -> list:
        res = self.request("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        out = []
        for p in res.get("list", []):
            sz = float(p.get("size") or 0.0)
            if sz <= 0:
                continue
            out.append({
                "symbol": p["symbol"],
                "side": "long" if p.get("side") == "Buy" else "short",
                "side_int": 1 if p.get("side") == "Buy" else -1,
                "size": sz,
                "entry_price": float(p.get("avgPrice") or 0.0),
                "mark_price": float(p.get("markPrice") or p.get("avgPrice") or 0.0),
                "unrealized_pnl": float(p.get("unrealisedPnl") or 0.0),
                "sl_price": float(p.get("stopLoss") or 0.0) if p.get("stopLoss") else 0.0,
                "tp_price": float(p.get("takeProfit") or 0.0) if p.get("takeProfit") else 0.0,
                "leverage": float(p.get("leverage") or 30.0),
            })
        return out

    def get_open_orders(self) -> list:
        res = self.request("GET", "/v5/order/realtime", {"category": "linear", "settleCoin": "USDT"})
        out = []
        for o in res.get("list", []):
            if o.get("StopOrderType") in ("TakeProfit", "StopLoss", "TrailingStop"):
                continue
            out.append({
                "order_id": o["orderId"],
                "symbol": o["symbol"],
                "side": "long" if o.get("side") == "Buy" else "short",
                "side_int": 1 if o.get("side") == "Buy" else -1,
                "price": float(o.get("price") or 0.0),
                "qty": float(o.get("qty") or 0.0),
            })
        return out

    def ensure_leverage(self, symbol: str, leverage: int = 30):
        try:
            self.request("POST", "/v5/position/set-leverage", {
                "category": "linear",
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            })
        except Exception:
            pass

    def set_tp_sl(self, symbol: str, sl_price: float = None, tp_price: float = None):
        spec = self.get_instrument_spec(symbol)
        tick = spec["tick_size"]
        payload = {"category": "linear", "symbol": symbol, "tpslMode": "Full", "positionIdx": 0}
        if sl_price is not None and sl_price > 0:
            payload["stopLoss"] = str(round_step(sl_price, tick, "nearest"))
        if tp_price is not None and tp_price > 0:
            payload["takeProfit"] = str(round_step(tp_price, tick, "nearest"))
        return self.request("POST", "/v5/position/trading-stop", payload)

    def place_order(
        self,
        symbol: str,
        side_int: int,
        order_type: str,
        qty: float,
        price: float = None,
        sl_price: float = None,
        tp_price: float = None,
        reduce_only: bool = False,
    ) -> dict:
        spec = self.get_instrument_spec(symbol)
        q_step = spec["qty_step"]
        tick = spec["tick_size"]
        qty_r = round_step(qty, q_step, "floor")
        if qty_r < spec["min_qty"]:
            raise ValueError(f"Quantité {qty_r} < min_qty {spec['min_qty']} sur {symbol}")

        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side_int == 1 else "Sell",
            "orderType": "Market" if order_type.upper() == "MARKET" else "Limit",
            "qty": str(qty_r),
            "positionIdx": 0,
        }
        if order_type.upper() == "LIMIT":
            if price is None or price <= 0:
                raise ValueError("Prix requis pour un ordre LIMIT")
            payload["price"] = str(round_step(price, tick, "nearest"))
            payload["timeInForce"] = "GTC"
        if reduce_only:
            payload["reduceOnly"] = True
        else:
            if sl_price is not None and sl_price > 0:
                payload["stopLoss"] = str(round_step(sl_price, tick, "nearest"))
            if tp_price is not None and tp_price > 0:
                payload["takeProfit"] = str(round_step(tp_price, tick, "nearest"))
        return self.request("POST", "/v5/order/create", payload)

    def cancel_order(self, symbol: str, order_id: str):
        return self.request("POST", "/v5/order/cancel", {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        })

    def close_position_market(self, pos: dict):
        opp_side = -1 if pos["side_int"] == 1 else 1
        return self.place_order(pos["symbol"], opp_side, "MARKET", pos["size"], reduce_only=True)


def load_state(path: str, initial_capital: float = 10000.0) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
                if isinstance(st, dict):
                    st.setdefault("initial_balance", initial_capital)
                    st.setdefault("hwm_balance", st.get("initial_balance", initial_capital))
                    st.setdefault("cur_utc_day", "")
                    st.setdefault("day_start_balance", st.get("initial_balance", initial_capital))
                    st.setdefault("day_locked_reason", "")
                    st.setdefault("day_pnl_history", {})
                    st.setdefault("days_traded", [])
                    st.setdefault("challenge_passed", False)
                    st.setdefault("symbols", {})
                    st.setdefault("dry_run_balance", initial_capital)
                    return st
        except Exception:
            pass
    return {
        "initial_balance": float(initial_capital),
        "hwm_balance": float(initial_capital),
        "cur_utc_day": "",
        "day_start_balance": float(initial_capital),
        "day_locked_reason": "",
        "day_pnl_history": {},
        "days_traded": [],
        "challenge_passed": False,
        "symbols": {},
        "dry_run_balance": float(initial_capital),
    }


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def compute_hyro_live_risk(
    state: dict,
    wallet_bal: float,
    equity: float,
    open_positions: list,
    p: dict,
) -> dict:
    """
    Calcule le risque autorisé pour un nouveau trade en appliquant strictement
    les 5 Règles HyroTrader One-Step + le partage du quota de consistance journalier (40% = +4.0%/jour).
    """
    init_bal = float(state.get("initial_balance", 10000.0))
    hwm_bal = max(float(state.get("hwm_balance", init_bal)), wallet_bal, equity)
    day_start = float(state.get("day_start_balance", wallet_bal))
    utc_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    day_realized_pct = ((wallet_bal - day_start) / init_bal) * 100.0
    day_total_pct = ((equity - day_start) / init_bal) * 100.0

    # Calcul du profit validé après règle de consistance des 40% (+4.0% max par jour UTC)
    day_hist = dict(state.get("day_pnl_history", {}))
    day_hist[utc_day] = day_total_pct
    max_day_cap_pct = 4.0
    excess_pct = sum(max(0.0, v - max_day_cap_pct) for v in day_hist.values())
    total_profit_pct = ((equity - init_bal) / init_bal) * 100.0
    credited_profit_pct = total_profit_pct - excess_pct

    # Règle 1 : Risque de base par trade (0.68% = 68$ sur 10k$)
    rk_base_pct = float(p.get("hyro_risk_pct", 0.68))
    risk_pct = rk_base_pct
    notes = [f"Base {rk_base_pct:.2f}%"]

    # Règle 2a : Disjoncteur journalier (-2.20%)
    daily_stop_pct = float(p.get("hyro_daily_stop_pct", 2.20))
    if day_total_pct <= -daily_stop_pct or day_realized_pct <= -daily_stop_pct:
        return {
            "allowed": False,
            "reason": f"Règle 2 (Disjoncteur journalier atteint : {day_total_pct:+.2f}% <= -{daily_stop_pct:.2f}%)",
            "risk_pct": 0.0,
            "day_total_pct": day_total_pct,
            "credited_profit_pct": credited_profit_pct,
        }

    # Règle 3a : Pause journalière de consistance (+3.60% réalisé ou total) — bloque toute NOUVELLE entrée mais laisse courir les gagnants déjà ouverts (bouclier DD gratuit !)
    pause_cap_pct = float(p.get("hyro_pause_at_cap_pct", 3.60))
    hard_tp_pct = float(p.get("hyro_hard_day_tp_pct", 0.0))
    if (hard_tp_pct > 0 and day_total_pct >= hard_tp_pct) or day_total_pct >= pause_cap_pct or day_realized_pct >= pause_cap_pct:
        return {
            "allowed": False,
            "reason": f"Règle 3 (Pause Consistance 40% active : jour {day_total_pct:+.2f}% >= +{pause_cap_pct:.2f}%)",
            "risk_pct": 0.0,
            "day_total_pct": day_total_pct,
            "credited_profit_pct": credited_profit_pct,
        }

    # Règle 5b : Plafond de positions simultanées (max 4)
    max_open = int(p.get("hyro_max_open_trades", 4))
    if len(open_positions) >= max_open:
        return {
            "allowed": False,
            "reason": f"Règle 5 (Max {max_open} positions simultanées déjà ouvertes)",
            "risk_pct": 0.0,
            "day_total_pct": day_total_pct,
            "credited_profit_pct": credited_profit_pct,
        }

    # Règle 3c : Écrêtage dynamique pour la règle des 40% (+4.0%/jour)
    if int(p.get("hyro_cap_scale", 1)) != 0:
        rem_day_cap = max(0.0, max_day_cap_pct - max(0.0, day_realized_pct))
        if risk_pct * 5.5 > rem_day_cap:
            risk_pct = max(0.45 * rk_base_pct, rem_day_cap / 5.5)
            notes.append(f"Cap40%->{risk_pct:.2f}%")

    # Règle 4 : Réduction du risque en Drawdown (dès -2.80% sous capital initial ou HWM -> x0.55)
    dd_static_pct = max(0.0, ((init_bal - equity) / init_bal) * 100.0)
    dd_hwm_pct = max(0.0, ((hwm_bal - equity) / init_bal) * 100.0)
    dd_thr_pct = float(p.get("hyro_dd_throttle_pct", 2.80))
    dd_mult = float(p.get("hyro_dd_mult", 0.55))
    if max(dd_static_pct, dd_hwm_pct) >= dd_thr_pct:
        risk_pct *= dd_mult
        notes.append(f"DD-Throttle(-{max(dd_static_pct, dd_hwm_pct):.2f}%)->x{dd_mult}")

    # Règle 5a : Verrouillage de fin de challenge (dès +6.50% validés -> x0.55)
    lock_at_pct = float(p.get("hyro_lock_at_pct", 6.50))
    lock_mult = float(p.get("hyro_lock_mult", 0.55))
    if credited_profit_pct >= lock_at_pct:
        risk_pct *= lock_mult
        notes.append(f"Lock(+{credited_profit_pct:.2f}%)->x{lock_mult}")

    # Règle 5c : Plafond de notionnel simultané <= 2.0x le solde
    max_notional_mult = float(p.get("hyro_max_notional_mult", 2.0))
    open_notional_usdt = sum(pos["size"] * pos["mark_price"] for pos in open_positions)
    rem_notional_usdt = max(0.0, max_notional_mult * wallet_bal - open_notional_usdt)
    sl_dist_frac = (float(p.get("sl_roi_pct", 50.0)) / 100.0) / float(p.get("leverage", 30.0))
    max_risk_from_notional_pct = ((rem_notional_usdt * sl_dist_frac) / init_bal) * 100.0
    if risk_pct > max_risk_from_notional_pct:
        risk_pct = max_risk_from_notional_pct
        notes.append(f"CapNotionnel2x->{risk_pct:.2f}%")

    # Règle 2b : Garde-fou journalier simultané (protège la limite journalière de -4.0%)
    if int(p.get("hyro_daily_guard", 1)) != 0 and open_positions:
        open_risk_pct = len(open_positions) * (rk_base_pct * 0.65)
        daily_buf_pct = (4.0 + day_realized_pct) - open_risk_pct - 0.25
        if daily_buf_pct < risk_pct:
            risk_pct = max(0.0, daily_buf_pct)
            notes.append(f"GardeFouJour->{risk_pct:.2f}%")

    if risk_pct < 0.20 * rk_base_pct:
        return {
            "allowed": False,
            "reason": f"Risque résiduel trop faible ({risk_pct:.2f}%) après plafonds Hyro ({', '.join(notes)})",
            "risk_pct": 0.0,
            "day_total_pct": day_total_pct,
            "credited_profit_pct": credited_profit_pct,
        }

    one_r_usdt = init_bal * (risk_pct / 100.0)
    margin_usdt = one_r_usdt / (float(p.get("sl_roi_pct", 50.0)) / 100.0)
    notional_usdt = margin_usdt * float(p.get("leverage", 30.0))

    return {
        "allowed": True,
        "reason": " | ".join(notes),
        "risk_pct": risk_pct,
        "one_r_usdt": one_r_usdt,
        "margin_usdt": margin_usdt,
        "notional_usdt": notional_usdt,
        "day_total_pct": day_total_pct,
        "credited_profit_pct": credited_profit_pct,
    }


def main():
    ap = argparse.ArgumentParser(description="Exécuteur Live CRYPTO V4 pour HyroTrader One-Step (Bybit API v5 / Démo / Dry-Run)")
    ap.add_argument("--env", default="demo", choices=["demo", "mainnet", "testnet", "dry_run"], help="demo = Bybit Demo Trading (Challenge HyroTrader)")
    ap.add_argument("--symbols", default="BTC,ETH,AVAX,INJ")
    ap.add_argument("--initial_capital", type=float, default=10000.0, help="Capital initial du challenge Hyro One-Step (défaut: 10 000 USDT)")
    ap.add_argument("--state_path", default=STATE_PATH_DEFAULT)
    ap.add_argument("--params", default="")
    args = ap.parse_args()

    p = scv4.parse_params(args.params)
    lev = int(round(float(p.get("leverage", 30.0))))
    syms = [to_bybit_symbol(s) for s in args.symbols.split(",") if s.strip()]

    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    is_dry = (args.env == "dry_run") or (not api_key or not api_secret)
    if is_dry and args.env != "dry_run":
        print("⚠️ BYBIT_API_KEY / BYBIT_API_SECRET non définis -> passage automatique en mode simulation DRY_RUN.", flush=True)

    client = None if is_dry else BybitV5Client(api_key, api_secret, env=args.env)
    state = load_state(args.state_path, args.initial_capital)

    utc_now = datetime.now(timezone.utc)
    utc_day = utc_now.strftime("%Y-%m-%d")

    # 1. Lecture du solde, des positions ouvertes et des ordres au repos
    if is_dry:
        wallet_bal = float(state.get("dry_run_balance", args.initial_capital))
        equity = wallet_bal
        open_positions = []
        open_orders = []
    else:
        acc = client.get_account_overview()
        wallet_bal = acc["wallet_balance"]
        equity = acc["equity"]
        open_positions = client.get_positions()
        open_orders = client.get_open_orders()

    if state["cur_utc_day"] != utc_day:
        state["cur_utc_day"] = utc_day
        state["day_start_balance"] = wallet_bal
        state["day_locked_reason"] = ""
        print(f"📅 Nouveau jour UTC ({utc_day}) : solde de référence journalier fixé à {wallet_bal:,.2f} USDT", flush=True)

    if max(wallet_bal, equity) > state.get("hwm_balance", args.initial_capital):
        state["hwm_balance"] = max(wallet_bal, equity)

    init_bal = float(state.get("initial_balance", args.initial_capital))
    day_start = float(state.get("day_start_balance", wallet_bal))
    day_total_pct = ((equity - day_start) / init_bal) * 100.0
    state["day_pnl_history"][utc_day] = day_total_pct

    if open_positions and utc_day not in state["days_traded"]:
        state["days_traded"].append(utc_day)

    excess_pct = sum(max(0.0, v - 4.0) for v in state["day_pnl_history"].values())
    total_profit_pct = ((equity - init_bal) / init_bal) * 100.0
    credited_profit_pct = total_profit_pct - excess_pct
    n_days_traded = len(state["days_traded"])

    print(
        f"⚡ [HYRO ONE-STEP LIVE | {args.env.upper()}] {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"   Solde : {wallet_bal:,.2f} USDT (Équité {equity:,.2f} USDT) | Jour UTC : {day_total_pct:+.2f}% (réf {day_start:,.2f}$)\n"
        f"   Profit Brut : {total_profit_pct:+.2f}% | Profit Retenu (40%) : {credited_profit_pct:+.2f}% / +10.00% | Jours tradés : {n_days_traded}/3",
        flush=True,
    )

    # ------------------------------------------------------------------
    # VERROU 0 : Challenge validé (+10.05% retenus et >= 3 jours tradés)
    # ------------------------------------------------------------------
    if state.get("challenge_passed") or (credited_profit_pct >= 10.05 and n_days_traded >= 3):
        if not state.get("challenge_passed"):
            state["challenge_passed"] = True
            if not is_dry:
                for pos in open_positions:
                    client.close_position_market(pos)
                for ord_d in open_orders:
                    client.cancel_order(ord_d["symbol"], ord_d["order_id"])
            msg = (
                f"🏆 CHALLENGE HYROTRADER ONE-STEP VALIDÉ !\n"
                f"Équité finale : {equity:,.2f} USDT ({total_profit_pct:+.2f}% brut | {credited_profit_pct:+.2f}% retenus après règle 40%)\n"
                f"Jours de trading : {n_days_traded} (min 3). Toutes les positions et ordres ont été fermés."
            )
            print(msg, flush=True)
            tg_send(msg)
        save_state(args.state_path, state)
        return

    # ------------------------------------------------------------------
    # VERROU 1 : Hard Daily Profit Lock (optionnel si hyro_hard_day_tp_pct > 0)
    # Par défaut (0.0), à +3.60% on bloque uniquement les NOUVELLES entrées et on annule les ordres limites non exécutés,
    # tout en laissant les positions gagnantes déjà ouvertes aller à leur TP (le surplus >4% sert de bouclier contre les -8% fixes !)
    # ------------------------------------------------------------------
    hard_day_tp = float(p.get("hyro_hard_day_tp_pct", 0.0))
    pause_cap_pct = float(p.get("hyro_pause_at_cap_pct", 3.60))
    if hard_day_tp > 0.0 and day_total_pct >= hard_day_tp and not state.get("day_locked_reason"):
        state["day_locked_reason"] = f"HARD_DAILY_PROFIT_LOCK (+{day_total_pct:.2f}% >= +{hard_day_tp:.2f}%)"
        if not is_dry:
            for pos in open_positions:
                client.close_position_market(pos)
            for ord_d in open_orders:
                client.cancel_order(ord_d["symbol"], ord_d["order_id"])
        msg = (
            f"🎯 [Hyro Règle 3 — Plafond 40%] Gain journalier à {day_total_pct:+.2f}% (seuil +{hard_day_tp:.2f}%) !\n"
            f"Clôture immédiate des positions ouvertes et pause jusqu'à 00:00 UTC."
        )
        print(msg, flush=True)
        tg_send(msg)
        save_state(args.state_path, state)
        return
    elif day_total_pct >= pause_cap_pct and open_orders and not is_dry:
        for ord_d in open_orders:
            client.cancel_order(ord_d["symbol"], ord_d["order_id"])
        open_orders = []
        print(f"🎯 [Hyro Règle 3] Journée à {day_total_pct:+.2f}% (>= +{pause_cap_pct:.2f}%) : ordres limites au repos annulés (positions ouvertes conservées sous Trailing SL).", flush=True)

    # ------------------------------------------------------------------
    # VERROU 2 : Disjoncteur Journalier (-2.20% sur la journée UTC)
    # ------------------------------------------------------------------
    daily_stop = float(p.get("hyro_daily_stop_pct", 2.20))
    if day_total_pct <= -daily_stop and not state.get("day_locked_reason"):
        state["day_locked_reason"] = f"DAILY_STOP_LOSS ({day_total_pct:+.2f}% <= -{daily_stop:.2f}%)"
        if not is_dry:
            for pos in open_positions:
                client.close_position_market(pos)
            for ord_d in open_orders:
                client.cancel_order(ord_d["symbol"], ord_d["order_id"])
        msg = (
            f"🛑 [Hyro Règle 2 — Disjoncteur Jour] Perte journalière à {day_total_pct:+.2f}% (seuil -{daily_stop:.2f}% pour protéger les -4.0%) !\n"
            f"Positions coupées et pause complète jusqu'à 00:00 UTC."
        )
        print(msg, flush=True)
        tg_send(msg)
        save_state(args.state_path, state)
        return

    if state.get("day_locked_reason"):
        print(f"⏸️ Journée UTC en pause ({state['day_locked_reason']}) -> aucune action jusqu'à 00:00 UTC.", flush=True)
        save_state(args.state_path, state)
        return

    # ------------------------------------------------------------------
    # ÉTAPE 2 : Gestion des positions ouvertes (Trailing SL 5 paliers tight_high)
    # ------------------------------------------------------------------
    pos_by_sym = {pos["symbol"]: pos for pos in open_positions}
    ord_by_sym = {}
    for o in open_orders:
        ord_by_sym.setdefault(o["symbol"], []).append(o)

    for sym, pos in pos_by_sym.items():
        st_sym = state["symbols"].setdefault(sym, {})
        prev_peak = float(st_sym.get("peak_px", pos["entry_price"]))
        trail_info = scv4.compute_live_trailing_sl(
            side=pos["side_int"],
            entry=pos["entry_price"],
            px=pos["mark_price"],
            current_sl=pos["sl_price"],
            p=p,
            peak_px=prev_peak,
        )
        st_sym["peak_px"] = trail_info["peak_px"]
        need_tp = pos["tp_price"] <= 0
        if trail_info["should_update"] or need_tp:
            new_sl = trail_info["target_sl_price"] if trail_info["tier"] > 0 else scv4._price_at_roi(pos["side_int"], pos["entry_price"], -float(p["sl_roi_pct"]), float(p["leverage"]))
            new_tp = trail_info["tp_price"]
            if not is_dry:
                client.set_tp_sl(sym, sl_price=new_sl, tp_price=new_tp)
            msg = (
                f"🔒 [{sym}] Trailing SL mis à jour (Palier {trail_info['tier']} | ROI pic {trail_info['roi_peak']:+.1f}%) : "
                f"SL -> {new_sl:.4f} ({trail_info['target_sl_roi']:+.0f}% ROI) | TP -> {new_tp:.4f}"
            )
            print(msg, flush=True)
            tg_send(msg)

    # ------------------------------------------------------------------
    # ÉTAPE 3 & 4 : Analyse SMC (15m/4h) & Ouverture calibrée Hyro One-Step
    # ------------------------------------------------------------------
    for sym in syms:
        if sym in pos_by_sym:
            continue

        st_sym = state["symbols"].setdefault(sym, {})
        st_sym["peak_px"] = 0.0
        last_sl_ts = float(st_sym.get("last_sl_ts", 0.0))
        hours_since_sl = (time.time() - last_sl_ts) / 3600.0 if last_sl_ts > 0 else 999.0
        last_sl_poi = float(st_sym.get("last_sl_poi", -1.0))

        try:
            df_15m = dc.fetch_klines(sym, "15m", limit=30)
            df_4h = dc.fetch_klines(sym, "4h", limit=20)
            # On utilise uniquement les bougies clôturées (avant la bougie en cours)
            sig = scv4.evaluate_live_signal(
                df_15m=df_15m.iloc[:-1],
                df_4h=df_4h.iloc[:-1],
                px=float(df_15m["close"].iloc[-1]),
                p=p,
                hours_since_last_sl=hours_since_sl,
                last_sl_poi=last_sl_poi,
            )
        except Exception as e:
            print(f"⚠️ [{sym}] Erreur lecture bougies 15m/4h : {e}", flush=True)
            continue

        cur_orders = ord_by_sym.get(sym, [])
        if sig["action"] == "SKIP":
            if cur_orders and not is_dry:
                for o in cur_orders:
                    client.cancel_order(sym, o["order_id"])
                print(f"🔄 [{sym}] Ordre limite annulé ({sig['reason']})", flush=True)
            else:
                print(f"⏸️ [{sym}] SKIP : {sig['reason']}", flush=True)
            continue

        # Calcul du sizing selon les 5 règles HyroTrader One-Step
        sizing = compute_hyro_live_risk(state, wallet_bal, equity, list(pos_by_sym.values()), p)
        if not sizing["allowed"]:
            print(f"🛡️ [{sym}] Entrée bloquée par Hyro Risk Overlay : {sizing['reason']}", flush=True)
            continue

        entry_px = sig["entry_price"]
        qty = sizing["notional_usdt"] / entry_px if entry_px > 0 else 0.0

        if sig["action"] == "LIMIT" and cur_orders:
            o0 = cur_orders[0]
            reprice_tol = float(p.get("reprice_tol_pct", 0.10)) / 100.0
            if o0["side_int"] == sig["side_int"] and abs(entry_px - o0["price"]) / o0["price"] <= reprice_tol:
                print(f"✅ [{sym}] Ordre LIMIT {sig['side'].upper()} déjà en place @ {o0['price']:.4f}", flush=True)
                continue
            if not is_dry:
                for o in cur_orders:
                    client.cancel_order(sym, o["order_id"])

        if is_dry:
            print(
                f"🧪 [DRY_RUN | {sym}] {sig['action']} {sig['side'].upper()} @ {entry_px:.4f} | "
                f"SL {sig['sl_price']:.4f} (-50%) | TP {sig['tp_price']:.4f} (+300%) | "
                f"Risque 1R={sizing['one_r_usdt']:.1f}$ ({sizing['risk_pct']:.2f}% | {sizing['reason']})",
                flush=True,
            )
        else:
            client.ensure_leverage(sym, lev)
            client.place_order(
                symbol=sym,
                side_int=sig["side_int"],
                order_type=sig["action"],
                qty=qty,
                price=entry_px if sig["action"] == "LIMIT" else None,
                sl_price=sig["sl_price"],
                tp_price=sig["tp_price"],
            )
            if utc_day not in state["days_traded"]:
                state["days_traded"].append(utc_day)
            msg = (
                f"🚀 [HYRO LIVE | {sym}] {sig['action']} {sig['side'].upper()} @ {entry_px:.4f}\n"
                f"• SL : {sig['sl_price']:.4f} (-50% ROI) | TP : {sig['tp_price']:.4f} (+300% ROI)\n"
                f"• Risque 1R : {sizing['one_r_usdt']:.1f} USDT ({sizing['risk_pct']:.2f}% | Marge {sizing['margin_usdt']:.1f}$ x{lev})\n"
                f"• Règles Hyro : {sizing['reason']}"
            )
            print(msg, flush=True)
            tg_send(msg)

    save_state(args.state_path, state)


if __name__ == "__main__":
    main()
