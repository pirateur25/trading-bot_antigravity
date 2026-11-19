"""
backtest/live_crypto_v4_hyro.py — Exécuteur Live autonome CRYPTO V4 pour **HyroTrader One-Step**
compatible **Bybit API v5**, **BloFin API v1** (les 2 exchanges officiels HyroTrader) et **Simulation Temps Réel (Dry-Run)**.

Pourquoi ce script résout l'erreur `HTTP 403 CloudFront (block access from your country)` sur GitHub Actions :
  1. Les runners GitHub Actions (`ubuntu-latest`) sont hébergés aux **États-Unis (Azure US)**, et le pare-feu Amazon CloudFront de Bybit bloque les IP américaines par défaut.
  2. Ce script intègre **4 solutions automatiques** :
     - **Option A (`BYBIT_PROXY_URL` ou `BYBIT_BASE_URL`)** : Supporte un proxy européen ou un relais Cloudflare Worker gratuit pour contourner le géoblocage US de GitHub Actions sur Bybit + teste automatiquement les domaines miroirs (`api-demo.bytick.com` / `api.bytick.com`).
     - **Option B (`--exchange blofin` : Clés API BloFin)** : Supporte nativement **BloFin** (`openapi.blofin.com` / `demo-trading-openapi.blofin.com`), le 2e exchange officiel de HyroTrader, qui n'a ni le blocage CloudFront US ni la pénalité de 40% sur les fills démo de Bybit !
     - **Option C (Exécution locale Linux / VPS EU)** : Tourne directement depuis la France sans aucun blocage IP.
     - **Option D (Bascule automatique en Simulation Temps Réel `DRY_RUN`)** : Si GitHub Actions (IP US) reçoit un `HTTP 403 CloudFront` de Bybit sans proxy, le bot **ne plante plus** : il bascule automatiquement sur le **Simulateur Paper-Trading Temps Réel** alimenté par les bougies 5m live d'Hyperliquid/KuCoin (non bloquées aux US) et simule en direct les ordres limites, positions, TP/SL, Trailing 5 paliers et les 5 règles Hyro dans `data/live_hyro_state.json` !
"""
import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_crypto as dc  # noqa: E402
import strategy_crypto_v4 as scv4  # noqa: E402

STATE_PATH_DEFAULT = os.path.join("data", "live_hyro_state.json")

BYBIT_ENDPOINTS = {
    "demo": ["https://api-demo.bybit.com", "https://api-demo.bytick.com"],
    "mainnet": ["https://api.bybit.com", "https://api.bytick.com"],
    "testnet": ["https://api-testnet.bybit.com", "https://api-testnet.bytick.com"],
}

BLOFIN_ENDPOINTS = {
    "demo": "https://demo-trading-openapi.blofin.com",
    "mainnet": "https://openapi.blofin.com",
    "testnet": "https://demo-trading-openapi.blofin.com",
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


def to_blofin_symbol(sym: str) -> str:
    base = to_bybit_symbol(sym)[:-4]
    return f"{base}-USDT"


def norm_coin(sym: str) -> str:
    s = sym.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if s.endswith("USDT"):
        return s[:-4]
    if s.endswith("USD"):
        return s[:-3]
    return s


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


# ==============================================================================
# 1. LECTURE TEMPS RÉEL DES BOUGIES 15M & 4H CLÔTURÉES (SANS BLOCAGE US)
# ==============================================================================
def fetch_closed_15m_4h(coin: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Télécharge en TEMPS RÉEL les dernières bougies 5m depuis Hyperliquid ou KuCoin (non géobloqués aux US sur GitHub Actions),
    et retourne `(df_15m_closed, df_4h_closed, last_close, recent_high_30m, recent_low_30m)`.
    """
    binance_sym = dc.ftmo_to_binance(f"{ norm_coin(coin) }USD")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (5 * 86400 * 1000)

    df_5m = None
    for live_fn in (dc.fetch_hyperliquid, dc.fetch_kucoin, dc.fetch_bybit_linear, dc.fetch_bybit):
        try:
            rows = live_fn(binance_sym, "5m", start_ms, now_ms)
            if rows and len(rows) >= 120:
                cand_df = dc._frame(rows)
                age_min = (pd.Timestamp.now(tz="UTC") - cand_df["time"].iloc[-1]).total_seconds() / 60.0
                if age_min <= 45.0:
                    df_5m = cand_df
                    break
        except Exception:
            continue

    if df_5m is None or len(df_5m) < 100:
        raise RuntimeError(f"Données 5m temps réel indisponibles pour {coin}")

    df_5m = df_5m.sort_values("time").reset_index(drop=True)
    last_px = float(df_5m["close"].iloc[-1])
    tail_6 = df_5m.iloc[-6:]
    recent_hi = float(tail_6["high"].max())
    recent_lo = float(tail_6["low"].min())

    now_utc = pd.Timestamp.now(tz="UTC")
    d = df_5m.set_index("time")

    r15 = d.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    r15 = r15[r15.index + pd.Timedelta(minutes=15) <= now_utc].reset_index()

    r4h = d.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    r4h = r4h[r4h.index + pd.Timedelta(hours=4) <= now_utc].reset_index()

    return r15, r4h, last_px, recent_hi, recent_lo


# ==============================================================================
# 2. CLIENT BYBIT API V5 (AVEC FALLBACK MIROIRS + SUPPORT PROXY/RELAIS EU)
# ==============================================================================
class BybitCloudFrontBlockedError(RuntimeError):
    """Levée lorsque l'IP US de GitHub Actions est bloquée par le CloudFront de Bybit (HTTP 403)."""


class BybitV5Client:
    """Client Bybit API v5 avec bascule automatique sur domaines miroirs et support `BYBIT_PROXY_URL` / `BYBIT_BASE_URL`."""

    def __init__(self, api_key: str, api_secret: str, env: str = "demo", timeout: int = 20):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        custom_base = (os.getenv("BYBIT_BASE_URL") or "").strip().rstrip("/")
        default_urls = BYBIT_ENDPOINTS.get(env, BYBIT_ENDPOINTS["demo"])
        self.base_urls = ([custom_base] + default_urls) if custom_base else list(default_urls)
        self.active_base_url = self.base_urls[0]
        self.recv_window = "10000"
        self.timeout = timeout
        self.session = requests.Session()
        proxy_url = (os.getenv("BYBIT_PROXY_URL") or os.getenv("HTTPS_PROXY") or "").strip()
        if proxy_url:
            self.session.proxies.update({"http": proxy_url, "https": proxy_url})
        self.instruments_cache = {}

    def _sign(self, ts_str: str, payload_str: str) -> str:
        raw = f"{ts_str}{self.api_key}{self.recv_window}{payload_str}"
        return hmac.new(self.api_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()

    def request(self, method: str, path: str, params: dict = None, auth: bool = True) -> dict:
        params = params or {}
        if method.upper() == "GET":
            query_str = urlencode(sorted(params.items())) if params else ""
            payload_str = query_str
            body = None
        else:
            query_str = ""
            payload_str = json.dumps(params, separators=(",", ":"))
            body = payload_str.encode("utf-8")

        last_err = None
        urls_to_try = [self.active_base_url] + [u for u in self.base_urls if u != self.active_base_url]

        for base_u in urls_to_try:
            url = f"{base_u}{path}" + (f"?{query_str}" if query_str else "")
            ts_str = str(int(time.time() * 1000))
            headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 Antigravity-HyroOneStep/1.0"}
            if auth:
                if not self.api_key or not self.api_secret:
                    raise RuntimeError("Clés BYBIT_API_KEY / BYBIT_API_SECRET manquantes.")
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-TIMESTAMP": ts_str,
                    "X-BAPI-RECV-WINDOW": self.recv_window,
                    "X-BAPI-SIGN": self._sign(ts_str, payload_str),
                })
            try:
                resp = self.session.request(method.upper(), url, headers=headers, data=body, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = RuntimeError(f"Erreur réseau Bybit ({base_u}{path}) : {e}")
                continue

            if resp.status_code == 403 and ("CloudFront" in resp.text or "block access from your country" in resp.text):
                last_err = BybitCloudFrontBlockedError(
                    f"Bybit bloque les adresses IP américaines des serveurs GitHub Actions (HTTP 403 CloudFront sur {base_u})."
                )
                continue

            if resp.status_code >= 400:
                raise RuntimeError(f"Erreur HTTP {resp.status_code} Bybit ({path}) : {resp.text[:300]}")

            self.active_base_url = base_u
            data = resp.json()
            ret_code = data.get("retCode", 0)
            if ret_code not in (0, 110043):
                raise RuntimeError(f"Erreur API Bybit ({path}, code={ret_code}) : {data.get('retMsg')}")
            return data.get("result", {})

        if last_err is not None:
            raise last_err
        raise RuntimeError(f"Échec requête Bybit sur {path}")

    def get_instrument_spec(self, symbol: str) -> dict:
        sym = to_bybit_symbol(symbol)
        if sym not in self.instruments_cache:
            res = self.request("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": sym}, auth=False)
            lst = res.get("list", [])
            if not lst:
                self.instruments_cache[sym] = {"qty_step": 0.001, "min_qty": 0.001, "tick_size": 0.01}
            else:
                item = lst[0]
                lot = item.get("lotSizeFilter", {})
                pf = item.get("priceFilter", {})
                self.instruments_cache[sym] = {
                    "qty_step": float(lot.get("qtyStep", 0.001)),
                    "min_qty": float(lot.get("minOrderQty", 0.001)),
                    "tick_size": float(pf.get("tickSize", 0.01)),
                }
        return self.instruments_cache[sym]

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
                "symbol": to_bybit_symbol(p["symbol"]),
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
            if o.get("stopOrderType") in ("TakeProfit", "StopLoss", "TrailingStop"):
                continue
            out.append({
                "order_id": o["orderId"],
                "symbol": to_bybit_symbol(o["symbol"]),
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
                "symbol": to_bybit_symbol(symbol),
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            })
        except Exception:
            pass

    def set_tp_sl(self, symbol: str, sl_price: float = None, tp_price: float = None, side_int: int = None, position_side: str = "net", margin_mode: str = "isolated"):
        sym = to_bybit_symbol(symbol)
        spec = self.get_instrument_spec(sym)
        tick = spec["tick_size"]
        payload = {"category": "linear", "symbol": sym, "tpslMode": "Full", "positionIdx": 0}
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
        sym = to_bybit_symbol(symbol)
        spec = self.get_instrument_spec(sym)
        q_step = spec["qty_step"]
        tick = spec["tick_size"]
        qty_r = round_step(qty, q_step, "floor")
        if qty_r < spec["min_qty"]:
            raise ValueError(f"Quantité {qty_r} < min_qty {spec['min_qty']} sur {sym}")

        payload = {
            "category": "linear",
            "symbol": sym,
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
            "symbol": to_bybit_symbol(symbol),
            "orderId": order_id,
        })

    def close_position_market(self, pos: dict):
        opp_side = -1 if pos["side_int"] == 1 else 1
        return self.place_order(pos["symbol"], opp_side, "MARKET", pos["size"], reduce_only=True)


# ==============================================================================
# 3. CLIENT BLOFIN API V1 (2e EXCHANGE OFFICIEL HYROTRADER — NON BLOQUÉ AUX US)
# ==============================================================================
class BloFinV1Client:
    """Client natif pour BloFin API v1 (`openapi.blofin.com` / `demo-trading-openapi.blofin.com`)."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str, env: str = "demo", timeout: int = 20):
        self.api_key = (api_key or "").strip()
        self.api_secret = (api_secret or "").strip()
        self.passphrase = (passphrase or "").strip()
        self.base_url = BLOFIN_ENDPOINTS.get(env, BLOFIN_ENDPOINTS["demo"])
        self.timeout = timeout
        self.session = requests.Session()
        self.instruments_cache = {}

    def _sign(self, request_path: str, method: str, ts_str: str, nonce: str, body_str: str = "") -> str:
        prehash = f"{request_path}{method.upper()}{ts_str}{nonce}{body_str}"
        mac = hmac.new(self.api_secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).hexdigest()
        return base64.b64encode(mac.encode("utf-8")).decode("utf-8")

    def request(self, method: str, path: str, params: dict = None, auth: bool = True) -> dict:
        params = params or {}
        if method.upper() == "GET":
            q = urlencode(sorted(params.items())) if params else ""
            req_path = f"{path}?{q}" if q else path
            body_str = ""
            body = None
        else:
            req_path = path
            body_str = json.dumps(params, separators=(",", ":")) if params else ""
            body = body_str.encode("utf-8") if body_str else None

        url = f"{self.base_url}{req_path}"
        ts_str = str(int(time.time() * 1000))
        nonce = str(uuid.uuid4())
        headers = {"Content-Type": "application/json", "User-Agent": "Antigravity-HyroOneStep/1.0"}
        if auth:
            headers.update({
                "ACCESS-KEY": self.api_key,
                "ACCESS-SIGN": self._sign(req_path, method, ts_str, nonce, body_str),
                "ACCESS-TIMESTAMP": ts_str,
                "ACCESS-NONCE": nonce,
                "ACCESS-PASSPHRASE": self.passphrase,
            })

        resp = self.session.request(method.upper(), url, headers=headers, data=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Erreur HTTP {resp.status_code} BloFin ({path}) : {resp.text[:300]}")
        data = resp.json()
        code = str(data.get("code", "0"))
        if code != "0":
            raise RuntimeError(f"Erreur API BloFin ({path}, code={code}) : {data.get('msg')}")
        return data.get("data", {})

    def get_instrument_spec(self, symbol: str) -> dict:
        inst_id = to_blofin_symbol(symbol)
        if inst_id not in self.instruments_cache:
            res = self.request("GET", "/api/v1/market/instruments", {"instId": inst_id}, auth=False)
            lst = res if isinstance(res, list) else []
            if not lst:
                self.instruments_cache[inst_id] = {"contract_val": 1.0, "min_size": 1.0, "lot_size": 1.0, "tick_size": 0.01}
            else:
                it = lst[0]
                self.instruments_cache[inst_id] = {
                    "contract_val": float(it.get("contractValue", 1.0)),
                    "min_size": float(it.get("minSize", 1.0)),
                    "lot_size": float(it.get("lotSize", 1.0)),
                    "tick_size": float(it.get("tickSize", 0.01)),
                }
        return self.instruments_cache[inst_id]

    def get_account_overview(self) -> dict:
        res = self.request("GET", "/api/v1/account/balance", {"accountType": "futures"})
        if isinstance(res, list) and res:
            res = res[0]
        equity = float(res.get("totalEquity") or 0.0)
        details = res.get("details", [])
        wallet = equity
        avail = equity
        for d in details:
            if d.get("currency") == "USDT":
                wallet = float(d.get("balance") or d.get("equity") or equity)
                avail = float(d.get("available") or wallet)
                if equity <= 0:
                    equity = float(d.get("equity") or wallet)
                break
        return {"equity": equity, "wallet_balance": wallet, "available_balance": avail}

    def get_positions(self) -> list:
        res = self.request("GET", "/api/v1/account/positions", {})
        out = []
        for p in (res if isinstance(res, list) else []):
            pos_sz = float(p.get("positions") or 0.0)
            if abs(pos_sz) <= 0:
                continue
            inst_id = p.get("instId", "")
            spec = self.get_instrument_spec(inst_id)
            coin_sz = abs(pos_sz) * spec["contract_val"]
            side_int = 1 if pos_sz > 0 or str(p.get("positionSide")).lower() == "long" else -1
            out.append({
                "symbol": to_bybit_symbol(inst_id),
                "side": "long" if side_int == 1 else "short",
                "side_int": side_int,
                "position_side": str(p.get("positionSide") or "net").lower(),
                "margin_mode": str(p.get("marginMode") or "isolated").lower(),
                "size": coin_sz,
                "contracts": abs(pos_sz),
                "entry_price": float(p.get("averagePrice") or 0.0),
                "mark_price": float(p.get("markPrice") or p.get("averagePrice") or 0.0),
                "unrealized_pnl": float(p.get("unrealizedPnl") or 0.0),
                "sl_price": 0.0,
                "tp_price": 0.0,
                "leverage": float(p.get("leverage") or 30.0),
            })
        return out

    def get_open_orders(self) -> list:
        res = self.request("GET", "/api/v1/trade/orders-pending", {})
        out = []
        for o in (res if isinstance(res, list) else []):
            inst_id = o.get("instId", "")
            out.append({
                "order_id": o["orderId"],
                "symbol": to_bybit_symbol(inst_id),
                "side": "long" if o.get("side") == "buy" else "short",
                "side_int": 1 if o.get("side") == "buy" else -1,
                "price": float(o.get("price") or 0.0),
                "qty": float(o.get("size") or 0.0),
            })
        return out

    def ensure_leverage(self, symbol: str, leverage: int = 30):
        try:
            self.request("POST", "/api/v1/account/set-leverage", {
                "instId": to_blofin_symbol(symbol),
                "leverage": str(leverage),
                "marginMode": "isolated",
            })
        except Exception:
            pass

    def cancel_all_tpsl(self, symbol: str):
        inst_id = to_blofin_symbol(symbol)
        try:
            pending = self.request("GET", "/api/v1/trade/orders-tpsl-pending", {"instId": inst_id})
            for item in (pending if isinstance(pending, list) else []):
                tpsl_id = item.get("tpslId") or item.get("orderId")
                if tpsl_id:
                    try:
                        self.request("POST", "/api/v1/trade/cancel-tpsl", [{"instId": inst_id, "tpslId": str(tpsl_id)}])
                    except Exception:
                        try:
                            self.request("POST", "/api/v1/trade/cancel-tpsl", {"instId": inst_id, "tpslId": str(tpsl_id)})
                        except Exception:
                            pass
        except Exception:
            pass

    def set_tp_sl(self, symbol: str, sl_price: float = None, tp_price: float = None, side_int: int = None, position_side: str = "net", margin_mode: str = "isolated"):
        inst_id = to_blofin_symbol(symbol)
        spec = self.get_instrument_spec(inst_id)
        tick = spec["tick_size"]
        # Sur BloFin, `side` est le sens de l'ordre de clôture : "sell" pour fermer un LONG, "buy" pour fermer un SHORT
        close_side = "sell" if (side_int is None or int(side_int) == 1) else "buy"
        self.cancel_all_tpsl(symbol)
        payload = {
            "instId": inst_id,
            "marginMode": margin_mode or "isolated",
            "positionSide": position_side or "net",
            "side": close_side,
            "size": "-1",
            "reduceOnly": "true",
        }
        if sl_price is not None and sl_price > 0:
            payload["slTriggerPrice"] = str(round_step(sl_price, tick, "nearest"))
            payload["slOrderPrice"] = "-1"
        if tp_price is not None and tp_price > 0:
            payload["tpTriggerPrice"] = str(round_step(tp_price, tick, "nearest"))
            payload["tpOrderPrice"] = "-1"
        return self.request("POST", "/api/v1/trade/order-tpsl", payload)

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
        inst_id = to_blofin_symbol(symbol)
        spec = self.get_instrument_spec(inst_id)
        contracts = round_step(qty / spec["contract_val"], spec["lot_size"], "floor")
        if contracts < spec["min_size"]:
            raise ValueError(f"Contrats {contracts} < min_size {spec['min_size']} sur {inst_id}")
        tick = spec["tick_size"]
        payload = {
            "instId": inst_id,
            "marginMode": "isolated",
            "side": "buy" if side_int == 1 else "sell",
            "orderType": "market" if order_type.upper() == "MARKET" else "limit",
            "size": str(contracts),
        }
        if order_type.upper() == "LIMIT" and price:
            payload["price"] = str(round_step(price, tick, "nearest"))
        if reduce_only:
            payload["reduceOnly"] = "true"
        else:
            if sl_price is not None and sl_price > 0:
                payload["slTriggerPrice"] = str(round_step(sl_price, tick, "nearest"))
                payload["slOrderPrice"] = "-1"
            if tp_price is not None and tp_price > 0:
                payload["tpTriggerPrice"] = str(round_step(tp_price, tick, "nearest"))
                payload["tpOrderPrice"] = "-1"
        return self.request("POST", "/api/v1/trade/order", payload)

    def cancel_order(self, symbol: str, order_id: str):
        return self.request("POST", "/api/v1/trade/cancel-order", {
            "instId": to_blofin_symbol(symbol),
            "orderId": order_id,
        })

    def close_position_market(self, pos: dict):
        opp_side = -1 if pos["side_int"] == 1 else 1
        return self.place_order(pos["symbol"], opp_side, "MARKET", pos["size"], reduce_only=True)


# ==============================================================================
# 4. GESTION DE L'ÉTAT PERSISTANT & SIMULATEUR TEMPS RÉEL (DRY_RUN)
# ==============================================================================
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
                    st.setdefault("active_positions", {})
                    st.setdefault("last_wallet_balance", initial_capital)
                    st.setdefault("dry_run_balance", initial_capital)
                    st.setdefault("dry_run_positions", {})
                    st.setdefault("dry_run_orders", {})
                    st.setdefault("cloudfront_warned", False)
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
        "active_positions": {},
        "last_wallet_balance": float(initial_capital),
        "dry_run_balance": float(initial_capital),
        "dry_run_positions": {},
        "dry_run_orders": {},
        "cloudfront_warned": False,
    }


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def update_dry_run_portfolio(state: dict, market_snapshots: dict, p: dict):
    """
    Met à jour en temps réel les positions et ordres limites simulés (`DRY_RUN`)
    à partir des dernières bougies 5m (high, low, close) de chaque crypto.
    """
    lev = float(p.get("leverage", 30.0))
    dry_pos = state.setdefault("dry_run_positions", {})
    dry_ord = state.setdefault("dry_run_orders", {})
    bal = float(state.get("dry_run_balance", state.get("initial_balance", 10000.0)))
    utc_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1) Exécution des ordres limites simulés si le prix 5m a touché la limite
    for sym, ord_d in list(dry_ord.items()):
        if sym not in market_snapshots or sym in dry_pos:
            continue
        _, _, last_px, hi_30m, lo_30m = market_snapshots[sym]
        side_int = int(ord_d["side_int"])
        lim_px = float(ord_d["price"])
        if (side_int == 1 and lo_30m <= lim_px) or (side_int == -1 and hi_30m >= lim_px):
            dry_pos[sym] = {
                "symbol": sym,
                "side": "long" if side_int == 1 else "short",
                "side_int": side_int,
                "size": float(ord_d["qty"]),
                "entry_price": lim_px,
                "mark_price": last_px,
                "sl_price": float(ord_d["sl_price"]),
                "tp_price": float(ord_d["tp_price"]),
                "margin_usdt": float(ord_d["margin_usdt"]),
                "leverage": lev,
                "poi": float(ord_d.get("poi", lim_px)),
            }
            dry_ord.pop(sym, None)
            if utc_day not in state["days_traded"]:
                state["days_traded"].append(utc_day)
            msg = f"🧪 [SIMULATION LIVE | {sym}] Ordre LIMIT {('LONG' if side_int == 1 else 'SHORT')} exécuté @ {lim_px:.4f} !"
            print(msg, flush=True)
            tg_send(msg)

    # 2) Vérification TP / SL et calcul du PnL latent des positions simulées ouvertes
    total_unr = 0.0
    for sym, pos in list(dry_pos.items()):
        if sym not in market_snapshots:
            continue
        _, _, last_px, hi_30m, lo_30m = market_snapshots[sym]
        pos["mark_price"] = last_px
        side_int = int(pos["side_int"])
        en = float(pos["entry_price"])
        sl = float(pos["sl_price"])
        tp = float(pos["tp_price"])
        margin = float(pos["margin_usdt"])

        hit_sl = (lo_30m <= sl) if side_int == 1 else (hi_30m >= sl)
        hit_tp = (hi_30m >= tp) if side_int == 1 else (lo_30m <= tp)

        if hit_sl or hit_tp:
            exit_px = sl if hit_sl else tp
            roi_gross = scv4._roi_at_price(side_int, en, exit_px, lev)
            roi_net = roi_gross - (2.0 * 0.00035 * lev * 100.0)
            pnl_usdt = margin * (roi_net / 100.0)
            bal += pnl_usdt
            state["dry_run_balance"] = bal
            if roi_gross < -5.0:
                st_s = state["symbols"].setdefault(sym, {})
                st_s["last_sl_ts"] = time.time()
                st_s["last_sl_poi"] = float(pos.get("poi", en))
            dry_pos.pop(sym, None)
            tag = "TP 🎯" if hit_tp else ("SL 🛑" if roi_gross < -5.0 else "Trailing SL 🔒")
            msg = (
                f"🧪 [SIMULATION LIVE | {sym}] Clôture {tag} @ {exit_px:.4f} | "
                f"PnL : {pnl_usdt:+.2f} USDT ({roi_net:+.1f}% ROI) | Nouveau solde : {bal:,.2f} USDT"
            )
            print(msg, flush=True)
            tg_send(msg)
        else:
            roi_cur = scv4._roi_at_price(side_int, en, last_px, lev)
            unr = margin * (roi_cur / 100.0)
            pos["unrealized_pnl"] = unr
            total_unr += unr

    equity = bal + total_unr
    open_positions = list(dry_pos.values())
    open_orders = [
        {
            "order_id": f"dry_{sym}",
            "symbol": sym,
            "side": o["side"],
            "side_int": o["side_int"],
            "price": o["price"],
            "qty": o["qty"],
        }
        for sym, o in dry_ord.items()
    ]
    return bal, equity, open_positions, open_orders


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

    # Règle 3a : Pause journalière de consistance (+3.60% réalisé ou total)
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
    ap = argparse.ArgumentParser(description="Exécuteur Live CRYPTO V4 pour HyroTrader One-Step (Bybit / BloFin / Dry-Run)")
    ap.add_argument("--exchange", default="auto", choices=["auto", "bybit", "blofin", "dry_run"], help="auto = BloFin si clés présentes, sinon Bybit, sinon Dry-Run")
    ap.add_argument("--env", default="demo", choices=["demo", "mainnet", "testnet", "dry_run"], help="demo = Compte Challenge Démo | mainnet = Compte financé")
    ap.add_argument("--symbols", default="BTC,ETH,AVAX,INJ")
    ap.add_argument("--initial_capital", type=float, default=10000.0, help="Capital initial du challenge Hyro One-Step (défaut: 10 000 USDT)")
    ap.add_argument("--state_path", default=STATE_PATH_DEFAULT)
    ap.add_argument("--params", default="")
    args = ap.parse_args()

    p = scv4.parse_params(args.params)
    lev = int(round(float(p.get("leverage", 30.0))))
    syms = [to_bybit_symbol(s) for s in args.symbols.split(",") if s.strip()]

    bybit_key = os.getenv("BYBIT_API_KEY", "").strip()
    bybit_sec = os.getenv("BYBIT_API_SECRET", "").strip()
    blofin_key = os.getenv("BLOFIN_API_KEY", "").strip()
    blofin_sec = os.getenv("BLOFIN_API_SECRET", "").strip()
    blofin_pass = os.getenv("BLOFIN_PASSPHRASE", "").strip()

    state = load_state(args.state_path, args.initial_capital)

    # 1. Préchargement des bougies 15m/4h clôturées en temps réel via Hyperliquid / KuCoin (jamais bloqué aux US)
    market_snapshots = {}
    for sym in syms:
        try:
            market_snapshots[sym] = fetch_closed_15m_4h(sym)
        except Exception as e:
            print(f"⚠️ [{sym}] Erreur lecture bougies temps réel : {e}", flush=True)

    # 2. Sélection du connecteur (BloFin, Bybit ou Simulation Temps Réel Dry-Run)
    exch = args.exchange
    if args.env == "dry_run" or exch == "dry_run":
        exch = "dry_run"
    elif exch == "auto":
        if blofin_key and blofin_sec and blofin_pass:
            exch = "blofin"
        elif bybit_key and bybit_sec:
            exch = "bybit"
        else:
            exch = "dry_run"

    client = None
    is_dry = (exch == "dry_run")

    if not is_dry:
        try:
            if exch == "blofin":
                client = BloFinV1Client(blofin_key, blofin_sec, blofin_pass, env=args.env)
            else:
                client = BybitV5Client(bybit_key, bybit_sec, env=args.env)
            acc = client.get_account_overview()
            wallet_bal = acc["wallet_balance"]
            equity = acc["equity"]
            open_positions = client.get_positions()
            open_orders = client.get_open_orders()
        except BybitCloudFrontBlockedError as e:
            print(f"⚠️ {e}\n   -> Bascule automatique sur le SIMULATEUR LIVE TEMPS RÉEL (DRY_RUN) pour ne manquer aucun signal !", flush=True)
            if not state.get("cloudfront_warned"):
                state["cloudfront_warned"] = True
                tg_send(
                    "⚠️ Info GitHub Actions : les serveurs GitHub étant aux USA, Bybit bloque leur IP (HTTP 403 CloudFront).\n"
                    "✅ Le bot a basculé automatiquement en Simulation Live Temps Réel (DRY_RUN sur bougies Hyperliquid/KuCoin).\n"
                    "💡 Pour passer les ordres réels depuis GitHub Actions : utilise soit un compte HyroTrader sur **BloFin** (secrets BLOFIN_API_KEY / BLOFIN_API_SECRET / BLOFIN_PASSPHRASE), soit un proxy européen (secret BYBIT_PROXY_URL), soit lance le script sur ton PC Linux."
                )
            is_dry = True
            exch = "dry_run (fallback CloudFront US)"

    if is_dry:
        wallet_bal, equity, open_positions, open_orders = update_dry_run_portfolio(state, market_snapshots, p)

    utc_now = datetime.now(timezone.utc)
    utc_day = utc_now.strftime("%Y-%m-%d")

    if state["cur_utc_day"] != utc_day:
        state["cur_utc_day"] = utc_day
        state["day_start_balance"] = equity
        state["day_locked_reason"] = ""
        print(f"📅 Nouveau jour UTC ({utc_day}) : équité de référence journalière fixée à {equity:,.2f} USDT (solde {wallet_bal:,.2f})", flush=True)

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
        f"⚡ [HYRO ONE-STEP LIVE | {exch.upper()} / {args.env.upper()}] {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
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
            if is_dry:
                state["dry_run_positions"] = {}
                state["dry_run_orders"] = {}
                state["dry_run_balance"] = equity
            else:
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
    # VERROU 1 : Règle 3 (Consistance 40%)
    # ------------------------------------------------------------------
    hard_day_tp = float(p.get("hyro_hard_day_tp_pct", 0.0))
    pause_cap_pct = float(p.get("hyro_pause_at_cap_pct", 3.60))
    if hard_day_tp > 0.0 and day_total_pct >= hard_day_tp and not state.get("day_locked_reason"):
        state["day_locked_reason"] = f"HARD_DAILY_PROFIT_LOCK (+{day_total_pct:.2f}% >= +{hard_day_tp:.2f}%)"
        if is_dry:
            state["dry_run_positions"] = {}
            state["dry_run_orders"] = {}
            state["dry_run_balance"] = equity
        else:
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
    elif day_total_pct >= pause_cap_pct and open_orders:
        if is_dry:
            state["dry_run_orders"] = {}
        else:
            for ord_d in open_orders:
                client.cancel_order(ord_d["symbol"], ord_d["order_id"])
        open_orders = []
        print(f"🎯 [Hyro Règle 3] Journée à {day_total_pct:+.2f}% (>= +{pause_cap_pct:.2f}%) : ordres limites au repos annulés.", flush=True)

    # ------------------------------------------------------------------
    # VERROU 2 : Disjoncteur Journalier (-2.20% sur la journée UTC)
    # ------------------------------------------------------------------
    daily_stop = float(p.get("hyro_daily_stop_pct", 2.20))
    if day_total_pct <= -daily_stop and not state.get("day_locked_reason"):
        state["day_locked_reason"] = f"DAILY_STOP_LOSS ({day_total_pct:+.2f}% <= -{daily_stop:.2f}%)"
        if is_dry:
            state["dry_run_positions"] = {}
            state["dry_run_orders"] = {}
            state["dry_run_balance"] = equity
        else:
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
    # ÉTAPE 2 : Suivi des ouvertures/clôtures & Trailing SL 5 paliers tight_high
    # ------------------------------------------------------------------
    pos_by_sym = {pos["symbol"]: pos for pos in open_positions}
    ord_by_sym = {}
    for o in open_orders:
        ord_by_sym.setdefault(o["symbol"], []).append(o)

    if not is_dry:
        prev_active = state.get("active_positions", {})
        prev_bal = float(state.get("last_wallet_balance", wallet_bal))

        # 2a) Détection des positions clôturées sur l'exchange entre deux cycles (TP / SL / Trailing)
        for sym, old_pos in list(prev_active.items()):
            if sym not in pos_by_sym:
                st_sym = state["symbols"].setdefault(sym, {})
                pnl_delta = wallet_bal - prev_bal
                if pnl_delta < -1.0:
                    st_sym["last_sl_ts"] = time.time()
                    st_sym["last_sl_poi"] = float(old_pos.get("poi", old_pos.get("entry_price", -1.0)))
                st_sym["active_sl_price"] = 0.0
                st_sym["active_tp_price"] = 0.0
                st_sym["peak_px"] = 0.0
                msg = (
                    f"🏁 [HYRO LIVE | {sym}] Position {old_pos.get('side', '').upper()} clôturée "
                    f"(entrée {old_pos.get('entry_price', 0.0):.4f}) | "
                    f"Solde : {wallet_bal:,.2f} USDT (Équité : {equity:,.2f} USDT)"
                )
                print(msg, flush=True)
                tg_send(msg)

        # 2b) Détection des nouvelles positions ouvertes (ex: ordre LIMIT exécuté entre deux cycles)
        for sym, pos in pos_by_sym.items():
            if sym not in prev_active:
                st_sym = state["symbols"].setdefault(sym, {})
                if not st_sym.get("just_opened_market"):
                    msg = (
                        f"✅ [HYRO LIVE | {sym}] TRADE PRIS (Ordre LIMIT exécuté) : {pos['side'].upper()} @ {pos['entry_price']:.4f}\n"
                        f"• Taille : {pos['size']:.4f} | SL initial : -50% ROI | TP : +300% ROI"
                    )
                    print(msg, flush=True)
                    tg_send(msg)
                st_sym.pop("just_opened_market", None)

        state["active_positions"] = {
            sym: {
                "side": pos["side"],
                "entry_price": float(pos["entry_price"]),
                "size": float(pos["size"]),
                "poi": float(state["symbols"].get(sym, {}).get("pending_poi", pos["entry_price"])),
            }
            for sym, pos in pos_by_sym.items()
        }
        state["last_wallet_balance"] = wallet_bal
        # Sauvegarde immédiate de l'état après détection des ouvertures/clôtures (évite tout doublon Telegram en cas d'erreur API ultérieure)
        save_state(args.state_path, state)

    for sym, pos in pos_by_sym.items():
        st_sym = state["symbols"].setdefault(sym, {})
        prev_peak = float(st_sym.get("peak_px", pos["entry_price"]))
        known_sl = pos["sl_price"] if pos["sl_price"] > 0 else float(st_sym.get("active_sl_price", 0.0))
        known_tp = pos["tp_price"] if pos["tp_price"] > 0 else float(st_sym.get("active_tp_price", 0.0))
        trail_info = scv4.compute_live_trailing_sl(
            side=pos["side_int"],
            entry=pos["entry_price"],
            px=pos["mark_price"],
            current_sl=known_sl,
            p=p,
            peak_px=prev_peak,
        )
        st_sym["peak_px"] = trail_info["peak_px"]
        need_tp = known_tp <= 0 or known_sl <= 0
        if trail_info["should_update"] or need_tp:
            new_sl = trail_info["target_sl_price"] if trail_info["tier"] > 0 else scv4._price_at_roi(pos["side_int"], pos["entry_price"], -float(p["sl_roi_pct"]), float(p["leverage"]))
            new_tp = trail_info["tp_price"]
            try:
                if is_dry:
                    state["dry_run_positions"][sym]["sl_price"] = new_sl
                    state["dry_run_positions"][sym]["tp_price"] = new_tp
                else:
                    client.set_tp_sl(
                        sym,
                        sl_price=new_sl,
                        tp_price=new_tp,
                        side_int=pos["side_int"],
                        position_side=pos.get("position_side", "net"),
                        margin_mode=pos.get("margin_mode", "isolated"),
                    )
                st_sym["active_sl_price"] = new_sl
                st_sym["active_tp_price"] = new_tp
                if trail_info["should_update"]:
                    msg = (
                        f"🔒 [{sym}] Trailing SL mis à jour (Palier {trail_info['tier']} | ROI pic {trail_info['roi_peak']:+.1f}%) : "
                        f"SL -> {new_sl:.4f} ({trail_info['target_sl_roi']:+.0f}% ROI) | TP -> {new_tp:.4f}"
                    )
                    print(msg, flush=True)
                    tg_send(msg)
            except Exception as e:
                print(f"⚠️ [{sym}] Échec mise à jour Trailing SL/TP : {e}", flush=True)

    # ------------------------------------------------------------------
    # ÉTAPE 3 & 4 : Analyse SMC (15m/4h) & Ouverture calibrée Hyro One-Step
    # ------------------------------------------------------------------
    for sym in syms:
        if sym in pos_by_sym:
            continue
        if sym not in market_snapshots:
            continue

        st_sym = state["symbols"].setdefault(sym, {})
        st_sym["peak_px"] = 0.0
        last_sl_ts = float(st_sym.get("last_sl_ts", 0.0))
        hours_since_sl = (time.time() - last_sl_ts) / 3600.0 if last_sl_ts > 0 else 999.0
        last_sl_poi = float(st_sym.get("last_sl_poi", -1.0))

        df_15m, df_4h, last_px, _, _ = market_snapshots[sym]
        sig = scv4.evaluate_live_signal(
            df_15m=df_15m,
            df_4h=df_4h,
            px=last_px,
            p=p,
            hours_since_last_sl=hours_since_sl,
            last_sl_poi=last_sl_poi,
        )

        cur_orders = ord_by_sym.get(sym, [])
        if sig["action"] == "SKIP":
            if cur_orders:
                if is_dry:
                    state["dry_run_orders"].pop(sym, None)
                else:
                    for o in cur_orders:
                        client.cancel_order(sym, o["order_id"])
                print(f"🔄 [{sym}] Ordre limite annulé ({sig['reason']})", flush=True)
            else:
                print(f"⏸️ [{sym}] SKIP : {sig['reason']}", flush=True)
            continue

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
            if is_dry:
                state["dry_run_orders"].pop(sym, None)
            else:
                for o in cur_orders:
                    client.cancel_order(sym, o["order_id"])

        if is_dry:
            if sig["action"] == "MARKET":
                state["dry_run_positions"][sym] = {
                    "symbol": sym,
                    "side": sig["side"],
                    "side_int": sig["side_int"],
                    "size": qty,
                    "entry_price": entry_px,
                    "mark_price": last_px,
                    "sl_price": sig["sl_price"],
                    "tp_price": sig["tp_price"],
                    "margin_usdt": sizing["margin_usdt"],
                    "leverage": float(lev),
                    "poi": sig["poi"],
                }
                pos_by_sym[sym] = state["dry_run_positions"][sym]
                if utc_day not in state["days_traded"]:
                    state["days_traded"].append(utc_day)
            else:
                state["dry_run_orders"][sym] = {
                    "symbol": sym,
                    "side": sig["side"],
                    "side_int": sig["side_int"],
                    "qty": qty,
                    "price": entry_px,
                    "sl_price": sig["sl_price"],
                    "tp_price": sig["tp_price"],
                    "margin_usdt": sizing["margin_usdt"],
                    "poi": sig["poi"],
                }
            msg = (
                f"🧪 [SIMULATION LIVE | {sym}] {sig['action']} {sig['side'].upper()} @ {entry_px:.4f} | "
                f"SL {sig['sl_price']:.4f} (-50%) | TP {sig['tp_price']:.4f} (+300%) | "
                f"Risque 1R={sizing['one_r_usdt']:.1f}$ ({sizing['risk_pct']:.2f}% | {sizing['reason']})"
            )
            print(msg, flush=True)
            tg_send(msg)
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
            st_sym["pending_poi"] = float(sig["poi"])
            st_sym["active_sl_price"] = float(sig["sl_price"])
            st_sym["active_tp_price"] = float(sig["tp_price"])
            if sig["action"] == "MARKET":
                state["active_positions"][sym] = {
                    "side": sig["side"],
                    "entry_price": float(entry_px),
                    "size": float(qty),
                    "poi": float(sig["poi"]),
                }
            if utc_day not in state["days_traded"]:
                state["days_traded"].append(utc_day)
            action_label = "TRADE PRIS (MARKET)" if sig["action"] == "MARKET" else "ORDRE LIMIT PLACÉ"
            msg = (
                f"🚀 [HYRO LIVE | {sym}] {action_label} {sig['side'].upper()} @ {entry_px:.4f}\n"
                f"• SL : {sig['sl_price']:.4f} (-50% ROI) | TP : {sig['tp_price']:.4f} (+300% ROI)\n"
                f"• Risque 1R : {sizing['one_r_usdt']:.1f} USDT ({sizing['risk_pct']:.2f}% | Marge {sizing['margin_usdt']:.1f}$ x{lev})\n"
                f"• Règles Hyro : {sizing['reason']}"
            )
            print(msg, flush=True)
            tg_send(msg)

    save_state(args.state_path, state)


if __name__ == "__main__":
    main()
