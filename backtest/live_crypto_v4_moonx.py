"""
backtest/live_crypto_v4_moonx.py — Exécuteur Live autonome CRYPTO V4 Optimisée sur MoonX (via connecteur MCP).

Fonctionnement (conçu pour tourner gratuitement toutes les 30 minutes via GitHub Actions + cron-job.org) :
  1. Connexion au serveur MCP MoonX (`https://api.moon-x.io/mcp?token=mcp_...`) via JSON-RPC 2.0 / SSE.
  2. Étape 1 — Lecture de l'état réel sur MoonX :
     - `get_account_overview` : lit le capital total à l'instant t (`Solde_t`) et la marge libre.
     - Sizing dynamique NON bridé à 18 USDT :
       * `1 R (risque SL) = Solde_t * (risk_pct / 100)`
       * `Marge isolée (levier x30, SL -50% ROI) = 2 * 1 R = Solde_t * (2 * risk_pct / 100)`
     - `list_futures_positions`, `list_futures_orders`, `list_futures_tp_sl_orders`, `get_futures_trade_history`.
  3. Étape 2 — Gestion des positions ouvertes (levier x30 isolé) :
     - Attache le TP (+300% ROI) et SL (-50% ROI) dès qu'un ordre limite a été exécuté.
     - Trailing du SL par paliers (+50% -> BE 0%, +100% -> +50%, +150% -> +100%, +200% -> +150%).
  4. Étape 3 & 4 — Détection SMC + Filtres Anti-Krach & Ouverture (sur BTC, ETH, AVAX, INJ par défaut) :
     - Biais H4 (8 bougies 4h), Fenêtre M15 encadrée [0.4% .. 1.6%] (16 bougies 15m), POI 50%,
       Cooldown 2h après SL, Anti-répétition du même POI invalidé.
     - Ré-analyse des ordres limites au repos (annule si thèse cassée ou écart > 0.1%).
     - Ouverture via `open_futures_position` (marché) ou `open_futures_limit_order` (limite) + `set_futures_tp_sl`.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_crypto as dc  # noqa: E402
import strategy_crypto_v4 as scv4  # noqa: E402


STATE_PATH_DEFAULT = os.path.join("data", "live_crypto_v4_state.json")


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


def build_moonx_mcp_url(raw_url_or_token: str) -> str:
    val = (raw_url_or_token or "").strip()
    if not val:
        raise ValueError(
            "Secret MOONX_MCP_URL (ou MOONX_TOKEN) manquant ! "
            "Ajoutez votre lien MCP MoonX (https://api.moon-x.io/mcp?token=mcp_...) dans GitHub Secrets."
        )
    if val.startswith("http://") or val.startswith("https://"):
        return val
    if val.startswith("mcp_"):
        return f"https://api.moon-x.io/mcp?token={val}"
    return f"https://api.moon-x.io/mcp?token={val}"


class MoonXMCPClient:
    """
    Client MCP (Model Context Protocol) compatible Streamable HTTP & JSON-RPC / SSE pour MoonX (`https://api.moon-x.io/mcp`).
    Découvre dynamiquement les outils (`tools/list`) et adapte les paramètres aux schémas JSON exacts du serveur.
    """

    def __init__(self, mcp_url: str, timeout: int = 30):
        self.url = build_moonx_mcp_url(mcp_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session_id = None
        self.req_id = 0
        self.tools_schema = {}

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "Antigravity-CryptoV4-MoonX/1.0",
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        # Si l'utilisateur a mis un token dans MOONX_TOKEN séparé sans ?token= dans l'URL
        extra_tok = (os.getenv("MOONX_TOKEN") or "").strip()
        if extra_tok and "token=" not in self.url:
            h["Authorization"] = f"Bearer {extra_tok}"
        return h

    def _parse_response(self, resp: requests.Response) -> dict:
        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid

        ctype = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text.strip()
        if not text:
            return {}

        if "text/event-stream" in ctype or text.startswith("event:") or text.startswith("data:"):
            last_obj = {}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    payload_str = line[5:].strip()
                    if not payload_str or payload_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(payload_str)
                        if isinstance(obj, dict):
                            last_obj = obj
                            if "result" in obj or "error" in obj:
                                return obj
                    except json.JSONDecodeError:
                        continue
            return last_obj

        try:
            return resp.json()
        except Exception:
            raise RuntimeError(f"Réponse MCP MoonX non-JSON (HTTP {resp.status_code}) : {text[:300]}")

    def rpc(self, method: str, params: dict = None, is_notification: bool = False) -> dict:
        if is_notification:
            payload = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                payload["params"] = params
        else:
            self.req_id += 1
            payload = {"jsonrpc": "2.0", "id": self.req_id, "method": method, "params": params or {}}

        resp = self.session.post(self.url, headers=self._headers(), json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"Erreur HTTP {resp.status_code} sur MoonX MCP ({method}) : {resp.text[:300]}")
        if is_notification:
            return {}
        data = self._parse_response(resp)
        if "error" in data and data["error"]:
            raise RuntimeError(f"Erreur RPC MoonX ({method}) : {data['error']}")
        return data.get("result", data)

    def connect(self):
        init_res = self.rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "crypto-v4-moonx-bot", "version": "1.0.0"},
            },
        )
        try:
            self.rpc("notifications/initialized", is_notification=True)
        except Exception:
            pass

        tools_res = self.rpc("tools/list", {})
        tools_list = tools_res.get("tools", []) if isinstance(tools_res, dict) else []
        for t in tools_list:
            name = t.get("name")
            if name:
                self.tools_schema[name] = t.get("inputSchema", {}) or {}
        print(f"[MoonX MCP] Connecté ({len(self.tools_schema)} outils disponibles : {', '.join(self.tools_schema.keys())})", flush=True)
        return init_res

    def call_tool(self, name: str, arguments: dict = None):
        res = self.rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if isinstance(res, dict) and "content" in res and isinstance(res["content"], list):
            texts = []
            for item in res["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            joined = "\n".join(texts).strip()
            if joined:
                try:
                    return json.loads(joined)
                except json.JSONDecodeError:
                    return {"raw_text": joined, "isError": res.get("isError", False)}
        return res

    def map_and_call(self, tool_name: str, canonical_args: dict):
        """
        Adapte automatiquement les clés de `canonical_args` au schéma `inputSchema` réel de l'outil MoonX.
        """
        schema = self.tools_schema.get(tool_name, {})
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if not props:
            return self.call_tool(tool_name, canonical_args)

        aliases = {
            "symbol": ["symbol", "pair", "instrument", "asset", "market", "ticker", "coin"],
            "side": ["side", "direction", "positionSide", "orderSide", "type"],
            "margin": ["margin", "marginUsdt", "margin_usdt", "amount", "collateral", "marginAmount", "sizeUsdt"],
            "quantity": ["quantity", "qty", "size", "amountCoins", "contracts"],
            "leverage": ["leverage", "lev", "multiplier"],
            "marginMode": ["marginMode", "margin_mode", "mode", "marginType"],
            "price": ["price", "limitPrice", "limit_price", "orderPrice", "entryPrice"],
            "takeProfitGainPercent": ["takeProfitGainPercent", "tpGainPercent", "take_profit_gain_percent", "tpRoiPercent"],
            "stopLossLossPercent": ["stopLossLossPercent", "slLossPercent", "stop_loss_loss_percent", "slRoiPercent"],
            "takeProfitPrice": ["takeProfitPrice", "tpPrice", "take_profit_price", "takeProfit", "tp"],
            "stopLossPrice": ["stopLossPrice", "slPrice", "stop_loss_price", "stopLoss", "sl"],
            "percentage": ["percentage", "closePercent", "percent", "qtyPercent"],
            "orderId": ["orderId", "order_id", "id"],
            "positionId": ["positionId", "position_id", "id", "posId"],
        }

        mapped = {}
        for canon_k, val in canonical_args.items():
            if val is None:
                continue
            candidates = aliases.get(canon_k, [canon_k])
            matched_key = None
            for cand in candidates:
                if cand in props:
                    matched_key = cand
                    break
            if matched_key is None:
                # Recherche insensible à la casse
                lower_map = {pk.lower(): pk for pk in props.keys()}
                for cand in candidates:
                    if cand.lower() in lower_map:
                        matched_key = lower_map[cand.lower()]
                        break
            if matched_key is not None:
                # Ajuster le format du symbole selon la description du schéma si besoin
                p_info = props.get(matched_key, {})
                desc = (str(p_info.get("description", "")) + " " + str(p_info.get("examples", ""))).upper()
                if canon_k == "symbol" and isinstance(val, str):
                    base_coin = val.upper().replace("USDT", "").replace("USD", "").replace("/", "").replace("-", "").replace("_", "")
                    if "BTC-USDT" in desc or "BTC-USD" in desc:
                        val = f"{base_coin}-USDT"
                    elif "BTC/USDT" in desc:
                        val = f"{base_coin}/USDT"
                    elif "BTCUSDT" in desc:
                        val = f"{base_coin}USDT"
                    elif "BTC" in desc and "USDT" not in desc:
                        val = base_coin
                mapped[matched_key] = val

        # Si aucun mapping n'a matché (schéma dynamique), passer les clés canoniques
        if not mapped:
            mapped = {k: v for k, v in canonical_args.items() if v is not None}
        else:
            # Si le schéma de MoonX prend déjà la marge (margin / marginUsdt / amount) et que quantity n'est pas dans required,
            # éviter d'envoyer à la fois margin et quantity pour ne pas créer de conflit de taille.
            req_fields = set(schema.get("required", []) if isinstance(schema, dict) else [])
            margin_keys = {"margin", "marginUsdt", "margin_usdt", "amount", "collateral", "marginAmount", "sizeUsdt"}
            qty_keys = {"quantity", "qty", "size", "amountCoins", "contracts"}
            has_margin_key = any(mk in mapped for mk in margin_keys)
            for qk in list(qty_keys):
                if has_margin_key and qk in mapped and qk not in req_fields:
                    mapped.pop(qk, None)

        return self.call_tool(tool_name, mapped)


def _to_float(val, default: float = -999999.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip().upper()
        for tok in ("USDT", "USDC", "USD", "$", "€", "%", ","):
            s = s.replace(tok, "")
        s = s.strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            return default
    return default


def _extract_list(obj) -> list:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in (
            "positions", "orders", "tpSlOrders", "tp_sl_orders", "tpslOrders",
            "trades", "history", "wallets", "accounts", "balances", "assets",
            "data", "items", "result", "rows", "list",
        ):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # Si un sous-dictionnaire contient la liste (ex: {"data": {"positions": [...]}})
        for v in obj.values():
            if isinstance(v, dict):
                sub = _extract_list(v)
                if sub:
                    return sub
    return []


def _norm_base_coin(sym: str) -> str:
    s = str(sym or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    for suf in ("PERP", "USDT", "USDC", "USD"):
        if s.endswith(suf) and len(s) > len(suf):
            s = s[:-len(suf)]
    return s


def _wallet_priority(label_or_obj) -> int:
    s = (label_or_obj if isinstance(label_or_obj, str) else json.dumps(label_or_obj, ensure_ascii=False)).lower()
    if any(w in s for w in ("spot", "forex", "fx", "earn", "funding", "copy")):
        if not any(w in s for w in ("future", "perp", "contract", "deriv")):
            return 2
    if any(w in s for w in ("future", "perp", "contract", "deriv")):
        return 0
    return 1


def _parse_float(d, keys: tuple, default: float = 0.0, require_positive: bool = False) -> float:
    if isinstance(d, list):
        sorted_items = sorted(
            [x for x in d if isinstance(x, (dict, list))],
            key=_wallet_priority,
        )
        for item in sorted_items:
            res = _parse_float(item, keys, default=-999999.0, require_positive=require_positive)
            if res != -999999.0 and (not require_positive or res > 0):
                return res
        return default

    if not isinstance(d, dict):
        return default

    # Si ce dictionnaire a des sous-sections explicites (ex: spot, futures, forex), chercher d'abord dans futures !
    sub_keys_lower = {str(k).lower() for k in d.keys()}
    has_wallet_sections = any(w in sub_keys_lower for w in ("futures", "future", "perp", "perpetual", "contract", "derivatives"))

    if not has_wallet_sections:
        for k in keys:
            if k in d and d[k] is not None and not isinstance(d[k], (dict, list)):
                v = _to_float(d[k], default=-999999.0)
                if v != -999999.0 and (not require_positive or v > 0):
                    return v

        lower_d = {str(k).lower().replace("_", ""): v for k, v in d.items() if not isinstance(v, (dict, list))}
        for k in keys:
            nk = k.lower().replace("_", "")
            if nk in lower_d and lower_d[nk] is not None:
                v = _to_float(lower_d[nk], default=-999999.0)
                if v != -999999.0 and (not require_positive or v > 0):
                    return v

    # Parcourir les sous-dictionnaires/listes en priorisant "futures" et en ignorant "spot"/"forex" à 0
    sub_items = [(k, v) for k, v in d.items() if isinstance(v, (dict, list))]
    sub_items.sort(key=lambda kv: _wallet_priority(str(kv[0]) + " " + json.dumps(kv[1], ensure_ascii=False)[:200]))
    for _, sub_v in sub_items:
        res = _parse_float(sub_v, keys, default=-999999.0, require_positive=require_positive)
        if res != -999999.0 and (not require_positive or res > 0):
            return res

    # Si has_wallet_sections était True mais rien trouvé dans les sous-sections, tenter au niveau racine
    if has_wallet_sections:
        for k in keys:
            if k in d and d[k] is not None and not isinstance(d[k], (dict, list)):
                v = _to_float(d[k], default=-999999.0)
                if v != -999999.0 and (not require_positive or v > 0):
                    return v

    return default


def parse_account_balance(
    overview,
    positions_list: list = None,
    orders_list: list = None,
    fallback_capital: float = 495.0,
) -> tuple[float, float, float]:
    """
    Extrait `(realized_capital_usdt, free_margin_usdt, used_margin_usdt)` depuis `get_account_overview`.
    Le `realized_capital_usdt` correspond au capital du compte AVANT le PnL latent des positions en cours
    (ex: 495.xx USDT = disponible + marge engagée, et non l'equity flottante avec PnL latent).
    """
    import re

    wallet_bal = _parse_float(
        overview,
        (
            "walletBalance", "futuresBalance", "accountBalance", "cashBalance",
            "realizedBalance", "totalWalletBalance", "usdtBalance", "totalBalance",
            "marginBalance", "balance", "total", "collateral",
        ),
        default=0.0,
        require_positive=True,
    )
    free_m = _parse_float(
        overview,
        (
            "availableBalance", "availableMargin", "freeMargin", "freeBalance",
            "available", "free", "transferable", "withdrawable", "canUse", "availableUsdt",
            "crossAvailable", "marginAvailable",
        ),
        default=0.0,
        require_positive=True,
    )
    used_m = _parse_float(
        overview,
        (
            "usedMargin", "lockedMargin", "positionMargin", "orderMargin",
            "initialMargin", "blockedMargin", "occupiedMargin", "isolatedMargin",
            "frozenMargin", "frozenBalance", "frozen", "locked", "marginUsed",
        ),
        default=0.0,
        require_positive=True,
    )
    total_eq = _parse_float(
        overview,
        ("totalEquity", "equity", "futuresEquity", "totalMarginBalance", "netAssetValue", "nav", "totalUsdt", "accountEquity"),
        default=0.0,
        require_positive=True,
    )
    un_pnl = _parse_float(
        overview,
        ("unrealizedPnl", "unRealizedProfit", "unrealizedProfit", "upl", "floatingPnl", "openPnl", "crossUnPnl", "pnl"),
        default=-999999.0,
        require_positive=False,
    )

    # Si get_account_overview a renvoyé du texte formaté (raw_text), extraire tous les montants par regex
    if isinstance(overview, dict) and "raw_text" in overview and wallet_bal <= 0 and free_m <= 0:
        txt = str(overview["raw_text"])
        m_bal = re.search(r"(?:wallet\s*balance|futures\s*balance|total\s*balance|balance|solde|capital)[^0-9\n]{0,25}([0-9]+(?:\.[0-9]+)?)", txt, re.I)
        m_av = re.search(r"(?:available|free|disponible|libre)[^0-9\n]{0,25}([0-9]+(?:\.[0-9]+)?)", txt, re.I)
        m_lk = re.search(r"(?:used|locked|position|blocked|occupied|engag)[^0-9\n]{0,25}([0-9]+(?:\.[0-9]+)?)", txt, re.I)
        if m_bal:
            wallet_bal = float(m_bal.group(1))
        if m_av:
            free_m = float(m_av.group(1))
        if m_lk:
            used_m = float(m_lk.group(1))

    # Additionner la marge engagée dans les positions et ordres ouverts si used_m non fourni dans overview
    pos_margin_sum = 0.0
    for p_it in (positions_list or []):
        if isinstance(p_it, dict):
            m_val = _parse_float(
                p_it,
                ("margin", "initialMargin", "isolatedMargin", "positionMargin", "usedMargin", "collateral", "marginUsdt"),
                default=0.0,
                require_positive=True,
            )
            if m_val > 0:
                pos_margin_sum += m_val
    for o_it in (orders_list or []):
        if isinstance(o_it, dict):
            m_val = _parse_float(
                o_it,
                ("margin", "initialMargin", "orderMargin", "lockedMargin", "marginUsdt"),
                default=0.0,
                require_positive=True,
            )
            if m_val > 0:
                pos_margin_sum += m_val
    if used_m <= 0 and pos_margin_sum > 0:
        used_m = pos_margin_sum

    # Calculer le capital réalisé avec les centimes exacts :
    # Attention : si MoonX appelle "balance" le solde disponible (wallet_bal == free_m) alors qu'il y a used_m > 0,
    # le vrai capital réalisé avant ouverture des positions est free_m + used_m !
    realized_cap = 0.0
    if free_m > 0 and used_m > 0:
        if wallet_bal > 0 and abs(wallet_bal - free_m) > 1.0:
            realized_cap = wallet_bal
        else:
            realized_cap = free_m + used_m
    elif wallet_bal > 0:
        realized_cap = wallet_bal + (used_m if free_m <= 0 else 0.0)
    elif total_eq > 0 and un_pnl != -999999.0:
        realized_cap = total_eq - un_pnl
    elif total_eq > 0:
        realized_cap = total_eq
    elif free_m > 0:
        realized_cap = free_m

    if free_m <= 0 and realized_cap > 0:
        free_m = max(0.0, realized_cap - used_m)
    if realized_cap <= 0:
        realized_cap = fallback_capital
    if free_m <= 0:
        free_m = max(0.0, realized_cap - used_m) if realized_cap > used_m else realized_cap

    return float(realized_cap), float(free_m), float(used_m)


def load_bot_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"notified_trade_ids": [], "last_sl_by_coin": {}, "active_pois": {}}


def save_bot_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def fetch_closed_15m_4h(coin: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Télécharge les dernières bougies 5m (10 derniers jours), construit les bougies 15m et 4h *clôturées*,
    et retourne `(df_15m, df_4h, last_price)`.
    """
    binance_sym = dc.ftmo_to_binance(f"{coin}USD")
    start_dt = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df_5m, _ = dc.load(binance_sym, "5m", start_dt, source="auto", use_cache=False)
    if df_5m is None or len(df_5m) < 100:
        raise RuntimeError(f"Données 5m insuffisantes pour {coin}")

    df_5m = df_5m.sort_values("time").reset_index(drop=True)
    last_px = float(df_5m["close"].iloc[-1])

    now_utc = pd.Timestamp.now(tz="UTC")
    d = df_5m.set_index("time")

    r15 = d.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    # Ne garder que les bougies 15m complètement clôturées
    r15 = r15[r15.index + pd.Timedelta(minutes=15) <= now_utc].reset_index()

    r4h = d.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    # Ne garder que les bougies 4h complètement clôturées
    r4h = r4h[r4h.index + pd.Timedelta(hours=4) <= now_utc].reset_index()

    return r15, r4h, last_px


def find_cancel_tool(client: MoonXMCPClient) -> str:
    for cand in (
        "cancel_futures_order",
        "cancel_futures_limit_order",
        "cancel_order",
        "close_futures_order",
        "delete_futures_order",
    ):
        if cand in client.tools_schema:
            return cand
    for name in client.tools_schema:
        if "cancel" in name.lower() and "order" in name.lower():
            return name
    return "cancel_futures_order"


def main():
    ap = argparse.ArgumentParser(description="Bot Live CRYPTO V4 Optimisée sur MoonX (MCP)")
    ap.add_argument("--symbols", default=",".join(scv4.CRYPTO_V4_SYMBOLS), help="Cryptos actives (défaut: BTC,ETH,AVAX,INJ)")
    ap.add_argument("--risk_pct", type=float, default=1.5, help="Risque par SL en %% du capital réalisé à l'instant t (défaut: 1.5%%)")
    ap.add_argument("--margin_mode", default="cross", choices=["cross", "isolated"], help="Mode de marge MoonX : cross (croisé) ou isolated (isolé)")
    ap.add_argument("--fallback_capital", type=float, default=495.0, help="Capital de secours si non lisible (défaut: 495 USDT)")
    ap.add_argument("--params", default="", help="Réglages supplémentaires k=v;k2=v2")
    ap.add_argument("--dry_run", default="non", choices=["oui", "non"], help="oui = mode test sans envoyer d'ordres réels")
    ap.add_argument("--inspect_tools", action="store_true", help="Affiche les schémas exacts des outils MoonX MCP et quitte")
    ap.add_argument("--state_file", default=STATE_PATH_DEFAULT)
    a = ap.parse_args()

    is_dry = (a.dry_run.lower() == "oui")
    margin_mode = a.margin_mode.lower()
    p = scv4.parse_params(a.params)
    lev = float(p.get("leverage", 30.0))
    tp_roi_pct = float(p.get("tp_roi_pct", 300.0))
    sl_roi_pct = float(p.get("sl_roi_pct", 50.0))
    reprice_tol = float(p.get("reprice_tol_pct", 0.1)) / 100.0

    coins = []
    for s in a.symbols.split(","):
        c = _norm_base_coin(s)
        if c and c not in coins:
            coins.append(c)

    mcp_secret = os.getenv("MOONX_MCP_URL") or os.getenv("MOONX_TOKEN") or ""
    client = MoonXMCPClient(mcp_secret)
    client.connect()

    if a.inspect_tools:
        print(json.dumps(client.tools_schema, indent=2, ensure_ascii=False))
        return

    state = load_bot_state(a.state_file)
    now_ts = time.time()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # =========================================================================
    # ÉTAPE 1 — LIRE L'ÉTAT DU COMPTE MOONX & SIZING EN % DU CAPITAL À L'INSTANT T
    # =========================================================================
    overview = client.call_tool("get_account_overview", {})
    positions_raw = client.call_tool("list_futures_positions", {})
    orders_raw = client.call_tool("list_futures_orders", {})
    tpsl_raw = client.call_tool("list_futures_tp_sl_orders", {})
    history_raw = client.call_tool("get_futures_trade_history", {})

    print(f"[MoonX Overview brut] {json.dumps(overview, ensure_ascii=False)}", flush=True)

    positions_list = _extract_list(positions_raw)
    orders_list = _extract_list(orders_raw)
    tpsl_list = _extract_list(tpsl_raw)
    history_list = _extract_list(history_raw)

    total_eq, free_margin, used_margin = parse_account_balance(
        overview,
        positions_list=positions_list,
        orders_list=orders_list,
        fallback_capital=a.fallback_capital,
    )
    # Sizing dynamique sur le capital réalisé du compte (hors PnL latent, ex: 495 USDT = 385 dispo + 110 engagé) :
    # 1R = Capital_réalisé_t * (risk_pct / 100) ; Marge à x30 (SL = -50% ROI) = 1R / 0.50 = 2 * 1R
    if a.risk_pct > 0:
        one_r_usdt = total_eq * (float(a.risk_pct) / 100.0)
        trade_margin_usdt = round(one_r_usdt / (sl_roi_pct / 100.0), 2)
    else:
        trade_margin_usdt = float(p.get("margin_usdt", 200.0))
        one_r_usdt = trade_margin_usdt * (sl_roi_pct / 100.0)

    print(
        f"[{now_str}] Capital réalisé MoonX : {total_eq:.2f} USDT (Dispo : {free_margin:.2f} USDT | Engagé : {used_margin:.2f} USDT) "
        f"-> 1R ({a.risk_pct:g}%) = {one_r_usdt:.2f} USDT | Marge/trade = {trade_margin_usdt:.2f} USDT (x{lev:g} {margin_mode})",
        flush=True,
    )

    # Indexer par coin (BTC, ETH, AVAX, INJ)
    open_pos_by_coin = {}
    for pos_item in positions_list:
        if not isinstance(pos_item, dict):
            continue
        c = _norm_base_coin(pos_item.get("symbol") or pos_item.get("pair") or pos_item.get("instrument") or pos_item.get("asset"))
        size_val = _parse_float(pos_item, ("size", "quantity", "qty", "amount", "contracts", "margin", "positionAmt"), default=1.0)
        if c and abs(size_val) > 0:
            open_pos_by_coin[c] = pos_item

    open_orders_by_coin = {}
    for ord_item in orders_list:
        if not isinstance(ord_item, dict):
            continue
        c = _norm_base_coin(ord_item.get("symbol") or ord_item.get("pair") or ord_item.get("instrument") or ord_item.get("asset"))
        if c:
            open_orders_by_coin.setdefault(c, []).append(ord_item)

    tpsl_by_coin = {}
    for t_item in tpsl_list:
        if not isinstance(t_item, dict):
            continue
        c = _norm_base_coin(t_item.get("symbol") or t_item.get("pair") or t_item.get("instrument") or t_item.get("asset"))
        if c:
            tpsl_by_coin.setdefault(c, []).append(t_item)

    # =========================================================================
    # NOTIFICATIONS CLÔTURES & DÉTECTION DES SL RÉCENTS (POUR COOLDOWN 2H & POI)
    # =========================================================================
    notified_ids = set(state.get("notified_trade_ids", []))
    last_sl_by_coin = state.get("last_sl_by_coin", {})
    active_pois = state.get("active_pois", {})

    for tr in history_list:
        if not isinstance(tr, dict):
            continue
        tid = str(tr.get("id") or tr.get("tradeId") or tr.get("orderId") or tr.get("closeTime") or tr.get("timestamp") or "")
        if not tid or tid in notified_ids:
            continue
        c = _norm_base_coin(tr.get("symbol") or tr.get("pair") or tr.get("instrument") or "")
        pnl = _parse_float(tr, ("realizedPnl", "pnl", "profit", "netPnl", "pnlUsdt"), default=0.0)
        roi = _parse_float(tr, ("roi", "roiPercent", "roe", "returnPercent"), default=0.0)
        en_px = _parse_float(tr, ("entryPrice", "openPrice", "avgEntryPrice"), default=0.0)
        ex_px = _parse_float(tr, ("exitPrice", "closePrice", "price"), default=0.0)
        side_str = str(tr.get("side") or tr.get("direction") or tr.get("positionSide") or "").upper()

        if pnl > 0:
            tg_send(
                f"✅ Clôture Gagnante MoonX — {c}\n"
                f"• Sens : {side_str}\n"
                f"• Entrée : {en_px:g} | Sortie : {ex_px:g}\n"
                f"• ROI : {roi:+.1f}% | PnL : {pnl:+.2f} USDT\n"
                f"• Nouveau solde : {total_eq:.2f} USDT"
            )
        elif pnl < -0.2 * one_r_usdt or roi <= -25.0:
            # Enregistrer le SL pour le Cooldown de 2h et le blocage du POI cassé
            prev_poi = float(active_pois.get(c, -1.0))
            last_sl_by_coin[c] = {"ts": now_ts, "poi": prev_poi}

        notified_ids.add(tid)

    state["notified_trade_ids"] = list(notified_ids)[-300:]
    state["last_sl_by_coin"] = last_sl_by_coin

    actions_log = []
    cancel_tool = find_cancel_tool(client)

    # =========================================================================
    # PARCOURS DES INSTRUMENTS ACTIFS (BTC, ETH, AVAX, INJ)
    # =========================================================================
    for coin in coins:
        try:
            df_15m, df_4h, mkt_px = fetch_closed_15m_4h(coin)
        except Exception as e:
            msg = f"⚠️ {coin}: impossible de lire les bougies 15m/4h ({e})"
            print(msg, flush=True)
            actions_log.append(msg)
            continue

        # ---------------------------------------------------------------------
        # ÉTAPE 2 — GÉRER LA POSITION OUVERTE (SI PRÉSENTE SUR CE SYMBOLE)
        # ---------------------------------------------------------------------
        if coin in open_pos_by_coin:
            pos = open_pos_by_coin[coin]
            # Règle PDF : jamais position + ordre au repos en même temps sur le même instrument
            if coin in open_orders_by_coin:
                for resting in open_orders_by_coin[coin]:
                    oid = resting.get("orderId") or resting.get("id")
                    if oid and not is_dry:
                        try:
                            client.map_and_call(cancel_tool, {"symbol": coin, "orderId": oid})
                        except Exception as ce:
                            print(f"[{coin}] Erreur annulation ordre doublon {oid}: {ce}", flush=True)

            raw_side = str(pos.get("side") or pos.get("direction") or pos.get("positionSide") or "LONG").upper()
            side_int = 1 if ("LONG" in raw_side or "BUY" in raw_side) else -1
            entry_px = _parse_float(pos, ("entryPrice", "openPrice", "avgEntryPrice", "avgPrice", "price"), default=mkt_px)
            cur_sl_px = _parse_float(pos, ("stopLossPrice", "slPrice", "stopLoss", "sl"), default=0.0)
            cur_tp_px = _parse_float(pos, ("takeProfitPrice", "tpPrice", "takeProfit", "tp"), default=0.0)
            pos_id = pos.get("positionId") or pos.get("id")

            # Chercher aussi dans list_futures_tp_sl_orders si non présent directement sur l'objet position
            if (cur_sl_px <= 0 or cur_tp_px <= 0) and coin in tpsl_by_coin:
                for t_item in tpsl_by_coin[coin]:
                    if cur_sl_px <= 0:
                        cur_sl_px = _parse_float(t_item, ("stopLossPrice", "slPrice", "stopLoss", "triggerPrice"), default=0.0)
                    if cur_tp_px <= 0:
                        cur_tp_px = _parse_float(t_item, ("takeProfitPrice", "tpPrice", "takeProfit", "triggerPrice"), default=0.0)

            trail = scv4.compute_live_trailing_sl(side_int, entry_px, mkt_px, current_sl=cur_sl_px, p=p)

            # 2.a) Si la position vient d'être remplie (ex: ordre limite exécuté) et n'a pas encore de TP/SL : attacher TP +300% / SL -50%
            has_tpsl = (cur_sl_px > 0 and cur_tp_px > 0) or (coin in tpsl_by_coin and len(tpsl_by_coin[coin]) > 0)
            if not has_tpsl and trail["tier"] == 0:
                msg = f"🛡️ {coin} ({'LONG' if side_int==1 else 'SHORT'} @ {entry_px:g}) : Attache TP +300% ({trail['tp_price']:g}) & SL -50% ({trail['target_sl_price']:g})"
                print(msg, flush=True)
                actions_log.append(msg)
                if not is_dry:
                    client.map_and_call(
                        "set_futures_tp_sl",
                        {
                            "symbol": coin,
                            "positionId": pos_id,
                            "takeProfitGainPercent": tp_roi_pct,
                            "stopLossLossPercent": sl_roi_pct,
                            "takeProfitPrice": trail["tp_price"],
                            "stopLossPrice": trail["target_sl_price"],
                            "percentage": 100,
                        },
                    )
            # 2.b) Si un palier de Trailing SL supérieur est atteint (+50% -> BE, +100% -> +50%, +150% -> +100%, +200% -> +150%)
            elif trail["should_update"]:
                msg = (
                    f"🔒 {coin} ({'LONG' if side_int==1 else 'SHORT'} | ROI {trail['roi_cur']:+.1f}%) : "
                    f"Trailing SL Palier {trail['tier']} -> SL remonté à {trail['target_sl_roi']:+.0f}% ROI ({trail['target_sl_price']:g})"
                )
                print(msg, flush=True)
                actions_log.append(msg)
                if not is_dry:
                    # stopLossLossPercent négatif ou stopLossPrice direct au prix verrouillé
                    client.map_and_call(
                        "set_futures_tp_sl",
                        {
                            "symbol": coin,
                            "positionId": pos_id,
                            "takeProfitGainPercent": tp_roi_pct,
                            "takeProfitPrice": trail["tp_price"],
                            "stopLossPrice": trail["target_sl_price"],
                            "percentage": 100,
                        },
                    )
            else:
                actions_log.append(
                    f"⏳ {coin} : Position {'LONG' if side_int==1 else 'SHORT'} active @ {entry_px:g} "
                    f"(Prix {mkt_px:g} | ROI {trail['roi_cur']:+.1f}% | Palier {trail['tier']})"
                )
            continue

        # ---------------------------------------------------------------------
        # ÉTAPE 3 — DÉTECTION & RÉ-ANALYSE DES ORDRES AU REPOS
        # ---------------------------------------------------------------------
        sl_info = last_sl_by_coin.get(coin, {})
        last_sl_ts = float(sl_info.get("ts", 0.0))
        last_sl_poi = float(sl_info.get("poi", -1.0))
        hours_since_sl = (now_ts - last_sl_ts) / 3600.0 if last_sl_ts > 0 else 999.0

        sig = scv4.evaluate_live_signal(
            df_15m,
            df_4h,
            px=mkt_px,
            p=p,
            hours_since_last_sl=hours_since_sl,
            last_sl_poi=last_sl_poi,
        )

        resting_orders = open_orders_by_coin.get(coin, [])
        kept_resting = False

        if resting_orders:
            for ro in resting_orders:
                oid = ro.get("orderId") or ro.get("id")
                ro_side_raw = str(ro.get("side") or ro.get("direction") or "").lower()
                ro_side = "long" if ("long" in ro_side_raw or "buy" in ro_side_raw) else "short"
                ro_px = _parse_float(ro, ("price", "limitPrice", "orderPrice"), default=0.0)

                # Garder l'ordre au repos uniquement si la thèse est toujours valide, même direction, et écart <= 0.1%
                if (
                    sig["action"] == "LIMIT"
                    and sig.get("side") == ro_side
                    and ro_px > 0
                    and abs(sig["entry_price"] - ro_px) / ro_px <= reprice_tol
                ):
                    kept_resting = True
                    actions_log.append(f"📌 {coin} : Ordre LIMITE {ro_side.upper()} conservé @ {ro_px:g} (POI {sig['poi']:g})")
                else:
                    msg = f"🔄 {coin} : Annulation ordre au repos @ {ro_px:g} ({sig.get('reason', 'ajustement POI > 0.1%')})"
                    print(msg, flush=True)
                    actions_log.append(msg)
                    if oid and not is_dry:
                        try:
                            client.map_and_call(cancel_tool, {"symbol": coin, "orderId": oid})
                        except Exception as ce:
                            print(f"[{coin}] Erreur annulation ordre {oid}: {ce}", flush=True)

        if kept_resting:
            continue

        if sig["action"] == "SKIP":
            print(f"[{coin}] PASS : {sig['reason']}", flush=True)
            continue

        # ---------------------------------------------------------------------
        # ÉTAPE 4 — OUVERTURE (MARCHÉ OU LIMITE) AVEC SIZING DYNAMIQUE À L'INSTANT T
        # ---------------------------------------------------------------------
        if free_margin < trade_margin_usdt:
            msg = f"⚠️ {coin} : Signal {sig['action']} {sig['side'].upper()} ignoré (marge libre {free_margin:.2f} < {trade_margin_usdt:.2f} USDT requis)"
            print(msg, flush=True)
            actions_log.append(msg)
            continue

        side_str = sig["side"]  # "long" ou "short"
        en_px = sig["entry_price"]
        sl_px = sig["sl_price"]
        tp_px = sig["tp_price"]
        qty_coins = round((trade_margin_usdt * lev) / en_px, 6)

        # Vérification stricte du côté gain/perte (Étape 4 du PDF)
        if side_str == "long" and not (tp_px > en_px and sl_px < en_px):
            continue
        if side_str == "short" and not (tp_px < en_px and sl_px > en_px):
            continue

        active_pois[coin] = float(sig["poi"])
        state["active_pois"] = active_pois

        if sig["action"] == "MARKET":
            msg = (
                f"🚀 {coin} : OUVERTURE MARCHÉ {side_str.upper()} @ ~{en_px:g} | "
                f"Marge {trade_margin_usdt:.2f} USDT (1R={one_r_usdt:.2f} USDT, {a.risk_pct:g}% de {total_eq:.0f}$) | "
                f"TP {tp_px:g} (+300%) | SL {sl_px:g} (-50%)"
            )
            print(msg, flush=True)
            actions_log.append(msg)
            if not is_dry:
                open_res = client.map_and_call(
                    "open_futures_position",
                    {
                        "symbol": coin,
                        "side": side_str,
                        "margin": trade_margin_usdt,
                        "quantity": qty_coins,
                        "leverage": int(lev),
                        "marginMode": margin_mode,
                        "takeProfitGainPercent": tp_roi_pct,
                        "stopLossLossPercent": sl_roi_pct,
                        "takeProfitPrice": tp_px,
                        "stopLossPrice": sl_px,
                    },
                )
                new_pos_id = open_res.get("positionId") or open_res.get("id") if isinstance(open_res, dict) else None
                try:
                    client.map_and_call(
                        "set_futures_tp_sl",
                        {
                            "symbol": coin,
                            "positionId": new_pos_id,
                            "takeProfitGainPercent": tp_roi_pct,
                            "stopLossLossPercent": sl_roi_pct,
                            "takeProfitPrice": tp_px,
                            "stopLossPrice": sl_px,
                            "percentage": 100,
                        },
                    )
                except Exception as te:
                    print(f"[{coin}] Note set_futures_tp_sl après ouverture : {te}", flush=True)
                free_margin = max(0.0, free_margin - trade_margin_usdt)
            tg_send(f"{'[SIMULATION] ' if is_dry else ''}{msg}")

        elif sig["action"] == "LIMIT":
            msg = (
                f"🎯 {coin} : ORDRE LIMITE {side_str.upper()} placé @ {en_px:g} (POI {sig['poi']:g}) | "
                f"Marge {trade_margin_usdt:.2f} USDT (1R={one_r_usdt:.2f} USDT, {a.risk_pct:g}% de {total_eq:.0f}$) | "
                f"TP prévu {tp_px:g} (+300%) | SL prévu {sl_px:g} (-50%)"
            )
            print(msg, flush=True)
            actions_log.append(msg)
            if not is_dry:
                client.map_and_call(
                    "open_futures_limit_order",
                    {
                        "symbol": coin,
                        "side": side_str,
                        "price": en_px,
                        "margin": trade_margin_usdt,
                        "quantity": qty_coins,
                        "leverage": int(lev),
                        "marginMode": margin_mode,
                        "takeProfitGainPercent": tp_roi_pct,
                        "stopLossLossPercent": sl_roi_pct,
                        "takeProfitPrice": tp_px,
                        "stopLossPrice": sl_px,
                    },
                )
                free_margin = max(0.0, free_margin - trade_margin_usdt)

    save_bot_state(a.state_file, state)

    raw_ov_preview = f"\n🔍 Brut MoonX : {json.dumps(overview, ensure_ascii=False)[:450]}" if is_dry else ""
    summary = (
        f"🤖 Run Live CRYPTO V4 ({now_str}){' [DRY-RUN]' if is_dry else ''}\n"
        f"• Capital réalisé MoonX : {total_eq:.2f} USDT (Dispo : {free_margin:.2f} USDT | Engagé : {used_margin:.2f} USDT)\n"
        f"• Sizing dynamique ({a.risk_pct:g}%) : 1R = {one_r_usdt:.2f} USDT | Marge = {trade_margin_usdt:.2f} USDT (x{lev:g} {margin_mode})\n"
        + ("\n".join(actions_log) if actions_log else "• Aucune nouvelle action sur ce créneau 30m.")
        + raw_ov_preview
    )
    print("\n" + summary, flush=True)
    if is_dry:
        tg_send(summary)


if __name__ == "__main__":
    main()

