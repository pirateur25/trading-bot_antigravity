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
                    parsed = json.loads(joined)
                    if res.get("isError") and isinstance(parsed, dict):
                        parsed["isError"] = True
                    return parsed
                except json.JSONDecodeError:
                    return {"raw_text": joined, "isError": res.get("isError", False)}
        return res

    @staticmethod
    def is_error(res) -> bool:
        if not isinstance(res, dict):
            return False
        if res.get("isError"):
            return True
        if res.get("error") or res.get("success") is False:
            return True
        raw = str(res.get("raw_text", "")).lower()
        return any(w in raw for w in ("error", "invalid", "failed", "not found", "unauthorized"))

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
            "takeProfitGainPercent": ["takeProfitGainPercent", "tpGainPercent", "take_profit_gain_percent", "tpRoiPercent", "tpPercent", "takeProfitPercent"],
            "stopLossLossPercent": ["stopLossLossPercent", "slLossPercent", "stop_loss_loss_percent", "slRoiPercent", "slPercent", "stopLossPercent"],
            "takeProfitPrice": ["takeProfitPrice", "tpPrice", "take_profit_price", "takeProfit", "tp", "tpTriggerPrice", "takeProfitTriggerPrice"],
            "stopLossPrice": ["stopLossPrice", "slPrice", "stop_loss_price", "stopLoss", "sl", "slTriggerPrice", "stopLossTriggerPrice"],
            "percentage": ["percentage", "closePercent", "percent", "qtyPercent"],
            "orderId": ["orderId", "order_id", "ordId", "orderID", "id", "_id"],
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
            if matched_key is None:
                print(f"[MCP {tool_name}] ⚠️ paramètre '{canon_k}' absent du schéma MoonX -> ignoré (schéma : {list(props.keys())})", flush=True)
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
            # MoonX précise "Use takeProfitPrice OR takeProfitGainPercent" : éviter d'envoyer les deux en doublon
            tp_px_keys = {"takeProfitPrice", "tpPrice", "take_profit_price", "takeProfit", "tp", "tpTriggerPrice", "takeProfitTriggerPrice"}
            tp_pct_keys = {"takeProfitGainPercent", "tpGainPercent", "take_profit_gain_percent", "tpRoiPercent", "tpPercent", "takeProfitPercent"}
            sl_px_keys = {"stopLossPrice", "slPrice", "stop_loss_price", "stopLoss", "sl", "slTriggerPrice", "stopLossTriggerPrice"}
            sl_pct_keys = {"stopLossLossPercent", "slLossPercent", "stop_loss_loss_percent", "slRoiPercent", "slPercent", "stopLossPercent"}
            if any(k in mapped for k in tp_px_keys):
                for pk in list(tp_pct_keys):
                    if pk in mapped and pk not in req_fields:
                        mapped.pop(pk, None)
            if any(k in mapped for k in sl_px_keys):
                for pk in list(sl_pct_keys):
                    if pk in mapped and pk not in req_fields:
                        mapped.pop(pk, None)

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

    # Cas 1 : Structure officielle MoonX Unified Balance (ex: unifiedBalance.total = 495.2226, availableFor.futures = 395.2226, LOCKED_MARGIN = 100)
    if isinstance(overview, dict) and isinstance(overview.get("unifiedBalance"), dict):
        ub = overview["unifiedBalance"]
        fb = overview.get("futuresBalances", {}) if isinstance(overview.get("futuresBalances"), dict) else {}
        av_for = ub.get("availableFor", {}) if isinstance(ub.get("availableFor"), dict) else {}
        u_tot = _to_float(ub.get("total"), default=0.0)
        u_free = _to_float(av_for.get("futures") or av_for.get("spot"), default=0.0)
        u_lock = _to_float(fb.get("LOCKED_MARGIN") or fb.get("lockedMargin"), default=0.0)
        if u_tot > 0:
            if u_lock <= 0 and u_free > 0 and u_tot > u_free:
                u_lock = u_tot - u_free
            if u_free <= 0:
                u_free = max(0.0, u_tot - max(0.0, u_lock))
            return float(u_tot), float(u_free), float(max(0.0, u_lock))

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
    Télécharge en TEMPS RÉEL les dernières bougies 5m (jusqu'à la minute actuelle !) depuis les API live
    (Hyperliquid, KuCoin, Bybit — et NON data.binance.vision qui s'arrête la veille à minuit),
    construit les bougies 15m et 4h *clôturées*, et retourne `(df_15m, df_4h, last_price)`.
    """
    binance_sym = dc.ftmo_to_binance(f"{coin}USD")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (5 * 86400 * 1000)  # 5 derniers jours en temps réel

    df_5m = None
    # Essayer les sources TEMPS RÉEL (qui incluent les bougies d'aujourd'hui jusqu'à la minute en cours)
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


_SL_KEYS = ("stopLossPrice", "slPrice", "stopLoss", "sl", "slTriggerPrice", "stopLossTriggerPrice", "stop_loss_price", "stop_loss")
_TP_KEYS = ("takeProfitPrice", "tpPrice", "takeProfit", "tp", "tpTriggerPrice", "takeProfitTriggerPrice", "take_profit_price", "take_profit")
_SYM_KEYS = ("symbol", "pair", "instrument", "asset", "tokenAddress", "market", "ticker", "coin")
_POS_ID_REF_KEYS = ("positionId", "position_id", "posId", "pos_id", "position")


def _extract_coin(item: dict, pos_id_to_coin: dict = None) -> str:
    """Identifie la crypto (BTC, ETH, ...) d'un objet MoonX via positionId puis via les champs symbol/tokenAddress/etc."""
    if not isinstance(item, dict):
        return ""
    if pos_id_to_coin:
        for pk in _POS_ID_REF_KEYS:
            pv = item.get(pk)
            if isinstance(pv, dict):
                pv = get_position_id(pv)
            if pv not in (None, "", 0) and str(pv) in pos_id_to_coin:
                return pos_id_to_coin[str(pv)]
    for sk in _SYM_KEYS:
        sv = item.get(sk)
        if sv:
            c = _norm_base_coin(sv)
            if c:
                return c
    return ""


def _item_matches_position(item: dict, coin: str, pos_id=None) -> bool:
    """Vérifie qu'un ordre TP/SL ou position appartient bien à (coin, pos_id) et jamais à une autre crypto."""
    if not isinstance(item, dict):
        return False
    st_raw = str(item.get("status") or item.get("state") or "").lower()
    if st_raw in ("cancelled", "canceled", "rejected", "expired", "triggered", "filled", "closed"):
        return False
    if pos_id:
        for pk in _POS_ID_REF_KEYS:
            pv = item.get(pk)
            if isinstance(pv, dict):
                pv = get_position_id(pv)
            if pv not in (None, "", 0):
                return str(pv) == str(pos_id)
    item_c = _extract_coin(item)
    return bool(item_c and item_c == coin)


def get_position_id(pos: dict):
    """Retrouve l'identifiant d'une position MoonX quel que soit le nom du champ (positionId, id, posId, ...)."""
    if not isinstance(pos, dict):
        return None
    for k in ("positionId", "position_id", "posId", "pos_id", "id", "_id", "positionID", "pid"):
        v = pos.get(k)
        if v not in (None, "", 0):
            return str(v)
    for k, v in pos.items():
        if str(k).lower().endswith("id") and v not in (None, "", 0) and not isinstance(v, (dict, list, bool)):
            if "order" in str(k).lower() or "user" in str(k).lower() or "account" in str(k).lower() or "member" in str(k).lower():
                continue
            return str(v)
    return None


def get_order_id(order: dict):
    """Retrouve l'identifiant d'un ordre MoonX (orderId, id, _id, ...)."""
    if not isinstance(order, dict):
        return None
    for k in ("orderId", "order_id", "ordId", "orderID", "id", "_id"):
        v = order.get(k)
        if v not in (None, "", 0):
            return str(v)
    for k, v in order.items():
        kl = str(k).lower()
        if kl.endswith("id") and v not in (None, "", 0) and not isinstance(v, (dict, list, bool)):
            if any(bad in kl for bad in ("position", "pos", "user", "account", "member")):
                continue
            return str(v)
    return None


def apply_tp_sl(client: MoonXMCPClient, coin: str, pos_id, side_int: int, entry_px: float,
                tp_px: float, sl_px: float, sl_roi_signed: float, tp_roi_pct: float, label: str) -> bool:
    """
    Pose / met à jour TP & SL d'une position MoonX, puis VÉRIFIE en relisant l'état réel.
    sl_roi_signed : ROI visé pour le SL (ex: -50 = perte -50%, 0 = BE, +50 = profit verrouillé).
    Envoie TOUJOURS les prix absolus (indispensable pour le trailing) + les pourcentages quand le SL est négatif.
    Retourne True si MoonX affiche bien le nouveau SL, sinon alerte Telegram.
    """
    args = {
        "symbol": coin,
        "positionId": pos_id,
        "takeProfitPrice": tp_px,
        "stopLossPrice": sl_px,
        "takeProfitGainPercent": tp_roi_pct,
        "percentage": 100,
    }
    if sl_roi_signed < 0:
        args["stopLossLossPercent"] = abs(sl_roi_signed)

    schema_props = (client.tools_schema.get("set_futures_tp_sl", {}) or {}).get("properties", {})
    if not pos_id:
        raw_pos = ""
        try:
            for it in _extract_list(client.call_tool("list_futures_positions", {})):
                if isinstance(it, dict) and _norm_base_coin(it.get("symbol") or it.get("pair") or it.get("instrument") or it.get("asset")) == coin:
                    pos_id = get_position_id(it)
                    raw_pos = json.dumps(it, ensure_ascii=False)[:700]
                    break
        except Exception:
            pass
        if not pos_id:
            tg_send(f"❌ {coin} {label} : positionId introuvable. Position brute MoonX : {raw_pos}\nVÉRIFIE MANUELLEMENT TON SL/TP !")
            return False
        args["positionId"] = str(pos_id)
    else:
        args["positionId"] = str(pos_id)

    price_ok = any(k.lower() in ("stoplossprice", "slprice", "stop_loss_price", "stoploss", "sl") for k in schema_props)
    if not price_ok and sl_roi_signed >= 0:
        # Schéma uniquement en pourcentages : on exprime le SL verrouillé en "perte négative"
        args["stopLossLossPercent"] = -sl_roi_signed

    try:
        res = client.map_and_call("set_futures_tp_sl", args)
    except Exception as e:
        tg_send(f"❌ {coin} {label} : set_futures_tp_sl a échoué ({e}). VÉRIFIE MANUELLEMENT TON SL/TP SUR MOONX !")
        print(f"[{coin}] set_futures_tp_sl exception : {e}", flush=True)
        return False

    print(f"[{coin}] set_futures_tp_sl réponse : {json.dumps(res, ensure_ascii=False)[:400]}", flush=True)
    if client.is_error(res):
        tg_send(f"❌ {coin} {label} : MoonX a refusé le TP/SL → {json.dumps(res, ensure_ascii=False)[:300]}\nVÉRIFIE MANUELLEMENT TON SL/TP !")
        return False

    # Vérification : relire les TP/SL réels (avec positionId/symbol si supporté par MoonX)
    try:
        time.sleep(1.0)
        chk = _extract_list(client.map_and_call("list_futures_tp_sl_orders", {"symbol": coin, "positionId": pos_id}))
        if not chk:
            chk = _extract_list(client.call_tool("list_futures_tp_sl_orders", {}))
        chk += _extract_list(client.call_tool("list_futures_positions", {}))
        found_sl = 0.0
        for it in chk:
            if _item_matches_position(it, coin, pos_id):
                v = _parse_float(it, _SL_KEYS, default=0.0)
                if v > 0 and entry_px * 0.2 <= v <= entry_px * 5.0:
                    found_sl = v
        if found_sl > 0 and abs(found_sl - sl_px) / max(sl_px, 1e-9) > 0.002:
            tg_send(f"⚠️ {coin} {label} : SL attendu {sl_px:g} mais MoonX affiche {found_sl:g}. Vérifie sur MoonX.")
            return False
    except Exception as e:
        print(f"[{coin}] vérification TP/SL impossible : {e}", flush=True)
    return True


def find_cancel_tool(client: MoonXMCPClient) -> str:
    for cand in (
        "cancel_futures_order",
        "cancel_futures_limit_order",
        "cancel_limit_order",
        "cancel_order",
        "close_futures_order",
        "delete_futures_order",
        "cancel_futures_limit",
        "close_futures_limit_order",
    ):
        if cand in client.tools_schema:
            return cand
    for name in client.tools_schema:
        nl = name.lower()
        if any(v in nl for v in ("cancel", "delete", "remove")) and ("order" in nl or "limit" in nl):
            if "tp" not in nl and "sl" not in nl:
                return name
    return ""


def cancel_resting_order(client: MoonXMCPClient, cancel_tool: str, coin: str, ro: dict, is_dry: bool) -> bool:
    """Annule un ordre au repos sur MoonX en vérifiant l'ID, le schéma et la réponse."""
    oid = get_order_id(ro)
    if is_dry:
        return True
    if not oid:
        msg = f"⚠️ {coin} : impossible d'annuler l'ordre au repos (ID introuvable dans {json.dumps(ro, ensure_ascii=False)[:250]})"
        print(msg, flush=True)
        tg_send(msg)
        return False
    if not cancel_tool or cancel_tool not in client.tools_schema:
        msg = f"⚠️ {coin} : aucun outil d'annulation d'ordre trouvé sur MoonX (outils : {', '.join(client.tools_schema.keys())})"
        print(msg, flush=True)
        tg_send(msg)
        return False
    try:
        c_schema = client.tools_schema.get(cancel_tool, {}) or {}
        print(
            f"[{coin}] Annulation ordre oid={oid} via {cancel_tool} "
            f"(schéma : {list((c_schema.get('properties') or {}).keys())})",
            flush=True,
        )
        res = client.map_and_call(cancel_tool, {"symbol": coin, "orderId": oid})
        print(f"[{coin}] {cancel_tool} réponse : {json.dumps(res, ensure_ascii=False)[:400]}", flush=True)
        if client.is_error(res):
            tg_send(f"❌ {coin} : échec annulation ordre {oid} → {json.dumps(res, ensure_ascii=False)[:250]}")
            return False
        return True
    except Exception as ce:
        msg = f"❌ {coin} : exception annulation ordre {oid} ({ce})"
        print(msg, flush=True)
        tg_send(msg)
        return False


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
    pos_id_to_coin = {}
    for pos_item in positions_list:
        if not isinstance(pos_item, dict):
            continue
        c = _extract_coin(pos_item)
        size_val = _parse_float(pos_item, ("size", "quantity", "qty", "amount", "contracts", "margin", "positionAmt"), default=1.0)
        if c and abs(size_val) > 0:
            open_pos_by_coin[c] = pos_item
            pid = get_position_id(pos_item)
            if pid:
                pos_id_to_coin[str(pid)] = c

    open_orders_by_coin = {}
    for ord_item in orders_list:
        if not isinstance(ord_item, dict):
            continue
        st_raw = str(ord_item.get("status") or ord_item.get("state") or "").lower()
        if st_raw in ("cancelled", "canceled", "rejected", "expired", "triggered", "filled", "closed"):
            continue
        c = _extract_coin(ord_item, pos_id_to_coin)
        if c:
            open_orders_by_coin.setdefault(c, []).append(ord_item)

    tpsl_by_coin = {}
    for t_item in tpsl_list:
        if not isinstance(t_item, dict):
            continue
        c = _extract_coin(t_item, pos_id_to_coin)
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
                    cancel_resting_order(client, cancel_tool, coin, resting, is_dry)

            raw_side = str(pos.get("side") or pos.get("direction") or pos.get("positionSide") or "LONG").upper()
            side_int = 1 if ("LONG" in raw_side or "BUY" in raw_side) else -1
            entry_px = _parse_float(pos, ("entryPrice", "openPrice", "avgEntryPrice", "avgPrice", "price"), default=mkt_px)
            cur_sl_px = _parse_float(pos, _SL_KEYS, default=0.0)
            cur_tp_px = _parse_float(pos, _TP_KEYS, default=0.0)
            pos_id = get_position_id(pos)

            # Si list_futures_tp_sl_orders({}) était vide, tenter avec positionId / symbol si le schéma l'accepte
            # IMPORTANT : toujours filtrer par (coin, pos_id) car MoonX peut renvoyer les ordres d'autres cryptos
            if coin not in tpsl_by_coin and pos_id:
                try:
                    sub_tpsl = _extract_list(client.map_and_call("list_futures_tp_sl_orders", {"symbol": coin, "positionId": pos_id}))
                    if sub_tpsl:
                        print(f"[{coin}] TP/SL bruts reçus pour pos_id={pos_id} : {json.dumps(sub_tpsl, ensure_ascii=False)[:400]}", flush=True)
                        matched_tpsl = [it for it in sub_tpsl if _item_matches_position(it, coin, pos_id)]
                        if matched_tpsl:
                            tpsl_by_coin[coin] = matched_tpsl
                except Exception:
                    pass

            # Lire SL/TP actuels : vérifier positionId/symbol ET ordre de grandeur vs entry_px (jamais mélanger BTC et ETH)
            for t_item in tpsl_by_coin.get(coin, []):
                if not _item_matches_position(t_item, coin, pos_id):
                    continue
                cands = []
                for kk in ("stopLossPrice", "slPrice", "stopLoss", "sl", "takeProfitPrice", "tpPrice", "takeProfit", "tp", "triggerPrice", "price"):
                    vv = _parse_float(t_item, (kk,), default=0.0)
                    if vv > 0 and entry_px * 0.2 <= vv <= entry_px * 5.0:
                        cands.append((kk, vv))
                for kk, vv in cands:
                    kl = kk.lower()
                    if kl in ("stoplossprice", "slprice", "stoploss", "sl"):
                        is_sl = True
                    elif kl in ("takeprofitprice", "tpprice", "takeprofit", "tp"):
                        is_sl = False
                    else:
                        typ = json.dumps(t_item, ensure_ascii=False).lower()
                        if "stop" in typ and "take" not in typ:
                            is_sl = True
                        elif "take" in typ and "stop" not in typ:
                            is_sl = False
                        else:
                            is_sl = (vv < entry_px) if side_int == 1 else (vv > entry_px)
                    if is_sl and cur_sl_px <= 0:
                        cur_sl_px = vv
                    elif (not is_sl) and cur_tp_px <= 0:
                        cur_tp_px = vv

            # MoonX n'expose pas SL/TP dans la position : utiliser la mémoire de ce que le BOT a posé (jamais reculer le SL)
            bot_sl = state.setdefault("bot_sl", {})
            mem = bot_sl.get(coin)
            if mem and abs(float(mem.get("entry", 0)) - entry_px) / max(entry_px, 1e-9) < 0.002:
                # Vérifier aussi que c'est bien la même position (même pos_id si mémorisé)
                if not mem.get("pos_id") or not pos_id or str(mem.get("pos_id")) == str(pos_id):
                    if cur_sl_px <= 0:
                        cur_sl_px = float(mem.get("sl", 0.0))
                    if cur_tp_px <= 0:
                        cur_tp_px = float(mem.get("tp", 0.0))
                else:
                    bot_sl.pop(coin, None)
            elif mem:
                bot_sl.pop(coin, None)  # nouvelle position sur ce coin : mémoire obsolète

            # Mode check_30m (identique au backtest) : trailing évalué sur le prix au moment du run 30m, et le SL ne recule jamais
            trail = scv4.compute_live_trailing_sl(side_int, entry_px, mkt_px, current_sl=cur_sl_px, p=p)

            side_lbl = 'LONG' if side_int == 1 else 'SHORT'
            known_pos = state.setdefault("known_positions", {})
            if coin not in known_pos:
                known_pos[coin] = {"entry": entry_px, "ts": now_ts}
                sl_txt = f"{cur_sl_px:g}" if cur_sl_px > 0 else "en cours"
                tp_txt = f"{cur_tp_px:g}" if cur_tp_px > 0 else "en cours"
                nt = f"📥 {coin} : position {side_lbl} détectée @ {entry_px:g} (ordre limite exécuté ou ouverture) | SL {sl_txt} | TP {tp_txt}"
                actions_log.append(nt)
                tg_send(("[SIMULATION] " if is_dry else "") + nt)
            has_tpsl = cur_sl_px > 0 and cur_tp_px > 0
            # Un SL initial ne doit être recalibré que s'il est PLUS LARGE / PLUS RISQUÉ que -50% (jamais reculer un SL déjà à BE ou meilleur !)
            if cur_sl_px > 0:
                sl_too_wide = (
                    (cur_sl_px < trail["target_sl_price"] * 0.9985) if side_int == 1
                    else (cur_sl_px > trail["target_sl_price"] * 1.0015)
                )
            else:
                sl_too_wide = False
            tp_misaligned = cur_tp_px > 0 and abs(cur_tp_px - trail["tp_price"]) / entry_px > 0.0015

            if trail["tier"] == 0 and (not has_tpsl or sl_too_wide or tp_misaligned):
                msg = (
                    f"🛡️ {coin} ({side_lbl} @ {entry_px:g}) : TP/SL manquant ou trop large -> pose TP +300% ({trail['tp_price']:g}) "
                    f"& SL -50% ({trail['target_sl_price']:g}) [actuel SL={cur_sl_px:g} TP={cur_tp_px:g}]"
                )
                print(msg, flush=True)
                actions_log.append(msg)
                if not is_dry:
                    ok = apply_tp_sl(client, coin, pos_id, side_int, entry_px, trail["tp_price"],
                                     trail["target_sl_price"], -sl_roi_pct, tp_roi_pct, "pose TP/SL")
                    if ok:
                        bot_sl[coin] = {"sl": trail["target_sl_price"], "tp": trail["tp_price"], "entry": entry_px, "pos_id": pos_id}
                    tg_send(f"{'✅' if ok else '⚠️'} {msg}")
                else:
                    tg_send(f"[SIMULATION] {msg}")
            elif trail["should_update"]:
                msg = (
                    f"🔒 {coin} ({side_lbl} | ROI {trail['roi_cur']:+.1f}%) : "
                    f"Trailing palier {trail['tier']} -> SL de {cur_sl_px:g} à {trail['target_sl_price']:g} ({trail['target_sl_roi']:+.0f}% ROI)"
                )
                print(msg, flush=True)
                actions_log.append(msg)
                if not is_dry:
                    ok = apply_tp_sl(client, coin, pos_id, side_int, entry_px, trail["tp_price"],
                                     trail["target_sl_price"], trail["target_sl_roi"], tp_roi_pct, f"trailing palier {trail['tier']}")
                    if ok:
                        bot_sl[coin] = {"sl": trail["target_sl_price"], "tp": trail["tp_price"], "entry": entry_px, "pos_id": pos_id}
                    tg_send(f"{'✅' if ok else '⚠️'} {msg}")
                else:
                    tg_send(f"[SIMULATION] {msg}")
            else:
                actions_log.append(
                    f"⏳ {coin} : {side_lbl} @ {entry_px:g} (Prix {mkt_px:g} | ROI {trail['roi_cur']:+.1f}% | "
                    f"Palier {trail['tier']} | SL {cur_sl_px:g} | TP {cur_tp_px:g})"
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
        cancel_failed = False

        if resting_orders:
            for ro in resting_orders:
                ro_side_raw = str(ro.get("side") or ro.get("direction") or "").lower()
                ro_side = "long" if ("long" in ro_side_raw or "buy" in ro_side_raw) else "short"
                ro_px = _parse_float(ro, ("price", "limitPrice", "orderPrice", "entryPrice"), default=0.0)

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
                    if sig["action"] == "LIMIT":
                        if sig.get("side") != ro_side:
                            reason_lbl = f"inversion {ro_side.upper()} -> {sig['side'].upper()}"
                        else:
                            reason_lbl = f"ajustement POI > {reprice_tol*100:g}% ({ro_px:g} -> {sig['entry_price']:g})"
                    else:
                        reason_lbl = sig.get("reason", f"passage en {sig['action']}")
                    msg = f"🔄 {coin} : Annulation ordre au repos @ {ro_px:g} ({reason_lbl})"
                    print(msg, flush=True)
                    actions_log.append(msg)
                    if cancel_resting_order(client, cancel_tool, coin, ro, is_dry):
                        ro_margin = _parse_float(ro, ("margin", "marginUsdt", "lockedMargin", "initialMargin"), default=0.0)
                        if ro_margin > 0 and not is_dry:
                            free_margin += ro_margin
                    else:
                        cancel_failed = True

        if kept_resting or cancel_failed:
            if cancel_failed:
                print(f"[{coin}] Nouvel ordre bloqué car l'annulation de l'ancien ordre au repos a échoué.", flush=True)
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
                    },
                )
                new_pos_id = get_position_id(open_res) if isinstance(open_res, dict) else None
                real_en_px = _parse_float(open_res, ("entryPrice", "openPrice", "avgEntryPrice", "avgPrice", "price"), default=0.0) if isinstance(open_res, dict) else 0.0
                if real_en_px <= 0:
                    try:
                        pos_after = _extract_list(client.call_tool("list_futures_positions", {}))
                        for p_af in pos_after:
                            if isinstance(p_af, dict) and _norm_base_coin(p_af.get("symbol") or p_af.get("pair") or p_af.get("instrument")) == coin:
                                real_en_px = _parse_float(p_af, ("entryPrice", "openPrice", "avgEntryPrice", "avgPrice", "price"), default=0.0)
                                if not new_pos_id:
                                    new_pos_id = get_position_id(p_af)
                                break
                    except Exception:
                        pass
                if real_en_px > 0:
                    en_px = real_en_px
                    s_int = 1 if side_str == "long" else -1
                    sl_px = scv4._price_at_roi(s_int, en_px, -sl_roi_pct, lev)
                    tp_px = scv4._price_at_roi(s_int, en_px, tp_roi_pct, lev)
                s_int_open = 1 if side_str == "long" else -1
                ok_open = apply_tp_sl(client, coin, new_pos_id, s_int_open, en_px, tp_px, sl_px, -sl_roi_pct, tp_roi_pct, "ouverture marché")
                if ok_open:
                    state.setdefault("bot_sl", {})[coin] = {"sl": sl_px, "tp": tp_px, "entry": en_px, "pos_id": new_pos_id}
                free_margin = max(0.0, free_margin - trade_margin_usdt)

            msg = (
                f"🚀 {coin} : OUVERTURE MARCHÉ {side_str.upper()} @ {en_px:g} | "
                f"Marge {trade_margin_usdt:.2f} USDT (1R={one_r_usdt:.2f} USDT, {a.risk_pct:g}% de {total_eq:.0f}$) | "
                f"TP {tp_px:g} (+300%) | SL {sl_px:g} (-50%)"
            )
            print(msg, flush=True)
            actions_log.append(msg)
            tg_send(f"{'[SIMULATION] ' if is_dry else ''}{msg}")

        elif sig["action"] == "LIMIT":
            lim_schema = client.tools_schema.get("open_futures_limit_order", {}) or {}
            lim_props = lim_schema.get("properties", {}) if isinstance(lim_schema, dict) else {}
            lim_props_low = {str(k).lower() for k in lim_props.keys()}
            has_lim_tp = any(k in lim_props_low for k in ("takeprofitprice", "tpprice", "take_profit_price", "takeprofit", "tp", "tptriggerprice", "takeprofittriggerprice", "takeprofitgainpercent", "tpgainpercent", "tppercent", "takeprofitpercent"))
            has_lim_sl = any(k in lim_props_low for k in ("stoplossprice", "slprice", "stop_loss_price", "stoploss", "sl", "sltriggerprice", "stoplosstriggerprice", "stoplosslosspercent", "sllosspercent", "slpercent", "stoplosspercent"))

            if not is_dry:
                print(f"[{coin}] Schéma open_futures_limit_order : {json.dumps(lim_schema, ensure_ascii=False)[:600]}", flush=True)
                lim_res = client.map_and_call(
                    "open_futures_limit_order",
                    {
                        "symbol": coin,
                        "side": side_str,
                        "price": en_px,
                        "margin": trade_margin_usdt,
                        "quantity": qty_coins,
                        "leverage": int(lev),
                        "marginMode": margin_mode,
                        "takeProfitPrice": tp_px,
                        "stopLossPrice": sl_px,
                        "takeProfitGainPercent": tp_roi_pct,
                        "stopLossLossPercent": sl_roi_pct,
                    },
                )
                print(f"[{coin}] open_futures_limit_order réponse : {json.dumps(lim_res, ensure_ascii=False)[:400]}", flush=True)
                if client.is_error(lim_res):
                    tg_send(f"❌ {coin} : ordre limite refusé par MoonX → {json.dumps(lim_res, ensure_ascii=False)[:300]}")
                else:
                    lim_pos_id = get_position_id(lim_res) if isinstance(lim_res, dict) else None
                    tpsl_attached = bool(has_lim_tp and has_lim_sl)
                    if not tpsl_attached and lim_pos_id:
                        tpsl_attached = apply_tp_sl(client, coin, lim_pos_id, 1 if side_str == "long" else -1, en_px, tp_px, sl_px, -sl_roi_pct, tp_roi_pct, "ordre limite")
                    note_tpsl = (
                        f"TP {tp_px:g} (+300%) | SL {sl_px:g} (-50%)"
                        if tpsl_attached
                        else f"TP prévu {tp_px:g} (+300%) | SL prévu {sl_px:g} (-50%) ⚠️ (non supporté en amont par MoonX sur ordre limite)"
                    )
                    msg = (
                        f"🎯 {coin} : ORDRE LIMITE {side_str.upper()} placé @ {en_px:g} (POI {sig['poi']:g}) | "
                        f"Marge {trade_margin_usdt:.2f} USDT (1R={one_r_usdt:.2f} USDT, {a.risk_pct:g}% de {total_eq:.0f}$) | "
                        f"{note_tpsl}"
                    )
                    print(msg, flush=True)
                    actions_log.append(msg)
                    tg_send(msg)
                    free_margin = max(0.0, free_margin - trade_margin_usdt)
            else:
                msg = (
                    f"🎯 {coin} : ORDRE LIMITE {side_str.upper()} placé @ {en_px:g} (POI {sig['poi']:g}) | "
                    f"Marge {trade_margin_usdt:.2f} USDT (1R={one_r_usdt:.2f} USDT, {a.risk_pct:g}% de {total_eq:.0f}$) | "
                    f"TP {tp_px:g} (+300%) | SL {sl_px:g} (-50%)"
                )
                print(msg, flush=True)
                actions_log.append(msg)
                tg_send(f"[SIMULATION] {msg}")

    # Nettoyer les positions mémorisées qui ne sont plus ouvertes (pour re-notifier la prochaine entrée)
    kp = state.get("known_positions", {})
    for c_old in list(kp.keys()):
        if c_old not in open_pos_by_coin:
            kp.pop(c_old, None)
    state["known_positions"] = kp
    state["raw_dumped"] = {k: v for k, v in state.get("raw_dumped", {}).items() if k in open_pos_by_coin}
    state.pop("peak_px", None)

    save_bot_state(a.state_file, state)

    raw_ov_preview = ""
    if is_dry:
        sch = client.tools_schema.get("set_futures_tp_sl", {})
        raw_ov_preview = (
            f"\n🔍 Schéma MoonX set_futures_tp_sl : {json.dumps(sch, ensure_ascii=False)[:700]}"
        )
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

