"""
backtest/strategy_crypto_v4.py — Stratégie CRYPTO V4 (A2-R — Stratégie SMC — Crypto Futures).

Spécifications exactes du PDF (MoonX / Cubrik) :
  1. Cadence :
     - Check toutes les 30 minutes (:00 et :30 UTC, soit à la clôture des bougies 5m :25 et :55).
  2. Sizing (fixe) :
     - Marge : 200 USDT fixe par instrument (`margin_usdt = 200.0`).
     - Levier : x30, mode isolé (`leverage = 30.0`).
     - Quantité : `(marge * levier) / prix = 6 000 / prix`.
     - Exposition : une par instrument (position ouverte OU ordre limite au repos, jamais les deux).
     - Unité de risque (1 R) : SL initial = -50 % de ROI sur marge = -100 USDT (variation de prix de 1.6667 %).
  3. Instruments (9 crypto futures) :
     - BTC, ETH, SOL, XRP, AVAX, ZEC, INJ, PUMP, HYPE (évalués à chaque run sans filtre de série de pertes).
  4. Étape 2 — Gestion des positions ouvertes (levier x30, isolé) :
     - ROI sur marge (%) = `levier * (prix - entree) / entree * 100` (long) | `levier * (entree - prix) / entree * 100` (short).
     - TP fixe : +300 % de ROI sur marge (+6.0 R = +600 USDT), posé dès l'ouverture (clôture 100 % de la position).
     - SL initial : -50 % de ROI sur marge (-1.0 R = -100 USDT), posé dès l'ouverture.
     - Trailing du SL (ne recule jamais, seul le palier le plus haut atteint s'applique) :
       * ROI >= +50 %  (+1 R) -> SL remonté au break-even (0 % ROI = 0 R, prix d'entrée)
       * ROI >= +100 % (+2 R) -> SL remonté à +50 % de profit (+1 R = +100 USDT)
       * ROI >= +150 % (+3 R) -> SL remonté à +100 % de profit (+2 R = +200 USDT)
       * ROI >= +200 % (+4 R) -> SL remonté à +150 % de profit (+3 R = +300 USDT)
  5. Étape 3 — Détection & Ré-analyse à chaque run (toutes les 30 min) :
     - Biais H4 : `net / range` des 8 dernières clôtures 4h. Haussier si > +0.35, baissier si < -0.35, sinon neutre.
     - Fenêtre M15 : `range (haut - bas) / prix >= 0.4 %` sur les 16 dernières clôtures 15m.
     - Direction : celle du biais H4 ; si neutre, signe du momentum M15 (`net / range`).
     - POI :
       * long  = `after_hi - 0.5 * (after_hi - win_lo)`
       * short = `after_lo + 0.5 * (win_hi - after_lo)`
     - Côté : un long exige `POI < prix`, un short exige `POI > prix`. Sinon, sauter (pas de fallback).
     - Type d'entrée :
       * Au marché si le prix traverse le POI (`long : prix <= POI * 1.0005` ; `short : prix >= POI * 0.9995`).
       * Sinon ordre limite au POI ajusté (`long : POI * 0.999` ; `short : POI * 1.001`).
     - Ordres au repos : ré-analysés à chaque run de 30 min ; annulés si la thèse est cassée (fenêtre M15 < 0.4 %,
       direction inversée, côté invalide), ou ajustés si le nouveau POI s'écarte de plus de 0.1 % du prix de l'ordre.
"""
import numpy as np
import pandas as pd

CRYPTO_V4_SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "AVAXUSD",
    "ZECUSD",
    "INJUSD",
    "PUMPUSD",
    "HYPEUSD",
]

PARAMS = {
    "margin_usdt": 200.0,       # Marge fixe 200 USDT par instrument
    "leverage": 30.0,           # Levier x30 isolé
    "tp_roi_pct": 300.0,        # TP fixe : +300% de ROI sur marge (+6 R)
    "sl_roi_pct": 50.0,         # SL initial : -50% de ROI sur marge (-1 R)
    "h4_lookback": 8,           # 8 dernières clôtures 4h
    "h4_thresh": 0.35,          # Seuil biais H4 : > +0.35 haussier, < -0.35 baissier
    "m15_lookback": 16,         # 16 dernières clôtures 15m
    "m15_min_range_pct": 0.4,   # Range (haut - bas) / prix >= 0.4%
    "poi_fib": 0.5,             # Retracement 50% (équilibre SMC)
    "mkt_tol_pct": 0.05,        # Entrée au marché si prix <= POI * 1.0005 (0.05%)
    "lim_offset_pct": 0.1,      # Ordre limite à POI * 0.999 (long) / POI * 1.001 (short)
    "reprice_tol_pct": 0.1,     # Remplace l'ordre au repos si écart > 0.1%
    "trail_mode": "check_30m",  # "check_30m" (vérifié au run 30m comme Cubrik) | "intra" (continu sur bougie 5m)
    "fee_pct": 0.02,            # Frais futures par ordre en % du notionnel (0.02% maker/taker moyen MoonX/Binance Futures, 0 = brut)
}

TRAIL_TIERS = (
    (50.0, 0.0),     # ROI +50%  -> SL au break-even (0% ROI = 0 R)
    (100.0, 50.0),   # ROI +100% -> SL à +50% ROI (+1 R)
    (150.0, 100.0),  # ROI +150% -> SL à +100% ROI (+2 R)
    (200.0, 150.0),  # ROI +200% -> SL à +150% ROI (+3 R)
)


def parse_params(text, base=None):
    p = dict(PARAMS if base is None else base)
    for part in [x.strip() for x in (text or "").replace("\n", ";").split(";") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"réglage invalide '{part}' (format attendu : nom=valeur)")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in PARAMS:
            raise ValueError(f"réglage inconnu '{k}'. Possibles : {', '.join(PARAMS)}")
        cur = PARAMS[k]
        p[k] = float(v) if isinstance(cur, float) else int(float(v)) if isinstance(cur, int) else v
    return p


def _price_at_roi(side: int, entry: float, roi_pct: float, leverage: float) -> float:
    """Prix correspondant à un `roi_pct` (% sur marge) pour un levier donné."""
    frac = (roi_pct / 100.0) / leverage
    return float(entry * (1.0 + frac)) if side == 1 else float(entry * (1.0 - frac))


def _roi_at_price(side: int, entry: float, price: float, leverage: float) -> float:
    """ROI sur marge (%) = levier * (prix - entree) / entree * 100 (long) ou inverse (short)."""
    if entry <= 0:
        return 0.0
    move = ((price - entry) / entry) if side == 1 else ((entry - price) / entry)
    return float(leverage * move * 100.0)


def run(pd_, capital=10000.0, risk_pct=1.0, fee_pct=None, leverage=30.0, p=None, i0=0, i1=None):
    """
    Exécute le backtest CRYPTO V4 sur `pd_` dans l'intervalle de bougies 5m `[i0, i1)`.
    Retourne `(trades_df, equity_array_on_slice)`.
    """
    p = dict(PARAMS if p is None else p)
    margin_usdt = float(p.get("margin_usdt", 200.0))
    lev = float(p.get("leverage", 30.0))
    tp_roi_pct = float(p.get("tp_roi_pct", 300.0))
    sl_roi_pct = float(p.get("sl_roi_pct", 50.0))
    h4_lb = int(p.get("h4_lookback", 8))
    h4_thresh = float(p.get("h4_thresh", 0.35))
    m15_lb = int(p.get("m15_lookback", 16))
    m15_min_rng = float(p.get("m15_min_range_pct", 0.4)) / 100.0
    poi_fib = float(p.get("poi_fib", 0.5))
    mkt_tol = float(p.get("mkt_tol_pct", 0.05)) / 100.0
    lim_off = float(p.get("lim_offset_pct", 0.1)) / 100.0
    reprice_tol = float(p.get("reprice_tol_pct", 0.1)) / 100.0
    trail_mode = str(p.get("trail_mode", "check_30m")).lower()

    # Frais en % du notionnel par ordre
    use_fee_pct = float(p.get("fee_pct", 0.02)) if fee_pct is None else float(fee_pct)
    fee = use_fee_pct / 100.0

    # 1 R correspond au risque initial SL = -sl_roi_pct% de la marge (soit 50% * 200 = 100 USDT)
    one_r_roi = sl_roi_pct if sl_roi_pct > 0 else 50.0
    one_r_usdt = margin_usdt * (one_r_roi / 100.0)

    B = pd_.base
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    open_ns, close_ns = B["open_ns"], B["close_ns"]
    n = pd_.n
    i1 = n if i1 is None else min(n, i1)
    i0 = max(0, i0)

    f15 = pd_.frames["15m"]
    h15, l15, c15, close_ns_15 = f15["h"], f15["l"], f15["c"], f15["close_ns"]

    f4h = pd_.frames["4h"]
    h4, l4, c4, close_ns_4h = f4h["h"], f4h["l"], f4h["c"], f4h["close_ns"]

    # Index de la dernière bougie 15m et 4h clôturée à ou avant la clôture de chaque bougie 5m
    j15_vec = np.searchsorted(close_ns_15, close_ns, side="right") - 1
    j4h_vec = np.searchsorted(close_ns_4h, close_ns, side="right") - 1

    # Les checks de 30 minutes ont lieu lorsque close_ns est un multiple exact de 30 minutes (1800s)
    ns_30m = 30 * 60 * 10**9
    is_check_30m = (close_ns % ns_30m) == 0

    pos = None    # dict de la position ouverte
    order = None  # dict de l'ordre limite au repos
    raw_trades = []

    def close_position(pos_dict, exit_idx, exit_px, reason_str):
        side = pos_dict["side"]
        en = pos_dict["entry"]
        roi_gross_pct = _roi_at_price(side, en, exit_px, lev)
        # Frais aller-retour exprimés en % de la marge : 2 * fee * lev * 100
        roi_fee_pct = 2.0 * fee * lev * 100.0
        roi_net_pct = roi_gross_pct - roi_fee_pct

        r_gross = roi_gross_pct / one_r_roi
        r_net = roi_net_pct / one_r_roi
        pnl_usdt = margin_usdt * (roi_net_pct / 100.0)
        pnl_gross_usdt = margin_usdt * (roi_gross_pct / 100.0)

        fill_i = pos_dict["fill_i"]
        sig_i = pos_dict["sig_i"]
        raw_trades.append({
            "fill_i": fill_i,
            "exit_i": exit_idx,
            "entry_time": pd.Timestamp(int(open_ns[fill_i]), tz="UTC"),
            "exit_time": pd.Timestamp(int(open_ns[exit_idx]), tz="UTC"),
            "signal_time": pd.Timestamp(int(open_ns[sig_i]), tz="UTC"),
            "side": "long" if side == 1 else "short",
            "entry": float(en),
            "exit": float(exit_px),
            "sl": float(pos_dict["sl"]),
            "tp": float(pos_dict["tp"]),
            "initial_sl": float(pos_dict["initial_sl"]),
            "poi": float(pos_dict["poi"]),
            "entry_type": pos_dict["entry_type"],
            "reason": reason_str,
            "roi_pct": float(roi_net_pct),
            "roi_gross_pct": float(roi_gross_pct),
            "r": float(r_net),
            "r_gross": float(r_gross),
            "pnl_usdt": float(pnl_usdt),
            "pnl_gross_usdt": float(pnl_gross_usdt),
            "margin_usdt": float(margin_usdt),
            "leverage": float(lev),
            "bars": int(exit_idx - fill_i),
            "session": pos_dict["entry_type"],
            "conf_tf": f"15m/4h ({pos_dict['entry_type']})",
            "zone_lo": float(min(en, pos_dict["poi"])),
            "zone_hi": float(max(en, pos_dict["poi"])),
            "zone_form": pd.Timestamp(int(open_ns[sig_i]), tz="UTC"),
            "zone_end": pd.Timestamp(int(close_ns[exit_idx]), tz="UTC"),
        })

    for k in range(i0, i1):
        hk = float(h[k])
        lk = float(l[k])
        ck = float(c[k])
        ok_p = float(o[k])

        # ------------------------------------------------------------------
        # 1) Exécution intra-bougie 5m : TP / SL de la position active ou fill de l'ordre limite
        # ------------------------------------------------------------------
        if pos is not None:
            side = pos["side"]
            sl = pos["sl"]
            tp = pos["tp"]
            hit_sl = (lk <= sl) if side == 1 else (hk >= sl)
            hit_tp = (hk >= tp) if side == 1 else (lk <= tp)

            if hit_sl or hit_tp:
                tier = pos["tier"]
                if hit_sl:
                    xp = float(min(sl, ok_p)) if (side == 1 and k > pos["fill_i"]) else (
                         float(max(sl, ok_p)) if (side == -1 and k > pos["fill_i"]) else float(sl))
                    if tier == 0:
                        reason = "SL (-50%)"
                    elif tier == 1:
                        reason = "BE (0%)"
                    else:
                        lock_pct = int(TRAIL_TIERS[tier - 1][1])
                        reason = f"TRAIL (+{lock_pct}%)"
                else:
                    xp = float(max(tp, ok_p)) if (side == 1 and k > pos["fill_i"]) else (
                         float(min(tp, ok_p)) if (side == -1 and k > pos["fill_i"]) else float(tp))
                    reason = "TP (+300%)"
                close_position(pos, k, xp, reason)
                pos = None
            elif trail_mode == "intra":
                best_px = hk if side == 1 else lk
                roi_peak = _roi_at_price(side, pos["entry"], best_px, lev)
                for tier_idx, (trig_roi, lock_roi) in enumerate(TRAIL_TIERS, 1):
                    if roi_peak >= trig_roi and tier_idx > pos["tier"]:
                        pos["tier"] = tier_idx
                        pos["sl"] = _price_at_roi(side, pos["entry"], lock_roi, lev)

        elif order is not None:
            side = order["side"]
            lim = order["limit"]
            filled = (lk <= lim) if side == 1 else (hk >= lim)
            if filled:
                en = float(lim)
                init_sl = _price_at_roi(side, en, -sl_roi_pct, lev)
                init_tp = _price_at_roi(side, en, tp_roi_pct, lev)
                pos = {
                    "side": side,
                    "entry": en,
                    "sl": init_sl,
                    "initial_sl": init_sl,
                    "tp": init_tp,
                    "tier": 0,
                    "fill_i": k,
                    "sig_i": order["sig_i"],
                    "poi": order["poi"],
                    "entry_type": "LIMIT",
                }
                order = None
                # Vérifie si la même bougie 5m a aussi touché le SL initial
                if (side == 1 and lk <= init_sl) or (side == -1 and hk >= init_sl):
                    close_position(pos, k, init_sl, "SL (-50%)")
                    pos = None

        # ------------------------------------------------------------------
        # 2) Cadence 30 minutes : Étape 2 (Trailing SL) & Étape 3 (Détection / Ré-analyse)
        # ------------------------------------------------------------------
        if not is_check_30m[k]:
            continue

        j15 = int(j15_vec[k])
        j4 = int(j4h_vec[k])
        if j15 < m15_lb - 1 or j4 < h4_lb - 1:
            continue

        px = ck
        if px <= 0 or np.isnan(px):
            continue

        # --- Étape 2 : Gérer la position ouverte (levier x30, isolé) ---
        if pos is not None:
            side = pos["side"]
            en = pos["entry"]
            roi_cur = _roi_at_price(side, en, px, lev)
            for tier_idx, (trig_roi, lock_roi) in enumerate(TRAIL_TIERS, 1):
                if roi_cur >= trig_roi and tier_idx > pos["tier"]:
                    pos["tier"] = tier_idx
                    new_sl = _price_at_roi(side, en, lock_roi, lev)
                    # Si le prix actuel a déjà franchi ce nouveau SL (impossible normalement car roi_cur >= trig_roi > lock_roi)
                    pos["sl"] = new_sl
            continue

        # --- Étape 3 : Détection & Ré-analyse (instruments sans position ouverte) ---
        # 3.1 Biais H4 : net/range des 8 dernières clôtures 4h
        s4_0 = j4 - h4_lb + 1
        s4_1 = j4 + 1
        net_h4 = float(c4[j4] - c4[s4_0])
        rng_h4 = float(np.max(h4[s4_0:s4_1]) - np.min(l4[s4_0:s4_1]))
        biais_h4 = (net_h4 / rng_h4) if rng_h4 > 0 else 0.0

        # 3.2 Fenêtre M15 : range (haut - bas) / prix >= 0.4 % sur les 16 dernières clôtures 15m
        s15_0 = j15 - m15_lb + 1
        s15_1 = j15 + 1
        w15_h = h15[s15_0:s15_1]
        w15_l = l15[s15_0:s15_1]
        win_hi = float(np.max(w15_h))
        win_lo = float(np.min(w15_l))
        rng_m15 = win_hi - win_lo

        if (rng_m15 / px) < m15_min_rng:
            # Fenêtre M15 trop étroite (< 0.4%) : annuler tout ordre au repos et sauter
            order = None
            continue

        # 3.3 Direction : celle du biais H4 ; si neutre, signe du momentum M15 (net/range)
        if biais_h4 > h4_thresh:
            direction = 1
        elif biais_h4 < -h4_thresh:
            direction = -1
        else:
            net_m15 = float(c15[j15] - c15[s15_0])
            mom_m15 = (net_m15 / rng_m15) if rng_m15 > 0 else 0.0
            if mom_m15 > 0:
                direction = 1
            elif mom_m15 < 0:
                direction = -1
            else:
                direction = 0

        if direction == 0:
            order = None
            continue

        # 3.4 Calcul du POI et validation du Côté
        if direction == 1:
            # long = after_hi - 0.5 * (after_hi - win_lo)
            i_lo = int(np.argmin(w15_l))
            after_hi = float(np.max(w15_h[i_lo:]))
            poi = after_hi - poi_fib * (after_hi - win_lo)

            # Côté : un long exige POI < prix. Sinon, sauter (et annuler l'ordre mort).
            if not (poi < px):
                order = None
                continue

            # Type d'entrée : au marché si prix <= POI * 1.0005, sinon limite à POI * 0.999
            if px <= poi * (1.0 + mkt_tol):
                order = None
                en = float(px)
                init_sl = _price_at_roi(1, en, -sl_roi_pct, lev)
                init_tp = _price_at_roi(1, en, tp_roi_pct, lev)
                pos = {
                    "side": 1,
                    "entry": en,
                    "sl": init_sl,
                    "initial_sl": init_sl,
                    "tp": init_tp,
                    "tier": 0,
                    "fill_i": k,
                    "sig_i": k,
                    "poi": poi,
                    "entry_type": "MARKET",
                }
            else:
                new_lim = float(poi * (1.0 - lim_off))
                if order is None or order["side"] != 1 or abs(new_lim - order["limit"]) / order["limit"] > reprice_tol:
                    order = {"side": 1, "limit": new_lim, "poi": poi, "sig_i": k}

        else:
            # short = after_lo + 0.5 * (win_hi - after_lo)
            i_hi = int(np.argmax(w15_h))
            after_lo = float(np.min(w15_l[i_hi:]))
            poi = after_lo + poi_fib * (win_hi - after_lo)

            # Côté : un short exige POI > prix. Sinon, sauter (et annuler l'ordre mort).
            if not (poi > px):
                order = None
                continue

            # Type d'entrée : au marché si prix >= POI * 0.9995, sinon limite à POI * 1.001
            if px >= poi * (1.0 - mkt_tol):
                order = None
                en = float(px)
                init_sl = _price_at_roi(-1, en, -sl_roi_pct, lev)
                init_tp = _price_at_roi(-1, en, tp_roi_pct, lev)
                pos = {
                    "side": -1,
                    "entry": en,
                    "sl": init_sl,
                    "initial_sl": init_sl,
                    "tp": init_tp,
                    "tier": 0,
                    "fill_i": k,
                    "sig_i": k,
                    "poi": poi,
                    "entry_type": "MARKET",
                }
            else:
                new_lim = float(poi * (1.0 + lim_off))
                if order is None or order["side"] != -1 or abs(new_lim - order["limit"]) / order["limit"] > reprice_tol:
                    order = {"side": -1, "limit": new_lim, "poi": poi, "sig_i": k}

    # 3) Courbe d'équité (en mode marge fixe 200 USDT : 1R = 100 USDT, ou % du capital)
    eq = np.full(n, float(capital), dtype=np.float64)
    bal = float(capital)
    eqpos = 0

    for t in raw_trades:
        fi = t["fill_i"]
        xi = t["exit_i"]
        if fi > eqpos:
            eq[eqpos:fi] = bal
            eqpos = fi
        # PnL en USDT (marge fixe 200 USDT, 1R = 100 USDT)
        pnl_cash = float(t["pnl_usdt"]) if risk_pct <= 0 else float(t["r"] * one_r_usdt)
        t["pnl"] = pnl_cash
        if xi >= eqpos:
            eq[eqpos:xi] = bal
            bal += pnl_cash
            eq[xi] = bal
            eqpos = xi + 1
        else:
            bal += pnl_cash

    eq[eqpos:] = bal

    for t in raw_trades:
        t.pop("fill_i", None)
        t.pop("exit_i", None)

    return pd.DataFrame(raw_trades), eq[i0:i1]


def compute_period_table(trades_df: pd.DataFrame, period: str = "M", start_capital: float = 1800.0, one_r_usdt: float = 100.0) -> pd.DataFrame:
    """
    Calcule le tableau détaillé par Mois (`period="M"`) ou par Année (`period="Y"`) :
      - `period`       : 'YYYY-MM' ou 'YYYY'
      - `trades`       : nombre de trades clôturés sur la période
      - `win_pct`      : % de trades gagnants (r > 0)
      - `be_pct`       : % de trades à Break-Even
      - `avg_r`        : RR moyen par trade sur la période
      - `sum_r`        : Gain net cumulé en R sur la période
      - `pnl_usdt`     : Gain net en USDT (base 200 USDT de marge/position, 1R = 100 USDT)
      - `roi_margin_pct`: Gain cumulé en % de la marge de 200 USDT (`sum_r * 50%`)
      - `return_pct`   : Gain en % du portefeuille (`pnl_usdt / start_capital * 100`)
      - `max_dd_r`     : Drawdown max intra-période en R
      - `max_dd_usdt`  : Drawdown max intra-période en USDT
      - `max_dd_pct`   : Drawdown max intra-période en % du capital de début de période
    """
    cols = [
        "period", "trades", "wins", "be", "losses", "win_pct",
        "avg_r", "sum_r", "pnl_usdt", "roi_margin_pct", "return_pct",
        "max_dd_r", "max_dd_usdt", "max_dd_pct",
    ]
    if trades_df is None or len(trades_df) == 0:
        return pd.DataFrame(columns=cols)

    df = trades_df.copy()
    dt = pd.to_datetime(df["exit_time"], utc=True)
    df["_sort_ts"] = dt
    df = df.sort_values("_sort_ts").reset_index(drop=True)
    if period.upper().startswith("M"):
        df["period"] = dt.dt.strftime("%Y-%m")
    else:
        df["period"] = dt.dt.strftime("%Y")

    rows = []
    running_cap = float(start_capital)

    for per, grp in df.groupby("period", sort=True):
        r_arr = grp["r"].to_numpy(dtype=float)
        pnl_arr = grp["pnl_usdt"].to_numpy(dtype=float) if "pnl_usdt" in grp.columns else (r_arr * one_r_usdt)
        n_tr = len(r_arr)
        wins = int((r_arr > 0.05).sum())
        be = int((np.abs(r_arr) <= 0.05).sum())
        losses = int((r_arr < -0.05).sum())
        win_pct = float((wins / n_tr) * 100.0) if n_tr > 0 else 0.0
        avg_r = float(r_arr.mean()) if n_tr > 0 else 0.0
        sum_r = float(r_arr.sum())
        pnl_usdt = float(pnl_arr.sum())
        roi_margin_pct = float(sum_r * 50.0)

        # Courbe intra-période pour le drawdown du mois / de l'année
        base_cap = max(running_cap, start_capital)
        eq_r = np.concatenate([[0.0], np.cumsum(r_arr)])
        peak_r = np.maximum.accumulate(eq_r)
        dd_r_arr = eq_r - peak_r
        max_dd_r = float(dd_r_arr.min())

        eq_u = np.concatenate([[base_cap], base_cap + np.cumsum(pnl_arr)])
        peak_u = np.maximum.accumulate(eq_u)
        dd_u_arr = eq_u - peak_u
        max_dd_usdt = float(dd_u_arr.min())
        max_dd_pct = float(((eq_u - peak_u) / np.maximum(peak_u, 1.0)).min() * 100.0)
        ret_pct = float((pnl_usdt / base_cap) * 100.0)

        running_cap = max(start_capital * 0.2, running_cap + pnl_usdt)

        rows.append({
            "period": per,
            "trades": n_tr,
            "wins": wins,
            "be": be,
            "losses": losses,
            "win_pct": win_pct,
            "avg_r": avg_r,
            "sum_r": sum_r,
            "pnl_usdt": pnl_usdt,
            "roi_margin_pct": roi_margin_pct,
            "return_pct": ret_pct,
            "max_dd_r": max_dd_r,
            "max_dd_usdt": max_dd_usdt,
            "max_dd_pct": max_dd_pct,
        })

    return pd.DataFrame(rows, columns=cols)


def format_crypto_v4_report(all_trades: pd.DataFrame, per_sym_stats: list, start_capital: float = 1800.0, p: dict = None) -> str:
    """
    Formate le rapport complet CRYPTO V4 (Bilan par crypto, Bilan Annuel, Bilan Mensuel avec Drawdown, Gains/RR et RR moyen).
    """
    p = dict(PARAMS if p is None else p)
    margin_usdt = float(p.get("margin_usdt", 200.0))
    lev = float(p.get("leverage", 30.0))
    one_r_usdt = margin_usdt * (float(p.get("sl_roi_pct", 50.0)) / 100.0)

    lines = [
        "⚡ CRYPTO V4 (A2-R — Stratégie SMC — Crypto Futures)",
        f"Sizing : Marge fixe {margin_usdt:.0f} USDT/trade | Levier x{lev:g} isolé (notionnel {margin_usdt * lev:,.0f} USDT) | 1R = -50% ROI ({one_r_usdt:.0f} USDT) | TP = +300% ROI (+6R)",
        f"Capital de référence portefeuille (9 cryptos x {margin_usdt:.0f} USDT) : {start_capital:,.0f} USDT",
        "",
        "📊 1. BILAN PAR CRYPTO (toute la période) :",
    ]

    for st in per_sym_stats:
        lines.append(
            f"  • {st['symbol']:8s} : {st['trades']:4d}t | WR {st['win_pct']:4.1f}% (BE {st['be_pct']:4.1f}%) | "
            f"RR moy {st['avg_r']:+.3f}R | Cumul {st['sum_r']:+7.1f}R ({st['pnl_usdt']:+8.0f} USDT) | "
            f"DD max {st['max_dd_r']:6.1f}R ({st['max_dd_usdt']:+7.0f} USDT / {st['max_dd_pct']:5.1f}%)"
        )

    if all_trades is None or len(all_trades) == 0:
        lines.append("\n⚠️ Aucun trade généré.")
        return "\n".join(lines)

    # Statistiques globales portefeuille
    df = all_trades.copy()
    df["_sort_ts"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values("_sort_ts").reset_index(drop=True)

    r_all = df["r"].to_numpy(dtype=float)
    pnl_all = df["pnl_usdt"].to_numpy(dtype=float)
    tot_t = len(r_all)
    wins = int((r_all > 0.05).sum())
    be = int((np.abs(r_all) <= 0.05).sum())
    losses = int((r_all < -0.05).sum())
    avg_r = float(r_all.mean())
    sum_r = float(r_all.sum())
    tot_usdt = float(pnl_all.sum())

    eq_r = np.concatenate([[0.0], np.cumsum(r_all)])
    dd_r = float((eq_r - np.maximum.accumulate(eq_r)).min())
    eq_u = np.concatenate([[start_capital], start_capital + np.cumsum(pnl_all)])
    peak_u = np.maximum.accumulate(eq_u)
    dd_u = float((eq_u - peak_u).min())
    dd_pct = float(((eq_u - peak_u) / np.maximum(peak_u, 1.0)).min() * 100.0)

    # Répartition par motif de clôture
    reason_counts = df["reason"].value_counts().to_dict() if "reason" in df.columns else {}
    r_parts = [f"{k}: {v}" for k, v in reason_counts.items()]

    lines.extend([
        "",
        "🌍 2. BILAN GLOBAL PORTEFEUILLE (toutes cryptos combinées) :",
        f"  • Total trades : {tot_t} | Gagnants : {wins} ({wins/tot_t*100:.1f}%) | Break-Even : {be} ({be/tot_t*100:.1f}%) | Perdants : {losses} ({losses/tot_t*100:.1f}%)",
        f"  • RR moyen global : {avg_r:+.3f} R / trade ({avg_r * 50.0:+.2f}% ROI marge / trade)",
        f"  • Gain net total  : {sum_r:+.1f} R | {tot_usdt:+,.0f} USDT ({tot_usdt/start_capital*100:+.1f}% sur capital {start_capital:,.0f} USDT)",
        f"  • Drawdown max    : {dd_r:.1f} R | {dd_u:+,.0f} USDT ({dd_pct:.1f}%)",
        f"  • Clôtures        : {' | '.join(r_parts)}",
    ])

    # Tableau annuel
    yr_df = compute_period_table(df, period="Y", start_capital=start_capital, one_r_usdt=one_r_usdt)
    lines.extend([
        "",
        "📅 3. BILAN PAR ANNÉE (Gains, RR moyen et Drawdown max annuel) :",
        "  Année | Trades |  WR%  |  RR moy  |  Gain (R) | Gain (USDT) |  DD max (R) | DD max (USDT / %)",
        "  ------+--------+-------+----------+-----------+-------------+-------------+-------------------",
    ])
    for _, row in yr_df.iterrows():
        lines.append(
            f"  {row['period']:5s} | {int(row['trades']):6d} | {row['win_pct']:4.1f}% | {row['avg_r']:+7.3f}R | "
            f"{row['sum_r']:+8.1f}R | {row['pnl_usdt']:+10.0f}$ | {row['max_dd_r']:10.1f}R | "
            f"{row['max_dd_usdt']:+8.0f}$ ({row['max_dd_pct']:5.1f}%)"
        )

    # Tableau mensuel
    mo_df = compute_period_table(df, period="M", start_capital=start_capital, one_r_usdt=one_r_usdt)
    lines.extend([
        "",
        "🗓️ 4. BILAN PAR MOIS (Gains/RR, RR moyen et Drawdown par mois) :",
        "  Mois    | Trades |  WR%  |  RR moy  |  Gain (R) | Gain (USDT) |  DD mois (R) | DD mois (USDT / %)",
        "  --------+--------+-------+----------+-----------+-------------+--------------+-------------------",
    ])
    for _, row in mo_df.iterrows():
        lines.append(
            f"  {row['period']:7s} | {int(row['trades']):6d} | {row['win_pct']:4.1f}% | {row['avg_r']:+7.3f}R | "
            f"{row['sum_r']:+8.1f}R | {row['pnl_usdt']:+10.0f}$ | {row['max_dd_r']:11.1f}R | "
            f"{row['max_dd_usdt']:+8.0f}$ ({row['max_dd_pct']:5.1f}%)"
        )

    return "\n".join(lines)

