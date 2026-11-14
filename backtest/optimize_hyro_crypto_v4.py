"""
backtest/optimize_hyro_crypto_v4.py — Optimiseur Multi-Paliers (Walk-Forward) & Simulateur 4 Ans (2022-2026)
spécialement conçu pour le challenge **HyroTrader One-Step (10 000 $)** avec la stratégie **CRYPTO V4**.

Règles officielles HyroTrader One-Step (octobre-novembre 2026) intégrées :
  - Objectif de profit     : +10.0 % (+1 000 $ sur 10 000 $)
  - Perte max journalière  : -4.0 % (-400 $)
  - Perte max totale       : -8.0 % statique / fixe (plancher fixe à 9 200 $)
  - Jours de trading min   : 3 jours minimum
  - Règle de consistance   : 40 % max par jour (aucune journée UTC ne peut compter pour plus de +4.0 % dans l'objectif de +10 %)
  - Plafond d'exposition   : Notionnel ouvert <= 2.0x le solde (marge <= 25 %)
  - Exécution Bybit/BloFin : Frais réels 0.020 % Maker (LIMIT) / 0.055 % Taker (MARKET / SL) sur BTC, ETH, AVAX, INJ.

Les 5 Règles du Risk Overlay Hyro One-Step intégrées et optimisées :
  1. `hyro_risk_pct`        : Sizing de base par trade (% du capital initial du challenge, calibré pour la règle des 40 %).
  2. `hyro_daily_stop_pct`  : Disjoncteur journalier (-X % réalisé sur la journée UTC -> coupe toute nouvelle entrée jusqu'à minuit UTC)
                              + Garde-fou simultané journalier (réduit/bloque un N-ième trade si le pire cas simultané menace les -4 %).
  3. `hyro_pause_at_cap_pct`: Pause journalière de consistance dès que la journée atteint +Y % (proche du plafond de +4.0 % / 40 %)
                              + Écrêtage dynamique du risque (`hyro_cap_scale`) pour qu'un TP (+6R) ne gaspille pas de profit au-delà de +4.0 %/jour.
  4. `hyro_dd_throttle_pct` : Réduction du risque en Drawdown (dès -D % de drawdown, multiplie le risque par `hyro_dd_mult`).
  5. `hyro_lock_at_pct`     : Verrouillage de fin de challenge (dès +L % de profit validé vers les +10 %, multiplie le risque par `hyro_lock_mult`)
                              + Plafond de positions simultanées (`hyro_max_open_trades`) et notionnel <= 2.0x.

Architecture de l'Optimiseur Multi-Paliers (Walk-Forward TRAIN < 2024-01-01 | TEST >= 2024-01-01) :
  - Palier 1 : Filtres d'Entrée & Anti-Krach SMC sur bougies 5m/15m/4h (`m15_max_range_pct`, `h4_thresh`, `sl_cooldown_hours`, `m15_min_range_pct`)
  - Palier 2 : Gestion de Position, Objectif TP & Paliers de Trailing SL (`tp_roi_pct`, `trail_variant`)
  - Palier 3 : Calibrage fin des 5 Règles Hyro One-Step (`hyro_risk_pct`, `hyro_daily_stop_pct`, `hyro_pause_at_cap_pct`, `hyro_dd_throttle_pct`, `hyro_lock_at_pct`, `hyro_max_open_trades`)
"""
import argparse
import csv
import gzip
import itertools
import json
import math
import os
import statistics
import subprocess
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone

DAY = 86400.0
SL_DIST_DEFAULT = 0.0166667  # 50% ROI / levier 30 = 1.6667%

# Règles officielles HyroTrader One-Step (10k$)
HYRO_ONESTEP_SPEC = {
    "name": "HyroTrader One-Step (10 000 $)",
    "firm": "HyroTrader",
    "account_usd": 10000.0,
    "price_usd": 79.0,
    "profit_split": 0.80,
    "target_pct": 10.0,
    "daily_loss_pct": 4.0,
    "max_dd_pct": 8.0,
    "dd_type": "static",
    "min_days": 3,
    "consistency_day_cap_pct": 4.0,  # 40% de 10% = +4.0% max comptabilisé par jour UTC
    "max_notional_mult": 2.0,        # Exposition notionnelle max = 2x le solde
    "fee_maker_pct": 0.020,          # Bybit/BloFin Maker (LIMIT)
    "fee_taker_pct": 0.055,          # Bybit/BloFin Taker (MARKET / SL)
    "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
}

# Paliers de Trailing SL testables au Palier 2
TRAIL_VARIANTS = {
    "standard": (
        (50.0, 0.0),     # +1R -> BE (0R)
        (100.0, 50.0),   # +2R -> +1R
        (150.0, 100.0),  # +3R -> +2R
        (200.0, 150.0),  # +4R -> +3R
    ),
    "lock_5r": (
        (50.0, 0.0),     # +1R -> BE (0R)
        (100.0, 50.0),   # +2R -> +1R
        (150.0, 100.0),  # +3R -> +2R
        (200.0, 150.0),  # +4R -> +3R
        (250.0, 210.0),  # +5R -> +4.2R (sécurise 84% du gain juste avant le TP +6R)
    ),
    "tight_high": (
        (50.0, 0.0),     # +1R -> BE (0R)
        (100.0, 50.0),   # +2R -> +1R
        (150.0, 105.0),  # +3R -> +2.1R
        (200.0, 160.0),  # +4R -> +3.2R
        (250.0, 215.0),  # +5R -> +4.3R
    ),
}

# Configuration de référence (avant optimisation dédiée Hyro One-Step)
REF_STRATEGY_PARAMS = {
    "leverage": 30.0,
    "tp_roi_pct": 300.0,
    "sl_roi_pct": 50.0,
    "h4_lookback": 8,
    "h4_thresh": 0.35,
    "m15_lookback": 16,
    "m15_min_range_pct": 0.40,
    "m15_max_range_pct": 1.60,
    "sl_cooldown_hours": 2.0,
    "block_same_poi": 1,
    "same_poi_tol_pct": 0.15,
    "poi_fib": 0.50,
    "mkt_tol_pct": 0.05,
    "lim_offset_pct": 0.10,
    "reprice_tol_pct": 0.10,
    "trail_variant": "standard",
}

REF_HYRO_OVERLAY = {
    "hyro_risk_pct": 0.68,           # Règle 1 : 0.68% de risque de base par trade (68$ sur 10k$)
    "hyro_daily_stop_pct": 2.2,      # Règle 2 : Disjoncteur journalier à -2.2% + garde-fou simultané
    "hyro_daily_guard": 1,
    "hyro_pause_at_cap_pct": 3.6,    # Règle 3 : Pause journalière à +3.6% (règle des 40% = +4.0%/j)
    "hyro_cap_scale": 1,             # Écrête dynamiquement le risque si un TP dépasserait +4.0% sur la journée
    "hyro_dd_throttle_pct": 2.8,     # Règle 4 : Dès -2.8% de DD (depuis initial ou HWM), réduit le risque à x0.55
    "hyro_dd_mult": 0.55,
    "hyro_lock_at_pct": 6.5,         # Règle 5 : Dès +6.5% de profit validé, réduit le risque à x0.55
    "hyro_lock_mult": 0.55,
    "hyro_max_open_trades": 4,       # Max 4 positions simultanées (et notionnel <= 2.0x)
    "hyro_max_notional_mult": 2.0,
}


def _parse_iso_ts(s: str) -> float:
    d = datetime.fromisoformat(str(s).strip())
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


# ==============================================================================
# 1. LECTURE ULTRA-RAPIDE DES BOUGIES 5M & PRÉCALCUL DES FENÊTRES 15M / 4H / 30M
# ==============================================================================
def load_5m_fast(sym: str, datadir: str = "data"):
    """Lit `data/<sym>.csv.gz` en ~0.4s via `gzip -dc` (ou fallback gzip Python) et précalcule les grilles 15m/4h."""
    path = os.path.join(datadir, f"{sym}.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(datadir, f"{sym}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    day_cache = {}
    ts5, o5, h5, l5, c5 = [], [], [], [], []

    proc = None
    if path.endswith(".gz"):
        try:
            proc = subprocess.Popen(["gzip", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
            stream = proc.stdout
        except Exception:
            proc = None
            stream = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        stream = open(path, "r", encoding="utf-8", errors="replace")

    try:
        stream.readline()  # en-tête
        for line in stream:
            p = line.split(",")
            if len(p) < 5:
                continue
            s = p[0]
            d_str = s[:10]
            d_ts = day_cache.get(d_str)
            if d_ts is None:
                d_ts = int(datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]), tzinfo=timezone.utc).timestamp())
                day_cache[d_str] = d_ts
            t = d_ts + int(s[11:13]) * 3600 + int(s[14:16]) * 60
            ts5.append(t)
            o5.append(float(p[1]))
            h5.append(float(p[2]))
            l5.append(float(p[3]))
            c5.append(float(p[4]))
    finally:
        if proc is not None:
            proc.stdout.close()
            proc.wait()
        else:
            stream.close()

    # Construction des bougies 15m (3 bougies 5m complètes, close_ts = bucket_start + 900)
    c15_ts, h15, l15, c15 = [], [], [], []
    # Construction des bougies 4h (>= 29 bougies 5m sur 48, close_ts = bucket_start + 14400)
    c4_ts, h4, l4, c4 = [], [], [], []

    n5 = len(ts5)
    # Regroupement 15m en un seul passage linéaire
    cur_b15 = -1
    cnt15 = 0
    hi15 = lo15 = cl15 = 0.0

    cur_b4 = -1
    cnt4 = 0
    hi4 = lo4 = cl4 = 0.0

    for i in range(n5):
        t = ts5[i]
        hi = h5[i]
        lo = l5[i]
        cl = c5[i]

        b15 = (t // 900) * 900
        if b15 != cur_b15:
            if cnt15 >= 3:
                c15_ts.append(cur_b15 + 900)
                h15.append(hi15)
                l15.append(lo15)
                c15.append(cl15)
            cur_b15 = b15
            cnt15 = 1
            hi15 = hi
            lo15 = lo
            cl15 = cl
        else:
            cnt15 += 1
            if hi > hi15:
                hi15 = hi
            if lo < lo15:
                lo15 = lo
            cl15 = cl

        b4 = (t // 14400) * 14400
        if b4 != cur_b4:
            if cnt4 >= 29:
                c4_ts.append(cur_b4 + 14400)
                h4.append(hi4)
                l4.append(lo4)
                c4.append(cl4)
            cur_b4 = b4
            cnt4 = 1
            hi4 = hi
            lo4 = lo
            cl4 = cl
        else:
            cnt4 += 1
            if hi > hi4:
                hi4 = hi
            if lo < lo4:
                lo4 = lo
            cl4 = cl

    if cnt15 >= 3:
        c15_ts.append(cur_b15 + 900)
        h15.append(hi15)
        l15.append(lo15)
        c15.append(cl15)
    if cnt4 >= 29:
        c4_ts.append(cur_b4 + 14400)
        h4.append(hi4)
        l4.append(lo4)
        c4.append(cl4)

    # Précalcul aux points de check 30m (là où (ts5[k] + 300) % 1800 == 0)
    # Pour éviter de recalculer max/min 15m et 4h à chaque essai de l'optimiseur !
    chk_k = []
    chk_biais_h4 = {6: [], 8: [], 10: []}
    chk_m15 = {}  # m15_lb -> list of tuples (rng_ratio, mom_m15, win_lo, after_hi_long, win_hi, after_lo_short)
    for lb in (12, 16, 20):
        chk_m15[lb] = []

    j15 = -1
    n15 = len(c15_ts)
    j4 = -1
    n4 = len(c4_ts)

    for k in range(n5):
        close_t = ts5[k] + 300
        if (close_t % 1800) != 0:
            continue
        while j15 + 1 < n15 and c15_ts[j15 + 1] <= close_t:
            j15 += 1
        while j4 + 1 < n4 and c4_ts[j4 + 1] <= close_t:
            j4 += 1
        if j15 < 20 or j4 < 10:
            continue
        px = c5[k]
        if px <= 0.0:
            continue

        chk_k.append(k)

        for h_lb in (6, 8, 10):
            s0 = j4 - h_lb + 1
            s1 = j4 + 1
            net4 = c4[j4] - c4[s0]
            rng4 = max(h4[s0:s1]) - min(l4[s0:s1])
            chk_biais_h4[h_lb].append((net4 / rng4) if rng4 > 0.0 else 0.0)

        for m_lb in (12, 16, 20):
            s0 = j15 - m_lb + 1
            s1 = j15 + 1
            w_h = h15[s0:s1]
            w_l = l15[s0:s1]
            win_hi = max(w_h)
            win_lo = min(w_l)
            rng15 = win_hi - win_lo
            rng_ratio = rng15 / px
            mom15 = ((c15[j15] - c15[s0]) / rng15) if rng15 > 0.0 else 0.0

            # argmin / argmax
            i_lo = 0
            v_lo = w_l[0]
            i_hi = 0
            v_hi = w_h[0]
            for idx in range(1, m_lb):
                if w_l[idx] < v_lo:
                    v_lo = w_l[idx]
                    i_lo = idx
                if w_h[idx] > v_hi:
                    v_hi = w_h[idx]
                    i_hi = idx
            after_hi = max(w_h[i_lo:])
            after_lo = min(w_l[i_hi:])
            chk_m15[m_lb].append((rng_ratio, mom15, win_lo, after_hi, win_hi, after_lo))

    return {
        "symbol": sym,
        "ts5": ts5,
        "o5": o5,
        "h5": h5,
        "l5": l5,
        "c5": c5,
        "chk_k": chk_k,
        "chk_biais_h4": chk_biais_h4,
        "chk_m15": chk_m15,
    }


def simulate_symbol_trades(data: dict, sp: dict) -> list:
    """
    Simule exactement la logique CRYPTO V4 sur un instrument à partir des tables 5m + checkpoints 30m précalculés.
    Exécution en ~0.025s par crypto (identique au moteur numpy `strategy_crypto_v4.py`).
    """
    sym = data["symbol"]
    ts5 = data["ts5"]
    o5 = data["o5"]
    h5 = data["h5"]
    l5 = data["l5"]
    c5 = data["c5"]
    chk_k = data["chk_k"]

    lev = float(sp.get("leverage", 30.0))
    tp_roi_pct = float(sp.get("tp_roi_pct", 300.0))
    sl_roi_pct = float(sp.get("sl_roi_pct", 50.0))
    one_r_roi = sl_roi_pct if sl_roi_pct > 0 else 50.0

    h4_lb = int(sp.get("h4_lookback", 8))
    h4_thresh = float(sp.get("h4_thresh", 0.35))
    m15_lb = int(sp.get("m15_lookback", 16))
    m15_min_rng = float(sp.get("m15_min_range_pct", 0.40)) / 100.0
    m15_max_rng = float(sp.get("m15_max_range_pct", 1.60)) / 100.0
    cd_bars_5m = int(round(float(sp.get("sl_cooldown_hours", 2.0)) * 12.0))
    block_same_poi = int(float(sp.get("block_same_poi", 1))) != 0
    same_poi_tol = float(sp.get("same_poi_tol_pct", 0.15)) / 100.0
    poi_fib = float(sp.get("poi_fib", 0.50))
    mkt_tol = float(sp.get("mkt_tol_pct", 0.05)) / 100.0
    lim_off = float(sp.get("lim_offset_pct", 0.10)) / 100.0
    reprice_tol = float(sp.get("reprice_tol_pct", 0.10)) / 100.0
    trail_tiers = TRAIL_VARIANTS.get(sp.get("trail_variant", "standard"), TRAIL_VARIANTS["standard"])

    biais_vec = data["chk_biais_h4"][h4_lb]
    m15_vec = data["chk_m15"][m15_lb]

    sl_frac = (sl_roi_pct / 100.0) / lev
    tp_frac = (tp_roi_pct / 100.0) / lev

    pos = None
    order = None
    trades = []
    last_sl_idx = -999999
    last_sl_poi = -1.0

    def record_close(pos_d, exit_idx, exit_px, reason_str):
        nonlocal last_sl_idx, last_sl_poi
        side = pos_d["side"]
        en = pos_d["entry"]
        move = ((exit_px - en) / en) if side == 1 else ((en - exit_px) / en)
        roi_gross_pct = lev * move * 100.0
        # Frais standard 0.02% par ordre dans le R par défaut (ajusté ensuite aux frais exacts Bybit Maker/Taker)
        roi_net_default = roi_gross_pct - (2.0 * 0.0002 * lev * 100.0)
        r_gross = roi_gross_pct / one_r_roi
        r_net_default = roi_net_default / one_r_roi
        if r_gross < -0.1:
            last_sl_idx = exit_idx
            last_sl_poi = pos_d["poi"]

        fill_i = pos_d["fill_i"]
        t_in = float(ts5[fill_i])
        t_out = float(ts5[exit_idx])
        if t_out <= t_in:
            t_out = t_in + 1.0
        trades.append({
            "t_in": t_in,
            "t_out": t_out,
            "symbol": sym,
            "side": "long" if side == 1 else "short",
            "entry": en,
            "exit": exit_px,
            "initial_sl": pos_d["initial_sl"],
            "sl_pct": sl_frac,
            "r_gross": r_gross,
            "r_net_default": r_net_default,
            "entry_type": pos_d["entry_type"],
            "reason": reason_str,
        })

    n_chk = len(chk_k)
    prev_k = chk_k[0] if n_chk > 0 else 0

    for c_idx in range(n_chk):
        k_target = chk_k[c_idx]

        # Si une position ou un ordre limite est actif entre prev_k + 1 et k_target, on parcourt les bougies 5m intermédiaires
        if pos is not None or order is not None:
            for k in range(prev_k + 1, k_target):
                if pos is not None:
                    side = pos["side"]
                    sl = pos["sl"]
                    tp = pos["tp"]
                    hk = h5[k]
                    lk = l5[k]
                    hit_sl = (lk <= sl) if side == 1 else (hk >= sl)
                    hit_tp = (hk >= tp) if side == 1 else (lk <= tp)
                    if hit_sl or hit_tp:
                        ok_p = o5[k]
                        tier = pos["tier"]
                        if hit_sl:
                            xp = (sl if sl < ok_p else ok_p) if (side == 1 and k > pos["fill_i"]) else (
                                 (sl if sl > ok_p else ok_p) if (side == -1 and k > pos["fill_i"]) else sl)
                            reason = "SL (-50%)" if tier == 0 else ("BE (0%)" if tier == 1 else f"TRAIL (+{int(trail_tiers[tier - 1][1])}%)")
                        else:
                            xp = (tp if tp > ok_p else ok_p) if (side == 1 and k > pos["fill_i"]) else (
                                 (tp if tp < ok_p else ok_p) if (side == -1 and k > pos["fill_i"]) else tp)
                            reason = f"TP (+{int(tp_roi_pct)}%)"
                        record_close(pos, k, xp, reason)
                        pos = None
                        break
                elif order is not None:
                    side = order["side"]
                    lim = order["limit"]
                    hk = h5[k]
                    lk = l5[k]
                    if (lk <= lim) if side == 1 else (hk >= lim):
                        en = lim
                        init_sl = en * (1.0 - sl_frac) if side == 1 else en * (1.0 + sl_frac)
                        init_tp = en * (1.0 + tp_frac) if side == 1 else en * (1.0 - tp_frac)
                        pos = {
                            "side": side, "entry": en, "sl": init_sl, "initial_sl": init_sl,
                            "tp": init_tp, "tier": 0, "fill_i": k, "poi": order["poi"], "entry_type": "LIMIT",
                        }
                        order = None
                        if (side == 1 and lk <= init_sl) or (side == -1 and hk >= init_sl):
                            record_close(pos, k, init_sl, "SL (-50%)")
                            pos = None
                            break

        # Maintenant on traite la bougie du checkpoint k = k_target
        k = k_target
        prev_k = k
        hk = h5[k]
        lk = l5[k]
        ck = c5[k]
        ok_p = o5[k]

        if pos is not None:
            side = pos["side"]
            sl = pos["sl"]
            tp = pos["tp"]
            hit_sl = (lk <= sl) if side == 1 else (hk >= sl)
            hit_tp = (hk >= tp) if side == 1 else (lk <= tp)
            if hit_sl or hit_tp:
                tier = pos["tier"]
                if hit_sl:
                    xp = (sl if sl < ok_p else ok_p) if (side == 1 and k > pos["fill_i"]) else (
                         (sl if sl > ok_p else ok_p) if (side == -1 and k > pos["fill_i"]) else sl)
                    reason = "SL (-50%)" if tier == 0 else ("BE (0%)" if tier == 1 else f"TRAIL (+{int(trail_tiers[tier - 1][1])}%)")
                else:
                    xp = (tp if tp > ok_p else ok_p) if (side == 1 and k > pos["fill_i"]) else (
                         (tp if tp < ok_p else ok_p) if (side == -1 and k > pos["fill_i"]) else tp)
                    reason = f"TP (+{int(tp_roi_pct)}%)"
                record_close(pos, k, xp, reason)
                pos = None
        elif order is not None:
            side = order["side"]
            lim = order["limit"]
            if (lk <= lim) if side == 1 else (hk >= lim):
                en = lim
                init_sl = en * (1.0 - sl_frac) if side == 1 else en * (1.0 + sl_frac)
                init_tp = en * (1.0 + tp_frac) if side == 1 else en * (1.0 - tp_frac)
                pos = {
                    "side": side, "entry": en, "sl": init_sl, "initial_sl": init_sl,
                    "tp": init_tp, "tier": 0, "fill_i": k, "poi": order["poi"], "entry_type": "LIMIT",
                }
                order = None
                if (side == 1 and lk <= init_sl) or (side == -1 and hk >= init_sl):
                    record_close(pos, k, init_sl, "SL (-50%)")
                    pos = None

        # Étape 2 : Gérer la position ouverte au check 30m (Trailing SL)
        if pos is not None:
            side = pos["side"]
            en = pos["entry"]
            move = ((ck - en) / en) if side == 1 else ((en - ck) / en)
            roi_cur = lev * move * 100.0
            for tier_idx, (trig_roi, lock_roi) in enumerate(trail_tiers, 1):
                if roi_cur >= trig_roi and tier_idx > pos["tier"]:
                    pos["tier"] = tier_idx
                    lock_frac = (lock_roi / 100.0) / lev
                    pos["sl"] = en * (1.0 + lock_frac) if side == 1 else en * (1.0 - lock_frac)
            continue

        # Filtre Cooldown après SL
        if cd_bars_5m > 0 and (k - last_sl_idx) < cd_bars_5m:
            order = None
            continue

        # Étape 3 : Détection & Ré-analyse
        biais_h4 = biais_vec[c_idx]
        rng_m15_ratio, mom_m15, win_lo, after_hi, win_hi, after_lo = m15_vec[c_idx]

        if rng_m15_ratio < m15_min_rng or (m15_max_rng > 0.0 and rng_m15_ratio > m15_max_rng):
            order = None
            continue

        if biais_h4 > h4_thresh:
            direction = 1
        elif biais_h4 < -h4_thresh:
            direction = -1
        else:
            direction = 1 if mom_m15 > 0.0 else (-1 if mom_m15 < 0.0 else 0)

        if direction == 0:
            order = None
            continue

        if direction == 1:
            poi = after_hi - poi_fib * (after_hi - win_lo)
            if block_same_poi and last_sl_poi > 0.0 and abs(poi - last_sl_poi) / last_sl_poi < same_poi_tol:
                order = None
                continue
            if not (poi < ck):
                order = None
                continue
            if ck <= poi * (1.0 + mkt_tol):
                order = None
                en = ck
                init_sl = en * (1.0 - sl_frac)
                init_tp = en * (1.0 + tp_frac)
                pos = {
                    "side": 1, "entry": en, "sl": init_sl, "initial_sl": init_sl,
                    "tp": init_tp, "tier": 0, "fill_i": k, "poi": poi, "entry_type": "MARKET",
                }
            else:
                new_lim = poi * (1.0 - lim_off)
                if order is None or order["side"] != 1 or abs(new_lim - order["limit"]) / order["limit"] > reprice_tol:
                    order = {"side": 1, "limit": new_lim, "poi": poi}
        else:
            poi = after_lo + poi_fib * (win_hi - after_lo)
            if block_same_poi and last_sl_poi > 0.0 and abs(poi - last_sl_poi) / last_sl_poi < same_poi_tol:
                order = None
                continue
            if not (poi > ck):
                order = None
                continue
            if ck >= poi * (1.0 - mkt_tol):
                order = None
                en = ck
                init_sl = en * (1.0 + sl_frac)
                init_tp = en * (1.0 - tp_frac)
                pos = {
                    "side": -1, "entry": en, "sl": init_sl, "initial_sl": init_sl,
                    "tp": init_tp, "tier": 0, "fill_i": k, "poi": poi, "entry_type": "MARKET",
                }
            else:
                new_lim = poi * (1.0 + lim_off)
                if order is None or order["side"] != -1 or abs(new_lim - order["limit"]) / order["limit"] > reprice_tol:
                    order = {"side": -1, "limit": new_lim, "poi": poi}

    return trades


# ==============================================================================
# 2. MOTEUR DE SIMULATION HYROTRADER ONE-STEP (LES 5 RÈGLES INTÉGRÉES)
# ==============================================================================
def prepare_hyro_events(raw_trades: list, spec: dict = None):
    """Ajuste chaque trade aux frais réels Bybit/BloFin de HyroTrader (0.020% Maker / 0.055% Taker) et construit l'échéancier."""
    spec = HYRO_ONESTEP_SPEC if spec is None else spec
    f_maker = spec["fee_maker_pct"] / 100.0
    f_taker = spec["fee_taker_pct"] / 100.0
    allowed = set(spec["symbols"])

    trades = []
    for t in raw_trades:
        if t["symbol"] not in allowed:
            continue
        fee_in = f_maker if t["entry_type"] == "LIMIT" else f_taker
        fee_out = f_maker if str(t["reason"]).startswith("TP") else f_taker
        extra_fee = max(0.0, (fee_in + fee_out - 0.0004) * 0.5)
        sl_pct = t["sl_pct"] if t["sl_pct"] > 0 else SL_DIST_DEFAULT
        r_net = t["r_net_default"] - (extra_fee / sl_pct)
        trades.append((t["t_in"], t["t_out"], r_net, sl_pct, t["symbol"], t["entry_type"], t["reason"]))

    trades.sort(key=lambda x: (x[0], x[1]))
    events = []
    for i, (t_in, t_out, *_rest) in enumerate(trades):
        events.append((t_in, 1, i))
        events.append((t_out, 0, i))
    events.sort()
    ev_ts = [e[0] for e in events]
    loss_mult = max(1.0, max((-t[2] for t in trades), default=1.0))
    return trades, events, ev_ts, loss_mult


def run_hyro_onestep_challenge(
    events: list,
    ev_ts: list,
    trades: list,
    loss_mult: float,
    start_ts: float,
    overlay: dict,
    horizon_days: float = 180.0,
    floating_mode: str = "worst",
    spec: dict = None,
):
    """
    Simule un challenge HyroTrader One-Step complet démarré à `start_ts` en appliquant
    strictement les règles officielles Hyro One-Step ET les 5 règles du Risk Overlay :
      - Objectif : +10.0% (après règle de consistance des 40%/jour, soit max +4.0% comptabilisé par jour UTC)
      - Perte max journalière : -4.0%
      - Perte max totale : -8.0% fixe (statique depuis le capital initial)
      - Minimum : 3 jours de trading
      - Plafond notionnel : 2.0x le solde
      - Les 5 Règles Risk Overlay (`hyro_risk_pct`, `hyro_daily_stop_pct` + `hyro_daily_guard`,
        `hyro_pause_at_cap_pct` + `hyro_cap_scale`, `hyro_dd_throttle_pct` + `hyro_dd_mult`,
        `hyro_lock_at_pct` + `hyro_lock_mult` + `hyro_max_open_trades`).
    Retourne un dictionnaire détaillé du challenge (`status`, `days`, `end_ts`, `final_bal`, `credited_profit`, `max_dd`, `worst_day`, `trades_taken`, `trades_skipped`).
    """
    spec = HYRO_ONESTEP_SPEC if spec is None else spec
    tgt_pct = spec["target_pct"] / 100.0               # +10.0%
    daily_limit = spec["daily_loss_pct"] / 100.0       # -4.0%
    total_limit = spec["max_dd_pct"] / 100.0           # -8.0% fixe
    min_days = int(spec["min_days"])                   # 3 jours min
    max_day_cap = spec["consistency_day_cap_pct"] / 100.0  # +4.0% max comptabilisé par jour UTC

    # Les 5 règles du Risk Overlay Hyro One-Step
    rk_base = float(overlay.get("hyro_risk_pct", 0.65)) / 100.0
    daily_stop = float(overlay.get("hyro_daily_stop_pct", 2.0)) / 100.0
    use_daily_guard = int(overlay.get("hyro_daily_guard", 1)) != 0
    pause_at_cap = float(overlay.get("hyro_pause_at_cap_pct", 3.8)) / 100.0
    use_cap_scale = int(overlay.get("hyro_cap_scale", 1)) != 0
    dd_thr_pct = float(overlay.get("hyro_dd_throttle_pct", 3.6)) / 100.0
    dd_mult = float(overlay.get("hyro_dd_mult", 0.55))
    lock_at_pct = float(overlay.get("hyro_lock_at_pct", 7.0)) / 100.0
    lock_mult = float(overlay.get("hyro_lock_mult", 0.55))
    max_open_trades = int(overlay.get("hyro_max_open_trades", 3))
    max_notional = float(overlay.get("hyro_max_notional_mult", spec["max_notional_mult"]))

    t_end = start_ts + horizon_days * DAY
    k = bisect_left(ev_ts, start_ts)
    n = len(events)

    bal = 1.0
    hwm = 1.0
    max_dd = 0.0
    worst_day_pnl = 0.0
    cur_day = int(start_ts // DAY)
    day_start = 1.0
    day_pnl = {}
    days_traded = set()
    open_r = {}
    taken = 0
    skipped = 0
    last_ts = start_ts

    while k < n:
        ts, typ, i = events[k]
        k += 1
        if ts >= t_end:
            break
        last_ts = ts
        d = int(ts // DAY)
        if d != cur_day:
            cur_day = d
            day_start = bal

        daily_floor = day_start - daily_limit
        total_floor = 1.0 - total_limit  # Perte max 8% FIXE sur capital initial
        floor = daily_floor if daily_floor > total_floor else total_floor

        excess_days = sum(max(0.0, p - max_day_cap) for p in day_pnl.values())
        credited_profit = (bal - 1.0) - excess_days

        if typ == 1:  # Ouverture d'un trade
            # Règle 5b : Plafond de positions ouvertes simultanées
            if max_open_trades > 0 and len(open_r) >= max_open_trades:
                skipped += 1
                continue

            # Règle 2a : Disjoncteur journalier (-hyro_daily_stop_pct)
            if daily_stop > 0.0 and (bal - day_start) <= -daily_stop:
                skipped += 1
                continue

            # Règle 3a : Pause journalière de consistance (40% = +4.0% max par jour UTC)
            cur_day_gain = day_pnl.get(d, 0.0)
            if pause_at_cap > 0.0 and cur_day_gain >= pause_at_cap:
                skipped += 1
                continue

            # Règle 1 : Risque de base par trade
            risk = rk_base

            # Règle 3b : Écrêtage dynamique pour ne pas dépasser inutilement le plafond journalier de +4.0% (40% de 10%)
            if use_cap_scale and max_day_cap > 0.0:
                rem_day_cap = max(0.0, max_day_cap - max(0.0, cur_day_gain))
                if risk * 5.5 > rem_day_cap:
                    risk = max(0.45 * rk_base, rem_day_cap / 5.5)

            # Règle 4 : Réduction du risque en Drawdown (protège à la fois le capital initial ET les gains accumulés depuis HWM)
            dd_static = max(0.0, 1.0 - bal)
            dd_from_hwm = max(0.0, hwm - bal)
            if dd_thr_pct > 0.0 and max(dd_static, dd_from_hwm) >= dd_thr_pct:
                risk *= dd_mult

            # Règle 5a : Verrouillage de fin de challenge à l'approche des +10%
            if lock_at_pct > 0.0 and credited_profit >= lock_at_pct:
                risk *= lock_mult

            # Règle 5c : Plafond de notionnel simultané (<= 2.0x le solde)
            sl_pct = trades[i][3]
            open_notional = sum(r_val / trades[idx][3] for idx, r_val in open_r.items())
            rem_notional = max(0.0, max_notional * bal - open_notional)
            if risk / sl_pct > rem_notional:
                risk = rem_notional * sl_pct
            if risk < 0.15 * rk_base:
                skipped += 1
                continue

            # Règle 2b : Garde-fou journalier simultané (protège la limite journalière de -4.0% si plusieurs trades sont ouverts)
            open_adv = sum(r_val * (loss_mult if trades[idx][2] < -0.1 else 0.55) for idx, r_val in open_r.items())
            if use_daily_guard and open_r:
                daily_buf = (bal - open_adv - daily_floor - 0.0015) / loss_mult
                if daily_buf < risk:
                    risk = daily_buf
                if risk < 0.25 * rk_base:
                    skipped += 1
                    continue

            this_adv = risk * (loss_mult if trades[i][2] < -0.1 else 0.55)
            if floating_mode == "worst" and (bal - (open_adv + this_adv)) < floor:
                return {
                    "status": "fail",
                    "fail_reason": "daily_loss" if daily_floor >= total_floor else "max_dd",
                    "days": (ts - start_ts) / DAY,
                    "end_ts": ts,
                    "final_bal": bal - (open_adv + this_adv),
                    "credited_profit": credited_profit,
                    "max_dd": max(max_dd, 1.0 - (bal - (open_adv + this_adv))),
                    "worst_day": min(worst_day_pnl, (bal - (open_adv + this_adv)) - day_start),
                    "trades_taken": taken + 1,
                    "trades_skipped": skipped,
                }

            open_r[i] = risk
            days_traded.add(d)
            taken += 1

        else:  # Clôture d'un trade
            risk = open_r.pop(i, None)
            if risk is None:
                continue
            pnl = risk * trades[i][2]
            bal += pnl
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            if day_pnl[d] < worst_day_pnl:
                worst_day_pnl = day_pnl[d]
            if bal > hwm:
                hwm = bal
            cur_dd = max(0.0, 1.0 - bal)
            if cur_dd > max_dd:
                max_dd = cur_dd
            days_traded.add(d)

            if bal < total_floor or bal < daily_floor:
                return {
                    "status": "fail",
                    "fail_reason": "daily_loss" if bal < daily_floor else "max_dd",
                    "days": (ts - start_ts) / DAY,
                    "end_ts": ts,
                    "final_bal": bal,
                    "credited_profit": (bal - 1.0),
                    "max_dd": max_dd,
                    "worst_day": worst_day_pnl,
                    "trades_taken": taken,
                    "trades_skipped": skipped,
                }

            excess_days = sum(max(0.0, p - max_day_cap) for p in day_pnl.values())
            credited_profit = (bal - 1.0) - excess_days

            if (bal - 1.0) >= tgt_pct and credited_profit >= tgt_pct and len(days_traded) >= min_days:
                return {
                    "status": "pass",
                    "fail_reason": "",
                    "days": (ts - start_ts) / DAY,
                    "end_ts": ts,
                    "final_bal": bal,
                    "credited_profit": credited_profit,
                    "max_dd": max_dd,
                    "worst_day": worst_day_pnl,
                    "trades_taken": taken,
                    "trades_skipped": skipped,
                }

    excess_days = sum(max(0.0, p - max_day_cap) for p in day_pnl.values())
    credited_profit = (bal - 1.0) - excess_days
    return {
        "status": "open",
        "fail_reason": "",
        "days": horizon_days,
        "end_ts": last_ts,
        "final_bal": bal,
        "credited_profit": credited_profit,
        "max_dd": max_dd,
        "worst_day": worst_day_pnl,
        "trades_taken": taken,
        "trades_skipped": skipped,
    }


def summarize_attempts(attempts: list, price_usd: float = 79.0) -> dict:
    n = len(attempts)
    if n == 0:
        return {
            "n": 0, "pass": 0.0, "fail": 0.0, "open": 0.0, "pass_resolved": 0.0,
            "p30": 0.0, "p45": 0.0, "p60": 0.0, "p90": 0.0,
            "med_days": 999.0, "p75_days": 999.0, "exp_cost_usd": 9999.0,
            "max_dd_med": 0.0, "worst_day_min": 0.0, "score": -999.0,
        }
    passes = [a["days"] for a in attempts if a["status"] == "pass"]
    fails = sum(1 for a in attempts if a["status"] == "fail")
    opens = sum(1 for a in attempts if a["status"] == "open")

    pass_rate = len(passes) / n
    fail_rate = fails / n
    resolved = len(passes) + fails
    pass_resolved = (len(passes) / resolved) if resolved > 0 else 0.0
    exp_cost_usd = (price_usd / pass_resolved) if pass_resolved > 0 else 9999.0

    def within(x):
        return sum(1 for d in passes if d <= x) / n

    med_days = statistics.median(passes) if passes else 180.0
    p75_days = sorted(passes)[int(0.75 * (len(passes) - 1))] if passes else 180.0
    max_dd_med = statistics.median(a["max_dd"] for a in attempts)
    worst_day_min = min(a["worst_day"] for a in attempts)

    # Score composite WFA centré sur la réussite réelle du challenge Hyro One-Step :
    #   + Taux de réussite global (pass_rate)
    #   + Taux de réussite des challenges résolus (pass_resolved)
    #   + Vitesse de validation (<= 45j et <= 60j)
    #   - Pénalité forte sur le taux d'échec (fail_rate)
    score = (
        pass_rate * 100.0
        + pass_resolved * 45.0
        + within(45) * 25.0
        + within(60) * 15.0
        - fail_rate * 85.0
        - max(0.0, med_days - 35.0) * 0.25
    )

    return {
        "n": n,
        "pass": pass_rate,
        "fail": fail_rate,
        "open": opens / n,
        "pass_resolved": pass_resolved,
        "p30": within(30),
        "p45": within(45),
        "p60": within(60),
        "p90": within(90),
        "med_days": med_days,
        "p75_days": p75_days,
        "exp_cost_usd": exp_cost_usd,
        "max_dd_med": max_dd_med,
        "worst_day_min": worst_day_min,
        "score": score,
    }


def simulate_sequential_campaigns(
    events: list,
    ev_ts: list,
    trades: list,
    loss_mult: float,
    start_ts: float,
    end_ts: float,
    overlay: dict,
    spec: dict = None,
):
    """
    Simule en continu (bout-à-bout) sur une période (ex: les 4 dernières années 2022-2026) :
    - On démarre un challenge Hyro One-Step 10 000 $ à `start_ts`.
    - Dès qu'un challenge est VALIDÉ (+10%) ou ÉCHOUÉ, on en relance immédiatement un nouveau le jour suivant,
      et en parallèle, chaque challenge validé devient un Compte Financé (split 80%) ou un Payout direct de +1 000 $ !
    """
    spec = HYRO_ONESTEP_SPEC if spec is None else spec
    cur_ts = start_ts
    campaigns = []
    while cur_ts + 5 * DAY < end_ts:
        rem_days = (end_ts - cur_ts) / DAY
        res = run_hyro_onestep_challenge(
            events, ev_ts, trades, loss_mult, cur_ts, overlay,
            horizon_days=rem_days, floating_mode="worst", spec=spec,
        )
        res["start_ts"] = cur_ts
        campaigns.append(res)
        if res["status"] == "open" and res["end_ts"] >= end_ts - 2 * DAY:
            break
        # On relance le challenge suivant 6 heures après la clôture du précédent
        cur_ts = max(cur_ts + DAY, res["end_ts"] + 6 * 3600.0)
    return campaigns


# ==============================================================================
# 3. OPTIMISEUR MULTI-PALIERS WALK-FORWARD (TRAIN < 2024-01-01 | TEST >= 2024-01-01)
# ==============================================================================
def fmt_st(st: dict) -> str:
    return (
        f"Score {st['score']:+6.1f} | Réussite {st['pass']*100:5.1f}% (résolus {st['pass_resolved']*100:5.1f}%) | "
        f"Échec {st['fail']*100:4.1f}% | <=30j {st['p30']*100:4.1f}% <=45j {st['p45']*100:4.1f}% <=60j {st['p60']*100:4.1f}% | "
        f"Médiane {st['med_days']:4.1f}j | Coût espéré {st['exp_cost_usd']:4.0f}$"
    )


def main():
    ap = argparse.ArgumentParser(description="Optimiseur Multi-Paliers & Simulation 4 Ans HyroTrader One-Step pour CRYPTO V4")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--split", default="2024-01-01", help="Date de coupure Walk-Forward TRAIN (< split) / TEST (>= split)")
    ap.add_argument("--sim_start", default="2022-01-01", help="Début de la simulation détaillée sur les 4 dernières années")
    ap.add_argument("--horizon", type=float, default=180.0, help="Horizon max d'un challenge en jours (illimité chez Hyro)")
    ap.add_argument("--step_days", type=float, default=4.0, help="Espacement des départs glissants en jours")
    ap.add_argument("--skip_p1_p2", type=int, default=0, help="1 = utilise directement les gagnants Palier 1 & 2 déjà validés pour accélérer le Palier 3")
    ap.add_argument("--eval_only", type=int, default=0, help="1 = évalue instantanément la configuration optimisée sur 4 ans sans relancer la grille")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    syms = HYRO_ONESTEP_SPEC["symbols"]

    print(f"⚡ Chargement des bougies 5m locales ({', '.join(syms)}) et précalcul des grilles 15m / 4h / 30m...", flush=True)
    market_data = {}
    for s in syms:
        t_s = time.time()
        market_data[s] = load_5m_fast(s, args.datadir)
        print(f"  ✅ {s:7s} : {len(market_data[s]['ts5']):7,d} bougies 5m ({len(market_data[s]['chk_k']):6,d} checks 30m) en {time.time() - t_s:.2f}s", flush=True)

    # Évaluation d'une paire (strategy_params, hyro_overlay)
    first_ts = min(market_data[s]["ts5"][0] for s in syms)
    last_ts = max(market_data[s]["ts5"][-1] for s in syms)
    split_ts = _parse_iso_ts(args.split)
    sim4y_ts = _parse_iso_ts(args.sim_start)

    all_starts = []
    cur = first_ts + 7 * DAY
    while cur + args.horizon * DAY <= last_ts:
        all_starts.append(cur)
        cur += args.step_days * DAY

    train_starts = [s for s in all_starts if s < split_ts]
    test_starts = [s for s in all_starts if s >= split_ts]
    sim4y_starts = [s for s in all_starts if s >= sim4y_ts]

    # Cache des trades par signature de paramètres stratégie pour ne jamais recalculer 2 fois les mêmes bougies
    trades_cache = {}

    def get_prepared_trades(sp: dict):
        key = (
            sp["m15_max_range_pct"], sp["m15_min_range_pct"], sp["h4_thresh"],
            sp["sl_cooldown_hours"], sp["same_poi_tol_pct"], sp["tp_roi_pct"], sp["trail_variant"]
        )
        if key not in trades_cache:
            raw = []
            for s in syms:
                raw.extend(simulate_symbol_trades(market_data[s], sp))
            trades_cache[key] = prepare_hyro_events(raw, HYRO_ONESTEP_SPEC)
        return trades_cache[key]

    def evaluate_combo(sp: dict, ov: dict, starts_list: list) -> dict:
        trades, events, ev_ts, loss_mult = get_prepared_trades(sp)
        att = [
            run_hyro_onestep_challenge(events, ev_ts, trades, loss_mult, s0, ov, args.horizon, "worst", HYRO_ONESTEP_SPEC)
            for s0 in starts_list
        ]
        return summarize_attempts(att, HYRO_ONESTEP_SPEC["price_usd"])

    # 0) Référence avant optimisation (paramètres initiaux avant WFA)
    base_ref_ov = {
        "hyro_risk_pct": 0.65,
        "hyro_daily_stop_pct": 2.0,
        "hyro_daily_guard": 1,
        "hyro_pause_at_cap_pct": 3.8,
        "hyro_cap_scale": 1,
        "hyro_dd_throttle_pct": 3.6,
        "hyro_dd_mult": 0.55,
        "hyro_lock_at_pct": 7.0,
        "hyro_lock_mult": 0.55,
        "hyro_max_open_trades": 3,
        "hyro_max_notional_mult": 2.0,
    }
    cur_sp = dict(REF_STRATEGY_PARAMS)
    cur_ov = dict(base_ref_ov)
    ref_train = evaluate_combo(cur_sp, cur_ov, train_starts)
    ref_test = evaluate_combo(cur_sp, cur_ov, test_starts)
    ref_4y = evaluate_combo(cur_sp, cur_ov, sim4y_starts)

    print("\n" + "=" * 95)
    print("🚀 OPTIMISATION MULTI-PALIERS HYROTRADER ONE-STEP (Walk-Forward TRAIN < 2024-01-01 | TEST >= 2024-01-01)")
    print("=" * 95)
    print(f"  • Réf. TRAIN (<2024) : {fmt_st(ref_train)}")
    print(f"  • Réf. TEST (>=2024) : {fmt_st(ref_test)}")
    print(f"  • Réf. 4 ANS (22-26) : {fmt_st(ref_4y)}")

    if args.eval_only:
        cur_sp = dict(
            cur_sp,
            m15_max_range_pct=1.40,
            h4_thresh=0.35,
            sl_cooldown_hours=3.0,
            m15_min_range_pct=0.35,
            tp_roi_pct=300.0,
            trail_variant="tight_high",
            same_poi_tol_pct=0.22,
        )
        cur_ov = dict(REF_HYRO_OVERLAY)
    elif args.skip_p1_p2:
        cur_sp = dict(
            cur_sp,
            m15_max_range_pct=1.40,
            h4_thresh=0.35,
            sl_cooldown_hours=3.0,
            m15_min_range_pct=0.35,
            tp_roi_pct=300.0,
            trail_variant="tight_high",
            same_poi_tol_pct=0.22,
        )
        p2_train = evaluate_combo(cur_sp, cur_ov, train_starts)
        print(
            f"\n--- Paliers 1 & 2 (pré-validés) : m15_max=1.40%, m15_min=0.35%, h4=0.35, cooldown=3.0h, trail=tight_high, poi_tol=0.22% ---\n"
            f"     TRAIN : {fmt_st(p2_train)}",
            flush=True,
        )
    else:
        # --------------------------------------------------------------------------
        # PALIER 1 : Filtres d'Entrée & Anti-Krach SMC (bougies 5m/15m/4h)
        # --------------------------------------------------------------------------
        print("\n--- Palier 1/3 : Filtres d'Entrée & Anti-Krach SMC (m15_max_range_pct, h4_thresh, sl_cooldown_hours, m15_min_range_pct) ---", flush=True)
        p1_grid = list(itertools.product(
            [1.30, 1.40, 1.50, 1.60, 1.75],  # m15_max_range_pct
            [0.30, 0.35, 0.40],              # h4_thresh
            [1.5, 2.0, 2.5, 3.0],            # sl_cooldown_hours
            [0.35, 0.40],                    # m15_min_range_pct
        ))
        p1_scores = {}
        best_p1_sp = dict(cur_sp)
        best_p1_score = ref_train["score"]

        for idx, (m_max, h_th, cd_h, m_min) in enumerate(p1_grid, 1):
            cand_sp = dict(cur_sp, m15_max_range_pct=m_max, h4_thresh=h_th, sl_cooldown_hours=cd_h, m15_min_range_pct=m_min)
            st = evaluate_combo(cand_sp, cur_ov, train_starts)
            p1_scores[(m_max, h_th, cd_h, m_min)] = (st, cand_sp)
            if idx % 30 == 0 or idx == len(p1_grid):
                print(f"  [Palier 1 | {idx:3d}/{len(p1_grid)} essais] meilleur score TRAIN = {max(best_p1_score, st['score']):+.1f}", flush=True)
            if st["score"] > best_p1_score:
                best_p1_score = st["score"]

        # Sélection par Score de Plateau sur TRAIN (moyenne du candidat et de ses voisins proches pour interdire tout pic isolé)
        best_p1_plateau = ref_train["score"]
        for (m_max, h_th, cd_h, m_min), (st, cand_sp) in p1_scores.items():
            neigh = [
                st2["score"]
                for (m2, h2, c2, mn2), (st2, _) in p1_scores.items()
                if abs(m2 - m_max) <= 0.16 and abs(h2 - h_th) <= 0.06 and abs(c2 - cd_h) <= 0.6 and abs(mn2 - m_min) <= 0.06
            ]
            plateau_sc = 0.55 * st["score"] + 0.45 * (sum(neigh) / len(neigh))
            if plateau_sc > best_p1_plateau + 0.3 and st["fail"] <= ref_train["fail"] + 0.01:
                best_p1_plateau = plateau_sc
                best_p1_sp = cand_sp

        cur_sp = best_p1_sp
        p1_train = evaluate_combo(cur_sp, cur_ov, train_starts)
        print(
            f"  ✅ Palier 1 retenu : m15_max_range_pct={cur_sp['m15_max_range_pct']}%, h4_thresh={cur_sp['h4_thresh']}, "
            f"sl_cooldown_hours={cur_sp['sl_cooldown_hours']}h, m15_min_range_pct={cur_sp['m15_min_range_pct']}%\n"
            f"     TRAIN : {fmt_st(p1_train)}",
            flush=True,
        )

        # --------------------------------------------------------------------------
        # PALIER 2 : Gestion de Position, TP & Paliers de Trailing SL
        # --------------------------------------------------------------------------
        print("\n--- Palier 2/3 : Objectif TP & Paliers de Trailing SL (tp_roi_pct, trail_variant, same_poi_tol_pct) ---", flush=True)
        p2_grid = list(itertools.product(
            [275.0, 300.0, 325.0],                      # tp_roi_pct (+5.5R, +6.0R, +6.5R)
            ["standard", "lock_5r", "tight_high"],      # trail_variant
            [0.15, 0.22],                               # same_poi_tol_pct
        ))
        best_p2_sp = dict(cur_sp)
        best_p2_score = p1_train["score"]
        for tp_v, tr_v, poi_tol in p2_grid:
            cand_sp = dict(cur_sp, tp_roi_pct=tp_v, trail_variant=tr_v, same_poi_tol_pct=poi_tol)
            st = evaluate_combo(cand_sp, cur_ov, train_starts)
            if st["score"] > best_p2_score + 0.2 and st["fail"] <= p1_train["fail"]:
                best_p2_score = st["score"]
                best_p2_sp = cand_sp

        cur_sp = best_p2_sp
        p2_train = evaluate_combo(cur_sp, cur_ov, train_starts)
        print(
            f"  ✅ Palier 2 retenu : tp_roi_pct={cur_sp['tp_roi_pct']}% (+{cur_sp['tp_roi_pct']/50:.1f}R), "
            f"trail={cur_sp['trail_variant']}, same_poi_tol={cur_sp['same_poi_tol_pct']}%\n"
            f"     TRAIN : {fmt_st(p2_train)}",
            flush=True,
        )

    if not args.eval_only:
        # --------------------------------------------------------------------------
        # PALIER 3 : Calibrage Fin des 5 Règles du Risk Overlay Hyro One-Step
        # --------------------------------------------------------------------------
        p3_grid = list(itertools.product(
            [0.65, 0.68, 0.75, 0.80],      # Règle 1 : hyro_risk_pct
            [2.0, 2.2, 2.5],               # Règle 2 : hyro_daily_stop_pct
            [3.6, 3.8],                    # Règle 3 : hyro_pause_at_cap_pct (plafond consistance 40% = 4.0%)
            [(2.8, 0.55), (3.2, 0.55), (3.5, 0.65)],  # Règle 4 : (hyro_dd_throttle_pct, hyro_dd_mult)
            [(6.5, 0.55), (7.0, 0.60)],               # Règle 5a: (hyro_lock_at_pct, hyro_lock_mult)
            [3, 4],                                   # Règle 5b: hyro_max_open_trades
        ))
        print(f"\n--- Palier 3/3 : Calibrage des 5 Règles Hyro One-Step Risk Overlay ({len(p3_grid)} combinaisons sur TRAIN) ---", flush=True)
        p3_results = []
        for idx, (rk, d_stop, p_cap, (dd_at, dd_m), (lk_at, lk_m), max_op) in enumerate(p3_grid, 1):
            cand_ov = {
                "hyro_risk_pct": rk,
                "hyro_daily_stop_pct": d_stop,
                "hyro_daily_guard": 1,
                "hyro_pause_at_cap_pct": p_cap,
                "hyro_cap_scale": 1,
                "hyro_dd_throttle_pct": dd_at,
                "hyro_dd_mult": dd_m,
                "hyro_lock_at_pct": lk_at,
                "hyro_lock_mult": lk_m,
                "hyro_max_open_trades": max_op,
                "hyro_max_notional_mult": 2.0,
            }
            st = evaluate_combo(cur_sp, cand_ov, train_starts)
            p3_results.append((st, cand_ov))
            if idx % 96 == 0 or idx == len(p3_grid):
                best_so_far = max(x[0]["score"] for x in p3_results)
                print(f"  [Palier 3 | {idx:3d}/{len(p3_grid)} essais] meilleur score TRAIN = {best_so_far:+.1f}", flush=True)

        p3_results.sort(key=lambda x: x[0]["score"], reverse=True)
        cur_ov = p3_results[0][1]

    opt_train = evaluate_combo(cur_sp, cur_ov, train_starts)
    opt_test = evaluate_combo(cur_sp, cur_ov, test_starts)
    opt_4y = evaluate_combo(cur_sp, cur_ov, sim4y_starts)

    print(
        f"  ✅ Palier 3 retenu (5 Règles Hyro One-Step) :\n"
        f"     • Règle 1 (Sizing)        : hyro_risk_pct = {cur_ov['hyro_risk_pct']}%\n"
        f"     • Règle 2 (Stop Jour)     : hyro_daily_stop_pct = -{cur_ov['hyro_daily_stop_pct']}% + garde-fou simultané ON\n"
        f"     • Règle 3 (Consistance)   : hyro_pause_at_cap_pct = +{cur_ov['hyro_pause_at_cap_pct']}% + écrêtage dynamique 40% ON\n"
        f"     • Règle 4 (DD Throttle)   : hyro_dd_throttle_pct = -{cur_ov['hyro_dd_throttle_pct']}% -> risque x{cur_ov['hyro_dd_mult']}\n"
        f"     • Règle 5 (Lock & Pos.)   : hyro_lock_at_pct = +{cur_ov['hyro_lock_at_pct']}% -> risque x{cur_ov['hyro_lock_mult']} | max {cur_ov['hyro_max_open_trades']} pos. (<= 2.0x notionnel)",
        flush=True,
    )

    # ==============================================================================
    # 4. SIMULATION COMPLÈTE SUR LES 4 DERNIÈRES ANNÉES (2022-01-01 -> 2026-11-09)
    # ==============================================================================
    trades_opt, events_opt, ev_ts_opt, loss_mult_opt = get_prepared_trades(cur_sp)
    trades_ref, events_ref, ev_ts_ref, loss_mult_ref = get_prepared_trades(REF_STRATEGY_PARAMS)

    # A) Départs glissants tous les 4 jours sur les 4 dernières années (2022 -> 2026)
    att_4y_opt = []
    for s0 in sim4y_starts:
        r = run_hyro_onestep_challenge(events_opt, ev_ts_opt, trades_opt, loss_mult_opt, s0, cur_ov, args.horizon, "worst", HYRO_ONESTEP_SPEC)
        r["start_ts"] = s0
        att_4y_opt.append(r)

    # B) Détail Année par Année sur les 4 dernières années (2022, 2023, 2024, 2025, 2026)
    yearly_4y = {}
    for a in att_4y_opt:
        yr = datetime.fromtimestamp(a["start_ts"], tz=timezone.utc).year
        yearly_4y.setdefault(yr, []).append(a)

    # C) Simulation SÉQUENTIELLE (bout-à-bout en conditions réelles du 1er janvier 2022 à aujourd'hui)
    seq_campaigns_opt = simulate_sequential_campaigns(
        events_opt, ev_ts_opt, trades_opt, loss_mult_opt, sim4y_ts, last_ts, cur_ov, HYRO_ONESTEP_SPEC
    )
    seq_campaigns_ref = simulate_sequential_campaigns(
        events_ref, ev_ts_ref, trades_ref, loss_mult_ref, sim4y_ts, last_ts, REF_HYRO_OVERLAY, HYRO_ONESTEP_SPEC
    )

    # Sauvegarde du JSON des meilleurs paramètres Hyro One-Step
    best_json_path = os.path.join(args.outdir, "best_params_hyro_onestep.json")
    payload = {
        "propfirm": "HyroTrader One-Step (10 000 $)",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy_params": cur_sp,
        "hyro_overlay": cur_ov,
        "metrics": {
            "train_pre2024": opt_train,
            "test_post2024": opt_test,
            "last_4_years_2022_2026": opt_4y,
        },
    }
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # Sauvegarde du CSV détaillé des campagnes séquentielles sur 4 ans
    seq_csv_path = os.path.join(args.outdir, "HYRO_ONESTEP_4Y_campaigns.csv")
    with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "campaign_id", "start_date", "end_date", "status", "fail_reason",
            "days", "final_return_pct", "credited_return_pct", "max_dd_pct",
            "worst_day_pct", "trades_taken", "trades_skipped",
        ])
        for idx, c in enumerate(seq_campaigns_opt, 1):
            d0 = datetime.fromtimestamp(c["start_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(c["end_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
            w.writerow([
                idx, d0, d1, c["status"].upper(), c["fail_reason"],
                f"{c['days']:.1f}", f"{(c['final_bal'] - 1.0)*100:+.2f}",
                f"{c['credited_profit']*100:+.2f}", f"{c['max_dd']*100:.2f}",
                f"{c['worst_day']*100:+.2f}", c["trades_taken"], c["trades_skipped"],
            ])

    # Construction du rapport complet 4 Ans + Comparatif Avant/Après Optimisation
    seq_passes = [c for c in seq_campaigns_opt if c["status"] == "pass"]
    seq_fails = [c for c in seq_campaigns_opt if c["status"] == "fail"]
    seq_opens = [c for c in seq_campaigns_opt if c["status"] == "open"]

    ref_seq_passes = [c for c in seq_campaigns_ref if c["status"] == "pass"]
    ref_seq_fails = [c for c in seq_campaigns_ref if c["status"] == "fail"]

    lines = [
        "===================================================================================================",
        "🏆 RAPPORT D'OPTIMISATION & SIMULATION 4 ANS (2022–2026) — HYROTRADER ONE-STEP (10 000 $) + CRYPTO V4",
        "===================================================================================================",
        f"Règles HyroTrader One-Step : Objectif +10% | Perte jour -4% | Perte max -8% FIXE | Min 3 jours |",
        f"                             Consistance 40%/jour (+4.0% max/jour) | Notionnel <= 2.0x | Frais Bybit 0.020%/0.055%",
        f"Actifs tradés (4)          : {', '.join(syms)} ({len(trades_opt)} trades qualifiés sur 2020–2026)",
        "",
        "---------------------------------------------------------------------------------------------------",
        "1. LES 5 RÈGLES HYRO ONE-STEP + PARAMÈTRES SMC RETENUS PAR L'OPTIMISEUR WALK-FORWARD",
        "---------------------------------------------------------------------------------------------------",
        f"  • Filtres SMC Anti-Krach (Palier 1 & 2) :",
        f"      - Range M15 autorisé    : [{cur_sp['m15_min_range_pct']:.2f}% .. {cur_sp['m15_max_range_pct']:.2f}%] (au lieu de 1.60% -> coupe les mèches de krach)",
        f"      - Seuil Biais H4        : ±{cur_sp['h4_thresh']:.2f} sur {cur_sp['h4_lookback']} bougies 4h",
        f"      - Cooldown après SL     : {cur_sp['sl_cooldown_hours']:.1f}h d'attente + blocage même POI ({cur_sp['same_poi_tol_pct']:.2f}%)",
        f"      - Objectif TP & Trail   : TP +{cur_sp['tp_roi_pct']:.0f}% ROI (+{cur_sp['tp_roi_pct']/50:.1f}R) | Trailing = {cur_sp['trail_variant']}",
        f"  • Les 5 Règles du Risk Overlay Hyro One-Step (Palier 3) :",
        f"      - Règle 1 (Sizing)      : {cur_ov['hyro_risk_pct']:.2f}% du capital initial par trade (1R = {10000*cur_ov['hyro_risk_pct']/100:.0f}$ sur 10k$)",
        f"      - Règle 2 (Stop Jour)   : Disjoncteur journalier à -{cur_ov['hyro_daily_stop_pct']:.1f}% (bien avant les -4.0%) + Garde-fou simultané ON",
        f"      - Règle 3 (Consistance) : Pause journalière dès +{cur_ov['hyro_pause_at_cap_pct']:.1f}% + Écrêtage dynamique pour la règle des 40% (+4.0%/j)",
        f"      - Règle 4 (DD Throttle) : Si Drawdown >= -{cur_ov['hyro_dd_throttle_pct']:.1f}% (sur -8.0% fixe) -> Risque multiplié par x{cur_ov['hyro_dd_mult']:.2f}",
        f"      - Règle 5 (Lock & Pos.) : Si Profit validé >= +{cur_ov['hyro_lock_at_pct']:.1f}% -> Risque x{cur_ov['hyro_lock_mult']:.2f} | Max {cur_ov['hyro_max_open_trades']} pos. (<= 2.0x solde)",
        "",
        "---------------------------------------------------------------------------------------------------",
        "2. GAIN APPORTÉ PAR L'OPTIMISATION (AVANT vs APRÈS OPTIMISATION WALK-FORWARD)",
        "---------------------------------------------------------------------------------------------------",
        f"  • Période TRAIN (< 2024-01-01, {len(train_starts)} départs glissants) :",
        f"      - Avant opt. : {fmt_st(ref_train)}",
        f"      - Après opt. : {fmt_st(opt_train)}",
        f"  • Période TEST HORS-ÉCHANTILLON (2024-01-01 -> 2026-11-09, {len(test_starts)} départs glissants) :",
        f"      - Avant opt. : {fmt_st(ref_test)}",
        f"      - Après opt. : {fmt_st(opt_test)}",
        f"  • SIMULATION SUR LES 4 DERNIÈRES ANNÉES (2022-01-01 -> 2026-11-09, {len(sim4y_starts)} départs glissants) :",
        f"      - Avant opt. : {fmt_st(ref_4y)}",
        f"      - Après opt. : {fmt_st(opt_4y)}",
        "",
        "---------------------------------------------------------------------------------------------------",
        "3. DÉTAIL ANNÉE PAR ANNÉE SUR LES 4 DERNIÈRES ANNÉES (2022 -> 2026, départs tous les 4 jours)",
        "---------------------------------------------------------------------------------------------------",
        "  Année | Départs | Réussite (Tous) | Réussite (Résolus) | Échecs | <= 30j | <= 45j | <= 60j | Médiane | DD médian",
        "  ------+---------+-----------------+--------------------+--------+--------+--------+--------+---------+----------",
    ]

    for yr in sorted(yearly_4y.keys()):
        st_y = summarize_attempts(yearly_4y[yr], HYRO_ONESTEP_SPEC["price_usd"])
        lines.append(
            f"  {yr:5d} | {st_y['n']:7d} | {st_y['pass']*100:14.1f}% | {st_y['pass_resolved']*100:17.1f}% | "
            f"{st_y['fail']*100:5.1f}% | {st_y['p30']*100:5.1f}% | {st_y['p45']*100:5.1f}% | {st_y['p60']*100:5.1f}% | "
            f"{st_y['med_days']:6.1f}j | {st_y['max_dd_med']*100:7.2f}%"
        )

    lines.extend([
        "",
        "---------------------------------------------------------------------------------------------------",
        "4. SIMULATION SÉQUENTIELLE BOUT-À-BOUT SUR 4 ANS (2022-01-01 -> 2026-11-09)",
        "   (Dès qu'un challenge se termine, un nouveau est lancé le lendemain)",
        "---------------------------------------------------------------------------------------------------",
        f"  • Total de campagnes jouées bout-à-bout sur 4 ans :",
        f"      - Configuration Optimisée : {len(seq_passes)} challenges VALIDÉS (+10%) contre {len(seq_fails)} échecs "
        f"(taux de succès résolu = {len(seq_passes)/max(1, len(seq_passes)+len(seq_fails))*100:.1f}%)",
        f"      - Configuration Référence : {len(ref_seq_passes)} challenges VALIDÉS (+10%) contre {len(ref_seq_fails)} échecs "
        f"(taux de succès résolu = {len(ref_seq_passes)/max(1, len(ref_seq_passes)+len(ref_seq_fails))*100:.1f}%)",
        "",
        "  #Camp | Début      | Fin        | Statut | Durée  | Gain Brut | Gain Retenu (40%) | DD Max | Pire Jour | Trades (Pris/Ignorés)",
        "  ------+------------+------------+--------+--------+-----------+-------------------+--------+-----------+----------------------",
    ])

    for idx, c in enumerate(seq_campaigns_opt, 1):
        d0 = datetime.fromtimestamp(c["start_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        d1 = datetime.fromtimestamp(c["end_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        st_lbl = "✅ PASS" if c["status"] == "pass" else ("❌ FAIL" if c["status"] == "fail" else "⏳ OPEN")
        lines.append(
            f"  #{idx:4d} | {d0} | {d1} | {st_lbl:6s} | {c['days']:5.1f}j | "
            f"{(c['final_bal']-1.0)*100:+8.2f}% | {c['credited_profit']*100:+16.2f}% | "
            f"-{c['max_dd']*100:4.2f}% | {c['worst_day']*100:+8.2f}% | {c['trades_taken']:4d} / {c['trades_skipped']:4d}"
        )

    report_txt = "\n".join(lines)
    rep_path = os.path.join(args.outdir, "HYRO_ONESTEP_4Y_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")

    print("\n" + report_txt, flush=True)
    print(f"\n💾 Paramètres sauvegardés dans : {best_json_path}")
    print(f"💾 Rapport 4 ans sauvegardé dans : {rep_path}")
    print(f"💾 Détail des campagnes 4 ans dans : {seq_csv_path} (durée totale : {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
