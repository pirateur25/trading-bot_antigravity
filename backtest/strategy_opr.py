"""
backtest/strategy_opr.py — Stratégie OPR (Opening Price Range) + Indicateur RV3 (Romain Bailleul) sur les indices US
(NAS100 / US100, USA500 / US500, US30).

Piliers fixes non négociables :
  1. Zone OPR :
     - Définie par la bougie M15 d'ouverture US : 15h30 -> 15h45 heure française (`Europe/Paris`, ou 09h30 -> 09h45 NY),
       c.-à-d. les 3 bougies M5 [15h30, 15h35, 15h40].
     - OPR High = plus haut (mèche incluse) de cette bougie M15.
     - OPR Low  = plus bas (mèche incluse) de cette bougie M15.
  2. Filtre obligatoire de tendance en H1 (`mm20_50`) :
     - Tendance haussière (+1) : MM20 > MM50 en H1 ET clôture de la bougie H1 en cours > MM50.
     - Tendance baissière (-1) : MM20 < MM50 en H1 ET clôture de la bougie H1 en cours < MM50.
  3. Unité d'exécution M5 (15h45 -> 19h30 heure française) :
     - Maximum 1 signal / ordre posé par jour et par indice (le premier des 2 setups valides qui se présente).
     - Deux setups d'entrée :
       * Setup 1 (Cassure) : clôture M5 au-dessus de l'OPR High (Long) -> BUY LIMIT sur OPR High ;
         clôture M5 sous l'OPR Low (Short) -> SELL LIMIT sur OPR Low.
       * Setup 2 (Réintégration) : sortie préalable sous l'OPR Low puis clôture M5 dans l'OPR -> BUY LIMIT sur OPR Low ;
         sortie préalable au-dessus de l'OPR High puis clôture M5 dans l'OPR -> SELL LIMIT sur OPR High.
     - L'ordre limite non déclenché est annulé si le prix atteint +1 R depuis l'entrée ou à 19h30 heure française.
  4. Modèle RV3 (Expected Move calibré sur 60 séances US de 09h30 à 16h00 NY / 15h30 à 22h00 FR) :
     - `Open` = prix d'ouverture US à 15h30 FR (`o[opr_i0]`).
     - `EM_brut` (Ligne A) = `Open * (VIX / 100.0) / 16.0` -> bornes `Open +/- EM_brut`.
     - Sur les 60 dernières séances US complètes :
       * `r_up[d] = (High_session[d] - Open[d]) / EM_brut[d]`
       * `r_dn[d] = (Open[d] - Low_session[d]) / EM_brut[d]`
     - Ligne B (Moyenne calibrée 60 séances) :
       * `dist_B_up = mean(r_up) * EM_brut` -> `B_up = Open + dist_B_up`
       * `dist_B_dn = mean(r_dn) * EM_brut` -> `B_dn = Open - dist_B_dn`
     - Ligne C (Enveloppe +1 sigma 60 séances) :
       * `dist_C_up = (mean(r_up) + std(r_up)) * EM_brut` -> `C_up = Open + dist_C_up`
       * `dist_C_dn = (mean(r_dn) + std(r_dn)) * EM_brut` -> `C_dn = Open - dist_C_dn`
     - Seuil de bruit (`sl_20pct`) = `em_noise_pct * EM_brut` (20% de l'EM brut par défaut).
  5. Sélection dynamique du Stop Loss (`SL_distance`) selon l'indice de volatilité (VXN / VIX / VXD) :
     - `SL_M5` = distance technique entre le prix d'entrée et le plus bas (Long) / plus haut (Short) de la bougie M5 de signal.
     - Si `VIX > 20` (Stress)   : `SL_distance = max(SL_M5, sl_atr_mult * ATR_H1_14)`
     - Si `15 <= VIX <= 20`     : `SL_distance = max(SL_M5, Seuil_Bruit_EM, Plancher_Indice)` (80 US30, 70 US100, 50 US500)
     - Si `VIX < 15` (Euphorie) : `SL_distance = max(SL_M5, Seuil_Bruit_EM)`
     - Emplacement du SL : `entry - SL_distance` (Long) | `entry + SL_distance` (Short).
  6. Validation de l'espace disponible avant BE (`Distance_BE = be_rr * SL_distance`) et gestion RV3 du TP :
     - Si un obstacle (Borne B/C du RV3, PDH/PDL, PSH/PSL, FVG H1 non mitigée) est à une distance `< Distance_BE`, le trade est annulé.
     - Pour le TP : vérification par rapport à l'enveloppe RV3 (B ou C selon `rv3_tp_mode`).
"""
import gzip
import os
import numpy as np
import pandas as pd

SETUP_ORDER = ["Cassure", "Reintegration"]

# Plancher minimal de SL en points (cours 2026) et son équivalent proportionnel au cours (pour 2015-2026)
MIN_SL_POINTS = {
    "US30": 80.0,
    "NAS100": 70.0,
    "US100": 70.0,
    "USA500": 50.0,
    "US500": 50.0,
}

# Équivalent proportionnel au prix de l'indice :
# - US30   : ~80 pts à 44 000 (0.18%), soit ~32 pts en 2015 (à 18 000)
# - NAS100 : ~70 pts à 31 000 (0.22%), soit ~9.5 pts en 2015 (à 4 300)
# - USA500 : ~12.5 pts (50 ticks ES) à 7 000 (0.18%), soit ~3.8 pts en 2015 (à 2 100)
MIN_SL_PCT = {
    "US30": 0.0018,
    "NAS100": 0.0022,
    "US100": 0.0022,
    "USA500": 0.0018,
    "US500": 0.0018,
}

# Fichier d'indice de volatilité associé à chaque indice US dans data/
VOL_FILES = {
    "US30": "VXDCLS-US30.csv.gz",
    "NAS100": "VXNCLS-US100.csv.gz",
    "US100": "VXNCLS-US100.csv.gz",
    "USA500": "VIXCLS-US500.csv.gz",
    "US500": "VIXCLS-US500.csv.gz",
}

PARAMS = {
    # --- Piliers fixes ---
    "trend_tf": "1h",
    "trend_mode": "mm20_50",
    "tz_mode": "paris",
    "end_hour": 18,             # Fin de prise de nouveau signal (18h30 FR par défaut, max 19h30 FR)
    "end_min": 30,
    "side": "both",
    "setups": "Cassure,Reintegration",
    "cancel_rr": 1.0,
    "rv3_lookback": 60,
    "vix_low": 15.0,
    "vix_high": 20.0,
    # --- Leviers optimisables (SL / BE / TP / Obstacles / RV3 / Filtres Anti-Range) ---
    "be_rr": 1.25,              # Break-Even à X R
    "em_noise_pct": 0.20,       # Seuil de bruit en fraction de l'EM brut (0.20 = 20% EM)
    "sl_vix15_mode": "m5_only", # Régime VIX < 15 : "m5_only" (SL sous bougie M5) | "half_noise" (10% EM) | "full_noise" (20% EM)
    "sl_floor_mode": "pct",     # Plancher 15 <= VIX <= 20 : "pct" (proportionnel au prix 2015-2026) | "fixed_pts" (50/70/80 pts fixes)
    "sl_atr_mult": 1.0,         # Multiplicateur ATR(14) H1 quand VIX > 20
    "sl_vix20_mode": "max_atr", # "max_atr" = max(SL_M5, mult*ATR_H1) | "add_atr" = SL_M5 + mult*ATR_H1
    "rr_break": 2.5,            # TP cible (en R) sur le setup Cassure
    "min_rr_reint": 2.0,        # RR minimum exigé sur le setup Réintégration (TP = borne opposée OPR)
    "res_be_mode": "rv3_b",     # Obstacles vérifiés avant BE : "none" | "rv3_b" | "rv3_c" | "sr_fvg" | "all"
    "rv3_tp_mode": "cap_c",     # Gestion du TP vs RV3 (B / C) : "none" | "reject_c" | "reject_b" | "cap_c" | "cap_b" | "ext_res"
    "max_rr_tp": 5.0,           # Plafond RR quand rv3_tp_mode == "ext_res"
    "fvg_h1_lookback_days": 15, # Jours de lookback pour les FVG H1 non mitigées
    # --- Filtres Anti-Range & Qualité de Cassure ---
    "min_vix": 14.0,            # Filtre anti-range : ignore les séances où VIX/VXN/VXD < min_vix (ex: 2017 VIX=10)
    "min_opr_em": 0.20,         # Ignore les OPR trop compressés (< 20% de l'EM brut = range étroit à l'open)
    "max_opr_em": 0.55,         # Ignore les OPR géants (> 55% de l'EM brut déjà consommé en 15 min)
    "min_mm_spread_atr": 0.15,  # Écart minimal |MM20 - MM50| en H1 en fraction de l'ATR H1 (0 = pas de filtre de range H1)
    "true_breakout_only": 1,    # 1 = la bougie M5 de Cassure doit franchir la borne OPR (pas la 4e bougie déjà dehors)
    "max_fill_bars": 6,         # Délai max (en bougies M5) pour déclencher l'ordre limite après le signal (6 = 30 min)
}


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


def _tz_vectors(pd_, tz_mode="paris"):
    tz_name = "America/New_York" if str(tz_mode).lower() == "ny" else "Europe/Paris"
    key = ("opr_tz_vec", tz_name)
    if key not in pd_.cache:
        dt = pd.to_datetime(pd_.base["open_ns"], utc=True).tz_convert(tz_name)
        day = (dt.year * 10000 + dt.month * 100 + dt.day).to_numpy(np.int32)
        mins = (dt.hour * 60 + dt.minute).to_numpy(np.int16)
        pd_.cache[key] = (day, mins)
    return pd_.cache[key]


def _load_vol_series(pd_, datadir="data"):
    sym = str(getattr(pd_, "symbol", "")).upper()
    key = ("opr_vol_map", sym)
    if key in pd_.cache:
        return pd_.cache[key]

    fname = VOL_FILES.get(sym)
    vol_map = {}
    if fname:
        candidates = [
            os.path.join(datadir, fname),
            os.path.join("data", fname),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", fname),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            op = gzip.open if path.endswith(".gz") else open
            with op(path, "rt", encoding="utf-8", errors="replace") as f:
                last_val = np.nan
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2 or not parts[0] or not parts[0][0].isdigit():
                        continue
                    d_str = parts[0].strip().replace("-", "")
                    if len(d_str) != 8 or not d_str.isdigit():
                        continue
                    d_int = int(d_str)
                    v_str = parts[1].strip()
                    try:
                        val = float(v_str)
                        last_val = val
                    except ValueError:
                        val = last_val
                    if not np.isnan(val):
                        vol_map[d_int] = val

    sorted_days = np.array(sorted(vol_map.keys()), dtype=np.int32)
    sorted_vals = np.array([vol_map[d] for d in sorted_days], dtype=np.float64)
    pd_.cache[key] = (sorted_days, sorted_vals)
    return pd_.cache[key]


def _vix_for_day(sorted_days, sorted_vals, day_int):
    if len(sorted_days) == 0:
        return np.nan
    pos = int(np.searchsorted(sorted_days, day_int, side="right")) - 1
    if pos < 0:
        return float(sorted_vals[0])
    return float(sorted_vals[pos])


def _h1_atr_and_fvgs(pd_):
    key = ("opr_h1_atr_fvgs",)
    if key in pd_.cache:
        return pd_.cache[key]

    f = pd_.frames["1h"]
    hh, hl, hc = f["h"], f["l"], f["c"]
    h_open_ns, h_close_ns = f["open_ns"], f["close_ns"]
    atr_h1 = f["atr"]
    nh = len(hc)

    b_open_ns = pd_.base["open_ns"]
    b_close_ns = pd_.base["close_ns"]
    bh, bl = pd_.base["h"], pd_.base["l"]
    nb = len(b_close_ns)

    j_h1 = np.searchsorted(h_close_ns, b_close_ns, side="right") - 1

    bear_q = np.flatnonzero(hh[2:] < hl[:-2])
    bull_q = np.flatnonzero(hl[2:] > hh[:-2])

    bear_fvgs = []
    for q in bear_q:
        form_idx = int(q + 2)
        form_ns = int(h_close_ns[form_idx])
        f_lo = float(hh[form_idx])
        f_hi = float(hl[q])
        mit_ns = np.iinfo(np.int64).max
        for m in range(form_idx + 1, nh):
            if hh[m] >= f_lo:
                b0 = int(np.searchsorted(b_open_ns, h_open_ns[m], side="left"))
                b1 = int(np.searchsorted(b_close_ns, h_close_ns[m], side="right"))
                for bi in range(b0, min(b1, nb)):
                    if bh[bi] >= f_lo:
                        mit_ns = int(b_close_ns[bi])
                        break
                if mit_ns == np.iinfo(np.int64).max:
                    mit_ns = int(h_close_ns[m])
                break
        bear_fvgs.append((form_ns, f_lo, f_hi, mit_ns))

    bull_fvgs = []
    for q in bull_q:
        form_idx = int(q + 2)
        form_ns = int(h_close_ns[form_idx])
        f_lo = float(hh[q])
        f_hi = float(hl[form_idx])
        mit_ns = np.iinfo(np.int64).max
        for m in range(form_idx + 1, nh):
            if hl[m] <= f_hi:
                b0 = int(np.searchsorted(b_open_ns, h_open_ns[m], side="left"))
                b1 = int(np.searchsorted(b_close_ns, h_close_ns[m], side="right"))
                for bi in range(b0, min(b1, nb)):
                    if bl[bi] <= f_hi:
                        mit_ns = int(b_close_ns[bi])
                        break
                if mit_ns == np.iinfo(np.int64).max:
                    mit_ns = int(h_close_ns[m])
                break
        bull_fvgs.append((form_ns, f_lo, f_hi, mit_ns))

    bear_arr = np.array(bear_fvgs, dtype=np.float64) if bear_fvgs else np.zeros((0, 4), dtype=np.float64)
    bull_arr = np.array(bull_fvgs, dtype=np.float64) if bull_fvgs else np.zeros((0, 4), dtype=np.float64)

    pd_.cache[key] = (j_h1, atr_h1, bear_arr, bull_arr)
    return pd_.cache[key]


def _trend_h1_mm20_50(pd_, tf="1h", mode="mm20_50"):
    key = ("opr_trend_v2", tf, mode)
    if key in pd_.cache:
        return pd_.cache[key]

    f = pd_.frames[tf]
    hc = f["c"]
    atr_h = f["atr"]
    nh = len(hc)
    b_close = pd_.base["c"]
    b_close_ns = pd_.base["close_ns"]
    nb = len(b_close)

    j = np.searchsorted(f["close_ns"], b_close_ns, side="right") - 1

    if mode == "ema_mm20_50":
        s_c = pd.Series(hc)
        ema20 = s_c.ewm(span=20, adjust=False).mean().to_numpy()
        ema50 = s_c.ewm(span=50, adjust=False).mean().to_numpy()
        st_h = np.zeros(nh, dtype=np.int8)
        sp_h = np.zeros(nh, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            bull = (np.arange(nh) >= 20) & (ema20 > ema50) & (hc > ema50)
            bear = (np.arange(nh) >= 20) & (ema20 < ema50) & (hc < ema50)
            sp_h = np.where(atr_h > 0, np.abs(ema20 - ema50) / atr_h, 0.0).astype(np.float32)
        st_h[bull] = 1
        st_h[bear] = -1
        j_clip = np.clip(j, 0, max(0, nh - 1))
        mapped = np.where(j >= 0, st_h[j_clip], 0).astype(np.int8)
        spread = np.where(j >= 0, sp_h[j_clip], 0.0).astype(np.float32)
        pd_.cache[key] = (mapped, spread)
        return pd_.cache[key]

    csum = np.concatenate([[0.0], np.cumsum(hc, dtype=np.float64)])
    exact_close = np.zeros(nb, dtype=bool)
    valid_j = j >= 0
    exact_close[valid_j] = f["close_ns"][j[valid_j]] == b_close_ns[valid_j]

    mapped = np.zeros(nb, dtype=np.int8)
    spread = np.zeros(nb, dtype=np.float32)
    ok = (j >= 49) & (~np.isnan(b_close))
    idx = np.flatnonzero(ok)
    if len(idx) > 0:
        ji = j[idx]
        ex = exact_close[idx]
        cur_c = b_close[idx]

        last_prev = np.where(ex, ji - 1, ji)
        sum19 = csum[last_prev + 1] - csum[last_prev - 18]
        sum49 = csum[last_prev + 1] - csum[last_prev - 48]

        mm20 = (sum19 + cur_c) / 20.0
        mm50 = (sum49 + cur_c) / 50.0

        bull = (mm20 > mm50) & (cur_c > mm50)
        bear = (mm20 < mm50) & (cur_c < mm50)
        st_sub = np.zeros(len(idx), dtype=np.int8)
        st_sub[bull] = 1
        st_sub[bear] = -1
        mapped[idx] = st_sub

        atr_sub = atr_h[ji]
        with np.errstate(invalid="ignore", divide="ignore"):
            sp_sub = np.where(atr_sub > 0, np.abs(mm20 - mm50) / atr_sub, 0.0)
        spread[idx] = sp_sub.astype(np.float32)

    pd_.cache[key] = (mapped, spread)
    return pd_.cache[key]


def _precompute_daily_sessions(pd_, tz_mode="paris"):
    """
    Précalcule une seule fois toutes les journées de trading et leurs statistiques de session US
    (15h30 -> 22h00 FR / 09h30 -> 16h00 NY) afin que :
      1) la calibration RV3 sur 60 séances (`r_up`, `r_dn`) soit instantanée,
      2) chaqueessai Optuna sur 12 ans s'exécute en quelques millisecondes.
    """
    key = ("opr_daily_sessions", tz_mode)
    if key in pd_.cache:
        return pd_.cache[key]

    day_vec, mins_vec = _tz_vectors(pd_, tz_mode)
    sorted_vdays, sorted_vvals = _load_vol_series(pd_)

    B = pd_.base
    o, h, l = B["o"], B["h"], B["l"]
    n = pd_.n

    if tz_mode == "ny":
        opr_start_min = 9 * 60 + 30    # 09h30 NY
        opr_end_min = 9 * 60 + 45      # 09h45 NY
        rth_end_min = 16 * 60          # 16h00 NY (clôture cash US)
    else:
        opr_start_min = 15 * 60 + 30   # 15h30 Paris
        opr_end_min = 15 * 60 + 45     # 15h45 Paris
        rth_end_min = 22 * 60          # 22h00 Paris (clôture cash US)

    # Trouver les frontières de jours rapidement avec np.flatnonzero
    if n == 0:
        pd_.cache[key] = []
        return []

    change_idx = np.flatnonzero(np.diff(day_vec) != 0) + 1
    starts = np.concatenate([[0], change_idx])
    ends = np.concatenate([change_idx, [n]])

    days_list = []
    prev_day_hi = np.nan
    prev_day_lo = np.nan

    # Historique glissant des ratios RV3 (r_up, r_dn) des séances US précédentes
    hist_r_up = []
    hist_r_dn = []

    for d_start, d_end in zip(starts, ends):
        d_start = int(d_start)
        d_end = int(d_end)
        cur_day = int(day_vec[d_start])

        pdh = prev_day_hi
        pdl = prev_day_lo
        prev_day_hi = float(np.max(h[d_start:d_end]))
        prev_day_lo = float(np.min(l[d_start:d_end]))

        m_slice = mins_vec[d_start:d_end]
        opr_rel = np.flatnonzero((m_slice >= opr_start_min) & (m_slice < opr_end_min))
        if len(opr_rel) < 3:
            continue

        opr_i0 = d_start + int(opr_rel[0])
        opr_i1 = d_start + int(opr_rel[-1])
        opr_hi = float(np.max(h[opr_i0:opr_i1 + 1]))
        opr_lo = float(np.min(l[opr_i0:opr_i1 + 1]))
        if opr_hi <= opr_lo or np.isnan(opr_hi) or np.isnan(opr_lo):
            continue

        open_us = float(o[opr_i0])
        vix_val = _vix_for_day(sorted_vdays, sorted_vvals, cur_day)
        em_brut = open_us * (vix_val / 100.0) / 16.0 if (vix_val == vix_val and vix_val > 0) else np.nan

        # Calibration RV3 sur les 60 dernières séances US (sans regarder la séance du jour !)
        if len(hist_r_up) >= 10:
            rup_60 = np.array(hist_r_up[-60:], dtype=np.float64)
            rdn_60 = np.array(hist_r_dn[-60:], dtype=np.float64)
            mean_up, std_up = float(np.mean(rup_60)), float(np.std(rup_60))
            mean_dn, std_dn = float(np.mean(rdn_60)), float(np.std(rdn_60))
        else:
            mean_up, std_up = 0.65, 0.45
            mean_dn, std_dn = 0.70, 0.48

        # Mettre à jour l'historique RV3 avec la séance RTH (15h30 -> 22h00 FR) à la fin de la journée
        rth_rel = np.flatnonzero((m_slice >= opr_start_min) & (m_slice < rth_end_min))
        if len(rth_rel) >= 3 and em_brut == em_brut and em_brut > 0:
            rth_i0 = d_start + int(rth_rel[0])
            rth_i1 = d_start + int(rth_rel[-1])
            rth_hi = float(np.max(h[rth_i0:rth_i1 + 1]))
            rth_lo = float(np.min(l[rth_i0:rth_i1 + 1]))
            r_up = (rth_hi - open_us) / em_brut
            r_dn = (open_us - rth_lo) / em_brut
            # Écarte les valeurs aberrantes extrêmes (> 4x l'EM brut, soit "séances exclues" dans RV3)
            if 0.0 <= r_up <= 4.0 and 0.0 <= r_dn <= 4.0:
                hist_r_up.append(r_up)
                hist_r_dn.append(r_dn)

        # Haut / Bas de la pré-session (00h00 -> 15h30 FR : Asie + Londres)
        pre_rel = np.flatnonzero(m_slice < opr_start_min)
        if len(pre_rel) > 0:
            p0 = d_start + int(pre_rel[0])
            p1 = d_start + int(pre_rel[-1])
            psh = float(np.max(h[p0:p1 + 1]))
            psl = float(np.min(l[p0:p1 + 1]))
        else:
            psh = np.nan
            psl = np.nan

        high_before_exec = float(np.max(h[d_start:opr_i1 + 1]))
        low_before_exec = float(np.min(l[d_start:opr_i1 + 1]))

        days_list.append({
            "day": cur_day,
            "d_start": d_start,
            "d_end": d_end,
            "opr_i0": opr_i0,
            "opr_i1": opr_i1,
            "opr_hi": opr_hi,
            "opr_lo": opr_lo,
            "open_us": open_us,
            "vix": vix_val,
            "em_brut": em_brut,
            "mean_up": mean_up,
            "std_up": std_up,
            "mean_dn": mean_dn,
            "std_dn": std_dn,
            "pdh": pdh,
            "pdl": pdl,
            "psh": psh,
            "psl": psl,
            "high_before_exec": high_before_exec,
            "low_before_exec": low_before_exec,
        })

    pd_.cache[key] = days_list
    return days_list


def _compute_sl_rv3(side, entry, candle_low, candle_high, vix_val, atr_h1_val, em_brut, min_sl_pts, min_sl_pct, p):
    """
    Calcul dynamique de la distance du Stop Loss selon les spécifications RV3 :
      1. Seuil de bruit EM = `em_noise_pct * EM_brut` (20% EM par défaut)
      2. Distance technique M5 : `sl_m5 = entry - candle_low` (Long) ou `candle_high - entry` (Short)
      3. Sélection selon le VIX :
         - Si VIX > vix_high (20) :
             * mode "max_atr" : `sl_dist = max(sl_m5, sl_atr_mult * atr_h1)`
             * mode "add_atr" : `sl_dist = sl_m5 + sl_atr_mult * atr_h1`
         - Si vix_low (15) <= VIX <= vix_high (20) :
             * `sl_floor_mode == "pct"` : plancher proportionnel au prix (`entry * min_sl_pct`, adapté 2015-2026)
             * `sl_floor_mode == "fixed_pts"` : plancher fixe 2026 (`min_sl_pts`)
             * `sl_dist = max(sl_m5, noise_thr, floor_val)`
         - Si VIX < vix_low (15) :
             * `sl_vix15_mode == "m5_only"`    : SL sous la bougie M5 de cassure (`max(sl_m5, 0.03 * em_brut)`)
             * `sl_vix15_mode == "half_noise"` : `max(sl_m5, 0.5 * noise_thr)` (10% EM)
             * `sl_vix15_mode == "full_noise"` : `max(sl_m5, noise_thr)` (20% EM)
    """
    vix_low = float(p.get("vix_low", 15.0))
    vix_high = float(p.get("vix_high", 20.0))
    em_noise_pct = float(p.get("em_noise_pct", 0.20))
    sl_atr_mult = float(p.get("sl_atr_mult", 1.0))
    sl_vix20_mode = str(p.get("sl_vix20_mode", "max_atr")).lower()
    sl_vix15_mode = str(p.get("sl_vix15_mode", "m5_only")).lower()
    sl_floor_mode = str(p.get("sl_floor_mode", "pct")).lower()

    atr_ok = (atr_h1_val == atr_h1_val) and (atr_h1_val > 0)
    vix_ok = (vix_val == vix_val) and (vix_val > 0)
    em_ok = (em_brut == em_brut) and (em_brut > 0)
    noise_thr = (em_noise_pct * em_brut) if em_ok else 0.0
    tiny_floor = (0.03 * em_brut) if em_ok else (entry * 0.0003)

    if side == "long":
        sl_m5 = entry - candle_low
    else:
        sl_m5 = candle_high - entry
    if sl_m5 <= 0:
        return np.nan

    floor_val = float(min_sl_pts) if sl_floor_mode == "fixed_pts" else float(entry * min_sl_pct)

    if vix_ok and vix_val > vix_high and atr_ok:
        if sl_vix20_mode == "add_atr":
            sl_dist = sl_m5 + sl_atr_mult * atr_h1_val
        else:
            sl_dist = max(sl_m5, sl_atr_mult * atr_h1_val)
    elif vix_ok and vix_val >= vix_low:
        sl_dist = max(sl_m5, noise_thr, floor_val)
    else:
        if sl_vix15_mode == "full_noise":
            sl_dist = max(sl_m5, noise_thr)
        elif sl_vix15_mode == "half_noise":
            sl_dist = max(sl_m5, 0.5 * noise_thr)
        else:
            sl_dist = max(sl_m5, tiny_floor)

    return float(entry - sl_dist) if side == "long" else float(entry + sl_dist)


def _obstacles_long(entry, b_up, c_up, pdh, pdh_mit, psh, psh_mit, bear_fvgs, sig_close_ns, lookback_ns, res_be_mode):
    """
    Liste triée des obstacles au-dessus de `entry` pour vérifier la règle :
    `Si Distance_Obstacle < Distance_BE -> ANNULER LE TRADE`.
    """
    if res_be_mode == "none":
        return []
    obs = []
    if res_be_mode in ("rv3_b", "all") and b_up == b_up and b_up > entry:
        obs.append(float(b_up))
    if res_be_mode in ("rv3_c", "all") and c_up == c_up and c_up > entry:
        obs.append(float(c_up))
    if res_be_mode in ("sr_fvg", "all"):
        if pdh == pdh and not pdh_mit and pdh > entry:
            obs.append(float(pdh))
        if psh == psh and not psh_mit and psh > entry:
            obs.append(float(psh))
        if len(bear_fvgs) > 0:
            min_form = sig_close_ns - lookback_ns
            m = (bear_fvgs[:, 0] <= sig_close_ns) & (bear_fvgs[:, 0] >= min_form) & (bear_fvgs[:, 3] > sig_close_ns) & (bear_fvgs[:, 1] > entry)
            if np.any(m):
                obs.extend(bear_fvgs[m, 1].tolist())
    obs.sort()
    return obs


def _obstacles_short(entry, b_dn, c_dn, pdl, pdl_mit, psl, psl_mit, bull_fvgs, sig_close_ns, lookback_ns, res_be_mode):
    """
    Liste triée (décroissante) des obstacles en dessous de `entry` pour un Short.
    """
    if res_be_mode == "none":
        return []
    obs = []
    if res_be_mode in ("rv3_b", "all") and b_dn == b_dn and b_dn < entry:
        obs.append(float(b_dn))
    if res_be_mode in ("rv3_c", "all") and c_dn == c_dn and c_dn < entry:
        obs.append(float(c_dn))
    if res_be_mode in ("sr_fvg", "all"):
        if pdl == pdl and not pdl_mit and pdl < entry:
            obs.append(float(pdl))
        if psl == psl and not psl_mit and psl < entry:
            obs.append(float(psl))
        if len(bull_fvgs) > 0:
            min_form = sig_close_ns - lookback_ns
            m = (bull_fvgs[:, 0] <= sig_close_ns) & (bull_fvgs[:, 0] >= min_form) & (bull_fvgs[:, 3] > sig_close_ns) & (bull_fvgs[:, 2] < entry)
            if np.any(m):
                obs.extend(bull_fvgs[m, 2].tolist())
    obs.sort(reverse=True)
    return obs


def _resolve_tp(side, setup, entry, risk, opr_lo, opr_hi, rr_break, min_rr_reint, max_rr_tp, b_lvl, c_lvl, sr_list, rv3_tp_mode):
    """
    Détermine le Take Profit final et sa validité selon `rv3_tp_mode` :
      - Setup Cassure : TP de base = `entry +/- rr_break * risk`.
      - Setup Réintégration : TP de base = borne opposée de l'OPR (valide si >= `min_rr_reint * risk`).
      - Ajustement RV3 (`rv3_tp_mode`) :
        * "none"     : garde le TP de base.
        * "reject_c" : annule le trade si le TP dépasse l'enveloppe 1 sigma C ("trade rêveur" au-delà de C).
        * "reject_b" : annule le trade si le TP dépasse la moyenne calibrée B.
        * "cap_c"    : si le TP dépasse C, le plafonne sur C (à condition que C offre au moins `min_rr_reint` R).
        * "cap_b"    : si le TP dépasse B, le plafonne sur B (à condition que B offre au moins `min_rr_reint` R).
        * "ext_res"  : sur Cassure, étend le TP sur le premier obstacle (ou borne B/C) situé entre `rr_break` R et `max_rr_tp` R.
    """
    min_ok_rr = min(rr_break, min_rr_reint)

    if side == "long":
        if setup == "Cassure":
            tp = entry + rr_break * risk
            if rv3_tp_mode == "ext_res":
                cands = [r for r in sr_list if entry + rr_break * risk <= r <= entry + max_rr_tp * risk]
                if cands:
                    tp = float(min(cands))
        else:
            tp = opr_hi
            if (tp - entry) < min_rr_reint * risk:
                return np.nan, False

        if rv3_tp_mode == "reject_c" and c_lvl == c_lvl and tp > c_lvl:
            return np.nan, False
        if rv3_tp_mode == "reject_b" and b_lvl == b_lvl and tp > b_lvl:
            return np.nan, False
        if rv3_tp_mode == "cap_c" and c_lvl == c_lvl and tp > c_lvl:
            tp = float(c_lvl)
            if (tp - entry) < min_ok_rr * risk:
                return np.nan, False
        if rv3_tp_mode == "cap_b" and b_lvl == b_lvl and tp > b_lvl:
            tp = float(b_lvl)
            if (tp - entry) < min_ok_rr * risk:
                return np.nan, False
        return float(tp), True

    else:
        if setup == "Cassure":
            tp = entry - rr_break * risk
            if rv3_tp_mode == "ext_res":
                cands = [r for r in sr_list if entry - max_rr_tp * risk <= r <= entry - rr_break * risk]
                if cands:
                    tp = float(max(cands))
        else:
            tp = opr_lo
            if (entry - tp) < min_rr_reint * risk:
                return np.nan, False

        if rv3_tp_mode == "reject_c" and c_lvl == c_lvl and tp < c_lvl:
            return np.nan, False
        if rv3_tp_mode == "reject_b" and b_lvl == b_lvl and tp < b_lvl:
            return np.nan, False
        if rv3_tp_mode == "cap_c" and c_lvl == c_lvl and tp < c_lvl:
            tp = float(c_lvl)
            if (entry - tp) < min_ok_rr * risk:
                return np.nan, False
        if rv3_tp_mode == "cap_b" and b_lvl == b_lvl and tp < b_lvl:
            tp = float(b_lvl)
            if (entry - tp) < min_ok_rr * risk:
                return np.nan, False
        return float(tp), True


def run(pd_, capital=10000.0, risk_pct=0.5, fee_pct=0.005, leverage=30.0, p=None, i0=0, i1=None):
    """
    Exécute le backtest OPR + RV3 sur `pd_` dans l'intervalle de bougies 5m `[i0, i1)`.
    Retourne `(trades_df, equity_array_on_slice)`.
    """
    p = dict(PARAMS if p is None else p)
    rr_break = float(p.get("rr_break", 2.5))
    min_rr_reint = float(p.get("min_rr_reint", 2.0))
    max_rr_tp = float(p.get("max_rr_tp", 5.0))
    cancel_rr = float(p.get("cancel_rr", 1.0))
    be_rr = float(p.get("be_rr", 1.0))
    em_noise_pct = float(p.get("em_noise_pct", 0.20))
    res_be_mode = str(p.get("res_be_mode", "all")).lower()
    rv3_tp_mode = str(p.get("rv3_tp_mode", "none")).lower()
    lookback_ns = int(float(p.get("fvg_h1_lookback_days", 20)) * 86400 * 10**9)
    tz_mode = str(p.get("tz_mode", "paris")).lower()
    side_filter = str(p.get("side", "both")).lower()
    allowed_setups = {s.strip() for s in str(p.get("setups", "Cassure,Reintegration")).split(",") if s.strip()}

    sym_up = str(getattr(pd_, "symbol", "NAS100")).upper()
    min_sl_pts = MIN_SL_POINTS.get(sym_up, 50.0)
    min_sl_pct = MIN_SL_PCT.get(sym_up, 0.0018)

    min_vix = float(p.get("min_vix", 0.0))
    min_opr_em = float(p.get("min_opr_em", 0.0))
    max_opr_em = float(p.get("max_opr_em", 99.0))
    min_mm_spread_atr = float(p.get("min_mm_spread_atr", 0.0))
    true_breakout_only = int(float(p.get("true_breakout_only", 1))) != 0
    max_fill_bars = int(float(p.get("max_fill_bars", 99)))

    B = pd_.base
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    open_ns, close_ns = B["open_ns"], B["close_ns"]
    n = pd_.n
    i1 = n if i1 is None else min(n, i1)
    fee = fee_pct / 100.0

    _, mins_vec = _tz_vectors(pd_, tz_mode)
    trend_vec, mm_spread_vec = _trend_h1_mm20_50(pd_, str(p.get("trend_tf", "1h")), str(p.get("trend_mode", "mm20_50")))
    j_h1, atr_h1_arr, bear_fvgs, bull_fvgs = _h1_atr_and_fvgs(pd_)
    daily_sessions = _precompute_daily_sessions(pd_, tz_mode)

    end_h = int(float(p.get("end_hour", 19)))
    end_m = int(float(p.get("end_min", 30)))
    if tz_mode == "ny":
        opr_end_min = 9 * 60 + 45
        end_session_min = (end_h - 6) * 60 + end_m
    else:
        opr_end_min = 15 * 60 + 45
        end_session_min = end_h * 60 + end_m
    # L'annulation de fin de séance (19h30 FR) reste active même si on arrête de chercher de nouveaux signaux plus tôt
    hard_end_min = (13 * 60 + 30) if tz_mode == "ny" else (19 * 60 + 30)

    start_search = max(0, i0)
    end_search = min(n, i1)

    signals = []
    for ds in daily_sessions:
        d_start = ds["d_start"]
        d_end = ds["d_end"]
        if d_end <= start_search or d_start >= end_search:
            continue

        opr_i0 = ds["opr_i0"]
        opr_i1 = ds["opr_i1"]
        if opr_i0 < start_search or opr_i1 >= end_search:
            continue

        opr_hi = ds["opr_hi"]
        opr_lo = ds["opr_lo"]
        open_us = ds["open_us"]
        vix_val = ds["vix"]
        em_brut = ds["em_brut"]

        # --- Filtre Anti-Range 1 : Volatilité minimale (évite les marchés morts type 2017 VIX < 14) ---
        if min_vix > 0.0 and vix_val == vix_val and vix_val < min_vix:
            continue

        # --- Filtre Anti-Range 2 : Taille de l'OPR relative à l'Expected Move ---
        if em_brut == em_brut and em_brut > 0.0:
            opr_em_ratio = (opr_hi - opr_lo) / em_brut
            if opr_em_ratio < min_opr_em or opr_em_ratio > max_opr_em:
                continue

        mean_up, std_up = ds["mean_up"], ds["std_up"]
        mean_dn, std_dn = ds["mean_dn"], ds["std_dn"]
        pdh, pdl = ds["pdh"], ds["pdl"]
        psh, psl = ds["psh"], ds["psl"]

        # Niveaux RV3 (A = EM brut, B = Moyenne calibrée 60j, C = Enveloppe +1 sigma)
        if em_brut == em_brut and em_brut > 0:
            a_up = open_us + em_brut
            a_dn = open_us - em_brut
            b_up = open_us + mean_up * em_brut
            b_dn = open_us - mean_dn * em_brut
            c_up = open_us + (mean_up + std_up) * em_brut
            c_dn = open_us - (mean_dn + std_dn) * em_brut
            sl_20pct = em_noise_pct * em_brut
        else:
            a_up = a_dn = b_up = b_dn = c_up = c_dn = sl_20pct = np.nan

        m_after = mins_vec[opr_i1 + 1:min(d_end, end_search)]
        exec_rel = np.flatnonzero((m_after >= opr_end_min) & (m_after < end_session_min))
        if len(exec_rel) < 1:
            continue

        hard_rel = np.flatnonzero((m_after >= opr_end_min) & (m_after < max(end_session_min, hard_end_min)))
        if len(hard_rel) < 2:
            continue

        exec_indices = (opr_i1 + 1) + exec_rel
        f_idx = int((opr_i1 + 1) + hard_rel[-1]) + 1

        pdh_mit = (pdh == pdh) and (ds["high_before_exec"] >= pdh)
        pdl_mit = (pdl == pdl) and (ds["low_before_exec"] <= pdl)
        psh_mit = (psh == psh) and (opr_hi >= psh)
        psl_mit = (psl == psl) and (opr_lo <= psl)

        exited_below = False
        exited_above = False

        for k in exec_indices:
            k = int(k)
            if k + 1 >= f_idx:
                break
            t_dir = int(trend_vec[k])
            ck = float(c[k])
            ok_p = float(o[k])
            prev_c = float(c[k - 1]) if k > 0 else ok_p
            hk = float(h[k])
            lk = float(l[k])

            # --- Filtre Anti-Range 3 : Écart minimal entre MM20 et MM50 H1 (évite les ranges H1 plats) ---
            if min_mm_spread_atr > 0.0 and float(mm_spread_vec[k]) < min_mm_spread_atr:
                if lk < opr_lo:
                    exited_below = True
                if hk > opr_hi:
                    exited_above = True
                if pdh == pdh and hk >= pdh:
                    pdh_mit = True
                if pdl == pdl and lk <= pdl:
                    pdl_mit = True
                if psh == psh and hk >= psh:
                    psh_mit = True
                if psl == psl and lk <= psl:
                    psl_mit = True
                continue

            sig_close_ns = int(close_ns[k])
            jh = int(j_h1[k])
            atr_h1_val = float(atr_h1_arr[jh]) if jh >= 0 else np.nan

            # --- Tendance H1 HAUSSIÈRE (+1) ---
            if t_dir == 1 and side_filter in ("both", "long"):
                cand_setup = None
                cand_entry = np.nan
                is_break_up = (ck > opr_hi) and ((not true_breakout_only) or (prev_c <= opr_hi) or (ok_p <= opr_hi))
                if "Cassure" in allowed_setups and is_break_up:
                    cand_setup = "Cassure"
                    cand_entry = opr_hi
                elif "Reintegration" in allowed_setups and exited_below and (opr_lo <= ck <= opr_hi):
                    cand_setup = "Reintegration"
                    cand_entry = opr_lo

                if cand_setup is not None:
                    sl = _compute_sl_rv3("long", cand_entry, lk, hk, vix_val, atr_h1_val, em_brut, min_sl_pts, min_sl_pct, p)
                    if sl == sl and sl < cand_entry:
                        risk = cand_entry - sl
                        be_dist = (be_rr if be_rr > 0 else 1.0) * risk
                        obs_list = _obstacles_long(cand_entry, b_up, c_up, pdh, pdh_mit, psh, psh_mit, bear_fvgs, sig_close_ns, lookback_ns, res_be_mode)
                        # Condition stricte RV3 : si Distance_Obstacle < Distance_BE -> ANNULER LE TRADE
                        blocked_be = any((obs - cand_entry) < be_dist for obs in obs_list)
                        if not blocked_be:
                            all_res = _obstacles_long(cand_entry, b_up, c_up, pdh, pdh_mit, psh, psh_mit, bear_fvgs, sig_close_ns, lookback_ns, "all")
                            tp, ok_tp = _resolve_tp("long", cand_setup, cand_entry, risk, opr_lo, opr_hi, rr_break, min_rr_reint, max_rr_tp, b_up, c_up, all_res, rv3_tp_mode)
                            if ok_tp:
                                signals.append({
                                    "sig_i": k,
                                    "opr_i0": opr_i0,
                                    "opr_i1": opr_i1,
                                    "f_idx": f_idx,
                                    "sign": -1,
                                    "side": "long",
                                    "entry": cand_entry,
                                    "sl": sl,
                                    "tp": tp,
                                    "opr_lo": opr_lo,
                                    "opr_hi": opr_hi,
                                    "setup": cand_setup,
                                    "open_us": open_us,
                                    "vix": vix_val,
                                    "em_brut": em_brut,
                                    "a_up": a_up,
                                    "a_dn": a_dn,
                                    "b_up": b_up,
                                    "b_dn": b_dn,
                                    "c_up": c_up,
                                    "c_dn": c_dn,
                                    "mean_up": mean_up,
                                    "std_up": std_up,
                                    "mean_dn": mean_dn,
                                    "std_dn": std_dn,
                                    "sl_20pct": sl_20pct,
                                    "pdh": pdh,
                                    "pdl": pdl,
                                    "psh": psh,
                                    "psl": psl,
                                })
                                break

            # --- Tendance H1 BAISSIÈRE (-1) (miroir exact) ---
            elif t_dir == -1 and side_filter in ("both", "short"):
                cand_setup = None
                cand_entry = np.nan
                is_break_dn = (ck < opr_lo) and ((not true_breakout_only) or (prev_c >= opr_lo) or (ok_p >= opr_lo))
                if "Cassure" in allowed_setups and is_break_dn:
                    cand_setup = "Cassure"
                    cand_entry = opr_lo
                elif "Reintegration" in allowed_setups and exited_above and (opr_lo <= ck <= opr_hi):
                    cand_setup = "Reintegration"
                    cand_entry = opr_hi

                if cand_setup is not None:
                    sl = _compute_sl_rv3("short", cand_entry, lk, hk, vix_val, atr_h1_val, em_brut, min_sl_pts, min_sl_pct, p)
                    if sl == sl and sl > cand_entry:
                        risk = sl - cand_entry
                        be_dist = (be_rr if be_rr > 0 else 1.0) * risk
                        obs_list = _obstacles_short(cand_entry, b_dn, c_dn, pdl, pdl_mit, psl, psl_mit, bull_fvgs, sig_close_ns, lookback_ns, res_be_mode)
                        blocked_be = any((cand_entry - obs) < be_dist for obs in obs_list)
                        if not blocked_be:
                            all_sup = _obstacles_short(cand_entry, b_dn, c_dn, pdl, pdl_mit, psl, psl_mit, bull_fvgs, sig_close_ns, lookback_ns, "all")
                            tp, ok_tp = _resolve_tp("short", cand_setup, cand_entry, risk, opr_lo, opr_hi, rr_break, min_rr_reint, max_rr_tp, b_dn, c_dn, all_sup, rv3_tp_mode)
                            if ok_tp:
                                signals.append({
                                    "sig_i": k,
                                    "opr_i0": opr_i0,
                                    "opr_i1": opr_i1,
                                    "f_idx": f_idx,
                                    "sign": 1,
                                    "side": "short",
                                    "entry": cand_entry,
                                    "sl": sl,
                                    "tp": tp,
                                    "opr_lo": opr_lo,
                                    "opr_hi": opr_hi,
                                    "setup": cand_setup,
                                    "open_us": open_us,
                                    "vix": vix_val,
                                    "em_brut": em_brut,
                                    "a_up": a_up,
                                    "a_dn": a_dn,
                                    "b_up": b_up,
                                    "b_dn": b_dn,
                                    "c_up": c_up,
                                    "c_dn": c_dn,
                                    "mean_up": mean_up,
                                    "std_up": std_up,
                                    "mean_dn": mean_dn,
                                    "std_dn": std_dn,
                                    "sl_20pct": sl_20pct,
                                    "pdh": pdh,
                                    "pdl": pdl,
                                    "psh": psh,
                                    "psl": psl,
                                })
                                break

            if lk < opr_lo:
                exited_below = True
            if hk > opr_hi:
                exited_above = True
            if pdh == pdh and hk >= pdh:
                pdh_mit = True
            if pdl == pdl and lk <= pdl:
                pdl_mit = True
            if psh == psh and hk >= psh:
                psh_mit = True
            if psl == psl and lk <= psl:
                psl_mit = True

    # 2) Simulation de chaque ordre limite (annulé si 1R atteint avant exécution, ou délai max dépassé, ou 19h30 atteint)
    raw_trades = []
    for s in signals:
        sig_i = s["sig_i"]
        f_idx = s["f_idx"]
        if sig_i + 1 >= f_idx:
            continue

        sign = s["sign"]
        en = s["entry"]
        sl = s["sl"]
        tp = s["tp"]
        risk_dist = abs(en - sl)
        if risk_dist <= 0:
            continue

        cancel_lvl = (en + cancel_rr * risk_dist) if sign == -1 else (en - cancel_rr * risk_dist)
        fill_limit_end = min(f_idx, i1, sig_i + 1 + max_fill_bars) if max_fill_bars > 0 else min(f_idx, i1)

        fill = -1
        for k in range(sig_i + 1, fill_limit_end):
            if sign == -1:  # Long : BUY LIMIT à en
                if l[k] <= en:
                    fill = k
                    break
                if cancel_rr > 0 and h[k] >= cancel_lvl:
                    break
            else:           # Short : SELL LIMIT à en
                if h[k] >= en:
                    fill = k
                    break
                if cancel_rr > 0 and l[k] <= cancel_lvl:
                    break

        if fill < 0:
            continue

        loss_unit = risk_dist + abs(en) * fee + abs(sl) * fee
        if loss_unit <= 0:
            continue
        if leverage > 0 and ((risk_pct / 100.0) / loss_unit) * abs(en) > leverage:
            continue

        cur_sl = sl
        be_done = False
        xi = i1 - 1
        xp = float(c[xi])
        reason = "END"

        for j in range(fill, i1):
            if sign == -1:  # Long
                if l[j] <= cur_sl:
                    xp = float(min(cur_sl, o[j])) if j > fill else float(cur_sl)
                    reason = "BE" if be_done else "SL"
                    xi = j
                    break
                if j > fill and h[j] >= tp:
                    xp = float(max(tp, o[j]))
                    reason = "TP"
                    xi = j
                    break
                if be_rr > 0 and not be_done and j > fill and h[j] >= en + be_rr * risk_dist:
                    cur_sl = en
                    be_done = True
            else:           # Short
                if h[j] >= cur_sl:
                    xp = float(max(cur_sl, o[j])) if j > fill else float(cur_sl)
                    reason = "BE" if be_done else "SL"
                    xi = j
                    break
                if j > fill and l[j] <= tp:
                    xp = float(min(tp, o[j]))
                    reason = "TP"
                    xi = j
                    break
                if be_rr > 0 and not be_done and j > fill and l[j] <= en - be_rr * risk_dist:
                    cur_sl = en
                    be_done = True

        gross_per_unit = (xp - en) if sign == -1 else (en - xp)
        fee_per_unit = (abs(en) + abs(xp)) * fee
        r_mult = (gross_per_unit - fee_per_unit) / loss_unit

        opr_i0 = s["opr_i0"]
        f_i = min(s["f_idx"] - 1, n - 1)

        raw_trades.append({
            "fill_i": fill,
            "exit_i": xi,
            "entry_time": pd.Timestamp(int(open_ns[fill]), tz="UTC"),
            "exit_time": pd.Timestamp(int(open_ns[xi]), tz="UTC"),
            "signal_time": pd.Timestamp(int(open_ns[sig_i]), tz="UTC"),
            "side": s["side"],
            "entry": en,
            "exit": xp,
            "sl": sl,
            "tp": tp,
            "reason": reason,
            "r": float(r_mult),
            "bars": int(xi - fill),
            "session": s["setup"],
            "conf_tf": f"5m ({s['setup']})",
            "zone_lo": s["opr_lo"],
            "zone_hi": s["opr_hi"],
            "zone_form": pd.Timestamp(int(open_ns[opr_i0]), tz="UTC"),
            "zone_end": pd.Timestamp(int(close_ns[f_i]), tz="UTC"),
            "open_us": s["open_us"],
            "vix": s["vix"],
            "em_brut": s["em_brut"],
            "a_up": s["a_up"],
            "a_dn": s["a_dn"],
            "b_up": s["b_up"],
            "b_dn": s["b_dn"],
            "c_up": s["c_up"],
            "c_dn": s["c_dn"],
            "mean_up": s["mean_up"],
            "std_up": s["std_up"],
            "mean_dn": s["mean_dn"],
            "std_dn": s["std_dn"],
            "sl_20pct": s["sl_20pct"],
            "pdh": s["pdh"],
            "pdl": s["pdl"],
            "psh": s["psh"],
            "psl": s["psl"],
        })

    # 3) Courbe d'équité et PnL en $
    eq = np.full(n, float(capital), dtype=np.float64)
    bal = float(capital)
    eqpos = 0
    open_q = []

    for t in raw_trades:
        fi = t["fill_i"]
        open_q.sort(key=lambda x: x[0])
        while open_q and open_q[0][0] <= fi:
            xi_done, pnl_done = open_q.pop(0)
            if xi_done >= eqpos:
                eq[eqpos:xi_done] = bal
                bal += pnl_done
                eq[xi_done] = bal
                eqpos = xi_done + 1
            else:
                bal += pnl_done
        if fi > eqpos:
            eq[eqpos:fi] = bal
            eqpos = fi
        risk_cash = bal * (risk_pct / 100.0)
        pnl_cash = t["r"] * risk_cash
        t["pnl"] = float(pnl_cash)
        open_q.append((t["exit_i"], pnl_cash))

    open_q.sort(key=lambda x: x[0])
    while open_q:
        xi_done, pnl_done = open_q.pop(0)
        if xi_done >= eqpos:
            eq[eqpos:xi_done] = bal
            bal += pnl_done
            eq[xi_done] = bal
            eqpos = xi_done + 1
        else:
            bal += pnl_done
    eq[eqpos:] = bal

    for t in raw_trades:
        t.pop("fill_i", None)
        t.pop("exit_i", None)

    return pd.DataFrame(raw_trades), eq[i0:i1]


def setup_stats(tr: pd.DataFrame, risk_pct: float, capital: float = 10000.0):
    """
    Calcule les statistiques détaillées par type de setup OPR (Cassure vs Reintegration).
    """
    out = {}
    for sname in SETUP_ORDER:
        if tr is None or len(tr) == 0 or "session" not in tr.columns:
            sub = pd.DataFrame()
        else:
            sub = tr[tr["session"] == sname].sort_values("entry_time")
        if len(sub) == 0:
            out[sname] = {
                "return_pct": 0.0,
                "max_dd_pct": 0.0,
                "trades": 0,
                "win_pct": 0.0,
                "avg_r": 0.0,
                "sum_r": 0.0,
            }
            continue
        r_arr = sub["r"].to_numpy(float)
        eq = np.empty(len(r_arr) + 1, dtype=float)
        eq[0] = capital
        for k, rv in enumerate(r_arr):
            eq[k + 1] = eq[k] * (1.0 + rv * (risk_pct / 100.0))
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100.0)
        ret = float((eq[-1] / capital - 1.0) * 100.0)
        out[sname] = {
            "return_pct": ret,
            "max_dd_pct": dd,
            "trades": int(len(r_arr)),
            "win_pct": float((r_arr > 0).mean() * 100.0),
            "avg_r": float(r_arr.mean()),
            "sum_r": float(r_arr.sum()),
        }
    return out
