"""
backtest/strategy_opr.py — Stratégie OPR (Opening Price Range) sur les indices US (NAS100 / US100, USA500 / US500, US30).

Règles exactes :
  1. Zone OPR :
     - Définie par la bougie M15 d'ouverture US : 15h30 -> 15h45 heure française (`Europe/Paris`),
       c.-à-d. les 3 bougies M5 [15h30, 15h35, 15h40].
     - OPR High = plus haut (mèche incluse) de cette bougie M15.
     - OPR Low  = plus bas (mèche incluse) de cette bougie M15.
  2. Filtre obligatoire de tendance en H1 (`mm20_50`) :
     - Tendance haussière (+1) : MM20 > MM50 en H1 ET clôture de la bougie H1 en cours > MM50.
     - Tendance baissière (-1) : MM20 < MM50 en H1 ET clôture de la bougie H1 en cours < MM50.
  3. Unité d'exécution : M5, de 15h45 (clôture de l'OPR) jusqu'à 19h30 heure française :
     - Maximum 1 signal / ordre posé par jour et par indice (le premier des 2 setups valides qui se présente).
     - Aucun nouveau trade ou ordre n'est pris après 19h30 heure française si le prix n'est pas déjà rentré en position.
  4. Placement du SL selon l'indice de volatilité associé (VXNCLS-US100, VIXCLS-US500, VXDCLS-US30) :
     - Soit `vix` la valeur de l'indice de volatilité et `atr_h1` l'ATR(14) calculé sur les 14 dernières bougies H1 clôturées :
       * Si `vix > 20` :
         - Long  : SL = low[bougie_signal] - atr_h1
         - Short : SL = high[bougie_signal] + atr_h1
       * Si `15 <= vix <= 20` :
         - Seuil minimal selon l'actif : US30 = 80 pts, NAS100/US100 = 70 pts, USA500/US500 = 50 pts.
         - Distance `inv = abs(entry - mèche_bougie_signal)` -> `sl_dist = max(inv, seuil_min)`.
         - Long  : SL = entry - sl_dist
         - Short : SL = entry + sl_dist
       * Si `vix < 15` :
         - Long  : SL = low[bougie_signal]
         - Short : SL = high[bougie_signal]
  5. Résistances / Supports potentiels (non mitigés / non franchis avant la bougie de signal) :
     - Pour un achat (Long), les résistances potentielles au-dessus de `entry` sont :
       * Previous Daily High (`PDH`) : plus haut de la journée de trading précédente (00h00 -> 23h59 heure FR), s'il n'a pas encore été franchi aujourd'hui avant le signal.
       * Haut de session précédente (`PSH`) : plus haut de la pré-session du jour (00h00 -> 15h30 heure FR, englobant Asie + Londres), s'il n'a pas encore été franchi depuis 15h30 avant le signal.
       * FVG H1 baissière non mitigée (`high[q+2] < low[q]` en H1) : bord bas `high[q+2]` d'une FVG H1 baissière formée avant le signal et jamais retouchée depuis sa formation.
     - Miroir exact en vente (Short) avec Previous Daily Low (`PDL`), Bas de session précédente (`PSL` 00h00 -> 15h30 FR) et bord haut (`low[q+2]`) d'une FVG H1 haussière non mitigée.
     - RÈGLE D'INVALIDATION : le trade est invalide s'il existe au moins une résistance potentielle située entre `entry` et `+1 R` (`entry < res <= entry + 1.0 * risk` en Long ; `entry - 1.0 * risk <= sup < entry` en Short).
  6. Règle de Take Profit (TP) :
     - Setup 1 (Cassure) : TP de `2.5 R` minimal, ou plus loin s'il existe une résistance potentielle non mitigée au-delà de `2.5 R` (dans la limite d'un plafond raisonnable `max_rr_target`, 8 R par défaut : on vise alors la première résistance `>= 2.5 R`).
     - Setup 2 (Réintégration) : TP sur la borne opposée de l'OPR (`OPR High` en Long, `OPR Low` en Short), valide uniquement si ce TP offre un RR >= `min_rr_reint` (2.5 R).
  7. Gestion de l'ordre limite et du trade :
     - L'ordre limite est annulé s'il n'a pas été déclenché alors que le prix atteint `+1 R` depuis l'entrée, ou à 19h30 heure française.
     - Passage à Break-Even (SL déplacé à `entry`) dès que le trade ouvert atteint `+1 R` (`be_rr = 1.0`).
"""
import gzip
import os
import numpy as np
import pandas as pd

SETUP_ORDER = ["Cassure", "Reintegration"]

# Plancher minimal de SL en points lorsque 15 <= indice de volatilité <= 20
MIN_SL_POINTS = {
    "US30": 80.0,
    "NAS100": 70.0,
    "US100": 70.0,
    "USA500": 50.0,
    "US500": 50.0,
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
    "rr_break": 2.5,         # TP minimal du setup Cassure (en R)
    "min_rr_reint": 2.5,     # RR minimal exigé pour le setup Réintégration (TP = borne opposée de l'OPR)
    "max_rr_tp": 8.0,        # plafond RR pour aller chercher une résistance plus lointaine que 2.5 R
    "cancel_rr": 1.0,        # annulation de l'ordre limite non déclenché si le prix touche +cancel_rr R
    "be_rr": 1.0,            # passage à Break-Even à +be_rr R (0 = désactivé)
    "vix_low": 15.0,         # seuil bas de l'indice de volatilité
    "vix_high": 20.0,        # seuil haut de l'indice de volatilité
    "use_res_filter": 1,     # 1 = invalide le trade si une résistance (PDH, PSH, FVG H1) bloque avant +1 R
    "use_res_tp": 1,         # 1 = étend le TP sur une résistance située au-delà de 2.5 R si elle existe
    "fvg_h1_lookback_days": 30,  # nombre de jours max en arrière pour les FVG H1 non mitigées
    "trend_tf": "1h",        # unité de temps du filtre de tendance
    "trend_mode": "mm20_50", # "mm20_50" (SMA20/50) ou "ema_mm20_50" (EMA20/50)
    "tz_mode": "paris",      # "paris" (15h30-19h30 Europe/Paris) ou "ny" (09h30-13h30 America/New_York)
    "end_hour": 19,          # heure de fin de prise de position (19h30 par défaut en mode paris)
    "end_min": 30,
    "side": "both",          # "both", "long", "short"
    "setups": "Cassure,Reintegration",
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
    """
    Charge l'indice de volatilité quotidien (VIXCLS / VXNCLS / VXDCLS) associé à `pd_.symbol`
    et renvoie un dictionnaire `{yyyymmdd_int: vix_float}` avec propagation (forward-fill)
    sur les jours fériés / week-ends.
    Pour éviter tout biais du futur (la clôture VIXCLS du jour J n'étant connue qu'à la clôture US),
    ainsi que pour coller à la valeur courante de la journée, on associe à chaque date la dernière
    valeur disponible <= date.
    """
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
    """Retourne la valeur de l'indice de volatilité pour `day_int` (YYYYMMDD)."""
    if len(sorted_days) == 0:
        return np.nan
    pos = int(np.searchsorted(sorted_days, day_int, side="right")) - 1
    if pos < 0:
        return float(sorted_vals[0])
    return float(sorted_vals[pos])


def _h1_atr_and_fvgs(pd_):
    """
    Précalcule sur l'unité H1 :
      - `j_h1[i]` : indice de la dernière bougie H1 clôturée à la fin de la bougie M5 `i`.
      - `atr_h1`  : ATR(14) des bougies H1.
      - `bear_fvgs` : liste des FVG H1 baissières `(form_h1_idx, form_close_ns, fvg_lo, fvg_hi, mit_close_ns)`
                      où `fvg_lo = high[q+2]`, `fvg_hi = low[q]` (avec `high[q+2] < low[q]`).
                      Une FVG H1 baissière (résistance pour un Long) est mitigée dès qu'une bougie ultérieure
                      touche `fvg_lo` (`high >= fvg_lo`).
      - `bull_fvgs` : liste des FVG H1 haussières `(form_h1_idx, form_close_ns, fvg_lo, fvg_hi, mit_close_ns)`
                      où `fvg_lo = high[q]`, `fvg_hi = low[q+2]` (avec `low[q+2] > high[q]`).
                      Une FVG H1 haussière (support pour un Short) est mitigée dès qu'une bougie ultérieure
                      touche `fvg_hi` (`low <= fvg_hi`).
    """
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

    # Détection des FVG H1 et calcul de l'instant exact (en nanosecondes sur les bougies M5) où elles sont touchées pour la 1re fois
    bear_q = np.flatnonzero(hh[2:] < hl[:-2])
    bull_q = np.flatnonzero(hl[2:] > hh[:-2])

    # Pour accélérer la recherche de mitigation sur M5, on commence par scanner en H1 puis on affine en M5 dans la bougie H1 qui touche
    bear_fvgs = []
    for q in bear_q:
        form_idx = int(q + 2)
        form_ns = int(h_close_ns[form_idx])
        f_lo = float(hh[form_idx])  # bord bas de la FVG H1 baissière = 1er obstacle pour un prix qui monte
        f_hi = float(hl[q])
        mit_ns = np.iinfo(np.int64).max
        for m in range(form_idx + 1, nh):
            if hh[m] >= f_lo:
                # Affine à la bougie M5 exacte à l'intérieur de la bougie H1 `m`
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
        f_hi = float(hl[form_idx])  # bord haut de la FVG H1 haussière = 1er obstacle pour un prix qui descend
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
    """
    Filtre de tendance H1 (mm20_50) :
      +1 (haussier) si MM20 > MM50 et close > MM50 en H1
      -1 (baissier) si MM20 < MM50 et close < MM50 en H1
       0 sinon.
    """
    key = ("opr_trend", tf, mode)
    if key in pd_.cache:
        return pd_.cache[key]

    f = pd_.frames[tf]
    hc = f["c"]
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
        with np.errstate(invalid="ignore"):
            bull = (np.arange(nh) >= 20) & (ema20 > ema50) & (hc > ema50)
            bear = (np.arange(nh) >= 20) & (ema20 < ema50) & (hc < ema50)
        st_h[bull] = 1
        st_h[bear] = -1
        mapped = np.where(j >= 0, st_h[np.clip(j, 0, max(0, nh - 1))], 0).astype(np.int8)
        pd_.cache[key] = mapped
        return mapped

    csum = np.concatenate([[0.0], np.cumsum(hc, dtype=np.float64)])
    exact_close = np.zeros(nb, dtype=bool)
    valid_j = j >= 0
    exact_close[valid_j] = f["close_ns"][j[valid_j]] == b_close_ns[valid_j]

    mapped = np.zeros(nb, dtype=np.int8)
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

    pd_.cache[key] = mapped
    return mapped


def _compute_sl(side, entry, candle_low, candle_high, vix_val, atr_h1_val, min_sl_pts, vix_low, vix_high):
    """
    Calcule le SL selon les 3 paliers d'indice de volatilité :
      - vix > vix_high (20)        : sous/au-dessus de la mèche de la bougie M5 +/- ATR(14) H1
      - vix_low (15) <= vix <= 20  : max(inv, seuil_min_actif) depuis l'entrée
      - vix < vix_low (15)         : sous/au-dessus de la mèche de la bougie M5
    """
    atr_ok = (atr_h1_val == atr_h1_val) and (atr_h1_val > 0)
    vix_ok = (vix_val == vix_val) and (vix_val > 0)

    if side == "long":
        inv = entry - candle_low
        if inv <= 0:
            return np.nan
        if vix_ok and vix_val > vix_high and atr_ok:
            return float(candle_low - atr_h1_val)
        elif vix_ok and vix_val >= vix_low:
            dist = max(inv, float(min_sl_pts))
            return float(entry - dist)
        else:
            return float(candle_low)
    else:
        inv = candle_high - entry
        if inv <= 0:
            return np.nan
        if vix_ok and vix_val > vix_high and atr_ok:
            return float(candle_high + atr_h1_val)
        elif vix_ok and vix_val >= vix_low:
            dist = max(inv, float(min_sl_pts))
            return float(entry + dist)
        else:
            return float(candle_high)


def _active_resistances_long(entry, pdh, pdh_mit, psh, psh_mit, bear_fvgs, sig_close_ns, lookback_ns):
    """
    Retourne la liste triée des résistances potentielles non mitigées strictement au-dessus de `entry` (pour un Long).
    """
    res = []
    if pdh == pdh and not pdh_mit and pdh > entry:
        res.append(float(pdh))
    if psh == psh and not psh_mit and psh > entry:
        res.append(float(psh))
    if len(bear_fvgs) > 0:
        # bear_fvgs colonnes : [0]=form_ns, [1]=fvg_lo (bord bas = obstacle), [2]=fvg_hi, [3]=mit_ns
        min_form = sig_close_ns - lookback_ns
        m = (bear_fvgs[:, 0] <= sig_close_ns) & (bear_fvgs[:, 0] >= min_form) & (bear_fvgs[:, 3] > sig_close_ns) & (bear_fvgs[:, 1] > entry)
        if np.any(m):
            res.extend(bear_fvgs[m, 1].tolist())
    res.sort()
    return res


def _active_supports_short(entry, pdl, pdl_mit, psl, psl_mit, bull_fvgs, sig_close_ns, lookback_ns):
    """
    Retourne la liste triée (du plus proche au plus bas, donc par ordre décroissant) des supports potentiels
    non mitigés strictement en dessous de `entry` (pour un Short).
    """
    sup = []
    if pdl == pdl and not pdl_mit and pdl < entry:
        sup.append(float(pdl))
    if psl == psl and not psl_mit and psl < entry:
        sup.append(float(psl))
    if len(bull_fvgs) > 0:
        # bull_fvgs colonnes : [0]=form_ns, [1]=fvg_lo, [2]=fvg_hi (bord haut = obstacle), [3]=mit_ns
        min_form = sig_close_ns - lookback_ns
        m = (bull_fvgs[:, 0] <= sig_close_ns) & (bull_fvgs[:, 0] >= min_form) & (bull_fvgs[:, 3] > sig_close_ns) & (bull_fvgs[:, 2] < entry)
        if np.any(m):
            sup.extend(bull_fvgs[m, 2].tolist())
    sup.sort(reverse=True)
    return sup


def run(pd_, capital=10000.0, risk_pct=0.5, fee_pct=0.005, leverage=30.0, p=None, i0=0, i1=None):
    """
    Exécute le backtest OPR sur `pd_` dans l'intervalle de bougies 5m `[i0, i1)`.
    Retourne `(trades_df, equity_array_on_slice)`.
    """
    p = dict(PARAMS if p is None else p)
    rr_break = float(p.get("rr_break", 2.5))
    min_rr_reint = float(p.get("min_rr_reint", 2.5))
    max_rr_tp = float(p.get("max_rr_tp", 8.0))
    cancel_rr = float(p.get("cancel_rr", 1.0))
    be_rr = float(p.get("be_rr", 1.0))
    vix_low = float(p.get("vix_low", 15.0))
    vix_high = float(p.get("vix_high", 20.0))
    use_res_filter = int(p.get("use_res_filter", 1))
    use_res_tp = int(p.get("use_res_tp", 1))
    lookback_ns = int(float(p.get("fvg_h1_lookback_days", 30)) * 86400 * 10**9)
    tz_mode = str(p.get("tz_mode", "paris")).lower()
    side_filter = str(p.get("side", "both")).lower()
    allowed_setups = {s.strip() for s in str(p.get("setups", "Cassure,Reintegration")).split(",") if s.strip()}

    sym_up = str(getattr(pd_, "symbol", "NAS100")).upper()
    min_sl_pts = MIN_SL_POINTS.get(sym_up, 50.0)

    B = pd_.base
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    open_ns, close_ns = B["open_ns"], B["close_ns"]
    n = pd_.n
    i1 = n if i1 is None else min(n, i1)
    fee = fee_pct / 100.0

    day_vec, mins_vec = _tz_vectors(pd_, tz_mode)
    trend_vec = _trend_h1_mm20_50(pd_, str(p.get("trend_tf", "1h")), str(p.get("trend_mode", "mm20_50")))
    sorted_vdays, sorted_vvals = _load_vol_series(pd_)
    j_h1, atr_h1_arr, bear_fvgs, bull_fvgs = _h1_atr_and_fvgs(pd_)

    if tz_mode == "ny":
        opr_start_min = 9 * 60 + 30    # 09h30 NY
        opr_end_min = 9 * 60 + 45      # 09h45 NY
        end_session_min = 13 * 60 + 30 # 13h30 NY
    else:
        opr_start_min = 15 * 60 + 30   # 15h30 Paris
        opr_end_min = 15 * 60 + 45     # 15h45 Paris
        end_session_min = int(p.get("end_hour", 19)) * 60 + int(p.get("end_min", 30))  # 19h30 Paris

    start_search = max(0, i0)
    end_search = min(n, i1)

    signals = []
    idx = start_search
    prev_day_hi = np.nan
    prev_day_lo = np.nan

    # Si i0 > 0, initialiser prev_day_hi / prev_day_lo sur la journée précédant i0
    if start_search > 0:
        first_d = int(day_vec[start_search])
        p_end = start_search
        while p_end > 0 and int(day_vec[p_end - 1]) == first_d:
            p_end -= 1
        if p_end > 0:
            prev_d = int(day_vec[p_end - 1])
            p_start = p_end - 1
            while p_start > 0 and int(day_vec[p_start - 1]) == prev_d:
                p_start -= 1
            prev_day_hi = float(np.max(h[p_start:p_end]))
            prev_day_lo = float(np.min(l[p_start:p_end]))

    while idx < end_search:
        cur_day = int(day_vec[idx])
        day_start = idx
        while idx < end_search and int(day_vec[idx]) == cur_day:
            idx += 1
        day_end = idx  # [day_start, day_end) = toutes les bougies 5m de ce jour

        pdh = prev_day_hi
        pdl = prev_day_lo
        # Mettre à jour pour le jour suivant avec le plus haut / plus bas complet du jour courant
        prev_day_hi = float(np.max(h[day_start:day_end]))
        prev_day_lo = float(np.min(l[day_start:day_end]))

        # 1) Identifier les bougies M5 de la pré-session (00h00 -> 15h30 FR) et de l'OPR (15h30 -> 15h45 FR)
        pre_indices = [
            k for k in range(day_start, day_end)
            if int(mins_vec[k]) < opr_start_min
        ]
        opr_indices = [
            k for k in range(day_start, day_end)
            if opr_start_min <= int(mins_vec[k]) < opr_end_min
        ]
        if len(opr_indices) < 3:
            continue

        opr_i0 = opr_indices[0]
        opr_i1 = opr_indices[-1]
        opr_hi = float(np.max(h[opr_indices]))
        opr_lo = float(np.min(l[opr_indices]))
        if opr_hi <= opr_lo or np.isnan(opr_hi) or np.isnan(opr_lo):
            continue

        # Haut / Bas de la session précédente (00h00 -> 15h30 heure FR : Asie + Londres)
        psh = float(np.max(h[pre_indices])) if pre_indices else np.nan
        psl = float(np.min(l[pre_indices])) if pre_indices else np.nan

        exec_indices = [
            k for k in range(opr_i1 + 1, day_end)
            if opr_end_min <= int(mins_vec[k]) < end_session_min
        ]
        if len(exec_indices) < 2:
            continue

        f_idx = exec_indices[-1] + 1
        vix_val = _vix_for_day(sorted_vdays, sorted_vvals, cur_day)

        # Suivi de la mitigation de PDH/PDL (depuis 00h00) et de PSH/PSL (depuis 15h30)
        high_before_exec = float(np.max(h[day_start:opr_i1 + 1]))
        low_before_exec = float(np.min(l[day_start:opr_i1 + 1]))
        pdh_mit = (pdh == pdh) and (high_before_exec >= pdh)
        pdl_mit = (pdl == pdl) and (low_before_exec <= pdl)
        psh_mit = (psh == psh) and (opr_hi >= psh)
        psl_mit = (psl == psl) and (opr_lo <= psl)

        exited_below = False
        exited_above = False

        for k in exec_indices[:-1]:
            t_dir = int(trend_vec[k])
            ck = float(c[k])
            hk = float(h[k])
            lk = float(l[k])
            sig_close_ns = int(close_ns[k])
            jh = int(j_h1[k])
            atr_h1_val = float(atr_h1_arr[jh]) if jh >= 0 else np.nan

            # --- Tendance H1 HAUSSIÈRE (+1) ---
            if t_dir == 1 and side_filter in ("both", "long"):
                cand_setup = None
                cand_entry = np.nan
                if "Cassure" in allowed_setups and ck > opr_hi:
                    cand_setup = "Cassure"
                    cand_entry = opr_hi
                elif "Reintegration" in allowed_setups and exited_below and (opr_lo <= ck <= opr_hi):
                    cand_setup = "Reintegration"
                    cand_entry = opr_lo

                if cand_setup is not None:
                    sl = _compute_sl("long", cand_entry, lk, hk, vix_val, atr_h1_val, min_sl_pts, vix_low, vix_high)
                    if sl == sl and sl < cand_entry:
                        risk = cand_entry - sl
                        res_list = _active_resistances_long(cand_entry, pdh, pdh_mit, psh, psh_mit, bear_fvgs, sig_close_ns, lookback_ns)
                        # Invalidation s'il existe une résistance potentielle avant d'atteindre +1 R
                        blocked_1r = use_res_filter and any(r <= cand_entry + 1.0 * risk for r in res_list)
                        if not blocked_1r:
                            if cand_setup == "Cassure":
                                tp = cand_entry + rr_break * risk
                                if use_res_tp:
                                    far_res = [r for r in res_list if cand_entry + rr_break * risk <= r <= cand_entry + max_rr_tp * risk]
                                    if far_res:
                                        tp = float(far_res[0])
                                valid_tp = True
                            else:
                                tp = opr_hi
                                valid_tp = (tp - cand_entry) >= min_rr_reint * risk

                            if valid_tp:
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
                                })
                                break

            # --- Tendance H1 BAISSIÈRE (-1) (miroir exact) ---
            elif t_dir == -1 and side_filter in ("both", "short"):
                cand_setup = None
                cand_entry = np.nan
                if "Cassure" in allowed_setups and ck < opr_lo:
                    cand_setup = "Cassure"
                    cand_entry = opr_lo
                elif "Reintegration" in allowed_setups and exited_above and (opr_lo <= ck <= opr_hi):
                    cand_setup = "Reintegration"
                    cand_entry = opr_hi

                if cand_setup is not None:
                    sl = _compute_sl("short", cand_entry, lk, hk, vix_val, atr_h1_val, min_sl_pts, vix_low, vix_high)
                    if sl == sl and sl > cand_entry:
                        risk = sl - cand_entry
                        sup_list = _active_supports_short(cand_entry, pdl, pdl_mit, psl, psl_mit, bull_fvgs, sig_close_ns, lookback_ns)
                        # Invalidation s'il existe un support potentiel avant d'atteindre +1 R
                        blocked_1r = use_res_filter and any(s_lvl >= cand_entry - 1.0 * risk for s_lvl in sup_list)
                        if not blocked_1r:
                            if cand_setup == "Cassure":
                                tp = cand_entry - rr_break * risk
                                if use_res_tp:
                                    far_sup = [s_lvl for s_lvl in sup_list if cand_entry - max_rr_tp * risk <= s_lvl <= cand_entry - rr_break * risk]
                                    if far_sup:
                                        tp = float(far_sup[0])
                                valid_tp = True
                            else:
                                tp = opr_lo
                                valid_tp = (cand_entry - tp) >= min_rr_reint * risk

                            if valid_tp:
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
                                })
                                break

            # Mise à jour de l'état de sortie hors de l'OPR et de la mitigation des niveaux sur la bougie `k`
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

    # 2) Simulation de chaque ordre limite (annulé si 1R atteint avant exécution ou si 19h30 atteint)
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

        fill = -1
        for k in range(sig_i + 1, min(f_idx, i1)):
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
