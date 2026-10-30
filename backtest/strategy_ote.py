# backtest/strategy.py
# Stratégie : tendance H4 + retracement (zone OTE) + confirmation (bulle apesanteur ou MSS) sur 5m / 15m / 1h.
#
# Raisonnement écrit pour une tendance BAISSIÈRE (position short). Le cas haussier est obtenu en
# miroir (prix inversés) : exactement la même logique, donc aucune différence de traitement.
#
# Principe, dans l'ordre :
#  1) TENDANCE H4 baissière (loi de Dow = creux et sommets de plus en plus bas, et/ou moyenne mobile).
#  2) RETRACEMENT : on repère la dernière jambe baissière (dernier sommet confirmé -> plus bas),
#     et on attend que le prix remonte dans la zone OTE (61,8 % - 78,6 % de cette jambe).
#     Les jambes sont calculées sur H4 et sur H1 (plus de trades).
#  3) CONFIRMATION dans la zone, sur 5m, 15m ou 1h :
#       - "bubble" : bulle apesanteur HAUTE (corps entièrement au-dessus de la bande de Bollinger haute)
#                    suivie d'une bougie BAISSIÈRE  -> entrée à la clôture de cette bougie ;
#       - "mss"    : le prix balaie un sommet (sweep) puis clôture sous le creux précédent (cassure de
#                    structure), et cette cassure est plus rapide que la montée qui a fait le sweep.
#  4) SL au plus haut local du retracement (+ petite marge), TP dans la zone OTE du mouvement de
#     retracement (par défaut au niveau 70,5 % de ce mouvement).
#
# Contrat avec le moteur : prepare(df) calcule tout (sans regarder le futur), signal(d, i) lit le résultat.
# Les paramètres se modifient ici, ou depuis le workflow GitHub (champ « params »), ex :
#   trend_mode=ma;tp_fib=0.62;confirms=bubble

import numpy as np
import pandas as pd

try:                                   # Numba (si installé) accélère fortement le MSS ; sinon le même code tourne en Python pur
    from numba import njit as _njit
except ImportError:                    # pragma: no cover
    def _njit(f):
        return f

PARAMS = {
    # --- tendance ---
    "trend_tf": "4h",
    "trend_mode": "both",        # "dow" (structure) | "ma" (moyenne mobile) | "both" (les deux d'accord)
    "ma_len": 20,               # EMA de tendance (mode ma / both)
    "ma_slope_bars": 5,         # la pente de l'EMA est mesurée sur N bougies H4
    # --- swings (ZigZag) : un retournement est confirmé quand le prix repart de N x ATR ---
    "atr_len": 14,
    "zz_atr_4h": 3.8115,
    "zz_atr_1h": 2.4454,
    "zz_atr_15m": 3.0,
    # --- retracement ---
    "range_tfs": "4h",       # jambes utilisées pour la zone OTE
    "trend_filter": 0,          # 1 = on ne trade que dans le sens de la tendance ; 0 = pas de filtre de tendance
    "zz_atr_trend": 7.9626,        # ZigZag de la TENDANCE (plus grand = tendance plus lente / plus longue)
    "range_origin": "leg",      # "leg" = le retracement part de l'ORIGINE du mouvement complet ; "pivot" = dernier pivot seulement
    "leg_lookback": 300,        # en "leg" : on remonte au plus de N bougies de l'unité du range
    "ote_lo": 0.618,
    "ote_hi": 0.706,
    "ote_react": 0,             # 1 = un retracement dont la zone OTE a DÉJÀ réagi est invalide ; 0 = règle désactivée
    "ote_react_level": 0.5378,     # niveau (en fraction du range, 0 = bas, 1 = haut) que le prix doit retoucher après la réaction
    # --- confirmations ---
    "confirms": "mss",   # "bubble", "mss" ou les deux
    "conf_tfs": "5m,15m,1h",    # unités de temps de confirmation (>= unité de base du backtest)
    "bb_len": 20,
    "bb_mult": 2.0,
    # bulle validée = corps hors bande ET distance >= seuil (en % du prix). Seuils tirés de ton tableau :
    "bubble_min_pct_5m": 0.0398,
    "bubble_min_pct_15m": 0.3388,
    "bubble_min_pct_1h": 0.4054,
    "bubble_min_sigma": 0.0,    # seuil optionnel en écarts-types (0 = désactivé)
    "mss_fractal": 4,           # un creux/sommet est confirmé après N bougies de chaque côté
    "mss_trough_pivot": 1,      # 0 = le creux L2 est simplement le plus bas entre H1 et H2 ; 1 = il doit aussi être confirmé sur N bougies
    "mss_min_sweep_pct": 0.0307,   # H2 doit dépasser H1 d'au moins ce % (0 = n'importe quel dépassement)
    "min_sl_pct": 0.0,          # distance minimale du stop en % du prix d'entrée (0 = pas de minimum)
    "mss_speed_ratio": 1.5,     # cassure en <= ratio x (durée de la montée du sweep)
    "max_time_ratio": 0.0,      # ratio max (durée impulsion Fib / durée retracement vers MSS) ; 0 = désactivé
    "entry_mode": "fvg_15m",    # "mss" (clôture MSS) | "fvg_15m" (FVG 15m entre H2 et MSS) | "fvg_15m_or_mss" | "fvg_conf"
    "entry_lvl": 0.0,           # niveau d'entrée dans la FVG (0.0 = bord proche du MSS, 0.5 = milieu)
    "fvg_min_atr": 0.0,         # taille minimale de la FVG d'entrée en ATR
    "ttl": 96,                  # durée de validité de l'ordre limit en bougies 5m (96 = 8h)
    # --- sortie ---
    "sl_buf_atr": 0.3407,          # marge au-dessus du plus haut local, en ATR de l'unité de confirmation
    "tp_fib": 3.0,            # TP = retracement de ce niveau du mouvement de retracement (0.618 à 0.786)
    "min_rr": 0.1101,              # ratio gain/risque minimum (0 = pas de filtre)
    "side": "long",             # "both" | "long" | "short"
    "be_rr": 1000.0,            # break-even : SL placé au prix d'entrée dès que le gain atteint X fois le risque (1000 = désactivé)
    "session_start": 0,         # heures UTC d'entrée autorisées [start, end)
    "session_end": 24,
}

ENGINE = {"allow_short": True, "max_bars": None}

TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
NS_MIN = 60 * 10**9


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


# ------------------------------------------------------------------ données multi-unités de temps

def make_frame(df, minutes, base_min):
    """Bougies de l'unité demandée, uniquement les bougies COMPLÈTES (ou >=60% pour 4h/1d sur forex/indices), avec leurs heures d'ouverture / clôture."""
    g = df.set_index("time")
    if minutes == base_min:
        f = g[["open", "high", "low", "close"]].copy()
    else:
        f = g.resample(f"{minutes}min").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                           close=("close", "last"), n=("close", "count"))
        full = min(minutes // base_min, float(np.quantile(f["n"].dropna(), 0.9))) if len(f) else minutes // base_min
        min_fill = 1.0 if minutes <= 15 else 0.6
        f = f[f["n"] >= max(1, int(np.ceil(full * min_fill)))][["open", "high", "low", "close"]]
    open_ns = pd.DatetimeIndex(f.index).as_unit("ns").asi8
    return {"o": f["open"].to_numpy(float), "h": f["high"].to_numpy(float), "l": f["low"].to_numpy(float),
            "c": f["close"].to_numpy(float), "open_ns": open_ns, "close_ns": open_ns + minutes * NS_MIN}


def add_indicators(f, p, tf):
    o, h, l, c = f["o"], f["h"], f["l"], f["c"]
    atr_key = ("atr", int(p["atr_len"]))
    if atr_key not in f:
        pc = np.concatenate([[np.nan], c[:-1]])
        tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
        f[atr_key] = pd.Series(tr).rolling(int(p["atr_len"])).mean().to_numpy()
    f["atr"] = f[atr_key]
    bb_key = ("bb", int(p["bb_len"]), float(p["bb_mult"]))
    if bb_key not in f:
        s = pd.Series(c)
        mid = s.rolling(int(p["bb_len"])).mean()
        sd = s.rolling(int(p["bb_len"])).std(ddof=0).to_numpy()  # écart-type de population, comme TradingView
        up, dn = (mid + p["bb_mult"] * sd).to_numpy(), (mid - p["bb_mult"] * sd).to_numpy()
        f[bb_key] = (up, dn, sd)
    up, dn, sd = f[bb_key]
    f["bb_up"], f["bb_dn"] = up, dn
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    thr = p.get(f"bubble_min_pct_{tf}", 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_up, d_dn = body_lo - up, dn - body_hi
        f["bub_up"] = (d_up > 0) & (d_up / body_lo * 100 >= thr) & (d_up / sd >= p["bubble_min_sigma"])
        f["bub_dn"] = (d_dn > 0) & (d_dn / body_hi * 100 >= thr) & (d_dn / sd >= p["bubble_min_sigma"])


# ------------------------------------------------------------------ swings (ZigZag causal)

def zigzag(h, l, atr, mult):
    """Sommets / creux confirmés quand le prix repart de mult x ATR. Tout est causal : l'état à la bougie t
    ne dépend que des bougies <= t. Renvoie, pour chaque bougie, la direction de la jambe en cours (d = +1
    montante, -1 descendante), son extrême provisoire (ext) et les 2 derniers sommets / creux confirmés."""
    n = len(h)
    nan = np.nan
    out = {k: np.full(n, nan) for k in ("ext", "cur_hi", "prev_hi", "cur_lo", "prev_lo")}
    out.update({k: np.full(n, -1, dtype=np.int64) for k in ("ext_i", "cur_hi_i", "cur_lo_i")})
    out["d"] = np.zeros(n, dtype=np.int8)
    d = 0
    started = False
    hh = ll = ext = nan
    hh_i = ll_i = ext_i = -1
    cur_hi = (nan, -1)
    prev_hi = (nan, -1)
    cur_lo = (nan, -1)
    prev_lo = (nan, -1)
    for t in range(n):
        a = atr[t]
        if a != a:
            continue
        thr = mult * a
        if not started:
            hh, hh_i, ll, ll_i, started = h[t], t, l[t], t, True
        elif d == 0:
            if h[t] > hh:
                hh, hh_i = h[t], t
            if l[t] < ll:
                ll, ll_i = l[t], t
            if hh - ll >= thr:
                if ll_i < hh_i:
                    cur_lo, d, ext, ext_i = (ll, ll_i), 1, hh, hh_i
                elif hh_i < ll_i:
                    cur_hi, d, ext, ext_i = (hh, hh_i), -1, ll, ll_i
        elif d == 1:
            if h[t] > ext:
                ext, ext_i = h[t], t
            if ext - l[t] >= thr:
                prev_hi, cur_hi = cur_hi, (ext, ext_i)
                d, ext, ext_i = -1, l[t], t
        else:
            if l[t] < ext:
                ext, ext_i = l[t], t
            if h[t] - ext >= thr:
                prev_lo, cur_lo = cur_lo, (ext, ext_i)
                d, ext, ext_i = 1, h[t], t
        out["d"][t] = d
        out["ext"][t], out["ext_i"][t] = ext, ext_i
        out["cur_hi"][t], out["cur_hi_i"][t], out["prev_hi"][t] = cur_hi[0], cur_hi[1], prev_hi[0]
        out["cur_lo"][t], out["cur_lo_i"][t], out["prev_lo"][t] = cur_lo[0], cur_lo[1], prev_lo[0]
    return out


def trend_bear(T, th, tl, tc, p):
    """Tendance baissière à chaque bougie de l'unité de tendance (bool). Loi de Dow, moyenne mobile, EMA simple (comme FVG), ou les deux."""
    mode = p["trend_mode"]
    n = len(tc)
    bear_dow = np.zeros(n, bool)
    bear_ma = np.zeros(n, bool)
    if mode in ("dow", "both"):
        zz = zigzag(th, tl, T["atr"], p["zz_atr_trend"])
        have = ~(np.isnan(zz["prev_hi"]) | np.isnan(zz["prev_lo"]) | np.isnan(zz["cur_hi"]) | np.isnan(zz["cur_lo"]))
        with np.errstate(invalid="ignore"):
            # sommets ET creux de plus en plus bas ; tendance annulée si le prix clôture au-dessus du dernier sommet
            bear_dow = have & (zz["cur_hi"] < zz["prev_hi"]) & (zz["cur_lo"] < zz["prev_lo"]) & (tc <= zz["cur_hi"])
    if mode in ("ma", "both", "ema"):
        ema = pd.Series(tc).ewm(span=p["ma_len"], adjust=False).mean().to_numpy()
        warmup = min(10, max(2, int(p["ma_len"]) // 10))
        if mode == "ema":
            with np.errstate(invalid="ignore"):
                bear_ma = (np.arange(n) >= warmup) & (tc < ema)
            return bear_ma
        k = p["ma_slope_bars"]
        slope = ema - np.concatenate([np.full(k, np.nan), ema[:-k]])
        with np.errstate(invalid="ignore"):
            bear_ma = (np.arange(n) >= warmup) & (tc < ema) & (slope < 0)
    if mode == "dow":
        return bear_dow
    if mode == "ma":
        return bear_ma
    return bear_dow & bear_ma


def bear_range(F, fh, fl, p, tf):
    """Dernière jambe baissière : H = dernier sommet confirmé, L = plus bas depuis ce sommet.
    Valide tant que le prix n'a pas repris le sommet H."""
    zz = zigzag(fh, fl, F["atr"], p.get("zz_atr_" + tf, 2.0))
    d, H = zz["d"], zz["cur_hi"]
    n = len(d)
    L = np.full(n, np.nan)
    Li = np.full(n, -1, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        down = (d == -1) & ~np.isnan(H)
        up = ((d == 1) & ~np.isnan(H) & ~np.isnan(zz["cur_lo"]) & (zz["cur_lo_i"] > zz["cur_hi_i"]) & (zz["ext"] < H))
        L[down], Li[down] = zz["ext"][down], zz["ext_i"][down]
        L[up], Li[up] = zz["cur_lo"][up], zz["cur_lo_i"][up]
        valid = (down | up) & (L < H)
    Hi = zz["cur_hi_i"].copy()
    if p.get("range_origin", "leg") == "leg":
        # Le haut du retracement = le plus haut depuis la DERNIÈRE FOIS où le prix était aussi bas que L
        # (= le début du mouvement complet, pas seulement le dernier pivot).
        look = int(p.get("leg_lookback", 300))
        H = H.copy()
        for t in np.where(valid)[0]:
            hi = int(Hi[t])
            a = max(0, hi - look)
            idx = np.where(fl[a:hi] <= L[t])[0]
            k = a + int(idx[-1]) + 1 if len(idx) else a
            j = k + int(np.argmax(fh[k:hi + 1]))
            if fh[j] > H[t]:
                H[t], Hi[t] = fh[j], j
    return {"H": H, "Hi": Hi, "L": L, "Li": Li, "valid": valid}


# ------------------------------------------------------------------ MSS après sweep

@_njit
def _mss_core(h, l, c, o, has_o, k, ratio, trough_pivot, min_sweep_pct):
    n = len(h)
    ev = np.zeros(n, np.bool_)
    o_h2 = np.full(n, np.nan)
    o_h1 = np.full(n, np.nan)
    o_l2 = np.full(n, np.nan)
    o_h2i = np.full(n, -1, np.int64)
    o_h1i = np.full(n, -1, np.int64)
    o_li = np.full(n, -1, np.int64)
    piv_i = np.empty(n, np.int64)
    piv_p = np.empty(n)
    npv = 0
    low_i = np.empty(n, np.int64)
    low_p = np.empty(n)
    nlw = 0
    runM, runM_i = -np.inf, -1
    last_h1, last_h2 = -1, -1                    # dernier couple (H1, H2) consommé
    ck_h1, ck_h2, ck_nl = -2, -2, -2             # clé du creux en cache
    cL, cLi = np.nan, -1
    ms = min_sweep_pct / 100.0
    for t in range(n):
        p = t - k
        if p >= k:
            # creux confirmé ?
            lmin = l[p - k]
            for q in range(p - k + 1, p):
                if l[q] < lmin:
                    lmin = l[q]
            rmin = l[p + 1]
            for q in range(p + 2, t + 1):
                if l[q] < rmin:
                    rmin = l[q]
            if l[p] < lmin and l[p] <= rmin:
                low_i[nlw] = p
                low_p[nlw] = l[p]
                nlw += 1
            # sommet confirmé ?
            lmax = h[p - k]
            for q in range(p - k + 1, p):
                if h[q] > lmax:
                    lmax = h[q]
            rmax = h[p + 1]
            for q in range(p + 2, t + 1):
                if h[q] > rmax:
                    rmax = h[q]
            if h[p] > lmax and h[p] >= rmax:
                piv_i[npv] = p
                piv_p[npv] = h[p]
                npv += 1
                runM = h[p + 1]
                runM_i = p + 1
                for q in range(p + 2, t + 1):
                    if h[q] > runM:
                        runM, runM_i = h[q], q
            elif npv > 0 and h[t] > runM:
                runM, runM_i = h[t], t
        elif npv > 0 and h[t] > runM:
            runM, runM_i = h[t], t
        if npv == 0:
            continue
        ia, pa = piv_i[npv - 1], piv_p[npv - 1]
        if runM > pa and abs(runM - pa) >= abs(pa) * ms:
            h1i, h1p, h2i, h2p = ia, pa, runM_i, runM
        elif npv >= 2 and pa > piv_p[npv - 2] and abs(pa - piv_p[npv - 2]) >= abs(piv_p[npv - 2]) * ms:
            h1i, h1p, h2i, h2p = piv_i[npv - 2], piv_p[npv - 2], ia, pa
        else:
            continue
        nl_key = nlw if trough_pivot else 0
        if h1i != ck_h1 or h2i != ck_h2 or nl_key != ck_nl:
            ck_h1, ck_h2, ck_nl = h1i, h2i, nl_key
            if trough_pivot:
                lo = np.searchsorted(low_i[:nlw], h1i, side="right")
                hi = np.searchsorted(low_i[:nlw], h2i, side="left")
                if hi <= lo:
                    cL, cLi = np.nan, -1
                else:
                    j = lo
                    for q in range(lo + 1, hi):
                        if low_p[q] < low_p[j]:
                            j = q
                    cL, cLi = low_p[j], low_i[j]
            else:
                j = h1i + 1
                for q in range(h1i + 2, h2i + 1):
                    if l[q] < l[j]:
                        j = q
                cL, cLi = l[j], j
        if cL != cL:
            continue
        if t > h2i and c[t] < cL and not (h1i == last_h1 and h2i == last_h2):
            last_h1, last_h2 = h1i, h2i
            if ((not has_o) or c[t] < o[t]) and (t - h2i) <= ratio * (h2i - cLi):
                ev[t] = True
                o_h2[t] = h2p
                o_h1[t] = h1p
                o_l2[t] = cL
                o_h2i[t] = h2i
                o_h1i[t] = h1i
                o_li[t] = cLi
    return ev, o_h2, o_h1, o_l2, o_h2i, o_h1i, o_li


def mss_events(h, l, c, k, ratio, o=None, trough_pivot=0, min_sweep_pct=0.0):
    """MSS baissier (définition de Yacine), recalculé en continu sur les sommets VOISINS.
    k = nombre de bougies de CHAQUE côté qui doivent être moins extrêmes pour valider un sommet (ou un creux).
      1) H1 = le dernier sommet confirmé (plus haut que les k bougies avant et les k bougies après) ;
      2) H2 = un plus haut qui dépasse H1 (sweep). H2 peut être encore provisoire : c'est le plus haut depuis H1 ;
      3) L2 = le plus bas (mèche comprise) entre H1 et H2 : c'est le creux à l'origine de H2. Avec trough_pivot=1 il doit en
         plus être confirmé (plus bas que les k bougies avant et après) ; sans creux confirmé, pas de MSS ;
      4) une bougie BAISSIÈRE dont la CLÔTURE (le corps) passe sous L2 = cassure ;
      5) cassure plus rapide que la montée : (t - H2) <= ratio x (H2 - L2), en bougies.
    Chaque couple (H1, H2) ne peut se déclencher qu'une fois. Causal : un sommet ou un creux n'est connu que k bougies
    après sa formation. Renvoie (événement, H2, H1, L2)."""
    h = np.ascontiguousarray(h, dtype=np.float64)
    l = np.ascontiguousarray(l, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    has_o = o is not None
    oo = np.ascontiguousarray(o, dtype=np.float64) if has_o else c
    return _mss_core(h, l, c, oo, has_o, int(k), float(ratio), bool(trough_pivot), float(min_sweep_pct))


@_njit
def _range_sm(bh, bl, bL, end, H0, Hb0, zlo, lvl):
    """Suit le retracement sur les bougies de base entre le creux bL et `end` (exclu).
    Règle (monde baissier ; le haussier est traité en miroir) :
      - une bougie (mèche ou corps) touche la zone OTE après le creux, puis une bougie suivante redescend
        jusqu'au niveau `lvl` (0,5) -> la zone OTE a déjà réagi : ce retracement est mort ;
      - le plus haut atteint depuis le creux devient le haut candidat d'un futur retracement ;
      - ce futur retracement n'existe que quand un creux PLUS BAS que le précédent se forme (ce creux devient son bas) ;
      - si le prix reprend le haut courant, on repart de ce nouveau haut.
    Renvoie H, indice de H, L, indice de L, mort (0/1), nombre de recyclages."""
    H, Hb = H0, Hb0
    L, Lb = bl[bL], bL
    touched = False
    dead = False
    peak, peak_b = -np.inf, -1
    dpeak, dpeak_b = -np.inf, -1
    nrec = 0
    for t in range(bL + 1, end):
        if bh[t] >= H:
            H, Hb = bh[t], t
            L, Lb = np.inf, -1
            touched = False
            dead = False
            peak, dpeak = -np.inf, -np.inf
            continue
        if bl[t] < L:
            if dead:
                H, Hb = dpeak, dpeak_b
                nrec += 1
                dead = False
            L, Lb = bl[t], t
            touched = False
            peak, peak_b = -np.inf, -1
            continue
        if dead:
            if bh[t] > dpeak:
                dpeak, dpeak_b = bh[t], t
            continue
        if touched and bl[t] <= L + lvl * (H - L):
            dead = True
            dpeak, dpeak_b = peak, peak_b
            if bh[t] > dpeak:
                dpeak, dpeak_b = bh[t], t
            continue
        if bh[t] > peak:
            peak, peak_b = bh[t], t
        if bh[t] >= L + zlo * (H - L):
            touched = True
    return H, Hb, L, Lb, dead, nrec


# ------------------------------------------------------------------ construction des signaux

DIAG = {}   # entonnoir : combien de candidats sont éliminés par chaque filtre, par type de confirmation


_DF_CACHE = {}


def _d(kind, reason):
    DIAG[(kind, reason)] = DIAG.get((kind, reason), 0) + 1


def run_world(F, B, jm, p, sign, cache=None):
    """sign=+1 : on cherche des SHORTS (tendance baissière). sign=-1 : monde en miroir -> des LONGS.
    Renvoie une liste de (indice_base, sl, tp, code) avec des prix déjà remis à l'échelle réelle."""
    def view(f):
        if sign == 1:
            return f["o"], f["h"], f["l"], f["c"]
        return -f["o"], -f["l"], -f["h"], -f["c"]

    if cache is None:
        cache = {}
    quiet = bool(p.get("_quiet", False))
    confirms = _csv(p["confirms"])
    nb = len(B["c"])
    tT = p["trend_tf"]
    if tT not in F:
        return [], 0.0
    T = F[tT]
    _, th, tl, tc = view(T)
    tb_key = ("tb", tT, sign, p["trend_mode"], int(p["atr_len"]), float(p["zz_atr_trend"]), int(p["ma_len"]), int(p["ma_slope_bars"]))
    if tb_key not in cache:
        cache[tb_key] = trend_bear(T, th, tl, tc, p)
    bear = cache[tb_key]
    if not quiet:
        zz_key = ("zz", tT, sign, int(p["atr_len"]), float(p["zz_atr_trend"]))
        if zz_key not in cache:
            cache[zz_key] = zigzag(th, tl, T["atr"], p["zz_atr_trend"])
        zzT = cache[zz_key]
        emaT = pd.Series(tc).ewm(span=p["ma_len"], adjust=False).mean().to_numpy()
    else:
        zzT = emaT = None
    ranges = {}
    for R in _csv(p["range_tfs"]):
        if R in F:
            _, rh, rl, _ = view(F[R])
            rg_key = ("rg", R, sign, int(p["atr_len"]), float(p.get("zz_atr_" + R, 2.0)), p.get("range_origin", "leg"), int(p.get("leg_lookback", 300)))
            if rg_key not in cache:
                cache[rg_key] = bear_range(F[R], rh, rl, p, R)
            ranges[R] = cache[rg_key]
    _, bh, bl, _ = view(B)
    s0, s1 = int(p.get("session_start", 0)), int(p.get("session_end", 24))
    b_hours = (B["close_ns"] // (60 * NS_MIN)) % 24 if (s0, s1) != (0, 24) else None
    out = []
    for Y in _csv(p["conf_tfs"]):
        if Y not in F:
            continue
        V = F[Y]
        vo, vh, vl, vc = view(V)
        iy_key = ("iY", Y)
        if iy_key not in cache:
            iY = np.searchsorted(B["close_ns"], V["close_ns"])
            okY = np.zeros(len(vo), bool)
            inside = iY < nb
            okY[inside] = B["close_ns"][iY[inside]] == V["close_ns"][inside]
            cache[iy_key] = (iY, okY)
        iY, okY = cache[iy_key]
        cands = []
        if "bubble" in confirms and len(vo) > 1:
            bub = V["bub_up"] if sign == 1 else V["bub_dn"]
            idx = np.where(bub[:-1] & (vc[1:] < vo[1:]))[0] + 1  # bulle puis bougie baissière
            cands += [(int(j), "bubble", None, None) for j in idx if okY[j]]
        if "mss" in confirms:
            mss_key = ("mss", Y, sign, int(p["mss_fractal"]), float(p["mss_speed_ratio"]), int(p["mss_trough_pivot"]), float(p["mss_min_sweep_pct"]))
            if mss_key not in cache:
                cache[mss_key] = mss_events(vh, vl, vc, p["mss_fractal"], p["mss_speed_ratio"], vo,
                                            p["mss_trough_pivot"], p["mss_min_sweep_pct"])
            ev, h2, h1m, l2m, h2i_arr, h1i_arr, li_arr = cache[mss_key]
            cands += [(int(j), "mss", h2[j], (h1m[j], h2[j], l2m[j], int(h1i_arr[j]), int(h2i_arr[j]), int(li_arr[j]))) for j in np.where(ev)[0] if okY[j]]
            if not quiet:
                _d("mss_brut", "0 événements MSS " + Y)
        cands.sort(key=lambda x: x[0])
        for j, kind, sweep, mssp in cands:
            i = int(iY[j])
            jt = jm[tT][i]
            if not quiet:
                _d(kind, "0 détectés")
            if jt < 0 or (p["trend_filter"] and not bear[jt]):
                if not quiet:
                    _d(kind, "1 hors tendance")
                continue
            if b_hours is not None:
                hr = int(b_hours[i])
                if not ((s0 <= hr < s1) if s0 < s1 else (hr >= s0 or hr < s1)):
                    if not quiet:
                        _d(kind, "1 hors session")
                    continue
            for R, rg in ranges.items():
                jr = jm[R][i]
                if jr < 0 or not rg["valid"][jr]:
                    if not quiet:
                        _d(kind, "2 pas de range valide")
                    continue
                H, L = rg["H"][jr], rg["L"][jr]
                Rf, li = F[R], int(rg["Li"][jr])
                bs = int(np.searchsorted(B["open_ns"], Rf["open_ns"][li], side="left"))
                be = int(np.searchsorted(B["open_ns"], Rf["close_ns"][li], side="left"))
                if be <= bs:
                    if not quiet:
                        _d(kind, "4 données manquantes")
                    continue
                bL = bs + int(np.argmin(bl[bs:be]))
                hs = int(np.searchsorted(B["open_ns"], Rf["open_ns"][int(rg["Hi"][jr])], side="left"))
                he = int(np.searchsorted(B["open_ns"], Rf["close_ns"][int(rg["Hi"][jr])], side="left"))
                hi_b = hs + int(np.argmax(bh[hs:he])) if he > hs else -1
                nrec = 0
                if p["ote_react"] and hi_b >= 0:
                    # début du déclencheur : avant lui, l'OTE ne doit pas avoir déjà réagi
                    if kind == "bubble" or mssp is None:
                        cj = j - 1
                    else:
                        cj = mssp[3] if mssp[3] >= 0 else j - 1
                    cut = int(np.searchsorted(B["open_ns"], V["open_ns"][max(cj, 0)], side="left"))
                    H, hi_b, L, bL2, dead, nrec = _range_sm(bh, bl, bL, cut, float(H), hi_b, p["ote_lo"], p["ote_react_level"])
                    if dead:
                        if not quiet:
                            _d(kind, "3 OTE déjà réagie (retracement invalide)")
                        continue
                    if bL2 < 0:
                        if not quiet:
                            _d(kind, "6 sommet du range repris")
                        continue
                    bL = bL2
                zl, zh = L + p["ote_lo"] * (H - L), L + p["ote_hi"] * (H - L)
                if kind == "bubble":
                    if not (zl <= max(vh[j - 1], vh[j]) <= zh and min(vl[j - 1], vl[j]) <= zh):
                        if not quiet:
                            _d(kind, "3 hors zone OTE")
                        continue
                else:
                    h2_idx = mssp[4]
                    # 1) H2 (mèche comprise = vh[h2_idx]) <= zh (ligne des 0.706)
                    # 2) Intersection entre la bougie de H2 [vl[h2_idx], vh[h2_idx]] et la zone OTE [zl, zh] non vide
                    if h2_idx < 0 or not (zl <= sweep <= zh and vl[h2_idx] <= zh):
                        if not quiet:
                            _d(kind, "3 hors zone OTE")
                        continue
                after_l, after_h = bl[bL + 1:i + 1], bh[bL + 1:i + 1]
                if len(after_h) == 0 or after_l.min() < bl[bL]:  # nouveau plus bas : la jambe a changé
                    if not quiet:
                        _d(kind, "5 nouveau extrême depuis le creux du range")
                    continue
                peak = after_h.max()
                if peak >= H:
                    if not quiet:
                        _d(kind, "6 sommet du range repris")
                    continue
                # Le sommet H2 doit être l'extrême absolu depuis le point d'ancrage du retracement Fibonacci (bL)
                if kind == "mss" and abs(sweep - peak) > 1e-9:
                    if not quiet:
                        _d(kind, "5 H2 n'est pas l'extrême du retracement Fib")
                    continue
                # Filtre du ratio temporel : durée de l'impulsion Fib (hi_b <-> bL) / durée du retracement (bL -> MSS)
                t_bL_ns = int(B["open_ns"][bL])
                t_hi_ns = int(B["open_ns"][hi_b]) if hi_b >= 0 else int(Rf["open_ns"][int(rg["Hi"][jr])])
                t_imp_ns = abs(t_bL_ns - t_hi_ns)
                t_ret_ns = max(1, int(B["close_ns"][i]) - t_bL_ns)
                time_ratio = float(t_imp_ns) / float(t_ret_ns)
                max_tr = float(p.get("max_time_ratio", 0.0) or 0.0)
                if max_tr > 0.0 and time_ratio > max_tr:
                    if not quiet:
                        _d(kind, "5 ratio temps impulsion/retracement trop grand")
                    continue
                atr = V["atr"][j]
                if atr != atr:
                    continue
                sl = peak + p["sl_buf_atr"] * atr
                tp = peak - p["tp_fib"] * (peak - bl[bL])
                k_pk = bL + 1 + int(np.argmax(after_h))
                emode = str(p.get("entry_mode", "mss") or "mss")
                elvl = float(p.get("entry_lvl", 0.0) or 0.0)
                fmin_atr = float(p.get("fvg_min_atr", 0.0) or 0.0)
                order_type = "market"
                entry = vc[j]
                fvg_lo_w = fvg_hi_w = np.nan
                fvg_t1_ns = -1
                if kind == "mss" and emode != "mss":
                    tf_fvg = "15m" if emode in ("fvg_15m", "fvg_15m_or_mss") and "15m" in F else Y
                    Ff = F[tf_fvg]
                    fh_w = Ff["h"] if sign == 1 else -Ff["l"]
                    fl_w = Ff["l"] if sign == 1 else -Ff["h"]
                    fatr = Ff["atr"]
                    jf_end = int(jm[tf_fvg][i])
                    t_h2_open = int(B["open_ns"][k_pk])
                    jf_h2 = int(np.searchsorted(Ff["close_ns"], t_h2_open, side="right"))
                    jf_found = -1
                    # On cherche la FVG la plus proche du MSS (donc la plus récente entre H2 et le MSS)
                    for qf in range(jf_end - 2, max(-1, jf_h2 - 1), -1):
                        if fh_w[qf + 2] < fl_w[qf] and (fl_w[qf] - fh_w[qf + 2]) >= fmin_atr * fatr[min(jf_end, len(fatr) - 1)]:
                            jf_found = qf
                            break
                    if jf_found >= 0:
                        fvg_lo_w = float(fh_w[jf_found + 2])
                        fvg_hi_w = float(fl_w[jf_found])
                        fvg_t1_ns = int(Ff["open_ns"][jf_found])
                        en_fvg = fvg_lo_w + elvl * (fvg_hi_w - fvg_lo_w)
                        if en_fvg > vc[j]:
                            entry = en_fvg
                            order_type = "limit"
                        else:
                            entry = vc[j]
                    elif emode in ("fvg_15m", "fvg_conf"):
                        if not quiet:
                            _d(kind, "6 pas de FVG d'entrée entre H2 et MSS")
                        continue
                if not (tp < entry < sl):
                    if not quiet:
                        _d(kind, "7 entrée déjà sous le TP")
                    continue
                # Plafond strict du RR à 1:8 quoiqu'il arrive
                if (entry - tp) > 8.0 * (sl - entry):
                    tp = entry - 8.0 * (sl - entry)
                if p["min_sl_pct"] and (sl - entry) / abs(entry) * 100 < p["min_sl_pct"]:
                    if not quiet:
                        _d(kind, "8 stop plus serré que min_sl_pct")
                    continue
                if p["min_rr"] and (entry - tp) < p["min_rr"] * (sl - entry):
                    if not quiet:
                        _d(kind, "8 ratio gain/risque trop faible")
                    continue
                if not quiet:
                    _d(kind, "9 SIGNAL")
                if quiet:
                    info = {}
                else:
                    # ---- infos détaillées du trade (prix réels, pas en miroir)
                    def ts(ix):
                        return pd.Timestamp(int(B["open_ns"][ix]), tz="UTC").strftime("%Y-%m-%d %H:%M") if ix >= 0 else ""
                    def ts_v(ix):
                        return pd.Timestamp(int(V["open_ns"][ix]), tz="UTC").strftime("%Y-%m-%d %H:%M") if ix >= 0 else ""
                    pts = sorted([(sign * H, ts(hi_b)), (sign * L, ts(bL))], key=lambda x: x[0])
                    leg = sorted([(sign * peak, ts(k_pk)), (sign * bl[bL], ts(bL))], key=lambda x: x[0])
                    z = sorted([sign * zl, sign * zh])
                    info = {"signal_time": pd.Timestamp(int(B["close_ns"][i]), tz="UTC").strftime("%Y-%m-%d %H:%M"),
                            "confirm": f"{kind}{Y}", "range_tf": R, "trend_tf": tT,
                            "fib_range_high": pts[1][0], "fib_range_high_time": pts[1][1],
                            "fib_range_low": pts[0][0], "fib_range_low_time": pts[0][1],
                            "ote_zone_low": z[0], "ote_zone_high": z[1],
                            "fib_leg_high": leg[1][0], "fib_leg_high_time": leg[1][1],
                            "fib_leg_low": leg[0][0], "fib_leg_low_time": leg[0][1], "tp_fib": p["tp_fib"],
                            "range_recycled": nrec, "time_ratio": round(time_ratio, 3)}
                    if fvg_t1_ns >= 0:
                        fz = sorted([sign * fvg_lo_w, sign * fvg_hi_w])
                        info.update({"fvg_lo": fz[0], "fvg_hi": fz[1],
                                     "fvg_t1": pd.Timestamp(fvg_t1_ns, tz="UTC").strftime("%Y-%m-%d %H:%M")})
                    jt_ = int(jt)
                    if sign == 1:
                        ph_, ch_, pl_, cl_ = (zzT[k][jt_] for k in ("prev_hi", "cur_hi", "prev_lo", "cur_lo"))
                    else:   # monde en miroir : les sommets du monde sont les creux réels
                        ph_, ch_, pl_, cl_ = (-zzT[k][jt_] for k in ("prev_lo", "cur_lo", "prev_hi", "cur_hi"))
                    tdir = ("BAISSIERE" if sign == 1 else "HAUSSIERE") if bear[jt_] else "AUCUNE (filtre désactivé)"
                    info.update({"trend_dir": tdir, "trend_mode": p["trend_mode"],
                                 "trend_prev_high": ph_, "trend_last_high": ch_, "trend_prev_low": pl_, "trend_last_low": cl_,
                                 "trend_ema": sign * emaT[jt_], "trend_close": sign * tc[jt_]})
                    if kind == "bubble":   # chiffres de la bulle, pour les comparer au graphique
                        jb = j - 1
                        if sign == 1:
                            edge, band = min(V["o"][jb], V["c"][jb]), V["bb_up"][jb]
                        else:
                            edge, band = max(V["o"][jb], V["c"][jb]), V["bb_dn"][jb]
                        info.update({"bubble_time": pd.Timestamp(int(V["open_ns"][jb]), tz="UTC").strftime("%Y-%m-%d %H:%M"),
                                     "bubble_kind": "HAUTE" if sign == 1 else "BASSE", "bubble_body_edge": edge, "bubble_band": band,
                                     "bubble_dist_pct": abs(edge - band) / edge * 100})
                    if mssp is not None:
                        info.update({"mss_first_extreme": sign * mssp[0], "mss_second_extreme": sign * mssp[1],
                                     "mss_origin_level": sign * mssp[2],
                                     "h1_time": ts_v(mssp[3]), "h2_time": ts_v(mssp[4]), "l2_time": ts_v(mssp[5])})
                out.append((i, sign * entry, order_type, sign * sl, sign * tp, f"{kind}{Y}|{R}", info))
                break
    return out, float(bear.mean())


def prepare(df):
    p = PARAMS
    quiet = bool(p.get("_quiet", False))
    open_ns = pd.DatetimeIndex(df["time"]).as_unit("ns").asi8
    base_min = int(round(np.median(np.diff(open_ns)) / NS_MIN))
    nb = len(df)
    c_arr = df["close"].to_numpy(float)
    df_key = (nb, int(open_ns[0]), int(open_ns[-1]), base_min, float(c_arr[0]), float(c_arr[-1]), float(c_arr[nb // 2]))
    if df_key not in _DF_CACHE:
        if len(_DF_CACHE) >= 16:
            _DF_CACHE.clear()
        B = {"o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float), "l": df["low"].to_numpy(float),
             "c": df["close"].to_numpy(float), "open_ns": open_ns, "close_ns": open_ns + base_min * NS_MIN}
        _DF_CACHE[df_key] = {"B": B, "frames": {}, "jm": {}, "cache": {}}
    entry = _DF_CACHE[df_key]
    B = entry["B"]
    dropped = [tf for tf in _csv(p["conf_tfs"]) if TF_MIN[tf] < base_min]
    if dropped and not quiet:
        print(f"Unité de base {base_min} min : confirmations {', '.join(dropped)} ignorées (trop fines).")
    need = {p["trend_tf"], "15m", *_csv(p["range_tfs"]), *_csv(p["conf_tfs"])}
    F = {}
    for tf in need:
        if TF_MIN[tf] >= base_min:
            if tf not in entry["frames"]:
                entry["frames"][tf] = make_frame(df, TF_MIN[tf], base_min)
                entry["jm"][tf] = np.searchsorted(entry["frames"][tf]["close_ns"], B["close_ns"], side="right") - 1
            F[tf] = entry["frames"][tf]
            add_indicators(F[tf], p, tf)
    jm = entry["jm"]

    setups = []
    for sign, side, name in ((1, -1, "short"), (-1, 1, "long")):
        if p["side"] not in ("both", name):
            continue
        res, share = run_world(F, B, jm, p, sign, cache=entry["cache"])
        setups += [(i, side, en, otype, sl, tp, code, info) for i, en, otype, sl, tp, code, info in res]
        if not quiet:
            label = "baissière" if sign == 1 else "haussière"
            print(f"Tendance {p['trend_tf']} {label} ({p['trend_mode']}) : {share * 100:.1f} % du temps | signaux {name} : {len(res)}")

    if not quiet:
        for kind in ("bubble", "mss"):
            rows = sorted((r, n) for (k, r), n in DIAG.items() if k == kind)
            if rows:
                print(f"Entonnoir {kind} (les 2 sens) : " + " | ".join(f"{r[2:]} {n}" if r[0] != "0" else f"détectés {n}" for r, n in rows))
    sig_side = np.zeros(nb, dtype=np.int8)
    sig_entry = np.full(nb, np.nan)
    sig_otype = np.array([""] * nb, dtype=object)
    sig_sl = np.full(nb, np.nan)
    sig_tp = np.full(nb, np.nan)
    sig_code = np.array([""] * nb, dtype=object)
    sig_info = np.array([None] * nb, dtype=object)
    for i, side, en, otype, sl, tp, code, info in sorted(setups, key=lambda x: x[0]):
        if sig_side[i] == 0:
            sig_side[i], sig_entry[i], sig_otype[i], sig_sl[i], sig_tp[i], sig_code[i], sig_info[i] = side, en, otype, sl, tp, code, info
    df["sig_side"], df["sig_entry"], df["sig_otype"], df["sig_sl"], df["sig_tp"], df["sig_code"], df["sig_info"] = (
        sig_side, sig_entry, sig_otype, sig_sl, sig_tp, sig_code, sig_info
    )
    df["sig_ttl"] = int(p.get("ttl", 96) or 0)
    return df


def signal(d, i):
    s = d["sig_side"][i]
    if s == 0:
        return None
    en = d["sig_entry"][i] if "sig_entry" in d and not np.isnan(d["sig_entry"][i]) else d["close"][i]
    otype = d["sig_otype"][i] if "sig_otype" in d and d["sig_otype"][i] else "market"
    ttl = int(d["sig_ttl"][i]) if "sig_ttl" in d else 96
    return {"side": "long" if s > 0 else "short", "sl": d["sig_sl"][i], "tp": d["sig_tp"][i], "meta": d["sig_code"][i], "info": d["sig_info"][i],
            "entry": float(en), "order_type": otype, "ttl": ttl, "sig_i": i}
