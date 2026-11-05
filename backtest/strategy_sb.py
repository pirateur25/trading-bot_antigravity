"""
backtest/strategy_sb.py — Stratégie ICT Silver Bullet (5m, fixe sans suroptimisation).

Règles exactes :
  1. Trois fenêtres horaires en heure de New York (America/New_York, gestion auto EST/EDT) :
       - London : 03h00 (D) à 04h00 (F)
       - NY AM  : 10h00 (D) à 11h00 (F)
       - NY PM  : 14h00 (D) à 15h00 (F)
  2. Dans chaque fenêtre [D, F), on repère la PREMIÈRE FVG 5m dont la bougie centrale (celle qui porte le gap)
     ouvre entre D inclus et F exclu (si le gap est sur D -> valide ; s'il est sur F -> non valide).
  3. Dès la clôture de la 3e bougie qui forme cette première FVG :
       - FVG haussière (low[i+1] > high[i-1]) :
           Ordre BUY LIMIT sur le haut du gap (entry = low[i+1]),
           SL au plus bas de la bougie qui a créé le gap (sl = low[i]),
           TP fixe à RR = 2.0 (tp = entry + 2 * (entry - sl)).
       - FVG baissière (high[i+1] < low[i-1]) :
           Ordre SELL LIMIT sur le bas du gap (entry = high[i+1]),
           SL au plus haut de la bougie qui a créé le gap (sl = high[i]),
           TP fixe à RR = 2.0 (tp = entry - 2 * (sl - entry)).
  4. L'ordre limite est retiré à F (fin de la fenêtre d'une heure).
     Si l'ordre est exécuté strictement avant F, on laisse courir le trade après F jusqu'au TP ou au SL.
"""
import numpy as np
import pandas as pd

SESSIONS = {
    3: "London",
    10: "NY AM",
    14: "NY PM",
}
SESSION_ORDER = ["London", "NY AM", "NY PM"]

PARAMS = {
    "rr": 2.0,
    "sessions": "London,NY AM,NY PM",
    "side": "both",
    "be_rr": 0.0,
    "trend_filter": 0,          # 0 = pas de filtre | 1 = uniquement dans le sens de la tendance
    "trend_tf": "1h",           # "1h" ou "4h"
    "trend_mode": "mm20_50",    # "mm20_50" | "ema_mm20_50" | "ema" | "ma" | "dow" | "both"
    "ma_len": 200,              # longueur EMA pour trend_mode in ("ema", "ma", "both")
    "ma_slope_bars": 5,         # pente EMA sur N bougies HTF pour trend_mode in ("ma", "both")
    "zz_atr_trend": 4.0,        # multiplicateur ATR du ZigZag pour trend_mode in ("dow", "both")
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


def _ny_vectors(pd_):
    key = ("ny_vec",)
    if key not in pd_.cache:
        ny_dt = pd.to_datetime(pd_.base["open_ns"], utc=True).tz_convert("America/New_York")
        ny_hour = ny_dt.hour.to_numpy(np.int16)
        ny_day = (ny_dt.year * 10000 + ny_dt.month * 100 + ny_dt.day).to_numpy(np.int32)
        pd_.cache[key] = (ny_hour, ny_day)
    return pd_.cache[key]


def _zigzag_causal(h, l, atr, mult):
    """ZigZag causal identique à strategy_ote.zigzag : renvoie les 2 derniers sommets et creux confirmés."""
    n = len(h)
    nan = np.nan
    out = {k: np.full(n, nan) for k in ("cur_hi", "prev_hi", "cur_lo", "prev_lo")}
    d = 0
    started = False
    hh = ll = ext = nan
    hh_i = ll_i = -1
    cur_hi = prev_hi = cur_lo = prev_lo = nan
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
                    cur_lo, d, ext = ll, 1, hh
                elif hh_i < ll_i:
                    cur_hi, d, ext = hh, -1, ll
        elif d == 1:
            if h[t] > ext:
                ext = h[t]
            if ext - l[t] >= thr:
                prev_hi, cur_hi = cur_hi, ext
                d, ext = -1, l[t]
        else:
            if l[t] < ext:
                ext = l[t]
            if h[t] - ext >= thr:
                prev_lo, cur_lo = cur_lo, ext
                d, ext = 1, h[t]
        out["cur_hi"][t], out["prev_hi"][t] = cur_hi, prev_hi
        out["cur_lo"][t], out["prev_lo"][t] = cur_lo, prev_lo
    return out


def _trend_dir_on_tf(pd_, tf, mode, ma_len, ma_slope_bars, zz_atr_trend):
    """
    Calcule la direction de tendance sur l'unité `tf` ("1h" ou "4h") et la projette causalement
    sur chaque bougie 5m de base :
      +1 = tendance haussière (longs autorisés)
      -1 = tendance baissière (shorts autorisés)
       0 = neutre / hors tendance (aucun trade)

    Modes disponibles :
      - "mm20_50"     : SMA 20 > SMA 50 et close > SMA 50 (haussier) ; SMA 20 < SMA 50 et close < SMA 50 (baissier)
      - "ema_mm20_50" : EMA 20 > EMA 50 et close > EMA 50 (haussier) ; EMA 20 < EMA 50 et close < EMA 50 (baissier)
      - "ema"         : close > EMA(ma_len) (haussier) ; close < EMA(ma_len) (baissier) (identique OTE/FVG)
      - "ma"          : close > EMA(ma_len) et pente EMA > 0 sur ma_slope_bars (haussier) ; inverse (baissier) (OTE)
      - "dow"         : structure ZigZag (HH + HL et close >= cur_lo pour haussier ; LH + LL et close <= cur_hi pour baissier) (OTE)
      - "both"        : accord simultané "dow" ET "ma" (OTE)
    """
    key = ("sb_trend", tf, mode, int(ma_len), int(ma_slope_bars), round(float(zz_atr_trend), 4))
    if key in pd_.cache:
        return pd_.cache[key]

    f = pd_.frames[tf]
    c = f["c"]
    h = f["h"]
    l = f["l"]
    n = len(c)
    s_c = pd.Series(c)

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    if mode == "mm20_50":
        mm20 = s_c.rolling(20).mean().to_numpy()
        mm50 = s_c.rolling(50).mean().to_numpy()
        with np.errstate(invalid="ignore"):
            bull = (mm20 > mm50) & (c > mm50)
            bear = (mm20 < mm50) & (c < mm50)
    elif mode == "ema_mm20_50":
        mm20 = s_c.ewm(span=20, adjust=False).mean().to_numpy()
        mm50 = s_c.ewm(span=50, adjust=False).mean().to_numpy()
        warmup = 10
        idx_ok = np.arange(n) >= warmup
        with np.errstate(invalid="ignore"):
            bull = idx_ok & (mm20 > mm50) & (c > mm50)
            bear = idx_ok & (mm20 < mm50) & (c < mm50)
    else:
        bull_dow = np.zeros(n, dtype=bool)
        bear_dow = np.zeros(n, dtype=bool)
        bull_ma = np.zeros(n, dtype=bool)
        bear_ma = np.zeros(n, dtype=bool)

        if mode in ("dow", "both"):
            zz_key = ("sb_zz", tf, round(float(zz_atr_trend), 4))
            if zz_key not in pd_.cache:
                zz_b = _zigzag_causal(h, l, f["atr"], float(zz_atr_trend))
                zz_l = _zigzag_causal(-l, -h, f["atr"], float(zz_atr_trend))
                pd_.cache[zz_key] = (zz_b, zz_l)
            zz_b, zz_l = pd_.cache[zz_key]
            have_b = ~(np.isnan(zz_b["prev_hi"]) | np.isnan(zz_b["prev_lo"]) | np.isnan(zz_b["cur_hi"]) | np.isnan(zz_b["cur_lo"]))
            have_l = ~(np.isnan(zz_l["prev_hi"]) | np.isnan(zz_l["prev_lo"]) | np.isnan(zz_l["cur_hi"]) | np.isnan(zz_l["cur_lo"]))
            with np.errstate(invalid="ignore"):
                bear_dow = have_b & (zz_b["cur_hi"] < zz_b["prev_hi"]) & (zz_b["cur_lo"] < zz_b["prev_lo"]) & (c <= zz_b["cur_hi"])
                bull_dow = have_l & (zz_l["cur_hi"] < zz_l["prev_hi"]) & (zz_l["cur_lo"] < zz_l["prev_lo"]) & ((-c) <= zz_l["cur_hi"])

        if mode in ("ema", "ma", "both"):
            ema = s_c.ewm(span=int(ma_len), adjust=False).mean().to_numpy()
            warmup = min(10, max(2, int(ma_len) // 10))
            idx_ok = np.arange(n) >= warmup
            if mode == "ema":
                with np.errstate(invalid="ignore"):
                    bull_ma = idx_ok & (c > ema)
                    bear_ma = idx_ok & (c < ema)
            else:
                k = max(1, int(ma_slope_bars))
                slope = ema - np.concatenate([np.full(k, np.nan), ema[:-k]])
                with np.errstate(invalid="ignore"):
                    bull_ma = idx_ok & (c > ema) & (slope > 0)
                    bear_ma = idx_ok & (c < ema) & (slope < 0)

        if mode == "dow":
            bull, bear = bull_dow, bear_dow
        elif mode in ("ema", "ma"):
            bull, bear = bull_ma, bear_ma
        else:
            bull, bear = (bull_dow & bull_ma), (bear_dow & bear_ma)

    st = np.zeros(n, dtype=np.int8)
    st[bull] = 1
    st[bear] = -1

    j_key = ("sb_j_map", tf)
    if j_key not in pd_.cache:
        pd_.cache[j_key] = np.searchsorted(f["close_ns"], pd_.base["close_ns"], side="right") - 1
    j = pd_.cache[j_key]

    mapped = np.where(j >= 0, st[np.clip(j, 0, max(0, n - 1))], 0).astype(np.int8)
    pd_.cache[key] = mapped
    return mapped


def run(pd_, capital=10000.0, risk_pct=0.5, fee_pct=0.005, leverage=30.0, p=None, i0=0, i1=None):
    """
    Exécute le backtest Silver Bullet sur `pd_` dans l'intervalle de bougies 5m `[i0, i1)`.
    Retourne `(trades_df, equity_array_on_slice)`.
    """
    p = dict(PARAMS if p is None else p)
    rr = float(p.get("rr", 2.0))
    be_rr = float(p.get("be_rr", 0.0))
    side_filter = str(p.get("side", "both")).lower()
    allowed_sessions = {s.strip() for s in str(p.get("sessions", "London,NY AM,NY PM")).split(",") if s.strip()}

    B = pd_.base
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    open_ns, close_ns = B["open_ns"], B["close_ns"]
    n = pd_.n
    i1 = n if i1 is None else min(n, i1)
    fee = fee_pct / 100.0

    ny_hour, ny_day = _ny_vectors(pd_)

    trend_filter = int(p.get("trend_filter", 0) or 0)
    trend_vec = None
    if trend_filter:
        t_tf = str(p.get("trend_tf", "1h") or "1h")
        t_mode = str(p.get("trend_mode", "mm20_50") or "mm20_50")
        t_ma_len = int(p.get("ma_len", 200) or 200)
        t_slope = int(p.get("ma_slope_bars", 5) or 5)
        t_zz = float(p.get("zz_atr_trend", 4.0) or 4.0)
        trend_vec = _trend_dir_on_tf(pd_, t_tf, t_mode, t_ma_len, t_slope, t_zz)

    # 1) Détection de la PREMIÈRE FVG 5m par fenêtre (jour NY, heure session D in {3, 10, 14})
    # La bougie centrale du gap est `i` (avec D <= open_time[i] < F, donc ny_hour[i] in SESSIONS).
    # La FVG est confirmée à la clôture de `i + 1`.
    signals = []
    seen_window = set()
    max_gap_ns = 15 * 60 * 10**9  # 3 bougies 5m consécutives sans coupure week-end

    start_search = max(1, i0)
    end_search = min(n - 2, i1 - 1)

    for i in range(start_search, end_search):
        sh = int(ny_hour[i])
        if sh not in SESSIONS:
            continue
        sess_name = SESSIONS[sh]
        if sess_name not in allowed_sessions:
            continue
        w_key = (int(ny_day[i]), sh)
        if w_key in seen_window:
            continue
        # Vérifie que i-1, i, i+1 sont bien 3 bougies 5m consécutives
        if (open_ns[i + 1] - open_ns[i - 1]) > max_gap_ns:
            continue

        bull = l[i + 1] > h[i - 1]
        bear = h[i + 1] < l[i - 1]
        if not bull and not bear:
            continue

        # Première FVG de la fenêtre [D, F) trouvée -> on verrouille la fenêtre
        seen_window.add(w_key)

        if bull:
            if side_filter not in ("both", "long"):
                continue
            if trend_vec is not None and trend_vec[i + 1] != 1:
                continue
            fvg_lo = float(h[i - 1])
            fvg_hi = float(l[i + 1])
            entry = fvg_hi          # haut du gap
            sl = float(l[i])        # plus bas de la bougie qui a créé le gap
            if sl >= entry:
                continue
            risk = entry - sl
            tp = entry + rr * risk
            sign = -1               # -1 = long (convention interne)
        else:
            if side_filter not in ("both", "short"):
                continue
            if trend_vec is not None and trend_vec[i + 1] != -1:
                continue
            fvg_lo = float(h[i + 1])
            fvg_hi = float(l[i - 1])
            entry = fvg_lo          # bas du gap (miroir short)
            sl = float(h[i])        # plus haut de la bougie qui a créé le gap
            if sl <= entry:
                continue
            risk = sl - entry
            tp = entry - rr * risk
            sign = 1                # +1 = short

        d_idx = i
        while d_idx > 0 and ny_day[d_idx - 1] == ny_day[i] and ny_hour[d_idx - 1] == sh:
            d_idx -= 1
        f_idx = i + 1
        while f_idx < n and ny_day[f_idx] == ny_day[i] and ny_hour[f_idx] == sh:
            f_idx += 1
        signals.append({
            "gap_i": i,
            "sig_i": i + 1,
            "d_idx": d_idx,
            "f_idx": f_idx,
            "sign": sign,
            "side": "long" if sign == -1 else "short",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "fvg_lo": fvg_lo,
            "fvg_hi": fvg_hi,
            "session": sess_name,
        })

    # 2) Simulation de chaque fenêtre (ordre limite retiré à F ; si exécuté avant F, on laisse courir le trade)
    raw_trades = []
    for s in signals:
        sig_i = s["sig_i"]
        f_idx = s["f_idx"]
        # L'ordre limite est placé à la clôture de sig_i (i+1) et retiré à F (f_idx)
        if sig_i + 1 >= f_idx:
            continue

        sign = s["sign"]
        en = s["entry"]
        sl = s["sl"]
        tp = s["tp"]

        fill = -1
        for k in range(sig_i + 1, min(f_idx, i1)):
            if sign == -1:  # Long : BUY LIMIT à en
                if l[k] <= en:
                    fill = k
                    break
                if h[k] >= tp:
                    break
            else:           # Short : SELL LIMIT à en
                if h[k] >= en:
                    fill = k
                    break
                if l[k] <= tp:
                    break

        if fill < 0:
            continue

        risk_dist = abs(en - sl)
        loss_unit = risk_dist + abs(en) * fee + abs(sl) * fee
        if loss_unit <= 0:
            continue
        # Vérification levier (pour 1$ de capital : qty_unit = (risk_pct/100) / loss_unit)
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

        d_i = s["d_idx"]
        f_i = min(s["f_idx"] - 1, n - 1)
        win_hi = float(np.max(h[d_i:f_i + 1]))
        win_lo = float(np.min(l[d_i:f_i + 1]))

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
            "session": s["session"],
            "conf_tf": f"5m ({s['session']})",
            "fvg_lo": s["fvg_lo"],
            "fvg_hi": s["fvg_hi"],
            "fvg_t1": pd.Timestamp(int(open_ns[s["gap_i"] - 1]), tz="UTC"),
            "fvg_t2": pd.Timestamp(int(close_ns[s["sig_i"]]), tz="UTC"),
            "zone_lo": win_lo,
            "zone_hi": win_hi,
            "zone_form": pd.Timestamp(int(open_ns[d_i]), tz="UTC"),
            "zone_end": pd.Timestamp(int(close_ns[f_i]), tz="UTC"),
        })

    # 3) Construction de la courbe d'équité et du PnL en $ (avec gestion propre des éventuelles superpositions entre sessions)
    eq = np.full(n, float(capital), dtype=np.float64)
    bal = float(capital)
    eqpos = 0
    open_q = []  # (exit_i, pnl_cash)

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


def session_stats(tr: pd.DataFrame, risk_pct: float, capital: float = 10000.0):
    """
    Calcule pour chacune des 3 sessions (London, NY AM, NY PM) :
      - return_pct : gain composé en % si on ne tradait que cette session
      - max_dd_pct : drawdown max en % sur la courbe de cette session
      - trades     : nombre de trades
      - win_pct    : taux de réussite (%)
      - avg_r      : espérance en R
      - sum_r      : cumul de R
    """
    out = {}
    for sess in SESSION_ORDER:
        if tr is None or len(tr) == 0 or "session" not in tr.columns:
            sub = pd.DataFrame()
        else:
            sub = tr[tr["session"] == sess].sort_values("entry_time")
        if len(sub) == 0:
            out[sess] = {
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
        out[sess] = {
            "return_pct": ret,
            "max_dd_pct": dd,
            "trades": int(len(r_arr)),
            "win_pct": float((r_arr > 0).mean() * 100.0),
            "avg_r": float(r_arr.mean()),
            "sum_r": float(r_arr.sum()),
        }
    return out
