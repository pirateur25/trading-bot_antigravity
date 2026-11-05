"""
backtest/strategy_opr.py — Stratégie OPR (Opening Price Range) sur les indices US (NAS100 / US100, USA500 / US500, US30).

Règles exactes :
  1. Zone OPR :
     - Définie par la bougie M15 d'ouverture US : 15h30 -> 15h45 heure française (ou 09h30 -> 09h45 heure NY selon `tz_mode`),
       c.-à-d. les 3 bougies M5 [15h30, 15h35, 15h40].
     - OPR High = plus haut (mèche incluse) de cette bougie M15.
     - OPR Low  = plus bas (mèche incluse) de cette bougie M15.
  2. Filtre obligatoire de tendance en H1 (`mm20_50`) :
     - Tendance haussière (+1) : MM20 > MM50 en H1 ET clôture de la bougie H1 > MM50.
     - Tendance baissière (-1) : MM20 < MM50 en H1 ET clôture de la bougie H1 < MM50.
  3. Unité d'exécution : M5, de 15h45 (clôture de l'OPR) jusqu'à 19h30 heure française :
     - Maximum 1 signal / ordre posé par jour et par indice (le premier des 2 setups qui se présente).
     - Aucun nouveau trade ou ordre n'est pris après 19h30 heure française si le prix n'est pas déjà rentré en position.
  4. Setup 1 — Cassure (Breakout) dans le sens de la tendance H1 :
     - Haussier (tendance H1 haussière) : si une bougie M5 clôture strictement au-dessus de l'OPR High (close > opr_hi),
       ordre BUY LIMIT sur l'OPR High (entry = opr_hi), SL sous la bougie M5 qui a cassé l'OPR High (sl = low[i]),
       TP fixe à RR = 2.5 (tp = entry + 2.5 * (entry - sl)).
     - Baissier (tendance H1 baissière, miroir exact) : si une bougie M5 clôture strictement en dessous de l'OPR Low (close < opr_lo),
       ordre SELL LIMIT sur l'OPR Low (entry = opr_lo), SL au-dessus de la bougie M5 qui a cassé l'OPR Low (sl = high[i]),
       TP fixe à RR = 2.5 (tp = entry - 2.5 * (sl - entry)).
  5. Setup 2 — Réintégration (Re-integration) après sortie par le côté opposé :
     - Haussier (tendance H1 haussière) : si le prix sort par le bas (sous l'OPR Low — pas de short car tendance haussière)
       puis qu'une bougie M5 réintègre l'OPR (clôture à l'intérieur : opr_lo <= close <= opr_hi),
       ordre BUY LIMIT sur l'OPR Low (entry = opr_lo), SL sous le bas de la bougie M5 qui réintègre (sl = low[i]),
       TP sur l'OPR High (tp = opr_hi). Valide UNIQUEMENT si ce TP donne un RR >= 2.0 ((opr_hi - opr_lo) >= 2.0 * (opr_lo - sl)).
     - Baissier (tendance H1 baissière, miroir exact) : si le prix sort par le haut (au-dessus de l'OPR High)
       puis qu'une bougie M5 réintègre l'OPR (opr_lo <= close <= opr_hi),
       ordre SELL LIMIT sur l'OPR High (entry = opr_hi), SL au-dessus du haut de la bougie M5 qui réintègre (sl = high[i]),
       TP sur l'OPR Low (tp = opr_lo). Valide UNIQUEMENT si ce TP donne un RR >= 2.0 ((opr_hi - opr_lo) >= 2.0 * (sl - opr_hi)).
  6. Gestion de l'ordre limite et du trade :
     - L'ordre limite est annulé s'il n'a pas encore été déclenché alors que le prix atteint +1 R depuis le niveau d'entrée
       (high >= entry + 1.0 * risk en achat ; low <= entry - 1.0 * risk en vente), ou si 19h30 est atteint sans exécution.
     - Une fois exécuté, Break-Even (SL déplacé à `entry`) dès que le prix atteint +1 R (`be_rr = 1.0`).
"""
import numpy as np
import pandas as pd

SETUP_ORDER = ["Cassure", "Reintegration"]

PARAMS = {
    "rr_break": 2.5,         # TP fixe du setup Cassure (en R)
    "min_rr_reint": 2.0,     # RR minimum exigé pour le setup Réintégration (TP = borne opposée de l'OPR)
    "cancel_rr": 1.0,        # annulation de l'ordre limite non déclenché si le prix touche +cancel_rr R
    "be_rr": 1.0,            # passage à Break-Even à +be_rr R (0 = désactivé)
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


def _trend_h1_mm20_50(pd_, tf="1h", mode="mm20_50"):
    """
    Filtre de tendance H1 (mm20_50) :
      +1 (haussier) si MM20 > MM50 et close > MM50 en H1
      -1 (baissier) si MM20 < MM50 et close < MM50 en H1
       0 sinon.
    Projeté sur chaque bougie 5m : utilise la bougie H1 en cours ou la dernière clôturée.
    Pour éviter tout biais du futur (lookahead), sur chaque bougie 5m `i`, on évalue :
      - les 19 et 49 dernières bougies H1 entièrement clôturées avant `close_ns[i]`
      - + la clôture courante à l'instant `close_ns[i]` (`B['c'][i]`), qui est exactement la clôture
        de la bougie H1 en cours à l'instant où la bougie 5m `i` se termine !
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

    # j[i] = indice de la dernière bougie H1 clôturée au moment où la bougie 5m `i` clôture
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

    # Mode "mm20_50" :
    # En temps réel à la fin de la bougie 5m `i` :
    #   - si `close_ns[i]` tombe pile sur la clôture d'une bougie H1, la bougie H1 en cours vient de clôturer (`hc[j[i]] == b_close[i]`) ;
    #     MM20 et MM50 sont les moyennes des 20 et 50 dernières bougies H1 (incluant `j[i]`).
    #   - si on est au milieu d'une heure (ex: 15h45, 15h50...), la bougie H1 en cours a pour clôture actuelle `b_close[i]`,
    #     et les 19 (resp. 49) bougies H1 précédentes sont `j[i]-18 .. j[i]` (resp. `j[i]-48 .. j[i]`).
    # Calcul vectorisé exact par sommes cumulées sur H1 :
    csum = np.concatenate([[0.0], np.cumsum(hc, dtype=np.float64)])
    exact_close = np.zeros(nb, dtype=bool)
    valid_j = j >= 0
    exact_close[valid_j] = f["close_ns"][j[valid_j]] == b_close_ns[valid_j]

    mapped = np.zeros(nb, dtype=np.int8)
    # Il faut au moins 49 bougies H1 clôturées (j >= 48 si en cours d'heure, j >= 49 si pile à l'heure)
    ok = (j >= 49) & (~ np.isnan(b_close))
    idx = np.flatnonzero(ok)
    if len(idx) > 0:
        ji = j[idx]
        ex = exact_close[idx]
        cur_c = b_close[idx]

        # Somme des 19 bougies H1 clôturées précédentes + bougie H1 en cours (ou 20 clôturées si pile à l'heure)
        # Quand ex est True, la bougie H1 `ji` vient de clôturer avec close == cur_c, donc les 19 précédentes sont ji-19..ji-1
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


def run(pd_, capital=10000.0, risk_pct=0.5, fee_pct=0.005, leverage=30.0, p=None, i0=0, i1=None):
    """
    Exécute le backtest OPR sur `pd_` dans l'intervalle de bougies 5m `[i0, i1)`.
    Retourne `(trades_df, equity_array_on_slice)`.
    """
    p = dict(PARAMS if p is None else p)
    rr_break = float(p.get("rr_break", 2.5))
    min_rr_reint = float(p.get("min_rr_reint", 2.0))
    cancel_rr = float(p.get("cancel_rr", 1.0))
    be_rr = float(p.get("be_rr", 1.0))
    tz_mode = str(p.get("tz_mode", "paris")).lower()
    side_filter = str(p.get("side", "both")).lower()
    allowed_setups = {s.strip() for s in str(p.get("setups", "Cassure,Reintegration")).split(",") if s.strip()}

    B = pd_.base
    o, h, l, c = B["o"], B["h"], B["l"], B["c"]
    open_ns, close_ns = B["open_ns"], B["close_ns"]
    n = pd_.n
    i1 = n if i1 is None else min(n, i1)
    fee = fee_pct / 100.0

    day_vec, mins_vec = _tz_vectors(pd_, tz_mode)
    trend_vec = _trend_h1_mm20_50(pd_, str(p.get("trend_tf", "1h")), str(p.get("trend_mode", "mm20_50")))

    # Heures d'ouverture OPR et de fin de session (en minutes depuis minuit)
    if tz_mode == "ny":
        opr_start_min = 9 * 60 + 30    # 09h30 NY
        opr_end_min = 9 * 60 + 45      # 09h45 NY
        end_session_min = 13 * 60 + 30 # 13h30 NY (= 19h30 Paris hors décalage DST d'une semaine)
    else:
        opr_start_min = 15 * 60 + 30   # 15h30 Paris
        opr_end_min = 15 * 60 + 45     # 15h45 Paris
        end_session_min = int(p.get("end_hour", 19)) * 60 + int(p.get("end_min", 30))  # 19h30 Paris

    start_search = max(0, i0)
    end_search = min(n, i1)

    # Regroupement par jour sur la plage [start_search, end_search)
    signals = []
    idx = start_search
    while idx < end_search:
        cur_day = int(day_vec[idx])
        day_start = idx
        while idx < end_search and int(day_vec[idx]) == cur_day:
            idx += 1
        day_end = idx  # [day_start, day_end) = toutes les bougies 5m de ce jour

        # 1) Identifier les bougies M5 qui forment la bougie M15 de 15h30 (open_min in {15h30, 15h35, 15h40})
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

        # Indice de fin de fenêtre de prise de position (19h30 heure française : dernière bougie qui clôture à 19h30 a open_min = 19h25 < 19h30)
        # Après 19h30 (c.-à-d. open_min >= 19h30), on ne prend plus de nouveau trade ni d'exécution d'ordre limite en attente.
        exec_indices = [
            k for k in range(opr_i1 + 1, day_end)
            if opr_end_min <= int(mins_vec[k]) < end_session_min
        ]
        if len(exec_indices) < 2:
            continue

        f_idx = exec_indices[-1] + 1  # première bougie à partir de 19h30 (exclue pour l'entrée)

        # Suivi d'une sortie préalable hors de l'OPR pour le Setup 2 (Réintégration)
        exited_below = False
        exited_above = False

        for k in exec_indices[:-1]:  # il faut au moins 1 bougie avant 19h30 après le signal pour remplir l'ordre limite
            t_dir = int(trend_vec[k])
            ck = float(c[k])
            hk = float(h[k])
            lk = float(l[k])

            # --- Tendance H1 HAUSSIÈRE (+1) ---
            if t_dir == 1 and side_filter in ("both", "long"):
                # Setup 1 : Cassure haussière (clôture M5 au-dessus de l'OPR High)
                if "Cassure" in allowed_setups and ck > opr_hi:
                    entry = opr_hi
                    sl = lk
                    if sl < entry:
                        risk = entry - sl
                        tp = entry + rr_break * risk
                        signals.append({
                            "sig_i": k,
                            "opr_i0": opr_i0,
                            "opr_i1": opr_i1,
                            "f_idx": f_idx,
                            "sign": -1,
                            "side": "long",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "opr_lo": opr_lo,
                            "opr_hi": opr_hi,
                            "setup": "Cassure",
                        })
                        break

                # Setup 2 : Réintégration haussière (sortie par le bas puis clôture M5 à l'intérieur de l'OPR)
                if "Reintegration" in allowed_setups and exited_below and (opr_lo <= ck <= opr_hi):
                    entry = opr_lo
                    sl = lk
                    tp = opr_hi
                    if sl < entry and (tp - entry) >= min_rr_reint * (entry - sl):
                        signals.append({
                            "sig_i": k,
                            "opr_i0": opr_i0,
                            "opr_i1": opr_i1,
                            "f_idx": f_idx,
                            "sign": -1,
                            "side": "long",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "opr_lo": opr_lo,
                            "opr_hi": opr_hi,
                            "setup": "Reintegration",
                        })
                        break

            # --- Tendance H1 BAISSIÈRE (-1) (miroir exact) ---
            elif t_dir == -1 and side_filter in ("both", "short"):
                # Setup 1 : Cassure baissière (clôture M5 en dessous de l'OPR Low)
                if "Cassure" in allowed_setups and ck < opr_lo:
                    entry = opr_lo
                    sl = hk
                    if sl > entry:
                        risk = sl - entry
                        tp = entry - rr_break * risk
                        signals.append({
                            "sig_i": k,
                            "opr_i0": opr_i0,
                            "opr_i1": opr_i1,
                            "f_idx": f_idx,
                            "sign": 1,
                            "side": "short",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "opr_lo": opr_lo,
                            "opr_hi": opr_hi,
                            "setup": "Cassure",
                        })
                        break

                # Setup 2 : Réintégration baissière (sortie par le haut puis clôture M5 à l'intérieur de l'OPR)
                if "Reintegration" in allowed_setups and exited_above and (opr_lo <= ck <= opr_hi):
                    entry = opr_hi
                    sl = hk
                    tp = opr_lo
                    if sl > entry and (entry - tp) >= min_rr_reint * (sl - entry):
                        signals.append({
                            "sig_i": k,
                            "opr_i0": opr_i0,
                            "opr_i1": opr_i1,
                            "f_idx": f_idx,
                            "sign": 1,
                            "side": "short",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "opr_lo": opr_lo,
                            "opr_hi": opr_hi,
                            "setup": "Reintegration",
                        })
                        break

            # Mise à jour de l'état de sortie hors de l'OPR (pour les bougies suivantes)
            if lk < opr_lo:
                exited_below = True
            if hk > opr_hi:
                exited_above = True

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

