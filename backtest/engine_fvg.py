# backtest/engine.py — lance la simulation sur les ordres limites de strategy.signals() (côtés long/short séparés puis fusionnés).
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import native
import v2_common


def run(pd_, sigs, capital=10000.0, risk_pct=1.0, fee_pct=0.005, leverage=50.0, be_rr=0.0, ttl=96, a=0, b=None, noentry=None, fexit=None, p=None):
    """Retourne (trades DataFrame, equity ndarray sur [a,b), stats). Un seul trade à la fois, les deux sens sont simulés ensemble."""
    b = pd_.n if b is None else b
    fee = fee_pct / 100.0
    if p is not None and (
        str(p.get("trail_mode", "be_only")) not in ("none", "be_only")
        or str(p.get("trail_check", "bar")) != "bar"
        or float(p.get("sl_cooldown_hours", 0.0) or 0.0) > 0.0
        or int(p.get("block_same_poi", 0) or 0) == 1
        or int(p.get("cancel_on_flip", 0) or 0) == 1
    ):
        return _run_v2(pd_, sigs, capital, risk_pct, fee, leverage, be_rr, ttl, a, b, noentry, fexit, p)
    # Le moteur C travaille en monde 'short'. Pour un mélange long/short on enchaîne les signaux dans l'ordre du temps :
    # on simule d'abord chaque monde séparément pour obtenir les occupations, puis on retient le premier signal libre.
    # Simplification exacte : les trades s'excluent mutuellement => on simule en séquence sur une liste fusionnée.
    return _run_merged(pd_, sigs, capital, risk_pct, fee, leverage, be_rr, ttl, a, b, noentry, fexit)


def _run_merged(pd_, sigs, capital, risk_pct, fee, lev, be_rr, ttl, a, b, noentry=None, fexit=None):
    B = pd_.base
    n = pd_.n
    worlds = {1: (B["o"], B["h"], B["l"], B["c"]), -1: (-B["o"], -B["l"], -B["h"], -B["c"])}
    if not sigs:
        eq = np.full(n, capital)
        return pd.DataFrame(), eq[a:b], stats(pd.DataFrame(), eq[a:b], capital)
    # on découpe en séquences de même sens : la simulation C gère l'occupation à l'intérieur d'un bloc ; entre deux blocs on
    # reprend le solde et l'heure de sortie. Pour rester simple et exact, on boucle signal par signal si les sens sont mixtes.
    sides = {s[1] for s in sigs}
    if len(sides) == 1:
        sign = sides.pop()
        o, h, l, c = worlds[sign]
        idx = np.array([s[0] for s in sigs], np.int32)
        cols, eq = native.simulate(o, h, l, c, idx, [s[2] for s in sigs], [s[3] for s in sigs], [s[4] for s in sigs],
                                   capital, risk_pct, fee, lev, be_rr, ttl, noentry, fexit)
        tr = _trades(pd_, sigs, cols, sign)
    else:
        tr, eq = _mixed(pd_, sigs, worlds, capital, risk_pct, fee, lev, be_rr, ttl, noentry, fexit)
    return tr, eq[a:b], stats(tr, eq[a:b], capital)


def _mixed(pd_, sigs, worlds, capital, risk_pct, fee, lev, be_rr, ttl, noentry=None, fexit=None):
    """Mélange long/short : on avance signal par signal ; chaque trade est simulé par le moteur C avec le solde courant."""
    n = pd_.n
    eq = np.full(n, np.nan)
    bal, busy, rows, last = capital, -1, [], 0
    for s in sigs:
        if s[0] < busy:
            continue
        sign = s[1]
        o, h, l, c = worlds[sign]
        lo = s[0]
        hi = min(n, lo + 1 + max(ttl, 1) + 20000)
        cols, e = native.simulate(o[lo:hi], h[lo:hi], l[lo:hi], c[lo:hi], np.array([0], np.int32), [s[2]], [s[3]], [s[4]],
                                  bal, risk_pct, fee, lev, be_rr, ttl, None if noentry is None else noentry[lo:hi], None if fexit is None else fexit[lo:hi])
        if len(cols["sig"]) == 0:
            # ordre annulé : occupe jusqu'à l'annulation (ttl) mais ne change pas le solde
            busy = lo + (ttl if ttl > 0 else 1)
            continue
        x = lo + int(cols["exit"][0])
        eq[last:lo] = bal
        seg = e[: x - lo + 1]
        eq[lo: x + 1] = seg
        bal = float(seg[-1])
        last = x + 1
        busy = x
        d = {k: v[0] for k, v in cols.items()}
        d["fill"] = lo + int(d["fill"]); d["exit"] = x; d["sig"] = s
        rows.append((s, d, sign))
    eq[last:] = bal
    tr = _rows_to_df(pd_, rows)
    return tr, eq


NAMES = ["SL", "TP", "BE", "END", "WE", "TRAIL"]


def _run_v2(pd_, sigs, capital, risk_pct, fee, lev, be_rr, ttl, a, b, noentry=None, fexit=None, p=None):
    """Simulation FVG 2.0 avec Trailing multi-paliers R, Cooldown post-SL, Anti-Repeat Zone/POI et annulation 30m sur flip."""
    B = pd_.base
    n = pd_.n
    if not sigs:
        eq = np.full(n, float(capital))
        return pd.DataFrame(), eq[a:b], stats(pd.DataFrame(), eq[a:b], capital)

    worlds = {1: (B["o"], B["h"], B["l"], B["c"]), -1: (-B["o"], -B["l"], -B["h"], -B["c"])}
    close_ns = B["close_ns"]
    p = p or {}
    trail_mode = str(p.get("trail_mode", "be_only"))
    trail_check = str(p.get("trail_check", "bar"))
    tiers = v2_common.get_trail_tiers_r(trail_mode, be_rr)
    sl_cooldown_ns = int(float(p.get("sl_cooldown_hours", 0.0) or 0.0) * 3600 * 10**9)
    block_same_poi = int(p.get("block_same_poi", 0) or 0) == 1
    same_poi_tol = float(p.get("same_poi_tol_pct", 0.15) or 0.15) / 100.0
    cancel_on_flip = int(p.get("cancel_on_flip", 0) or 0) == 1
    v2_st = v2_common.get_pair_v2_state(pd_, p) if (cancel_on_flip or trail_check != "bar") else None

    eq = np.full(n, float(capital))
    bal = float(capital)
    busy = -1
    last_eq_idx = 0
    cooldown_until_ns = -1
    last_sl_zone = None
    last_sl_poi = np.nan
    last_sl_sign = 0
    rows = []

    for s in sigs:
        s0 = int(s[0])
        if s0 < busy or s0 >= n - 1:
            continue
        if sl_cooldown_ns > 0 and int(close_ns[s0]) < cooldown_until_ns:
            continue
        sign = int(s[1])
        en, sl, tp = float(s[2]), float(s[3]), float(s[4])
        zid = int(s[10])
        real_en = abs(en)
        if block_same_poi and last_sl_sign == sign:
            if last_sl_zone == (sign, zid):
                continue
            if not np.isnan(last_sl_poi) and last_sl_poi > 0 and abs(real_en - last_sl_poi) / last_sl_poi <= same_poi_tol:
                continue

        o, h, l, c = worlds[sign]
        end = min(n - 1, s0 + ttl) if ttl > 0 else n - 1
        fill = -1
        for t in range(s0 + 1, end + 1):
            if noentry is not None and noentry[t]:
                break
            if h[t] >= en:
                fill = t
                break
            if l[t] <= tp or h[t] >= sl:
                break
            if cancel_on_flip and v2_st is not None and v2_common.should_cancel_resting_on_flip(t, -sign, v2_st, p):
                break
        if fill < 0:
            continue

        risk_dist = sl - en
        loss_unit = risk_dist + abs(en) * fee + abs(sl) * fee
        if risk_dist <= 0 or loss_unit <= 0:
            continue
        risk_cash = bal * (risk_pct / 100.0)
        qty = risk_cash / loss_unit
        if qty * abs(en) > bal * lev:
            continue

        efee = qty * abs(en) * fee
        bal -= efee
        if fill > last_eq_idx:
            eq[last_eq_idx:fill] = bal + efee
            last_eq_idx = fill

        cur_sl = sl
        cur_tier = 0
        xbar = n - 1
        xprice = c[n - 1]
        reason = 3  # END
        for t in range(fill, n):
            if t > fill and fexit is not None and fexit[t]:
                xbar, xprice, reason = t, o[t], 4
                break
            if h[t] >= cur_sl:
                xbar = t
                xprice = max(cur_sl, o[t]) if t > fill else cur_sl
                reason = 5 if cur_tier >= 2 else (2 if cur_tier == 1 else 0)
                break
            if t > fill and l[t] <= tp:
                xbar = t
                xprice = min(tp, o[t])
                reason = 1
                break
            if t > fill and tiers:
                is_30m = bool(v2_st["is_30m_close"][t]) if v2_st is not None else True
                new_sl, new_tier = v2_common.compute_new_trail_sl(
                    "short", en, cur_sl, risk_dist, cur_tier, h[t], l[t], c[t], is_30m, tiers, trail_check
                )
                if new_tier > cur_tier:
                    cur_sl = new_sl
                    cur_tier = new_tier
            eq[t] = bal + (en - c[t]) * qty

        gross = (en - xprice) * qty
        xfee = qty * abs(xprice) * fee
        bal += gross - xfee
        net = gross - efee - xfee
        r_mult = net / risk_cash if risk_cash > 0 else 0.0
        eq[xbar] = bal
        last_eq_idx = xbar + 1
        busy = xbar

        if r_mult < -0.1:
            if sl_cooldown_ns > 0:
                cooldown_until_ns = int(close_ns[xbar]) + sl_cooldown_ns
            if block_same_poi:
                last_sl_zone = (sign, zid)
                last_sl_poi = real_en
                last_sl_sign = sign
        elif r_mult > 0.1:
            last_sl_zone = None
            last_sl_poi = np.nan
            last_sl_sign = 0

        d = {
            "sig": s,
            "fill": fill,
            "exit": xbar,
            "reason": reason,
            "entry": en,
            "xprice": xprice,
            "sl": sl,
            "tp": tp,
            "pnl": net,
            "r": r_mult,
        }
        rows.append((s, d, sign))

    if last_eq_idx < n:
        eq[last_eq_idx:] = bal
    tr = _rows_to_df(pd_, rows)
    return tr, eq[a:b], stats(tr, eq[a:b], capital)


def _trades(pd_, sigs, cols, sign):
    rows = []
    for k in range(len(cols["sig"])):
        d = {c: cols[c][k] for c in cols}
        d["sig"] = sigs[int(d["sig"])]
        rows.append((d["sig"], d, sign))
    return _rows_to_df(pd_, rows)


def _rows_to_df(pd_, rows):
    if not rows:
        return pd.DataFrame()
    t = pd_.times
    out = []
    T = lambda W, i: pd.Timestamp(int(W["open_ns"][i]), tz="UTC")
    for s, d, sign in rows:
        W = pd_.worlds[sign][s[5]]
        out.append({"entry_time": t.iloc[int(d["fill"])], "exit_time": t.iloc[int(d["exit"])], "signal_time": t.iloc[s[0]],
                    "side": "short" if sign == 1 else "long", "entry": sign * d["entry"], "exit": sign * d["xprice"],
                    "sl": sign * d["sl"], "tp": sign * d["tp"], "reason": NAMES[int(d["reason"])], "pnl": d["pnl"], "r": d["r"],
                    "bars": int(d["exit"]) - int(d["fill"]), "conf_tf": s[5], "mss_h2": sign * s[8], "mss_l2": sign * s[9],
                    "h1_time": T(W, s[7]), "h2_time": T(W, s[6]), "l2_time": T(W, s[15]), "zone_form": pd.Timestamp(s[16], tz="UTC"),
                    "fvg_lo": W["h"][s[17] + 2] if sign == 1 else -W["l"][s[17]], "fvg_hi": W["l"][s[17]] if sign == 1 else -W["h"][s[17] + 2],
                    "fvg_t1": T(W, s[17]), "fvg_t2": pd.Timestamp(int(W["close_ns"][s[17] + 2]), tz="UTC"),
                    "mss_h1": sign * s[14], "zone_lo": s[12] if sign == 1 else -s[13], "zone_hi": s[13] if sign == 1 else -s[12]})
    return pd.DataFrame(out)


def stats(tr, eq, capital):
    final = float(eq[-1]) if len(eq) else capital
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    dd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    s = {"return_pct": (final / capital - 1) * 100, "max_dd_pct": dd, "trades": len(tr)}
    if len(tr):
        w, lo = tr[tr["pnl"] > 0]["pnl"].sum(), -tr[tr["pnl"] <= 0]["pnl"].sum()
        s.update(win_pct=float((tr["pnl"] > 0).mean() * 100), avg_r=float(tr["r"].mean()), pf=float(w / lo) if lo > 0 else float("inf"),
                 longs=int((tr["side"] == "long").sum()), shorts=int((tr["side"] == "short").sum()))
    return s
