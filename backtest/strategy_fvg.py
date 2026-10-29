# backtest/strategy.py — stratégie « MSS dans un FVG HTF non mitigé » (short ; le long est le miroir exact).
#
# 1) Zones : FVG baissiers non mitigés en H1 / H4 / Daily (voir zones.py).
# 2) MSS baissier en m5 et/ou m15 dont le sommet H2 (le plus haut, celui qui balaie H1) est DANS une de ces zones :
#    H1 = sommet confirmé sur k bougies de chaque côté ; H2 > H1 ; L2 = plus bas (mèche comprise) entre H1 et H2 ;
#    une bougie baissière CLÔTURE sous L2 ; cassure plus rapide que (ratio x) la distance entre les deux sommets.
# 3) Ordre : SELL LIMIT sur le premier FVG baissier m5/m15 de la jambe H2 -> cassure ; SL au-dessus de H2 (+ marge ATR).
# 4) TP : RR fixe, ou prochain FVG haussier HTF non mitigé sous l'entrée.
# Les réglages sont dans PARAMS ; ils se modifient en ligne de commande / workflow avec --params "k=v;k2=v2".
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native
import zones

PARAMS = {
    # --- confirmation MSS ---
    "conf_tfs": "5m",        # "5m", "15m" ou "5m,15m"
    "mss_k": 2,              # un sommet/creux est confirmé si plus extrême que k bougies de chaque côté
    "speed_ratio": 1.5,      # cassure en <= ratio x distance (bougies) ; 1e9 = pas de limite
    "speed_ref": 0,          # 0 = distance entre les 2 sommets (H1->H2) ; 1 = distance L2->H2 (variante de ton premier bot)
    "trough_pivot": 0,       # 0 = L2 est le vrai plus bas (mèche comprise) entre H1 et H2 ; 1 = il doit en plus être un creux confirmé
    "min_sweep_pct": 0.0,    # H2 doit dépasser H1 d'au moins ce % du prix
    "min_corr_pct": 0.0,     # correction minimale H1 -> L2 en % du prix (sinon on prend un H1 plus ancien) ; 0 = pas de seuil
    "min_corr_atr": 0.0,     # idem en multiples d'ATR de l'unité de confirmation (0 = pas de seuil)
    "max_back": 5,           # nb maximum de sommets plus anciens essayés pour trouver un H1 qui respecte la correction minimale
    # --- zones HTF ---
    "htf": "4h",             # FVG utilisés : sous-ensemble de 1h,4h,1d
    "mit_tf": "1h",          # une zone est mitigée par une bougie CLÔTURÉE de cette unité : 1h ou 4h
    "mit_depth": 0.0,        # 0 = touchée (mèche) ; 1 = comblée en entier
    "zone_pad_atr": 0.0,     # tolérance au-dessus de la zone (en ATR de l'unité de confirmation)
    # --- entrée / sortie ---
    "entry_lvl": 0.0,        # niveau de l'ordre limite dans le FVG de la jambe : 0 = bord proche .. 1 = bord lointain
    "fvg_min_atr": 0.0,      # taille minimale du FVG d'entrée (en ATR)
    "sl_buf_atr": 0.0,       # marge du SL au-dessus de H2 (en ATR de l'unité de confirmation)
    "tp_mode": "rr",         # "rr" (TP fixe) ou "fvg" (prochain FVG HTF non mitigé)
    "rr": 2.0,
    "min_rr": 1.0,           # mode fvg : RR minimum de la cible
    "max_rr": 1e9,           # mode fvg : plafond de RR
    "tp_edge": 0.0,          # mode fvg : 0 = bord proche du FVG cible, 1 = bord lointain
    "be_rr": 0.0,            # break-even à X R (0 = désactivé)
    "ttl": 96,               # un ordre limite non rempli est annulé après N bougies de 5m (0 = jamais)
    "min_sl_pct": 0.0,       # distance minimale du SL en % du prix
    # --- filtres ---
    "side": "both",          # "short", "long" ou "both"
    "trend": "none",         # "none" ou "4h:200", "1d:50", ... (clôture vs EMA : short seulement sous l'EMA, long seulement au-dessus)
    "session_start": 0,      # heures UTC d'entrée autorisées [start, end)
    "session_end": 24,
}


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_params(text, base=None):
    """'k=v;k2=v2' -> dict (chaque valeur est convertie dans le type de la valeur par défaut)."""
    p = dict(PARAMS if base is None else base)
    for part in [x.strip() for x in (text or "").replace("\n", ";").split(";") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"réglage invalide '{part}' (format attendu : nom=valeur)")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in PARAMS:
            raise ValueError(f"réglage inconnu '{k}'. Possibles : {', '.join(PARAMS)}")
        cur = PARAMS[k]
        p[k] = float(v) if isinstance(cur, float) else int(float(v)) if isinstance(cur, int) else v.replace(" ", "")
    return p


def events(pd_, sign, tf, p):
    key = ("ev", sign, tf, p["mss_k"], p["speed_ratio"], p["speed_ref"], p["trough_pivot"], p["min_sweep_pct"],
           p["min_corr_pct"], p["min_corr_atr"], p["max_back"])
    if key not in pd_.cache:
        f = pd_.worlds[sign][tf]
        pd_.cache[key] = native.mss_events(f["h"], f["l"], f["c"], f["o"], p["mss_k"], p["speed_ratio"], p["speed_ref"],
                                           p["trough_pivot"], p["min_sweep_pct"], f["atr"], p["min_corr_pct"], p["min_corr_atr"], p["max_back"])
    return pd_.cache[key]


def signals(pd_, p, a=0, b=None):
    """Tous les ordres limites (côté, bougie de signal en 5m, entrée, SL, TP) pour les bougies de base dans [a, b)."""
    b = pd_.n if b is None else b
    out = []
    for sign in (1, -1):
        if p["side"] not in ("both", "short" if sign == 1 else "long"):
            continue
        W = pd_.worlds[sign]
        htf = _csv(p["htf"])
        pz = zones.zone_set(pd_, sign, htf, p["mit_tf"], p["mit_depth"], "bear")
        oz = zones.zone_set(pd_, sign, htf, p["mit_tf"], p["mit_depth"], "bull") if p["tp_mode"] == "fvg" else \
            {"lo": np.zeros(0), "hi": np.zeros(0), "form": np.zeros(0, np.int64), "mit": np.zeros(0, np.int64)}
        P = np.array([p["zone_pad_atr"], p["entry_lvl"], p["sl_buf_atr"], 1.0 if p["tp_mode"] == "fvg" else 0.0, p["rr"],
                      p["min_rr"], p["tp_edge"], p["max_rr"], p["fvg_min_atr"]], dtype=np.float64)
        for tf in _csv(p["conf_tfs"]):
            conf = W[tf]
            ev = events(pd_, sign, tf, p)
            if len(ev["t"]) == 0:
                continue
            ok, entry, sl, tp, zid = native.build_signals(ev, conf, pz, oz, P)
            bi = ev["t"] if tf == "5m" else np.searchsorted(pd_.base["close_ns"], conf["close_ns"][ev["t"]])
            m = ok & (bi >= a) & (bi < b) & (bi < pd_.n - 1)
            if p["trend"] != "none":
                ttf, ln = p["trend"].split(":")
                m &= pd_.trend_state(ttf, int(ln))[np.clip(bi, 0, pd_.n - 1)] == -sign
            if (p["session_start"], p["session_end"]) != (0, 24):
                s0, s1 = int(p["session_start"]), int(p["session_end"])
                hr = pd_.hour[np.clip(bi, 0, pd_.n - 1)]
                m &= ((hr >= s0) & (hr < s1)) if s0 < s1 else ((hr >= s0) | (hr < s1))
            if p["min_sl_pct"] > 0:
                m &= (sl - entry) >= p["min_sl_pct"] / 100.0 * np.abs(entry)
            for i in np.flatnonzero(m):
                q0, t0 = int(ev["h2i"][i]), int(ev["t"][i])   # premier FVG de la jambe (même règle que le noyau C), pour l'affichage
                jf = next((q for q in range(q0, t0 - 1) if conf["h"][q + 2] < conf["l"][q]
                           and conf["l"][q] - conf["h"][q + 2] >= p["fvg_min_atr"] * conf["atr"][t0]), q0)
                out.append((int(bi[i]), sign, float(entry[i]), float(sl[i]), float(tp[i]), tf, int(ev["h2i"][i]), int(ev["h1i"][i]),
                            float(ev["h2p"][i]), float(ev["l2p"][i]), int(zid[i]), int(ev["t"][i]), float(pz["lo"][zid[i]]), float(pz["hi"][zid[i]]), float(ev["h1p"][i]), int(ev["li"][i]),
                            int(pz["form"][zid[i]]), jf))
    out.sort(key=lambda x: (x[0], x[1]))
    return out
