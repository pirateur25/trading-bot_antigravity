# backtest/zones.py — FVG des unités de temps élevées (H1, H4, Daily) et leur mitigation.
# Raisonnement baissier (monde "sign=+1"). FVG baissier = 3 bougies a,b,c avec high(c) < low(a) : zone [high(c), low(a)].
# FVG haussier (utilisé comme cible de TP) = low(c) > high(a) : zone [high(a), low(c)].
# Une zone existe à la CLÔTURE de la bougie c. Elle est mitigée dès qu'une bougie CLÔTURÉE de l'unité mit_tf (H1 ou H4),
# ouverte après sa formation, l'a touchée (mèche ou corps) — profondeur mit_depth : 0 = simple touche, 1 = comblée en entier.
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native


def raw_zones(f, kind):
    h, l = f["h"], f["l"]
    if kind == "bear":
        j = np.flatnonzero(h[2:] < l[:-2])
        lo, hi = h[j + 2], l[j]
    else:
        j = np.flatnonzero(l[2:] > h[:-2])
        lo, hi = h[j], l[j + 2]
    return lo, hi, f["close_ns"][j + 2]


def zone_set(pd_, sign, tfs, mit_tf, depth, kind):
    """Zones triées par heure de formation, avec l'heure de leur mitigation (INT64_MAX = jamais mitigée)."""
    key = ("zones", sign, tuple(tfs), mit_tf, depth, kind)
    if key in pd_.cache:
        return pd_.cache[key]
    W = pd_.worlds[sign]
    m = W[mit_tf]
    parts = []
    for tf in tfs:
        lo, hi, form = raw_zones(W[tf], kind)
        mit = native.mitigate(lo, hi, form, 0 if kind == "bear" else 1, depth, m["h"], m["l"], m["open_ns"], m["close_ns"])
        parts.append((lo, hi, form, mit))
    lo, hi, form, mit = (np.concatenate([p[i] for p in parts]) for i in range(4))
    o = np.argsort(form, kind="stable")
    res = {"lo": np.ascontiguousarray(lo[o]), "hi": np.ascontiguousarray(hi[o]),
           "form": np.ascontiguousarray(form[o]), "mit": np.ascontiguousarray(mit[o])}
    pd_.cache[key] = res
    return res
