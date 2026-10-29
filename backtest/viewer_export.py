"""
backtest/viewer_export.py — visualiseur de trades (graphique en chandeliers interactif, lightweight-charts).

DEUX ÉTAPES :
 1) UNE SEULE FOIS sur ton PC (les bougies sont "installées" dans le fichier) :
        python backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDCHF,USDJPY,XAUUSD --out viewer/viewer.html
    -> viewer/viewer.html contient tout l'historique 5m des paires choisies (a ouvrir dans le navigateur, hors-ligne sauf la librairie du graphique).
 2) A chaque backtest, run_backtest.py envoie sur Telegram un petit fichier  viewer_trades.json  (seulement les trades).
    Dans le visualiseur : bouton « Charger trades » (ou glisser-déposer le fichier). Rien d'autre à modifier.
Pour ajouter des paires plus tard (ex. cryptos) : relancer l'étape 1 avec --symbols plus longue.
"""
import argparse
import base64
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ep(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return None
    return int(pd.Timestamp(s).timestamp())


def nz(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def pick_div(price):
    return 100 if price >= 1000 else 1000 if price >= 50 else 100000 if price >= 1 else 10**7


def encode_pair(P, name, since=None):
    """Bougies 5m en différences entières, gzip + base64 (même format que l'ancien visualiseur, un seul segment)."""
    div = pick_div(float(np.nanmedian(P.base["c"])))
    ts = (P.base["open_ns"] // 10**9).astype(np.int64)
    i0 = int(np.searchsorted(ts, pd.Timestamp(since).timestamp())) if since else 0
    O, H, L, C = (np.rint(P.base[k][i0:] * div).astype(np.int64) for k in "ohlc")
    t = ts[i0:] // 300
    v = [int(t[0]), int(O[0])]
    pt, pc = t[0], O[0]
    for i in range(len(t)):
        v += [int(t[i] - pt), int(O[i] - pc), int(H[i] - O[i]), int(O[i] - L[i]), int(C[i] - O[i])]
        pt, pc = t[i], C[i]
    b64 = base64.b64encode(gzip.compress(json.dumps([v], separators=(",", ":")).encode(), 9)).decode()
    return {"name": name, "div": div, "b64": b64}


def trades_to_json(tr, strat, sym, be_rr=0.0):
    """DataFrame de trades (FVG ou OTE) -> liste de dicts pour le visualiseur."""
    out = []
    for r in tr.to_dict("records"):
        x = dict(sym=sym, strat=strat, side=r["side"], te=ep(r["entry_time"]), tx=ep(r["exit_time"]), entry=float(r["entry"]), exit=float(r["exit"]),
                 sl=float(r["sl"]), tp=float(r["tp"]), reason=r["reason"], r=round(float(r["r"]), 3), pnl=round(float(r["pnl"]), 2), be=float(be_rr or 0))
        if strat == "FVG":
            x.update(ts=ep(r["signal_time"]), tf=r["conf_tf"], h1p=nz(r["mss_h1"]), h1t=ep(r["h1_time"]), h2p=nz(r["mss_h2"]), h2t=ep(r["h2_time"]),
                     l2p=nz(r["mss_l2"]), l2t=ep(r["l2_time"]), zlo=nz(r["zone_lo"]), zhi=nz(r["zone_hi"]), zt=ep(r["zone_form"]),
                     fl=nz(r["fvg_lo"]), fh=nz(r["fvg_hi"]), ft1=ep(r["fvg_t1"]))
        else:
            g = lambda k: r.get(k)  # noqa: E731
            x.update(ts=ep(g("signal_time")) or ep(r["entry_time"]), tf=str(g("confirm") or ""), rh=nz(g("fib_range_high")), rht=ep(g("fib_range_high_time")),
                     rl=nz(g("fib_range_low")), rlt=ep(g("fib_range_low_time")), ozl=nz(g("ote_zone_low")), ozh=nz(g("ote_zone_high")),
                     lh=nz(g("fib_leg_high")), lht=ep(g("fib_leg_high_time")), ll=nz(g("fib_leg_low")), llt=ep(g("fib_leg_low_time")),
                     bt=ep(g("bubble_time")), m1=nz(g("mss_first_extreme")), m2=nz(g("mss_second_extreme")), mo=nz(g("mss_origin_level")))
        out.append(x)
    return out


def write_trades_json(trades, path):
    json.dump({"version": 2, "trades": trades}, open(path, "w"), separators=(",", ":"))
    return path


def build(symbols, datadir, out, since=None):
    import data_forex as data
    import opt_common as oc
    pairs = []
    for s in symbols:
        oc.ensure_data(s, datadir)   # crypto absente : téléchargement automatique
        P = data.PairData(s, datadir)
        e = encode_pair(P, s, since)
        pairs.append(e)
        print(f"{s}: {len(e['b64']) / 1e6:.1f} Mo", flush=True)
    html = open(os.path.join(HERE, "..", "viewer", "template.html"), encoding="utf-8").read().replace("/*PAIRS*/[]", json.dumps(pairs, separators=(",", ":")))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write(html)
    print(f"{out} : {len(html) / 1e6:.1f} Mo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDCHF,USDJPY,XAUUSD")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--out", default="viewer/viewer.html")
    ap.add_argument("--since", default="", help="ne garder que les bougies depuis cette date (réduit la taille), ex. 2023-01-01")
    a = ap.parse_args()
    build([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.out, a.since or None)

