"""
backtest/viewer_export.py — visualiseur de trades (graphique en chandeliers interactif, lightweight-charts).

DEUX ÉTAPES :
 1) UNE SEULE FOIS sur ton PC (les bougies sont "installées" dans le fichier) :
        python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100 --out viewer/viewer.html
    -> viewer/viewer.html contient tout l'historique 5m des paires choisies (à ouvrir dans le navigateur, hors-ligne sauf la librairie du graphique).
    (Fonctionne même sans numpy/pandas installés sur le PC local !)
 2) A chaque backtest, run_backtest.py envoie sur Telegram un petit fichier  viewer_trades.json  (seulement les trades).
    Dans le visualiseur : bouton « Charger trades » (ou glisser-déposer le fichier). Rien d'autre à modifier.
Pour ajouter des paires plus tard (ex. cryptos) : relancer l'étape 1 avec --symbols plus longue.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import gzip
import json
import os
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _is_nan(v):
    return isinstance(v, float) and v != v


def ep(s):
    if s is None or _is_nan(s) or s == "":
        return None
    if pd is not None:
        return int(pd.Timestamp(s).timestamp())
    dt = datetime.fromisoformat(str(s).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def nz(v):
    return None if v is None or _is_nan(v) else float(v)


def pick_div(price):
    return 100 if price >= 1000 else 1000 if price >= 50 else 100000 if price >= 1 else 10**7


def encode_pair(P, name, since=None):
    """Bougies 5m en différences entières, gzip + base64 (depuis un objet PairData numpy/pandas)."""
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


def _resolve_file(symbol, datadir="data"):
    sym = symbol.upper()
    for e in (".csv.gz", ".csv"):
        p = os.path.join(datadir, sym + e)
        if os.path.exists(p):
            return p
    if os.path.isdir(datadir):
        prefixes = [sym]
        if sym in ("USA500", "US500"):
            prefixes = ["USA500", "US500", "SPX500", "SP500"]
        elif sym in ("NAS100", "US100"):
            prefixes = ["NAS100", "USATECH", "USTEC", "US100"]
        for fname in sorted(os.listdir(datadir)):
            up = fname.upper()
            if any(up.startswith(pref) for pref in prefixes) and (up.endswith(".CSV.GZ") or up.endswith(".CSV")):
                src = os.path.join(datadir, fname)
                ext = ".csv.gz" if up.endswith(".CSV.GZ") else ".csv"
                dst = os.path.join(datadir, sym + ext)
                try:
                    os.rename(src, dst)
                    print(f"[data] Fichier renommé automatiquement : {fname} -> {sym + ext}", flush=True)
                    return dst
                except OSError:
                    return src
    raise FileNotFoundError(f"Fichier introuvable pour {symbol} dans {datadir}/ ({symbol}.csv.gz)")


def _parse_ts_stdlib(t_str):
    t_str = str(t_str).strip()
    if t_str.isdigit():
        v = int(t_str)
        return v // 1000 if v > 10**11 else v
    if " GMT" in t_str:
        t_str = t_str.split(" GMT")[0].strip()
    if t_str.endswith(" UTC"):
        t_str = t_str[:-4].strip()
    elif t_str.endswith("Z"):
        t_str = t_str[:-1].strip()
    if len(t_str) >= 10 and t_str[2] in (".", "/") and t_str[5] in (".", "/"):
        d_part, rest = (t_str.split(" ", 1) + ["00:00:00"])[:2]
        dd, mm, yyyy = d_part[0:2], d_part[3:5], d_part[6:10]
        t_str = f"{yyyy}-{mm}-{dd} {rest}"
    elif len(t_str) >= 10 and t_str[4] in (".", "/") and t_str[7] in (".", "/"):
        t_str = t_str[:4] + "-" + t_str[5:7] + "-" + t_str[8:]
    dt = datetime.fromisoformat(t_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def encode_pair_stdlib(symbol, datadir="data", since=None):
    """Même encodage 5m en pur Python standard (sans numpy ni pandas), compatible exports JForex/Dukascopy."""
    path = _resolve_file(symbol, datadir)
    since_ts = _parse_ts_stdlib(since) if since else None

    op = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="replace")) if path.endswith(".gz") else (lambda p: open(p, "r", encoding="utf-8", errors="replace"))
    with op(path) as f:
        first_line = f.readline().strip()
    sep = ";" if first_line.count(";") > first_line.count(",") else ("\t" if first_line.count("\t") > first_line.count(",") else ",")

    rows = []
    seen = set()
    with op(path) as f:
        rdr = csv.reader(f, delimiter=sep)
        header = next(rdr, None)
        if not header:
            raise ValueError(f"Fichier vide : {path}")
        hl = [h.strip().lower() for h in header]
        has_header = not any(ch.isdigit() for ch in hl[0])
        if has_header:
            t_idx = next((i for i, h in enumerate(hl) if h in ("time", "gmt time", "local time", "datetime", "date", "timestamp", "open_time", "time (utc)", "time_utc")), 0)
            o_idx = next((i for i, h in enumerate(hl) if h in ("open", "o", "open_price")), 1)
            h_idx = next((i for i, h in enumerate(hl) if h in ("high", "h", "high_price")), 2)
            l_idx = next((i for i, h in enumerate(hl) if h in ("low", "l", "low_price")), 3)
            c_idx = next((i for i, h in enumerate(hl) if h in ("close", "c", "close_price")), 4)
        else:
            t_idx, o_idx, h_idx, l_idx, c_idx = 0, 1, 2, 3, 4
            ts = _parse_ts_stdlib(header[t_idx])
            if since_ts is None or ts >= since_ts:
                seen.add(ts)
                rows.append((ts, float(header[o_idx]), float(header[h_idx]), float(header[l_idx]), float(header[c_idx])))

        for r in rdr:
            if len(r) < 5:
                continue
            ts = _parse_ts_stdlib(r[t_idx])
            if since_ts is not None and ts < since_ts:
                continue
            if ts in seen:
                continue
            seen.add(ts)
            rows.append((ts, float(r[o_idx]), float(r[h_idx]), float(r[l_idx]), float(r[c_idx])))

    rows.sort(key=lambda x: x[0])
    mid_c = rows[len(rows) // 2][4]
    div = pick_div(mid_c)
    t0 = rows[0][0] // 300
    o0 = round(rows[0][1] * div)
    v = [t0, o0]
    pt, pc = t0, o0
    for ts, o_f, h_f, l_f, c_f in rows:
        ti = ts // 300
        oi = round(o_f * div)
        hi = round(h_f * div)
        li = round(l_f * div)
        ci = round(c_f * div)
        v.extend((ti - pt, oi - pc, hi - oi, oi - li, ci - oi))
        pt, pc = ti, ci
    b64 = base64.b64encode(gzip.compress(json.dumps([v], separators=(",", ":")).encode(), 9)).decode()
    return {"name": symbol, "div": div, "b64": b64}


def trades_to_json(tr, strat, sym, be_rr=0.0):
    """DataFrame de trades (FVG, OTE ou SB) -> liste de dicts pour le visualiseur."""
    out = []
    for r in tr.to_dict("records"):
        x = dict(sym=sym, strat=strat, side=r["side"], te=ep(r["entry_time"]), tx=ep(r["exit_time"]), entry=float(r["entry"]), exit=float(r["exit"]),
                 sl=float(r["sl"]), tp=float(r["tp"]), reason=r["reason"], r=round(float(r["r"]), 3), pnl=round(float(r["pnl"]), 2), be=float(be_rr or 0))
        if strat == "FVG":
            x.update(ts=ep(r["signal_time"]), tf=r["conf_tf"], h1p=nz(r["mss_h1"]), h1t=ep(r["h1_time"]), h2p=nz(r["mss_h2"]), h2t=ep(r["h2_time"]),
                     l2p=nz(r["mss_l2"]), l2t=ep(r["l2_time"]), zlo=nz(r["zone_lo"]), zhi=nz(r["zone_hi"]), zt=ep(r["zone_form"]),
                     fl=nz(r["fvg_lo"]), fh=nz(r["fvg_hi"]), ft1=ep(r["fvg_t1"]))
        elif strat in ("OPR", "SB"):
            x.update(ts=ep(r["signal_time"]), tf=r.get("conf_tf", "5m"), sess=r.get("session", ""),
                     zlo=nz(r.get("zone_lo")), zhi=nz(r.get("zone_hi")), zt=ep(r.get("zone_form")), zt2=ep(r.get("zone_end")),
                     fl=nz(r.get("fvg_lo")), fh=nz(r.get("fvg_hi")), ft1=ep(r.get("fvg_t1")))
        else:
            g = lambda k: r.get(k)  # noqa: E731
            x.update(ts=ep(g("signal_time")) or ep(r["entry_time"]), tf=str(g("confirm") or ""), rh=nz(g("fib_range_high")), rht=ep(g("fib_range_high_time")),
                     rl=nz(g("fib_range_low")), rlt=ep(g("fib_range_low_time")), ozl=nz(g("ote_zone_low")), ozh=nz(g("ote_zone_high")),
                     lh=nz(g("fib_leg_high")), lht=ep(g("fib_leg_high_time")), ll=nz(g("fib_leg_low")), llt=ep(g("fib_leg_low_time")),
                     bt=ep(g("bubble_time")), m1=nz(g("mss_first_extreme")), m1t=ep(g("h1_time")),
                     m2=nz(g("mss_second_extreme")), m2t=ep(g("h2_time")), mo=nz(g("mss_origin_level")), mot=ep(g("l2_time")),
                     fl=nz(g("fvg_lo")), fh=nz(g("fvg_hi")), ft1=ep(g("fvg_t1")), tr_ratio=nz(g("time_ratio")))
        out.append(x)
    return out


def write_trades_json(trades, path):
    json.dump({"version": 2, "trades": trades}, open(path, "w"), separators=(",", ":"))
    return path


def build(symbols, datadir, out, since=None):
    pairs = []
    if np is not None and pd is not None:
        import data_forex as data
        import opt_common as oc
        for s in symbols:
            oc.ensure_data(s, datadir)
            P = data.PairData(s, datadir)
            e = encode_pair(P, s, since)
            pairs.append(e)
            print(f"{s}: {len(e['b64']) / 1e6:.1f} Mo", flush=True)
    else:
        for s in symbols:
            print(f"Encodage de {s} (mode standard sans numpy)...", flush=True)
            e = encode_pair_stdlib(s, datadir, since)
            pairs.append(e)
            print(f"{s}: {len(e['b64']) / 1e6:.1f} Mo", flush=True)
    html = open(os.path.join(HERE, "..", "viewer", "template.html"), encoding="utf-8").read().replace("/*PAIRS*/[]", json.dumps(pairs, separators=(",", ":")))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write(html)
    print(f"{out} : {len(html) / 1e6:.1f} Mo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--out", default="viewer/viewer.html")
    ap.add_argument("--since", default="", help="ne garder que les bougies depuis cette date (réduit la taille), ex. 2023-01-01")
    a = ap.parse_args()
    build([s.strip().upper() for s in a.symbols.split(",") if s.strip()], a.datadir, a.out, a.since or None)
