"""
backtest/robustness_crypto_v4.py — Test de ROBUSTESSE (pas d'optimisation) des réglages de CRYPTO V4.

On part des réglages actuels et on fait varier UN SEUL paramètre à la fois, autour de sa valeur actuelle (±), pour vérifier que
les réglages sont sur un PLATEAU (la perf reste proche quand on bouge un peu) et non sur un pic isolé (= signe de surapprentissage).

Pour chaque variante : R moyen par trade (net de frais), erreur-type, séparé TRAIN (< --split) / TEST (>= --split), R cumulé, DD en R,
nombre d'années positives. Rien n'est adopté automatiquement : le rapport signale seulement une variante qui améliore TRAIN *et* TEST
de plus d'une erreur-type chacune (indice d'amélioration réelle plutôt que de bruit).

Usage : python backtest/robustness_crypto_v4.py --symbols BTC,ETH,AVAX,INJ --start 2020-01-01 --split 2024-01-01
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_forex as data_fx  # noqa: E402
import run_crypto_v4 as rc  # noqa: E402
import strategy_crypto_v4 as scv4  # noqa: E402

# (nom lisible, paramètre, valeur) — voisinage proche des valeurs actuelles
VARIANTS = [
    ("m15_max_range_pct", 1.2), ("m15_max_range_pct", 1.4), ("m15_max_range_pct", 1.8), ("m15_max_range_pct", 2.0),
    ("m15_min_range_pct", 0.3), ("m15_min_range_pct", 0.5),
    ("sl_cooldown_hours", 1.0), ("sl_cooldown_hours", 3.0), ("sl_cooldown_hours", 4.0),
    ("h4_thresh", 0.25), ("h4_thresh", 0.30), ("h4_thresh", 0.40), ("h4_thresh", 0.45),
    ("h4_lookback", 6), ("h4_lookback", 10),
    ("m15_lookback", 12), ("m15_lookback", 20),
    ("poi_fib", 0.45), ("poi_fib", 0.55),
    ("tp_roi_pct", 250.0), ("tp_roi_pct", 350.0),
]

# Variantes de trailing (contribution de chaque palier) — appliquées en remplaçant scv4.TRAIL_TIERS le temps du run
TRAIL_VARIANTS = {
    "trailing: aucun palier": (),
    "trailing: BE seulement (+50% -> 0)": ((50.0, 0.0),),
    "trailing: sans palier +200%": scv4.TRAIL_TIERS[:3],
}


def eval_trades(tr, split_ts):
    """Statistiques R d'un ensemble de trades (DataFrame avec colonnes r, exit_time)."""
    if tr is None or len(tr) == 0:
        return None
    t = tr.copy()
    t["_x"] = pd.to_datetime(t["exit_time"], utc=True)
    t = t.sort_values("_x").reset_index(drop=True)

    def stats(d):
        r = d["r"].to_numpy(dtype=float)
        n = len(r)
        if n == 0:
            return {"n": 0, "avg": 0.0, "se": 0.0, "sum": 0.0}
        sd = float(r.std(ddof=1)) if n > 1 else 0.0
        return {"n": n, "avg": float(r.mean()), "se": sd / np.sqrt(n) if n > 1 else 0.0, "sum": float(r.sum())}

    r_all = t["r"].to_numpy(dtype=float)
    eq = np.cumsum(r_all)
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    yearly = t.groupby(t["_x"].dt.year)["r"].mean()
    return {
        "all": stats(t), "train": stats(t[t["_x"] < split_ts]), "test": stats(t[t["_x"] >= split_ts]),
        "dd_r": dd, "years_pos": int((yearly > 0).sum()), "years": int(len(yearly)),
    }


def run_variant(Ps, base_p, overrides, start_cap, risk_pct, fee_pct, lev, a):
    p = dict(base_p)
    p.update(overrides)
    frames = []
    for sym, P in Ps.items():
        i0 = P.idx_at(a.start) if a.start else 0
        i1 = P.idx_at(pd.Timestamp(a.end) + pd.Timedelta(days=1)) if a.end else P.n
        tr, _ = scv4.run(P, capital=start_cap, risk_pct=risk_pct, fee_pct=fee_pct, leverage=float(p["leverage"]), p=p, i0=i0, i1=i1)
        if len(tr):
            frames.append(tr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt_line(name, ev, base):
    if ev is None:
        return f"{name:44s} | aucun trade"
    tr, te, al = ev["train"], ev["test"], ev["all"]
    d_tr = (tr["avg"] - base["train"]["avg"])
    d_te = (te["avg"] - base["test"]["avg"])
    flag = ""
    if d_tr > base["train"]["se"] and d_te > base["test"]["se"]:
        flag = "  <-- meilleur sur TRAIN ET TEST (> 1 erreur-type)"
    elif d_tr < -base["train"]["se"] and d_te < -base["test"]["se"]:
        flag = "  <-- pire sur TRAIN ET TEST"
    return (f"{name:44s} | n={al['n']:5d} | TRAIN {tr['avg']:+.3f}R (±{tr['se']:.3f}) | TEST {te['avg']:+.3f}R (±{te['se']:.3f}) | "
            f"cumul {al['sum']:+7.1f}R | DD {ev['dd_r']:6.1f}R | années+ {ev['years_pos']}/{ev['years']}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(scv4.CRYPTO_V4_SYMBOLS))
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--capital", type=float, default=600.0)
    ap.add_argument("--risk_pct", type=float, default=1.5)
    ap.add_argument("--fee_pct", type=float, default=0.02)
    ap.add_argument("--wall_time", type=float, default=300.0, help="durée max en minutes (arrêt propre avec rapport partiel)")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    t0 = time.time()
    split_ts = pd.Timestamp(a.split, tz="UTC")
    syms = []
    for raw in a.symbols.split(","):
        ns = rc.normalize_crypto_sym(raw)
        if ns and ns not in syms:
            syms.append(ns)

    Ps = {}
    for sym in syms:
        rc.ensure_crypto_data(sym, a.datadir, a.start)
        Ps[sym] = data_fx.PairData(sym, a.datadir)
        print(f"[data] {sym} chargé ({Ps[sym].n} bougies 5m)", flush=True)

    base_p = scv4.parse_params("")
    base_p["fee_pct"] = a.fee_pct
    base_p["compound"] = 0   # le R par trade ne dépend pas du sizing ; compound=0 évite tout recalcul inutile

    end_txt = a.end or "aujourd'hui"
    lines = [f"ROBUSTESSE CRYPTO V4 — {', '.join(syms)} | {a.start} -> {end_txt} | TRAIN < {a.split} <= TEST | frais {a.fee_pct:g}%/ordre", ""]

    print("[run] référence...", flush=True)
    base_tr = run_variant(Ps, base_p, {}, a.capital, a.risk_pct, a.fee_pct, 30.0, a)
    base = eval_trades(base_tr, split_ts)
    if base is None:
        print("Aucun trade en référence : arrêt.")
        return
    lines.append(fmt_line("RÉFÉRENCE (réglages actuels)", base, base))
    lines.append("")
    results = []
    stopped = False

    cur_param = None
    for k, v in VARIANTS:
        if (time.time() - t0) / 60.0 > a.wall_time:
            stopped = True
            break
        if k != cur_param:
            lines.append(f"-- {k} (actuel = {base_p[k]:g})")
            cur_param = k
        t1 = time.time()
        ev = eval_trades(run_variant(Ps, base_p, {k: v}, a.capital, a.risk_pct, a.fee_pct, 30.0, a), split_ts)
        line = fmt_line(f"   {k} = {v:g}", ev, base)
        lines.append(line)
        results.append((k, v, ev))
        print(f"{line}   [{time.time() - t1:.0f}s]", flush=True)

    lines.append("")
    lines.append("-- Contribution du trailing (variantes structurelles, à lire comme un test de nécessité, pas comme une piste d'optimisation)")
    orig_tiers = scv4.TRAIL_TIERS
    try:
        for name, tiers in TRAIL_VARIANTS.items():
            if (time.time() - t0) / 60.0 > a.wall_time:
                stopped = True
                break
            scv4.TRAIL_TIERS = tiers
            ev = eval_trades(run_variant(Ps, base_p, {}, a.capital, a.risk_pct, a.fee_pct, 30.0, a), split_ts)
            line = fmt_line(f"   {name}", ev, base)
            lines.append(line)
            print(line, flush=True)
    finally:
        scv4.TRAIL_TIERS = orig_tiers

    # lecture du plateau : à quel point la perf bouge-t-elle quand on s'éloigne un peu ?
    lines.append("")
    lines.append("== LECTURE")
    plateau, fragile = 0, []
    for k in dict.fromkeys(k for k, _, _ in results):
        evs = [ev for kk, _, ev in results if kk == k and ev is not None]
        if not evs:
            continue
        spread = max(e["all"]["avg"] for e in evs) - min(e["all"]["avg"] for e in evs)
        worst = min(e["all"]["avg"] for e in evs)
        if worst > 0 and spread < 0.05:
            plateau += 1
        else:
            fragile.append((k, spread, worst))
    lines.append(f"- {plateau} paramètre(s) sur {len(set(k for k, _, _ in results))} restent positifs avec un écart de R moyen < 0.05R autour de leur valeur actuelle (plateau).")
    for k, spread, worst in fragile:
        lines.append(f"- {k} : sensible (écart {spread:.3f}R entre variantes, pire variante {worst:+.3f}R) -> ne pas toucher sans validation hors échantillon.")
    lines.append("- Une variante marquée « meilleur sur TRAIN ET TEST » est seulement une piste : à confirmer sur une 2e période / d'autres cryptos avant de l'adopter.")
    if stopped:
        lines.append("- ATTENTION : arrêt anticipé (wall_time atteint), rapport partiel.")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "ROBUSTNESS_crypto_v4.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    rc.tg(report[:3900])


if __name__ == "__main__":
    main()
