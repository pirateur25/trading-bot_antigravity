"""
backtest/risk_overlay_crypto_v4.py — Test de règles de GESTION DU RISQUE sur les trades CRYPTO V4 déjà générés.

La stratégie (entrées / SL / TP / trailing) n'est PAS modifiée : on rejoue chronologiquement
`results/ALL_CRYPTO_V4_trades.csv` avec des politiques de sizing différentes :
  - réduction du risque en drawdown (palier avec hystérésis, ou linéaire)
  - plafond de positions ouvertes simultanément (les cryptos sont corrélées)

Méthode anti-surapprentissage :
  1. Choix de la règle sur TRAIN (entrées avant --split), jugement sur TEST (après --split), jamais l'inverse.
  2. Mesure de stabilité : corrélation de rang (Spearman) du Calmar TRAIN vs TEST sur toute la grille.
  3. Monte-Carlo par blocs mensuels (rééchantillonnage des mois) pour comparer les distributions de drawdown.
  4. Aucune règle n'est adoptée automatiquement : le rapport indique seulement si elle bat la référence
     sur TRAIN *et* sur TEST *et* en Monte-Carlo.

Sizing : à l'ouverture, 1R = capital RÉALISÉ à cet instant (trades déjà clôturés) * risk% * multiplicateur ;
marge = 1R / (SL ROI 50%) ; PnL du trade = marge * roi_pct / 100 (roi_pct vient du CSV, net de frais).
Le drawdown est mesuré sur le capital réalisé (sans PnL latent).

Dépendances : bibliothèque standard uniquement.
Usage : python backtest/risk_overlay_crypto_v4.py [--csv results/ALL_CRYPTO_V4_trades.csv] [--split 2024-01-01]
"""
import argparse
import csv
import itertools
import os
import random
import statistics
from datetime import datetime, timezone

SL_ROI = 0.50          # 1R = -50% de ROI sur la marge
MONTH_SEC = 32 * 86400


def _ts(s):
    d = datetime.fromisoformat(str(s).strip())
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def load_trades(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t_in, t_out = _ts(r["entry_time"]), _ts(r["exit_time"])
            if t_out <= t_in:          # SL touché dans la bougie d'entrée : la clôture doit passer APRÈS l'ouverture
                t_out = t_in + 1.0
            out.append({"sym": r["symbol"], "t_in": t_in, "t_out": t_out, "roi": float(r["roi_pct"]) / 100.0})
    out.sort(key=lambda x: (x["t_in"], x["t_out"]))
    return out


# --------------------------------------------------------------------------- politiques
def policy_mult(pol, dd, reduced):
    """Retourne (multiplicateur de risque, nouvel état 'réduit') selon le drawdown courant `dd` (0..1)."""
    kind = pol["kind"]
    if kind == "none":
        return 1.0, False
    if kind == "flat":                # risque FIXE réduit (référence honnête : se dé-lever sans règle)
        return pol["m"], False
    if kind == "step":
        x = pol["x"] / 100.0
        if reduced:
            if dd <= x / 2.0:         # hystérésis : on ne revient à 100% que quand le DD est revenu à la moitié du seuil
                reduced = False
        elif dd >= x:
            reduced = True
        return (pol["m"] if reduced else 1.0), reduced
    if kind == "lin":
        x = pol["x"] / 100.0
        return max(pol["floor"], 1.0 - dd / x), False
    raise ValueError(kind)


def pol_name(pol, conc):
    if pol["kind"] == "none":
        base = "Référence (aucune réduction)"
    elif pol["kind"] == "flat":
        base = f"Risque FIXE {1.5 * pol['m']:.2f}% (sans règle)"
    elif pol["kind"] == "step":
        base = f"DD>={pol['x']:g}% -> risque x{pol['m']:g} (retour à 100% si DD<={pol['x'] / 2:g}%)"
    else:
        base = f"Linéaire : x(1-DD/{pol['x']:g}%) plancher x{pol['floor']:g}"
    return base + (f" | max {conc} pos." if conc else " | pos. illimitées")


# --------------------------------------------------------------------------- rejeu
def replay(trades, pol, conc=None, start_cap=600.0, risk_pct=1.5):
    events = []
    for i, t in enumerate(trades):
        events.append((t["t_in"], 1, i))
        events.append((t["t_out"], 0, i))   # à égalité de temps, les clôtures passent avant les ouvertures
    events.sort()
    bal = hwm = float(start_cap)
    reduced = False
    open_m = {}
    exits_t, exits_bal = [], []
    taken = skipped = 0
    rk = risk_pct / 100.0
    max_dd = 0.0
    for ts, typ, i in events:
        if typ == 1:
            if conc and len(open_m) >= conc:
                skipped += 1
                continue
            dd = 1.0 - bal / hwm if hwm > 0 else 0.0
            mult, reduced = policy_mult(pol, dd, reduced)
            margin = (max(10.0, bal) * rk * mult) / SL_ROI
            open_m[i] = margin
            taken += 1
        else:
            if i not in open_m:
                continue
            margin = open_m.pop(i)
            bal = max(10.0, bal + margin * trades[i]["roi"])
            hwm = max(hwm, bal)
            max_dd = max(max_dd, 1.0 - bal / hwm)
            exits_t.append(ts)
            exits_bal.append(bal)
    return summarize(trades, exits_t, exits_bal, start_cap, max_dd, taken, skipped)


def summarize(trades, exits_t, exits_bal, start_cap, max_dd, taken, skipped):
    if not exits_t:
        return {"final": start_cap, "cagr": 0.0, "max_dd": 0.0, "calmar": 0.0, "worst6": 0.0, "neg_m": 0.0, "taken": 0, "skipped": skipped}
    final = exits_bal[-1]
    years = max((exits_t[-1] - trades[0]["t_in"]) / (365.25 * 86400), 0.25)
    cagr = (final / start_cap) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    # soldes de fin de mois (dernier solde connu) pour pire fenêtre 6 mois et % de mois négatifs
    month_bal = {}
    for ts, b in zip(exits_t, exits_bal):
        d = datetime.fromtimestamp(ts, timezone.utc)
        month_bal[(d.year, d.month)] = b
    keys = sorted(month_bal)
    series = [start_cap] + [month_bal[k] for k in keys]
    rets = [series[i + 1] / series[i] - 1.0 for i in range(len(series) - 1)]
    worst6 = min((series[i + 6] / series[i] - 1.0 for i in range(len(series) - 6)), default=0.0)
    neg_m = sum(1 for r in rets if r < 0) / max(len(rets), 1)
    calmar = cagr / max_dd if max_dd > 1e-9 else 0.0
    return {"final": final, "cagr": cagr, "max_dd": max_dd, "calmar": calmar, "worst6": worst6, "neg_m": neg_m,
            "taken": taken, "skipped": skipped}


# --------------------------------------------------------------------------- Monte-Carlo (blocs mensuels)
def month_blocks(trades):
    blocks = {}
    for t in trades:
        d = datetime.fromtimestamp(t["t_in"], timezone.utc)
        key = (d.year, d.month)
        start = datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp()
        blocks.setdefault(key, []).append((t, start))
    return [blocks[k] for k in sorted(blocks)]


def make_sim(blocks, rng):
    sim = []
    for k in range(len(blocks)):
        blk = blocks[rng.randrange(len(blocks))]
        for t, start in blk:
            sim.append({"sym": t["sym"], "t_in": t["t_in"] - start + k * MONTH_SEC,
                        "t_out": t["t_out"] - start + k * MONTH_SEC, "roi": t["roi"]})
    sim.sort(key=lambda x: (x["t_in"], x["t_out"]))
    return sim


def pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def monte_carlo(blocks, configs, n_sims, seed):
    rng = random.Random(seed)
    res = {i: {"final": [], "max_dd": [], "calmar": []} for i in range(len(configs))}
    for _ in range(n_sims):
        sim = make_sim(blocks, rng)      # MÊME tirage pour toutes les politiques => comparaison appariée
        for i, (pol, conc) in enumerate(configs):
            r = replay(sim, pol, conc)
            res[i]["final"].append(r["final"])
            res[i]["max_dd"].append(r["max_dd"])
            res[i]["calmar"].append(r["calmar"])
    out = []
    for i in range(len(configs)):
        d = res[i]
        out.append({
            "med_final": statistics.median(d["final"]), "p5_final": pct(d["final"], 0.05),
            "med_dd": statistics.median(d["max_dd"]), "p95_dd": pct(d["max_dd"], 0.95),
            "p_dd50": sum(1 for x in d["max_dd"] if x >= 0.5) / n_sims,
            "med_calmar": statistics.median(d["calmar"]),
        })
    return out


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


# --------------------------------------------------------------------------- main
def build_grid():
    pols = [{"kind": "none"}] + [{"kind": "flat", "m": m} for m in (0.5, 0.67, 0.83)]
    for x, m in itertools.product((10, 15, 20, 25, 30), (0.75, 0.5, 0.33)):
        pols.append({"kind": "step", "x": x, "m": m})
    for x in (40, 60):
        pols.append({"kind": "lin", "x": x, "floor": 0.25})
    return [(pol, conc) for pol in pols for conc in (None, 4, 3, 2)]


def fmt_row(name, r):
    return (f"{name[:78]:78s} | final {r['final']:>10,.0f} | CAGR {r['cagr'] * 100:6.1f}% | DDmax {r['max_dd'] * 100:5.1f}% | "
            f"Calmar {r['calmar']:5.2f} | pire6m {r['worst6'] * 100:6.1f}% | mois<0 {r['neg_m'] * 100:4.1f}% | pris {r['taken']} (ignorés {r['skipped']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ALL_CRYPTO_V4_trades.csv")
    ap.add_argument("--split", default="2024-01-01", help="début de la période TEST (TRAIN = avant)")
    ap.add_argument("--capital", type=float, default=600.0)
    ap.add_argument("--risk_pct", type=float, default=1.5)
    ap.add_argument("--sims", type=int, default=500, help="nombre de tirages Monte-Carlo")
    ap.add_argument("--topk", type=int, default=6, help="nb de règles (meilleures sur TRAIN) passées au Monte-Carlo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    trades = load_trades(a.csv)
    split_ts = _ts(a.split)
    train = [t for t in trades if t["t_in"] < split_ts]
    test = [t for t in trades if t["t_in"] >= split_ts]
    grid = build_grid()

    rows = []
    for pol, conc in grid:
        rows.append({
            "name": pol_name(pol, conc), "pol": pol, "conc": conc,
            "full": replay(trades, pol, conc, a.capital, a.risk_pct),
            "train": replay(train, pol, conc, a.capital, a.risk_pct),
            "test": replay(test, pol, conc, a.capital, a.risk_pct),
        })
    ref = next(r for r in rows if r["pol"]["kind"] == "none" and r["conc"] is None)
    flats = [r for r in rows if r["pol"]["kind"] in ("none", "flat") and r["conc"] is None]
    cand = [r for r in rows if r["pol"]["kind"] in ("step", "lin")]

    lines = []
    lines.append(f"RISK OVERLAY CRYPTO V4 — {len(trades)} trades ({len(train)} TRAIN avant {a.split} / {len(test)} TEST), risque de base {a.risk_pct:g}%, capital {a.capital:g}")
    lines.append("La stratégie n'est pas modifiée : seuls le sizing et le nombre de positions simultanées changent.")
    lines.append("Capital TRAIN et TEST repartent chacun de la même valeur de départ (comparaison de style, pas de continuité).\n")
    lines.append("== 1. RÉFÉRENCE (rejeu avec capital réalisé à l'ouverture)")
    lines.append("FULL  " + fmt_row(ref["name"], ref["full"]))
    lines.append("TRAIN " + fmt_row(ref["name"], ref["train"]))
    lines.append("TEST  " + fmt_row(ref["name"], ref["test"]))

    # stabilité TRAIN -> TEST
    sp_calmar = spearman([r["train"]["calmar"] for r in rows], [r["test"]["calmar"] for r in rows])
    sp_dd = spearman([-r["train"]["max_dd"] for r in rows], [-r["test"]["max_dd"] for r in rows])
    lines.append(f"\n== 2. STABILITÉ TRAIN -> TEST sur {len(rows)} règles : Spearman Calmar = {sp_calmar:+.2f} | Spearman (-DDmax) = {sp_dd:+.2f}")
    lines.append("   (proche de +1 = la hiérarchie des règles se retrouve hors échantillon ; proche de 0 = bruit)")

    # sélection sur TRAIN uniquement
    lines.append("\n== 2b. RÉFÉRENCES « SE DÉ-LEVER SANS RÈGLE » (risque fixe plus faible, positions illimitées)")
    for r in flats:
        lines.append("FULL  " + fmt_row(r["name"], r["full"]))
    ranked = sorted(cand, key=lambda r: r["train"]["calmar"], reverse=True)
    lines.append(f"\n== 3. TOP {a.topk} RÈGLES CHOISIES SUR TRAIN (puis jugées sur TEST)")
    top = ranked[:a.topk]
    for r in top:
        lines.append(f"- {r['name']}")
        lines.append("    TRAIN " + fmt_row("", r["train"])[80:])
        lines.append("    TEST  " + fmt_row("", r["test"])[80:])
        lines.append(f"    TEST vs réf. : DDmax {r['test']['max_dd'] * 100:.1f}% (réf {ref['test']['max_dd'] * 100:.1f}%) | "
                     f"Calmar {r['test']['calmar']:.2f} (réf {ref['test']['calmar']:.2f}) | CAGR {r['test']['cagr'] * 100:.1f}% (réf {ref['test']['cagr'] * 100:.1f}%)")

    # Monte-Carlo : référence + top + quelques références simples
    mc_cfg = [(r["pol"], r["conc"]) for r in flats] + [(r["pol"], r["conc"]) for r in top]
    simple = [({"kind": "none"}, 3), ({"kind": "step", "x": 20, "m": 0.5}, None), ({"kind": "step", "x": 20, "m": 0.5}, 3)]
    for pol, conc in simple:
        if (pol, conc) not in mc_cfg:
            mc_cfg.append((pol, conc))
    blocks = month_blocks(trades)
    mc = monte_carlo(blocks, mc_cfg, a.sims, a.seed)
    lines.append(f"\n== 4. MONTE-CARLO ({a.sims} historiques synthétiques par rééchantillonnage des {len(blocks)} mois, tirages identiques pour toutes les règles)")
    lines.append(f"{'règle':78s} | médiane finale | 5e pct finale | DDmax méd. | DDmax 95e pct | P(DD>=50%) | Calmar méd.")
    for (pol, conc), m in zip(mc_cfg, mc):
        lines.append(f"{pol_name(pol, conc)[:78]:78s} | {m['med_final']:>14,.0f} | {m['p5_final']:>13,.0f} | {m['med_dd'] * 100:9.1f}% | {m['p95_dd'] * 100:12.1f}% | {m['p_dd50'] * 100:9.1f}% | {m['med_calmar']:10.2f}")

    # verdict mécanique (ne modifie rien)
    n_fl = len(flats)
    lines.append("\n== 5. VERDICT MÉCANIQUE : une règle n'est retenue que si elle bat le MEILLEUR « risque fixe » (même sans règle) en Calmar")
    lines.append("   sur TRAIN, sur TEST et en Monte-Carlo (médiane), sans augmenter le DDmax 95e pct Monte-Carlo vs ce meilleur risque fixe.")
    best_tr = max(flats, key=lambda r: r["train"]["calmar"])["train"]["calmar"]
    best_te = max(flats, key=lambda r: r["test"]["calmar"])["test"]["calmar"]
    best_mc = max(mc[:n_fl], key=lambda m: m["med_calmar"])
    verdict = []
    for idx, r in enumerate(top):
        m = mc[n_fl + idx]
        ok = (r["train"]["calmar"] > best_tr and r["test"]["calmar"] > best_te
              and m["med_calmar"] > best_mc["med_calmar"] and m["p95_dd"] <= best_mc["p95_dd"])
        verdict.append(ok)
        lines.append(f"  [{'OUI' if ok else 'non'}] {r['name']}  (Calmar TRAIN {r['train']['calmar']:.2f}/{best_tr:.2f} | TEST {r['test']['calmar']:.2f}/{best_te:.2f} | MC {m['med_calmar']:.2f}/{best_mc['med_calmar']:.2f})")
    if not any(verdict):
        lines.append("  => Aucune règle ne bat le simple fait de réduire le risque fixe : une règle de drawdown ne crée pas d'avantage, elle dé-lève seulement.")
        lines.append("     Le levier utile est alors le niveau de risque choisi (risk_pct) selon le drawdown que tu peux supporter.")

    report = "\n".join(lines)
    print(report)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "RISK_OVERLAY_crypto_v4.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(a.outdir, "RISK_OVERLAY_crypto_v4_grid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regle", "periode", "final", "cagr", "max_dd", "calmar", "pire_6m", "mois_negatifs", "pris", "ignores"])
        for r in rows:
            for per in ("full", "train", "test"):
                s = r[per]
                w.writerow([r["name"], per, round(s["final"], 2), round(s["cagr"], 4), round(s["max_dd"], 4), round(s["calmar"], 3),
                            round(s["worst6"], 4), round(s["neg_m"], 4), s["taken"], s["skipped"]])


if __name__ == "__main__":
    main()
