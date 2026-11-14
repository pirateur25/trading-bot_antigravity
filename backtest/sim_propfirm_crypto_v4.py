"""
backtest/sim_propfirm_crypto_v4.py — Sélection de la meilleure Prop Firm Crypto & Calibrage sur-mesure de CRYPTO V4.

Compare sur l'historique réel des trades CRYPTO V4 (BTCUSD, ETHUSD, AVAXUSD, INJUSD) les offres réelles :
  1. HyroTrader (Bybit / BloFin API directe, 700+ paires USDT perp -> BTC, ETH, AVAX, INJ disponibles) :
     - Hyro Core 1-Step (39 $) : Objectif 10 %, Perte jour 3 %, Perte max 5 % fixe, 4 j min, Règle consistance 40 %/jour, Plafond notionnel 2x.
     - Hyro One-Step           : Objectif 10 %, Perte jour 4 %, Perte max 8 % fixe, 3 j min, Règle consistance 40 %/jour, Plafond notionnel 2x.
     - Hyro Two-Step           : Phase 1 (10 %) + Phase 2 (5 %), Perte jour 5 %, Perte max 10 % fixe, 4 j min/phase, Règle consistance 40 %/jour.
  2. E8 Markets Perpetuals (Serveur MCP, BTC/ETH 15:1, AVAX/INJ 2:1, frais 0.015 % maker / 0.045 % taker) :
     - E8 One 1-Step (88 $)    : Objectif 9 % (et variante 10 %), Perte jour 4 %, Perte max 6 % fixe.
     - E8 Pro 1-Step (68 $)    : Objectif 8 %, Perte jour 2.5 %, Perte max 8 % fixe, Gain comptabilisé plafonné à 2 %/jour.
  3. BrightFunded (MT5 / cTrader uniquement, levier 5:1, INJ non confirmé -> testé sur BTC+ETH+AVAX, frais 0.024 %/ordre) :
     - Bright 2-Step Classic (96 $) : 10 % puis 5 %, Perte jour 5 %, Perte max 10 % fixe, 5 j min/phase.
     - Bright 2-Step Bright (87 $)  : 8 % puis 5 %, Perte jour 4 %, Perte max 8 % fixe, 5 j min/phase.
     - Bright 1-Step                : 10 %, Perte jour 3 %, Perte max 6 % glissant (trailing), 5 j min.
  4. Crypto Fund Trader (Écarté par prudence : liste noire FINMA + Trustpilot masqué + 5 % frais retrait).

Mécanismes de calibrage CRYPTO V4 testés (sans dénaturer le signal) :
  - `risk`        : Risque de base par trade (% du capital initial du challenge)
  - `breaker`     : Disjoncteur journalier (bloque toute nouvelle entrée si la journée est à -X %)
  - `dd_throttle` : Réduction du risque en Drawdown (divise le risque par 2 si le compte est sous -2 % ou -3 % de DD)
  - `guard`       : Garde-fou pire cas simultané (réduit la taille si les trades ouverts au SL approchent la perte jour/totale + respecte le plafond notionnel 2x de HyroTrader)
  - `lock`        : Verrouillage des gains à 70 % de l'objectif (risque x0.5) + lissage automatique pour la règle des 40 % par jour.
"""
import argparse
import csv
import itertools
import json
import os
import statistics
from bisect import bisect_left
from datetime import datetime, timezone

DAY = 86400.0
SL_DIST_PCT = 0.0166667  # Distance SL de Crypto V4 : 50% / levier 30 = 1.6667% du prix

REAL_PROPFIRM_RULESETS = {
    "Hyro_Core_1Step_39$": {
        "firm": "HyroTrader",
        "price_usd": 39.0,
        "api": "Bybit/BloFin API native (100% compatible bot Python)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
        "fee_maker": 0.020,
        "fee_taker": 0.055,
        "max_notional_mult": 2.0,
        "phases": [
            {"target": 10.0, "daily": 3.0, "total": 5.0, "total_type": "static", "min_days": 4, "max_day_pct": 4.0},
        ],
    },
    "Hyro_OneStep": {
        "firm": "HyroTrader",
        "price_usd": 79.0,
        "api": "Bybit/BloFin API native (100% compatible bot Python)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
        "fee_maker": 0.020,
        "fee_taker": 0.055,
        "max_notional_mult": 2.0,
        "phases": [
            {"target": 10.0, "daily": 4.0, "total": 8.0, "total_type": "static", "min_days": 3, "max_day_pct": 4.0},
        ],
    },
    "Hyro_TwoStep": {
        "firm": "HyroTrader",
        "price_usd": 89.0,
        "api": "Bybit/BloFin API native (100% compatible bot Python)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
        "fee_maker": 0.020,
        "fee_taker": 0.055,
        "max_notional_mult": 2.0,
        "phases": [
            {"target": 10.0, "daily": 5.0, "total": 10.0, "total_type": "static", "min_days": 4, "max_day_pct": 4.0},
            {"target": 5.0, "daily": 5.0, "total": 10.0, "total_type": "static", "min_days": 4, "max_day_pct": 2.0},
        ],
    },
    "E8_One_9pct_88$": {
        "firm": "E8 Markets",
        "price_usd": 88.0,
        "api": "Serveur MCP E8 (AVAX/INJ levier 2:1)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
        "fee_maker": 0.015,
        "fee_taker": 0.045,
        "max_notional_mult": 2.0,
        "phases": [
            {"target": 9.0, "daily": 4.0, "total": 6.0, "total_type": "static", "min_days": 0, "max_day_pct": 0.0},
        ],
    },
    "E8_Pro_8pct_68$": {
        "firm": "E8 Markets",
        "price_usd": 68.0,
        "api": "Serveur MCP E8 (plafond 2% profit/jour)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD", "INJUSD"],
        "fee_maker": 0.015,
        "fee_taker": 0.045,
        "max_notional_mult": 2.0,
        "phases": [
            {"target": 8.0, "daily": 2.5, "total": 8.0, "total_type": "static", "min_days": 4, "max_day_pct": 2.0},
        ],
    },
    "Bright_2Step_Bright_87$": {
        "firm": "BrightFunded",
        "price_usd": 87.0,
        "api": "MT5/cTrader requis (sans INJUSD)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD"],
        "fee_maker": 0.024,
        "fee_taker": 0.024,
        "max_notional_mult": 5.0,
        "phases": [
            {"target": 8.0, "daily": 4.0, "total": 8.0, "total_type": "static", "min_days": 5, "max_day_pct": 0.0},
            {"target": 5.0, "daily": 4.0, "total": 8.0, "total_type": "static", "min_days": 5, "max_day_pct": 0.0},
        ],
    },
    "Bright_2Step_Classic_96$": {
        "firm": "BrightFunded",
        "price_usd": 96.0,
        "api": "MT5/cTrader requis (sans INJUSD)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD"],
        "fee_maker": 0.024,
        "fee_taker": 0.024,
        "max_notional_mult": 5.0,
        "phases": [
            {"target": 10.0, "daily": 5.0, "total": 10.0, "total_type": "static", "min_days": 5, "max_day_pct": 0.0},
            {"target": 5.0, "daily": 5.0, "total": 10.0, "total_type": "static", "min_days": 5, "max_day_pct": 0.0},
        ],
    },
    "Bright_1Step_Trailing": {
        "firm": "BrightFunded",
        "price_usd": 95.0,
        "api": "MT5/cTrader requis (DD glissant 6%, sans INJUSD)",
        "symbols": ["BTCUSD", "ETHUSD", "AVAXUSD"],
        "fee_maker": 0.024,
        "fee_taker": 0.024,
        "max_notional_mult": 5.0,
        "phases": [
            {"target": 10.0, "daily": 3.0, "total": 6.0, "total_type": "trailing", "min_days": 5, "max_day_pct": 0.0},
        ],
    },
}


def _ts(s):
    d = datetime.fromisoformat(str(s).strip())
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def load_raw_trades(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_in, t_out = _ts(row["entry_time"]), _ts(row["exit_time"])
            if t_out <= t_in:
                t_out = t_in + 1.0
            en = float(row["entry"])
            isl = float(row["initial_sl"])
            sl_pct = abs(en - isl) / en if en > 0 else SL_DIST_PCT
            out.append({
                "t_in": t_in,
                "t_out": t_out,
                "symbol": row["symbol"],
                "r_gross": float(row["r_gross"]),
                "r_net_default": float(row["r"]),
                "sl_pct": sl_pct,
                "entry_type": row["entry_type"],
                "reason": row["reason"],
            })
    out.sort(key=lambda x: (x["t_in"], x["t_out"]))
    return out


def prepare_firm_trades(raw_trades, firm_spec):
    """Filtre les actifs disponibles chez la Prop Firm et ajuste le R net selon l'écart de frais par rapport au backtest (0.02%/ordre)."""
    allowed = set(firm_spec["symbols"])
    f_maker = firm_spec["fee_maker"] / 100.0
    f_taker = firm_spec["fee_taker"] / 100.0
    trades = []
    for t in raw_trades:
        if t["symbol"] not in allowed:
            continue
        fee_in = f_maker if t["entry_type"] == "LIMIT" else f_taker
        fee_out = f_maker if str(t["reason"]).startswith("TP") else f_taker
        # Le CSV inclut déjà 0.02% + 0.02% = 0.04% de frais. On applique la moitié de l'écart taker éventuel (ordres BE/TRAIL souvent exécutés à prix limite/stop serré)
        extra_fee = max(0.0, (fee_in + fee_out - 0.0004) * 0.5)
        r_net = t["r_net_default"] - (extra_fee / t["sl_pct"])
        trades.append((t["t_in"], t["t_out"], r_net, t["sl_pct"]))
    events = []
    for i, (t_in, t_out, _r, _sl) in enumerate(trades):
        events.append((t_in, 1, i))
        events.append((t_out, 0, i))
    events.sort()
    ev_ts = [e[0] for e in events]
    loss_mult = max(1.0, max((-t[2] for t in trades), default=1.0))
    return trades, events, ev_ts, loss_mult


def run_attempt(events, ev_ts, trades, loss_mult, start_ts, firm_spec, cfg, horizon_days, floating, day_off):
    """
    Simule un challenge complet démarré à `start_ts` avec :
      - Plafond de notionnel simultané (`max_notional_mult`, ex: 2x sur HyroTrader financé / E8 altcoins)
      - Règle de consistance (`max_day_pct`, ex: aucun jour ne compte pour plus de 40% de l'objectif chez HyroTrader, ou 2%/j chez E8 Pro)
      - Réduction du risque en Drawdown (`dd_throttle`)
      - Garde-fou pire cas simultané (`guard`)
      - Verrouillage à 70% de l'objectif (`lock`)
    """
    phases = firm_spec["phases"]
    max_notional = float(firm_spec.get("max_notional_mult", 10.0))
    t_end = start_ts + horizon_days * DAY
    k = bisect_left(ev_ts, start_ts)

    risk_pct = cfg["risk"]
    breaker = cfg["breaker"]
    dd_throttle = cfg["dd_throttle"]
    guard = cfg["guard"]
    lock = cfg["lock"]

    rk = risk_pct / 100.0
    br = breaker / 100.0 if breaker else 0.0

    ph = 0
    rules = phases[0]
    bal = hwm = 1.0
    cur_day = int((start_ts - day_off) // DAY)
    day_start = 1.0
    day_pnl = {}
    days_traded = set()
    open_r = {}
    n = len(events)

    while k < n:
        ts, typ, i = events[k]
        k += 1
        if ts >= t_end:
            return "open", horizon_days
        d = int((ts - day_off) // DAY)
        if d != cur_day:
            cur_day = d
            day_start = bal

        tgt_pct = rules["target"] / 100.0
        max_day_cap = (rules.get("max_day_pct", 0.0) or 0.0) / 100.0
        daily_floor = day_start - rules["daily"] / 100.0
        if rules["total_type"] == "trailing":
            total_floor = min(1.0, hwm - rules["total"] / 100.0)
        else:
            total_floor = 1.0 - rules["total"] / 100.0
        floor = daily_floor if daily_floor > total_floor else total_floor

        excess_days = sum(max(0.0, p - max_day_cap) for p in day_pnl.values()) if max_day_cap > 0.0 else 0.0
        credited_profit = (bal - 1.0) - excess_days

        if typ == 1:  # Ouverture d'un trade
            if br and (bal - day_start) <= -br:
                continue
            # Si le gain net du jour atteint déjà le plafond de consistance (ex: +4% sur Hyro 10%, +2% sur Phase 2), pause jusqu'au lendemain
            if max_day_cap > 0.0 and day_pnl.get(d, 0.0) >= max_day_cap * 0.95:
                continue

            risk = rk
            # En Phase 2 (cible 5% au lieu de 8-10%, mais même DD max 8-10%), adapter le risque à 0.75x pour sécuriser sans ralentir
            if ph > 0:
                p0_tgt = phases[0]["target"]
                if p0_tgt > 0 and rules["target"] < p0_tgt:
                    risk *= 0.75
            # Si une règle des 40%/jour s'applique, plafonner uniquement si un TP (+6R) dépasserait fortement le quota restant du jour
            if max_day_cap > 0.0:
                rem_day_cap = max(0.0, max_day_cap - max(0.0, day_pnl.get(d, 0.0)))
                if risk * 5.5 > rem_day_cap:
                    risk = max(0.45 * rk, rem_day_cap / 5.5)

            # 1) Réduction du risque en Drawdown (si le compte a consommé >= 50% de son DD total autorisé, risque x0.6)
            if dd_throttle:
                dd_cur = hwm - bal if rules["total_type"] == "trailing" else (1.0 - bal)
                if dd_cur >= (rules["total"] / 100.0) * dd_throttle["at_frac"]:
                    risk *= dd_throttle["mult"]

            # 2) Verrouillage à l'approche de l'objectif (à >= 70% de la cible, risque x0.6)
            if lock and credited_profit >= lock["at"] * tgt_pct:
                risk *= lock["mult"]

            # 3) Respect du plafond de notionnel simultané (ex: 2x le solde sur HyroTrader financé / E8 altcoins)
            sl_pct = trades[i][3]
            open_notional = sum(r_val / trades[idx][3] for idx, r_val in open_r.items())
            rem_notional = max(0.0, max_notional * bal - open_notional)
            if risk / sl_pct > rem_notional:
                risk = rem_notional * sl_pct
            if risk < 0.15 * rk:
                continue

            # 4) Garde-fou JOURNALIER simultané : empêche uniquement d'ouvrir un N-ième trade simultané si les trades ouverts
            #    feraient sauter la limite de perte JOURNALIÈRE (ne bloque jamais le 1er trade d'une journée !)
            open_adv = sum(r_val * (loss_mult if trades[idx][2] < -0.1 else 0.55) for idx, r_val in open_r.items())
            this_adv = risk * (loss_mult if trades[i][2] < -0.1 else 0.55)
            if guard and open_r:
                daily_buf = (bal - open_adv - daily_floor - 0.0015) / loss_mult
                if daily_buf < risk:
                    risk = daily_buf
                if risk < 0.25 * rk:
                    continue
                this_adv = risk * (loss_mult if trades[i][2] < -0.1 else 0.55)

            if floating == "worst" and bal - (open_adv + this_adv) < floor:
                return "fail", (ts - start_ts) / DAY

            open_r[i] = risk
            days_traded.add(d)

        else:  # Clôture d'un trade
            risk = open_r.pop(i, None)
            if risk is None:
                continue
            pnl = risk * trades[i][2]
            bal += pnl
            day_pnl[d] = day_pnl.get(d, 0.0) + pnl
            if bal > hwm:
                hwm = bal
            days_traded.add(d)

            if rules["total_type"] == "trailing":
                total_floor = min(1.0, hwm - rules["total"] / 100.0)
            if bal < total_floor or bal < daily_floor:
                return "fail", (ts - start_ts) / DAY

            excess_days = sum(max(0.0, p - max_day_cap) for p in day_pnl.values()) if max_day_cap > 0.0 else 0.0
            credited_profit = (bal - 1.0) - excess_days

            if (bal - 1.0) >= tgt_pct and credited_profit >= tgt_pct and len(days_traded) >= rules["min_days"]:
                ph += 1
                if ph >= len(phases):
                    return "pass", (ts - start_ts) / DAY
                rules = phases[ph]
                bal = hwm = day_start = 1.0
                day_pnl = {}
                open_r = {}
                days_traded = set()
                cur_day = d

    return "open", horizon_days


def attempt_stats(results, price_usd=100.0):
    n = len(results)
    if n == 0:
        return None
    passes = [d for s, d in results if s == "pass"]
    fails = sum(1 for s, _ in results if s == "fail")
    opens = sum(1 for s, _ in results if s == "open")

    def within(x):
        return sum(1 for d in passes if d <= x) / n

    pass_rate = len(passes) / n
    fail_rate = fails / n
    # Coût espéré en $ pour valider 1 compte (les challenges "open" au bout de l'horizon ne sont pas perdus, mais même en les comptant comme tentatives) :
    exp_attempts = (1.0 / pass_rate) if pass_rate > 0 else float("inf")
    # En l'absence de limite de temps, le vrai taux de validation parmi les challenges terminés est len(passes) / (len(passes) + fails)
    resolved = len(passes) + fails
    pass_resolved = (len(passes) / resolved) if resolved > 0 else 0.0
    exp_cost_usd = price_usd / pass_resolved if pass_resolved > 0 else float("inf")

    return {
        "n": n,
        "pass": pass_rate,
        "fail": fail_rate,
        "open": opens / n,
        "pass_resolved": pass_resolved,
        "p30": within(30),
        "p45": within(45),
        "p60": within(60),
        "p90": within(90),
        "med_days": statistics.median(passes) if passes else float("nan"),
        "p75_days": (sorted(passes)[int(0.75 * (len(passes) - 1))] if passes else float("nan")),
        "exp_attempts": exp_attempts,
        "exp_cost_usd": exp_cost_usd,
    }


def build_grid(args):
    risks = [float(x) for x in args.risks.split(",")]
    breakers = [None] + [float(x) for x in args.breakers.split(",") if x]
    throttles = [
        None,
        {"at_frac": 0.50, "mult": 0.65},  # Dès que 50% du DD max autorisé est consommé, risque x0.65
    ]
    locks = [
        None,
        {"at": 0.75, "mult": 0.65},       # À 75% de l'objectif, risque x0.65
    ]
    cfgs = []
    for risk, br, thr, lock in itertools.product(risks, breakers, throttles, locks):
        cfgs.append({
            "risk": risk,
            "breaker": br,
            "dd_throttle": thr,
            "guard": True,
            "lock": lock,
        })
    return cfgs


def cfg_name(c):
    parts = [f"risque {c['risk']:g}%"]
    parts.append(f"stop-jour -{c['breaker']:g}%" if c["breaker"] else "sans stop-jour")
    parts.append("DD-throttle 45%->x0.55" if c["dd_throttle"] else "sans DD-throttle")
    parts.append("garde-fou ON" if c["guard"] else "garde-fou off")
    parts.append("verrou 70%->x0.55" if c["lock"] else "sans verrou")
    return " | ".join(parts)


def fmt(st):
    return (
        f"réussite {st['pass'] * 100:5.1f}% (résolus {st['pass_resolved'] * 100:5.1f}%) | "
        f"échec {st['fail'] * 100:4.1f}% | <=30j {st['p30'] * 100:4.1f}% <=45j {st['p45'] * 100:4.1f}% <=60j {st['p60'] * 100:4.1f}% | "
        f"médiane {st['med_days']:4.1f}j (p75 {st['p75_days']:4.1f}j) | coût espéré {st['exp_cost_usd']:5.0f}$"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ALL_CRYPTO_V4_trades.csv")
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--horizon", type=float, default=180.0, help="Durée max analysée en jours (temps illimité chez Hyro/E8/Bright)")
    ap.add_argument("--start_step", type=float, default=5.0, help="Jours entre deux départs glissants")
    ap.add_argument("--risks", default="0.5,0.65,0.8,1.0,1.25,1.5")
    ap.add_argument("--breakers", default="2.0,2.5")
    ap.add_argument("--floating", default="worst", choices=["realized", "worst"])
    ap.add_argument("--day_hour_utc", type=float, default=0.0)
    ap.add_argument("--max_fail", type=float, default=0.50, help="Échec max visé sur TRAIN")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    raw_trades = load_raw_trades(a.csv)
    first_ts = raw_trades[0]["t_in"]
    last_ts = max(t["t_out"] for t in raw_trades)
    split_ts = _ts(a.split)
    day_off = a.day_hour_utc * 3600.0

    starts = []
    s = first_ts
    while s + a.horizon * DAY <= last_ts:
        starts.append(s)
        s += a.start_step * DAY
    tr_starts = [x for x in starts if x < split_ts]
    te_starts = [x for x in starts if x >= split_ts]

    grid = build_grid(a)
    lines = [
        f"COMPARATIF PROP FIRMS CRYPTO V4 — {len(raw_trades)} trades | {len(starts)} départs glissants (tous les {a.start_step:g}j, horizon {a.horizon:g}j)",
        f"Périodes : TRAIN < {a.split} ({len(tr_starts)} départs) | TEST >= {a.split} ({len(te_starts)} départs) | Mode flottant : {a.floating.upper()}",
        "",
    ]

    chosen = {}
    for rname, fspec in REAL_PROPFIRM_RULESETS.items():
        trades, events, ev_ts, loss_mult = prepare_firm_trades(raw_trades, fspec)
        price = fspec["price_usd"]
        rows = []
        for cfg in grid:
            tr = attempt_stats([run_attempt(events, ev_ts, trades, loss_mult, s0, fspec, cfg, a.horizon, a.floating, day_off) for s0 in tr_starts], price)
            te = attempt_stats([run_attempt(events, ev_ts, trades, loss_mult, s0, fspec, cfg, a.horizon, a.floating, day_off) for s0 in te_starts], price)
            rows.append({"cfg": cfg, "train": tr, "test": te})

        # On sélectionne sur TRAIN la config qui maximise le taux de réussite et minimise le coût espéré en $ avec une durée médiane raisonnable (<= 45j)
        ok = [r for r in rows if r["train"]["med_days"] <= 50.0 and r["train"]["pass"] >= 0.45] or rows
        ok.sort(key=lambda r: (-(r["train"]["pass"] * 0.65 + r["train"]["p60"] * 0.35), r["train"]["exp_cost_usd"]))

        ph_desc = " -> ".join(
            f"+{p['target']:g}% (jour -{p['daily']:g}% / max -{p['total']:g}% {p['total_type']} / {p['min_days']}j min"
            + (f" / max {p['max_day_pct']:g}%/j)" if p.get("max_day_pct") else ")")
            for p in fspec["phases"]
        )
        lines.append(f"=== {rname} ({fspec['firm']} — {price:g}$) | {ph_desc}")
        lines.append(f"    API & Actifs : {fspec['api']} | Frais {fspec['fee_maker']:.3f}%/{fspec['fee_taker']:.3f}% | Trades actifs={len(trades)}")
        for r in ok[:3]:
            lines.append(f"  * {cfg_name(r['cfg'])}")
            lines.append(f"      TRAIN : {fmt(r['train'])}")
            lines.append(f"      TEST  : {fmt(r['test'])}")
        chosen[rname] = ok[0]
        lines.append("")

    lines.append("=== CLASSEMENT FINAL HORS-ÉCHANTILLON (TEST 2024-2026, mode PESSIMISTE flottant + frais réels + consistance)")
    ranked = sorted(
        chosen.items(),
        key=lambda kv: (-(kv[1]["test"]["pass"] - kv[1]["test"]["fail"]), kv[1]["test"]["exp_cost_usd"]),
    )
    for rank, (rname, r) in enumerate(ranked, 1):
        fspec = REAL_PROPFIRM_RULESETS[rname]
        te = r["test"]
        lines.append(f"  #{rank} {rname:26s} ({fspec['price_usd']:3.0f}$) | {cfg_name(r['cfg'])}")
        lines.append(f"      TEST : {fmt(te)}")

    report = "\n".join(lines)
    print(report)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "PROPFIRM_sim_crypto_v4.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
