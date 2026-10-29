"""backtest/opt_common.py — briques communes aux optimiseurs Optuna (OTE et FVG), avec Walk-Forward et détection de plateaux."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests


def parse_kv(text):
    """'k=v;k2=v2' -> dict (valeurs converties en int / float / str)."""
    out = {}
    for part in [x.strip() for x in (text or "").split(";") if x.strip()]:
        k, v = [x.strip() for x in part.split("=", 1)]
        for conv in (int, float):
            try:
                v = conv(v)
                break
            except ValueError:
                pass
        out[k] = v
    return out


def pair_score(st, min_trades, dd_limit=14.0):
    """Score asymétrique anti-overfitting :
    - Si return_pct <= 0 : la perte et l'excès de drawdown s'ADDITIONNENT (jamais de division par le DD sur une perte !),
      et la pénalité n'est jamais réduite quand il y a peu de trades.
    - Si return_pct > 0 : Calmar plafonné à +2.0 (pour éviter qu'un seul actif aberrant masque les autres),
      pondéré par min(1, trades/min_trades), moins pénalité si le DD dépasse dd_limit.
    """
    tr = st.get("trades", 0)
    if tr == 0:
        return -0.25
    ret = float(st.get("return_pct", 0.0))
    dd = abs(float(st.get("max_dd_pct", 0.0)))
    dd_excess = max(0.0, (dd - dd_limit) / 5.0)
    if ret <= 0.0:
        return float(np.clip(ret / 4.0 - max(0.0, dd - 6.0) / 6.0, -4.0, -0.05))
    calmar = float(np.clip(ret / max(dd, 3.5), 0.0, 2.5))
    ret_bonus = float(np.clip(ret / 15.0, 0.0, 2.0))
    weight = min(1.0, tr / float(max(min_trades, 1.0)))
    return float((0.6 * calmar + 0.4 * ret_bonus) * weight - dd_excess)


def half_score(rows, min_trades, min_total_trades=10, dd_limit=14.0):
    """Score moyen sur le panier avec pénalité de sous-activité, de paires perdantes et de déséquilibre Long/Short."""
    tot_t = sum(st.get("trades", 0) for st in rows.values())
    if tot_t < min_total_trades:
        return -10.0 + (tot_t / float(max(min_total_trades, 1))) * 5.0
    n = max(1, len(rows))
    scores = [pair_score(st, min_trades, dd_limit=dd_limit) for st in rows.values()]
    mean_sc = float(np.mean(scores))
    # Pénalité si plus de la moitié des paires du panier perdent de l'argent
    losers = sum(1 for st in rows.values() if st.get("return_pct", 0.0) < -0.5)
    lose_ratio = losers / float(n)
    consistency_pen = max(0.0, lose_ratio - 0.35) * 1.5
    # Pénalité si un sens (achat ou vente) est lourdement perdant pendant que l'autre gagne
    rl = sum(st.get("r_long", 0.0) for st in rows.values())
    rs = sum(st.get("r_short", 0.0) for st in rows.values())
    side_pen = 0.25 if (rl < -5.0 and rs > 5.0) or (rs < -5.0 and rl > 5.0) else 0.0
    return mean_sc - consistency_pen - side_pen


def side_r(tr):
    """(somme des R des achats, somme des R des ventes) : sert à détecter un profit porté par un seul sens (= dérive du marché)."""
    if tr is None or len(tr) == 0:
        return 0.0, 0.0
    return float(tr.loc[tr["side"] == "long", "r"].sum()), float(tr.loc[tr["side"] == "short", "r"].sum())


def ensure_data(sym, datadir, start="2021-01-01"):
    """Crypto absente de data/ : téléchargement auto (Binance Vision -> KuCoin -> Bybit)."""
    if any(os.path.exists(os.path.join(datadir, sym + e)) for e in (".csv.gz", ".csv")):
        return
    import data_crypto as dc
    df, src = dc.load(dc.ftmo_to_binance(sym), "5m", start, source="auto", use_cache=False)
    os.makedirs(datadir, exist_ok=True)
    df[["time", "open", "high", "low", "close", "volume"]].to_csv(os.path.join(datadir, sym + ".csv.gz"), index=False)
    print(f"{sym}: {len(df)} bougies 5m ({src})")


def fmt(rows):
    return " | ".join(f"{s}: {st['return_pct']:+.1f}% DD {st['max_dd_pct']:.1f}% {st['trades']}t" for s, st in rows.items())


def tg(text=None, photo=None, document=None):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        return
    try:
        if text:
            for i in range(0, len(text), 4000):
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": text[i:i + 4000]}, timeout=20)
        if photo and os.path.exists(photo):
            with open(photo, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data={"chat_id": chat}, files={"photo": f}, timeout=30)
        if document and os.path.exists(document):
            with open(document, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat}, files={"document": f}, timeout=60)
    except requests.RequestException as e:
        print("TELEGRAM erreur:", e)


def apply_space(trial, space, frozen, base):
    """Tire les leviers de `space` dans `trial` ; les leviers figés prennent leur valeur fixe. Renvoie le dict de paramètres."""
    p = dict(base)
    p.update(frozen)
    for k, spec in space.items():
        if k in frozen:
            continue
        if spec[0] == "cat":
            p[k] = trial.suggest_categorical(k, spec[1])
        elif spec[0] == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2])
        else:
            p[k] = trial.suggest_int(k, spec[1], spec[2])
    return p


def plateau_center(done_trials, space, frozen, base):
    """Sélectionne le centre du plateau stable (médiane/mode du top 15% des essais) plutôt que le pic isolé."""
    valid = [t for t in done_trials if t.value is not None and t.value > -5.0]
    if len(valid) < 5:
        return None
    valid.sort(key=lambda t: t.value, reverse=True)
    top = valid[:max(3, len(valid) // 7)]
    p = dict(base)
    p.update(frozen)
    for k, spec in space.items():
        if k in frozen:
            continue
        vals = [t.user_attrs["p"][k] for t in top if k in t.user_attrs.get("p", {})]
        if not vals:
            continue
        if spec[0] == "cat":
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in spec[1]):
                med = float(np.median(vals))
                p[k] = min(spec[1], key=lambda x: abs(float(x) - med))
            else:
                p[k] = max(set(vals), key=vals.count)
        elif spec[0] == "int":
            p[k] = int(round(float(np.median(vals))))
        else:
            p[k] = round(float(np.median(vals)), 4)
    if "session" in p and isinstance(p["session"], str) and "-" in p["session"]:
        s0, s1 = p["session"].split("-")
        p["session_start"], p["session_end"] = int(s0), int(s1)
    return p


def save_slice_plot(done_trials, space, frozen, outpath, label):
    """Trace les 'Parameter Slices' (Zones Plates vs Pics) pour tous les paramètres optimisés."""
    keys = [k for k in space if k not in frozen]
    if not keys or len(done_trials) < 3:
        return None
    cols = min(3, len(keys))
    rows_n = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.8 * cols, 3.2 * rows_n), squeeze=False)
    for idx, k in enumerate(keys):
        ax = axes[idx // cols][idx % cols]
        xs, ys = [], []
        for t in done_trials:
            if t.value is not None and t.value > -9.5 and k in t.user_attrs.get("p", {}):
                xs.append(t.user_attrs["p"][k])
                ys.append(t.value)
        if xs:
            if isinstance(xs[0], str):
                cats = sorted(set(xs))
                xi = [cats.index(v) for v in xs]
                ax.scatter(xi, ys, alpha=0.55, s=18, c=ys, cmap="RdYlGn")
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=20)
            else:
                ax.scatter(xs, ys, alpha=0.55, s=18, c=ys, cmap="RdYlGn")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(k, fontsize=10)
        ax.set_ylabel("Score WFA / Calmar", fontsize=8)
        ax.grid(alpha=0.25)
    for idx in range(len(keys), rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle(f"Analyse Zones Plates (Parameter Slices) — {label.upper()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    return outpath


def run_study(label, ev, suggest_fn, baseline, space, frozen, a, best_file, summary_file):
    """Boucle Optuna commune avec Walk-Forward, détection de plateau, garde-fou Référence et graphique de tranches."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(a.outdir, 'study_' + label + '.db')}"
    study = optuna.create_study(study_name=f"{label}_v4", storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(35, max(10, a.n_trials // 4))))
    if len(study.trials) == 0:
        try:
            study.enqueue_trial({k: v for k, v in baseline.items() if k in space and k not in frozen})
        except Exception as e:
            print("baseline non injectée:", e)
    t0, limit = time.time(), a.wall_time * 60 - 180

    def objective(trial):
        if time.time() - t0 > limit:
            study.stop()
            raise optuna.TrialPruned()
        p = suggest_fn(trial)
        sc = ev.objective(p)
        trial.set_user_attr("p", p)
        return sc

    def cb(st, tr):
        n = len(st.trials)
        if tr.state.name == "COMPLETE" and st.best_trial.number == tr.number:
            json.dump(st.best_trial.user_attrs["p"], open(os.path.join(a.outdir, best_file), "w"), indent=1)
        if n % a.save_every == 0:
            print(f"  [{n} essais] meilleur score WFA {st.best_value:.3f}", flush=True)

    study.optimize(objective, n_trials=a.n_trials, callbacks=[cb], catch=(Exception,), show_progress_bar=False)
    done = [t for t in study.trials if t.state.name == "COMPLETE" and "p" in t.user_attrs]
    if not done:
        print("Aucun essai valide.")
        return None
    best = study.best_trial
    p_peak = best.user_attrs["p"]
    p_plat = plateau_center(done, space, frozen, baseline)
    sc_plat = ev.objective(p_plat) if p_plat is not None else float("-inf")
    base_score = ev.objective(baseline)

    # Priorité au centre du plateau s'il est proche du pic (écart <= 0.15 ou >= 65% du score pic)
    use_plat = p_plat is not None and sc_plat > -5.0 and (sc_plat >= best.value - 0.15 or (best.value > 0 and sc_plat >= best.value * 0.65))
    cand_p = p_plat if use_plat else p_peak
    cand_sc = sc_plat if use_plat else best.value

    # Garde-fou : si ni le plateau ni le pic ne battent la référence sur le Walk-Forward, on conserve la référence
    use_ref = base_score > cand_sc + 1e-6
    if use_ref:
        p = dict(baseline)
        mode_lbl = "🛡️ RÉFÉRENCE CONSERVÉE (plus robuste que les essais)"
    elif use_plat:
        p = p_plat
        mode_lbl = "🟢 CENTRE DU PLATEAU STABLE"
    else:
        p = p_peak
        mode_lbl = "🔶 PIC OPTUNA"

    if p_plat is not None:
        json.dump(p_plat, open(os.path.join(a.outdir, f"plateau_{best_file}"), "w"), indent=1)
    json.dump(p, open(os.path.join(a.outdir, best_file), "w"), indent=1)

    tried = sorted({k for k in space if k not in frozen})
    rows = {"train": ev.evaluate(p, "train"), "test": ev.evaluate(p, "test")}
    brows = {"train": ev.evaluate(baseline, "train"), "test": ev.evaluate(baseline, "test")}
    ntrades = sum(r["trades"] for r in rows["train"].values())
    L = [f"✅ Optuna {label.upper()} (Walk-Forward) — {len(done)} essais",
         f"Mode retenu : {mode_lbl} (score pic {best.value:.3f} | plateau {sc_plat:.3f} | réf {base_score:.3f})",
         f"Leviers optimisés ({len(tried)}) : {', '.join(tried)}",
         "Leviers figés : " + (", ".join(f"{k}={v}" for k, v in frozen.items()) or "aucun"),
         f"TRAIN (2021-2023) retenu : {fmt(rows['train'])}", f"TRAIN référence : {fmt(brows['train'])}",
         f"TEST (2024+) retenu : {fmt(rows['test'])}", f"TEST référence : {fmt(brows['test'])}"]
    tst = sum(r["return_pct"] for r in rows["test"].values())
    tb = sum(r["return_pct"] for r in brows["test"].values())
    for per, rr in (("TRAIN", rows["train"]), ("TEST", rows["test"])):
        sl, ss = sum(r.get("r_long", 0) for r in rr.values()), sum(r.get("r_short", 0) for r in rr.values())
        L.append(f"Sens {per} : achats {sl:+.1f} R | ventes {ss:+.1f} R")
        tot = sl + ss
        if p.get("side") == "both" and tot > 0 and max(sl, ss) > 0.8 * tot and min(sl, ss) < 0.2 * tot:
            L.append(f"⚠️ {per} : le profit vient presque d'un seul sens → dérive du marché probable.")
    if ntrades < 60:
        L.append(f"⚠️ Seulement {ntrades} trades sur le train : configuration trop sélective.")
    if use_ref:
        L.append("➡️ Aucun essai n'a battu la référence sur le Walk-Forward : la référence a été conservée dans le JSON.")
    elif tst < tb:
        L.append("⚠️ Meilleur sur Walk-Forward mais moins bon que la référence sur TEST 2024+.")
    else:
        L.append("👍 Meilleur que la référence sur TRAIN (Walk-Forward) ET sur TEST 2024+.")
    txt = "\n".join(L)
    print("\n" + txt)
    print(json.dumps(p, indent=1))
    open(os.path.join(a.outdir, summary_file), "w", encoding="utf-8").write(txt + "\n")
    slice_png = save_slice_plot(done, space, frozen, os.path.join(a.outdir, f"slices_{label}.png"), label)
    try:
        study.trials_dataframe().to_csv(os.path.join(a.outdir, f"optuna_log_{label}.csv"), index=False)
    except Exception as e:
        print("export log:", e)
    tg(txt, photo=slice_png)
    tg(document=os.path.join(a.outdir, best_file))
    return p
