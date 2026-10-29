"""backtest/opt_common.py — briques communes aux optimiseurs Optuna (OTE et FVG)."""
import os

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


def pair_score(st, min_trades):
    """Rendement / drawdown (plafonné à ±3) pondéré par min(1, trades/min_trades).
    Avant : une paire sous le minimum valait -3 => le paysage était plat (tout = -3) et Optuna ne trouvait rien.
    Maintenant : peu de trades = score réduit proportionnellement, donc l'optimiseur a un gradient pour y remonter."""
    if st["trades"] == 0:
        return 0.0
    s = float(np.clip(st["return_pct"] / max(abs(st["max_dd_pct"]), 2.0), -3, 3))
    return s * min(1.0, st["trades"] / float(min_trades))


def half_score(rows, min_trades):
    return float(np.mean([pair_score(st, min_trades) for st in rows.values()]))


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


def tg(text=None, document=None):
    token, chat = os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID")
    if not token or not chat:
        return
    try:
        if text:
            for i in range(0, len(text), 4000):
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": text[i:i + 4000]}, timeout=20)
        if document and os.path.exists(document):
            with open(document, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data={"chat_id": chat}, files={"document": f}, timeout=60)
    except requests.RequestException as e:
        print("TELEGRAM erreur:", e)


def run_study(label, ev, suggest_fn, baseline, space, frozen, a, best_file, summary_file):
    """Boucle Optuna commune. `baseline` = réglages de référence : ils sont testés EN PREMIER (trial 0) et servent de comparaison
    finale (si l'optimisation ne bat pas la référence sur TRAIN, ou si elle perd sur TEST, le résumé le dit clairement)."""
    import json
    import time
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{os.path.join(a.outdir, 'study_' + label + '.db')}"
    study = optuna.create_study(study_name=f"{label}_v2", storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(40, max(10, a.n_trials // 5))))
    if len(study.trials) == 0:
        try:
            study.enqueue_trial({k: v for k, v in baseline.items() if k in space and k not in frozen})
        except Exception as e:  # référence hors bornes : on continue sans
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
            print(f"  [{n} essais] meilleur score train {st.best_value:.3f}", flush=True)

    study.optimize(objective, n_trials=a.n_trials, callbacks=[cb], catch=(Exception,), show_progress_bar=False)
    done = [t for t in study.trials if t.state.name == "COMPLETE" and "p" in t.user_attrs]
    if not done:
        print("Aucun essai valide.")
        return None
    best = study.best_trial
    p = best.user_attrs["p"]
    json.dump(p, open(os.path.join(a.outdir, best_file), "w"), indent=1)
    tried = sorted({k for k in space if k not in frozen})
    base_score = next((t.value for t in done if t.number == 0), None)
    rows = {"train": ev.evaluate(p, "train"), "test": ev.evaluate(p, "test")}
    brows = {"train": ev.evaluate(baseline, "train"), "test": ev.evaluate(baseline, "test")}
    ntrades = sum(r["trades"] for r in rows["train"].values())
    L = [f"✅ Optuna {label.upper()} terminé — {len(done)} essais",
         f"Leviers optimisés ({len(tried)}) : {', '.join(tried)}",
         "Leviers figés : " + (", ".join(f"{k}={v}" for k, v in frozen.items()) or "aucun"),
         f"Score TRAIN (pire des 2 moitiés) : optimisé {best.value:.3f} | référence {base_score if base_score is None else round(base_score, 3)}",
         f"TRAIN optimisé : {fmt(rows['train'])}", f"TRAIN référence : {fmt(brows['train'])}",
         f"TEST 2024+ optimisé : {fmt(rows['test'])}", f"TEST 2024+ référence : {fmt(brows['test'])}"]
    tst = sum(r["return_pct"] for r in rows["test"].values())
    tb = sum(r["return_pct"] for r in brows["test"].values())
    for per, rr in (("TRAIN", rows["train"]), ("TEST", rows["test"])):
        sl, ss = sum(r.get("r_long", 0) for r in rr.values()), sum(r.get("r_short", 0) for r in rr.values())
        L.append(f"Sens {per} : achats {sl:+.1f} R | ventes {ss:+.1f} R")
        tot = sl + ss
        if p.get("side") == "both" and tot > 0 and max(sl, ss) > 0.8 * tot and min(sl, ss) < 0.2 * tot:
            L.append(f"⚠️ {per} : le profit vient presque d'un seul sens → dérive du marché probable, pas un avantage réel.")
    if ntrades < 40:
        L.append(f"⚠️ Seulement {ntrades} trades sur le train : configuration trop sélective, résultat non fiable.")
    if base_score is not None and best.value <= base_score + 1e-9:
        L.append("➡️ L'optimisation n'a PAS battu la référence sur le train : garde la référence.")
    elif tst < tb:
        L.append("⚠️ Meilleur que la référence sur TRAIN mais PIRE sur TEST (somme des rendements) : probable sur-apprentissage.")
    else:
        L.append("👍 Meilleur que la référence sur TRAIN et sur TEST.")
    txt = "\n".join(L)
    print("\n" + txt)
    print(json.dumps(p, indent=1))
    open(os.path.join(a.outdir, summary_file), "w", encoding="utf-8").write(txt + "\n")
    try:
        study.trials_dataframe().to_csv(os.path.join(a.outdir, f"optuna_log_{label}.csv"), index=False)
    except Exception as e:
        print("export log:", e)
    tg(txt)
    tg(document=os.path.join(a.outdir, best_file))
    return p


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
