"""
backtest/v2_common.py — Briques 2.0 issues de CRYPTO V4 pour OTE 2.0, FVG 2.0 et OPR 2.0 (OTR 2.0),
ainsi que le moteur d'optimisation Walk-Forward multi-paliers (3 paliers séquentiels + 4 plis temporels).

Innovations CRYPTO V4 transposées aux stratégies Forex / Métaux / Indices US :
  1. Trailing SL progressif multi-paliers en R (`trail_mode` : "be_only", "step_1r", "step_05r", "step_buffered")
     et cadence de vérification (`trail_check` : "bar", "close_30m", "hybrid_30m").
  2. Biais directionnel H4 par efficience (`biais_h4 = net_H4 / range_H4` sur `h4_lookback` bougies 4h clôturées)
     avec relais Momentum M15 (`mom_m15 = net_M15 / range_M15` sur 16 bougies 15m clôturées) :
     `h4_bias_mode` in {"none", "contra_block", "crypto_v4", "strict"}.
  3. Fenêtre de volatilité M15 normalisée par l'ATR H1 (`m15_min_atr`, `m15_max_atr`) :
     bloque les marchés morts (range M15 trop étroit) et les krachs / annonces violentes (range M15 explosif).
  4. Cooldown après Stop-Loss (`sl_cooldown_hours`) et anti-répétition du même POI / zone invalidée
     (`block_same_poi`, `same_poi_tol_pct`).
  5. Ré-analyse dynamique toutes les 30 min des ordres limites au repos (`cancel_on_flip`) :
     annule l'ordre en attente si le biais H4/M15 se retourne violemment avant exécution.
"""
import json
import os
import time

import numpy as np
import pandas as pd

NS_MIN = 60 * 10**9
NS_HOUR = 3600 * 10**9
NS_30M = 30 * NS_MIN

V2_DEFAULT_PARAMS = {
    # --- 1) Trailing SL multi-paliers (inspiré de CRYPTO V4) ---
    "trail_mode": "be_only",        # "be_only" (1.0) | "step_1r" (Crypto V4) | "step_05r" (fin) | "step_buffered"
    "trail_check": "bar",           # "bar" (chaque bougie 5m) | "close_30m" (clôture 30m) | "hybrid_30m" (BE barre + paliers 30m)
    # --- 2) Biais H4 efficience & Momentum M15 (inspiré de CRYPTO V4) ---
    "h4_bias_mode": "none",         # "none" (1.0) | "contra_block" | "crypto_v4" | "strict"
    "h4_lookback": 8,               # Nb de bougies H4 clôturées pour net/range (8 = 32h)
    "h4_thresh": 0.35,              # Seuil d'efficience directionnelle H4 (0.25 .. 0.45)
    "m15_lookback": 16,             # Nb de bougies M15 clôturées (16 = 4h)
    # --- 3) Fenêtre de volatilité M15 (normalisée par ATR H1) ---
    "m15_min_atr": 0.0,             # Plancher range M15 (16 bougies) / ATR_H1 (0.0 = désactivé, ex: 1.0, 1.3)
    "m15_max_atr": 0.0,             # Plafond anti-krach range M15 / ATR_H1 (0.0 = désactivé, ex: 4.0, 5.0)
    # --- 4) Cooldown après SL & Anti-répétition du même POI / zone cassée ---
    "sl_cooldown_hours": 0.0,       # Pause en heures après un SL < -0.1R (0.0 = désactivé, ex: 1.0, 2.0, 4.0)
    "block_same_poi": 0,            # 1 = interdit de reprendre la même zone / POI qui vient de toucher SL
    "same_poi_tol_pct": 0.15,       # Tolérance d'identité du POI en % du prix (ex: 0.10%, 0.15%, 0.25%)
    # --- 5) Ré-évaluation 30m des ordres limites au repos ---
    "cancel_on_flip": 0,            # 1 = annule l'ordre limite en attente si le biais H4/M15 s'inverse fortement
}


def _rolling_efficiency(h, l, c, lb):
    """Calcule de façon causale (net / range) et (range) sur une fenêtre glissante de `lb` bougies."""
    n = len(c)
    eff = np.zeros(n, dtype=np.float64)
    rng = np.zeros(n, dtype=np.float64)
    lb = max(2, int(lb))
    if n < lb:
        return eff, rng
    s_h = pd.Series(h)
    s_l = pd.Series(l)
    win_hi = s_h.rolling(lb).max().to_numpy(dtype=np.float64)
    win_lo = s_l.rolling(lb).min().to_numpy(dtype=np.float64)
    rng = np.maximum(0.0, win_hi - win_lo)
    net = np.zeros(n, dtype=np.float64)
    net[lb - 1:] = c[lb - 1:] - c[:n - lb + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(rng > 0.0, net / rng, 0.0)
    eff = np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.nan_to_num(rng, nan=0.0, posinf=0.0, neginf=0.0)
    return eff, rng


def compute_h4_m15_state(b_close_ns, f4, f15, f1h, h4_lb=8, m15_lb=16, cache_dict=None, h4_lookback=None, m15_lookback=None):
    """
    Précalcule et projette de manière strictement causale sur les bougies 5m de base :
      - `biais_h4`    : efficience directionnelle H4 `[-1, +1]` sur `h4_lb` bougies 4h clôturées
      - `mom_m15`     : efficience / momentum M15 `[-1, +1]` sur `m15_lb` bougies 15m clôturées
      - `rng_m15_atr` : amplitude M15 (haut - bas sur `m15_lb` bougies) divisée par l'ATR(14) H1 clôturé
      - `is_30m` / `is_30m_close` : booléen vrai sur les clôtures de bougies 30m (`:00` et `:30`)
    """
    h4_lb = int(h4_lookback if h4_lookback is not None else (h4_lb or 8))
    m15_lb = int(m15_lookback if m15_lookback is not None else (m15_lb or 16))
    key = ("v2_h4_m15_state", h4_lb, m15_lb, len(b_close_ns))
    if cache_dict is not None and key in cache_dict:
        return cache_dict[key]

    nb = len(b_close_ns)
    eff4, _ = _rolling_efficiency(f4["h"], f4["l"], f4["c"], h4_lb)
    eff15, rng15 = _rolling_efficiency(f15["h"], f15["l"], f15["c"], m15_lb)

    j4 = np.searchsorted(f4["close_ns"], b_close_ns, side="right") - 1
    j15 = np.searchsorted(f15["close_ns"], b_close_ns, side="right") - 1
    j1h = np.searchsorted(f1h["close_ns"], b_close_ns, side="right") - 1

    biais_h4 = np.where(j4 >= h4_lb - 1, eff4[np.clip(j4, 0, max(0, len(eff4) - 1))], 0.0)
    mom_m15 = np.where(j15 >= m15_lb - 1, eff15[np.clip(j15, 0, max(0, len(eff15) - 1))], 0.0)
    rng_m15 = np.where(j15 >= m15_lb - 1, rng15[np.clip(j15, 0, max(0, len(rng15) - 1))], 0.0)

    atr1h_arr = f1h.get("atr")
    if atr1h_arr is not None and len(atr1h_arr) > 0:
        atr1h = np.where(j1h >= 0, atr1h_arr[np.clip(j1h, 0, len(atr1h_arr) - 1)], np.nan)
    else:
        atr1h = np.full(nb, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        rng_m15_atr = np.where((atr1h > 0) & (~np.isnan(atr1h)), rng_m15 / atr1h, 0.0)

    is_30m = (b_close_ns % NS_30M) == 0

    res = {
        "biais_h4": biais_h4.astype(np.float64),
        "mom_m15": mom_m15.astype(np.float64),
        "rng_m15_atr": rng_m15_atr.astype(np.float64),
        "is_30m": is_30m,
        "is_30m_close": is_30m,
    }
    if cache_dict is not None:
        cache_dict[key] = res
    return res


def get_pair_v2_state(pd_, p):
    """Récupère l'état V2 (H4/M15) directement depuis un objet PairData `pd_`."""
    h4_lb = int(p.get("h4_lookback", 8) or 8)
    m15_lb = int(p.get("m15_lookback", 16) or 16)
    return compute_h4_m15_state(
        pd_.base["close_ns"],
        pd_.frames["4h"],
        pd_.frames["15m"],
        pd_.frames["1h"],
        h4_lb=h4_lb,
        m15_lb=m15_lb,
        cache_dict=pd_.cache,
    )


def pass_v2_entry_filters(idx: int, side_int: int, v2_state: dict, p: dict) -> bool:
    """
    Vérifie les filtres d'entrée CRYPTO V4 à la bougie `idx` pour `side_int` (+1 long, -1 short) :
      1) `h4_bias_mode` ("none", "contra_block", "crypto_v4", "strict")
      2) `m15_min_atr` et `m15_max_atr` (fenêtre de volatilité M15 / ATR H1)
    """
    if v2_state is None:
        return True

    h4_mode = str(p.get("h4_bias_mode", "none") or "none").lower()
    if h4_mode != "none":
        thresh = float(p.get("h4_thresh", 0.35) or 0.35)
        bh4 = float(v2_state["biais_h4"][idx])
        if h4_mode == "contra_block":
            if side_int == 1 and bh4 < -thresh:
                return False
            if side_int == -1 and bh4 > thresh:
                return False
        elif h4_mode == "strict":
            if side_int == 1 and bh4 < thresh:
                return False
            if side_int == -1 and bh4 > -thresh:
                return False
        elif h4_mode == "crypto_v4":
            if bh4 > thresh:
                dir_ok = (side_int == 1)
            elif bh4 < -thresh:
                dir_ok = (side_int == -1)
            else:
                mm15 = float(v2_state["mom_m15"][idx])
                dir_ok = (side_int == 1 and mm15 > 0.0) or (side_int == -1 and mm15 < 0.0)
            if not dir_ok:
                return False

    min_atr = float(p.get("m15_min_atr", 0.0) or 0.0)
    max_atr = float(p.get("m15_max_atr", 0.0) or 0.0)
    if min_atr > 0.0 or max_atr > 0.0:
        ratr = float(v2_state["rng_m15_atr"][idx])
        if ratr > 0.0:
            if min_atr > 0.0 and ratr < min_atr:
                return False
            if max_atr > 0.0 and ratr > max_atr:
                return False

    return True


def should_cancel_resting_on_flip(idx: int, side_int: int, v2_state: dict, p: dict) -> bool:
    """
    Étape 3 CRYPTO V4 : à chaque clôture 30m, annule un ordre limite en attente si le biais H4
    ET le momentum M15 se sont retournés contre la direction de l'ordre.
    """
    if not int(float(p.get("cancel_on_flip", 0) or 0)) or v2_state is None:
        return False
    is_30m_arr = v2_state.get("is_30m_close", v2_state.get("is_30m"))
    if is_30m_arr is not None and not bool(is_30m_arr[idx]):
        return False
    thresh = float(p.get("h4_thresh", 0.35) or 0.35)
    bh4 = float(v2_state["biais_h4"][idx])
    mm15 = float(v2_state["mom_m15"][idx])
    if side_int == 1:
        return (bh4 < -thresh and mm15 < 0.0) or (mm15 < -0.65)
    return (bh4 > thresh and mm15 > 0.0) or (mm15 > 0.65)


def get_trail_tiers_r(trail_mode: str, be_rr: float) -> list[tuple[float, float]]:
    """
    Construit la liste ordonnée des paliers `(trigger_r, lock_r)` en multiples du risque initial `1 R`.
    """
    mode = str(trail_mode or "be_only").lower()
    be_val = float(be_rr) if (be_rr is not None and 0.0 < float(be_rr) < 100.0) else 0.0

    if mode in ("none", "be_only"):
        return [(be_val, 0.0)] if be_val > 0.0 else []

    be0 = be_val if (0.0 < be_val < 2.0) else 1.0

    if mode == "step_05r":
        raw = [
            (be0, 0.0),
            (1.5, 0.5),
            (2.0, 1.0),
            (2.5, 1.5),
            (3.0, 2.0),
            (3.5, 2.5),
            (4.0, 3.0),
            (5.0, 4.0),
        ]
    elif mode == "step_buffered":
        raw = [
            (be0, 0.0),
            (2.0, 0.75),
            (3.0, 1.75),
            (4.0, 2.75),
            (5.0, 3.75),
            (6.0, 4.75),
        ]
    else:  # "step_1r" / "crypto_v4" (loi exacte Crypto V4 en R : +1R->0R, +2R->+1R, +3R->+2R, +4R->+3R)
        raw = [
            (be0, 0.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 3.0),
            (5.0, 4.0),
            (6.0, 5.0),
        ]

    out = []
    last_trig = 0.0
    for trig, lock in raw:
        if trig > last_trig and lock < trig:
            out.append((float(trig), float(lock)))
            last_trig = float(trig)
    return out


def compute_new_trail_sl(
    side,
    entry: float,
    cur_sl: float,
    risk_dist: float,
    cur_tier: int,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    is_30m_bar: bool,
    tiers: list[tuple[float, float]],
    trail_check: str = "bar",
) -> tuple[float, int]:
    """
    Met à jour `(cur_sl, cur_tier)` de manière monotone (le SL ne recule jamais) :
      - `trail_check == "bar"`        : tous les paliers sont évalués sur l'extrême 5m (`bar_high` / `bar_low`)
      - `trail_check == "close_30m"`  : tous les paliers sont évalués sur la clôture 30m (`bar_close`) comme Crypto V4
      - `trail_check == "hybrid_30m"` : le palier 1 (Break-Even) est évalué sur l'extrême 5m et les paliers suivants (> 0R verrouillé) sur clôture 30m.
    """
    if not tiers or risk_dist <= 0.0:
        return cur_sl, cur_tier

    side_int = 1 if side in (1, "long", "LONG") else -1
    r_ext = ((bar_high - entry) / risk_dist) if side_int == 1 else ((entry - bar_low) / risk_dist)
    r_cls = ((bar_close - entry) / risk_dist) if side_int == 1 else ((entry - bar_close) / risk_dist)

    new_sl = cur_sl
    new_tier = cur_tier

    for idx, (trig_r, lock_r) in enumerate(tiers, 1):
        if idx <= new_tier:
            continue
        if trail_check == "close_30m":
            if not is_30m_bar or r_cls < trig_r:
                continue
        elif trail_check == "hybrid_30m":
            if idx == 1 or lock_r <= 0.15:
                if r_ext < trig_r:
                    continue
            else:
                if not is_30m_bar or r_cls < trig_r:
                    continue
        else:  # "bar"
            if r_ext < trig_r:
                continue

        cand_sl = (entry + lock_r * risk_dist) if side_int == 1 else (entry - lock_r * risk_dist)
        if (side_int == 1 and cand_sl > new_sl) or (side_int == -1 and cand_sl < new_sl):
            new_sl = float(cand_sl)
        new_tier = idx

    return new_sl, new_tier


def compute_multi_fold_bounds(open_ns, split_str: str, n_folds: int = 4, warmup_days: int = 180):
    """
    Découpe la période TRAIN `[i_start, i_split)` en `n_folds` plis Walk-Forward équilibrés
    (ex: 4 plis de ~2.1 ans chacun sur 2015–2023 = ~150 000 bougies 5m par pli !).
    """
    i_start = int(len(open_ns) and (open_ns < open_ns[0] + warmup_days * 86400 * 10**9).sum())
    t = pd.Timestamp(split_str)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    i_split = int(np.searchsorted(open_ns, t.value, side="left"))
    span = max(n_folds, i_split - i_start)
    folds = []
    for k in range(n_folds):
        a = i_start + (k * span) // n_folds
        b = i_split if k == n_folds - 1 else (i_start + ((k + 1) * span) // n_folds)
        folds.append((int(a), int(b)))
    return {
        "start": int(i_start),
        "split": int(i_split),
        "end": int(len(open_ns)),
        "folds": folds,
    }


def score_multi_fold(f_full: dict, fold_rows_list: list[dict], pair_target: float, min_tot: int, dd_limit: float = 16.0) -> float:
    """
    Score Walk-Forward multi-plis :
      - 45% sur la courbe continue TRAIN complète (vrai Max Drawdown multi-années)
      - 55% sur les `K` plis temporels (50% pire pli + 50% moyenne des plis).
    """
    import opt_common as oc
    tot_tr = sum(r.get("trades", 0) for r in f_full.values())
    if tot_tr < min_tot:
        return -10.0 + (tot_tr / float(max(1, min_tot))) * 5.0

    n_folds = max(1, len(fold_rows_list))
    fold_min = max(6.0, pair_target / float(n_folds))
    fold_min_tot = max(12, min_tot // n_folds)

    s_full = oc.half_score(f_full, pair_target, min_total_trades=min_tot, dd_limit=dd_limit)
    f_scores = [
        oc.half_score(fr, fold_min, min_total_trades=fold_min_tot, dd_limit=max(10.0, dd_limit - 4.0))
        for fr in fold_rows_list
    ]
    s_wfa = 0.5 * min(f_scores) + 0.5 * float(np.mean(f_scores))
    return float(0.45 * s_full + 0.55 * s_wfa)


def run_staged_study(
    label: str,
    ev,
    stages: list[dict],
    full_space: dict,
    initial_frozen: dict,
    baseline_v1: dict,
    a,
    best_file: str,
    summary_file: str,
):
    """
    Optimiseur Optuna par PALIERS SÉQUENTIELS (Multi-Stage Walk-Forward) :
      - Chaque palier (`stage`) libère uniquement son groupe cohérent de leviers tout en gardant les autres figés
        sur le meilleur centre de plateau du palier précédent.
      - À la fin de chaque palier, le centre du plateau stable (top 15%) et le pic sont comparés au palier précédent :
        on ne valide le nouveau palier que s'il améliore (ou égale) le score Walk-Forward sur TRAIN !
      - En fin d'optimisation, compare `Référence 1.0` vs `Palier 1` vs `Palier 2` vs `Palier 3 (2.0 Final)`
        sur TRAIN et sur TEST hors-échantillon.
    """
    import optuna
    import opt_common as oc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    total_wall_sec = float(getattr(a, "wall_time", 300.0)) * 60.0 - 120.0
    total_trials = int(getattr(a, "n_trials", 300))

    cur_best_p = dict(baseline_v1)
    cur_best_p.update(initial_frozen)
    v1_score = float(ev.objective(cur_best_p))
    cur_best_sc = v1_score

    stage_history = [("Référence 1.0", dict(cur_best_p), v1_score)]
    all_done_trials = []

    weights = [float(st.get("weight", 1.0)) for st in stages]
    w_sum = sum(weights) if sum(weights) > 0 else float(len(stages))

    print(f"\n🚀 Début Optimisation Multi-Paliers {label.upper()} ({len(stages)} paliers, {total_trials} essais total) | Score WFA Réf 1.0 = {v1_score:+.3f}", flush=True)

    for s_idx, st in enumerate(stages, 1):
        st_name = st["name"]
        st_keys = [k for k in st["keys"] if k in full_space and k not in initial_frozen]
        if not st_keys:
            continue

        st_trials = max(20, int(round(total_trials * (float(st.get("weight", 1.0)) / w_sum))))
        st_wall_limit = t0 + total_wall_sec * (sum(weights[:s_idx]) / w_sum)

        st_frozen = dict(cur_best_p)
        for k in st_keys:
            st_frozen.pop(k, None)
        st_space = {k: full_space[k] for k in st_keys}

        db_path = os.path.join(a.outdir, f"study_{label}_p{s_idx}.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name=f"{label}_palier_{s_idx}",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + s_idx, n_startup_trials=min(60, max(15, st_trials // 4))),
        )

        # Toujours injecter le gagnant du palier précédent comme essai #0
        try:
            study.enqueue_trial({k: cur_best_p[k] for k in st_keys if k in cur_best_p and k in st_space})
        except Exception:
            pass

        print(f"\n--- Palier {s_idx}/{len(stages)} : {st_name} ({st_trials} essais | leviers: {', '.join(st_keys)}) ---", flush=True)

        def make_obj(st_sp, st_fr, base_p, wall_lim):
            def _obj(trial):
                if time.time() > wall_lim:
                    study.stop()
                    raise optuna.TrialPruned()
                p_cand = oc.apply_space(trial, st_sp, st_fr, base_p)
                if "session" in p_cand and isinstance(p_cand["session"], str) and "-" in p_cand["session"]:
                    s0, s1 = p_cand["session"].split("-")
                    p_cand["session_start"], p_cand["session_end"] = int(s0), int(s1)
                if "max_rr" in p_cand and "min_rr" in p_cand and float(p_cand["max_rr"]) < float(p_cand["min_rr"]):
                    p_cand["max_rr"] = p_cand["min_rr"]
                sc = float(ev.objective(p_cand))
                trial.set_user_attr("p", p_cand)
                return sc
            return _obj

        save_ev = max(10, int(getattr(a, "save_every", 25)))

        def _cb(st_obj, tr_obj):
            n_done = len(st_obj.trials)
            if n_done % save_ev == 0 and st_obj.best_value is not None:
                print(f"  [Palier {s_idx} | {n_done}/{st_trials} essais] meilleur WFA {st_obj.best_value:+.3f}", flush=True)

        study.optimize(
            make_obj(st_space, st_frozen, cur_best_p, st_wall_limit),
            n_trials=st_trials,
            callbacks=[_cb],
            catch=(Exception,),
            show_progress_bar=False,
        )

        done = [t for t in study.trials if t.state.name == "COMPLETE" and "p" in t.user_attrs]
        all_done_trials.extend(done)
        if not done:
            stage_history.append((f"Palier {s_idx} ({st_name}) [inchangé]", dict(cur_best_p), cur_best_sc))
            continue

        best_t = study.best_trial
        p_peak = best_t.user_attrs["p"]
        sc_peak = float(best_t.value)

        p_plat = oc.plateau_center(done, st_space, st_frozen, cur_best_p)
        sc_plat = float(ev.objective(p_plat)) if p_plat is not None else float("-inf")

        use_plat = (
            p_plat is not None
            and sc_plat > -5.0
            and (sc_plat >= sc_peak - 0.12 or (sc_peak > 0 and sc_plat >= sc_peak * 0.75))
        )
        cand_p = p_plat if use_plat else p_peak
        cand_sc = sc_plat if use_plat else sc_peak
        tag = "Plateau" if use_plat else "Pic"

        if cand_sc >= cur_best_sc - 1e-6:
            cur_best_p = dict(cand_p)
            cur_best_sc = cand_sc
            stage_history.append((f"Palier {s_idx} ({st_name}) [{tag}]", dict(cur_best_p), cur_best_sc))
            print(f"  -> Palier {s_idx} validé ({tag}) : score WFA = {cur_best_sc:+.3f}", flush=True)
        else:
            stage_history.append((f"Palier {s_idx} ({st_name}) [gardé Palier {s_idx-1}]", dict(cur_best_p), cur_best_sc))
            print(f"  -> Palier {s_idx} n'améliore pas ({cand_sc:+.3f} < {cur_best_sc:+.3f}) : on conserve l'état précédent.", flush=True)

        json.dump(cur_best_p, open(os.path.join(a.outdir, best_file), "w"), indent=2)

    # Évaluation finale TRAIN & TEST hors-échantillon (Référence 1.0 vs 2.0 Retenu)
    rows_v2 = {"train": ev.evaluate(cur_best_p, "train"), "test": ev.evaluate(cur_best_p, "test")}
    rows_v1 = {"train": ev.evaluate(baseline_v1, "train"), "test": ev.evaluate(baseline_v1, "test")}

    # Sélection automatique des actifs les plus rentables sur TRAIN (sans regarder TEST)
    profitable_syms = [
        s for s, st in rows_v2["train"].items()
        if st.get("return_pct", 0.0) > 0.0 and st.get("trades", 0) >= 8
    ] or list(rows_v2["train"].keys())
    cur_best_p["_recommended_symbols"] = profitable_syms
    cur_best_p["profitable_pairs"] = profitable_syms
    json.dump(cur_best_p, open(os.path.join(a.outdir, best_file), "w"), indent=2)

    split_str = getattr(a, "split", "2024-01-01")
    lines = [
        f"🏆 OPTIMISATION MULTI-PALIERS {label.upper()} (Walk-Forward {getattr(ev, 'n_folds', 4)} plis) — {len(all_done_trials)} essais cumulés",
        "Progression par palier sur TRAIN :",
    ]
    for s_lbl, _, s_val in stage_history:
        lines.append(f"  • {s_lbl:45s} : score WFA = {s_val:+.3f}")

    lines.extend([
        "",
        f"📊 TRAIN (2015 -> {split_str}) :",
        f"  • 2.0 Retenu    : {oc.fmt(rows_v2['train'])}",
        f"  • 1.0 Référence : {oc.fmt(rows_v1['train'])}",
        f"🧪 TEST HORS-ÉCHANTILLON ({split_str}+) :",
        f"  • 2.0 Retenu    : {oc.fmt(rows_v2['test'])}",
        f"  • 1.0 Référence : {oc.fmt(rows_v1['test'])}",
        f"🎯 Actifs les plus rentables sélectionnés sur TRAIN : {', '.join(profitable_syms)}",
    ])

    for per, rr in (("TRAIN", rows_v2["train"]), ("TEST", rows_v2["test"])):
        sl = sum(r.get("r_long", 0.0) for r in rr.values())
        ss = sum(r.get("r_short", 0.0) for r in rr.values())
        lines.append(f"Sens {per} (2.0) : achats {sl:+.1f} R | ventes {ss:+.1f} R")

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open(os.path.join(a.outdir, summary_file), "w", encoding="utf-8").write(txt + "\n")

    active_space = {k: v for k, v in full_space.items() if k not in initial_frozen}
    slice_png = oc.save_slice_plot(all_done_trials, active_space, initial_frozen, os.path.join(a.outdir, f"slices_{label}.png"), label)
    oc.tg(txt, photo=slice_png)
    oc.tg(document=os.path.join(a.outdir, best_file))
    return cur_best_p

