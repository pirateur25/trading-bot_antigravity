# Contexte du projet (reprise de conversation)

Dépôt : `/home/pirateur25/Documents/Trading/trading-bot_antigravity` (Linux). Réponses en français, concises.
Backtests lancés sur GitHub Actions (workflow `.github/workflows/backtest.yml`), résultats via Telegram.

## Stratégies
- **OTE** (`backtest/strategy_ote.py`) et **FVG** (`backtest/strategy_fvg.py`) : optimisées Optuna sur 12 ans (Forex : EURUSD, GBPUSD, USDJPY), params dans `results/best_params_ote.json` / `best_params_fvg.json`.
- **OPR + RV3 — Opening Price Range & Expected Move Calibré (Romain Bailleul)** (`backtest/strategy_opr.py`, optimisateur Walk-Forward `backtest/optimize_opr.py`, workflow `.github/workflows/optimize_opr.yml`) : stratégie M5 sur les indices US (`NAS100` / `US100`, `USA500` / `US500`, `US30`) :
  - **Zone OPR (non négociable)** : plus haut (`OPR High`) et plus bas (`OPR Low`) mèche comprise de la bougie M15 de **15h30 → 15h45 heure française** (`Europe/Paris`, soit les 3 bougies M5 15h30, 15h35, 15h40).
  - **Filtre de tendance H1 obligatoire (`mm20_50`)** : haussière si `MM20 > MM50` et `clôture H1 en cours > MM50` ; baissière en miroir (`MM20 < MM50` et `clôture < MM50`).
  - **1 seul trade / ordre maximum par jour et par indice**, pris entre 15h45 et **19h30 heure française** :
    1. **Setup Cassure** : clôture M5 au-dessus de l'`OPR High` (haussier) → BUY LIMIT sur `OPR High` (et miroir en baissier sous l'`OPR Low`).
    2. **Setup Réintégration** : sortie préalable sous l'`OPR Low` puis clôture M5 à l'intérieur de l'OPR (`OPR Low <= close <= OPR High`) → BUY LIMIT sur `OPR Low`, TP sur `OPR High` (et miroir en baissier).
  - **Modèle RV3 (calibré sur 60 séances US `15h30 -> 22h00` FR)** :
    - **Ligne A (`EM_brut`)** : `EM_brut = Open_15h30 * (VIX / 100) / 16` (`VXNCLS-US100`, `VIXCLS-US500`, `VXDCLS-US30`).
    - **Ligne B (Moyenne calibrée 60j)** : `B_up = Open + mean(r_up) * EM_brut`, `B_dn = Open - mean(r_dn) * EM_brut`.
    - **Ligne C (Enveloppe +1σ 60j)** : `C_up = Open + (mean(r_up) + std(r_up)) * EM_brut`, `C_dn = Open - (mean(r_dn) + std(r_dn)) * EM_brut`.
    - **Seuil de bruit (20% EM)** : `Seuil_Bruit = em_noise_pct * EM_brut` (`0.20 * EM_brut` par défaut).
  - **Sélection dynamique du SL (`SL_distance`)** :
    - `VIX > 20` (Stress) : `SL_distance = max(SL_M5, sl_atr_mult * ATR_H1_14)` (ou `add_atr` selon levier Optuna).
    - `15 <= VIX <= 20` (Standard) : `SL_distance = max(SL_M5, Seuil_Bruit, Plancher_Indice)` avec plancher proportionnel au prix (`MIN_SL_PCT` : `0.18%` US30/USA500, `0.22%` NAS100, soit ~80/70/12.5 pts aux cours actuels mais adapté aux cours 2015–2021).
    - `VIX < 15` (Euphorie) : `sl_vix15_mode` (`"m5_only"` = SL sous la bougie M5 de cassure, `"half_noise"` = 10% EM, `"full_noise"` = 20% EM).
  - **Filtres Anti-Range & Qualité de Cassure** :
    - `min_vix` : ignore les séances de range mort où `VIX/VXN/VXD < min_vix` (ex: 2017).
    - `min_opr_em` / `max_opr_em` : ignore les OPR trop étroits (`< 20% EM`) ou déjà épuisés (`> 55% EM`).
    - `min_mm_spread_atr` : exige un écart minimal `|MM20 - MM50| / ATR_H1` en H1 pour éviter les ranges plats.
    - `true_breakout_only` : exige que la bougie M5 de Cassure franchisse effectivement la borne OPR (`close[k-1] <= OPR High < close[k]`).
    - `max_fill_bars` & `end_hour` : délai max de déclenchement de l'ordre limite et heure limite de nouveau signal (`17h30`, `18h30`, `19h30` FR).
  - **Filtre d'espace disponible avant BE (`res_be_mode`) & TP (`rv3_tp_mode`)** : annulation si obstacle (`B`, `C`, `PDH/PDL`, `PSH/PSL`, `FVG H1`) avant `+be_rr * SL_distance` ; gestion du TP par rapport aux bornes RV3 (`B`/`C`) et résistances.
  - **Visualiseur RV3** (`viewer/template.html` & `viewer/viewer.html`) : affiche l'Open US (jaune), les lignes A (gris), B (cyan/rouge), C (teal/rouge foncé), les zones ombrées `B -> C` et le tableau complet de calibration RV3 en bas à droite.
- Simulation FTMO (`backtest/sim_challenge.py`) avec superposition des trades : OPR seule / OTE+FVG+OPR / OTE+FVG.
- Visualiseur : `viewer/template.html` -> `viewer/viewer.html` via
  `python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30 --out viewer/viewer.html` (fonctionne sans numpy).
