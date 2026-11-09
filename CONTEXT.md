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
    - **Calibration EM (`em_cal`) et Seuil de bruit (20% EM)** : `em_cal = ((mean_up + mean_dn) / (2 * 0.68)) * EM_brut` aligne les indices 30j (`VIXCLS`, `VXNCLS`, `VXDCLS`) sur l'échelle `VIX1D` de Romain Bailleul (`Seuil_Bruit = em_noise_pct * em_cal`, soit ~29% de la borne B).
  - **Sélection dynamique du SL (`SL_distance`)** :
    - `VIX > 20` (Stress) : `SL_distance = max(SL_M5, sl_atr_mult * ATR_H1_14)` (`sl_vix20_mode = "max_atr"`).
    - `15 <= VIX <= 20` (Standard) : `SL_distance = max(SL_M5, Seuil_Bruit, Plancher_Indice)` avec plancher proportionnel au prix (`MIN_SL_PCT` : `0.12%` US30/USA500, `0.15%` NAS100, équivalent au SL indicatif 9 pts sur ES à 7778).
    - `VIX < 15` (Euphorie) : `sl_vix15_mode` (`"m5_only"` = SL sous la bougie M5 de cassure, `"half_noise"` = 10% EM calibré).
  - **Filtres Anti-Range, Anti-Épuisement H1 & Qualité de Cassure** :
    - `min_vix` : ignore les séances de range mort où `VIX/VXN/VXD < min_vix` (ex: 2017).
    - `min_opr_em` / `max_opr_em` : ignore les OPR trop étroits (`< 20% EM_cal`) ou déjà épuisés (`> 55% EM_cal`).
    - `min_mm_spread_atr` / `max_mm_spread_atr` : exige un écart `|MM20 - MM50| / ATR_H1` en H1 compris entre `min_mm_spread_atr` (évite les ranges plats) et `max_mm_spread_atr` (évite d'acheter/vendre en bout de tendance H1 sur-étendue).
    - `true_breakout_only` : exige que la bougie M5 de Cassure franchisse effectivement la borne OPR (`close[k-1] <= OPR High < close[k]`).
    - `max_fill_bars` & `end_hour` : délai max de déclenchement de l'ordre limite et heure limite de nouveau signal (`17h30`, `18h30`, `19h30` FR).
  - **Filtre d'espace disponible avant BE (`res_be_mode`) & TP dynamique (`rv3_tp_mode = "rv3_dyn"`)** :
    - Interdiction absolue d'entrer si la moyenne 60j `B` est déjà dépassée ou à moins de `1.0 R`, ou si un obstacle (`B`, `C`, `PDH/PDL`, `PSH/PSL`, `FVG H1`) bloque le Break-Even (`be_rr >= 1.0`).
    - Mode `"rv3_dyn"` : RR minimum strict de `2.0 R` et extension dynamique vers les cibles structurelles / bornes RV3 `B`/`C` jusqu'à `3.6 R` (moyenne ~`2.6 R`).
  - **Visualiseur RV3** (`viewer/template.html` & `viewer/viewer.html`) : affiche l'Open US (jaune), les lignes A (gris), B (cyan/rouge), C (teal/rouge foncé), les zones ombrées `B -> C` et le tableau complet de calibration RV3 en bas à droite.
- **CRYPTO V4 — A2-R Stratégie SMC Crypto Futures (MoonX / Cubrik)** (`backtest/strategy_crypto_v4.py`, `backtest/run_crypto_v4.py`, workflow `.github/workflows/backtest_crypto_v4.yml`) :
  - **9 cryptos Futures** : `BTC, ETH, SOL, XRP, AVAX, ZEC, INJ, PUMP, HYPE` (téléchargement auto 5m via Binance USD-M Futures, Binance Spot, Bybit Linear Futures, KuCoin, Hyperliquid dans `backtest/data_crypto.py`).
  - **Sizing fixe** : marge `200 USDT` par instrument, levier `×30` isolé (`notionnel = 6 000 USDT`), 1 position ou ordre au repos max par instrument.
  - **TP / SL & Trailing SL (sur ROI marge)** :
    - `1 R = -50 %` de ROI sur marge (`-100 USDT`, mouvement de prix de `1.6667 %`).
    - `TP fixe = +300 %` de ROI sur marge (`+6 R = +600 USDT`, mouvement de prix de `10.0 %`).
    - `Trailing SL` : `ROI >= +50%` $\rightarrow$ SL au BE (`0%` / `0 R`) ; `ROI >= +100%` $\rightarrow$ SL à `+50%` (`+1 R`) ; `ROI >= +150%` $\rightarrow$ SL à `+100%` (`+2 R`) ; `ROI >= +200%` $\rightarrow$ SL à `+150%` (`+3 R`).
  - **Détection & Ré-analyse toutes les 30 min** :
    - Biais H4 : `net/range` sur 8 clôtures 4h (`> +0.35` haussier, `< -0.35` baissier, sinon neutre).
    - Fenêtre M15 : `range (haut - bas) / prix >= 0.4 %` sur 16 clôtures 15m ; direction H4 ou momentum M15 si H4 neutre.
    - POI : `long = after_hi - 0.5 * (after_hi - win_lo)` (`POI < prix`), `short = after_lo + 0.5 * (win_hi - after_lo)` (`POI > prix`).
    - Entrée : au marché si `prix <= POI * 1.0005` (long) / `prix >= POI * 0.9995` (short), sinon limite à `POI * 0.999` (long) / `POI * 1.001` (short), ré-ajustée si écart `> 0.1 %`.
  - **Reporting complet** : RR moyen, Gains/RR et Drawdown max par Mois (`CRYPTO_V4_monthly.csv`), par Année (`CRYPTO_V4_yearly.csv`), par crypto et portefeuille global (`PORTFOLIO_CRYPTO_V4.png`).
- Simulation FTMO (`backtest/sim_challenge.py`) avec superposition des trades : OPR seule / OTE+FVG+OPR / OTE+FVG.
- Visualiseur : `viewer/template.html` -> `viewer/viewer.html` via
  `python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30 --out viewer/viewer.html` (fonctionne sans numpy).
