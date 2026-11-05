# Contexte du projet (reprise de conversation)

Dépôt : `/home/pirateur25/Documents/Trading/trading-bot_antigravity` (Linux). Réponses en français, concises.
Backtests lancés sur GitHub Actions (workflow `.github/workflows/backtest.yml`), résultats via Telegram.

## Stratégies
- **OTE** (`backtest/strategy_ote.py`) et **FVG** (`backtest/strategy_fvg.py`) : optimisées Optuna sur 12 ans (Forex : EURUSD, GBPUSD, USDJPY), params dans `results/best_params_ote.json` / `best_params_fvg.json`.
- **OPR — Opening Price Range** (`backtest/strategy_opr.py`) : stratégie M5 sur les indices US (`NAS100` / `US100`, `USA500` / `US500`, `US30`) :
  - **Zone OPR** : plus haut (`OPR High`) et plus bas (`OPR Low`) mèche comprise de la bougie M15 de **15h30 → 15h45 heure française** (`Europe/Paris`, soit les 3 bougies M5 15h30, 15h35, 15h40).
  - **Filtre de tendance H1 obligatoire (`mm20_50`)** : haussière si `MM20 > MM50` et `clôture H1 en cours > MM50` ; baissière en miroir (`MM20 < MM50` et `clôture < MM50`).
  - **1 seul trade / ordre maximum par jour et par indice**, pris entre 15h45 et **19h30 heure française** :
    1. **Setup Cassure** : clôture M5 au-dessus de l'`OPR High` (haussier) → BUY LIMIT sur `OPR High`, SL sous la bougie M5 de cassure, TP fixe à `RR = 2.5` (et miroir en baissier sous l'`OPR Low`).
    2. **Setup Réintégration** : sortie préalable sous l'`OPR Low` puis clôture M5 à l'intérieur de l'OPR (`OPR Low <= close <= OPR High`) → BUY LIMIT sur `OPR Low`, SL sous la bougie M5 qui réintègre, TP sur `OPR High`, valide uniquement si `RR >= 2.0` (et miroir en baissier).
  - **Gestion de l'ordre et du trade** : ordre limite annulé s'il n'est pas déclenché avant que le prix atteigne `+1 R` ou à `19h30` ; passage à **Break-Even à `+1 R`** (`be_rr = 1.0`).
- Simulation FTMO (`backtest/sim_challenge.py`) avec superposition des trades : OPR seule / OTE+FVG+OPR / OTE+FVG.
- Visualiseur : `viewer/template.html` -> `viewer/viewer.html` via
  `python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30 --out viewer/viewer.html` (fonctionne sans numpy).
