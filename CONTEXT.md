# Contexte du projet (reprise de conversation)

Dépôt : `/home/pirateur25/Documents/Trading/trading-bot_antigravity` (Linux). Réponses en français, concises.
Backtests lancés sur GitHub Actions (workflow `.github/workflows/backtest.yml`), résultats via Telegram.

## Stratégies
- **OTE** (`backtest/strategy_ote.py`) et **FVG** (`backtest/strategy_fvg.py`) : optimisées Optuna sur 12 ans (Forex : EURUSD, GBPUSD, USDJPY), params dans `results/best_params_ote.json` / `best_params_fvg.json`.
- **OPR — Opening Price Range** (`backtest/strategy_opr.py`) : stratégie M5 sur les indices US (`NAS100` / `US100`, `USA500` / `US500`, `US30`) :
  - **Zone OPR** : plus haut (`OPR High`) et plus bas (`OPR Low`) mèche comprise de la bougie M15 de **15h30 → 15h45 heure française** (`Europe/Paris`, soit les 3 bougies M5 15h30, 15h35, 15h40).
  - **Filtre de tendance H1 obligatoire (`mm20_50`)** : haussière si `MM20 > MM50` et `clôture H1 en cours > MM50` ; baissière en miroir (`MM20 < MM50` et `clôture < MM50`).
  - **1 seul trade / ordre maximum par jour et par indice**, pris entre 15h45 et **19h30 heure française** :
    1. **Setup Cassure** : clôture M5 au-dessus de l'`OPR High` (haussier) → BUY LIMIT sur `OPR High` (et miroir en baissier sous l'`OPR Low`).
    2. **Setup Réintégration** : sortie préalable sous l'`OPR Low` puis clôture M5 à l'intérieur de l'OPR (`OPR Low <= close <= OPR High`) → BUY LIMIT sur `OPR Low`, TP sur `OPR High`, valide uniquement si `RR >= 2.5` (et miroir en baissier).
  - **Placement du SL selon l'indice de volatilité associé (`VXNCLS-US100`, `VIXCLS-US500`, `VXDCLS-US30`)** :
    - Si `indice > 20` : SL sous la mèche de la bougie M5 de signal moins l'`ATR(14) H1` (et `+ ATR(14) H1` en short).
    - Si `15 <= indice <= 20` : distance SL = `max(inv, plancher_actif)` où `inv` = distance entrée ↔ mèche bougie signal, et plancher = **80 pts** (`US30`), **70 pts** (`NAS100`/`US100`), **50 pts** (`USA500`/`US500`).
    - Si `indice < 15` : SL classique sous/au-dessus de la mèche de la bougie M5 de signal.
  - **Filtre d'invalidation par résistance / support avant `+1 R`** : trade annulé s'il existe une résistance non mitigée entre l'entrée et `+1 R` parmi :
    - *Previous Daily High / Low* (`PDH` / `PDL`),
    - *Haut / Bas de session précédente* (`00h00 -> 15h30` heure FR : Asie + Londres),
    - *FVG H1* opposée non mitigée.
  - **Take Profit (TP)** : `2.5 R` minimal, ou étendu sur la première résistance non mitigée située au-delà de `2.5 R` (sur le setup Cassure).
  - **Gestion de l'ordre et du trade** : ordre limite annulé s'il n'est pas déclenché avant que le prix atteigne `+1 R` ou à `19h30` ; passage à **Break-Even à `+1 R`** (`be_rr = 1.0`).
- Simulation FTMO (`backtest/sim_challenge.py`) avec superposition des trades : OPR seule / OTE+FVG+OPR / OTE+FVG.
- Visualiseur : `viewer/template.html` -> `viewer/viewer.html` via
  `python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500,US30 --out viewer/viewer.html` (fonctionne sans numpy).
