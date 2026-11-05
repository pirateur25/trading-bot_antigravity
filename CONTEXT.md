# Contexte du projet (reprise de conversation)

Dépôt : `/home/pirateur25/Documents/Trading/trading-bot_antigravity` (Linux). Réponses en français, concises.
Backtests lancés sur GitHub Actions (workflow `.github/workflows/backtest.yml`), résultats via Telegram. Terminal local possible uniquement si le workspace est ouvert sur ce dossier.

## Stratégies
- **OTE** (`backtest/strategy_ote.py`) et **FVG** (`backtest/strategy_fvg.py`) : optimisées Optuna sur 12 ans, params dans `results/best_params_ote.json` / `best_params_fvg.json`.
- **Silver Bullet SB** (`backtest/strategy_sb.py`) : stratégie fixe 5m, sessions heure NY (London 3h-4h, NY AM 10h-11h, NY PM 14h-15h), 1re FVG 5m de la fenêtre [D,F), limit sur le bord du gap, SL sur la bougie centrale, TP RR=2 par défaut (`rr`), ordre retiré à F.
- Récap par session (London / NY AM / NY PM) par paire et global dans `backtest/run_backtest.py`.
- Simulation FTMO (`backtest/sim_challenge.py`) avec superposition des trades, SB seule / OTE+FVG+SB / OTE+FVG.
- Visualiseur : `viewer/template.html` -> `viewer/viewer.html` via
  `python3 backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDJPY,NAS100,USA500 --out viewer/viewer.html` (fonctionne sans numpy).

## Données (`data/`)
- EURUSD, GBPUSD, USDJPY, NAS100, USA500 en `.csv.gz` 5m. Lecture tolérante aux en-têtes (JForex/Dukascopy, timestamps ms) dans `backtest/data_forex.py::load_csv`; renommage auto des fichiers `usatech*`/`USA500*` -> `NAS100`/`USA500` (sauf si le nom cible existe déjà).
- Coûts/levier indices dans `backtest/realism.py` (NAS100, USA500).

## Problème en cours
Nouveau fichier NAS100 (depuis 2015, `usatechidxusd-m5-bid-...`) donne SB NAS100 -35.9% (750t, -0.08R) contre +78.7% (323t) avec l'ancien fichier.
Constats (via les `viewer_trades.json`) :
- Ancien fichier : prix ~1000-1500 (pas du vrai NAS100, ~17x trop bas), bougies bien plus larges, 0 trade NY PM, TP à 3R (donc run avec rr=3 ou ancienne version).
- Nouveau : prix ~17000-21000, TP à 2R, 3 sessions.
- Le premier trade du nouveau run est le 2025-01-02 : il n'a donc PAS utilisé 2015 (vérifier input `start` du workflow, et que le fichier 2015 est bien renommé en `data/NAS100.csv.gz` + commité).
À faire : `ls -la data`, `zcat data/NAS100.csv.gz | head -3 / tail -2`, `git status`; relancer avec `start=2015-01-01`, `params_sb` vide (RR2) puis `rr=3` pour comparer.

