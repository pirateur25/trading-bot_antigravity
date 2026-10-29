# 🤖 Trading Bot — OTE + FVG HTF + Simulation FTMO

Bot de trading algorithmique backtesté pour la validation d'un compte prop firm (FTMO). Deux stratégies complémentaires, presque décorrélées, combinées pour maximiser les chances de validation.

## 📊 Résultats de référence (TRAIN 2021–2023, TEST 2024–2025)

| Stratégie | Actifs | Espérance | Trades/an | DD max | Test 2025 |
|-----------|--------|-----------|-----------|--------|-----------|
| OTE (A) | EURUSD,USDJPY,XAUUSD | +0.14 R/trade | ~80 | -10% | +0.16 R |
| FVG (B) | EURUSD,GBPUSD,USDJPY,XAUUSD | +0.30 R/trade | ~100 | -13% | +0.23 R |
| **Combiné** | 4–5 paires | ~+0.22 R | ~160 | -9% | ✅ |

**Simulation FTMO 10k$ (départs glissants, pire cas intra-bougie) :**
- Risque FVG 0.75% + OTE 0.5% → **~95% réussite, 0% échec, durée médiane ~190 jours**

> ⚠️ Ces résultats sont issus de données sur lesquelles les stratégies ont été partiellement réglées. Fais 2–3 mois de démo avant de payer un challenge.

---

## 🗂️ Structure du dépôt

```
trading-bot/
├── .github/workflows/
│   ├── backtest.yml           # Backtest OTE + FVG → Telegram
│   ├── optimize_fvg.yml       # Optimisation Optuna FVG (bug mkdir fixé)
│   ├── optimize_ote.yml       # Optimisation Optuna OTE
│   └── fetch_crypto.yml       # Téléchargement auto données crypto
├── backtest/
│   ├── strategy_ote.py        # Stratégie A : OTE (ZigZag + Fibonacci + MSS/Bulle)
│   ├── strategy_fvg.py        # Stratégie B : FVG HTF + MSS (moteur C)
│   ├── engine_ote.py          # Moteur simulation OTE (Python)
│   ├── engine_fvg.py          # Moteur simulation FVG (C via ctypes)
│   ├── kernels.c              # Noyau C (MSS, FVG, simulation)
│   ├── native.py              # Wrapper ctypes → kernels.c
│   ├── zones.py               # FVG zones HTF (H1/H4/Daily)
│   ├── data_forex.py          # Chargement données forex locales (.csv.gz)
│   ├── data_crypto.py         # Téléchargement crypto (Binance→KuCoin→Bybit)
│   ├── realism.py             # Coûts FTMO réels par actif (spread + commission)
│   ├── macro_filter.py        # Filtre annonces macro (macro_calendar_utc.csv)
│   ├── optimize_fvg.py        # Optimiseur Optuna FVG
│   ├── optimize_ote.py        # Optimiseur Optuna OTE
│   ├── sim_challenge.py       # Simulation challenge FTMO (1 phase & 2 phases)
│   └── run_backtest.py        # Point d'entrée principal
├── data/                      # Historiques 5m (.csv.gz) — forex + crypto
├── results/
│   ├── best_params_fvg.json   # Meilleurs paramètres FVG
│   └── best_params_ote.json   # Meilleurs paramètres OTE
├── macro_calendar_utc.csv     # Calendrier événements high/medium 2021–2025
└── requirements.txt
```

---

## 🚀 Démarrage rapide

### 1. Configurer les secrets GitHub
Dans Settings → Secrets and variables → Actions :
- `BOT_TOKEN` : token de ton bot Telegram
- `CHAT_ID` : ton chat ID Telegram

### 2. Placer les données forex
Les fichiers `.csv.gz` (EURUSD, GBPUSD, USDCHF, USDJPY, XAUUSD, etc.) doivent être dans `data/`. Format : colonnes `time,open,high,low,close` en UTC, bougies 5m.

### 3. Lancer un backtest
GitHub → Actions → **Backtest OTE + FVG** → Run workflow.

Options principales :
- `symbols` : paires forex à tester
- `crypto` : cryptos FTMO (ex: `BTCUSD,ETHUSD`) — données téléchargées auto
- `mode` : `both` / `ote_only` / `fvg_only`
- `sim_challenge` : `oui` pour simuler le challenge FTMO
- `ftmo_mode` : `2phases` (classic) ou `1phase` (stellar)

### 4. Optimiser les paramètres
- **FVG** : Actions → **Optimisation FVG HTF + MSS (Optuna)** → Run workflow
- **OTE** : Actions → **Optimisation OTE ZigZag + MSS (Optuna)** → Run workflow

Les résultats sont :
1. Sauvegardés dans `results_opt/` (artifact GitHub)
2. Envoyés sur Telegram (`best_params_fvg.json` / `best_params_ote.json`)

### 5. Télécharger les données crypto
Actions → **Téléchargement données crypto** → Run workflow (spécifie les cryptos).

---

## 📐 Les deux stratégies

### Stratégie A — OTE (ZigZag + Zone Fibonacci + MSS/Bulle)
1. **Tendance H4** (optionnelle) : loi de Dow via ZigZag ATR ou EMA
2. **Retracement** : jambe baissière H4/H1 identifiée par ZigZag causal
3. **Zone OTE** : entre 61.8% et 78.6% de la jambe (paramétrable)
4. **Confirmation** en 5m/15m/1h : MSS (cassure de structure) ou Bulle Bollinger
5. **SL** : plus haut local + buffer ATR | **TP** : niveau Fibonacci de la jambe

### Stratégie B — FVG HTF + MSS (moteur C)
1. **Zones** : FVG (Fair Value Gaps) baissiers en H1/H4/Daily, non mitigés
2. **MSS** : cassure de structure baissière en 5m/15m dans une zone FVG
3. **Entrée** : ordre limite sur le premier FVG de la jambe MSS
4. **SL** : au-dessus de H2 du MSS | **TP** : RR fixe ou prochain FVG HTF

---

## 💰 Coûts FTMO modélisés (`realism.py`)

| Actif | Fee/ordre (aprox.) |
|-------|-------------------|
| EURUSD | 0.0063% |
| USDJPY | 0.0066% |
| XAUUSD | 0.0107% |
| BTCUSD | 0.0656% |
| ETHUSD | 0.0777% |
| XRPUSD | 0.1214% |

---

## ⚠️ Pièges à éviter (déjà corrigés)

- ✅ Risque > 1R à cause des frais → taille calculée avec coûts inclus
- ✅ Optimisation sur la période de test → TRAIN 2021–2023 / TEST 2024+
- ✅ Bug `tee: results_opt/log.txt: No such file or directory` → `mkdir -p results_opt` avant
- ✅ Config trop peu de trades (14–24/an) → `min_trades` imposé dans Optuna
- ✅ Binance bloqué sur GitHub → fallback KuCoin puis Bybit
- ✅ Datetimes naïfs mixés avec UTC → tout en UTC

---

## 📨 Rapport Telegram

À chaque backtest, tu reçois sur Telegram :
- 🚀 Message de démarrage
- 📈 Résumé par paire (rendement, DD, espérance, trades, par année, par raison)
- 📊 Courbe d'équité (PNG)
- 📄 CSV détaillé des trades (pour visualisation future)
- 🎯 Simulation FTMO (% réussite, durée médiane, plusieurs niveaux de risque)

### Colonnes du CSV de trades (FVG)
`entry_time, exit_time, signal_time, side, entry, exit, sl, tp, reason, pnl, r, bars, conf_tf, mss_h2, mss_l2, h1_time, h2_time, l2_time, zone_form, fvg_lo, fvg_hi, fvg_t1, fvg_t2, mss_h1, zone_lo, zone_hi`

### Colonnes du CSV de trades (OTE)
`entry_time, exit_time, side, entry, exit, sl, tp, reason, r, pnl, setup, ...`

---

## 🔬 Protocole de validation

- **TRAIN** : 2021–2022 (h1) + 2023 (h2) — score = min(score_h1, score_h2)
- **TEST** : 2024–2025 — lu **une seule fois**, à la fin
- **Score** : moyenne sur paires de `clip(return / max(|dd|, 2%), -3, 3)`
- **Minimum trades** : 15 (FVG) / 10 (OTE) par moitié, sinon score = -3
- **Reprise Optuna** : si le job plante, il repart du checkpoint sqlite automatiquement

---

## 📝 Notes importantes

- Les données historiques 5m forex sont à placer dans `data/` (format JForex)
- Les cryptos disponibles sur FTMO : BTC, ETH, XRP, LTC, ADA, DOGE, DASH, XMR, NEO, DOT
- Levier Swing FTMO : 1:30 forex/or, 1:2 crypto
- Swaps non modélisés (compte Swing recommandé pour éviter le problème du week-end)


---

## 🗺️ Visualiseur de trades (comme celui de Claude)

1. **Une seule fois** (sur ton PC, après avoir mis les données dans `data/`) :
   `python backtest/viewer_export.py build --symbols EURUSD,GBPUSD,USDCHF,USDJPY,XAUUSD,BTCUSD --out viewer/viewer.html`
   -> `viewer/viewer.html` contient déjà toutes les bougies 5m (~2,5 Mo par paire). Ajoute les paires voulues (cryptos incluses) puis garde ce fichier.
2. À chaque backtest, Telegram envoie **`viewer_trades.json`** (seulement les trades OTE et FVG avec tous leurs niveaux).
   Ouvre `viewer.html` > bouton « Charger trades » (ou glisse le fichier). Les trades s'accumulent si tu charges plusieurs fichiers.

## 🎛️ Optimiseurs : leviers
Chaque résumé Telegram liste les leviers **optimisés** et **figés**. Par défaut : zone OTE figée à 0,618–0,786, OTE `side` limité à short/both, FVG `side=both`.
Workflows : champs `freeze` (`k=v;k2=v2`) et `unfreeze` (`ote_lo,ote_hi`) pour changer ça. Les réglages de référence sont testés en premier et comparés à la fin (train **et** test).
