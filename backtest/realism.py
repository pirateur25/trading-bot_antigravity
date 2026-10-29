"""
backtest/realism.py — Coûts réalistes FTMO par actif.

Pour les Forex : commission fixe de 5$ par lot standard (100 000 unités) = ~ spread_pct du notionnel.
On modélise tout en % du prix par ordre (aller ET retour = 2x).

Spread FTMO observé (captures d'écran, octobre 2026) :
  Forex : spread en pips (1 pip = 0.0001 pour EUR/USD etc, 0.01 pour JPY)
  Indices : spread en points
  Crypto : spread + commission 0.065% par ordre

fee_per_order_pct = (spread/2 + commission_per_order) / prix_typique * 100
  — ce % est appliqué à l'entrée ET à la sortie dans engine_fvg (fee_pct * 2 au total).
"""

# Coûts par actif : fee_pct = coût appliqué à CHAQUE ordre (entrée ou sortie), en % du prix notionnel.
# Pour le forex, commission FTMO = 5$/lot = 0.005% du notionnel (sur 100k).
# Spread ajouté au-dessus : spread_pips / prix * 50% (demi-spread par ordre).

FTMO_COSTS = {
    # FOREX (commission 0.005% + demi-spread)
    "EURUSD": 0.005 + 0.00003 / 1.113 * 100 / 2,   # ~0.00634%
    "GBPUSD": 0.005 + 0.00010 / 1.330 * 100 / 2,   # ~0.00876%
    "USDCHF": 0.005 + 0.00006 / 0.829 * 100 / 2,   # ~0.00862%
    "USDJPY": 0.005 + 0.005   / 157.9 * 100 / 2,   # ~0.00658%
    "AUDUSD": 0.005 + 0.00006 / 0.703 * 100 / 2,   # ~0.00927%
    "NZDUSD": 0.005 + 0.00004 / 0.568 * 100 / 2,   # ~0.00852%
    "USDCAD": 0.005 + 0.00008 / 1.405 * 100 / 2,   # ~0.00785%
    "EURGBP": 0.005 + 0.00008 / 0.836 * 100 / 2,   # ~0.00979%
    "EURJPY": 0.005 + 0.014   / 177.0 * 100 / 2,   # ~0.00896%
    "GBPJPY": 0.005 + 0.030   / 210.0 * 100 / 2,   # ~0.01214%
    "AUDJPY": 0.005 + 0.016   / 111.0 * 100 / 2,   # ~0.01221%
    # XAUUSD (or) — spread typique ~0.30 USD, commission comme forex
    "XAUUSD": 0.005 + 0.30    / 2650.0 * 100 / 2,  # ~0.01066%
    # XAGUSD (argent) — spread typique ~0.03 USD
    "XAGUSD": 0.005 + 0.03    / 30.0  * 100 / 2,   # ~0.10%  (cher!)
    # INDICES (pas de commission, spread seul)
    "NAS100": 0.0 + 1.45 / 20000.0 * 100 / 2,      # ~0.00363%
    "US30":   0.0 + 2.38 / 40000.0 * 100 / 2,      # ~0.00298%
    "GER40":  0.0 + 3.33 / 18000.0 * 100 / 2,      # ~0.00925%
    # CRYPTO (commission 0.065% par ordre + demi-spread)
    "BTCUSD":  0.065 + 1.00    / 81000.0 * 100 / 2,   # ~0.06562%
    "ETHUSD":  0.065 + 0.60    / 2370.0  * 100 / 2,   # ~0.07766%
    "XRPUSD":  0.065 + 0.0015  / 1.330   * 100 / 2,   # ~0.12141%
    "LTCUSD":  0.065 + 0.15    / 57.7    * 100 / 2,   # ~0.19%
    "ADAUSD":  0.065 + 0.0003  / 0.207   * 100 / 2,   # ~0.13%
    "DOGEUSD": 0.065 + 0.00009 / 0.080   * 100 / 2,   # ~0.12%
    "DASHUSD": 0.065 + 0.03    / 54.0    * 100 / 2,   # ~0.09%
    "XMRUSD":  0.065 + 0.30    / 610.0   * 100 / 2,   # ~0.09%
    "NEOUSD":  0.065 + 0.10    / 2.08    * 100 / 2,   # ~0.31%  (très cher)
    "DOTUSD":  0.065 + 0.0023  / 1.01    * 100 / 2,   # ~0.18%
}

DEFAULT_COST = 0.005  # fallback si actif inconnu


def fee_for(symbol: str) -> float:
    """Coût par ordre en % du notionnel pour un actif FTMO donné."""
    sym = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    # Normalisation : BTCUSDT -> BTCUSD
    if sym.endswith("USDT"):
        sym = sym[:-1]
    return FTMO_COSTS.get(sym, DEFAULT_COST)


def levier_for(symbol: str) -> float:
    """Levier maximum FTMO (Swing) par catégorie d'actif."""
    sym = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    if sym.endswith("USDT"):
        sym = sym[:-1]
    if sym in ("BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "ADAUSD", "DOGEUSD",
               "DASHUSD", "XMRUSD", "NEOUSD", "DOTUSD"):
        return 2.0   # crypto : levier très limité sur FTMO (x2 max typique)
    if sym in ("XAUUSD", "XAGUSD"):
        return 50.0  # métaux précieux
    if sym in ("NAS100", "US30", "GER40"):
        return 30.0  # indices
    return 30.0  # forex compte Swing : levier 1:30 (FTMO Swing)
