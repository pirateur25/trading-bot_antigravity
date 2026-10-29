"""
backtest/macro_filter.py — Filtre d'annonces macro.

Charge macro_calendar_utc.csv (colonnes: time_utc, currency, impact, event).
Pour un tableau de timestamps UTC (numpy int64 nanosecondes), renvoie un masque booléen
indiquant si le timestamp est dans une fenêtre d'exclusion autour d'une annonce.
"""
import os
import numpy as np
import pandas as pd

_CAL = None
_CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "macro_calendar_utc.csv")


def _load():
    global _CAL
    if _CAL is not None:
        return _CAL
    if not os.path.exists(_CAL_PATH):
        _CAL = np.zeros(0, dtype=np.int64)
        return _CAL
    df = pd.read_csv(_CAL_PATH, parse_dates=["time_utc"])
    if df["time_utc"].dt.tz is None:
        df["time_utc"] = df["time_utc"].dt.tz_localize("UTC")
    else:
        df["time_utc"] = df["time_utc"].dt.tz_convert("UTC")
    _CAL = df["time_utc"].astype(np.int64).to_numpy()
    return _CAL


def macro_block_mask(timestamps_ns: np.ndarray, window_before_min: int = 15, window_after_min: int = 15) -> np.ndarray:
    """
    Retourne un masque True pour chaque timestamp qui est dans une fenêtre d'exclusion autour d'une annonce.
    timestamps_ns : tableau numpy int64 en nanosecondes UTC.
    """
    events_ns = _load()
    if len(events_ns) == 0:
        return np.zeros(len(timestamps_ns), dtype=bool)
    before_ns = window_before_min * 60 * 10**9
    after_ns  = window_after_min  * 60 * 10**9
    mask = np.zeros(len(timestamps_ns), dtype=bool)
    for ev in events_ns:
        mask |= (timestamps_ns >= ev - before_ns) & (timestamps_ns <= ev + after_ns)
    return mask
