# backtest/data.py — chargement des historiques 5m locaux (data/PAIRE.csv.gz) et construction des unités de temps.
import os
import numpy as np
import pandas as pd

TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
NS_MIN = 60 * 10**9


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def make_frame(df, minutes, base_min=5, min_fill=1.0):
    """Bougies de l'unité demandée (bougies incomplètes écartées : on n'utilise jamais une bougie en cours de formation)."""
    g = df.set_index("time")
    if minutes == base_min:
        f = g[["open", "high", "low", "close"]].copy()
    else:
        f = g.resample(f"{minutes}min").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                           close=("close", "last"), n=("close", "count"))
        full = min(minutes // base_min, float(np.quantile(f["n"], 0.9)))   # nb de bougies 5m d'une bougie « complète » (s'adapte aux indices qui ne cotent pas 24 h)
        f = f[f["n"] >= max(1, int(np.ceil(full * min_fill)))][["open", "high", "low", "close"]]
    open_ns = pd.DatetimeIndex(f.index).as_unit("ns").asi8
    o, h, l, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return {"o": o, "h": h, "l": l, "c": c, "open_ns": open_ns, "close_ns": open_ns + minutes * NS_MIN, "atr": atr}


def neg(f):
    """Monde miroir (cas haussier) : tous les prix sont inversés ; les ATR et les heures ne changent pas."""
    g = dict(f)
    g["o"], g["h"], g["l"], g["c"] = -f["o"], -f["l"], -f["h"], -f["c"]
    return g


class PairData:
    """Toutes les données d'une paire + caches (événements MSS, zones FVG HTF, filtres de tendance)."""

    def __init__(self, symbol, datadir="data"):
        self.symbol = symbol
        path = next(os.path.join(datadir, symbol + e) for e in (".csv.gz", ".csv") if os.path.exists(os.path.join(datadir, symbol + e)))
        df = load_csv(path)
        self.times = df["time"]
        self.base = make_frame(df, 5, 5)
        self.frames = {"5m": self.base}
        for tf in ("15m", "1h", "4h", "1d"):
            self.frames[tf] = make_frame(df, TF_MIN[tf], 5, 1.0 if tf == "15m" else 0.6)
        self.n = len(df)
        self.worlds = {1: self.frames, -1: {k: neg(v) for k, v in self.frames.items()}}
        self.cache = {}
        hours = (self.base["close_ns"] // (60 * NS_MIN)) % 24
        self.hour = hours.astype(np.int16)

    def idx_at(self, ts):
        return int(np.searchsorted(self.base["open_ns"], pd.Timestamp(ts, tz="UTC").value, side="left"))

    def trend_state(self, tf, length):
        """+1 si la dernière clôture TF est au-dessus de l'EMA, -1 si en dessous (par bougie de base, sans regarder le futur)."""
        key = ("trend", tf, length)
        if key not in self.cache:
            f = self.frames[tf]
            ema = pd.Series(f["c"]).ewm(span=length, adjust=False).mean().to_numpy()
            st = np.where(np.arange(len(ema)) < length, 0, np.sign(f["c"] - ema)).astype(np.int8)
            j = np.searchsorted(f["close_ns"], self.base["close_ns"], side="right") - 1
            self.cache[key] = np.where(j >= 0, st[np.clip(j, 0, None)], 0).astype(np.int8)
        return self.cache[key]
