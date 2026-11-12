# backtest/data.py — chargement des historiques 5m locaux (data/PAIRE.csv.gz) et construction des unités de temps.
import os
import numpy as np
import pandas as pd

TF_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
NS_MIN = 60 * 10**9


def load_csv(path):
    import gzip as _gz
    op = (lambda p: _gz.open(p, "rt", encoding="utf-8", errors="replace")) if path.endswith(".gz") else (lambda p: open(p, "r", encoding="utf-8", errors="replace"))
    with op(path) as f:
        first_line = f.readline().strip()
    sep = ";" if first_line.count(";") > first_line.count(",") else ("\t" if first_line.count("\t") > first_line.count(",") else ",")
    df = pd.read_csv(path, sep=sep)

    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("time", "gmt time", "local time", "datetime", "date", "timestamp", "open_time", "time (utc)", "time_utc"):
            col_map[c] = "time"
        elif cl in ("open", "o", "open_price", "askopen", "bidopen"):
            col_map[c] = "open"
        elif cl in ("high", "h", "high_price", "askhigh", "bidhigh"):
            col_map[c] = "high"
        elif cl in ("low", "l", "low_price", "asklow", "bidlow"):
            col_map[c] = "low"
        elif cl in ("close", "c", "close_price", "askclose", "bidclose"):
            col_map[c] = "close"
        elif cl in ("volume", "vol", "v", "tick_volume"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    if "time" not in df.columns:
        first_col = str(df.columns[0]).strip()
        if any(ch.isdigit() for ch in first_col) and len(df.columns) >= 5:
            df = pd.read_csv(path, sep=sep, header=None)
            base_cols = ["time", "open", "high", "low", "close"] + (["volume"] if len(df.columns) >= 6 else [])
            df.columns = base_cols + [f"col_{i}" for i in range(len(base_cols), len(df.columns))]
        else:
            df = df.rename(columns={df.columns[0]: "time"})

    # Conversion des colonnes OHLC si elles n'étaient pas encore nommées en minuscules
    if not all(k in df.columns for k in ("open", "high", "low", "close")) and len(df.columns) >= 5:
        rename_pos = {df.columns[1]: "open", df.columns[2]: "high", df.columns[3]: "low", df.columns[4]: "close"}
        df = df.rename(columns=rename_pos)

    if pd.api.types.is_numeric_dtype(df["time"]):
        v0 = float(df["time"].dropna().iloc[0])
        unit = "ns" if v0 > 1e14 else ("ms" if v0 > 1e11 else "s")
        df["time"] = pd.to_datetime(df["time"], unit=unit, utc=True)
    else:
        s_col = df["time"].astype(str).str.strip()
        s0 = s_col.iloc[0] if len(s_col) else ""
        if " GMT" in s0:
            s_col = s_col.str.replace(r"\s*GMT([+-]\d{4})?", "", regex=True).str.strip()
        dayfirst = len(s0) >= 10 and s0[2] in (".", "/") and s0[5] in (".", "/")
        df["time"] = pd.to_datetime(s_col, dayfirst=dayfirst, utc=True, format="mixed")

    for k in ("open", "high", "low", "close"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
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


def resolve_data_path(symbol, datadir="data"):
    """Trouve data/<SYMBOL>.csv.gz (et renomme automatiquement les exports bruts type USA500IDXUSD_5m_...csv.gz)."""
    sym = symbol.upper()
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(datadir, sym + ext)
        if os.path.exists(p):
            return p
    if os.path.isdir(datadir):
        prefixes = [sym]
        if sym in ("USA500", "US500"):
            prefixes = ["USA500", "US500", "SPX500", "SP500"]
        elif sym in ("NAS100", "US100"):
            prefixes = ["NAS100", "USATECH", "USTEC", "US100"]
        elif sym in ("US30", "DJ30"):
            prefixes = ["US30", "USA30", "DJ30", "WS30"]
        for fname in sorted(os.listdir(datadir)):
            up = fname.upper()
            if any(up.startswith(pref) for pref in prefixes) and (up.endswith(".CSV.GZ") or up.endswith(".CSV")):
                src = os.path.join(datadir, fname)
                ext = ".csv.gz" if up.endswith(".CSV.GZ") else ".csv"
                dst = os.path.join(datadir, sym + ext)
                try:
                    os.rename(src, dst)
                    print(f"[data] Fichier renommé automatiquement : {fname} -> {sym + ext}")
                    return dst
                except OSError:
                    return src
    raise FileNotFoundError(f"Historique introuvable pour {symbol} dans {datadir}/")


class PairData:
    """Toutes les données d'une paire + caches (événements MSS, zones FVG HTF, filtres de tendance)."""

    def __init__(self, symbol, datadir="data", df=None):
        self.symbol = symbol
        if df is None:
            path = resolve_data_path(symbol, datadir)
            df = load_csv(path)
        else:
            df = df.sort_values("time").reset_index(drop=True)
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
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return int(np.searchsorted(self.base["open_ns"], t.value, side="left"))

    def trend_state(self, tf, length):
        """+1 si la dernière clôture TF est au-dessus de l'EMA, -1 si en dessous (par bougie de base, sans regarder le futur)."""
        key = ("trend", tf, length)
        if key not in self.cache:
            f = self.frames[tf]
            ema = pd.Series(f["c"]).ewm(span=length, adjust=False).mean().to_numpy()
            warmup = min(10, max(2, length // 10))
            st = np.where(np.arange(len(ema)) < warmup, 0, np.sign(f["c"] - ema)).astype(np.int8)
            j = np.searchsorted(f["close_ns"], self.base["close_ns"], side="right") - 1
            self.cache[key] = np.where(j >= 0, st[np.clip(j, 0, None)], 0).astype(np.int8)
        return self.cache[key]
