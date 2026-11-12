# backtest/data.py
# Téléchargement de l'historique de bougies.
# Sources essayées dans l'ordre (source="auto") :
#   1) data.binance.vision : fichiers officiels Binance (prix exacts de Binance, sans limite)
#   2) KuCoin  (API publique, par tranches de 1500 bougies)
#   3) Bybit   (API publique, par tranches de 1000 bougies)
# Résultat : DataFrame avec les colonnes time (UTC), open, high, low, close, volume.

import io
import os
import time
import zipfile
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

# Symboles crypto disponibles sur FTMO et Crypto V4 (MoonX / Futures)
FTMO_CRYPTO = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "AVAX": "AVAXUSDT",
    "ZEC": "ZECUSDT",
    "INJ": "INJUSDT",
    "PUMP": "PUMPUSDT",
    "HYPE": "HYPEUSDT",
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "XRPUSD": "XRPUSDT",
    "AVAXUSD": "AVAXUSDT",
    "ZECUSD": "ZECUSDT",
    "INJUSD": "INJUSDT",
    "PUMPUSD": "PUMPUSDT",
    "HYPEUSD": "HYPEUSDT",
    "LTCUSD": "LTCUSDT",
    "ADAUSD": "ADAUSDT",
    "DOGEUSD": "DOGEUSDT",
    "DASHUSD": "DASHUSDT",
    "XMRUSD": "XMRUSDT",
    "NEOUSD": "NEOUSDT",
    "DOTUSD": "DOTUSDT",
}


# Traduit un symbole (ex: BTC, BTCUSD, HYPE, HYPEUSD) en symbole USDT Binance/Bybit
def ftmo_to_binance(ftmo_sym):
    s = ftmo_sym.strip().upper()
    if s in FTMO_CRYPTO:
        return FTMO_CRYPTO[s]
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"

SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
KUCOIN_TYPES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1hour", "4h": "4hour", "1d": "1day"}
BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
COLS = ["time", "open", "high", "low", "close", "volume"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _frame(rows):
    """rows : liste de [ts_ms, open, high, low, close, volume]"""
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _get(url, params=None, retries=3):
    """GET avec quelques essais ; renvoie la réponse, ou None si 404 / erreur réseau."""
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  réseau: {e}")
            time.sleep(1 + k)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code in (429, 418) or r.status_code >= 500:
            time.sleep(2 * (k + 1))
            continue
        return None
    return None


# ---------------------------------------------------------------- Binance (data.binance.vision)

def _parse_vision_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, usecols=range(6))
    ts = pd.to_numeric(df[0], errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()].astype("int64")
    ts = ts.where(ts < 10**14, ts // 1000)  # certains fichiers récents sont en microsecondes
    df[0] = ts
    return df.values.tolist()


def _fetch_binance_vision_base(base, symbol, interval, start_ms, end_ms):
    today = datetime.now(timezone.utc).date()
    first_this_month = today.replace(day=1)
    prev_month_start = (first_this_month - timedelta(days=1)).replace(day=1)
    start_d = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_d = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    rows = []
    cur = start_d.replace(day=1)
    while cur <= end_d:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        got = None
        if cur < first_this_month:  # mois terminé : fichier mensuel
            url = f"{base}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{cur:%Y-%m}.zip"
            r = _get(url)
            if r is not None:
                got = _parse_vision_zip(r.content)
        if got is not None:
            rows.extend(got)
        elif cur >= prev_month_start:  # mois récent : fichiers journaliers (le mensuel peut ne pas être publié)
            day = cur
            while day < nxt and day < today:
                if start_d <= day <= end_d:
                    url = f"{base}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
                    r = _get(url)
                    if r is not None:
                        rows.extend(_parse_vision_zip(r.content))
                day += timedelta(days=1)
        cur = nxt
    return rows


def fetch_binance_futures(symbol, interval, start_ms, end_ms):
    return _fetch_binance_vision_base("https://data.binance.vision/data/futures/um", symbol, interval, start_ms, end_ms)


def fetch_binance_vision(symbol, interval, start_ms, end_ms):
    return _fetch_binance_vision_base("https://data.binance.vision/data/spot", symbol, interval, start_ms, end_ms)


# ---------------------------------------------------------------- KuCoin

def fetch_kucoin(symbol, interval, start_ms, end_ms):
    sym = symbol[:-4] + "-" + symbol[-4:]  # BTCUSDT -> BTC-USDT
    step = SECONDS[interval]
    start_s, cur_end = start_ms // 1000, end_ms // 1000
    rows = []
    while cur_end > start_s:
        cur_start = max(start_s, cur_end - 1500 * step)
        r = _get("https://api.kucoin.com/api/v1/market/candles",
                 {"type": KUCOIN_TYPES[interval], "symbol": sym, "startAt": cur_start, "endAt": cur_end})
        data = (r.json().get("data") if r is not None else None) or []
        for t, o, c, h, l, v, _ in data:  # KuCoin : time, open, close, high, low, volume, turnover
            rows.append([int(t) * 1000, o, h, l, c, v])
        if not data:
            break
        cur_end = cur_start
        time.sleep(0.15)
    return rows


# ---------------------------------------------------------------- Bybit (Linear Futures & Spot)

def _fetch_bybit_cat(category, symbol, interval, start_ms, end_ms):
    step = SECONDS[interval] * 1000
    cur_end = end_ms
    rows = []
    empty_streak = 0
    while cur_end > start_ms:
        cur_start = max(start_ms, cur_end - 1000 * step)
        r = _get("https://api.bybit.com/v5/market/kline",
                 {"category": category, "symbol": symbol, "interval": BYBIT_INTERVALS[interval],
                  "start": cur_start, "end": cur_end, "limit": 1000})
        data = []
        if r is not None:
            data = (r.json().get("result") or {}).get("list") or []
        for row in data:  # Bybit : start, open, high, low, close, volume, turnover
            t, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]
            rows.append([int(t), o, h, l, c, v])
        if not data:
            empty_streak += 1
            if empty_streak >= 3:
                break
        else:
            empty_streak = 0
        cur_end = cur_start
        time.sleep(0.12)
    return rows


def fetch_bybit_linear(symbol, interval, start_ms, end_ms):
    return _fetch_bybit_cat("linear", symbol, interval, start_ms, end_ms)


def fetch_bybit(symbol, interval, start_ms, end_ms):
    return _fetch_bybit_cat("spot", symbol, interval, start_ms, end_ms)


# ---------------------------------------------------------------- Hyperliquid (pour HYPE, PUMP, etc.)

def fetch_hyperliquid(symbol, interval, start_ms, end_ms):
    coin = symbol[:-4] if symbol.endswith("USDT") else (symbol[:-3] if symbol.endswith("USD") else symbol)
    step = SECONDS[interval] * 1000
    cur_start = max(start_ms, end_ms - 5000 * step * 60)
    rows = []
    cur = cur_start
    while cur < end_ms:
        nxt = min(end_ms, cur + 4500 * step)
        try:
            r = requests.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": cur, "endTime": nxt}},
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json() or []
                for item in data:
                    rows.append([int(item["t"]), item["o"], item["h"], item["l"], item["c"], item.get("v", 0.0)])
        except Exception:
            break
        cur = nxt
        time.sleep(0.15)
    return rows


SOURCES = {
    "binance_futures": fetch_binance_futures,
    "binance_vision": fetch_binance_vision,
    "bybit_linear": fetch_bybit_linear,
    "bybit": fetch_bybit,
    "kucoin": fetch_kucoin,
    "hyperliquid": fetch_hyperliquid,
}


def load(symbol, interval, start, end=None, source="auto", use_cache=True):
    """Renvoie (DataFrame, nom_de_la_source). start/end au format 'YYYY-MM-DD' (end exclu, défaut = aujourd'hui)."""
    if interval not in SECONDS:
        raise ValueError(f"Unité de temps inconnue: {interval} (choix: {', '.join(SECONDS)})")
    start_ms = int(_parse_date(start).timestamp() * 1000)
    end_dt = _parse_date(end) if end else datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("La date de fin doit être après la date de début.")

    names = list(SOURCES) if source == "auto" else [source]
    for name in names:
        cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{start}_{end or 'now'}_{name}.csv")
        if use_cache and os.path.exists(cache) and end:
            df = pd.read_csv(cache, parse_dates=["time"])
            print(f"{symbol} {interval}: {len(df)} bougies (cache, {name})")
            return df, name
        print(f"{symbol} {interval}: téléchargement via {name}...")
        try:
            rows = SOURCES[name](symbol, interval, start_ms, end_ms)
        except Exception as e:  # une source en panne ne doit pas bloquer les autres
            print(f"  échec {name}: {e}")
            continue
        if not rows:
            print(f"  aucune donnée via {name}")
            continue
        df = _frame(rows)
        df = df[(df["time"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
                & (df["time"] < pd.Timestamp(end_ms, unit="ms", tz="UTC"))].reset_index(drop=True)
        if df.empty:
            continue
        print(f"  {len(df)} bougies, du {df['time'].iloc[0]} au {df['time'].iloc[-1]}")
        if use_cache and end:
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(cache, index=False)
        return df, name
    raise RuntimeError(f"Impossible de récupérer l'historique de {symbol} {interval} (toutes les sources ont échoué).")


def load_extended(symbol, interval, start, end=None):
    """
    Charge en mémoire (sans rien écrire sur disque) l'historique le plus long possible :
      1) Binance Futures (data.binance.vision/futures/um) depuis `start`
      2) complète tout ce qui précède le 1er historique Futures avec les bougies Binance Spot
      3) si Futures est vide, Spot seul ; si Spot est vide aussi, repli sur `load()` (KuCoin/Bybit/...)
    Renvoie (DataFrame, description_source). start/end au format 'YYYY-MM-DD' (end exclu).
    """
    if interval not in SECONDS:
        raise ValueError(f"Unité de temps inconnue: {interval}")
    start_ms = int(_parse_date(start).timestamp() * 1000)
    end_dt = _parse_date(end) if end else datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("La date de fin doit être après la date de début.")

    def _clip(df):
        return df[(df["time"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC"))
                  & (df["time"] < pd.Timestamp(end_ms, unit="ms", tz="UTC"))].reset_index(drop=True)

    print(f"{symbol} {interval}: téléchargement en ligne Binance Futures depuis {start}...", flush=True)
    fut = None
    try:
        rows = fetch_binance_futures(symbol, interval, start_ms, end_ms)
        if rows:
            fut = _clip(_frame(rows))
            if fut.empty:
                fut = None
    except Exception as e:
        print(f"  échec binance_futures: {e}", flush=True)

    first_fut_ms = int(fut["time"].iloc[0].timestamp() * 1000) if fut is not None else end_ms
    spot = None
    if first_fut_ms - start_ms > 2 * 86400 * 1000:
        print(f"{symbol} {interval}: complément Binance Spot de {start} jusqu'au début des Futures...", flush=True)
        try:
            rows = fetch_binance_vision(symbol, interval, start_ms, first_fut_ms)
            if rows:
                spot = _frame(rows)
                spot = _clip(spot)
                spot = spot[spot["time"] < pd.Timestamp(first_fut_ms, unit="ms", tz="UTC")].reset_index(drop=True)
                if spot.empty:
                    spot = None
        except Exception as e:
            print(f"  échec binance_vision (spot): {e}", flush=True)

    parts = [d for d in (spot, fut) if d is not None]
    if not parts:
        print(f"{symbol} {interval}: Binance indisponible -> repli sur les autres sources", flush=True)
        return load(symbol, interval, start, end, source="auto", use_cache=False)

    df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if spot is not None and fut is not None:
        src = "binance_spot+binance_futures"
    elif spot is not None:
        src = "binance_spot"
    else:
        src = "binance_futures"
    print(f"  {len(df)} bougies ({src}), du {df['time'].iloc[0]} au {df['time'].iloc[-1]}", flush=True)
    return df, src


def load_csv(path):
    """Charge un CSV local (colonnes time, open, high, low, close, volume)."""
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df.sort_values("time").reset_index(drop=True)
              
