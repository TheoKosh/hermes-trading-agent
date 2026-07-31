"""Load or hydrate recent OHLC history for ML training across supported symbols/intervals."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import requests

STATE_DIR = os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")
FEATURE_DIR = os.path.join(STATE_DIR, "features")
DEFAULT_PATH = os.path.join(FEATURE_DIR, "linkusd_1h.json")

SYMBOL_INTERVAL_TO_YFINANCE = {
    ("LINKUSD", "1h"): "LINK-USD",
    ("ETHUSD", "1h"): "ETH-USD",
    ("BTCUSD", "1h"): "BTC-USD",
    ("LINKUSD", "1d"): "LINK-USD",
    ("ETHUSD", "1d"): "ETH-USD",
    ("BTCUSD", "1d"): "BTC-USD",
}


def _kraken_ohlc(pair: str = "LINKUSD", interval: int = 60, days: int = 120) -> pd.DataFrame:
    since = None
    rows: list[list[Any]] = []
    for _ in range(20):
        params = {"pair": pair, "interval": interval}
        if since:
            params["since"] = str(since)
        r = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()["result"]
        pair_key = next(k for k in data if k != "last")
        batch = data[pair_key]
        if not batch:
            break
        rows.extend(batch)
        since = int(data["last"])
        if len(batch) < 200:
            break
    if not rows:
        raise ValueError("kraken_ohlc_empty")
    frame = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )
    frame = frame.astype({"timestamp": int, "open": float, "high": float, "low": float, "close": float, "volume": float})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _yfinance_ohlc(symbol: str = "LINK-USD", interval: str = "1h", days: int = 120) -> pd.DataFrame:
    import yfinance as yf
    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("yfinance_empty")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].reset_index().rename(columns={"index": "timestamp", "Datetime": "timestamp"})


def _default_path(symbol: str, interval: str) -> str:
    return os.path.join(FEATURE_DIR, f"{symbol.lower()}_{interval}.json")


def load_or_hydrate(path: str = DEFAULT_PATH, days: int = 120) -> pd.DataFrame:
    return load_dataset(path=path, days=days)


def load_dataset(symbol: str = "LINKUSD", interval: str = "1h", path: str | None = None, days: int = 120) -> pd.DataFrame:
    path = path or _default_path(symbol, interval)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            df = pd.read_json(path, orient="records")
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if len(df) > 200:
                return df.tail(500)
        except Exception:
            pass
    try:
        df = _kraken_ohlc(pair=symbol, interval=60 if interval == "1h" else 1440, days=days)
    except Exception:
        ticker = SYMBOL_INTERVAL_TO_YFINANCE.get((symbol, interval), f"{symbol}-USD")
        interval_arg = "1h" if interval == "1h" else "1d"
        df = _yfinance_ohlc(symbol=ticker, interval=interval_arg, days=days)
    df.to_json(path, orient="records", date_format="iso")
    return df


__all__ = ["load_or_hydrate", "load_dataset", "DEFAULT_PATH"]
