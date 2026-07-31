"""Cached live market view for paper futures positions; no broker execution."""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf

from strategies.futures_momentum import SYMBOLS, generate_signal
from integrations.forward_validation import record_paper_signal

_CACHE: dict[str, Any] = {"updated_at": 0.0, "payload": None}
_LOCK = threading.Lock()


def _regime(candles: list[dict[str, Any]]) -> str:
    if len(candles) < 21:
        return "INSUFFICIENT_DATA"
    close = pd.Series([x["c"] for x in candles])
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    return "TRENDING_UP" if ema9 > ema21 else "TRENDING_DOWN" if ema9 < ema21 else "RANGE"


def _candles(symbol: str) -> list[dict[str, Any]]:
    start = datetime.now(timezone.utc) - timedelta(days=5)
    raw = yf.download(symbol, start=start, end=datetime.now(timezone.utc), interval="15m", progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        return []
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(symbol, axis=1, level=1)
    return [{"t": pd.Timestamp(idx).isoformat(), "o": float(row["Open"]), "h": float(row["High"]), "l": float(row["Low"]), "c": float(row["Close"]), "v": float(row["Volume"])} for idx, row in raw.iterrows()]


def futures_live_paper_positions() -> dict[str, Any]:
    now = time.time()
    ttl = max(30, int(os.environ.get("HERMES_FUTURES_LIVE_CACHE_SECONDS", "60")))
    with _LOCK:
        if _CACHE["payload"] and now - float(_CACHE["updated_at"]) < ttl:
            return _CACHE["payload"]
        positions = []
        for symbol in SYMBOLS:
            try:
                candles = _candles(symbol)
                signal = generate_signal(candles[-200:]) if candles else None
                paper_position = signal.direction.upper() if signal else "FLAT"
                event = record_paper_signal(symbol, paper_position, float(candles[-1]["c"]) if candles else None, candles[-1]["t"] if candles else None)
                positions.append({"symbol": symbol, "price": float(candles[-1]["c"]) if candles else None, "regime": _regime(candles), "paper_position": paper_position, "score": int(signal.score) if signal else 0, "updated_at": candles[-1]["t"] if candles else None, "closed_event": event, "error": None if candles else "no_market_data"})
            except Exception as exc:
                positions.append({"symbol": symbol, "price": None, "regime": "UNKNOWN", "paper_position": "FLAT", "score": 0, "updated_at": None, "error": str(exc)})
        payload = {"positions": positions, "broker_positions": False, "paper_only": True, "live_trading": False, "cache_seconds": ttl, "updated_at": now}
        _CACHE.update({"updated_at": now, "payload": payload})
        return payload
