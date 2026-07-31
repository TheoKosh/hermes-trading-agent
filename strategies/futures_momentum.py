"""Lucid futures strategy module: env-hygienic, session-aware."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import Signal

SYMBOLS = ("MNQ=F", "MYM=F", "MES=F")
INTERVAL = "15m"
OUTDIR = os.environ.get("AUDIT_DIR", "/Users/vera/trading-system/backtests")
STATE_DIR = os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")
MARKET_OPEN_HOUR_CEST = 0
MAX_TRADES_PER_DAY = 10
FLATTEN_HOUR_CEST = 22


def generate_signal(candles: list[dict[str, Any]]) -> Signal | None:
    if not candles or len(candles) < 50:
        return None

    df = pd.DataFrame(candles)
    df = df.rename(columns={"h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["high", "low", "close", "volume"]].astype(float)

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap"] = pv.cumsum() / df["volume"].cumsum().replace(0, 1e-12)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    df["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    df["rsi"] = 100.0 - 100.0 / (1.0 + rs)
    df["rsi"] = df["rsi"].fillna(50.0)

    current = df.iloc[-1]
    if current.isna().any():
        return None

    price = float(current["close"])
    vwap = float(current["vwap"])
    ema9 = float(current["ema9"])
    ema21 = float(current["ema21"])
    rsi = float(current["rsi"])
    atr = float(current["atr"])

    direction = None
    score = 0
    if ema9 > ema21:
        direction = "buy"
        score = 20
    elif ema9 < ema21:
        direction = "sell"
        score = 20

    if direction == "buy":
        score += 15 if 40 < rsi < 70 else 0
        score += 10 if rsi < 30 else 0
        score += 15 if price > vwap else 0
        score += 10 if abs((price - vwap) / price) < 0.01 else 0
    elif direction == "sell":
        score += 15 if 30 < rsi < 60 else 0
        score += 10 if rsi > 70 else 0
        score += 15 if price < vwap else 0
        score += 10 if abs((price - vwap) / price) < 0.01 else 0

    if score >= 45 and direction is not None:
        score += 5

    if atr / price > 0.001:
        score += 5

    if score < 50 or direction is None:
        return None

    sl_pts = max(atr * 0.5, 5.0)
    tp_pts = sl_pts * 2.0

    return Signal(
        direction=direction,
        score=int(score),
        strategy="futures-momentum",
        stop_loss=price - sl_pts if direction == "buy" else price + sl_pts,
        take_profit=price + tp_pts if direction == "buy" else price - tp_pts,
        meta={
            "symbol": "session",
            "timeframe": INTERVAL,
            "vwap": vwap,
            "rsi": rsi,
            "atr": atr,
        },
    )


def health_check() -> dict[str, Any]:
    return {
        "module": "lucid",
        "market": "futures",
        "symbols": list(SYMBOLS),
        "timeframe": INTERVAL,
        "execution": ["paper", "live_env_required"],
        "paper_mode": True,
        "state_dir": STATE_DIR,
        "flatten_rule": f"{FLATTEN_HOUR_CEST}:45 CEST",
    }


def save_backtest(results: dict[str, Any], name: str = "lucid_momentum") -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
