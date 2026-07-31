"""Kraken strategy module: regime-aware, env-hygienic."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import Signal

SYMBOL = "LINKUSD"
INTERVAL = "1h"
OUTDIR = os.environ.get("AUDIT_DIR", "/Users/vera/trading-system/backtests")
STATE_DIR = os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")
KRAKEN_PUBLIC_OHLC = "https://api.kraken.com/0/public/OHLC"


def generate_signal(snapshot: Any) -> Signal | None:
    closes = np.asarray(snapshot.closes, dtype=float)
    highs = np.asarray(snapshot.highs, dtype=float)
    lows = np.asarray(snapshot.lows, dtype=float)
    volumes = np.asarray(snapshot.volumes, dtype=float)

    if len(closes) < 50:
        return None

    df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": volumes})
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

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
    regime = _detect_regime(df["close"].values, float(current["ema21"]), float(current["ema50"]))

    if regime == "TRENDING_UP":
        direction = "buy"
    elif regime == "TRENDING_DOWN":
        direction = "sell"
    else:
        return None

    score = 20
    score += 25 if (direction == "buy" and price > current["vwap"]) or (direction == "sell" and price < current["vwap"]) else 0
    score += 20 if (direction == "buy" and 40 < float(current["rsi"]) < 70) or (direction == "sell" and 30 < float(current["rsi"]) < 60) else 0
    score += 15 if (direction == "buy" and bool(current["ema9"] > current["ema21"])) or (direction == "sell" and bool(current["ema9"] < current["ema21"])) else 0
    score += 10 if abs((price - current["vwap"]) / price) < 0.01 else 0

    if score < 50:
        return None

    atr_val = float(current["atr"])
    sl_pts = max(atr_val * 1.5, price * 0.005)
    tp_pts = sl_pts * 2.0

    return Signal(
        direction=direction,
        score=int(score),
        strategy="kraken-swing",
        stop_loss=price - sl_pts if direction == "buy" else price + sl_pts,
        take_profit=price + tp_pts if direction == "buy" else price - tp_pts,
        meta={
            "regime": regime,
            "vwap": float(current["vwap"]),
            "rsi": float(current["rsi"]),
            "atr": atr_val,
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "timestamp": snapshot.timestamp,
        },
    )


def _detect_regime(closes: np.ndarray, ema21: float, ema50: float) -> str:
    if len(closes) < 50:
        return "RANGE"
    last = float(closes[-1])
    if last > ema21 and ema21 > ema50:
        return "TRENDING_UP"
    if last < ema21 and ema21 < ema50:
        return "TRENDING_DOWN"
    return "RANGE"


def save_backtest(results: dict[str, Any], name: str = "kraken_swing") -> str:
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_state() -> dict[str, Any]:
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, "kraken_state.json")
    if os.path.exists(path):
        try:
            return json.loads(open(path).read())
        except Exception:
            pass
    return {"trades": 0, "wins": 0, "last_signal_bar": -1}


def save_state(state: dict[str, Any]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, "kraken_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def health_check() -> dict[str, Any]:
    return {
        "module": "kraken",
        "market": "crypto",
        "symbol": SYMBOL,
        "timeframe": INTERVAL,
        "regimes_supported": ["TRENDING_UP", "TRENDING_DOWN"],
        "execution": ["paper", "live_env_required"],
        "paper_mode": True,
        "state_dir": STATE_DIR,
    }
