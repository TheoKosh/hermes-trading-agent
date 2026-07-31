#!/usr/bin/env python3
"""Kraken backtest runner — clean, env-hygienic, no secrets."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.kraken_swing import generate_signal, save_backtest, OUTDIR

SYMBOL = "LINKUSD"
INTERVAL = "1h"
DAYS = 90
START = datetime.now(timezone.utc) - timedelta(days=DAYS)


def run() -> None:
    # Use public Kraken OHLC for crypto if available, otherwise Yahoo
    try:
        import requests as _req

        r = _req.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": SYMBOL, "interval": 60},
            timeout=20,
        )
        data = r.json()
        pair = next(k for k in data["result"] if k != "last")
        bars = data["result"][pair]
        closes = [float(b[4]) for b in bars]
        highs = [float(b[2]) for b in bars]
        lows = [float(b[3]) for b in bars]
        vols = [float(b[6]) for b in bars]
        timestamps = [datetime.fromtimestamp(b[0], tz=timezone.utc).isoformat() for b in bars]
    except Exception:
        df = yf.download(f"{SYMBOL}=X", start=START, end=datetime.now(timezone.utc), interval=INTERVAL, progress=False, auto_adjust=True)
        df.index = pd.DatetimeIndex(df.index)
        closes = np.asarray(df["Close"].values).flatten().astype(float).tolist()
        highs = np.asarray(df["High"].values).flatten().astype(float).tolist()
        lows = np.asarray(df["Low"].values).flatten().astype(float).tolist()
        vols = np.asarray(df["Volume"].values).flatten().astype(float).tolist()
        timestamps = [pd.Timestamp(idx).isoformat() for idx in df.index]

    trades: list[dict[str, Any]] = []
    in_pos = False
    pos: dict[str, Any] = {}

    for i in range(50, len(closes)):
        snapshot = {
            "closes": tuple(closes[max(0, i - 49) : i + 1]),
            "highs": tuple(highs[max(0, i - 49) : i + 1]),
            "lows": tuple(lows[max(0, i - 49) : i + 1]),
            "volumes": tuple(vols[max(0, i - 49) : i + 1]),
            "vwap": float(np.nan),
            "timestamp": timestamps[i],
        }
        # Compute VWAP for the window
        window_highs = np.array(highs[max(0, i - 49) : i + 1])
        window_lows = np.array(lows[max(0, i - 49) : i + 1])
        window_closes = np.array(closes[max(0, i - 49) : i + 1])
        window_vols = np.array(vols[max(0, i - 49) : i + 1])
        tp = (window_highs + window_lows + window_closes) / 3.0
        vwap = float(np.cumsum(tp * window_vols)[-1] / max(np.cumsum(window_vols)[-1], 1e-9))
        snapshot["vwap"] = vwap

        if in_pos:
            price = closes[i]
            if pos["direction"] == "buy":
                if price >= pos["take_profit"] or price <= pos["stop_loss"]:
                    ret = (price - pos["entry"]) / pos["entry"]
                    trades.append({
                        "returns": ret, "direction": "buy", "strategy": pos["strategy"],
                        "entry_price": float(pos["entry"]),
                        "exit_price": float(price),
                        "stop_loss": float(pos["stop_loss"]),
                        "take_profit": float(pos["take_profit"]),
                        "entry_time": pos.get("entry_time"),
                        "exit_time": timestamps[i],
                    })
                    in_pos = False
            else:
                if price <= pos["take_profit"] or price >= pos["stop_loss"]:
                    ret = (pos["entry"] - price) / pos["entry"]
                    trades.append({
                        "returns": ret, "direction": "sell", "strategy": pos["strategy"],
                        "entry_price": float(pos["entry"]),
                        "exit_price": float(price),
                        "stop_loss": float(pos["stop_loss"]),
                        "take_profit": float(pos["take_profit"]),
                        "entry_time": pos.get("entry_time"),
                        "exit_time": timestamps[i],
                    })
                    in_pos = False

        if not in_pos:
            sig = generate_signal(type("Snap", (), snapshot)())
            if sig:
                in_pos = True
                pos = {
                    "direction": sig.direction,
                    "entry": closes[i],
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "strategy": sig.strategy,
                    "entry_time": timestamps[i],
                }

    from strategies.base import performance_metrics

    metrics = performance_metrics(trades)
    print(json.dumps({"module": "kraken", "symbol": SYMBOL, "metrics": metrics}, indent=2))
    payload = {
        "metrics": metrics,
        "trade_log": trades,
        "source": "kraken-public",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "start": timestamps[50] if timestamps else None,
        "end": timestamps[-1] if timestamps else None,
        "bars": len(closes),
    }
    path = save_backtest(payload, name="kraken_swing")
    print("Saved:", path)


if __name__ == "__main__":
    run()
