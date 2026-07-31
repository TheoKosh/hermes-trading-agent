#!/usr/bin/env python3
"""Lucid backtest runner — clean, env-hygienic, no secrets."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.futures_momentum import generate_signal, save_backtest, STATE_DIR, OUTDIR
from strategies.base import performance_metrics

SYMBOLS = ["MNQ=F", "MYM=F", "MES=F"]
INTERVAL = "15m"
DAYS = 45
START = datetime.now(timezone.utc) - timedelta(days=DAYS)
MAX_TRADES_PER_DAY = 10


def run() -> None:
    all_trades: list[dict[str, Any]] = []
    regimes: dict[str, str] = {}
    for sym in SYMBOLS:
        df = yf.download(sym, start=START, end=datetime.now(timezone.utc), interval=INTERVAL, progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            continue
        df.index = pd.DatetimeIndex(df.index)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=1)
        candles: list[dict[str, Any]] = []
        for i in range(len(df)):
            candles.append(
                {
                    "t": pd.Timestamp(df.index[i]).isoformat(),
                    "o": float(df["Open"].iloc[i]),
                    "h": float(df["High"].iloc[i]),
                    "l": float(df["Low"].iloc[i]),
                    "c": float(df["Close"].iloc[i]),
                    "v": float(df["Volume"].iloc[i]),
                }
            )

        if len(candles) >= 21:
            closes = pd.Series([bar["c"] for bar in candles])
            ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
            regimes[sym] = "TRENDING_UP" if ema9 > ema21 else "TRENDING_DOWN" if ema9 < ema21 else "RANGE"
        else:
            regimes[sym] = "INSUFFICIENT_DATA"

        trades: list[dict[str, Any]] = []
        in_pos = False
        pos: dict[str, Any] = {}
        day_trades = 0
        day_pnl = 0.0
        today_key: str | None = None

        for i in range(50, len(candles)):
            bar_time = datetime.fromisoformat(candles[i]["t"])
            day_key = bar_time.strftime("%Y-%m-%d")
            if day_key != today_key:
                today_key = day_key
                day_trades = 0
                day_pnl = 0.0
                if in_pos:
                    price = float(candles[i]["c"])
                    ret = ((price - pos["entry"]) / pos["entry"]) if pos["direction"] == "buy" else ((pos["entry"] - price) / pos["entry"])
                    trades.append({"returns": ret, "direction": pos["direction"], "strategy": pos["strategy"], "symbol": sym, "exit_time": candles[i]["t"], "outcome": "win" if ret > 0 else "loss"})
                    in_pos = False

            if day_trades >= MAX_TRADES_PER_DAY:
                continue
            if day_pnl <= -500.0:
                continue

            window = candles[max(0, i - 199) : i + 1]
            sig = generate_signal(window)
            price = float(candles[i]["c"])

            if not in_pos and sig is not None:
                in_pos = True
                pos = {
                    "direction": sig.direction,
                    "entry": price,
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "strategy": sig.strategy,
                }
                continue

            if in_pos:
                exit_reason = None
                if pos["direction"] == "buy":
                    if price >= pos["take_profit"]:
                        exit_reason = "TP"
                    elif price <= pos["stop_loss"]:
                        exit_reason = "SL"
                else:
                    if price <= pos["take_profit"]:
                        exit_reason = "TP"
                    elif price >= pos["stop_loss"]:
                        exit_reason = "SL"

                if exit_reason:
                    ret = ((price - pos["entry"]) / pos["entry"]) if pos["direction"] == "buy" else ((pos["entry"] - price) / pos["entry"])
                    trades.append({"returns": ret, "direction": pos["direction"], "strategy": pos["strategy"], "symbol": sym, "exit_time": candles[i]["t"], "outcome": "win" if ret > 0 else "loss"})
                    day_trades += 1
                    day_pnl += ret * 25000
                    in_pos = False

        all_trades.extend(trades)

    metrics = performance_metrics(all_trades)
    print(json.dumps({"module": "lucid", "symbols": ",".join(s.replace("=F", "") for s in SYMBOLS), "metrics": metrics}, indent=2))
    payload = {"metrics": metrics, "trade_log": all_trades, "regimes": regimes, "initial_capital": 25000.0, "paper_only": True}
    path = save_backtest(payload, name="lucid_momentum")
    print("Saved:", path)


if __name__ == "__main__":
    run()
