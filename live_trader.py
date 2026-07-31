#!/usr/bin/env python3
"""Live trader runner — env-hygienic, paper-first, paper mode is default."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# State lives in STATE_DIR (/app/state on Railway, local state/ by default)
STATE_DIR = os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")
os.makedirs(STATE_DIR, exist_ok=True)

# Paper mode by default. Explicit LIVE=true required for live execution.
LIVE_MODE = os.environ.get("LIVE", "false").lower() == "true"

# Env-hygienic: all secrets from environment only
TRADERSPOST_WEBHOOK_URL = os.environ.get("TRADERSPOST_WEBHOOK_URL", "")
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")


def save_state(name: str, state: dict[str, Any]) -> None:
    path = os.path.join(STATE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(name: str) -> dict[str, Any]:
    path = os.path.join(STATE_DIR, f"{name}.json")
    if os.path.exists(path):
        try:
            return json.loads(open(path).read())
        except Exception:
            pass
    return {"trades": 0, "wins": 0, "last_bar": -1}


def validate_live_env() -> bool:
    missing = []
    if not TRADERSPOST_WEBHOOK_URL:
        missing.append("TRADERSPOST_WEBHOOK_URL")
    if not KRAKEN_API_KEY:
        missing.append("KRAKEN_API_KEY")
    if not KRAKEN_API_SECRET:
        missing.append("KRAKEN_API_SECRET")
    if missing:
        print(f"LIVE mode blocked: missing {', '.join(missing)}")
        return False
    return True


def execute_trade(signal: dict[str, Any]) -> bool:
    """Execute trade. Paper by default, live if LIVE=true and env is valid."""
    if LIVE_MODE:
        if not validate_live_env():
            return False
        # No broker execution adapter is wired here. Fail closed rather than
        # reporting success for a trade that was never submitted.
        print(f"LIVE mode blocked: execution adapter not implemented for {signal}")
        return False
    else:
        # Paper: just record it locally
        state = load_state("paper_log")
        state["trades"] = state.get("trades", 0) + 1
        if signal.get("direction") == "buy" and signal.get("returns", 0) > 0:
            state["wins"] = state.get("wins", 0) + 1
        save_state("paper_log", state)
        print(f"[PAPER] {signal}")
        return True


def run_kraken() -> None:
    from strategies.kraken_swing import generate_signal, SYMBOL, INTERVAL, load_state as kraken_load_state, save_state as kraken_save_state, health_check

    print(health_check())
    state = kraken_load_state()
    print(f"Loaded state: {state}")

    # Fetch bars from Kraken public API
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": SYMBOL, "interval": INTERVAL},
            timeout=20,
        )
        data = r.json()
        pair = next(k for k in data["result"] if k != "last")
        bars = data["result"][pair]
        closes = np.array([float(b[4]) for b in bars], dtype=float)
        highs = np.array([float(b[2]) for b in bars], dtype=float)
        lows = np.array([float(b[3]) for b in bars], dtype=float)
        vols = np.array([float(b[6]) for b in bars], dtype=float)
        ts = [datetime.fromtimestamp(b[0], tz=timezone.utc).isoformat() for b in bars]
    except Exception as e:
        print(f"Kraken fetch failed: {e}")
        return

    i = len(closes) - 1
    snapshot = {
        "closes": tuple(closes[max(0, i - 49) : i + 1]),
        "highs": tuple(highs[max(0, i - 49) : i + 1]),
        "lows": tuple(lows[max(0, i - 49) : i + 1]),
        "volumes": tuple(vols[max(0, i - 49) : i + 1]),
        "vwap": float(np.nan),
        "timestamp": ts[i],
    }
    window_highs = highs[max(0, i - 49) : i + 1]
    window_lows = lows[max(0, i - 49) : i + 1]
    window_closes = closes[max(0, i - 49) : i + 1]
    window_vols = vols[max(0, i - 49) : i + 1]
    tp = (window_highs + window_lows + window_closes) / 3.0
    vwap = float(np.cumsum(tp * window_vols)[-1] / max(np.cumsum(window_vols)[-1], 1e-9))
    snapshot["vwap"] = vwap

    sig = generate_signal(type("Snap", (), snapshot)())
    if sig:
        state["last_signal_bar"] = i
        kraken_save_state(state)
        execute_trade({
            "module": "kraken",
            "direction": sig.direction,
            "score": sig.score,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "meta": sig.meta,
        })
    else:
        print("Kraken: no signal")


def run_lucid() -> None:
    from strategies.futures_momentum import generate_signal, health_check

    print(health_check())
    for sym in ["MNQ=F", "MYM=F", "MES=F"]:
        try:
            df = yf.download(sym, period="5d", interval="15m", progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                continue
            df.index = pd.DatetimeIndex(df.index)
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
        except Exception as e:
            print(f"Lucid fetch failed for {sym}: {e}")
            continue

        sig = generate_signal(candles)
        if sig:
            execute_trade({
                "module": "lucid",
                "symbol": sym,
                "direction": sig.direction,
                "score": sig.score,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
            })
        else:
            print(f"Lucid {sym}: no signal")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python live_trader.py kraken|lucid|both")
        return 1

    target = sys.argv[1].lower()
    if target == "kraken":
        run_kraken()
    elif target == "lucid":
        run_lucid()
    elif target == "both":
        run_kraken()
        run_lucid()
    else:
        print(f"Unknown target: {target}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
