"""Multi-asset, multi-feature paper scanner.

Signals are analysis-only. A model trained on LINKUSD is never silently applied
to another asset; each asset must earn its own validated model first.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests

SUPPORTED_ASSETS = ("LINKUSD", "BTCUSD", "ETHUSD")
KRAKEN_PAIRS = {"LINKUSD": "LINKUSD", "BTCUSD": "XBTUSD", "ETHUSD": "ETHUSD"}


def _fetch(asset: str) -> pd.DataFrame:
    pair = KRAKEN_PAIRS[asset]
    response = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": 60},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError("kraken_error:" + ",".join(data["error"]))
    key = next(k for k in data["result"] if k != "last")
    rows = data["result"][key]
    if len(rows) < 60:
        raise ValueError("insufficient_bars")
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vwap_raw", "volume", "count"])
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = frame[col].astype(float)
    return frame


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    typical = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = (typical * out["volume"]).cumsum() / out["volume"].cumsum().replace(0, 1e-12)
    tr = pd.concat([(out["high"] - out["low"]), (out["high"] - out["close"].shift()).abs(), (out["low"] - out["close"].shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, 1e-12))).fillna(50)
    out["return_1"] = out["close"].pct_change()
    out["return_6"] = out["close"].pct_change(6)
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean().replace(0, 1e-12)
    return out.dropna().reset_index(drop=True)


def scan_asset(asset: str) -> dict[str, Any]:
    asset = asset.upper()
    if asset not in SUPPORTED_ASSETS:
        raise ValueError("unsupported_asset")
    features = _features(_fetch(asset))
    row = features.iloc[-1]
    price, ema21, ema50 = float(row.close), float(row.ema21), float(row.ema50)
    if price > ema21 > ema50:
        regime, direction = "TRENDING_UP", "buy"
    elif price < ema21 < ema50:
        regime, direction = "TRENDING_DOWN", "sell"
    else:
        regime, direction = "RANGE", "hold"
    confirmations = {
        "vwap_alignment": bool((direction == "buy" and price > row.vwap) or (direction == "sell" and price < row.vwap)),
        "rsi_neutral": bool(30 < float(row.rsi) < 70),
        "ema_acceleration": bool((direction == "buy" and row.ema9 > row.ema21) or (direction == "sell" and row.ema9 < row.ema21)),
        "volume_confirmed": bool(float(row.volume_ratio) >= 1.0),
    }
    score = int(20 + 20 * sum(bool(v) for v in confirmations)) if direction != "hold" else 0
    return {
        "asset": asset,
        "bars": len(features),
        "datapoints": ["open", "high", "low", "close", "volume", "ema9", "ema21", "ema50", "vwap", "atr", "rsi", "return_1", "return_6", "volume_ratio"],
        "features": {k: round(float(row[k]), 8) for k in ("close", "ema9", "ema21", "ema50", "vwap", "atr", "rsi", "return_1", "return_6", "volume_ratio")},
        "regime": regime,
        "direction": direction if score >= 50 else "hold",
        "score": score,
        "confirmations": confirmations,
        "model": {"status": "not_applied", "reason": "asset-specific validation required"},
        "mode": "paper_only",
    }


def scan_universe() -> dict[str, Any]:
    results, errors = [], []
    for asset in SUPPORTED_ASSETS:
        try:
            results.append(scan_asset(asset))
        except Exception as exc:
            errors.append({"asset": asset, "error": str(exc)})
    return {"assets": list(SUPPORTED_ASSETS), "results": results, "errors": errors, "mode": "paper_only", "live_trading": False}
