#!/usr/bin/env python3
"""
Wrapper to run data-integrity audits as part of the trading loop.

This module is intended to be imported by auto_trader.py or dashboard/main.py.
It exposes:
  - run_preflight(webhook_url=None): startup checks
  - audit_yahoo_batch(symbol, timeframe, candles): per-batch audit
  - record_decision_outcome(symbol, action, outcome): feedback for audit layer
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from dashboard.audit_layer import evaluate_feed, record_alert


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_preflight(webhook_url: str = "") -> bool:
    from audit_preflight import preflight
    return preflight(webhook_url)


def audit_yahoo_batch(
    symbol: str,
    timeframe: str,
    candles: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    if not candles:
        record_alert("yahoo", "error", f"{symbol} {timeframe}: empty batch rejected")
        return False, {"rejected": True, "check": "empty_batch"}
    prices = [c["c"] for c in candles if c.get("c") is not None]
    volumes = [c["v"] for c in candles if c.get("v") is not None]
    timestamps = [c["t"] for c in candles if c.get("t") is not None]
    last_bar_ts = candles[-1].get("t") if candles else None
    now = time.time()
    ok, primary, meta = evaluate_feed(
        source="yahoo",
        symbol=symbol,
        timeframe=timeframe,
        prices=prices,
        volumes=volumes,
        timestamps=timestamps,
        last_bar_ts=last_bar_ts,
        last_heartbeat_ts=last_bar_ts,
        source_cfg={"name": "yahoo", "mode": "live"},
    )
    if not ok and primary is not None:
        record_alert(
            "yahoo",
            "error",
            f"{symbol} {timeframe}: {primary.check} — {primary.message}",
        )
    state_path = os.environ.get("STATE_FILE", os.path.expanduser("~/auto_trader_state.json"))
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
    except Exception:
        state = {}
    audits = state.setdefault("feed_audits", {})
    key = f"{symbol}|{timeframe}"
    audits[key] = {
        "ts": _utcnow_iso(),
        "ok": ok,
        "check": primary.check if primary else "pass",
        "message": primary.message if primary else "",
        "meta": meta,
    }
    try:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass
    return ok, meta


def record_decision_outcome(symbol: str, action: str, outcome: str) -> None:
    path = os.environ.get("TRADE_LOG", os.path.expanduser("~/auto_trader_trades.json"))
    try:
        with open(path, "r") as f:
            trades = json.load(f)
    except Exception:
        trades = []
    trades.append({
        "time": _utcnow_iso(),
        "symbol": symbol,
        "action": action,
        "outcome": outcome,
        "audit": True,
    })
    try:
        with open(path, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass
