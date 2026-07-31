#!/usr/bin/env python3
"""
Continuous data audit layer for the Hermes trading agent.
Evaluates every incoming data batch against:
  - freshness
  - liveness / heartbeat
  - source authenticity
  - sanity bounds
  - gap detection
  - cross-source consistency
and updates persistent state that the dashboard can read.

Design rule: FAIL CLOSED. If data fails any check, the perception layer
must not pass it downstream. Default action is to halt new decisions on
the affected instrument and immediately alert.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
LIVE_FLAG_FILE = os.environ.get("LIVE_FLAG_FILE", os.path.expanduser("~/auto_trader_LIVE_FLAG.json"))
TRADE_LOG = os.environ.get("TRADE_LOG", os.path.expanduser("~/auto_trader_trades.json"))
DASH_AUDIT_LOG = STATE_DIR / "dash_audit.jsonl"
DASH_AUDIT_STATE = STATE_DIR / "dash_audit_state.json"

# Hard thresholds per feed type
FRESHNESS_LIMITS = {
    "tick": 10,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}
SANITY_PRICE_WINDOW = {
    "YM=F": (0.0, 99999.0),
    "NQ=F": (0.0, 99999.0),
}


class AuditDecision:
    PASS = "PASS"
    REJECTED = "REJECTED"
    SANDBOX = "SANDBOX"
    STALE = "STALE"


class FeedAuditResult:
    __slots__ = (
        "source", "symbol", "timeframe", "status", "check", "message",
        "ts", "last_bar_ts", "last_heartbeat_ts", "meta",
    )

    def __init__(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        status: str,
        check: str,
        message: str = "",
        ts: float | None = None,
        last_bar_ts: float | None = None,
        last_heartbeat_ts: float | None = None,
        meta: dict | None = None,
    ):
        self.source = source
        self.symbol = symbol
        self.timeframe = timeframe
        self.status = status
        self.check = check
        self.message = message
        self.ts = ts or time.time()
        self.last_bar_ts = last_bar_ts
        self.last_heartbeat_ts = last_heartbeat_ts
        self.meta = meta or {}


def _now_ts() -> float:
    return time.time()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_cest() -> str:
    return (datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def _append_audit_log(row: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(DASH_AUDIT_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _load_audit_state() -> dict:
    try:
        if DASH_AUDIT_STATE.exists():
            return json.loads(DASH_AUDIT_STATE.read_text())
    except Exception:
        pass
    return {
        "sources": {},
        "rejections": [],
        "alerts": [],
        "last_audit_ts": 0.0,
        "uptime_start": _now_ts(),
    }


def _save_audit_state(state: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        DASH_AUDIT_STATE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _reject(
    source: str,
    check: str,
    message: str,
    symbol: str = "",
    timeframe: str = "",
    last_bar_ts: float | None = None,
    last_heartbeat_ts: float | None = None,
    meta: dict | None = None,
) -> FeedAuditResult:
    result = FeedAuditResult(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        status=AuditDecision.REJECTED,
        check=check,
        message=message,
        ts=_now_ts() if '_now_ts' in globals() else time.time(),
        last_bar_ts=last_bar_ts,
        last_heartbeat_ts=last_heartbeat_ts,
        meta=meta,
    )
    # Force populate ts if globals lookup failed due to import ordering
    if result.ts is None:
        result.ts = time.time()
    try:
        state = _load_audit_state()
        src = state["sources"].setdefault(source, {"rejections": 0, "last_pass_ts": 0.0, "last_status": AuditDecision.PASS})
        src["rejections"] = int(src.get("rejections", 0)) + 1
        src["last_status"] = AuditDecision.REJECTED
        src["last_reject_ts"] = result.ts
        entry = {
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "check": check,
            "message": message,
            "ts": result.ts,
            "meta": meta or {},
        }
        state["rejections"].append(entry)
        if len(state["rejections"]) > 1000:
            state["rejections"] = state["rejections"][-1000:]
        alert = {
            "level": "error",
            "source": source,
            "check": check,
            "message": message,
            "ts": _utcnow_iso(),
        }
        state["alerts"].append(alert)
        if len(state["alerts"]) > 200:
            state["alerts"] = state["alerts"][-200:]
        _save_audit_state(state)
        _append_audit_log(entry)
    except Exception:
        pass
    return result


def _pass(
    source: str,
    check: str,
    symbol: str = "",
    timeframe: str = "",
    last_bar_ts: float | None = None,
    last_heartbeat_ts: float | None = None,
    meta: dict | None = None,
) -> FeedAuditResult:
    result = FeedAuditResult(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        status=AuditDecision.PASS,
        check=check,
        ts=time.time(),
        last_bar_ts=last_bar_ts,
        last_heartbeat_ts=last_heartbeat_ts,
        meta=meta,
    )
    try:
        state = _load_audit_state()
        src = state["sources"].setdefault(source, {"rejections": 0, "last_pass_ts": 0.0, "last_status": AuditDecision.PASS})
        src["last_pass_ts"] = result.ts
        src["last_status"] = AuditDecision.PASS
        _save_audit_state(state)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_source_authenticity(source_cfg: dict | None) -> FeedAuditResult | None:
    if source_cfg is None:
        return _reject("unknown", "source_authenticity", "Missing source config", meta={"reason": "no cfg"})
    mode = source_cfg.get("mode", "unknown")
    if mode not in {"live", "production"}:
        return _reject(
            source_cfg.get("name", "unknown"),
            "source_authenticity",
            f"Source authenticity failed: mode={mode}",
            meta={"mode": mode},
        )
    return _pass(source_cfg.get("name", "unknown"), "source_authenticity")


def check_freshness(
    source: str,
    symbol: str,
    timeframe: str,
    last_bar_ts: float,
    now: float | None = None,
) -> FeedAuditResult:
    now = now or time.time()
    limit = FRESHNESS_LIMITS.get(timeframe, 300)
    age = now - last_bar_ts
    if age < 0:
        return _reject(
            source, "freshness", f"Bar timestamp is in the future by {abs(age):.1f}s",
            symbol=symbol, timeframe=timeframe, last_bar_ts=last_bar_ts,
            meta={"age_s": age, "limit_s": limit},
        )
    if age > limit:
        return _reject(
            source, "freshness", f"Bar stale by {age:.1f}s > {limit}s",
            symbol=symbol, timeframe=timeframe, last_bar_ts=last_bar_ts,
            meta={"age_s": age, "limit_s": limit},
        )
    return _pass(source, "freshness", symbol=symbol, timeframe=timeframe, last_bar_ts=last_bar_ts)


def check_liveness(
    source: str,
    symbol: str,
    timeframe: str,
    last_heartbeat_ts: float,
    now: float | None = None,
) -> FeedAuditResult:
    now = now or time.time()
    if last_heartbeat_ts <= 0:
        return _reject(
            source, "liveness", "No heartbeat recorded",
            symbol=symbol, timeframe=timeframe, last_heartbeat_ts=0.0,
            meta={},
        )
    age = now - last_heartbeat_ts
    limit = max(FRESHNESS_LIMITS.get(timeframe, 300), 120)
    if age > limit:
        return _reject(
            source, "liveness", f"Heartbeat stale by {age:.1f}s > {limit}s",
            symbol=symbol, timeframe=timeframe, last_heartbeat_ts=last_heartbeat_ts,
            meta={"age_s": age, "limit_s": limit},
        )
    return _pass(source, "liveness", symbol=symbol, timeframe=timeframe, last_heartbeat_ts=last_heartbeat_ts)


def check_sanity_bounds(
    source: str,
    symbol: str,
    timeframe: str,
    prices: list[float],
    volumes: list[float],
) -> FeedAuditResult | None:
    if not prices:
        return _reject(source, "sanity_bounds", "Empty price series", symbol=symbol, timeframe=timeframe)
    price = prices[-1]
    if not isinstance(price, (int, float)) or not math.isfinite(float(price)) or price <= 0:
        return _reject(source, "sanity_bounds", f"Negative/non-finite price: {price}", symbol=symbol, timeframe=timeframe)
    if len(volumes) >= 20 and sum(volumes[-20:]) == 0:
        return _reject(source, "sanity_bounds", "Zero volume across last 20 bars", symbol=symbol, timeframe=timeframe)
    if symbol in SANITY_PRICE_WINDOW:
        lo, hi = SANITY_PRICE_WINDOW[symbol]
        if not (lo <= price <= hi):
            return _reject(source, "sanity_bounds", f"Price {price} outside window [{lo}, {hi}]", symbol=symbol, timeframe=timeframe)
    return _pass(source, "sanity_bounds", symbol=symbol, timeframe=timeframe, meta={"last_price": price})


def check_gaps(
    source: str,
    symbol: str,
    timeframe: str,
    timestamps: list[float],
    max_gap_mult: int = 2,
) -> FeedAuditResult | None:
    if len(timestamps) < 10:
        return None
    expected_gap = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(timeframe, 300)
    gaps = []
    for i in range(1, min(len(timestamps), 50)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > expected_gap * max_gap_mult:
            gaps.append(dt)
    if gaps:
        return _reject(
            source, "gap_detection", f"Detected {len(gaps)} gap(s) up to {max(gaps):.0f}s",
            symbol=symbol, timeframe=timeframe,
            meta={"gaps": gaps[:5], "expected_gap_s": expected_gap},
        )
    return _pass(source, "gap_detection", symbol=symbol, timeframe=timeframe)


def check_cross_source_consistency(
    source_a: str,
    source_b: str,
    symbol: str,
    timeframe: str,
    price_a: float,
    price_b: float,
    tolerance: float = 0.015,
) -> FeedAuditResult | None:
    if price_a <= 0 or price_b <= 0:
        return _reject(
            source_a, "cross_source", "Cross-source zero price",
            symbol=symbol, timeframe=timeframe,
            meta={"price_a": price_a, "price_b": price_b},
        )
    drift = abs(price_a - price_b) / max(price_a, 1e-9)
    if drift > tolerance:
        return _reject(
            source_a, "cross_source", f"Cross-source drift {drift:.2%} > {tolerance:.2%}",
            symbol=symbol, timeframe=timeframe,
            meta={"price_a": price_a, "price_b": price_b, "drift_pct": drift},
        )
    return _pass(source_a, "cross_source", symbol=symbol, timeframe=timeframe, meta={"drift_pct": drift})


# ---------------------------------------------------------------------------
# Aggregate + decision gate
# ---------------------------------------------------------------------------

def evaluate_feed(
    source: str,
    symbol: str,
    timeframe: str,
    *,
    prices: list[float] | None = None,
    volumes: list[float] | None = None,
    timestamps: list[float] | None = None,
    last_bar_ts: float | None = None,
    last_heartbeat_ts: float | None = None,
    source_cfg: dict | None = None,
    peer_source: tuple[str, float] | None = None,
) -> tuple[bool, FeedAuditResult | None, dict]:
    checks = {}
    failures = []

    # 1. Source authenticity
    r_auth = check_source_authenticity(source_cfg)
    if r_auth:
        checks["source_authenticity"] = r_auth.status
        if r_auth.status == AuditDecision.REJECTED:
            failures.append(r_auth)

    # 2. Freshness
    if last_bar_ts is not None:
        r = check_freshness(source, symbol, timeframe, last_bar_ts)
        checks["freshness"] = r.status
        if r.status == AuditDecision.REJECTED:
            failures.append(r)

    # 3. Liveness
    if last_heartbeat_ts is not None:
        r = check_liveness(source, symbol, timeframe, last_heartbeat_ts)
        checks["liveness"] = r.status
        if r.status == AuditDecision.REJECTED:
            failures.append(r)

    # 4. Sanity bounds
    if prices is not None and volumes is not None:
        r_sanity = check_sanity_bounds(source, symbol, timeframe, prices, volumes)
        if r_sanity:
            checks["sanity_bounds"] = r_sanity.status
            if r_sanity.status == AuditDecision.REJECTED:
                failures.append(r_sanity)

    # 5. Gap detection
    if timestamps is not None:
        r_gap = check_gaps(source, symbol, timeframe, timestamps)
        if r_gap:
            checks["gap_detection"] = r_gap.status
            if r_gap.status == AuditDecision.REJECTED:
                failures.append(r_gap)

    # 6. Cross-source consistency
    if peer_source is not None and prices is not None and len(prices) > 0:
        peer_name, peer_price = peer_source
        r_peer = check_cross_source_consistency(source, peer_name, symbol, timeframe, prices[-1], peer_price)
        if r_peer:
            checks["cross_source"] = r_peer.status
            if r_peer.status == AuditDecision.REJECTED:
                failures.append(r_peer)

    if failures:
        primary = failures[0]
        return False, primary, {"checks": checks, "failures": [f.message for f in failures]}
    return True, None, {"checks": checks, "failures": []}


def record_alert(source: str, level: str, message: str) -> None:
    try:
        state = _load_audit_state()
        state["alerts"].append(
            {"level": level, "source": source, "message": message, "ts": _utcnow_iso()}
        )
        if len(state["alerts"]) > 200:
            state["alerts"] = state["alerts"][-200:]
        _save_audit_state(state)
    except Exception:
        pass
