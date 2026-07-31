"""Explicit six-layer reasoning pipeline with full traceability."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PerceptionResult:
    symbol: str
    timestamp: str
    bars: int
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureResult:
    ema9: float
    ema21: float
    ema50: float
    vwap: float
    atr: float
    rsi: float
    price: float


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    confidence: float
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hypothesis:
    direction: str
    score: int
    evidence: tuple[str, ...]
    strategy: str


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    sl_distance: float = 0.0
    tp_distance: float = 0.0


@dataclass(frozen=True)
class DecisionResult:
    action: str  # "buy" | "sell" | "hold"
    direction: str | None
    score: int
    strategy: str
    stop_loss: float | None
    take_profit: float | None
    reason: str


@dataclass(frozen=True)
class ReasoningChain:
    perception: PerceptionResult
    features: FeatureResult
    regime: RegimeResult
    hypotheses: tuple[Hypothesis, ...]
    risk: RiskCheckResult
    decision: DecisionResult
    symbol: str
    timeframe: str


__all__ = [
    "PerceptionResult",
    "FeatureResult",
    "RegimeResult",
    "Hypothesis",
    "RiskCheckResult",
    "DecisionResult",
    "ReasoningChain",
]
