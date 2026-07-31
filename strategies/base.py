"""Shared strategy base types, metrics, and env-backed paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Signal:
    direction: str  # "buy" | "sell"
    score: int
    strategy: str
    stop_loss: float
    take_profit: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketSnapshot:
    closes: Sequence[float]
    highs: Sequence[float]
    lows: Sequence[float]
    volumes: Sequence[float]
    vwap: float
    timestamp: str


BASE_DIR = Path(os.environ.get("BASE_DIR", "/Users/vera/trading-system")).resolve()
AUDIT_DIR = BASE_DIR / "backtests"
STATE_DIR = BASE_DIR / "state"


def performance_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0}
    returns = [float(t.get("returns", 0.0)) for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(returns),
        "win_rate": len(wins) / len(returns),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy": sum(returns) / len(returns),
    }


def kelly_fraction(win_rate: float, win_loss_ratio: float, cap: float = 0.25) -> float:
    if win_loss_ratio <= 0 or not 0 < win_rate < 1:
        return 0.0
    q = 1.0 - win_rate
    return max(0.0, min((win_rate * win_loss_ratio - q) / win_loss_ratio, cap))
