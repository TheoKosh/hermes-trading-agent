"""Council personas with distinct reasoning logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PersonaPosition:
    persona: str
    stance: str  # "support" | "oppose" | "neutral"
    reason: str
    layer: str  # one of the 6 layers
    confidence: float = 0.0


@dataclass(frozen=True)
class CouncilSession:
    trigger: str
    chain_symbol: str
    chain_timeframe: str
    chain_decision: str
    chain_score: int
    positions: tuple[PersonaPosition, ...]
    synthesis: str
    recommendation: str
    target_layer: str
    applied: bool = False
    outcome_after: str | None = None
    outcome_delta: float = 0.0


def _risk_position(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]]) -> PersonaPosition:
    reasons = []
    stance = "neutral"
    confidence = 0.5
    if chain.get("risk", {}).get("passed") is False:
        stance = "oppose"
        reasons.append("risk check failed")
        confidence = 0.95
    if trade and trade.get("outcome") == "loss":
        stance = "oppose"
        reasons.append("prior trade lost")
        confidence = max(confidence, 0.8)
    if not reasons:
        reasons.append("risk parameters within current bounds")
    return PersonaPosition(
        persona="The Risk Officer",
        stance=stance,
        reason="; ".join(reasons),
        layer="risk-check",
        confidence=confidence,
    )


def _quant_position(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]]) -> PersonaPosition:
    sample = max(len(history), 1)
    reasons = []
    stance = "neutral"
    confidence = min(0.3 + sample * 0.02, 0.85)
    if sample < 5:
        stance = "oppose"
        reasons.append(f"sample size too small: {sample}")
        confidence = 0.7
    else:
        reasons.append(f"sample size={sample}")
    if chain.get("decision", {}).get("score", 0) < 60:
        stance = "oppose"
        reasons.append("score below confidence threshold")
        confidence = max(confidence, 0.75)
    if not reasons:
        reasons.append("signal strength meets minimum sample threshold")
    return PersonaPosition(
        persona="The Quant",
        stance=stance,
        reason="; ".join(reasons),
        layer="hypothesis",
        confidence=confidence,
    )


def _contrarian_position(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]]) -> PersonaPosition:
    decision = chain.get("decision", {}).get("action", "hold")
    regime = chain.get("regime", {}).get("regime", "RANGE")
    reasons = []
    stance = "neutral"
    confidence = 0.6
    if decision != "hold":
        stance = "oppose"
        reasons.append(f"counter-case: inverse {decision} in {regime}")
        confidence = 0.7
    if regime == "RANGE":
        stance = "oppose"
        reasons.append("range regimes favor fade setups over directional entries")
        confidence = max(confidence, 0.65)
    if not reasons:
        reasons.append("no clear opposite edge identified")
    return PersonaPosition(
        persona="The Contrarian",
        stance=stance,
        reason="; ".join(reasons),
        layer="hypothesis",
        confidence=confidence,
    )


def _macro_position(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]]) -> PersonaPosition:
    regime = chain.get("regime", {}).get("regime", "RANGE")
    reasons = []
    stance = "neutral"
    confidence = 0.5
    if regime == "RANGE":
        stance = "oppose"
        reasons.append("range regime: reduce directional exposure")
        confidence = 0.75
    if regime in ("TRENDING_UP", "TRENDING_DOWN") and chain.get("features", {}).get("atr", 0) > 0:
        stance = "support"
        reasons.append(f"trending regime supports momentum layer: {regime}")
        confidence = 0.7
    if not reasons:
        reasons.append("regime context insufficient for macro adjustment")
    return PersonaPosition(
        persona="The Macro Strategist",
        stance=stance,
        reason="; ".join(reasons),
        layer="regime",
        confidence=confidence,
    )


def _historian_position(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]]) -> PersonaPosition:
    if not history:
        return PersonaPosition(
            persona="The Historian",
            stance="neutral",
            reason="no prior similar sessions recorded",
            layer="decision",
            confidence=0.4,
        )
    similar = [
        h for h in history
        if isinstance(h, dict) and isinstance(h.get("decision"), dict) and h.get("decision", {}).get("action") == chain.get("decision", {}).get("action")
    ]
    wins = sum(1 for h in similar if isinstance(h, dict) and isinstance(h.get("trade"), dict) and h.get("trade", {}).get("outcome") == "win")
    rate = wins / len(similar) if similar else 0.0
    stance = "support" if rate >= 0.6 else "oppose" if rate < 0.4 else "neutral"
    reasons = [f"historical win rate for similar action={rate:.2f} over {len(similar)} records"]
    return PersonaPosition(
        persona="The Historian",
        stance=stance,
        reason="; ".join(reasons),
        layer="decision",
        confidence=min(max(rate, 0.4), 0.85),
    )


PERSONAS = (
    _risk_position,
    _quant_position,
    _contrarian_position,
    _macro_position,
    _historian_position,
)


def convene(chain: dict[str, Any], trade: dict[str, Any] | None, history: list[dict[str, Any]], trigger: str) -> CouncilSession:
    positions = tuple(func(chain, trade, history) for func in PERSONAS)
    stances = [p.stance for p in positions]
    support = stances.count("support")
    oppose = stances.count("oppose")
    synthesis = f"support={support}, oppose={oppose}, neutral={stances.count('neutral')}"
    if support >= 3:
        recommendation = "proceed with current logic; confidence is consensus-backed"
        target_layer = "hypothesis"
    elif oppose >= 3:
        recommendation = "pause or downgrade confidence until opposition is addressed"
        target_layer = "risk-check"
    else:
        recommendation = "mixed signal; tighten observation and reduce exposure until council converges"
        target_layer = "decision"
    suspicious = support == 5 or oppose == 5
    if suspicious:
        synthesis += "; suspicious consensus—possible shared blind spot"
    return CouncilSession(
        trigger=trigger,
        chain_symbol=chain.get("symbol", ""),
        chain_timeframe=chain.get("timeframe", ""),
        chain_decision=chain.get("decision", {}).get("action", "hold"),
        chain_score=int(chain.get("decision", {}).get("score", 0)),
        positions=positions,
        synthesis=synthesis,
        recommendation=recommendation,
        target_layer=target_layer,
    )


__all__ = [
    "PersonaPosition",
    "CouncilSession",
    "convene",
]
