"""Optional LiteLLM-backed gateway with local cost guard + response cache.

Default is OFF. Set HERMES_LLM_GATEWAY_ENABLED=true to route calls through LiteLLM.
This module is isolated from live trading paths and safe to remove for rollback.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
CACHE_PATH = Path(os.environ.get("HERMES_LLM_CACHE_PATH", STATE_DIR / "llm_cache.json"))
ENABLED = os.environ.get("HERMES_LLM_GATEWAY_ENABLED", "false").lower() == "true"
CACHE_TTL_SECONDS = int(os.environ.get("HERMES_LLM_CACHE_TTL", "3600"))
MONTHLY_BUDGET_USD = float(os.environ.get("HERMES_LLM_MONTHLY_BUDGET_USD", "0.00"))

# Conservative default pricing. Override at call-site via estimated_cost_usd if needed.
MODEL_PRICE_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.00500, "output": 0.01500},
    "claude-3-5-haiku": {"input": 0.00080, "output": 0.00400},
    "claude-3-5-sonnet": {"input": 0.00300, "output": 0.01500},
}


class LLMGatewayDisabled(RuntimeError):
    pass


class LLMBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class GatewayStats:
    calls: int
    cache_hits: int
    estimated_spend_usd: float
    budget_usd: float


def _load_cache() -> dict[str, Any]:
    try:
        return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {"entries": {}, "stats": {}}
    except Exception:
        return {"entries": {}, "stats": {}}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _cache_key(model: str, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    filtered = {k: v for k, v in kwargs.items() if k not in {"stream", "api_key"}}
    raw = json.dumps({"model": model, "messages": messages, "kwargs": filtered}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def estimate_cost_usd(model: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> float:
    p = MODEL_PRICE_PER_1K.get(model, MODEL_PRICE_PER_1K["gpt-4o-mini"])
    return (prompt_tokens / 1000.0) * p["input"] + (completion_tokens / 1000.0) * p["output"]


def stats() -> GatewayStats:
    cache = _load_cache()
    s = cache.setdefault("stats", {})
    return GatewayStats(
        calls=int(s.get("calls", 0)),
        cache_hits=int(s.get("cache_hits", 0)),
        estimated_spend_usd=round(float(s.get("estimated_spend_usd", 0.0)), 6),
        budget_usd=MONTHLY_BUDGET_USD,
    )


def completion(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """Cached, budget-gated LiteLLM completion.

    Requires env HERMES_LLM_GATEWAY_ENABLED=true. If budget is 0, paid calls are blocked
    unless cache already has a fresh response.
    """
    key = _cache_key(model, messages, kwargs)
    cache = _load_cache()
    entries = cache.setdefault("entries", {})
    hit = entries.get(key)
    now = time.time()
    if hit and now - float(hit.get("ts", 0)) <= CACHE_TTL_SECONDS:
        cache.setdefault("stats", {})["cache_hits"] = int(cache["stats"].get("cache_hits", 0)) + 1
        _save_cache(cache)
        return hit["response"]

    if not ENABLED:
        raise LLMGatewayDisabled("LiteLLM gateway disabled; set HERMES_LLM_GATEWAY_ENABLED=true")

    expected = float(kwargs.pop("estimated_cost_usd", 0.0))
    spent = float(cache.setdefault("stats", {}).get("estimated_spend_usd", 0.0))
    if MONTHLY_BUDGET_USD <= 0 or spent + expected > MONTHLY_BUDGET_USD:
        raise LLMBudgetExceeded(f"LLM budget blocked: ${spent + expected:.6f} > ${MONTHLY_BUDGET_USD:.6f}")

    from litellm import completion as litellm_completion  # optional dependency; imported only when enabled

    resp = litellm_completion(model=model, messages=messages, **kwargs)
    usage = getattr(resp, "usage", None) or {}
    actual = estimate_cost_usd(
        model,
        int(getattr(usage, "prompt_tokens", usage.get("prompt_tokens", 0)) or 0),
        int(getattr(usage, "completion_tokens", usage.get("completion_tokens", 0)) or 0),
    ) or expected

    cache["stats"]["calls"] = int(cache["stats"].get("calls", 0)) + 1
    cache["stats"]["estimated_spend_usd"] = round(spent + actual, 8)
    entries[key] = {"ts": now, "response": resp}
    _save_cache(cache)
    return resp


__all__ = ["completion", "estimate_cost_usd", "stats", "LLMGatewayDisabled", "LLMBudgetExceeded"]
