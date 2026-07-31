"""Bounded AI review of paper-trading mistakes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from integrations.free_claude_code_adapter import completion, status as fcc_status
from integrations.ml_ai_advisor import local_conclusion, _free_model_allowed
from integrations.self_evolution_adapter import record_staged_mutation


LEARNING_SYSTEM_PROMPT = """You review paper-trading mistakes. Use only the supplied verified outcomes and model metrics. Return concise JSON with: mistakes, hypotheses, candidate_adjustments, confidence. Do not provide hidden chain-of-thought. Candidate adjustments are paper-only proposals; never change live code, risk, models, or orders. Require tests, walk-forward validation, positive expectancy, and human approval."""


def _paper_outcomes() -> list[dict[str, Any]]:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    path = state / "trade_memory.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data.get("trades", [])
    except (OSError, json.JSONDecodeError):
        return []


def review_learning() -> dict[str, Any]:
    base = local_conclusion()
    outcomes = _paper_outcomes()[-30:]
    mistakes = [x for x in outcomes if str(x.get("outcome", "")).lower() in {"loss", "failed", "blocked"} or (isinstance(x.get("pnl"), (int, float)) and x["pnl"] < 0)]
    model = os.environ.get("HERMES_ML_AI_MODEL", "openrouter/free")
    result: dict[str, Any] = {
        "status": "paper_only",
        "model": base.get("model"),
        "mistake_count": len(mistakes),
        "ai_requests": 0,
        "tokens": 0,
        "candidate_logged": False,
        "live_mutation": False,
        "live_trading": False,
    }
    if os.environ.get("HERMES_ML_AI_ENABLED", "false").lower() != "true":
        result["ai_status"] = "disabled"
        return result
    if os.environ.get("HERMES_ML_AI_FREE_ONLY", "true").lower() == "true" and not _free_model_allowed(model):
        result["ai_status"] = "blocked_paid_model"
        return result
    if not fcc_status()["enabled"]:
        result["ai_status"] = "blocked_no_configured_free_endpoint"
        return result
    payload = {"model": base.get("model"), "outcomes": outcomes, "known_plan": base.get("plan")}
    try:
        response = completion([
            {"role": "system", "content": LEARNING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ], model=model)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(response, dict) else ""
        result.update({"ai_status": "complete", "ai_requests": 1, "tokens": int(usage.get("total_tokens", 0) or 0), "review": text})
        record = record_staged_mutation(f"paper-learning/review-{int(time.time())}", text, "AI paper-mistake review; candidate only", {"mistake_count": len(mistakes)})
        result["candidate_logged"] = True
        result["mutation_stage"] = record["stage"]
    except Exception as exc:
        result.update({"ai_status": "error", "error": str(exc)})
    return result


def automatic_review_once() -> dict[str, Any]:
    """Run at most once per interval and within a daily free-request cap."""
    state_dir = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    state_path = state_dir / "learning_review_state.json"
    now = time.time()
    interval = max(300, int(os.environ.get("HERMES_ML_AI_AUTO_INTERVAL_SECONDS", "1800")))
    daily_cap = max(1, int(os.environ.get("HERMES_ML_AI_AUTO_DAILY_CAP", "8")))
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    used = int(state.get("requests", 0)) if state.get("day") == day else 0
    if os.environ.get("HERMES_ML_AI_AUTO_REVIEW", "false").lower() != "true":
        return {"status": "disabled", "automatic": True, "requests_today": used, "daily_cap": daily_cap}
    if now - float(state.get("last_run", 0)) < interval:
        return {"status": "not_due", "automatic": True, "requests_today": used, "daily_cap": daily_cap}
    if used >= daily_cap:
        return {"status": "daily_cap_reached", "automatic": True, "requests_today": used, "daily_cap": daily_cap}
    result = review_learning()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"day": day, "requests": used + int(result.get("ai_requests", 0)), "last_run": now}, sort_keys=True))
    result.update({"automatic": True, "requests_today": used + int(result.get("ai_requests", 0)), "daily_cap": daily_cap})
    return result


def automatic_review_status() -> dict[str, Any]:
    state_dir = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    state_path = state_dir / "learning_review_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return {
        "automatic": os.environ.get("HERMES_ML_AI_AUTO_REVIEW", "false").lower() == "true",
        "interval_seconds": max(300, int(os.environ.get("HERMES_ML_AI_AUTO_INTERVAL_SECONDS", "1800"))),
        "daily_cap": max(1, int(os.environ.get("HERMES_ML_AI_AUTO_DAILY_CAP", "8"))),
        "requests_today": int(state.get("requests", 0)) if state.get("day") == day else 0,
        "last_run": state.get("last_run"),
        "free_only": os.environ.get("HERMES_ML_AI_FREE_ONLY", "true").lower() == "true",
        "candidate_only": True,
        "live_mutation": False,
        "live_trading": False,
    }
