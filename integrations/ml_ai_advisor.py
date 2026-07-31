"""No-token ML learning conclusion and guarded optional AI bridge."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _model_dir() -> Path:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    return Path(os.environ.get("MODEL_DIR", state / "models"))


def _latest_manifest() -> tuple[dict[str, Any] | None, str | None]:
    candidates = sorted(_model_dir().glob("lgbm_meta_*.json"))
    for path in reversed(candidates):
        try:
            return json.loads(path.read_text()), str(path)
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def local_conclusion() -> dict[str, Any]:
    manifest, path = _latest_manifest()
    if not manifest:
        return {
            "status": "blocked",
            "conclusion": "No valid model manifest; learning is paused.",
            "plan": ["collect verified OHLCV", "train a new candidate", "run walk-forward validation"],
            "model": None,
            "token_usage": {"local_ml_calls": 0, "external_ai_calls": 0},
        }
    metrics = manifest.get("metrics", {})
    precision, auc = metrics.get("precision"), metrics.get("auc")
    gate_pass = isinstance(precision, (int, float)) and isinstance(auc, (int, float)) and precision >= 0.55 and auc >= 0.60
    if gate_pass:
        conclusion = "The model clears the statistical gate, but still requires paper review and positive expectancy before any promotion."
        plan = ["run fee-adjusted paper replay", "verify realized expectancy", "compare against the current baseline", "request human approval before promotion"]
    else:
        conclusion = "The model is not reliable enough: its quality gate is blocked, so it must remain paper-only."
        plan = ["keep all actions on hold", "collect more verified data", "retrain with walk-forward validation", "reject candidates with negative expectancy", "re-evaluate regime segments"]
    return {
        "status": "eligible_for_paper_review" if gate_pass else "paper_only",
        "conclusion": conclusion,
        "plan": plan,
        "model": {"version": manifest.get("version"), "precision": precision, "auc": auc, "path": path},
        "token_usage": {"local_ml_calls": 0, "external_ai_calls": 0, "external_ai_enabled": False},
        "safety": {"live_mutation": False, "live_trading": False, "human_approval_required": True},
    }


ADVISOR_SYSTEM_PROMPT = """You are a cautious paper-trading ML reviewer. Be concise, factual, and explicit about uncertainty. Analyze only the supplied verified metrics and outcomes; never invent data. Give conclusions and a bounded learning plan, not hidden chain-of-thought. Never recommend live execution, change risk controls, mutate code/models, or place orders. Require positive expectancy, valid data, walk-forward evidence, and human approval before promotion. Treat financial output as analysis, not personalized financial advice."""


def _free_model_allowed(model: str) -> bool:
    """Allow only clearly free model routes when the free-only gate is on."""
    normalized = model.lower().strip()
    return normalized == "openrouter/free" or normalized.endswith(":free") or normalized.endswith("/free")


def ai_conclusion() -> dict[str, Any]:
    """Optional AI critique; never changes code, models, risk, or orders."""
    result = local_conclusion()
    if os.environ.get("HERMES_ML_AI_ENABLED", "false").lower() != "true":
        result["ai"] = {"status": "disabled", "requests": 0, "tokens": 0}
        return result
    model = os.environ.get("HERMES_ML_AI_MODEL", "openrouter/free")
    if os.environ.get("HERMES_ML_AI_FREE_ONLY", "true").lower() == "true" and not _free_model_allowed(model):
        result["ai"] = {"status": "blocked_paid_model", "requests": 0, "tokens": 0, "model": model}
        return result
    prompt = [{"role": "system", "content": ADVISOR_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps({"model": result.get("model"), "status": result.get("status"), "plan": result.get("plan")}, sort_keys=True)}]
    try:
        from integrations.free_claude_code_adapter import completion, status as fcc_status
        if not fcc_status()["enabled"]:
            result["ai"] = {"status": "blocked_no_configured_free_endpoint", "requests": 0, "tokens": 0}
            return result
        response = completion(prompt, model=os.environ.get("HERMES_ML_AI_MODEL"))
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        result["ai"] = {"status": "complete", "requests": 1, "tokens": int(usage.get("total_tokens", 0) or 0), "response": response.get("choices", [{}])[0].get("message", {}).get("content", "")}
        return result
    except Exception as exc:
        result["ai"] = {"status": "error", "requests": 1, "tokens": 0, "error": str(exc)}
        return result


def status() -> dict[str, Any]:
    result = local_conclusion()
    result["communication"] = {
        "mode": "local_deterministic_conclusion",
        "llm_requests": 0,
        "tokens_consumed": 0,
        "external_ai_opt_in": False,
        "reason": "ML training/inference has no LLM or AI API calls",
    }
    return result
