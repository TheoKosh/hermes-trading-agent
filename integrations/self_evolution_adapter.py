"""Safe integration surface for Hermes Agent Self-Evolution.

The upstream project is an offline optimizer that operates *on* Hermes Agent.
It is deliberately not imported into the trading worker and cannot mutate live
skills, prompts, models, or trading logic from this runtime.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "0a929e3"
UPSTREAM_REPO = "https://github.com/NousResearch/hermes-agent-self-evolution"


def status() -> dict[str, Any]:
    repo = Path(os.environ.get("HERMES_AGENT_REPO", "")) if os.environ.get("HERMES_AGENT_REPO") else None
    output = Path(os.environ.get("HERMES_SELF_EVOLUTION_OUTPUT", "/app/state/self_evolution"))
    return {
        "enabled": os.environ.get("HERMES_SELF_EVOLUTION_ENABLED", "false").lower() == "true",
        "approval_required": True,
        "mode": "paper_candidate_evolution",
        "analysis_only": True,
        "paper_mutation": os.environ.get("HERMES_SELF_EVOLUTION_PAPER_MUTATION", "false").lower() == "true",
        "mutation_target": "isolated_candidate_workspace",
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "hermes_repo_configured": bool(repo and repo.exists()),
        "output_dir": str(output),
        "mutation_log": str(output / "mutations.jsonl"),
        "live_mutation": False,
        "live_trading_access": False,
        "guardrails": [
            "full test suite before candidate acceptance",
            "size and structural constraints",
            "holdout evaluation required",
            "human PR review required",
            "no direct production mutation",
        ],
    }


def record_staged_mutation(
    target: str,
    candidate: str,
    reason: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a candidate mutation record; never applies the mutation."""
    if not target or ".." in target or any(ch in target for ch in "\\\n\r"):
        raise ValueError("invalid_target")
    output = Path(os.environ.get("HERMES_SELF_EVOLUTION_OUTPUT", "/app/state/self_evolution"))
    path = output / "mutations.jsonl"
    record = {
        "timestamp": time.time(),
        "target": target,
        "candidate": candidate,
        "reason": reason,
        "metrics": metrics or {},
        "stage": "candidate_pending_human_review",
        "applied": False,
        "live_mutation": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def mutation_log(limit: int = 100) -> dict[str, Any]:
    output = Path(os.environ.get("HERMES_SELF_EVOLUTION_OUTPUT", "/app/state/self_evolution"))
    path = output / "mutations.jsonl"
    records: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 500)):]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"records": records, "count": len(records), "live_mutation": False, "approval_required": True}


def apply_paper_candidate(target: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a candidate into the isolated paper workspace, never production."""
    if not target or ".." in target or any(ch in target for ch in "\\\n\r"):
        raise ValueError("invalid_target")
    if os.environ.get("HERMES_SELF_EVOLUTION_PAPER_MUTATION", "false").lower() != "true":
        raise RuntimeError("paper_mutation_disabled")
    output = Path(os.environ.get("HERMES_SELF_EVOLUTION_OUTPUT", "/app/state/self_evolution"))
    candidate = output / "candidates" / target
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    record = record_staged_mutation(target, str(candidate), "paper candidate applied", metadata)
    record.update({"applied": True, "paper_only": True, "live_mutation": False, "path": str(candidate)})
    return record


def dry_run(skill: str) -> dict[str, Any]:
    if not skill or any(ch in skill for ch in "../\\\n\r"):
        raise ValueError("invalid_skill")
    result = status()
    result.update({"status": "ready" if result["hermes_repo_configured"] else "needs_repo", "skill": skill, "would_run": "evolution.skills.evolve_skill --dry-run"})
    return result
