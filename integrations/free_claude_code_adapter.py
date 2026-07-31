"""Optional Free Claude Code OpenAI-compatible router.

FCC is a local/provider proxy (Python >=3.14) and is intentionally not installed
inside the Python 3.11 trading worker. This adapter lets local TradingAgents or
other analysis callers use an explicitly configured FCC endpoint without making
FCC a production dependency or exposing a public proxy.
"""
from __future__ import annotations

import os
from typing import Any

import requests

UPSTREAM_COMMIT = "a8b6dab"
UPSTREAM_REPO = "https://github.com/Alishahryar1/free-claude-code"


def _base_url() -> str:
    return os.environ.get("HERMES_FCC_BASE_URL", "").strip().rstrip("/")


def status() -> dict[str, Any]:
    base = _base_url()
    return {
        "enabled": os.environ.get("HERMES_FCC_ENABLED", "false").lower() == "true" and bool(base),
        "configured": bool(base),
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "base_url": base or None,
        "max_output_tokens": int(os.environ.get("HERMES_FCC_MAX_OUTPUT_TOKENS", "256")),
        "reasoning": "enabled_bounded" if os.environ.get("HERMES_FCC_REASONING", "false").lower() == "true" else "off_by_default",
        "production_installed": False,
        "security": "explicit_endpoint_only",
    }


def compact_messages(messages: list[dict[str, Any]], max_chars: int = 12000) -> list[dict[str, Any]]:
    """Keep system context plus the newest messages under a hard input cap."""
    if not isinstance(messages, list):
        raise ValueError("messages_must_be_list")
    system = [m for m in messages if m.get("role") == "system"][:1]
    rest = [m for m in messages if m.get("role") != "system"]
    selected: list[dict[str, Any]] = []
    total = sum(len(str(m.get("content", ""))) for m in system)
    for message in reversed(rest):
        size = len(str(message.get("content", "")))
        if total + size > max_chars and selected:
            break
        selected.append(message)
        total += size
    return system + list(reversed(selected))


def completion(messages: list[dict[str, Any]], model: str | None = None) -> dict[str, Any]:
    cfg = status()
    if not cfg["enabled"]:
        return {"status": "disabled", "reason": "FCC endpoint not explicitly enabled"}
    payload = {
        "model": model or os.environ.get("HERMES_FCC_MODEL", "openrouter/free"),
        "messages": compact_messages(messages, int(os.environ.get("HERMES_FCC_MAX_INPUT_CHARS", "12000"))),
        "max_tokens": cfg["max_output_tokens"],
        "temperature": 0,
        "stream": False,
    }
    if os.environ.get("HERMES_FCC_REASONING", "false").lower() == "true":
        payload["reasoning_effort"] = os.environ.get("HERMES_FCC_REASONING_EFFORT", "low")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("HERMES_FCC_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(f"{_base_url()}/v1/chat/completions", json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()
