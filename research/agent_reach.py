"""Thin research/skill integration for Agent-Reach access."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def agent_reach_available() -> bool:
    return shutil.which("agent-reach") is not None or _import_agent_reach() is not None


def _import_agent_reach():
    try:
        import agent_reach  # noqa: F401
        return agent_reach
    except Exception:
        return None


def search(query: str, channel: str | None = None) -> dict[str, object]:
    if not agent_reach_available():
        return {"error": "agent_reach_not_installed", "query": query}
    cmd = [sys.executable, "-m", "agent_reach.cli", "search", query]
    if channel:
        cmd += ["--channel", channel]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"query": query, "channel": channel, "stdout": out.stdout, "stderr": out.stderr, "returncode": out.returncode}
    except Exception as exc:
        return {"error": str(exc), "query": query}


def read(url: str) -> dict[str, object]:
    if not agent_reach_available():
        return {"error": "agent_reach_not_installed", "url": url}
    try:
        out = subprocess.run([sys.executable, "-m", "agent_reach.cli", "read", url], capture_output=True, text=True, check=False)
        return {"url": url, "stdout": out.stdout, "stderr": out.stderr, "returncode": out.returncode}
    except Exception as exc:
        return {"error": str(exc), "url": url}


__all__ = ["agent_reach_available", "search", "read"]
