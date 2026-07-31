"""Persistent state directory helper for Railway/local."""
from __future__ import annotations

import json
import os
from pathlib import Path

STATE_DIR = Path(
    os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")
)


def _ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(name: str, default):
    p = STATE_DIR / f"{name}.json"
    if not p.exists():
        return default
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(name: str, value) -> None:
    _ensure(STATE_DIR / f"{name}.json")
    p = STATE_DIR / f"{name}.json"
    with open(p, "w") as f:
        json.dump(value, f, indent=2)


def append_jsonl(name: str, row) -> None:
    _ensure(STATE_DIR / f"{name}.jsonl")
    p = STATE_DIR / f"{name}.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")
