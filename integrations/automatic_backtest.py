"""Bounded automatic paper backtests; never promotes or trades candidates."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _paths() -> tuple[Path, Path]:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    output = state / "backtests"
    return state / "automatic_backtest_state.json", output


def _load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _run_one(module: str, output: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["AUDIT_DIR"] = str(output)
    completed = subprocess.run([sys.executable, "-m", module], cwd=str(Path(__file__).resolve().parents[1]), env=env, capture_output=True, text=True, timeout=300)
    result: dict[str, Any] = {"module": module, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    filename = "kraken_swing.json" if module.endswith("run_kraken") else "lucid_momentum.json"
    artifact = output / filename
    if artifact.exists():
        try:
            payload = json.loads(artifact.read_text())
            result["metrics"] = payload.get("metrics", payload) if isinstance(payload, dict) else payload
            result["artifact"] = str(artifact)
        except (OSError, json.JSONDecodeError):
            result["artifact_error"] = "invalid_json"
    return result


def automatic_backtest_once() -> dict[str, Any]:
    state_path, output = _paths()
    now = time.time()
    interval = max(1800, int(os.environ.get("HERMES_BACKTEST_INTERVAL_SECONDS", "21600")))
    daily_cap = max(1, int(os.environ.get("HERMES_BACKTEST_DAILY_CAP", "4")))
    state = _load_state(state_path)
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    used = int(state.get("runs", 0)) if state.get("day") == day else 0
    base = {"automatic": True, "interval_seconds": interval, "daily_cap": daily_cap, "runs_today": used, "paper_only": True, "promoted": False, "live_trading": False}
    if os.environ.get("HERMES_BACKTEST_AUTO", "false").lower() != "true":
        return {**base, "status": "disabled"}
    if now - float(state.get("last_run", 0)) < interval:
        return {**base, "status": "not_due"}
    if used >= daily_cap:
        return {**base, "status": "daily_cap_reached"}
    output.mkdir(parents=True, exist_ok=True)
    results = [_run_one("backtests.run_kraken", output), _run_one("backtests.run_lucid", output)]
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"day": day, "runs": used + 1, "last_run": now, "results": results}, sort_keys=True))
    return {**base, "status": "complete", "runs_today": used + 1, "results": results}


def automatic_backtest_status() -> dict[str, Any]:
    state_path, _ = _paths()
    state = _load_state(state_path)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return {
        "automatic": os.environ.get("HERMES_BACKTEST_AUTO", "false").lower() == "true",
        "interval_seconds": max(1800, int(os.environ.get("HERMES_BACKTEST_INTERVAL_SECONDS", "21600"))),
        "daily_cap": max(1, int(os.environ.get("HERMES_BACKTEST_DAILY_CAP", "4"))),
        "runs_today": int(state.get("runs", 0)) if state.get("day") == day else 0,
        "last_run": state.get("last_run"),
        "paper_only": True,
        "promoted": False,
        "live_trading": False,
        "candidate_gate": "metrics_and_human_review_required",
        "results": state.get("results", []),
    }
