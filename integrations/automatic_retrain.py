"""Bounded automatic retraining for paper-only ML candidates."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _state_path() -> Path:
    return Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")) / "automatic_training_state.json"


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def automatic_retrain_once() -> dict[str, Any]:
    path = _state_path()
    now = time.time()
    interval = max(1800, int(os.environ.get("HERMES_ML_TRAIN_INTERVAL_SECONDS", "21600")))
    daily_cap = max(1, int(os.environ.get("HERMES_ML_TRAIN_DAILY_CAP", "4")))
    state = _load(path)
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    used = int(state.get("runs", 0)) if state.get("day") == day else 0
    base = {"automatic": True, "interval_seconds": interval, "daily_cap": daily_cap, "runs_today": used, "paper_only": True, "promotion_allowed": False, "live_mutation": False, "live_trading": False}
    if os.environ.get("HERMES_ML_AUTOTRAIN", "false").lower() != "true":
        return {**base, "status": "disabled"}
    if now - float(state.get("last_run", 0)) < interval:
        return {**base, "status": "not_due"}
    if used >= daily_cap:
        return {**base, "status": "daily_cap_reached"}
    env = os.environ.copy()
    state_dir = Path(env.get("STATE_DIR", "/Users/vera/trading-system/state"))
    audit_dir = state_dir / "backtests"
    audit_dir.mkdir(parents=True, exist_ok=True)
    env["AUDIT_DIR"] = str(audit_dir)
    results = []
    for module in ("ml.train_kraken_meta", "ml.train_futures_meta"):
        completed = subprocess.run([sys.executable, "-m", module], cwd=str(Path(__file__).resolve().parents[1]), env=env, capture_output=True, text=True, timeout=600)
        results.append({"module": module, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]})
    returncode = max(item["returncode"] for item in results)
    result = {"returncode": returncode, "runs": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": day, "runs": used + 1, "last_run": now, "result": result}, sort_keys=True))
    return {**base, "status": "complete" if returncode == 0 else "error", "runs_today": used + 1, "result": result}


def automatic_retrain_status() -> dict[str, Any]:
    state = _load(_state_path())
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return {
        "automatic": os.environ.get("HERMES_ML_AUTOTRAIN", "false").lower() == "true",
        "interval_seconds": max(1800, int(os.environ.get("HERMES_ML_TRAIN_INTERVAL_SECONDS", "21600"))),
        "daily_cap": max(1, int(os.environ.get("HERMES_ML_TRAIN_DAILY_CAP", "4"))),
        "runs_today": int(state.get("runs", 0)) if state.get("day") == day else 0,
        "last_run": state.get("last_run"),
        "last_result": state.get("result"),
        "paper_only": True,
        "promotion_allowed": False,
        "live_mutation": False,
        "live_trading": False,
    }
