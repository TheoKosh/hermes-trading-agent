"""Recursive paper-only retraining protocol with persisted stage state."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from integrations.automatic_backtest import automatic_backtest_once
from integrations.automatic_retrain import automatic_retrain_once
from integrations.ml_improvement_observer import improvement_snapshot
from integrations.ml_learning_review import automatic_review_once

_LOCK = threading.Lock()


def _path() -> Path:
    return Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state")) / "recursive_training_state.json"


def recursive_training_step() -> dict[str, Any]:
    """Advance one candidate cycle; each stage retains its own hard limits."""
    if os.environ.get("HERMES_RECURSIVE_TRAINING", "false").lower() != "true":
        return {"status": "disabled", "paper_only": True, "promotion_allowed": False, "live_mutation": False, "live_trading": False}
    if not _LOCK.acquire(blocking=False):
        return {"status": "busy", "paper_only": True, "promotion_allowed": False, "live_mutation": False, "live_trading": False}
    try:
        training = automatic_retrain_once()
        backtest = automatic_backtest_once()
        review = automatic_review_once()
        observation = improvement_snapshot()
        result = {
            "timestamp": time.time(),
            "status": "complete",
            "stages": {"training": training, "backtest": backtest, "review": review},
            "observation": {"latest_model": observation.get("latest_model"), "metric_deltas": observation.get("metric_deltas"), "quality_gate": observation.get("quality_gate")},
            "paper_only": True,
            "promotion_allowed": False,
            "live_mutation": False,
            "live_trading": False,
        }
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True))
        return result
    finally:
        _LOCK.release()


def recursive_training_status() -> dict[str, Any]:
    path = _path()
    try:
        saved = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    return {
        "enabled": os.environ.get("HERMES_RECURSIVE_TRAINING", "false").lower() == "true",
        "last_cycle": saved,
        "protocol": ["crypto_candidate_train", "futures_candidate_train", "paper_backtest", "AI_error_review", "read_only_quality_observation"],
        "paper_only": True,
        "promotion_allowed": False,
        "live_mutation": False,
        "live_trading": False,
    }
