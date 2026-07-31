"""Read-only observability for ML improvement and regression tracking."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def improvement_snapshot() -> dict[str, Any]:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    model_dir = Path(os.environ.get("MODEL_DIR", str(state / "models")))
    manifests: list[dict[str, Any]] = []
    for path in sorted(model_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        data = _read_json(path)
        if data and isinstance(data.get("metrics"), dict):
            manifests.append({"version": data.get("version", path.stem), "metrics": data["metrics"], "symbol": data.get("symbol"), "interval": data.get("interval"), "asset_class": data.get("asset_class", "crypto"), "artifact": str(path)})
    latest = manifests[-1] if manifests else None
    previous = manifests[-2] if len(manifests) > 1 else None
    deltas: dict[str, float] = {}
    if latest and previous and latest.get("symbol") == previous.get("symbol"):
        for key, value in latest["metrics"].items():
            old = previous["metrics"].get(key)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                deltas[key] = float(value - old)
    backtest_state = _read_json(state / "automatic_backtest_state.json") or {}
    mutation_path = state / "self_evolution" / "mutations.jsonl"
    mutation_count = len(mutation_path.read_text().splitlines()) if mutation_path.exists() else 0
    review_state = _read_json(state / "learning_review_state.json") or {}
    backtests = backtest_state.get("results", [])
    expectancies = [r.get("metrics", {}).get("expectancy") for r in backtests if isinstance(r, dict) and isinstance(r.get("metrics"), dict) and isinstance(r["metrics"].get("expectancy"), (int, float))]
    latest_auc = latest["metrics"].get("auc") if latest else None
    gate = bool(isinstance(latest_auc, (int, float)) and latest_auc >= 0.6 and expectancies and all(x > 0 for x in expectancies))
    return {
        "latest_model": latest,
        "previous_model": previous,
        "metric_deltas": deltas,
        "model_count": len(manifests),
        "model_timeline": manifests[-20:],
        "backtest_runs": backtest_state.get("runs", 0),
        "backtest_results": backtests,
        "ai_review_requests": review_state.get("requests", 0),
        "mutation_candidates": mutation_count,
        "quality_gate": "PASS" if gate else "BLOCK",
        "promotion_allowed": False,
        "paper_only": True,
        "live_mutation": False,
        "live_trading": False,
    }
