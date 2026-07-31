"""Read-only futures paper-trading dashboard data."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def futures_paper_status() -> dict[str, Any]:
    state = Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))
    model_dir = Path(os.environ.get("FUTURES_MODEL_DIR", str(state / "models" / "futures")))
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(model_dir.glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0):
        manifest = _json(path)
        symbol = manifest.get("symbol")
        if symbol:
            latest[str(symbol)] = {"version": manifest.get("version"), "metrics": manifest.get("metrics", {}), "interval": manifest.get("interval"), "artifact": str(path)}
    from integrations.automatic_backtest import automatic_backtest_status
    backtests = automatic_backtest_status().get("results", [])
    futures_backtest = next((x for x in backtests if isinstance(x, dict) and x.get("module") == "backtests.run_lucid"), None)
    simulation: dict[str, Any] | None = None
    if futures_backtest:
        payload = _json(Path(str(futures_backtest.get("artifact", ""))))
        trades = payload.get("trade_log", []) if isinstance(payload.get("trade_log", []), list) else []
        capital = float(payload.get("initial_capital", 25000.0))
        curve = []
        wins = 0
        for idx, trade in enumerate(trades[-200:]):
            ret = float(trade.get("returns", 0.0))
            capital *= 1.0 + ret
            wins += ret > 0
            curve.append({"index": idx, "capital": capital, "symbol": trade.get("symbol"), "time": trade.get("exit_time")})
        simulation = {**futures_backtest, "metrics": payload.get("metrics", futures_backtest.get("metrics", {})), "regimes": payload.get("regimes", {}), "capital_curve": curve, "simulated_capital": capital, "realized_win_rate": wins / len(trades[-200:]) if trades else None}
    return {
        "assets": latest,
        "asset_count": len(latest),
        "simulation": simulation,
        "paper_only": True,
        "live_trading": False,
        "promotion_allowed": False,
        "note": "Futures models are asset-specific paper candidates; no order execution is enabled.",
    }
