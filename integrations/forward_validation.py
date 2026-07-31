"""Persisted forward paper outcomes and promotion evidence; never executes orders."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    return Path(os.environ.get("STATE_DIR", "/Users/vera/trading-system/state"))


def _paths() -> tuple[Path, Path]:
    state = _state_dir()
    return state / "forward_paper_open.json", state / "forward_paper_ledger.jsonl"


def _load_open(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_open(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))


def record_paper_signal(symbol: str, direction: str, price: float | None, timestamp: str | None) -> dict[str, Any] | None:
    """Track a paper position from a live market signal, closing on flip/flat."""
    if not symbol or direction not in {"BUY", "SELL", "FLAT"} or not isinstance(price, (int, float)) or not math.isfinite(float(price)) or float(price) <= 0:
        return None
    open_path, ledger_path = _paths()
    positions = _load_open(open_path)
    current = positions.get(symbol)
    event = None
    if current and direction != current["direction"]:
        entry = float(current["entry_price"])
        exit_price = float(price)
        ret = (exit_price - entry) / entry if current["direction"] == "BUY" else (entry - exit_price) / entry
        event = {"symbol": symbol, "direction": current["direction"], "entry_price": entry, "exit_price": exit_price, "entry_time": current.get("entry_time"), "exit_time": timestamp, "returns": ret, "outcome": "win" if ret > 0 else "loss", "validated": True, "mode": "paper_forward"}
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        positions.pop(symbol, None)
    if direction in {"BUY", "SELL"} and (not current or current["direction"] != direction):
        positions[symbol] = {"direction": direction, "entry_price": float(price), "entry_time": timestamp}
    _save_open(open_path, positions)
    return event


def forward_validation_status(min_trades: int = 50) -> dict[str, Any]:
    open_path, ledger_path = _paths()
    rows: list[dict[str, Any]] = []
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("validated") and isinstance(row.get("returns"), (int, float)) and math.isfinite(float(row["returns"])):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    returns = [float(x["returns"]) for x in rows]
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    expectancy = sum(returns) / len(returns) if returns else None
    profit_factor = sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else None)
    win_rate = len(wins) / len(returns) if returns else None
    gate = bool(len(returns) >= min_trades and expectancy is not None and expectancy > 0 and profit_factor is not None and profit_factor > 1 and win_rate is not None and win_rate >= 0.60)
    return {"validated_round_trips": len(returns), "open_paper_positions": _load_open(open_path), "win_rate": win_rate, "expectancy": expectancy, "profit_factor": profit_factor, "min_trades": min_trades, "forward_gate": "PASS" if gate else "BLOCK", "live_eligible": False, "paper_only": True, "live_trading": False, "reason": "human approval and broker fill reconciliation required even after gate pass"}
