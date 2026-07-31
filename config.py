"""Clean config: single source of truth for paths and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Runtime dirs
    root: Path = Path("/Users/vera/trading-system")
    audit_dir: Path = root / "backtests"
    state_dir: Path = root / "state"
    docs_dir: Path = root / "docs"

    # Default trading universe
    # Yahoo-style symbols for yfinance direct download
    futures_universe: tuple[str, ...] = ("MNQ=F", "MYM=F", "MES=F", "/NQ", "/YM")
    crypto_universe: tuple[str, ...] = ("LINKUSD",)

    # Risk defaults
    kelly_cap: float = 0.25
    min_trades_for_significance: int = 30

    def __post_init__(self) -> None:
        # Allow env overrides for containerized runs
        object.__setattr__(self, "audit_dir", Path(os.environ.get("AUDIT_DIR", str(self.audit_dir))))
        object.__setattr__(self, "state_dir", Path(os.environ.get("STATE_DIR", str(self.state_dir))))


CONFIG = Config()
