"""Default free sources for Hermes web access."""
from __future__ import annotations

from typing import Any

from net.client import SourceConfig, register_source

FREE_SOURCES: tuple[SourceConfig, ...] = (
    SourceConfig(
        name="kraken-public",
        base_url="https://api.kraken.com",
        cost_tag="free",
        cost_per_1k_calls=0.0,
        rpm=15,
        cache_ttl=60,
        allow_paths=("/0/public/OHLC", "/0/public/Ticker", "/0/public/Depth", "/0/public/AssetPairs"),
    ),
    SourceConfig(
        name="yahoo",
        base_url="https://query1.finance.yahoo.com",
        cost_tag="free",
        rpm=10,
        cache_ttl=120,
    ),
    SourceConfig(
        name="rss",
        base_url="",
        cost_tag="free",
        rpm=20,
        cache_ttl=300,
    ),
    SourceConfig(
        name="fred",
        base_url="https://api.stlouisfed.org",
        cost_tag="free",
        rpm=10,
        cache_ttl=3600,
    ),
)


def register_all(client: Any | None = None) -> None:
    for cfg in FREE_SOURCES:
        register_source(cfg, client)
