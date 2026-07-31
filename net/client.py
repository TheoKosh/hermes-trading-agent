"""Unified outbound web-access wrapper with metering, caching, and rate limiting."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Hard-coded spend ceiling. No call tagged "paid" may push total_cost above this.
MAX_MONTHLY_SPEND_USD: float = float(os.environ.get("HERMES_WEB_MAX_SPEND", "0.00"))
CACHE_DIR: str = os.environ.get("HERMES_WEB_CACHE", "/Users/vera/trading-system/state/web/cache")
CACHE_TTL_SECONDS: int = int(os.environ.get("HERMES_WEB_CACHE_TTL", "300"))

# Default rate limits by source name. Requests beyond these are refused or delayed.
SOURCE_LIMITS: Dict[str, Dict[str, Any]] = {
    "kraken-public": {"rpm": 15, "cost_per_1k": 0.0},
    "yahoo": {"rpm": 10, "cost_per_1k": 0.0},
    "rss": {"rpm": 20, "cost_per_1k": 0.0},
    "fred": {"rpm": 10, "cost_per_1k": 0.0},
    "default": {"rpm": 5, "cost_per_1k": 0.0},
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    base_url: str
    cost_tag: str = "free"  # "free" | "paid"
    cost_per_1k_calls: float = 0.0
    rpm: int = 5
    cache_ttl: int = CACHE_TTL_SECONDS
    allow_paths: Tuple[str, ...] = ()
    deny_paths: Tuple[str, ...] = ()


class BudgetExceeded(Exception):
    """Raised when a call would exceed the hard monthly spend cap."""


class RateLimitExceeded(Exception):
    """Raised when a source exceeds its RPM limit."""


class _Meter:
    def __init__(self) -> None:
        self.total_cost_usd: float = 0.0
        self.calls: Dict[str, list[float]] = {}
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.feed: list[dict[str, Any]] = []

    def _prune(self, source: str) -> None:
        now = time.time()
        window = 60.0
        self.calls[source] = [t for t in self.calls.get(source, []) if now - t < window]

    def check_rate(self, source: str, rpm: int) -> None:
        self._prune(source)
        if len(self.calls.get(source, [])) >= rpm:
            raise RateLimitExceeded(f"{source} rate limit exceeded: {rpm}/min")

    def record_call(self, source: str, path: str, cost: float) -> None:
        if cost > 0 and self.total_cost_usd + cost > MAX_MONTHLY_SPEND_USD:
            raise BudgetExceeded(
                f"Monthly spend cap exceeded: ${self.total_cost_usd + cost:.4f} > ${MAX_MONTHLY_SPEND_USD:.4f}"
            )
        self.calls.setdefault(source, []).append(time.time())
        if cost > 0:
            self.total_cost_usd += cost
        self.feed.append(
            {
                "ts": time.time(),
                "source": source,
                "path": path,
                "cost": cost,
                "total_cost_usd": round(self.total_cost_usd, 6),
            }
        )
        if len(self.feed) > 200:
            self.feed[:] = self.feed[-200:]

    def cache_key(self, source: str, method: str, path: str, params: Dict[str, Any]) -> str:
        raw = json.dumps({"source": source, "method": method, "path": path, "params": params}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def cached(self, key: str, ttl: int) -> Optional[Any]:
        hit = self.cache.get(key)
        if not hit:
            return None
        ts, payload = hit
        if time.time() - ts > ttl:
            del self.cache[key]
            return None
        return payload

    def store(self, key: str, payload: Any) -> None:
        self.cache[key] = (time.time(), payload)


class WebClient:
    def __init__(self, meter: Optional[_Meter] = None) -> None:
        self.meter = meter or _Meter()
        self.sources: Dict[str, SourceConfig] = {}
        self.session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def register(self, cfg: SourceConfig) -> None:
        self.sources[cfg.name] = cfg

    def request(self, source: str, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        cfg = self.sources.get(source, SourceConfig(name=source, base_url="", rpm=SOURCE_LIMITS.get("default", {}).get("rpm", 5)))
        if cfg.allow_paths and not any(path == allowed or path.startswith(allowed.rstrip("/") + "/") for allowed in cfg.allow_paths):
            raise ValueError(f"Path not allowed for {source}: {path}")
        if cfg.deny_paths and any(path == denied or path.startswith(denied.rstrip("/") + "/") for denied in cfg.deny_paths):
            raise ValueError(f"Path denied for {source}: {path}")
        if cfg.cost_tag != "free" and (cfg.cost_per_1k_calls > 0 or MAX_MONTHLY_SPEND_USD == 0):
            if MAX_MONTHLY_SPEND_USD == 0:
                raise BudgetExceeded("Paid calls are blocked when MAX_MONTHLY_SPEND_USD=0")
            est_cost = cfg.cost_per_1k_calls
            if self.meter.total_cost_usd + est_cost > MAX_MONTHLY_SPEND_USD:
                raise BudgetExceeded(
                    f"Call to {source} would exceed monthly cap: est=${est_cost:.4f}, total=${self.meter.total_cost_usd:.4f} > ${MAX_MONTHLY_SPEND_USD:.4f}"
                )

        self.meter.check_rate(source, cfg.rpm)

        key = self.meter.cache_key(source, method, path, params or {})
        cached = self.meter.cached(key, cfg.cache_ttl)
        if cached is not None:
            return {"cached": True, "data": cached, "source": source, "path": path}

        url = cfg.base_url + path if cfg.base_url and path.startswith("/") else path
        if not url.startswith("http"):
            url = cfg.base_url + path

        resp = self.session.request(method, url, params=params, timeout=kwargs.get("timeout", 15))
        resp.raise_for_status()
        payload = resp.json() if "application/json" in resp.headers.get("content-type", "") else {"text": resp.text}

        cost = (cfg.cost_per_1k_calls / 1000.0) if cfg.cost_per_1k_calls else 0.0
        self.meter.record_call(source, path, cost)
        self.meter.store(key, payload)

        return {
            "cached": False,
            "data": payload,
            "source": source,
            "path": path,
            "status": resp.status_code,
            "cost": cost,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.meter.total_cost_usd, 6),
            "max_monthly_spend_usd": MAX_MONTHLY_SPEND_USD,
            "calls_by_source": {k: len(v) for k, v in self.meter.calls.items()},
            "cache_entries": len(self.meter.cache),
        }


_default_client = WebClient()


def web_get(source: str, path: str, *, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return _default_client.request(source, "GET", path, params=params, **kwargs)


def register_source(cfg: SourceConfig, client: Optional[WebClient] = None) -> None:
    (client or _default_client).register(cfg)


def web_stats() -> Dict[str, Any]:
    return _default_client.stats()


def web_feed(limit: int = 50) -> Dict[str, Any]:
    meter = _default_client.meter
    recent = meter.feed[-limit:]
    recent.reverse()
    return {
        "entries": recent,
        "count": len(meter.feed),
        "total_cost_usd": round(meter.total_cost_usd, 6),
    }


__all__ = [
    "WebClient",
    "SourceConfig",
    "BudgetExceeded",
    "RateLimitExceeded",
    "web_get",
    "register_source",
    "web_stats",
    "web_feed",
    "MAX_MONTHLY_SPEND_USD",
]
