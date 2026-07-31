"""Optional token-saving adapter for TauricResearch TradingAgents.

Disabled by default. This is analysis-only and never submits orders.
The configuration deliberately minimizes LLM/API spend:
- one market analyst by default
- zero debate/risk rounds
- cheap quick/deep model defaults
- bounded news inputs
- zero SDK retries (the outer caller can retry deliberately)
- persistent data/memory paths on Railway volume
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

UPSTREAM_COMMIT = "a33fd4c"


def enabled() -> bool:
    return os.environ.get("TRADINGAGENTS_ENABLED", "false").lower() == "true"


def token_saving_config() -> dict[str, Any]:
    return {
        "llm_provider": os.environ.get("TRADINGAGENTS_LLM_PROVIDER", "openai_compatible"),
        "backend_url": os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL") or os.environ.get("HERMES_FCC_BASE_URL", ""),
        "deep_think_llm": os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", "gpt-4o-mini"),
        "quick_think_llm": os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM", "gpt-4o-mini"),
        "max_debate_rounds": int(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "0")),
        "max_risk_discuss_rounds": int(os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS", "0")),
        "llm_max_retries": int(os.environ.get("TRADINGAGENTS_LLM_MAX_RETRIES", "0")),
        "news_article_limit": int(os.environ.get("TRADINGAGENTS_NEWS_LIMIT", "5")),
        "global_news_article_limit": int(os.environ.get("TRADINGAGENTS_GLOBAL_NEWS_LIMIT", "3")),
        "checkpoint_enabled": False,
        "data_cache_dir": os.environ.get("TRADINGAGENTS_CACHE_DIR", "/app/state/tradingagents/cache"),
        "memory_log_path": os.environ.get("TRADINGAGENTS_MEMORY_LOG_PATH", "/app/state/tradingagents/memory.md"),
        "output_language": "English",
    }


def status() -> dict[str, Any]:
    config = token_saving_config()
    try:
        import tradingagents  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    return {
        "enabled": enabled(),
        "installed": installed,
        "upstream_commit": UPSTREAM_COMMIT,
        "analysis_only": True,
        "token_saving_config": config,
    }


def analyze(symbol: str, analysis_date: str | None = None) -> dict[str, Any]:
    """Run one optional paper analysis; never routes to execution."""
    if not enabled():
        return {"status": "disabled", "analysis_only": True}
    if not symbol or len(symbol) > 32 or any(ch in symbol for ch in "\n\r/\\"):
        raise ValueError("invalid_symbol")
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError as exc:
        return {"status": "unavailable", "analysis_only": True, "error": str(exc)}

    config = DEFAULT_CONFIG.copy()
    config.update(token_saving_config())
    run_date = analysis_date or date.today().isoformat()
    _, decision = TradingAgentsGraph(selected_analysts=("market",), debug=False, config=config).propagate(symbol, run_date)
    return {
        "status": "ok",
        "analysis_only": True,
        "symbol": symbol,
        "date": run_date,
        "decision": decision,
        "config": {k: config[k] for k in ("quick_think_llm", "deep_think_llm", "max_debate_rounds", "max_risk_discuss_rounds", "news_article_limit", "llm_max_retries")},
    }


__all__ = ["enabled", "token_saving_config", "status", "analyze"]
