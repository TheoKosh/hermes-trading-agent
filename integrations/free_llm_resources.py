"""Curated provider metadata from free-llm-api-resources.

This is metadata only; it never scrapes, signs up for, or calls providers.
"""
from __future__ import annotations

UPSTREAM_REPO = "https://github.com/cheahjs/free-llm-api-resources"
UPSTREAM_COMMIT = "a7ed456"

FREE_PROVIDERS = (
    {"id": "openrouter", "model": "openrouter/free", "base_url": "https://openrouter.ai/api/v1", "auth": "OPENROUTER_API_KEY", "note": "shared free quota; rate limits apply"},
    {"id": "google-ai-studio", "model": "gemini-2.5-flash-lite", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "auth": "GEMINI_API_KEY", "note": "free quota; data-use policy applies"},
    {"id": "cerebras", "model": "gpt-oss-120b", "base_url": "https://api.cerebras.ai/v1", "auth": "CEREBRAS_API_KEY", "note": "free quota; rate limits apply"},
    {"id": "groq", "model": "openai/gpt-oss-20b", "base_url": "https://api.groq.com/openai/v1", "auth": "GROQ_API_KEY", "note": "free quota; rate limits apply"},
    {"id": "cloudflare", "model": "@cf/openai/gpt-oss-20b", "base_url": "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1", "auth": "CLOUDFLARE_API_TOKEN", "note": "requires account id"},
)


def catalog() -> dict:
    return {"source": UPSTREAM_REPO, "source_commit": UPSTREAM_COMMIT, "providers": list(FREE_PROVIDERS), "auto_signup": False, "auto_calls": False}
