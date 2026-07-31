# Hermes Trading Agent — Railway Deployment Package

## Structure
- `dashboard/main.py` — FastAPI runtime with reasoning stream, council endpoints, and web-access stats.
- `net/client.py` — unified outbound web wrapper with metering, caching, rate limiting, and hard spend cap (`HERMES_WEB_MAX_SPEND=0`).
- `net/sources.py` — free-source registry exposed as environment-backed config.
- `strategies/` — Kraken swing and Lucid momentum strategies, plus council/pipeline modules.
- `Procfile` — Railway worker definition.
- `railway.json` — Railway deploy/health config.

## Deploy steps
1. `git push` to a repo.
2. Railway New Project → Deploy from GitHub → select this repo/service.
3. Use worker process type; no execution-related secrets required.
4. Health-check path: `/health`.
5. Deployed URL exposes the same `dashboard/index.html` loaded at `/`.

## Guardrails at runtime
- Execution mode is observation/paper only. No exchange order credentials are set in the deployment environment.
- Web spend is hard-capped with default `HERMES_WEB_MAX_SPEND=0.00`, so paid calls are refused before they occur.
- Dashboard shows paper mode label and live reasoning/council/web-access panels.
