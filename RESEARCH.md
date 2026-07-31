# Research references

## Agent-Reach
Lightweight web-access skill assets copied from `https://github.com/Panniantong/Agent-Reach`.
- `research/agent_reach_skill/SKILL.md`
- `research/agent_reach.py`
Use only when `agent-reach` is installed in the execution environment. The dashboard does not depend on it.

## NautilusTrader
Use as read-only research reference only. Do not add the upstream source tree into this repo.
Canonical repo: `https://github.com/nautechsystems/nautilus_trader.git`
Useful for:
- bar/ticks readers in `nautilus_trader.data`
- Rust-backed adapter ideas, not runtime deps

## ML dataset directions
Current `ml/data.py` supports:
- Kraken public OHLC
- Yahoo Finance fallback via `yfinance`
Next dataset options to add:
- public Binance klines for additional liquidity
- local Parquet/CSV dataset loader for offline training
- walk-forward split utilities in `ml/train_kraken_meta.py`

## NautilusTrader dataset/research integration

Reference only: `https://github.com/nautechsystems/nautilus_trader.git`

Useful ML/reference surface:
- `nautilus_trader.data.readers` for bar/quote/readers concepts
- adapter examples under `python/examples/<exchange>/data_tester.py`
- Rust core for backtesting; not added as a runtime Python dependency here

Recommended usage pattern:
- keep Nautilus as read-only research reference
- import adapter ideas into `ml/data.py` and `strategies/` only when needed
- do not vendor the upstream tree in this workspace

## Python-project-Scripts reference material

Reference only: `https://github.com/larymak/Python-project-Scripts.git`

Use this repo for:
- reusable notebook/data-science script patterns
- notebook-style data cleanup, plotting, and EDA helpers
- copy-paste helpers for `ml/`, `backtests/`, and `dashboard/` prototyping

Recommendation:
- do not vendor the full repo
- copy individual scripts into `scripts/` only when needed
- keep imports bounded to pandas/matplotlib/scikit-learn/LightGBM already in use
