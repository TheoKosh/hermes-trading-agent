# Hermes Reasoning Path — as implemented

This document maps the current trading-system code onto the six-layer reasoning path.

## 1. Perception layer

Current status: **implicit**.

Raw market data is fetched at the start of each tick:
- Kraken: `live_trader.py` pulls from `https://api.kraken.com/0/public/OHLC`
- Lucid: `live_trader.py` pulls via `yfinance` for `MNQ=F`, `MYM=F`, `MES=F`

The data is normalized into arrays/candles. There is **no explicit validation or rejection** of ticks/gaps/stale data before downstream use. This is a flagged gap.

## 2. Feature layer

Current status: **implicit, inside strategies**.

- `strategies/kraken_swing.py`: EMA(9/21/50), ATR(14), RSI(14), session VWAP
- `strategies/futures_momentum.py`: EMA(9/21), ATR(14), RSI(14), session VWAP

These are computed directly inside `generate_signal` with no intermediate structure. The feature set is reasonable, though there is overlap between the two strategies.

## 3. Context/regime layer

Current status: **partial**.

- `kraken_swing.py` has `_detect_regime(closes, ema21, ema50)` returning `TRENDING_UP`, `TRENDING_DOWN`, or `RANGE`.
- `futures_momentum.py` has **no explicit regime classification** — it always tries to fire on every session.
- Neither strategy systematically suppresses a mean-reversion signal during a strong trend. Regime is only used to pick direction in Kraken, not to gate the strategy.

## 4. Hypothesis layer

Current status: **implicit**.

Candidate actions are generated inside `generate_signal`, but there is no explicit `Hypothesis` object. The evidence behind a signal is flattened into a single opaque `score` integer. There is no per-candidate breakdown or one-sentence explanation.

## 5. Risk-check layer

Current status: **minimal, hardcoded**.

- Stop-loss distance is `max(atr * 1.5, 5.0)` for Kraken and `max(atr * 0.5, 5.0)` for Lucid.
- Take-profit is SL × 2.0 in both cases.
- Position sizing is not implemented in strategy modules; `live_trader.py` does not implement correlation caps, drawdown checks, or max-position constraints.
- Risk is therefore a soft heuristic, not a non-negotiable safety rail.

## 6. Decision/execution layer

Current status:
- `live_trader.py` selects the single signal returned by `generate_signal` and either logs it (paper) or prints a live placeholder.
- The full reasoning chain is **not persisted** alongside the trade.
- Backtests record only `returns`, `direction`, and `strategy` in trade lists — no feature snapshot or regime evidence.

## Flagged behavior-affecting changes needed

These gaps are **not** silently patched in this pass; they are documented for explicit review:

- Perception: add tick validation and stale-data rejection; bad ticks currently propagate.
- Regime: add regime gating to Lucid; suppress mean-reversion signals during strong trend.
- Hypothesis: replace opaque score with explicit `Hypothesis` objects and one-sentence evidence strings.
- Risk: implement hard constraints in code, not comments — max position size, correlation check, drawdown pause.
- Decision trace: persist full `ReasoningChain` to state so auditable history is possible.
- Execution: wire a backend API so the dashboard can read live traces instead of simulated data.

## Current pipeline diagram (text)

```
Raw data fetch  →  signal generation  →  paper/live log
```

Implicit layers:
```
Raw data fetch  →  [percept→features→regime→hypothesis→risk→decision]  →  paper/live log
```

The six-layer structure exists conceptually in rule order but is not encoded as explicit, serializable, traceable objects.
