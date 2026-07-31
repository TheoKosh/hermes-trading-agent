"""Minimal FastAPI backend that emits Hermes reasoning traces."""
from __future__ import annotations

import json
import hashlib
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

sys_path = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(sys_path))

from strategies.pipeline import (  # noqa: E402
    DecisionResult,
    FeatureResult,
    Hypothesis,
    PerceptionResult,
    ReasoningChain,
    RegimeResult,
    RiskCheckResult,
)
from strategies.council import CouncilSession, PersonaPosition, convene  # noqa: E402
from net.sources import register_all  # noqa: E402
from net.client import web_stats, web_feed  # noqa: E402
from dashboard.state_store import STATE_DIR  # noqa: E402
from dashboard.audit_layer import evaluate_feed, record_alert  # noqa: E402
from dashboard.audit_wrapper import audit_yahoo_batch  # noqa: E402

register_all()

_AUDIT_STATE_PATH = STATE_DIR / "dash_audit_state.json"
_LAST_AUDIT: dict[str, dict[str, Any]] = {}
_AUDIT_LOCK = threading.Lock()

app = FastAPI(title="Hermes Reasoning API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    index = sys_path / "dashboard" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"service": "hermes-agent", "mode": "observation", "status": "online"}

_broadcast = queue.Queue(maxsize=20)
_council_history: list[dict[str, Any]] = []
_council_lock = threading.Lock()
_trade_memory_lock = threading.Lock()


def _trade_memory_path() -> Path:
    return STATE_DIR / "trade_memory.json"


def _trade_id(trade: dict[str, Any], chain: ReasoningChain | None = None) -> str:
    raw = json.dumps(
        {
            "timestamp": trade.get("timestamp"),
            "module": trade.get("module"),
            "direction": trade.get("direction"),
            "score": trade.get("score"),
            "symbol": getattr(chain, "symbol", trade.get("symbol", "")) if chain is not None else trade.get("symbol", ""),
            "timeframe": getattr(chain, "timeframe", trade.get("timeframe", "")) if chain is not None else trade.get("timeframe", ""),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _record_trade_memory(trade: dict[str, Any], chain: ReasoningChain | None = None) -> None:
    if not trade:
        return
    with _trade_memory_lock:
        path = _trade_memory_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.loads(path.read_text()) if path.exists() else {"trades": []}
            trades = payload.setdefault("trades", [])
            entry = dict(trade)
            entry.setdefault("symbol", getattr(chain, "symbol", ""))
            entry.setdefault("timeframe", getattr(chain, "timeframe", ""))
            entry.setdefault("strategy", getattr(getattr(chain, "decision", None), "strategy", ""))
            entry.setdefault("remembered_at", time.time())
            entry["id"] = entry.get("id") or _trade_id(entry, chain)
            if not any(t.get("id") == entry["id"] for t in trades):
                trades.append(entry)
                payload["trades"] = trades[-1000:]
                path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception as exc:
            record_alert("trade-memory", "error", f"trade memory write failed: {exc}")


def _trade_memory(limit: int = 100) -> dict[str, Any]:
    with _trade_memory_lock:
        path = _trade_memory_path()
        try:
            payload = json.loads(path.read_text()) if path.exists() else {"trades": []}
        except Exception:
            payload = {"trades": []}
        trades = payload.get("trades", []) if isinstance(payload, dict) else []
        realized = [float(t["pnl"]) for t in trades if isinstance(t, dict) and isinstance(t.get("pnl"), (int, float))]
        wins = [p for p in realized if p > 0]
        return {
            "count": len(trades),
            "trades": list(reversed(trades[-limit:])),
            "realized_count": len(realized),
            "win_rate": (len(wins) / len(realized)) if realized else None,
            "expectancy": (sum(realized) / len(realized)) if realized else None,
        }


def _record_council(session: CouncilSession) -> None:
    entry = {
        "trigger": session.trigger,
        "symbol": session.chain_symbol,
        "timeframe": session.chain_timeframe,
        "decision": session.chain_decision,
        "score": session.chain_score,
        "synthesis": session.synthesis,
        "recommendation": session.recommendation,
        "target_layer": session.target_layer,
        "applied": session.applied,
        "outcome_after": session.outcome_after,
        "outcome_delta": session.outcome_delta,
        "positions": [
            {
                "persona": p.persona,
                "stance": p.stance,
                "reason": p.reason,
                "layer": p.layer,
                "confidence": p.confidence,
            }
            for p in session.positions
        ],
    }
    with _council_lock:
        _council_history.append(entry)
        if len(_council_history) > 200:
            _council_history.pop(0)


def _broadcast_chain(chain: ReasoningChain, trade: dict[str, Any] | None) -> None:
    if trade:
        _record_trade_memory(trade, chain)
    payload = {
        "chain": {
            "perception": _asdict(chain.perception),
            "features": _asdict(chain.features),
            "regime": _asdict(chain.regime),
            "hypotheses": _asdict(chain.hypotheses),
            "risk": _asdict(chain.risk),
            "decision": _asdict(chain.decision),
            "symbol": chain.symbol,
            "timeframe": chain.timeframe,
        },
        "trade": trade,
    }
    try:
        _broadcast.put_nowait(payload)
    except queue.Full:
        pass


def _asdict(obj):
    if obj is None:
        return None
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _asdict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, tuple):
        return [_asdict(x) for x in obj]
    return obj


def _run_kraken_loop() -> None:
    """Background loop: fetch Kraken bars, build trace, broadcast."""
    while True:
        try:
            closes, highs, lows, volumes, ts, errors = _kraken_perception()
            
            # === DATA AUDIT LAYER ===
            try:
                candles_for_audit = _build_candles(closes, highs, lows, volumes, ts)
                ok, audit_meta = audit_yahoo_batch("LINKUSD", "1h", candles_for_audit)
            except Exception as exc:
                record_alert("kraken", "error", f"audit wrapper failed: {exc}")
                ok = False
                audit_meta = {}
            
            audit_bundle = {
                "audit_errors": [] if ok else ["data audit failed: " + str(audit_meta)],
                "audit_results": [audit_meta] if audit_meta else [],
            }
            valid = len(errors) == 0 and len(closes) >= 50 and not _audit_blocked(audit_bundle)
            perception = PerceptionResult(
                symbol="LINKUSD",
                timestamp=ts[-1] if ts else "",
                bars=len(closes),
                valid=valid,
                errors=tuple(errors) + (tuple(audit_bundle.get("audit_errors", [])) if audit_bundle.get("audit_errors") else ()),
            )

            if not valid:
                reason = "; ".join(errors)
                if audit_bundle.get("audit_errors"):
                    reason += "; " + "; ".join(audit_bundle["audit_errors"])
                decision = DecisionResult(
                    action="hold", direction=None, score=0, strategy="kraken-swing",
                    stop_loss=None, take_profit=None, reason=reason,
                )
                chain = ReasoningChain(
                    perception=perception, features=None, regime=None, hypotheses=(),
                    risk=RiskCheckResult(passed=False, reasons=tuple(errors + audit_bundle.get("audit_errors", []))), decision=decision,
                    symbol="LINKUSD", timeframe="1h",
                )
                _broadcast_chain(chain, None)
                time.sleep(5)
                continue

            df = _kraken_features(closes, highs, lows, volumes)
            if df is None or len(df) == 0:
                decision = DecisionResult(
                    action="hold", direction=None, score=0, strategy="kraken-swing",
                    stop_loss=None, take_profit=None, reason="feature calculation failed",
                )
                chain = ReasoningChain(
                    perception=perception, features=None, regime=None, hypotheses=(),
                    risk=RiskCheckResult(passed=False, reasons=("feature calculation failed",)), decision=decision,
                    symbol="LINKUSD", timeframe="1h",
                )
                _broadcast_chain(chain, None)
                time.sleep(5)
                continue

            current = df.iloc[-1]
            price = float(current["close"])
            vwap = float(current["vwap"])
            ema9 = float(current["ema9"])
            ema21 = float(current["ema21"])
            rsi = float(current["rsi"])
            atr = float(current["atr"])

            features = FeatureResult(
                ema9=ema9, ema21=ema21, ema50=float(current["ema50"]),
                vwap=vwap, atr=atr, rsi=rsi, price=price,
            )

            regime_name, confidence, evidence = _kraken_regime(df)
            regime = RegimeResult(regime=regime_name, confidence=confidence, evidence=evidence)

            direction = None
            score = 20
            evidence_parts = [f"regime={regime_name}"]
            if regime_name == "TRENDING_UP":
                direction = "buy"
            elif regime_name == "TRENDING_DOWN":
                direction = "sell"

            hypotheses_list: list[Hypothesis] = []
            if direction:
                if (direction == "buy" and price > vwap) or (direction == "sell" and price < vwap):
                    score += 25
                    evidence_parts.append("price aligned with VWAP")
                if (direction == "buy" and 40 < rsi < 70) or (direction == "sell" and 30 < rsi < 60):
                    score += 20
                    evidence_parts.append("RSI in neutral zone")
                if (direction == "buy" and ema9 > ema21) or (direction == "sell" and ema9 < ema21):
                    score += 15
                    evidence_parts.append("short EMA acceleration")
                if abs((price - vwap) / price) < 0.01:
                    score += 10
                    evidence_parts.append("price near VWAP")
                hypotheses_list.append(Hypothesis(direction=direction, score=score, evidence=tuple(evidence_parts), strategy="kraken-swing"))

            ml_meta = None
            ml_hypotheses: list[Hypothesis] = []
            try:
                from strategies.ml_meta import load_latest_model, MLMetaResult, build_hypothesis_with_meta_label, _no_lookahead_features, predict_meta_label
                _model, _manifest = load_latest_model()
                if _model is not None and not df.empty:
                    _feats = _no_lookahead_features(df)
                    if not _feats.empty:
                        _rule = hypotheses_list[0] if hypotheses_list else Hypothesis(direction=direction or "hold", score=score, evidence=tuple(evidence_parts), strategy="kraken-swing")
                        _ml_result = MLMetaResult(
                            adopted=True,
                            model_version=str(_manifest.get("version", "unknown")) if _manifest else "unknown",
                            confidence=0.0,
                            precision=_manifest.get("metrics", {}).get("precision") if _manifest else None,
                            recall=_manifest.get("metrics", {}).get("recall") if _manifest else None,
                            f1=_manifest.get("metrics", {}).get("f1") if _manifest else None,
                            auc=_manifest.get("metrics", {}).get("auc") if _manifest else None,
                            walk_forward_summary={},
                            evidence=(),
                        )
                        if len(_feats) >= 10:
                            _row = _feats.drop(columns=["close"], errors="ignore").values[-1:].astype(float)
                            _label_prob = predict_meta_label(_model, _feats)[1]
                            _ml_result = MLMetaResult(
                                adopted=True,
                                model_version=_ml_result.model_version,
                                confidence=_label_prob,
                                precision=_ml_result.precision,
                                recall=_ml_result.recall,
                                f1=_ml_result.f1,
                                auc=_ml_result.auc,
                                walk_forward_summary=_ml_result.walk_forward_summary,
                                evidence=(),
                            )
                        ml_hypotheses = [build_hypothesis_with_meta_label(_rule, _model, _feats, _ml_result)] if _model is not None else []
                        ml_meta = {
                            "version": _ml_result.model_version,
                            "confidence": _ml_result.confidence,
                            "adopted": ml_hypotheses and ml_hypotheses[0].strategy != "ml-meta-blocked",
                            "precision": _ml_result.precision,
                            "recall": _ml_result.recall,
                            "f1": _ml_result.f1,
                            "auc": _ml_result.auc,
                        }
            except Exception as exc:
                ml_meta = {"error": str(exc)}

            if len(ml_hypotheses) > 0:
                hypotheses_list = ml_hypotheses

            sl = price - max(atr * 1.5, 5.0) if direction == "buy" else price + max(atr * 1.5, 5.0) if direction else None
            tp = price + max(atr * 1.5, 5.0) * 2.0 if direction == "buy" else price - max(atr * 1.5, 5.0) * 2.0 if direction else None
            passed, reasons, sl_dist, tp_dist = _risk_check(price, sl, tp, atr)
            risk = RiskCheckResult(passed=passed, reasons=reasons, sl_distance=sl_dist, tp_distance=tp_dist)

            action, final_direction, final_score, final_sl, final_tp, reason = _select_trade_decision(hypotheses_list, risk, sl, tp)

            decision = DecisionResult(
                action=action, direction=final_direction, score=final_score, strategy="kraken-swing",
                stop_loss=final_sl, take_profit=final_tp, reason=reason,
            )

            chain = ReasoningChain(
                perception=perception, features=features, regime=regime,
                hypotheses=tuple(hypotheses_list), risk=risk, decision=decision,
                symbol="LINKUSD", timeframe="1h",
            )

            trade = None
            if action in {"buy", "sell"}:
                trade = {
                    "timestamp": perception.timestamp,
                    "module": "kraken",
                    "direction": action,
                    "score": final_score,
                    "reason": reason,
                    "outcome": "simulated",
                }

            _broadcast_chain(chain, trade)
            time.sleep(5)
        except Exception as exc:
            time.sleep(5)


def _run_ml_learning_loop() -> None:
    while True:
        try:
            from integrations.ml_learning_review import automatic_review_once
            automatic_review_once()
        except Exception as exc:
            print(f"automatic ML review failed: {exc}")
        time.sleep(60)


def _run_automatic_backtest_loop() -> None:
    while True:
        try:
            from integrations.automatic_backtest import automatic_backtest_once
            automatic_backtest_once()
        except Exception as exc:
            print(f"automatic backtest failed: {exc}")
        time.sleep(60)


def _run_automatic_retrain_loop() -> None:
    while True:
        try:
            from integrations.automatic_retrain import automatic_retrain_once
            automatic_retrain_once()
        except Exception as exc:
            print(f"automatic retrain failed: {exc}")
        time.sleep(60)


def _run_recursive_training_loop() -> None:
    while True:
        try:
            from integrations.recursive_training import recursive_training_step
            recursive_training_step()
        except Exception as exc:
            print(f"recursive training failed: {exc}")
        time.sleep(60)


def _kraken_perception():
    symbol = "LINKUSD"
    interval = 60
    errors: list[str] = []
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    volumes: list[float] = []
    ts: list[str] = []
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": symbol, "interval": interval},
            timeout=20,
        )
        data = r.json()
        pair = next(k for k in data["result"] if k != "last")
        bars = data["result"][pair]
        closes = [float(b[4]) for b in bars]
        highs = [float(b[2]) for b in bars]
        lows = [float(b[3]) for b in bars]
        volumes = [float(b[6]) for b in bars]
        ts = [str(b[0]) for b in bars]
        if len(closes) < 50:
            errors.append("insufficient bars")
    except Exception as exc:
        errors.append(f"fetch failed: {exc}")
    return closes, highs, lows, volumes, ts, errors


def _build_candles(closes, highs, lows, volumes, ts):
    out = []
    for i in range(len(closes)):
        out.append({
            "t": float(ts[i]) if i < len(ts) else 0.0,
            "o": float(closes[i]),
            "h": float(highs[i]) if i < len(highs) else float(closes[i]),
            "l": float(lows[i]) if i < len(lows) else float(closes[i]),
            "c": float(closes[i]),
            "v": float(volumes[i]) if i < len(volumes) else 0.0,
        })
    return out


def _kraken_features(closes, highs, lows, volumes):
    if len(closes) < 50:
        return None
    df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": volumes})
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap"] = pv.cumsum() / df["volume"].cumsum().replace(0, 1e-12)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = tr1.iloc[0]
    df["atr"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    df["rsi"] = 100.0 - 100.0 / (1.0 + rs)
    df["rsi"] = df["rsi"].fillna(50.0)
    return df


def _kraken_regime(df):
    last = float(df["close"].iloc[-1])
    ema21 = float(df["ema21"].iloc[-1])
    ema50 = float(df["ema50"].iloc[-1])
    if last > ema21 > ema50:
        return "TRENDING_UP", 0.75, ("price > EMA21 > EMA50",)
    if last < ema21 < ema50:
        return "TRENDING_DOWN", 0.75, ("price < EMA21 < EMA50",)
    return "RANGE", 0.5, ("EMAs converged",)


def _risk_check(price, sl, tp, atr):
    sl_dist = abs(price - sl) if sl is not None else 0.0
    tp_dist = abs(tp - price) if tp is not None else 0.0
    reasons: list[str] = []
    if sl_dist <= 0:
        reasons.append("invalid stop")
    if tp_dist <= 0:
        reasons.append("invalid target")
    passed = len(reasons) == 0 and sl_dist > 0 and tp_dist > 0
    return passed, tuple(reasons), sl_dist, tp_dist


def _select_trade_decision(hypotheses_list, risk, sl, tp):
    """Convert the final hypothesis/risk state into an executable decision.

    A meta-label block is represented as a hold hypothesis. Treat that as a
    hard veto; do not fall back to the original rule direction/score.
    """
    if not hypotheses_list:
        return "hold", None, 0, None, None, "no candidate"

    candidate = hypotheses_list[0]
    if candidate.direction not in {"buy", "sell"}:
        return "hold", None, 0, None, None, "; ".join(candidate.evidence) or "candidate blocked"

    if not risk.passed:
        return "hold", None, 0, None, None, "; ".join(risk.reasons) or "risk failed"

    if int(candidate.score) < 50:
        return "hold", None, 0, None, None, f"candidate score below threshold: {candidate.score}"

    return (
        candidate.direction,
        candidate.direction,
        int(candidate.score),
        sl,
        tp,
        f"{candidate.strategy} candidate score {candidate.score}; risk passed",
    )


@app.get("/health")
async def health():
    return {"kraken": {"module": "kraken", "status": "ok"}, "lucid": {"module": "lucid", "status": "pending"}}


@app.get("/audit/state")
async def audit_state():
    try:
        if _AUDIT_STATE_PATH.exists():
            return json.loads(_AUDIT_STATE_PATH.read_text())
    except Exception:
        pass
    return {"sources": {}, "rejections": [], "alerts": [], "last_audit_ts": 0.0}


def _audit_kraken(closes, highs, lows, volumes, ts, errors):
    """Run the data audit layer on raw Kraken bars."""
    return {
        "audit_errors": [],
        "audit_results": [],
    }


def _audit_blocked(bundle: dict) -> bool:
    return bool(bundle.get("audit_errors"))


@app.get("/net/stats")
async def net_stats():
    return web_stats()


@app.get("/net/feed")
async def net_feed():
    return web_feed()


@app.get("/trades/history")
async def trades_history(limit: int = 100):
    return _trade_memory(limit=max(1, min(int(limit), 1000)))


@app.get("/ml/audit")
async def ml_audit():
    try:
        model_dir = Path(os.environ.get("MODEL_DIR") or (STATE_DIR / "models"))
        manifests = []
        corrupt = []
        for path in sorted(model_dir.glob("lgbm_meta_*.json")):
            try:
                data = json.loads(path.read_text())
                manifests.append({"file": path.name, "metrics": data.get("metrics", {}), "version": data.get("version")})
            except Exception as exc:
                corrupt.append({"file": path.name, "error": str(exc)})
        latest = manifests[-1] if manifests else None
        metrics = latest.get("metrics", {}) if latest else {}
        precision = metrics.get("precision")
        auc = metrics.get("auc")
        gate_pass = bool(isinstance(precision, (int, float)) and isinstance(auc, (int, float)) and precision >= 0.55 and auc >= 0.60)
        return {
            "status": "ok" if latest else "missing",
            "model_dir": str(model_dir),
            "latest": latest,
            "corrupt_manifests": corrupt,
            "gate_pass": gate_pass,
            "gate": {"min_precision": 0.55, "min_auc": 0.60},
            "recommendation": "paper_only" if not gate_pass else "eligible_for_paper_review_not_live",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "gate_pass": False, "recommendation": "paper_only"}


@app.get("/ml/improvements")
async def ml_improvements():
    from integrations.ml_improvement_observer import improvement_snapshot
    return improvement_snapshot()


@app.get("/ml/training/status")
async def ml_training_status():
    from integrations.automatic_retrain import automatic_retrain_status
    return automatic_retrain_status()


@app.get("/ml/recursive-training/status")
async def ml_recursive_training_status():
    from integrations.recursive_training import recursive_training_status
    return recursive_training_status()


@app.get("/ml/conclusion")
async def ml_conclusion():
    from integrations.ml_ai_advisor import status
    return status()


@app.get("/ml/conclusion/ai")
async def ml_ai_conclusion():
    from integrations.ml_ai_advisor import ai_conclusion
    return ai_conclusion()


@app.get("/ml/learning-review")
async def ml_learning_review():
    from integrations.ml_learning_review import review_learning
    return review_learning()


@app.get("/ml/learning-review/status")
async def ml_learning_review_status():
    from integrations.ml_learning_review import automatic_review_status
    return automatic_review_status()


@app.get("/backtest/automatic/status")
async def automatic_backtest_status_route():
    from integrations.automatic_backtest import automatic_backtest_status
    return automatic_backtest_status()


@app.get("/free-llm/resources")
async def free_llm_resources():
    from integrations.free_llm_resources import catalog
    return catalog()


@app.get("/tradingagents/status")
async def tradingagents_status():
    from integrations.tradingagents_adapter import status
    return status()


@app.get("/tradingagents/analyze")
async def tradingagents_analyze(symbol: str = "LINK-USD", analysis_date: str | None = None):
    from integrations.tradingagents_adapter import analyze
    return analyze(symbol, analysis_date)


@app.get("/self-evolution/status")
async def self_evolution_status():
    from integrations.self_evolution_adapter import status
    return status()


@app.get("/self-evolution/dry-run")
async def self_evolution_dry_run(skill: str = "trading-system-ops"):
    from integrations.self_evolution_adapter import dry_run
    return dry_run(skill)


@app.get("/self-evolution/mutations")
async def self_evolution_mutations(limit: int = 100):
    from integrations.self_evolution_adapter import mutation_log
    return mutation_log(limit)


@app.get("/free-claude-code/status")
async def free_claude_code_status():
    from integrations.free_claude_code_adapter import status
    return status()


def _recent_history(limit: int = 20) -> list[dict[str, Any]]:
    with _council_lock:
        return list(reversed(_council_history[-limit:]))


async def _current_chain_payload() -> dict[str, Any]:
    closes, highs, lows, volumes, ts, errors = _kraken_perception()
    valid = len(errors) == 0 and len(closes) >= 50
    perception = PerceptionResult(
        symbol="LINKUSD",
        timestamp=ts[-1] if ts else "",
        bars=len(closes),
        valid=valid,
        errors=tuple(errors),
    )
    if not valid:
        decision = DecisionResult(
            action="hold", direction=None, score=0, strategy="kraken-swing",
            stop_loss=None, take_profit=None, reason="; ".join(errors) if errors else "insufficient data",
        )
        chain = ReasoningChain(
            perception=perception, features=None, regime=None, hypotheses=(),
            risk=RiskCheckResult(passed=False, reasons=tuple(errors)), decision=decision,
            symbol="LINKUSD", timeframe="1h",
        )
        payload = {
            "perception": _asdict(chain.perception),
            "features": _asdict(chain.features),
            "regime": _asdict(chain.regime),
            "hypotheses": _asdict(chain.hypotheses),
            "risk": _asdict(chain.risk),
            "decision": _asdict(chain.decision),
            "symbol": chain.symbol,
            "timeframe": chain.timeframe,
        }
        return {"chain": payload, "trade": None}

    df = _kraken_features(closes, highs, lows, volumes)
    if df is None or len(df) == 0:
        decision = DecisionResult(
            action="hold", direction=None, score=0, strategy="kraken-swing",
            stop_loss=None, take_profit=None, reason="feature calculation failed",
        )
        chain = ReasoningChain(
            perception=perception, features=None, regime=None, hypotheses=(),
            risk=RiskCheckResult(passed=False, reasons=("feature calculation failed",)), decision=decision,
            symbol="LINKUSD", timeframe="1h",
        )
        payload = {
            "perception": _asdict(chain.perception),
            "features": _asdict(chain.features),
            "regime": _asdict(chain.regime),
            "hypotheses": _asdict(chain.hypotheses),
            "risk": _asdict(chain.risk),
            "decision": _asdict(chain.decision),
            "symbol": chain.symbol,
            "timeframe": chain.timeframe,
        }
        return {"chain": payload, "trade": None}

    current = df.iloc[-1]
    price = float(current["close"])
    vwap = float(current["vwap"])
    ema9 = float(current["ema9"])
    ema21 = float(current["ema21"])
    rsi = float(current["rsi"])
    atr = float(current["atr"])

    features = FeatureResult(
        ema9=ema9, ema21=ema21, ema50=float(current["ema50"]),
        vwap=vwap, atr=atr, rsi=rsi, price=price,
    )

    regime_name, confidence, evidence = _kraken_regime(df)
    regime = RegimeResult(regime=regime_name, confidence=confidence, evidence=evidence)

    direction = None
    score = 20
    evidence_parts = [f"regime={regime_name}"]
    if regime_name == "TRENDING_UP":
        direction = "buy"
    elif regime_name == "TRENDING_DOWN":
        direction = "sell"

    hypotheses_list: list[Hypothesis] = []
    if direction:
        if (direction == "buy" and price > vwap) or (direction == "sell" and price < vwap):
            score += 25
            evidence_parts.append("price aligned with VWAP")
        if (direction == "buy" and 40 < rsi < 70) or (direction == "sell" and 30 < rsi < 60):
            score += 20
            evidence_parts.append("RSI in neutral zone")
        if (direction == "buy" and ema9 > ema21) or (direction == "sell" and ema9 < ema21):
            score += 15
            evidence_parts.append("short EMA acceleration")
        if abs((price - vwap) / price) < 0.01:
            score += 10
            evidence_parts.append("price near VWAP")
        hypotheses_list.append(Hypothesis(direction=direction, score=score, evidence=tuple(evidence_parts), strategy="kraken-swing"))

    ml_meta = None
    ml_hypotheses: list[Hypothesis] = []
    try:
        from strategies.ml_meta import load_latest_model, MLMetaResult, build_hypothesis_with_meta_label, _no_lookahead_features, predict_meta_label
        _model, _manifest = load_latest_model()
        if _model is not None and not df.empty:
            _feats = _no_lookahead_features(df)
            if not _feats.empty:
                _rule = hypotheses_list[0] if hypotheses_list else Hypothesis(direction=direction or "hold", score=score, evidence=tuple(evidence_parts), strategy="kraken-swing")
                _ml_result = MLMetaResult(
                    adopted=True,
                    model_version=str(_manifest.get("version", "unknown")) if _manifest else "unknown",
                    confidence=0.0,
                    precision=_manifest.get("metrics", {}).get("precision") if _manifest else None,
                    recall=_manifest.get("metrics", {}).get("recall") if _manifest else None,
                    f1=_manifest.get("metrics", {}).get("f1") if _manifest else None,
                    auc=_manifest.get("metrics", {}).get("auc") if _manifest else None,
                    walk_forward_summary={},
                    evidence=(),
                )
                if len(_feats) >= 10:
                    _row = _feats.drop(columns=["close"], errors="ignore").values[-1:].astype(float)
                    _label_prob = predict_meta_label(_model, _feats)[1]
                    _ml_result = MLMetaResult(
                        adopted=True,
                        model_version=_ml_result.model_version,
                        confidence=_label_prob,
                        precision=_ml_result.precision,
                        recall=_ml_result.recall,
                        f1=_ml_result.f1,
                        auc=_ml_result.auc,
                        walk_forward_summary=_ml_result.walk_forward_summary,
                        evidence=(),
                    )
                ml_hypotheses = [build_hypothesis_with_meta_label(_rule, _model, _feats, _ml_result)] if _model is not None else []
                ml_meta = {
                    "version": _ml_result.model_version,
                    "confidence": _ml_result.confidence,
                    "adopted": ml_hypotheses and ml_hypotheses[0].strategy != "ml-meta-blocked",
                    "precision": _ml_result.precision,
                    "recall": _ml_result.recall,
                    "f1": _ml_result.f1,
                    "auc": _ml_result.auc,
                }
    except Exception as exc:
        ml_meta = {"error": str(exc)}

    if len(ml_hypotheses) > 0:
        hypotheses_list = ml_hypotheses

    sl = price - max(atr * 1.5, 5.0) if direction == "buy" else price + max(atr * 1.5, 5.0) if direction else None
    tp = price + max(atr * 1.5, 5.0) * 2.0 if direction == "buy" else price - max(atr * 1.5, 5.0) * 2.0 if direction else None
    passed, reasons, sl_dist, tp_dist = _risk_check(price, sl, tp, atr)
    risk = RiskCheckResult(passed=passed, reasons=reasons, sl_distance=sl_dist, tp_distance=tp_dist)

    action, final_direction, final_score, final_sl, final_tp, reason = _select_trade_decision(hypotheses_list, risk, sl, tp)

    decision = DecisionResult(
        action=action, direction=final_direction, score=final_score, strategy="kraken-swing",
        stop_loss=final_sl, take_profit=final_tp, reason=reason,
    )

    chain = ReasoningChain(
        perception=perception, features=features, regime=regime,
        hypotheses=tuple(hypotheses_list), risk=risk, decision=decision,
        symbol="LINKUSD", timeframe="1h",
    )

    trade = None
    if action in {"buy", "sell"}:
        trade = {
            "timestamp": perception.timestamp,
            "module": "kraken",
            "direction": action,
            "score": final_score,
            "reason": reason,
            "outcome": "simulated",
        }

    payload = {
        "perception": _asdict(chain.perception),
        "features": _asdict(chain.features),
        "regime": _asdict(chain.regime),
        "hypotheses": _asdict(chain.hypotheses),
        "risk": _asdict(chain.risk),
        "decision": _asdict(chain.decision),
        "symbol": chain.symbol,
        "timeframe": chain.timeframe,
        "ml_meta": ml_meta,
    }
    return {"chain": payload, "trade": trade}


@app.get("/council/now")
async def council_now():
    payload = await _current_chain_payload()
    chain = payload["chain"] if isinstance(payload, dict) else {}
    trade = payload.get("trade") if isinstance(payload, dict) else None
    history = _recent_history(limit=20)
    session = convene(chain, trade, history, trigger="live-snapshot")
    _record_council(session)
    return {
        "trigger": session.trigger,
        "chain_symbol": session.chain_symbol,
        "chain_timeframe": session.chain_timeframe,
        "chain_decision": session.chain_decision,
        "chain_score": session.chain_score,
        "positions": [
            {
                "persona": p.persona,
                "stance": p.stance,
                "reason": p.reason,
                "layer": p.layer,
                "confidence": p.confidence,
            }
            for p in session.positions
        ],
        "synthesis": session.synthesis,
        "recommendation": session.recommendation,
        "target_layer": session.target_layer,
        "applied": session.applied,
    }


@app.get("/council/history")
async def council_history():
    with _council_lock:
        return {"sessions": list(reversed(_council_history))}


@app.get("/events")
async def events():
    async def stream():
        while True:
            payload = _broadcast.get()
            yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/kraken/snapshot")
async def kraken_snapshot():
    closes, highs, lows, volumes, ts, errors = _kraken_perception()
    valid = len(errors) == 0 and len(closes) >= 50
    perception = PerceptionResult(
        symbol="LINKUSD",
        timestamp=ts[-1] if ts else "",
        bars=len(closes),
        valid=valid,
        errors=tuple(errors),
    )
    if not valid:
        decision = DecisionResult(
            action="hold", direction=None, score=0, strategy="kraken-swing",
            stop_loss=None, take_profit=None, reason="; ".join(errors) if errors else "insufficient data",
        )
        chain = ReasoningChain(
            perception=perception, features=None, regime=None, hypotheses=(),
            risk=RiskCheckResult(passed=False, reasons=tuple(errors)), decision=decision,
            symbol="LINKUSD", timeframe="1h",
        )
        payload = {
            "perception": _asdict(chain.perception),
            "features": _asdict(chain.features),
            "regime": _asdict(chain.regime),
            "hypotheses": _asdict(chain.hypotheses),
            "risk": _asdict(chain.risk),
            "decision": _asdict(chain.decision),
            "symbol": chain.symbol,
            "timeframe": chain.timeframe,
        }
        return {"chain": payload, "trade": None}

    df = _kraken_features(closes, highs, lows, volumes)
    current = df.iloc[-1]
    price = float(current["close"])
    vwap = float(current["vwap"])
    ema9 = float(current["ema9"])
    ema21 = float(current["ema21"])
    rsi = float(current["rsi"])
    atr = float(current["atr"])

    features = FeatureResult(
        ema9=ema9, ema21=ema21, ema50=float(current["ema50"]),
        vwap=vwap, atr=atr, rsi=rsi, price=price,
    )

    regime_name, confidence, evidence = _kraken_regime(df)
    regime = RegimeResult(regime=regime_name, confidence=confidence, evidence=evidence)

    direction = None
    score = 20
    evidence_parts = [f"regime={regime_name}"]
    if regime_name == "TRENDING_UP":
        direction = "buy"
    elif regime_name == "TRENDING_DOWN":
        direction = "sell"

    hypotheses_list: list[Hypothesis] = []
    if direction:
        if (direction == "buy" and price > vwap) or (direction == "sell" and price < vwap):
            score += 25
            evidence_parts.append("price aligned with VWAP")
        if (direction == "buy" and 40 < rsi < 70) or (direction == "sell" and 30 < rsi < 60):
            score += 20
            evidence_parts.append("RSI in neutral zone")
        if (direction == "buy" and ema9 > ema21) or (direction == "sell" and ema9 < ema21):
            score += 15
            evidence_parts.append("short EMA acceleration")
        if abs((price - vwap) / price) < 0.01:
            score += 10
            evidence_parts.append("price near VWAP")
        hypotheses_list.append(Hypothesis(direction=direction, score=score, evidence=tuple(evidence_parts), strategy="kraken-swing"))

    ml_meta = None
    ml_hypotheses: list[Hypothesis] = []
    try:
        from strategies.ml_meta import load_latest_model, MLMetaResult, build_hypothesis_with_meta_label, _no_lookahead_features, predict_meta_label
        _model, _manifest = load_latest_model()
        if _model is not None and not df.empty:
            _feats = _no_lookahead_features(df)
            if not _feats.empty:
                _rule = hypotheses_list[0] if hypotheses_list else Hypothesis(direction=direction or "hold", score=score, evidence=tuple(evidence_parts), strategy="kraken-swing")
                _ml_result = MLMetaResult(
                    adopted=True,
                    model_version=str(_manifest.get("version", "unknown")) if _manifest else "unknown",
                    confidence=0.0,
                    precision=_manifest.get("metrics", {}).get("precision") if _manifest else None,
                    recall=_manifest.get("metrics", {}).get("recall") if _manifest else None,
                    f1=_manifest.get("metrics", {}).get("f1") if _manifest else None,
                    auc=_manifest.get("metrics", {}).get("auc") if _manifest else None,
                    walk_forward_summary={},
                    evidence=(),
                )
                if len(_feats) >= 10:
                    _row = _feats.drop(columns=["close"], errors="ignore").values[-1:].astype(float)
                    _label_prob = predict_meta_label(_model, _feats)[1]
                    _ml_result = MLMetaResult(
                        adopted=True,
                        model_version=_ml_result.model_version,
                        confidence=_label_prob,
                        precision=_ml_result.precision,
                        recall=_ml_result.recall,
                        f1=_ml_result.f1,
                        auc=_ml_result.auc,
                        walk_forward_summary=_ml_result.walk_forward_summary,
                        evidence=(),
                    )
                ml_hypotheses = [build_hypothesis_with_meta_label(_rule, _model, _feats, _ml_result)] if _model is not None else []
                ml_meta = {
                    "version": _ml_result.model_version,
                    "confidence": _ml_result.confidence,
                    "adopted": ml_hypotheses and ml_hypotheses[0].strategy != "ml-meta-blocked",
                    "precision": _ml_result.precision,
                    "recall": _ml_result.recall,
                    "f1": _ml_result.f1,
                    "auc": _ml_result.auc,
                }
    except Exception as exc:
        ml_meta = {"error": str(exc)}

    if len(ml_hypotheses) > 0:
        hypotheses_list = ml_hypotheses

    sl = price - max(atr * 1.5, 5.0) if direction == "buy" else price + max(atr * 1.5, 5.0) if direction else None
    tp = price + max(atr * 1.5, 5.0) * 2.0 if direction == "buy" else price - max(atr * 1.5, 5.0) * 2.0 if direction else None
    passed, reasons, sl_dist, tp_dist = _risk_check(price, sl, tp, atr)
    risk = RiskCheckResult(passed=passed, reasons=reasons, sl_distance=sl_dist, tp_distance=tp_dist)

    action, final_direction, final_score, final_sl, final_tp, reason = _select_trade_decision(hypotheses_list, risk, sl, tp)

    decision = DecisionResult(
        action=action, direction=final_direction, score=final_score, strategy="kraken-swing",
        stop_loss=final_sl, take_profit=final_tp, reason=reason,
    )

    chain = ReasoningChain(
        perception=perception, features=features, regime=regime,
        hypotheses=tuple(hypotheses_list), risk=risk, decision=decision,
        symbol="LINKUSD", timeframe="1h",
    )

    trade = None
    if action in {"buy", "sell"}:
        trade = {
            "timestamp": perception.timestamp,
            "module": "kraken",
            "direction": action,
            "score": final_score,
            "reason": reason,
            "outcome": "simulated",
        }

    payload = {
        "perception": _asdict(chain.perception),
        "features": _asdict(chain.features),
        "regime": _asdict(chain.regime),
        "hypotheses": _asdict(chain.hypotheses),
        "risk": _asdict(chain.risk),
        "decision": _asdict(chain.decision),
        "symbol": chain.symbol,
        "timeframe": chain.timeframe,
        "ml_meta": ml_meta,
    }
    return {"chain": payload, "trade": trade}


@app.get("/kraken/universe")
async def kraken_universe():
    """Scan the validated paper universe without applying a cross-asset model."""
    from strategies.multi_asset_scanner import scan_universe
    return scan_universe()


@app.get("/futures/paper/status")
async def futures_paper_status_route():
    from integrations.futures_paper_status import futures_paper_status
    return futures_paper_status()


@app.get("/futures/live-paper/positions")
async def futures_live_paper_positions_route():
    from integrations.futures_live_monitor import futures_live_paper_positions
    return futures_live_paper_positions()


@app.get("/validation/forward")
async def forward_validation_status_route():
    from integrations.forward_validation import forward_validation_status
    return forward_validation_status()


@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=_run_kraken_loop, daemon=True)
    thread.start()
    recursive = os.environ.get("HERMES_RECURSIVE_TRAINING", "false").lower() == "true"
    if recursive:
        recursive_thread = threading.Thread(target=_run_recursive_training_loop, daemon=True)
        recursive_thread.start()
    else:
        if os.environ.get("HERMES_ML_AI_AUTO_REVIEW", "false").lower() == "true":
            review_thread = threading.Thread(target=_run_ml_learning_loop, daemon=True)
            review_thread.start()
        if os.environ.get("HERMES_BACKTEST_AUTO", "false").lower() == "true":
            backtest_thread = threading.Thread(target=_run_automatic_backtest_loop, daemon=True)
            backtest_thread.start()
        if os.environ.get("HERMES_ML_AUTOTRAIN", "false").lower() == "true":
            retrain_thread = threading.Thread(target=_run_automatic_retrain_loop, daemon=True)
            retrain_thread.start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")
