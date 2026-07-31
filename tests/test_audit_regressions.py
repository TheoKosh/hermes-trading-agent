from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd


class TestAuditLayerValidation(unittest.TestCase):
    def test_sanity_bounds_rejects_nan_last_price(self):
        from dashboard.audit_layer import AuditDecision, check_sanity_bounds

        result = check_sanity_bounds("kraken", "LINKUSD", "1h", [1.0, float("nan")], [10.0, 10.0])

        self.assertIsNotNone(result)
        self.assertEqual(result.status, AuditDecision.REJECTED)
        self.assertEqual(result.check, "sanity_bounds")

    def test_freshness_rejects_future_bar_timestamp(self):
        from dashboard.audit_layer import AuditDecision, check_freshness

        result = check_freshness("kraken", "LINKUSD", "1h", last_bar_ts=2_000.0, now=1_000.0)

        self.assertEqual(result.status, AuditDecision.REJECTED)
        self.assertEqual(result.check, "freshness")

    def test_reject_updates_source_status_and_count(self):
        import importlib
        import dashboard.audit_layer as audit_layer

        with tempfile.TemporaryDirectory() as tmp:
            old_state_dir = audit_layer.STATE_DIR
            old_state = audit_layer.DASH_AUDIT_STATE
            old_log = audit_layer.DASH_AUDIT_LOG
            audit_layer.STATE_DIR = audit_layer.Path(tmp)
            audit_layer.DASH_AUDIT_STATE = audit_layer.STATE_DIR / "dash_audit_state.json"
            audit_layer.DASH_AUDIT_LOG = audit_layer.STATE_DIR / "dash_audit.jsonl"
            try:
                audit_layer._reject("kraken", "freshness", "stale", symbol="LINKUSD", timeframe="1h")
                state = audit_layer._load_audit_state()
            finally:
                audit_layer.STATE_DIR = old_state_dir
                audit_layer.DASH_AUDIT_STATE = old_state
                audit_layer.DASH_AUDIT_LOG = old_log

        self.assertEqual(state["sources"]["kraken"]["last_status"], audit_layer.AuditDecision.REJECTED)
        self.assertEqual(state["sources"]["kraken"]["rejections"], 1)


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestLiveTraderLucid(unittest.TestCase):
    def test_live_mode_placeholder_does_not_report_execution_success(self):
        import live_trader

        old_live = live_trader.LIVE_MODE
        old_url = live_trader.TRADERSPOST_WEBHOOK_URL
        old_key = live_trader.KRAKEN_API_KEY
        old_secret = live_trader.KRAKEN_API_SECRET
        live_trader.LIVE_MODE = True
        live_trader.TRADERSPOST_WEBHOOK_URL = "https://example.test/webhook"
        live_trader.KRAKEN_API_KEY = "key"
        live_trader.KRAKEN_API_SECRET = "secret"
        try:
            with patch.object(live_trader.requests, "post") as post:
                ok = live_trader.execute_trade({"direction": "buy"})
        finally:
            live_trader.LIVE_MODE = old_live
            live_trader.TRADERSPOST_WEBHOOK_URL = old_url
            live_trader.KRAKEN_API_KEY = old_key
            live_trader.KRAKEN_API_SECRET = old_secret

        self.assertFalse(ok)
        post.assert_not_called()

    def test_run_lucid_uses_strategy_signal_without_import_or_pd_failure(self):
        import live_trader

        rows = 60
        idx = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "Open": [100.0] * rows,
                "High": [101.0] * rows,
                "Low": [99.0] * rows,
                "Close": [100.5] * rows,
                "Volume": [10.0] * rows,
            },
            index=idx,
        )
        fake_signal = types.SimpleNamespace(
            direction="buy",
            score=55,
            stop_loss=99.0,
            take_profit=103.0,
        )

        with patch.object(live_trader.yf, "download", return_value=df), \
             patch("strategies.futures_momentum.generate_signal", return_value=fake_signal), \
             patch.object(live_trader, "execute_trade", return_value=True) as execute_trade, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            live_trader.run_lucid()

        self.assertEqual(execute_trade.call_count, 3, stdout.getvalue())
        self.assertNotIn("Lucid fetch failed", stdout.getvalue())


class TestWebClientGuards(unittest.TestCase):
    def test_record_call_does_not_mutate_total_cost_when_budget_exceeded(self):
        import net.client as client

        meter = client._Meter()
        original_cap = client.MAX_MONTHLY_SPEND_USD
        client.MAX_MONTHLY_SPEND_USD = 0.001
        try:
            with self.assertRaises(client.BudgetExceeded):
                meter.record_call("paid-source", "/path", 0.002)
            self.assertEqual(meter.total_cost_usd, 0.0)
            self.assertEqual(meter.feed, [])
        finally:
            client.MAX_MONTHLY_SPEND_USD = original_cap

    def test_webclient_rejects_paths_outside_source_allowlist_before_network(self):
        import net.client as client

        wc = client.WebClient()
        wc.register(client.SourceConfig(name="locked", base_url="https://example.test", allow_paths=("/allowed",)))
        wc.session.request = Mock(side_effect=AssertionError("network should not be called"))

        with self.assertRaises(ValueError):
            wc.request("locked", "GET", "/denied")


class TestDashboardDecisionGate(unittest.TestCase):
    def test_broadcast_persists_trade_memory(self):
        import dashboard.main as main
        from strategies.pipeline import DecisionResult, PerceptionResult, ReasoningChain, RiskCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            old_state_dir = main.STATE_DIR
            main.STATE_DIR = main.Path(tmp)
            try:
                chain = ReasoningChain(
                    perception=PerceptionResult(symbol="LINKUSD", timestamp="1700000000", bars=60, valid=True),
                    features=None,
                    regime=None,
                    hypotheses=(),
                    risk=RiskCheckResult(passed=True),
                    decision=DecisionResult(action="buy", direction="buy", score=70, strategy="kraken-swing", stop_loss=9, take_profit=12, reason="test"),
                    symbol="LINKUSD",
                    timeframe="1h",
                )
                trade = {"timestamp": "1700000000", "module": "kraken", "direction": "buy", "score": 70, "outcome": "simulated"}
                main._broadcast_chain(chain, trade)
                main._broadcast_chain(chain, trade)
                memory = main._trade_memory(limit=10)
            finally:
                main.STATE_DIR = old_state_dir

        self.assertEqual(memory["count"], 1)
        self.assertEqual(memory["trades"][0]["symbol"], "LINKUSD")

    def test_ml_audit_reports_corrupt_manifest_and_gate(self):
        import dashboard.main as main
        import strategies.ml_meta as ml

        with tempfile.TemporaryDirectory() as tmp:
            old_model_dir = os.environ.get("MODEL_DIR")
            os.environ["MODEL_DIR"] = tmp
            try:
                model_dir = ml.Path(tmp)
                good = model_dir / "lgbm_meta_v2.json"
                bad = model_dir / "lgbm_meta_v1.json"
                good.write_text('{"version":"lgbm_meta_v2","metrics":{"precision":0.56,"auc":0.61}}')
                bad.write_text('{"version":')
                audit = asyncio.run(main.ml_audit())
            finally:
                if old_model_dir is None:
                    os.environ.pop("MODEL_DIR", None)
                else:
                    os.environ["MODEL_DIR"] = old_model_dir

        self.assertTrue(audit["gate_pass"])
        self.assertEqual(len(audit["corrupt_manifests"]), 1)

    def test_website_defines_called_pollers_and_trade_memory_panel(self):
        from pathlib import Path
        html = Path(os.path.join(ROOT, "dashboard", "index.html")).read_text()

        self.assertIn('id="trade-memory"', html)
        self.assertIn('id="ml-audit"', html)
        self.assertIn('id="ml-conclusion"', html)
        self.assertIn('id="ml-learning-review"', html)
        self.assertIn('RUN PAPER ERROR REVIEW', html)
        self.assertIn('id="automatic-backtest-status"', html)
        self.assertIn('AUTOMATIC PAPER BACKTEST · CANDIDATE GATE', html)
        self.assertIn('MACHINE CONCLUSION · ZERO-TOKEN LEARNER', html)
        self.assertIn('id="tradingagents-status"', html)
        self.assertIn('id="self-evolution-status"', html)
        self.assertIn('id="mutation-log"', html)
        self.assertIn('MUTATION LOG · CANDIDATES ONLY', html)
        self.assertIn('id="free-claude-code-status"', html)
        self.assertIn('id="live-series"', html)
        self.assertIn('id="multi-asset-scanner"', html)
        self.assertIn('MULTI-ASSET PAPER SCANNER · 14 DATA POINTS', html)
        self.assertIn('LIVE MARKET SERIES · BKLIT UI', html)
        self.assertIn('SELF-EVOLUTION SAFETY GATE', html)
        self.assertIn('FREE CLAUDE CODE TOKEN ROUTER', html)
        self.assertIn('async function loadCouncil', html)
        self.assertIn('async function loadWebStats', html)
        self.assertIn('async function loadTradeMemory', html)
        self.assertIn('async function loadMlAudit', html)
        self.assertIn('async function loadTradingAgentsStatus', html)
        self.assertIn('async function loadSelfEvolutionStatus', html)
        self.assertIn('async function loadFreeClaudeCodeStatus', html)
        self.assertIn('async function loadLiveMarketSeries', html)

    def test_ml_meta_hold_blocks_rule_signal_from_becoming_trade_action(self):
        import dashboard.main as main
        from strategies.pipeline import Hypothesis

        closes = [100.0 + i * 0.1 for i in range(60)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        volumes = [10.0 for _ in closes]
        ts = [str(1_700_000_000 + i * 3600) for i in range(60)]
        df = main._kraken_features(closes, highs, lows, volumes)

        fake_ml = types.ModuleType("strategies.ml_meta")

        class MLMetaResult:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        setattr(fake_ml, "MLMetaResult", MLMetaResult)
        setattr(fake_ml, "load_latest_model", lambda: (object(), {"version": "test", "metrics": {}}))
        setattr(fake_ml, "_no_lookahead_features", lambda _df: df)
        setattr(fake_ml, "predict_meta_label", lambda _model, _feats: (0, 0.1))
        setattr(
            fake_ml,
            "build_hypothesis_with_meta_label",
            lambda *_args, **_kwargs: Hypothesis(
                direction="hold", score=0, evidence=("meta-label blocked",), strategy="ml-meta-blocked"
            ),
        )

        previous_ml = sys.modules.get("strategies.ml_meta")
        sys.modules["strategies.ml_meta"] = fake_ml
        try:
            with patch.object(main, "_kraken_perception", return_value=(closes, highs, lows, volumes, ts, [])), \
                 patch.object(main, "_kraken_regime", return_value=("TRENDING_UP", 0.75, ("price > EMA21 > EMA50",))), \
                 patch.object(main, "_risk_check", return_value=(True, (), 5.0, 10.0)):
                payload = asyncio.run(main._current_chain_payload())
        finally:
            if previous_ml is not None:
                sys.modules["strategies.ml_meta"] = previous_ml
            else:
                sys.modules.pop("strategies.ml_meta", None)

        self.assertEqual(payload["chain"]["decision"]["action"], "hold")
        self.assertIsNone(payload["trade"])


if __name__ == "__main__":
    unittest.main()
