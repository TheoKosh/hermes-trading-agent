from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestFuturesPaperStatus(unittest.TestCase):
    def test_reports_asset_specific_models_and_paper_simulation(self):
        from integrations.futures_paper_status import futures_paper_status

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            models = state / "models" / "futures"
            models.mkdir(parents=True)
            (models / "mnq.json").write_text(json.dumps({"version": "futures_meta_MNQ", "symbol": "MNQ=F", "interval": "15m", "metrics": {"auc": 0.57, "precision": 0.4}}))
            artifact = state / "backtests" / "lucid_momentum.json"
            artifact.parent.mkdir()
            artifact.write_text(json.dumps({"metrics": {"expectancy": -0.001}, "initial_capital": 25000.0, "regimes": {"MNQ=F": "TRENDING_UP"}, "trade_log": [{"returns": 0.01, "symbol": "MNQ=F", "exit_time": "2026-01-01"}, {"returns": -0.005, "symbol": "MNQ=F", "exit_time": "2026-01-02"}]}))
            (state / "automatic_backtest_state.json").write_text(json.dumps({"results": [{"module": "backtests.run_lucid", "artifact": str(artifact), "metrics": {"expectancy": -0.001}}]}))
            with patch.dict(os.environ, {"STATE_DIR": str(state)}, clear=False):
                result = futures_paper_status()

        self.assertEqual(result["asset_count"], 1)
        self.assertIn("MNQ=F", result["assets"])
        self.assertEqual(result["simulation"]["module"], "backtests.run_lucid")
        self.assertEqual(result["simulation"]["regimes"]["MNQ=F"], "TRENDING_UP")
        self.assertEqual(len(result["simulation"]["capital_curve"]), 2)
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["live_trading"])
        self.assertFalse(result["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
