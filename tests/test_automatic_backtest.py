from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class TestAutomaticBacktest(unittest.TestCase):
    def test_disabled_by_default(self):
        from integrations.automatic_backtest import automatic_backtest_status

        with patch.dict(os.environ, {"HERMES_BACKTEST_AUTO": "false"}, clear=False):
            result = automatic_backtest_status()

        self.assertFalse(result["automatic"])
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["promoted"])
        self.assertFalse(result["live_trading"])

    def test_interval_and_daily_cap(self):
        import integrations.automatic_backtest as module

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_BACKTEST_AUTO": "true", "HERMES_BACKTEST_INTERVAL_SECONDS": "1800", "HERMES_BACKTEST_DAILY_CAP": "1", "STATE_DIR": tmp}, clear=False), patch.object(module, "_run_one", return_value={"returncode": 0, "metrics": {"expectancy": 0.1}}):
                first = module.automatic_backtest_once()
                second = module.automatic_backtest_once()

        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "not_due")
        self.assertTrue(first["paper_only"])
        self.assertFalse(first["promoted"])


if __name__ == "__main__":
    unittest.main()
