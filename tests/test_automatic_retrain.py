from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class TestAutomaticRetrain(unittest.TestCase):
    def test_disabled_by_default(self):
        from integrations.automatic_retrain import automatic_retrain_status

        with patch.dict(os.environ, {"HERMES_ML_AUTOTRAIN": "false"}, clear=False):
            result = automatic_retrain_status()

        self.assertFalse(result["automatic"])
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["live_trading"])

    def test_retrain_interval_is_bounded_and_candidate_only(self):
        import integrations.automatic_retrain as module

        completed = type("Completed", (), {"returncode": 0, "stdout": "trained", "stderr": ""})()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_ML_AUTOTRAIN": "true", "HERMES_ML_TRAIN_INTERVAL_SECONDS": "1800", "HERMES_ML_TRAIN_DAILY_CAP": "1", "STATE_DIR": tmp}, clear=False), patch.object(module.subprocess, "run", return_value=completed) as run:
                first = module.automatic_retrain_once()
                second = module.automatic_retrain_once()

        self.assertEqual(run.call_count, 2)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(second["status"], "not_due")
        self.assertTrue(first["paper_only"])
        self.assertFalse(first["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
