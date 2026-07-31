from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class TestRecursiveTraining(unittest.TestCase):
    def test_disabled_by_default(self):
        from integrations.recursive_training import recursive_training_step

        with patch.dict(os.environ, {"HERMES_RECURSIVE_TRAINING": "false"}, clear=False):
            result = recursive_training_step()

        self.assertEqual(result["status"], "disabled")
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["live_trading"])

    def test_cycle_calls_all_bounded_stages_and_persists(self):
        import integrations.recursive_training as module

        with tempfile.TemporaryDirectory() as tmp:
            env = {"HERMES_RECURSIVE_TRAINING": "true", "STATE_DIR": tmp}
            with patch.dict(os.environ, env, clear=False), patch.object(module, "automatic_retrain_once", return_value={"status": "not_due"}) as train, patch.object(module, "automatic_backtest_once", return_value={"status": "not_due"}) as backtest, patch.object(module, "automatic_review_once", return_value={"status": "not_due"}) as review, patch.object(module, "improvement_snapshot", return_value={"quality_gate": "BLOCK", "latest_model": None, "metric_deltas": {}}):
                result = module.recursive_training_step()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(train.call_count, 1)
        self.assertEqual(backtest.call_count, 1)
        self.assertEqual(review.call_count, 1)
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["live_mutation"])


if __name__ == "__main__":
    unittest.main()
