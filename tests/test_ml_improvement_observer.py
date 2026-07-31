from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMLImprovementObserver(unittest.TestCase):
    def test_metric_delta_and_gate_remain_fail_closed(self):
        from integrations.ml_improvement_observer import improvement_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            models = state / "models"
            models.mkdir()
            (models / "old.json").write_text(json.dumps({"version": "old", "metrics": {"auc": 0.50, "precision": 0.50}}))
            (models / "new.json").write_text(json.dumps({"version": "new", "metrics": {"auc": 0.54, "precision": 0.52}}))
            with patch.dict(os.environ, {"STATE_DIR": str(state), "MODEL_DIR": str(models)}, clear=False):
                result = improvement_snapshot()

        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["latest_model"]["version"], "new")
        self.assertAlmostEqual(result["metric_deltas"]["auc"], 0.04)
        self.assertEqual(result["quality_gate"], "BLOCK")
        self.assertFalse(result["promotion_allowed"])
        self.assertFalse(result["live_mutation"])
        self.assertFalse(result["live_trading"])


if __name__ == "__main__":
    unittest.main()
