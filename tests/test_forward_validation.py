from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class TestForwardValidation(unittest.TestCase):
    def test_paper_round_trip_is_validated_and_gate_stays_blocked(self):
        from integrations.forward_validation import forward_validation_status, record_paper_signal

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"STATE_DIR": tmp}, clear=False):
                self.assertIsNone(record_paper_signal("MNQ=F", "BUY", 100.0, "t1"))
                event = record_paper_signal("MNQ=F", "FLAT", 101.0, "t2")
                status = forward_validation_status(min_trades=50)

        self.assertEqual(event["outcome"], "win")
        self.assertTrue(event["validated"])
        self.assertEqual(status["validated_round_trips"], 1)
        self.assertAlmostEqual(status["expectancy"], 0.01)
        self.assertEqual(status["forward_gate"], "BLOCK")
        self.assertFalse(status["live_eligible"])
        self.assertTrue(status["paper_only"])
        self.assertFalse(status["live_trading"])


if __name__ == "__main__":
    unittest.main()
