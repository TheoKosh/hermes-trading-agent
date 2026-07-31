from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestFuturesLiveMonitor(unittest.TestCase):
    def test_live_market_data_is_paper_position_only(self):
        import integrations.futures_live_monitor as module

        candles = [{"t": f"2026-01-01T{i:02d}:00:00+00:00", "o": 100 + i, "h": 101 + i, "l": 99 + i, "c": 100 + i, "v": 1000.0} for i in range(60)]
        module._CACHE.update({"updated_at": 0.0, "payload": None})
        with patch.dict(os.environ, {"HERMES_FUTURES_LIVE_CACHE_SECONDS": "30"}, clear=False), patch.object(module, "_candles", return_value=candles):
            result = module.futures_live_paper_positions()

        self.assertEqual(len(result["positions"]), 3)
        self.assertFalse(result["broker_positions"])
        self.assertTrue(result["paper_only"])
        self.assertFalse(result["live_trading"])
        self.assertEqual(result["positions"][0]["regime"], "TRENDING_UP")
        self.assertIn(result["positions"][0]["paper_position"], {"BUY", "SELL", "FLAT"})


if __name__ == "__main__":
    unittest.main()
