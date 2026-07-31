from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd


class TestMultiAssetScanner(unittest.TestCase):
    def _bars(self):
        n = 80
        close = [100 + i * 0.2 for i in range(n)]
        return pd.DataFrame({
            "timestamp": list(range(n)),
            "open": close,
            "high": [x + 1 for x in close],
            "low": [x - 1 for x in close],
            "close": close,
            "vwap_raw": close,
            "volume": [1000 + i for i in range(n)],
            "count": [1] * n,
        })

    def test_assets_have_independent_multi_feature_paper_scan(self):
        from strategies.multi_asset_scanner import SUPPORTED_ASSETS, scan_asset

        with patch("strategies.multi_asset_scanner._fetch", return_value=self._bars()):
            results = [scan_asset(asset) for asset in SUPPORTED_ASSETS]

        self.assertEqual(tuple(x["asset"] for x in results), SUPPORTED_ASSETS)
        self.assertTrue(all(x["mode"] == "paper_only" for x in results))
        self.assertTrue(all(x["model"]["status"] == "not_applied" for x in results))
        self.assertTrue(all(len(x["datapoints"]) >= 10 for x in results))
        self.assertTrue(all("rsi" in x["features"] and "volume_ratio" in x["features"] for x in results))

    def test_unknown_asset_rejected(self):
        from strategies.multi_asset_scanner import scan_asset

        with self.assertRaises(ValueError):
            scan_asset("DOGEUSD")


if __name__ == "__main__":
    unittest.main()
