from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestFuturesTraining(unittest.TestCase):
    def test_futures_candidate_is_asset_specific_and_paper_only(self):
        from ml import train_futures_meta as module

        idx = pd.date_range("2026-01-01", periods=360, freq="15min", tz="UTC")
        close = 100 + np.cumsum(np.sin(np.arange(360) / 7) * 0.2 + 0.05)
        frame = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": np.full(360, 1000.0)}, index=idx)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"STATE_DIR": tmp, "FUTURES_MODEL_DIR": str(Path(tmp) / "models"), "AUDIT_DIR": str(Path(tmp) / "backtests")}, clear=False), patch.object(module, "_frame", return_value=frame):
                result = module._train_symbol("MNQ=F")

            self.assertEqual(result["status"], "complete")
            manifest = result["manifest"]
            self.assertEqual(manifest["symbol"], "MNQ=F")
            self.assertEqual(manifest["asset_class"], "futures")
            self.assertTrue(Path(manifest["path"]).exists())
            self.assertTrue((Path(tmp) / "models" / f"{manifest['version']}.json").exists())


if __name__ == "__main__":
    unittest.main()
