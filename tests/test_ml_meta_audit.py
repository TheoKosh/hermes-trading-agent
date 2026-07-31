from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestMLMetaAudit(unittest.TestCase):
    def _df(self, n=80):
        return pd.DataFrame({
            "open": np.linspace(10, 12, n),
            "high": np.linspace(10.2, 12.2, n),
            "low": np.linspace(9.8, 11.8, n),
            "close": np.linspace(10, 12, n),
            "volume": np.full(n, 100.0),
        })

    def test_dataset_drops_rows_without_future_label(self):
        from strategies.ml_meta import _dataset

        holding = 3
        data = _dataset(self._df(), holding_bars=holding)

        self.assertEqual(len(data), 80 - 19 - holding)
        self.assertFalse(data.index.isin(range(80 - holding, 80)).any())

    def test_walk_forward_mean_auc_none_when_no_auc_values(self):
        import strategies.ml_meta as ml

        rows = [
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None},
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None},
        ]
        summary = ml._summarize_folds(rows)

        self.assertIsNone(summary["mean_auc"])


if __name__ == "__main__":
    unittest.main()
