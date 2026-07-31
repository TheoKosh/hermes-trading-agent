from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestSelfEvolutionAdapter(unittest.TestCase):
    def test_status_is_offline_pr_only_by_default(self):
        from integrations.self_evolution_adapter import status

        result = status()

        self.assertFalse(result["enabled"])
        self.assertTrue(result["approval_required"])
        self.assertEqual(result["mode"], "paper_candidate_evolution")
        self.assertTrue(result["approval_required"])
        self.assertFalse(result["live_mutation"])
        self.assertFalse(result["live_trading_access"])
        self.assertIn("full test suite before candidate acceptance", result["guardrails"])

    def test_dry_run_rejects_path_traversal(self):
        from integrations.self_evolution_adapter import dry_run

        with self.assertRaises(ValueError):
            dry_run("../../live_trading")

    def test_staged_mutation_is_logged_and_never_applied(self):
        from integrations.self_evolution_adapter import mutation_log, record_staged_mutation

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_SELF_EVOLUTION_OUTPUT": tmp}, clear=False):
                record = record_staged_mutation("skills/example", "candidate-1", "paper improvement", {"auc": 0.7})
                log = mutation_log()

        self.assertFalse(record["applied"])
        self.assertFalse(record["live_mutation"])
        self.assertEqual(log["count"], 1)
        self.assertEqual(log["records"][0]["stage"], "candidate_pending_human_review")

    def test_paper_candidate_can_mutate_isolated_workspace_only(self):
        from integrations.self_evolution_adapter import apply_paper_candidate

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_SELF_EVOLUTION_OUTPUT": tmp, "HERMES_SELF_EVOLUTION_PAPER_MUTATION": "true"}, clear=False):
                record = apply_paper_candidate("skills/example.md", "candidate", {"auc": 0.7})

            self.assertTrue(record["applied"])
            self.assertTrue(record["paper_only"])
            self.assertFalse(record["live_mutation"])
            self.assertEqual(Path(record["path"]).read_text(), "candidate")


if __name__ == "__main__":
    unittest.main()
