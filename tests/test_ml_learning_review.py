from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class TestMLLearningReview(unittest.TestCase):
    def test_disabled_review_makes_no_ai_request(self):
        from integrations.ml_learning_review import review_learning

        with patch.dict(os.environ, {"HERMES_ML_AI_ENABLED": "false"}, clear=False):
            result = review_learning()

        self.assertEqual(result["ai_status"], "disabled")
        self.assertEqual(result["ai_requests"], 0)
        self.assertFalse(result["live_mutation"])

    def test_paper_ai_review_logs_candidate_only(self):
        import integrations.ml_learning_review as module

        fake = {"usage": {"total_tokens": 12}, "choices": [{"message": {"content": "candidate adjustment"}}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_ML_AI_ENABLED": "true", "HERMES_ML_AI_FREE_ONLY": "true", "HERMES_ML_AI_MODEL": "openrouter/free", "HERMES_SELF_EVOLUTION_OUTPUT": tmp}, clear=False), patch.object(module, "fcc_status", return_value={"enabled": True}), patch.object(module, "completion", return_value=fake):
                result = module.review_learning()

        self.assertEqual(result["ai_status"], "complete")
        self.assertEqual(result["ai_requests"], 1)
        self.assertEqual(result["tokens"], 12)
        self.assertTrue(result["candidate_logged"])
        self.assertEqual(result["mutation_stage"], "candidate_pending_human_review")

    def test_automatic_review_has_interval_and_daily_cap(self):
        import integrations.ml_learning_review as module

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_ML_AI_AUTO_REVIEW": "true", "HERMES_ML_AI_AUTO_INTERVAL_SECONDS": "300", "HERMES_ML_AI_AUTO_DAILY_CAP": "1", "STATE_DIR": tmp}, clear=False), patch.object(module, "review_learning", return_value={"ai_requests": 1, "ai_status": "complete"}):
                first = module.automatic_review_once()
                second = module.automatic_review_once()

        self.assertEqual(first["ai_status"], "complete")
        self.assertEqual(second["status"], "not_due")
        self.assertEqual(second["requests_today"], 1)

    def test_automatic_status_is_observable_without_calling_ai(self):
        import integrations.ml_learning_review as module

        with patch.dict(os.environ, {"HERMES_ML_AI_AUTO_REVIEW": "true", "HERMES_ML_AI_AUTO_INTERVAL_SECONDS": "1800", "HERMES_ML_AI_AUTO_DAILY_CAP": "8"}, clear=False):
            result = module.automatic_review_status()

        self.assertTrue(result["automatic"])
        self.assertEqual(result["interval_seconds"], 1800)
        self.assertEqual(result["daily_cap"], 8)
        self.assertTrue(result["candidate_only"])
        self.assertFalse(result["live_mutation"])


if __name__ == "__main__":
    unittest.main()
