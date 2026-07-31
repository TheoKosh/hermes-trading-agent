from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMLAIAdvisor(unittest.TestCase):
    def test_local_conclusion_is_zero_token_and_paper_only_on_bad_model(self):
        from integrations.ml_ai_advisor import status

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MODEL_DIR": tmp}, clear=False):
                result = status()

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["communication"]["llm_requests"], 0)
        self.assertEqual(result["communication"]["tokens_consumed"], 0)

    def test_good_model_still_requires_paper_review(self):
        from integrations.ml_ai_advisor import status

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lgbm_meta_v1.json"
            path.write_text(json.dumps({"version": "v1", "metrics": {"precision": 0.70, "auc": 0.75}}))
            with patch.dict(os.environ, {"MODEL_DIR": tmp}, clear=False):
                result = status()

        self.assertEqual(result["status"], "eligible_for_paper_review")
        self.assertFalse(result["safety"]["live_mutation"])
        self.assertTrue(result["safety"]["human_approval_required"])

    def test_ai_bridge_is_disabled_without_explicit_opt_in(self):
        from integrations.ml_ai_advisor import ai_conclusion

        with patch.dict(os.environ, {"HERMES_ML_AI_ENABLED": "false"}, clear=False):
            result = ai_conclusion()

        self.assertEqual(result["ai"]["status"], "disabled")
        self.assertEqual(result["ai"]["requests"], 0)

    def test_distilled_advisor_logic_is_financially_fail_closed(self):
        from integrations.ml_ai_advisor import ADVISOR_SYSTEM_PROMPT

        self.assertIn("paper-trading ML reviewer", ADVISOR_SYSTEM_PROMPT)
        self.assertIn("positive expectancy", ADVISOR_SYSTEM_PROMPT)
        self.assertIn("human approval", ADVISOR_SYSTEM_PROMPT)
        self.assertIn("never invent data", ADVISOR_SYSTEM_PROMPT)

    def test_paid_model_is_blocked_in_free_only_mode(self):
        from integrations.ml_ai_advisor import ai_conclusion

        with patch.dict(os.environ, {"HERMES_ML_AI_ENABLED": "true", "HERMES_ML_AI_FREE_ONLY": "true", "HERMES_ML_AI_MODEL": "gpt-4o"}, clear=False):
            result = ai_conclusion()

        self.assertEqual(result["ai"]["status"], "blocked_paid_model")
        self.assertEqual(result["ai"]["requests"], 0)


if __name__ == "__main__":
    unittest.main()
