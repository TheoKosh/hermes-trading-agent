from __future__ import annotations

import os
import unittest


class TestTradingAgentsAdapter(unittest.TestCase):
    def setUp(self):
        self.keys = [
            "TRADINGAGENTS_ENABLED",
            "TRADINGAGENTS_LLM_PROVIDER",
            "TRADINGAGENTS_DEEP_THINK_LLM",
            "TRADINGAGENTS_QUICK_THINK_LLM",
            "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
            "TRADINGAGENTS_MAX_RISK_ROUNDS",
            "TRADINGAGENTS_NEWS_LIMIT",
            "TRADINGAGENTS_LLM_MAX_RETRIES",
        ]
        self.old = {k: os.environ.get(k) for k in self.keys}

    def tearDown(self):
        for k, value in self.old.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value

    def test_disabled_analysis_is_paper_only_and_makes_no_import_call(self):
        from integrations.tradingagents_adapter import analyze

        os.environ["TRADINGAGENTS_ENABLED"] = "false"
        result = analyze("LINK-USD")

        self.assertEqual(result["status"], "disabled")
        self.assertTrue(result["analysis_only"])

    def test_token_saving_defaults_are_bounded(self):
        from integrations.tradingagents_adapter import token_saving_config

        for key in self.keys:
            os.environ.pop(key, None)
        config = token_saving_config()

        self.assertEqual(config["quick_think_llm"], "gpt-4o-mini")
        self.assertEqual(config["deep_think_llm"], "gpt-4o-mini")
        self.assertEqual(config["max_debate_rounds"], 0)
        self.assertEqual(config["max_risk_discuss_rounds"], 0)
        self.assertEqual(config["news_article_limit"], 5)
        self.assertEqual(config["llm_max_retries"], 0)
        self.assertFalse(config["checkpoint_enabled"])

    def test_invalid_symbol_is_rejected_before_upstream(self):
        from integrations.tradingagents_adapter import analyze

        os.environ["TRADINGAGENTS_ENABLED"] = "true"
        with self.assertRaises(ValueError):
            analyze("../../secret")


if __name__ == "__main__":
    unittest.main()
