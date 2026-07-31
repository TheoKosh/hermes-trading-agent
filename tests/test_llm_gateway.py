from __future__ import annotations

import importlib
import os
import tempfile
import unittest


class TestLLMGateway(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["HERMES_LLM_CACHE_PATH"] = os.path.join(self.tmp.name, "cache.json")
        os.environ["HERMES_LLM_GATEWAY_ENABLED"] = "false"
        os.environ["HERMES_LLM_MONTHLY_BUDGET_USD"] = "0.00"
        import cost_control.llm_gateway as gw
        self.gw = importlib.reload(gw)

    def tearDown(self):
        self.tmp.cleanup()
        for k in ["HERMES_LLM_CACHE_PATH", "HERMES_LLM_GATEWAY_ENABLED", "HERMES_LLM_MONTHLY_BUDGET_USD"]:
            os.environ.pop(k, None)

    def test_disabled_gateway_blocks_uncached_calls(self):
        with self.assertRaises(self.gw.LLMGatewayDisabled):
            self.gw.completion("gpt-4o-mini", [{"role": "user", "content": "hi"}], estimated_cost_usd=0.001)

    def test_zero_budget_blocks_paid_uncached_calls(self):
        os.environ["HERMES_LLM_GATEWAY_ENABLED"] = "true"
        self.gw = importlib.reload(self.gw)
        with self.assertRaises(self.gw.LLMBudgetExceeded):
            self.gw.completion("gpt-4o-mini", [{"role": "user", "content": "hi"}], estimated_cost_usd=0.001)

    def test_cost_estimator(self):
        self.assertAlmostEqual(self.gw.estimate_cost_usd("gpt-4o-mini", 1000, 1000), 0.00075)


if __name__ == "__main__":
    unittest.main()
