from __future__ import annotations

import os
import unittest


class TestFreeClaudeCodeAdapter(unittest.TestCase):
    def setUp(self):
        self.old = {k: os.environ.get(k) for k in ("HERMES_FCC_ENABLED", "HERMES_FCC_BASE_URL", "HERMES_FCC_MAX_OUTPUT_TOKENS", "HERMES_FCC_REASONING")}
        for key in self.old:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_disabled_by_default(self):
        from integrations.free_claude_code_adapter import status

        result = status()

        self.assertFalse(result["enabled"])
        self.assertFalse(result["configured"])
        self.assertFalse(result["production_installed"])
        self.assertEqual(result["max_output_tokens"], 256)

    def test_bounded_reasoning_is_explicitly_opt_in(self):
        from integrations.free_claude_code_adapter import status

        os.environ["HERMES_FCC_REASONING"] = "true"
        result = status()

        self.assertEqual(result["reasoning"], "enabled_bounded")

    def test_compact_messages_keeps_system_and_newest_messages(self):
        from integrations.free_claude_code_adapter import compact_messages

        messages = [{"role": "system", "content": "system"}] + [
            {"role": "user", "content": "x" * 100} for _ in range(200)
        ]
        compacted = compact_messages(messages, max_chars=350)

        self.assertEqual(compacted[0]["role"], "system")
        self.assertLessEqual(sum(len(str(m["content"])) for m in compacted), 350)
        self.assertEqual(compacted[-1]["content"], "x" * 100)


if __name__ == "__main__":
    unittest.main()
