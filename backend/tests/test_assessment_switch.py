"""The AI gap analysis sends text to OpenAI, so it must be off unless explicitly switched on."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from backend import api


class AssessmentSwitchTests(unittest.TestCase):
    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREDICTLEADS_ASSESSMENT", None)
            self.assertIsNone(api._default_assessment_provider())

    def test_other_values_do_not_turn_it_on(self) -> None:
        with mock.patch.dict(os.environ, {"PREDICTLEADS_ASSESSMENT": "yes"}):
            self.assertIsNone(api._default_assessment_provider())

    def test_openai_value_builds_the_provider(self) -> None:
        with mock.patch.dict(os.environ, {"PREDICTLEADS_ASSESSMENT": "openai", "OPENAI_API_KEY": "sk-test-not-real"}):
            provider = api._default_assessment_provider()
            self.assertEqual(provider.model, "gpt-4.1-mini")


if __name__ == "__main__":
    unittest.main()
