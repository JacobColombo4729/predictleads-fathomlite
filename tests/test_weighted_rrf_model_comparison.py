import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import evaluate_bge_transition_openai as evaluator
import run_weighted_rrf_model_comparison as comparison


class AlternativeModelPayloadTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "profile_id": "synthetic-dev-profile",
            "job_id": "stable-job-id",
            "candidate_evidence": {"work_history": ["Recorded process checks."]},
            "job_evidence": {"description_text": "Maintain process logs under supervision."},
        }

    def test_gpt41_request_uses_responses_structured_schema_without_reasoning_parameter(self):
        prompt = (comparison.REVISED_DIR / "audit_prompt_frozen.md").read_text(encoding="utf-8")
        with comparison._model_context():
            payload = comparison._payload(self.row, prompt)

        self.assertEqual(payload["model"], "gpt-4.1-2025-04-14")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_output_tokens"], 1800)
        self.assertIs(payload["store"], False)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertNotIn("reasoning", payload)

    def test_request_identity_is_model_specific_and_blinded(self):
        prompt = (comparison.REVISED_DIR / "audit_prompt_frozen.md").read_text(encoding="utf-8")
        with comparison._model_context():
            key = comparison._request_identity(
                self.row, prompt,
                json.loads((comparison.REVISED_DIR / "audit_schema_frozen.json").read_text(encoding="utf-8")),
            )
            payload = comparison._payload(self.row, prompt)

        self.assertEqual(len(key), 64)
        user_content = payload["input"][0]["content"]
        self.assertNotIn("selection_reason", user_content)
        self.assertNotIn("grade", user_content)
        self.assertNotIn("rank", user_content)
        self.assertNotIn("method", user_content)

    def test_gpt41_reservation_includes_two_full_output_caps_at_verified_rates(self):
        per_attempt = evaluator._maximum_request_cost(
            3000, comparison.MAX_OUTPUT_TOKENS,
            comparison.INPUT_USD_PER_MILLION, comparison.OUTPUT_USD_PER_MILLION,
        )
        reservation = per_attempt * comparison.MAX_ATTEMPTS

        self.assertAlmostEqual(per_attempt, (3000 * 2.0 + 1800 * 8.0) / 1_000_000)
        self.assertAlmostEqual(reservation, per_attempt * 2)
        self.assertGreater(reservation, evaluator._maximum_request_cost(3000, 1800) * 2)

    def test_pinned_evaluator_model_and_prices_remain_unchanged(self):
        self.assertEqual(evaluator.MODEL, "gpt-4.1-mini-2025-04-14")
        self.assertEqual(evaluator.INPUT_USD_PER_MILLION, 0.40)
        self.assertEqual(evaluator.OUTPUT_USD_PER_MILLION, 1.60)


if __name__ == "__main__":
    unittest.main()
