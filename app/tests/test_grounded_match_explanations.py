import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from grounded_match_explanations import (
    MockProvider,
    OpenAIProvider,
    ProviderFailure,
    ResponseValidationError,
    explain_match,
)


FIXTURE_PATH = PROJECT_ROOT / "code" / "fixtures" / "grounded_explanation_pair.json"


def load_pair():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def valid_response():
    return {
        "profile_id": "synthetic_customer_support_technical_training",
        "job_id": "synthetic_technical_support_001",
        "evidence_sufficiency": "sufficient",
        "insufficiency_reason": None,
        "transferable_experience": [
            {
                "job_duty": "Triage customer-reported software issues and document reproducible steps.",
                "candidate_evidence_ids": ["candidate.experience[0]"],
                "job_evidence_ids": ["job.description_text[1]"],
                "rationale": "The candidate has issue-triage and escalation experience relevant to this duty.",
            }
        ],
        "qualifications": [
            {
                "qualification_id": "q1",
                "qualification": "Demonstrated experience supporting customers with software products.",
                "classification": "required",
                "status": "supported",
                "candidate_evidence_ids": ["candidate.experience[0]"],
                "job_evidence_ids": ["job.description_text[5]"],
                "rationale": "The profile explicitly describes customer service and account support.",
            },
            {
                "qualification_id": "q2",
                "qualification": "Familiarity with TCP/IP networking.",
                "classification": "required",
                "status": "unknown",
                "candidate_evidence_ids": ["candidate.evidenced_skills[2]"],
                "job_evidence_ids": ["job.description_text[6]"],
                "rationale": "Basic networking coursework is mentioned, but it does not establish TCP/IP familiarity.",
            },
            {
                "qualification_id": "q3",
                "qualification": "Experience with SQL queries.",
                "classification": "preferred",
                "status": "unknown",
                "candidate_evidence_ids": [],
                "job_evidence_ids": ["job.description_text[9]"],
                "rationale": "The profile does not state SQL experience.",
            },
        ],
        "next_steps": [
            {
                "action": "Check whether the networking coursework covered TCP/IP concepts, and add a specific example if it did.",
                "qualification_ids": ["q2"],
                "evidence_ids": ["job.description_text[6]", "candidate.evidenced_skills[2]"],
            },
            {
                "action": "If relevant to your goals, consider a small SQL practice exercise and record what you can demonstrate.",
                "qualification_ids": ["q3"],
                "evidence_ids": ["job.description_text[9]"],
            },
        ],
    }


class GroundedMatchExplanationTests(unittest.TestCase):
    def setUp(self):
        pair = load_pair()
        self.candidate = pair["candidate"]
        self.job = pair["job"]

    def test_fixture_transfer_claim_resolves_to_exact_candidate_and_job_evidence(self):
        result = explain_match(self.candidate, self.job, provider=MockProvider(valid_response()))
        transfer = result["transferable_experience"][0]
        by_id = {item["evidence_id"]: item["text"] for item in result["evidence_catalog"]}
        self.assertIn("issue triage", by_id[transfer["candidate_evidence_ids"][0]])
        self.assertIn("Triage customer-reported", by_id[transfer["job_evidence_ids"][0]])

    def test_unknown_required_and_preferred_qualifications_are_not_gaps(self):
        result = explain_match(self.candidate, self.job, provider=MockProvider(valid_response()))
        qualifications = {q["qualification_id"]: q for q in result["qualifications"]}
        self.assertEqual(qualifications["q2"]["classification"], "required")
        self.assertEqual(qualifications["q2"]["status"], "unknown")
        self.assertEqual(qualifications["q3"]["classification"], "preferred")
        self.assertEqual(qualifications["q3"]["status"], "unknown")
        self.assertEqual(qualifications["q2"]["candidate_evidence_ids"], ["candidate.evidenced_skills[2]"])

    def test_explicit_gap_requires_candidate_evidence_and_keeps_modality(self):
        candidate = dict(self.candidate)
        candidate["evidenced_skills"] = [*candidate["evidenced_skills"], "I have never used SQL"]
        response = valid_response()
        response["qualifications"][-1] = {
            **response["qualifications"][-1],
            "status": "explicit_gap",
            "candidate_evidence_ids": ["candidate.evidenced_skills[4]"],
            "rationale": "The profile explicitly says SQL has not been used.",
        }
        result = explain_match(candidate, self.job, provider=MockProvider(response))
        self.assertEqual(result["qualifications"][-1]["classification"], "preferred")
        self.assertEqual(result["qualifications"][-1]["status"], "explicit_gap")

    def test_next_step_rejects_candidate_evidence_not_linked_to_qualification(self):
        response = valid_response()
        response["next_steps"][0]["evidence_ids"] = ["candidate.experience[0]"]
        with self.assertRaises(ResponseValidationError):
            explain_match(self.candidate, self.job, provider=MockProvider(response))

    def test_unsupported_evidence_reference_is_rejected(self):
        response = valid_response()
        response["transferable_experience"][0]["candidate_evidence_ids"] = ["candidate.fake"]
        with self.assertRaises(ResponseValidationError):
            explain_match(self.candidate, self.job, provider=MockProvider(response))

    def test_response_identity_must_match_selected_pair(self):
        response = valid_response()
        response["job_id"] = "another-job"
        with self.assertRaisesRegex(ResponseValidationError, "identity"):
            explain_match(self.candidate, self.job, provider=MockProvider(response))

    def test_missing_description_returns_insufficient_without_calling_provider(self):
        job = {"job_id": "job-missing", "title": "Support Specialist", "description_text": None}
        provider = MockProvider(valid_response())
        result = explain_match(self.candidate, job, provider=provider)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(result["evidence_sufficiency"], "insufficient")
        self.assertEqual(result["qualifications"], [])
        self.assertIn("null", result["insufficiency_reason"])

    def test_sparse_candidate_returns_insufficient_without_provider_call(self):
        provider = MockProvider(valid_response())
        result = explain_match({"profile_id": "empty"}, self.job, provider=provider)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(result["evidence_sufficiency"], "insufficient")

    def test_provider_failure_propagates_as_safe_feature_error(self):
        def fail(_payload):
            raise TimeoutError("timeout")

        with self.assertRaisesRegex(ProviderFailure, "TimeoutError"):
            explain_match(self.candidate, self.job, provider=MockProvider(fail))

    def test_openai_transport_sends_strict_schema_and_parses_mocked_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"output_text": json.dumps(valid_response())}).encode("utf-8")

        with patch("grounded_match_explanations.urllib.request.urlopen", return_value=Response()) as send:
            result = explain_match(self.candidate, self.job, provider=OpenAIProvider("test-secret", "test-model"))
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(result["job_id"], self.job["job_id"])

    def test_openai_http_failure_does_not_expose_response_body(self):
        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses", 429, "rate limited", {}, None,
        )
        with patch("grounded_match_explanations.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(ProviderFailure, "HTTP 429") as raised:
                OpenAIProvider("test-secret").generate({"candidate": {}, "job": {}})
        self.assertNotIn("test-secret", str(raised.exception))

    def test_next_step_must_reference_unknown_or_explicit_gap(self):
        response = valid_response()
        response["next_steps"][0]["qualification_ids"] = ["q1"]
        response["next_steps"][0]["evidence_ids"] = ["candidate.experience[0]", "job.description_text[5]"]
        with self.assertRaises(ResponseValidationError):
            explain_match(self.candidate, self.job, provider=MockProvider(response))


if __name__ == "__main__":
    unittest.main()
