import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from requirement_aware_reranker import (
    AssessmentValidationError,
    ProviderConfig,
    ProviderUnavailable,
    ordering_key,
    validate_assessment,
    verify_provider,
)


class RequirementAwareRerankerTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "profile_id": "synthetic_1",
            "work_history": ["Inspected assemblies before release."],
            "hard_constraints": {"schedule": "day shift only"},
        }
        self.job = {
            "job_id": "job_1",
            "title": "Quality Technician",
            "description": "Must hold an active RN license. Experience with ISO 13485 preferred.",
        }
        self.valid = {
            "profile_id": "synthetic_1",
            "job_id": "job_1",
            "duty_transfer_score": 2,
            "duty_transfer_rationale": "Inspection experience transfers to quality work.",
            "duty_evidence": [{
                "job_excerpt": "Quality Technician",
                "candidate_excerpt": "Inspected assemblies before release.",
                "rationale": "Both concern inspection before release.",
            }],
            "requirements": [
                {"job_excerpt": "Must hold an active RN license.", "classification": "mandatory",
                 "status": "unknown", "candidate_excerpt": None,
                 "rationale": "No license evidence is provided."},
                {"job_excerpt": "Experience with ISO 13485 preferred.", "classification": "preferred",
                 "status": "explicitly_contradicted", "candidate_excerpt": "Inspected assemblies before release.",
                 "rationale": "This candidate evidence does not document that preference."},
            ],
            "constraints": [],
            "unknowns": ["RN license status is not stated."],
            "evidence_sufficiency": "sufficient",
        }

    def test_missing_mandatory_qualification_is_unknown_not_conflict(self):
        validated = validate_assessment(self.candidate, self.job, self.valid)
        key = ordering_key(validated, 3, "job_1")
        self.assertEqual(key[1], 0)
        self.assertEqual(key[0], 0)

    def test_only_explicit_mandatory_contradiction_counts(self):
        assessment = dict(self.valid)
        assessment["requirements"] = [
            {"job_excerpt": "Must hold an active RN license.", "classification": "mandatory",
             "status": "explicitly_contradicted", "candidate_excerpt": "Inspected assemblies before release.",
             "rationale": "The candidate explicitly states they do not hold the required license."}
        ]
        validated = validate_assessment(self.candidate, self.job, assessment)
        self.assertEqual(ordering_key(validated, 1, "job_1")[1], 1)

    def test_contradiction_requires_candidate_evidence(self):
        assessment = dict(self.valid)
        assessment["requirements"] = [dict(self.valid["requirements"][0], status="explicitly_contradicted")]
        with self.assertRaises(AssessmentValidationError):
            validate_assessment(self.candidate, self.job, assessment)

    def test_unquoted_or_fabricated_evidence_is_rejected(self):
        assessment = dict(self.valid)
        assessment["duty_evidence"] = [dict(self.valid["duty_evidence"][0],
                                             candidate_excerpt="Managed a clinical team.")]
        with self.assertRaises(AssessmentValidationError):
            validate_assessment(self.candidate, self.job, assessment)

    def test_unknowns_do_not_demote_and_profile_ranks_do_not_collide(self):
        base = {"duty_transfer_score": 2, "constraints": [], "requirements": []}
        with_unknown = dict(base, requirements=[{
            "classification": "mandatory", "status": "unknown"}])
        supported = dict(base, requirements=[{
            "classification": "mandatory", "status": "supported"}])
        self.assertEqual(ordering_key(supported, 1, "a_job"),
                         ordering_key(with_unknown, 1, "a_job"))
        self.assertLess(ordering_key(base, 1, "a_job"), ordering_key(base, 2, "a_job"))
        self.assertLess(ordering_key(base, 1, "a_job"), ordering_key(base, 1, "b_job"))

    def test_provider_requires_immutable_identity(self):
        with self.assertRaises(ProviderUnavailable):
            ProviderConfig.from_environment({})
        config = ProviderConfig.from_environment({
            "PREDICTLEADS_RERANK_BASE_URL": "https://example.invalid/v1",
            "PREDICTLEADS_RERANK_MODEL": "deployment-test",
            "PREDICTLEADS_RERANK_REVISION": "revision-abc",
        })
        self.assertEqual(config.completion_url, "https://example.invalid/v1/chat/completions")

    def test_provider_preflight_requires_matching_served_model(self):
        config = ProviderConfig("https://example.invalid/v1", "deployment-test", "revision-abc")
        assessment = {"profile_id": "provider_preflight", "job_id": "provider_preflight",
                      "duty_transfer_score": 1, "duty_transfer_rationale": "Probe.",
                      "duty_evidence": [], "requirements": [], "constraints": [],
                      "unknowns": [], "evidence_sufficiency": "insufficient"}

        class Response:
            headers = {"x-request-id": "request-1"}
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"model": "unexpected-model", "choices": [{"message": {
                    "content": json.dumps(assessment)}}]}

        with patch("requirement_aware_reranker.requests.post", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "returned model"):
                verify_provider(config)


if __name__ == "__main__":
    unittest.main()
