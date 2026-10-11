import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from run_weighted_rrf_grader_audit import (
    REVISED_AUDIT_PROMPT,
    REVISED_AUDIT_SCHEMA,
    _payload,
    _revised_audit_configuration_for_freeze,
    _semantic_concerns,
    _validate_audit,
)


class WeightedRrfGraderAuditValidationTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "profile_id": "profile-a",
            "job_id": "job-a",
            "candidate_evidence": {
                "work_history": ["Recorded process checks and escalated exceptions."],
                "qualification_disclosures": {
                    "unknown": ["Wastewater process experience"],
                    "explicit_absence": ["Drinking-water operator license"],
                },
            },
            "job_evidence": {
                "description_text": (
                    "Wastewater Operator in Training. Under supervision, prepare for entry-level certification. "
                    "Obtain and maintain a Florida Class C Wastewater Operator License within one year."
                ),
            },
        }

    def audit(self, qualification):
        return {
            "profile_id": "profile-a",
            "job_id": "job-a",
            "grade": 2,
            "status": "judged",
            "unknown_reason": None,
            "supporting_candidate_evidence_ids": ["C0001"],
            "supporting_job_evidence_ids": ["J0001"],
            "transferable_duty_findings": [],
            "domain_difference_findings": [],
            "qualification_findings": [qualification],
            "material_gaps": [],
            "explicit_constraint_conflicts": [],
            "required_qualifications": "unknown",
            "career_transition_relevance": {"level": "moderate", "rationale": "Process monitoring transfers."},
            "title_outside_stated_targets": False,
            "evidence_sufficiency": "sufficient",
        }

    def qualification(self, *, classification="mandatory_at_entry", status="unknown",
                      candidate_ids=None, job_ids=None, name="Wastewater Operator License"):
        return {
            "qualification": name,
            "job_classification": classification,
            "candidate_status": status,
            "candidate_evidence_ids": [] if candidate_ids is None else candidate_ids,
            "job_evidence_ids": ["J0001"] if job_ids is None else job_ids,
            "analysis": "Candidate status is assessed only from cited evidence.",
        }

    def test_silent_unknown_qualification_needs_job_but_not_candidate_citation(self):
        raw = self.audit(self.qualification(candidate_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)

        self.assertEqual(errors, [])
        self.assertIsNotNone(normalized)

    def test_supported_qualification_requires_candidate_evidence(self):
        raw = self.audit(self.qualification(status="supported", candidate_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)

        self.assertIsNone(normalized)
        self.assertIn("supported_qualification_requires_candidate_evidence", errors)

    def test_contradiction_requires_candidate_and_job_evidence(self):
        raw = self.audit(self.qualification(status="explicitly_contradicted", candidate_ids=["C0003"], job_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)

        self.assertIsNone(normalized)
        self.assertIn("qualification_requires_job_evidence", errors)
        self.assertIn("contradicted_qualification_requires_job_evidence", errors)

    def test_unknown_qualification_still_requires_job_evidence(self):
        raw = self.audit(self.qualification(status="unknown", candidate_ids=[], job_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)

        self.assertIsNone(normalized)
        self.assertIn("qualification_requires_job_evidence", errors)

    def test_different_water_credentials_flag_semantic_concern_not_mechanical_failure(self):
        raw = self.audit(self.qualification(status="explicitly_contradicted", candidate_ids=["C0003"]))

        normalized, errors = _validate_audit(raw, self.row)
        concerns = _semantic_concerns(normalized or {}, self.row)

        self.assertEqual(errors, [])
        self.assertEqual(len(concerns), 2)
        self.assertIn("unsupported_drinking_water_wastewater_credential_equivalence",
                      {item["code"] for item in concerns})

    def test_wastewater_license_absence_in_material_gap_is_not_inferred_from_silence(self):
        raw = self.audit(self.qualification(status="unknown", candidate_ids=[]))
        raw["material_gaps"] = ["Candidate lacks a wastewater operator license."]

        normalized, errors = _validate_audit(raw, self.row)
        concerns = _semantic_concerns(normalized or {}, self.row)

        self.assertEqual(errors, [])
        self.assertIn("unsupported_wastewater_credential_absence_claim",
                      {item["code"] for item in concerns})

    def test_one_year_operator_training_flags_entry_timing_for_review(self):
        raw = self.audit(self.qualification(status="unknown", candidate_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)
        concerns = _semantic_concerns(normalized or {}, self.row)

        self.assertEqual(errors, [])
        self.assertIn("post_entry_credential_timing_classified_as_entry_mandatory",
                      {item["code"] for item in concerns})

    def test_explicit_post_joining_classification_is_not_recast_as_entry_requirement(self):
        raw = self.audit(self.qualification(classification="obtainable_after_joining", status="unknown", candidate_ids=[]))

        normalized, errors = _validate_audit(raw, self.row)
        concerns = _semantic_concerns(normalized or {}, self.row)

        self.assertEqual(errors, [])
        self.assertNotIn("post_entry_credential_timing_classified_as_entry_mandatory",
                         {item["code"] for item in concerns})

    def test_revised_prompt_separates_duties_eligibility_and_unknowns(self):
        prompt = REVISED_AUDIT_PROMPT.casefold()

        self.assertIn("duties fit", prompt)
        self.assertIn("qualification eligibility", prompt)
        self.assertIn("silence in the candidate profile always means `unknown`", prompt)
        self.assertIn("different credentials are not equivalent", prompt)
        self.assertIn("a deadline or acquisition window alone does not establish", prompt)
        self.assertIn("specific source evidence ids", prompt)
        self.assertIn("original definitions", prompt)
        self.assertNotIn("8aceeb9d-e12a-47a9-9427-653ad8c7ffa1", prompt)

    def test_revised_schema_has_explicit_entry_and_after_joining_classes(self):
        classes = set(REVISED_AUDIT_SCHEMA["$defs"]["qualification_finding"]["properties"]["job_classification"]["enum"])

        self.assertEqual(classes, {
            "required_at_entry", "explicitly_obtainable_after_joining", "preferred", "unclear", "not_stated",
        })

    def test_revised_protocol_accepts_unknown_candidate_with_specific_job_citation(self):
        raw = self.audit(self.qualification(
            classification="explicitly_obtainable_after_joining", status="unknown", candidate_ids=[]))

        with _revised_audit_configuration_for_freeze(REVISED_AUDIT_PROMPT):
            normalized, errors = _validate_audit(raw, self.row)

        self.assertEqual(errors, [])
        self.assertIsNotNone(normalized)

    def test_revised_payload_excludes_selection_and_prior_result_fields(self):
        payload = _payload(self.row, REVISED_AUDIT_PROMPT)
        user_input = payload["input"][0]["content"]

        self.assertNotIn("selection_reason", user_input)
        self.assertNotIn("original_grade", user_input)
        self.assertNotIn("method", user_input)
        self.assertNotIn("rank", user_input)
        self.assertNotIn("score", user_input)


if __name__ == "__main__":
    unittest.main()