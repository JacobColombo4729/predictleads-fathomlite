import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import run_weighted_rrf_qualification_protocol as protocol


class QualificationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.raw_row = {
            "profile_id": "synthetic_case",
            "job_id": "job_case",
            "candidate_evidence": {
                "work_history": ["Recorded process readings and reported exceptions."],
                "education_certifications": ["Associate degree in applied science"],
                "qualification_disclosures": {"explicit_absence": ["Drinking-water operator license"]},
            },
            "job_evidence": {
                "description_text": "Must hold a wastewater operator license at entry.\nObtain a class B credential within one year after joining.",
            },
        }
        self.row = protocol.normalize_row(self.raw_row)

    def qualification(self, name, classification="mandatory", timing="at_entry", status="unknown"):
        return {"qualification": name, "classification": classification, "timing": timing,
                "candidate_status": status}

    def test_candidate_silence_stays_unknown_and_does_not_make_conflict(self):
        result = protocol.calculate_eligibility(
            [self.qualification("Wastewater operator license")], {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "unknown")
        self.assertEqual(result["unknown_entry_requirements"], ["Wastewater operator license"])
        self.assertEqual(result["entry_conflicts"], [])

    def test_different_credentials_do_not_establish_contradiction(self):
        judgment = {"qualifications": [{
            "qualification": "Wastewater operator license",
            "candidate_status": "explicitly_contradicted",
            "candidate_evidence_ids": ["C0003"],
        }]}
        concerns = protocol.semantic_concerns(judgment, self.row)
        self.assertIn("cross_credential_evidence_does_not_support_wastewater_finding",
                  [item["code"] for item in concerns])

    def test_post_entry_deadline_is_not_an_entry_conflict(self):
        result = protocol.calculate_eligibility(
            [self.qualification("Class B credential", timing="explicitly_obtainable_after_joining",
                                status="explicitly_contradicted")], {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "unknown")
        self.assertEqual(result["post_entry_obligations"], ["Class B credential"])
        self.assertEqual(result["entry_conflicts"], [])

    def test_mixed_supported_and_unknown_entry_requirements_are_unknown(self):
        result = protocol.calculate_eligibility([
            self.qualification("High school diploma", status="supported"),
            self.qualification("Driver license", status="unknown"),
        ], {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "unknown")

    def test_explicit_mandatory_entry_contradiction_is_conflict(self):
        result = protocol.calculate_eligibility(
            [self.qualification("Required license", status="explicitly_contradicted")],
            {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "entry_conflict")
        self.assertEqual(result["entry_conflicts"], ["Required license"])

    def test_unsupported_contradiction_claim_is_unknown_in_python_summary(self):
        result = protocol.calculate_eligibility(
            [self.qualification("Logistics supervisory experience", status="explicitly_contradicted")],
            {"status": "sufficient"},
            [{"code": "contradiction_not_explicit_in_candidate_source",
              "qualification": "Logistics supervisory experience"}],
        )
        self.assertEqual(result["entry_status"], "unknown")
        self.assertEqual(result["entry_conflicts"], [])
        self.assertEqual(result["unknown_entry_requirements"], ["Logistics supervisory experience"])

    def test_missing_mandatory_requirements_never_establish_eligibility(self):
        result = protocol.calculate_eligibility(
            [self.qualification("Preferred ERP experience", "preferred", "unclear", "supported")],
            {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "unknown")

    def test_insufficient_coverage_prevents_supported_summary(self):
        result = protocol.calculate_eligibility(
            [self.qualification("High school diploma", status="supported")],
            {"status": "unclear"})
        self.assertEqual(result["entry_status"], "unknown")

    def test_unclear_requirement_classification_prevents_supported_summary(self):
        result = protocol.calculate_eligibility([
            self.qualification("Possible license", "unclear", "at_entry", "supported")],
            {"status": "sufficient"})
        self.assertEqual(result["entry_status"], "unknown")
        self.assertEqual(result["unresolved_job_classification"], ["Possible license"])

    def test_precise_excerpt_and_location_are_validated_and_context_returned(self):
        raw = self.valid_response()
        validated, errors = protocol._validate(raw, self.row)
        self.assertEqual(errors, [])
        finding = validated["qualifications"][0]
        self.assertEqual(finding["job_source_location"], "job_evidence/full_description:1-1")
        self.assertEqual(finding["job_surrounding_context"], "Must hold a wastewater operator license at entry.\nObtain a class B credential within one year after joining.")
        raw["qualifications"][0]["job_excerpt"] = "Must hold a wastewater license."
        _, errors = protocol._validate(raw, self.row)
        self.assertIn("job_excerpt_location_mismatch:0", errors)

    def test_deadline_excerpt_must_be_exact_source_text(self):
        raw = self.valid_response()
        raw["qualifications"][0]["deadline_excerpt"] = "within two years"
        _, errors = protocol._validate(raw, self.row)
        self.assertIn("deadline_not_exact_job_excerpt:0", errors)

    def test_unknown_gap_and_rationale_contradictions_are_semantic_concerns(self):
        judgment = {
            "qualifications": [self.qualification("Wastewater operator license", status="unknown")],
            "material_gaps": [{"qualification": "Wastewater operator license", "claim": "Candidate lacks the license.",
                               "candidate_evidence_ids": [], "job_evidence_ids": ["JLINE0001"]}],
            "explicit_constraint_conflicts": [],
            "career_transition_relevance": {"level": "moderate", "rationale": "Candidate lacks the wastewater operator license."},
        }
        concerns = protocol.semantic_concerns(judgment, self.row)
        codes = {item["code"] for item in concerns}
        self.assertIn("unsupported_material_gaps_claim", codes)
        self.assertIn("rationale_overstates_candidate_qualification_absence", codes)
        self.assertTrue(all(item["blocks_grade_aggregate"] for item in concerns))

    def test_contradicted_status_needs_direct_negative_candidate_text(self):
        candidate_ids = [item["evidence_id"] for item in protocol._candidate_catalog(self.row["candidate_evidence"])]
        judgment = {"qualifications": [{
            "qualification": "Wastewater operator license", "candidate_status": "explicitly_contradicted",
            "candidate_evidence_ids": [candidate_ids[0]],
        }]}
        codes = {item["code"] for item in protocol.semantic_concerns(judgment, self.row)}
        self.assertIn("contradiction_not_explicit_in_candidate_source", codes)

    def test_wastewater_case_is_complete_and_not_duplicated(self):
        frozen = PROJECT_ROOT.parents[1] / "data/matching/weighted_rrf_20261008T233000Z/diagnostics/dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009/revised_protocol_v2_20261009/reviewer_input_blinded.jsonl"
        rows = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines() if line.strip()]
        case = next(row for row in rows if row["profile_id"] == "dev_brewery_cellar_to_water_operations"
                    and row["job_id"] == "8aceeb9d-e12a-47a9-9427-653ad8c7ffa1")
        normalized = protocol.normalize_row(case)
        job = normalized["job_evidence"]
        self.assertEqual(len(job["full_description"]), 4351)
        self.assertIn("Possess a valid driver", job["full_description"])
        self.assertIn("Florida Class C Wastewater Operator License within one (1) year", job["full_description"])
        self.assertEqual(sum(value == job["full_description"] for value in job.values()), 1)
        self.assertNotIn("explicit_requirements_and_duties_as_supplied", job)
        self.assertNotIn("full_prepared_description", job)
        self.assertEqual(len(job["description_line_evidence"]), 39)

    def valid_response(self):
        candidate_ids = [item["evidence_id"] for item in protocol._candidate_catalog(self.row["candidate_evidence"])]
        first_line_id = self.row["job_evidence"]["description_line_evidence"][0]["evidence_id"]
        second_line_id = self.row["job_evidence"]["description_line_evidence"][1]["evidence_id"]
        return {
            "profile_id": self.row["profile_id"], "job_id": self.row["job_id"],
            "grade": 2, "status": "judged", "unknown_reason": None,
            "supporting_candidate_evidence_ids": [candidate_ids[0]],
            "supporting_job_evidence_ids": [first_line_id],
            "transferable_duty_findings": [{"finding": "Checks transfer.", "candidate_evidence_ids": [candidate_ids[0]],
                                             "job_evidence_ids": [first_line_id]}],
            "domain_difference_findings": [],
            "qualifications": [{
                "qualification": "Wastewater operator license",
                "job_excerpt": "Must hold a wastewater operator license at entry.",
                "job_line_start": 1, "job_line_end": 1, "job_evidence_ids": [first_line_id],
                "classification": "mandatory", "timing": "at_entry", "deadline_excerpt": None,
                "candidate_status": "unknown", "candidate_excerpt": None, "candidate_evidence_ids": [],
                "explanation": "Candidate profile does not state license status.",
            }],
            "extraction_coverage": {"status": "sufficient", "rationale": "All supplied lines were reviewed.",
                                    "job_evidence_ids": [first_line_id, second_line_id]},
            "material_gaps": [], "explicit_constraint_conflicts": [],
            "career_transition_relevance": {"level": "moderate", "rationale": "Some process checks transfer."},
            "title_outside_stated_targets": False, "evidence_sufficiency": "sufficient",
        }


if __name__ == "__main__":
    unittest.main()