from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
spec = importlib.util.spec_from_file_location(
    "analyze_matching_text_errors", PROJECT_ROOT / "code/analyze_matching_text_errors.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Cannot load analyze_matching_text_errors")
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


class MatchingErrorAnalysisTests(unittest.TestCase):
    def test_long_citation_maps_to_retained_and_omitted_facets_without_claiming_causality(self):
        facets = [
            {"evidence_id": "e1", "category": "duties", "requirement_modality": "unspecified",
             "source_line": 1, "source_text": "Inspect pumps and record pressure readings."},
            {"evidence_id": "e2", "category": "required_qualifications", "requirement_modality": "required",
             "source_line": 2, "source_text": "Must hold a current boiler operator license."},
        ]
        result = analysis._align_citation_to_facets(
            "Job Description: Inspect pumps and record pressure readings. Must hold a current boiler operator license. Other text.",
            facets,
            "Inspect pumps and record pressure readings. Must hold a current boiler operator license. Other text.",
            "Job title: Technician. Duty: Inspect pumps and record pressure readings.",
            "Job title: Technician. Duty: Inspect pumps and record pressure readings.",
            {"retained_evidence_ids": ["e1"], "omitted_evidence": [{"evidence_id": "e2"}]},
        )
        self.assertEqual(result["alignment_status"], "matched_to_source_facets")
        self.assertEqual([row["evidence_id"] for row in result["aligned_source_facets"]], ["e1", "e2"])
        self.assertEqual(result["aligned_source_facets"][1]["treatment_input_status"], "omitted_by_token_budget")
        self.assertIn("not causal evidence", result["warning"])

    def test_unknown_or_preferred_is_not_reported_as_explicit_mandatory_conflict(self):
        profile = {"qualification_disclosures": {
            "explicit_absence": ["ISO 13485 auditor credential"],
            "unknown": ["Emergency-management certification"],
        }}
        facets = [
            {"category": "preferred_qualifications", "requirement_modality": "preferred",
             "source_text": "ISO 13485 auditor credential preferred."},
            {"category": "unclassified", "requirement_modality": "unspecified",
             "source_text": "Emergency-management certification experience is unclear."},
        ]
        judgment = {"explicit_constraint_conflicts": [
            "Candidate lacks ISO 13485 auditor credential.",
            "Candidate lacks emergency-management certification.",
        ]}
        result = analysis._explicit_requirement_review(profile, judgment, facets)
        self.assertEqual(result["evaluator_claim_assessments"][0]["status"],
                         "candidate_explicit_absence_without_matching_mandatory_job_text")
        self.assertEqual(result["evaluator_claim_assessments"][1]["status"],
                         "profile_disclosure_is_unknown_not_explicit_absence")


if __name__ == "__main__":
    unittest.main()