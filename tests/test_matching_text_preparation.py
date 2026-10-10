from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "matching_text_preparation", PROJECT_ROOT / "code/matching_text_preparation.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Cannot load matching_text_preparation")
preparation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preparation
spec.loader.exec_module(preparation)
runner_spec = importlib.util.spec_from_file_location(
    "run_matching_text_preparation_experiment", PROJECT_ROOT / "code/run_matching_text_preparation_experiment.py",
)
if runner_spec is None or runner_spec.loader is None:
    raise ImportError("Cannot load run_matching_text_preparation_experiment")
experiment_runner = importlib.util.module_from_spec(runner_spec)
sys.modules[runner_spec.name] = experiment_runner
runner_spec.loader.exec_module(experiment_runner)


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        tokens = text.split()
        return (["[CLS]", *tokens, "[SEP]"] if add_special_tokens else tokens)


class MatchingTextPreparationTests(unittest.TestCase):
    def _record(self, description: str) -> dict:
        return {
            "job_id": "job-1",
            "title_raw": "Field Operations Technician",
            "title_searchable": "Field Operations Technician",
            "company_name_included": "Example Corp",
            "historical_snapshot": True,
            "snapshot_date": "2026-10-01",
            "status_raw": None,
            "status_derived": "unknown",
            "description_usable": True,
            "description_text": description,
            "onet_data": {"occupation_name": "Field Technicians"},
        }

    def test_explicit_categories_modality_negation_and_ambiguous_text_are_preserved(self):
        description = (
            "Career at Example Corp:\n"
            "We are proud to be a growing organization with a vibrant culture.\n"
            "Requirements:\n"
            "Must have two years of field inspection experience.\n"
            "A forklift license is not required.\n"
            "Preferred Qualifications:\n"
            "Experience with SAP is preferred.\n"
            "Responsibilities:\n"
            "Inspect sites and document field conditions.\n"
            "Skills:\n"
            "Use Excel and GIS tools.\n"
            "Working Conditions:\n"
            "Rotating shifts and local travel.\n"
            "Qualifications:\n"
            "A varied background may be helpful.\n"
        )
        record = self._record(description)
        row = preparation.build_dataset_row(record, 0, WhitespaceTokenizer())
        items = {item["source_text"]: item for item in row["matching_text"]["evidence"]}

        self.assertEqual(items["Must have two years of field inspection experience."]["category"],
                         "required_qualifications")
        self.assertEqual(items["Must have two years of field inspection experience."]["requirement_modality"],
                         "required")
        self.assertEqual(items["A forklift license is not required."]["category"], "unclassified")
        self.assertEqual(items["A forklift license is not required."]["requirement_modality"], "not_required")
        self.assertEqual(items["Experience with SAP is preferred."]["category"],
                         "preferred_qualifications")
        self.assertEqual(items["Inspect sites and document field conditions."]["category"], "duties")
        self.assertEqual(items["Use Excel and GIS tools."]["category"], "skills_tools")
        self.assertEqual(items["Rotating shifts and local travel."]["category"], "working_conditions")
        self.assertEqual(items["A varied background may be helpful."]["category"], "unclassified")
        self.assertTrue(any(item["category"] == "company_background" for item in row["matching_text"]["evidence"]))
        self.assertEqual(row["source_record"], record)
        self.assertEqual(row["matching_text"]["historical_markers"]["historical_snapshot"], True)
        self.assertEqual(row["matching_text"]["historical_markers"]["status_raw"], None)
        self.assertEqual(preparation.validate_dataset_row(row), [])

    def test_source_spans_are_exact_and_null_description_state_is_preserved(self):
        record = self._record(
            "  Requirements:\r\n    Must have a valid operator license.\r\n"
            "  Disclaimer: Franchisees set their own wage and benefit programs.\r\n"
        )
        row = preparation.build_dataset_row(record, 7, WhitespaceTokenizer())
        description = record["description_text"]
        for item in row["matching_text"]["evidence"]:
            self.assertEqual(description[item["source_char_start"]:item["source_char_end"]], item["source_text"])
        missing = self._record("")
        missing["description_text"] = None
        empty_row = preparation.build_dataset_row(missing, 8, WhitespaceTokenizer())
        self.assertEqual(empty_row["matching_text"]["description_state"], "null")
        self.assertEqual(empty_row["source_record"]["description_text"], None)

    def test_rendering_excludes_company_background_and_respects_token_budget(self):
        description = (
            "About the Company:\n" + "We are proud to offer a vibrant culture. " * 30 + "\n"
            "Required Qualifications:\nMust have an operator license.\n"
            "Responsibilities:\nInspect equipment and record results.\n"
            "Skills:\nUse GIS software and Excel.\n"
            "Working Conditions:\nTravel locally on rotating shifts.\n"
        )
        row = preparation.build_dataset_row(self._record(description), 0, WhitespaceTokenizer())
        text = row["bge_document_text"]
        trace = row["bge_render_trace"]
        self.assertNotIn("vibrant culture", text)
        self.assertIn("Must have an operator license", text)
        self.assertIn("Inspect equipment and record results", text)
        self.assertGreater(trace["company_background_excluded_count"], 0)
        self.assertLessEqual(trace["encoded_tokens_including_specials"], 512)
        self.assertEqual(trace["representation_id"], preparation.REPRESENTATION_ID)

    def test_jsonl_reader_preserves_embedded_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            value = {"text": "first\u2028middle\u0085last"}
            path.write_text(experiment_runner._canonical(value) + "\n", encoding="utf-8")
            self.assertEqual(experiment_runner._read_jsonl(path), [value])

    def test_extraction_review_validates_exact_ids_and_categories(self):
        source_line_id = "job-1:S000001"
        evidence_id = "job-1:E000001"
        row = {
            "job_id": "job-1",
            "source_line_catalog": [{"source_line_id": source_line_id}],
            "evidence_catalog": [{"evidence_id": evidence_id}],
        }
        result = {
            "job_id": "job-1",
            "review_status": "reviewed",
            "reviewed_evidence": [{
                "evidence_id": evidence_id,
                "reviewer_category": "duties",
                "reviewer_modality": "unspecified",
                "assessment": "supported",
            }],
            "missed_material_evidence_ids": [source_line_id],
            "unsupported_classification_evidence_ids": [],
            "unknown_evidence_ids": [],
        }
        validated, errors = experiment_runner._validate_extraction_review(result, row)
        self.assertEqual(errors, [])
        self.assertIsNotNone(validated)
        result["missed_material_evidence_ids"] = ["invented-line-id"]
        validated, errors = experiment_runner._validate_extraction_review(result, row)
        self.assertIsNone(validated)
        self.assertIn("missed_material_evidence_ids_contains_invalid_id", errors)


if __name__ == "__main__":
    unittest.main()