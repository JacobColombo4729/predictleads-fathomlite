"""Tests for the role-family classifier's leak guard (synthetic data only).

The classifier's labels come from each listing's O*NET occupation name. An earlier version trained on text that
contained that name and reported 94.7% accuracy; the real figure without it is far lower. These tests keep the
occupation out of every model input.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import role_family_classifier as rfc  # noqa: E402

OCCUPATION = "Sales Engineers"


def _record(job_id: str, title: str, occupation: str | None, company: str) -> dict:
    return {
        "job_id": job_id, "title_searchable": title, "description_text": "Demonstrate the product to prospects.",
        "onet_data": {"occupation_name": occupation} if occupation else None,
        "company_reference": {"id": company, "type": "company"},
    }


class ClassifierTextTests(unittest.TestCase):
    def test_title_text_never_contains_the_occupation(self) -> None:
        text = rfc.classifier_text("Pre-Sales Engineer")
        self.assertEqual(text, "Job title: Pre-Sales Engineer.")
        self.assertNotIn(OCCUPATION, text)
        self.assertNotIn("Occupation", text)

    def test_description_is_cut_and_still_has_no_occupation(self) -> None:
        text = rfc.classifier_text("Pre-Sales Engineer", "x" * 5000, description_chars=100)
        self.assertEqual(len(text), len("Job title: Pre-Sales Engineer.\n") + 100)
        self.assertNotIn("Occupation", text)

    def test_no_description_when_not_requested(self) -> None:
        self.assertEqual(rfc.classifier_text("A", "long text", 0), "Job title: A.")


class LoadLabelsTests(unittest.TestCase):
    def test_labels_come_from_the_occupation_but_inputs_do_not(self) -> None:
        records = [
            _record("a", "Solutions Engineer", OCCUPATION, "c1"),
            _record("b", "Data Scientist II", "Data Scientists", "c2"),
            _record("c", "Barista", "Baristas", "c3"),       # not in a family: Other
            _record("d", "Unlabelled role", None, "c4"),      # no occupation: unlabelled
        ]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "jobs_prepared_corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            data = rfc._load_labels(Path(folder))
        self.assertEqual(data["labels"], ["Technical Sales / Solutions Engineer", "Data Scientist", rfc.OTHER, None])
        # the model inputs (titles, descriptions) never hold the occupation name
        for text in data["titles"] + data["descs"]:
            self.assertNotIn(OCCUPATION, text)
        # the occupation is kept separate, only for the explicit leak control
        self.assertEqual(data["occupations"][0], OCCUPATION)

    def test_company_is_the_split_group(self) -> None:
        records = [_record("a", "T", OCCUPATION, "c1"), _record("b", "T", OCCUPATION, "c1")]
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "jobs_prepared_corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            data = rfc._load_labels(Path(folder))
        self.assertEqual(data["groups"], ["c1", "c1"])


class FamilyTableTests(unittest.TestCase):
    def test_compass_still_gets_its_mapping(self) -> None:
        self.assertEqual(rfc.FAMILY_OF_OCCUPATION["Data Scientists"], "Data Scientist")
        self.assertEqual(len(set(rfc.FAMILY_OF_OCCUPATION.values())), 16)


if __name__ == "__main__":
    unittest.main()
