from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORE_SPEC = importlib.util.spec_from_file_location(
    "retail_review_store", PROJECT_ROOT / "code" / "retail_review_store.py"
)
if STORE_SPEC is None or STORE_SPEC.loader is None:
    raise ImportError("Could not load code/retail_review_store.py")
store = importlib.util.module_from_spec(STORE_SPEC)
sys.modules[STORE_SPEC.name] = store
STORE_SPEC.loader.exec_module(store)

ENGINE_SPEC = importlib.util.spec_from_file_location(
    "job_matching_engine", PROJECT_ROOT / "code" / "job_matching_engine.py"
)
if ENGINE_SPEC is None or ENGINE_SPEC.loader is None:
    raise ImportError("Could not load code/job_matching_engine.py")
engine = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = engine
ENGINE_SPEC.loader.exec_module(engine)


class RetailReviewStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.blinded_path = self.root / "blinded.csv"
        self.judgment_path = self.root / "retail_review_judgments.csv"
        fields = ["profile_id", "candidate_evidence", "job_id", "source_title", "description_full_prepared"]
        with self.blinded_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"profile_id": store.PROFILE_ID, "candidate_evidence": "{}", "job_id": "retail-1", "source_title": "Inventory Associate", "description_full_prepared": "Stock inventory."})
            writer.writerow({"profile_id": "synthetic_software_development", "candidate_evidence": "{}", "job_id": "other-1", "source_title": "Developer", "description_full_prepared": "Write code."})
            writer.writerow({"profile_id": store.PROFILE_ID, "candidate_evidence": "{}", "job_id": "retail-2", "source_title": "Retail Associate", "description_full_prepared": "Help customers."})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_only_requested_profile_is_loaded(self) -> None:
        jobs = store.load_retail_jobs(self.blinded_path)
        self.assertEqual([row["job_id"] for row in jobs], ["retail-1", "retail-2"])

    def test_save_blank_insufficient_evidence_and_resume(self) -> None:
        store.write_saved_judgment(
            self.judgment_path, store.PROFILE_ID, "retail-1",
            relevance_score="1", review_status="rated", reviewer="reviewer", notes="plausible",
        )
        store.write_saved_judgment(
            self.judgment_path, store.PROFILE_ID, "retail-2",
            relevance_score=None, review_status="insufficient_evidence",
            insufficient_evidence_reason="Description too sparse", notes="Need source details",
        )
        rows = store.load_saved_judgments(self.judgment_path)
        self.assertEqual(rows[(store.PROFILE_ID, "retail-1")]["relevance_score"], "1")
        self.assertEqual(rows[(store.PROFILE_ID, "retail-2")]["relevance_score"], "")
        self.assertEqual(rows[(store.PROFILE_ID, "retail-2")]["review_status"], "insufficient_evidence")
        self.assertEqual(rows[(store.PROFILE_ID, "retail-2")]["insufficient_evidence_reason"], "Description too sparse")
        counts = store.review_counts(["retail-1", "retail-2", "retail-3"], rows)
        self.assertEqual(counts["rated"], 1)
        self.assertEqual(counts["insufficient_evidence"], 1)
        self.assertEqual(counts["unreviewed"], 1)

        store.save_progress(self.root / "progress.json", 1, "retail-2")
        self.assertEqual(store.load_progress(self.root / "progress.json", ["retail-1", "retail-2"]), 1)
        resumed_rows = store.load_saved_judgments(self.judgment_path)
        self.assertEqual(resumed_rows[(store.PROFILE_ID, "retail-1")]["relevance_score"], "1")

    def test_previous_profile_judgments_are_preserved_and_engine_reads_labels(self) -> None:
        store.write_saved_judgment(
            self.judgment_path, "synthetic_software_development", "software-1",
            relevance_score="2", review_status="rated", notes="previous decision",
        )
        store.write_saved_judgment(
            self.judgment_path, store.PROFILE_ID, "retail-1",
            relevance_score="0", review_status="rated",
        )
        rows = store.load_saved_judgments(self.judgment_path)
        self.assertEqual(rows[("synthetic_software_development", "software-1")]["relevance_score"], "2")
        labels = engine.read_review_labels(self.judgment_path)
        self.assertEqual(labels[(store.PROFILE_ID, "retail-1")], 0)
        self.assertNotIn((store.PROFILE_ID, "retail-2"), labels)

    def test_blank_selection_is_valid_but_invalid_scores_are_rejected(self) -> None:
        store.write_saved_judgment(
            self.judgment_path, store.PROFILE_ID, "retail-1",
            relevance_score=None, review_status="unreviewed", notes="come back",
        )
        with self.assertRaises(ValueError):
            store.write_saved_judgment(
                self.judgment_path, store.PROFILE_ID, "retail-2",
                relevance_score="3", review_status="rated",
            )


if __name__ == "__main__":
    unittest.main()
