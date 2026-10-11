from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "code" / "evaluate_description_coverage.py"
SPEC = importlib.util.spec_from_file_location("evaluate_description_coverage", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError("Could not load code/evaluate_description_coverage.py")
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


class SavedExperimentEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recommendations = self.root / "recommendations.csv"
        self.labels = self.root / "review.csv"
        rows = []
        for method, condition, ids in (
            ("keyword", "all_eligible", ["good-1", "good-2", "good-3", "good-4", "good-5"]),
            ("tfidf_cosine", "all_eligible", ["bad-1", "bad-2", "bad-3", "bad-4", "bad-5"]),
            ("keyword", "description_available", ["good-1", "good-2", "good-3", "good-4", "good-5"]),
            ("tfidf_cosine", "description_available", ["bad-1", "bad-2", "bad-3", "bad-4", "bad-5"]),
        ):
            rows.extend({"profile_id": "p", "method": method, "condition": condition,
                         "rank": rank, "job_id": job_id} for rank, job_id in enumerate(ids, 1))
        with self.recommendations.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_labels(self, values: dict[str, str]) -> None:
        with self.labels.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["profile_id", "job_id", "relevance_score"])
            writer.writeheader()
            for job_id, score in values.items():
                writer.writerow({"profile_id": "p", "job_id": job_id, "relevance_score": score})

    def test_blank_labels_are_rejected_as_no_metrics(self) -> None:
        self._write_labels({"good-1": "", "bad-1": ""})
        with self.assertRaisesRegex(ValueError, "No genuine relevance labels"):
            evaluator.evaluate_saved_experiment(self.recommendations, self.labels)

    def test_saved_shared_pool_labels_distinguish_methods_and_conditions(self) -> None:
        labels = {f"good-{i}": "2" for i in range(1, 6)}
        labels.update({f"bad-{i}": "0" for i in range(1, 6)})
        self._write_labels(labels)
        metrics = evaluator.evaluate_saved_experiment(self.recommendations, self.labels)
        lookup = {(row["method"], row["condition"]): row for row in metrics}
        self.assertEqual(len(metrics), 4)
        self.assertEqual(lookup[("keyword", "all_eligible")]["ndcg_at_k"], 1.0)
        self.assertEqual(lookup[("tfidf_cosine", "all_eligible")]["ndcg_at_k"], 0.0)
        self.assertEqual(lookup[("keyword", "description_available")]["ndcg_at_k"], 1.0)
        self.assertEqual(lookup[("tfidf_cosine", "description_available")]["ndcg_at_k"], 0.0)

    def test_evaluator_can_limit_comparison_to_one_profile(self) -> None:
        rows = []
        with self.recommendations.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        other = [{**row, "profile_id": "other"} for row in rows]
        with self.recommendations.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows + other)
        self._write_labels({**{f"good-{i}": "2" for i in range(1, 6)},
                            **{f"bad-{i}": "0" for i in range(1, 6)}})
        result = evaluator.evaluate_saved_experiment(self.recommendations, self.labels, profile_id="p")
        self.assertEqual(len(result), 4)
        self.assertEqual({row["profile_id"] for row in result}, {"p"})


if __name__ == "__main__":
    unittest.main()
