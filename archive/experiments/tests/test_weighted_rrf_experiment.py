from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_weighted_rrf_experiment", PROJECT_ROOT / "code/run_weighted_rrf_experiment.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Cannot load run_weighted_rrf_experiment")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


class WeightedRRFTests(unittest.TestCase):
    def test_equal_ratio_matches_original_rrf_exactly(self):
        tfidf = [{"job_id": "a", "rank": 1}, {"job_id": "b", "rank": 2}]
        bge = [{"job_id": "b", "rank": 1}, {"job_id": "c", "rank": 2}]
        self.assertEqual(experiment.weighted_rrf_scores(tfidf, bge, 60, 1, 1),
                         experiment.hybrid._rrf_scores(tfidf, bge, 60))

    def test_declared_weights_are_positive_and_shift_only_rank_contributions(self):
        tfidf = [{"job_id": "lexical", "rank": 1}, {"job_id": "shared", "rank": 2}]
        bge = [{"job_id": "semantic", "rank": 1}, {"job_id": "shared", "rank": 2}]
        lexical_weighted = experiment.weighted_rrf_scores(tfidf, bge, 60, 2, 1)
        semantic_weighted = experiment.weighted_rrf_scores(tfidf, bge, 60, 1, 2)
        self.assertGreater(lexical_weighted["lexical"], lexical_weighted["semantic"])
        self.assertGreater(semantic_weighted["semantic"], semantic_weighted["lexical"])
        self.assertAlmostEqual(lexical_weighted["shared"], semantic_weighted["shared"])

    def test_nonpositive_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment.weighted_rrf_scores([], [], 60, 0, 1)


if __name__ == "__main__":
    unittest.main()