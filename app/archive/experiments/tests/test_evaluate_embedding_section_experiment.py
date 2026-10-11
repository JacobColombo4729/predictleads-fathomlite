from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_name, relative in (
    ("job_matching_engine", Path("code/job_matching_engine.py")),
    ("embedding_comparison", Path("code/embedding_comparison.py")),
    ("evaluate_embedding_comparison", Path("code/evaluate_embedding_comparison.py")),
    ("section_aware_embedding", Path("code/section_aware_embedding.py")),
):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

spec = importlib.util.spec_from_file_location(
    "evaluate_embedding_section_experiment",
    PROJECT_ROOT / "code" / "evaluate_embedding_section_experiment.py",
)
if spec is None or spec.loader is None:
    raise ImportError("Cannot load section-aware evaluator")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


class SectionAwareExperimentEvaluationTests(unittest.TestCase):
    def test_variant_rankings_are_scored_separately_from_baseline_embeddings(self) -> None:
        rankings = [
            {"profile_id": "p", "method": method, "rank": str(rank), "job_id": f"{method}-{rank}"}
            for method in ("keyword", "tfidf_cosine", "embedding_cosine",
                           experiment.REPRESENTATION_ID)
            for rank in range(1, 6)
        ]
        baseline, variant = experiment._split_rankings(rankings)
        self.assertEqual(len(baseline), 15)
        self.assertEqual(len(variant), 5)
        self.assertEqual({row["method"] for row in baseline}, {"keyword", "tfidf_cosine", "embedding_cosine"})
        self.assertEqual({row["method"] for row in variant}, {"embedding_cosine"})
        self.assertEqual({row["job_id"] for row in variant},
                         {f"{experiment.REPRESENTATION_ID}-{rank}" for rank in range(1, 6)})


if __name__ == "__main__":
    unittest.main()