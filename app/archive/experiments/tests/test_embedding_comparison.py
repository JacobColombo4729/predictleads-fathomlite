from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_name, relative in (
    ("job_matching_engine", Path("code/job_matching_engine.py")),
    ("embedding_comparison", Path("code/embedding_comparison.py")),
    ("evaluate_embedding_comparison", Path("code/evaluate_embedding_comparison.py")),
):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

engine = sys.modules["job_matching_engine"]
comparison = sys.modules["embedding_comparison"]
evaluator = sys.modules["evaluate_embedding_comparison"]


class EmbeddingComparisonTests(unittest.TestCase):
    def test_unknown_pending_judgments_remain_null_not_zero(self) -> None:
        pool = [{"profile_id": "p", "job_id": "j1", "presentation_id": "P1"}]
        response = {"profile_id": "p", "job_id": "j1", "grade": None,
                    "status": "pending_evaluator", "unknown_reason": "not evaluated yet"}
        valid, errors = evaluator.validate_judgments(pool, [response])
        self.assertEqual(errors, [])
        self.assertIsNone(valid[0]["grade"])
        self.assertEqual(valid[0]["status"], "pending_evaluator")
        self.assertNotIn(("p", "j1"), {(row["profile_id"], row["job_id"]): row["grade"]
                                        for row in valid if row["status"] == "judged"})

    def test_precision_withholds_for_short_result_list(self) -> None:
        pool = [{"profile_id": "p", "job_id": f"j{i}"} for i in range(1, 5)]
        rankings = [{"profile_id": "p", "method": method, "rank": rank, "job_id": f"j{rank}"}
                    for method in evaluator.METHODS for rank in range(1, 5)]
        judgments = [{"profile_id": "p", "job_id": f"j{i}", "grade": 2, "status": "judged"}
                     for i in range(1, 5)]
        metrics = evaluator.calculate_metrics(rankings, pool, judgments,
                                              {"heldout_profile_ids": ["p"]})
        for row in metrics["per_profile_metrics"]:
            self.assertIsNone(row["plausible_or_strong_precision_at_5"])
            self.assertIsNone(row["strong_only_precision_at_5"])
            self.assertEqual(row["plausible_or_strong_precision_status"], "fewer_than_five_results")

    def test_precision_withholds_when_any_top5_judgment_is_unknown(self) -> None:
        pool = [{"profile_id": "p", "job_id": f"j{i}"} for i in range(1, 6)]
        rankings = [{"profile_id": "p", "method": method, "rank": rank, "job_id": f"j{rank}"}
                    for method in evaluator.METHODS for rank in range(1, 6)]
        judgments = [{"profile_id": "p", "job_id": f"j{i}", "grade": 2, "status": "judged"}
                     for i in range(1, 5)]
        metrics = evaluator.calculate_metrics(rankings, pool, judgments,
                                              {"heldout_profile_ids": ["p"]})
        self.assertTrue(all(row["plausible_or_strong_precision_at_5"] is None
                            for row in metrics["per_profile_metrics"]))
        self.assertTrue(all(row["unknown_top5_ranks"] == [5]
                            for row in metrics["per_profile_metrics"]))

    def test_ndcg_withholds_if_common_pool_judgments_are_incomplete(self) -> None:
        pool = [{"profile_id": "p", "job_id": f"j{i}"} for i in range(1, 6)]
        rankings = [{"profile_id": "p", "method": method, "rank": rank, "job_id": f"j{rank}"}
                    for method in evaluator.METHODS for rank in range(1, 6)]
        judgments = [{"profile_id": "p", "job_id": f"j{i}", "grade": i % 4, "status": "judged"}
                     for i in range(1, 5)]
        metrics = evaluator.calculate_metrics(rankings, pool, judgments,
                                              {"heldout_profile_ids": ["p"]})
        self.assertTrue(all(row["ndcg_at_5"] is None for row in metrics["per_profile_metrics"]))
        self.assertTrue(all(row["ndcg_status"] == "incomplete_common_pool_judgments"
                            for row in metrics["per_profile_metrics"]))

    def test_blinded_union_is_stable_deduplicated_and_has_no_ranking_fields(self) -> None:
        job = engine.JobRecord("job-1", {"job_id": "job-1", "title_raw": "Job one",
                                         "title_searchable": "Job one", "description_text": "Work text",
                                         "description_usable": True}, True)
        corpus = engine.MatchingCorpus([job], None, None)
        fixture = {"profile_id": "p", "fixture_type": "career_transition", "synthetic_fixture": True,
                   "work_history": ["Prior work"], "accomplishments": ["Achievement"],
                   "experience_level": "mid-level", "skills_with_evidence": [], "education_certifications": [],
                   "desired_roles": ["New role"], "career_transition_goal": "Explore", "location": None,
                   "work_arrangement_preferences": [], "soft_preferences": [], "hard_constraints": {}}
        rankings = [
            {"profile_id": "p", "method": "keyword", "rank": 1, "score": 0.8, "job_id": "job-1"},
            {"profile_id": "p", "method": "tfidf_cosine", "rank": 2, "score": 0.4, "job_id": "job-1"},
            {"profile_id": "p", "method": "embedding_cosine", "rank": 1, "score": 0.7, "job_id": "job-1"},
        ]
        pool_a = comparison.build_blinded_pool(rankings, corpus, {"p": fixture}, ["p"], 42)
        pool_b = comparison.build_blinded_pool(rankings, corpus, {"p": fixture}, ["p"], 42)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(len(pool_a), 1)
        self.assertNotIn("method", pool_a[0])
        self.assertNotIn("rank", pool_a[0])
        self.assertNotIn("score", pool_a[0])
        self.assertEqual(pool_a[0]["job_id"], "job-1")

    def test_all_rankers_share_description_and_hard_filter_eligibility(self) -> None:
        jobs = [
            engine.JobRecord("open", {"job_id": "open", "title_raw": "Inventory clerk",
                                       "title_searchable": "Inventory clerk", "description_text": "Count inventory daily.",
                                       "description_usable": True, "status_derived": "open"}, True),
            engine.JobRecord("closed", {"job_id": "closed", "title_raw": "Inventory clerk",
                                         "title_searchable": "Inventory clerk", "description_text": "Count stock daily.",
                                         "description_usable": True, "status_derived": "closed"}, True),
        ]
        corpus = engine.fit_tfidf(jobs)
        indices = engine.condition_job_indices(corpus, "description_available")
        profile = engine.CandidateProfile("p", desired_roles=("inventory clerk",),
                                          preferences={"exclude_explicitly_closed": True})
        filtered = comparison.filtered_indices_for_profile(corpus, profile, indices)
        self.assertEqual([corpus.jobs[index].job_id for index in filtered], ["open"])
        keyword = engine.rank_keyword(corpus, profile, 10, candidate_indices=indices,
                                      condition="description_available")
        tfidf = engine.rank_tfidf(corpus, profile, 10, candidate_indices=indices,
                                  condition="description_available")
        self.assertEqual({row["job_id"] for row in keyword}, {"open"})
        self.assertEqual({row["job_id"] for row in tfidf}, {"open"})

        class QueryModel:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0]], dtype=np.float32)

        previous_model = comparison._ACTIVE_MODEL
        comparison._ACTIVE_MODEL = QueryModel()
        try:
            embedding = comparison._embedding_rank(corpus, profile, indices,
                                                  np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                                                  {0: 0, 1: 1}, 10)
        finally:
            comparison._ACTIVE_MODEL = previous_model
        self.assertEqual({row["job_id"] for row in embedding}, {"open"})

    def test_embedding_cache_key_changes_with_model_revision_or_text(self) -> None:
        ids = ["j1", "j2"]
        texts = ["title and description one", "title and description two"]
        original = comparison._cache_key(ids, texts, "model@rev-a")
        self.assertNotEqual(original, comparison._cache_key(ids, texts, "model@rev-b"))
        self.assertNotEqual(original, comparison._cache_key(ids, [texts[0], "changed"], "model@rev-a"))
        self.assertNotEqual(original, comparison._cache_key(["j2", "j1"], texts, "model@rev-a"))

    def test_seeded_split_has_four_dev_and_mixed_eight_heldout(self) -> None:
        fixtures, _ = comparison._load_fixtures(comparison.DEFAULT_PROFILES)
        split = comparison._seeded_split(fixtures)
        self.assertEqual(len(split["development_profile_ids"]), 4)
        self.assertEqual(len(split["heldout_profile_ids"]), 8)
        self.assertGreater(split["heldout_fixture_type_counts"].get("conventional_match", 0), 0)
        self.assertGreater(split["heldout_fixture_type_counts"].get("career_transition", 0), 0)

    def test_invalid_grade_is_not_silently_coerced(self) -> None:
        pool = [{"profile_id": "p", "job_id": "j"}]
        response = {"profile_id": "p", "job_id": "j", "grade": "3", "status": "judged",
                    "unknown_reason": None, "supporting_candidate_evidence": [], "supporting_job_evidence": [],
                    "material_gaps": [], "explicit_constraint_conflicts": [], "required_qualifications": "unknown",
                    "career_transition_relevance": {"level": "unknown", "rationale": "unknown"},
                    "title_outside_stated_targets": False, "evidence_sufficiency": "sufficient"}
        valid, errors = evaluator.validate_judgments(pool, [response])
        self.assertEqual(valid, [])
        self.assertEqual(errors[0]["error"], "grade_must_be_integer_0_to_3_or_null")


if __name__ == "__main__":
    unittest.main()
