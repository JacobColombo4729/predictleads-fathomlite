"""Offline tests for Jacob's ML_Rank ranker (synthetic listings and a fake text-match ranker; no data, no models)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from backend.ml_rank_ranker import MlRankRankingBackend  # noqa: E402

WEIGHTS = {"fit": 0.3, "still_open": 0.2, "accessibility": 0.1, "pay": 0.0}


class FakeTextMatch:
    """Stands in for the RRF ranker: fixed text-match scores, best first, limited to the eligible listings."""

    supports_compass = True

    def __init__(self, scores: dict[str, float]):
        self.job_ids = list(scores)
        self.scores = scores
        self.calls: list[dict] = []

    def rank(self, profile, limit, candidate_indices, **kwargs):
        self.calls.append(kwargs)
        ids = sorted((self.job_ids[i] for i in candidate_indices), key=lambda j: -self.scores[j])[:limit]
        return [{"job_id": j, "rank": r, "score": self.scores[j]} for r, j in enumerate(ids, 1)]

    def query_vector(self, profile):
        return "vector"


def snapshot() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "role_bucket": ["data", "data", "data", "data", "data"],
        "p_still_open": [0.1, 0.9, 0.5, 0.95, 0.99],
        "accessibility": [0, 2, 1, 4, 4],
        "pay": [30.0, 40.0, None, 50.0, 60.0],
        "level": ["entry", "entry", "entry", "manager", "entry"],
        "hidden_occupation": [False, False, False, False, False],
        "likely_ghost": [False, False, False, False, True],
    })


class MlRankRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        # f has no ML_Rank score (not in the snapshot)
        self.base = FakeTextMatch({"a": 1.0, "b": 0.9, "c": 0.5, "d": 0.8, "e": 0.7, "f": 0.6})
        self.ranker = MlRankRankingBackend(self.base, snapshot(), {"weights": WEIGHTS, "hidden_levels": ["manager", "leadership"]})

    def test_hides_manager_levels_and_likely_ghosts(self) -> None:
        ids = [r["job_id"] for r in self.ranker.rank(None, 10, range(6))]
        self.assertNotIn("d", ids)  # manager level
        self.assertNotIn("e", ids)  # likely ghost
        self.assertEqual(set(ids), {"a", "b", "c", "f"})

    def test_still_open_can_outrank_a_slightly_better_text_match(self) -> None:
        ranked = self.ranker.rank(None, 10, range(6))
        ids = [r["job_id"] for r in ranked]
        self.assertLess(ids.index("b"), ids.index("a"))  # b: 90% of a's match but far likelier to stay open
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3, 4])
        self.assertEqual([r["score"] for r in ranked], sorted((r["score"] for r in ranked), reverse=True))

    def test_parts_add_up_to_the_score_and_fit_is_relative_to_the_best_match(self) -> None:
        ranked = {r["job_id"]: r for r in self.ranker.rank(None, 10, range(6))}
        for r in ranked.values():
            self.assertAlmostEqual(sum(p["part"] for p in r["breakdown"].values()), r["score"], places=1)
        self.assertEqual(ranked["a"]["breakdown"]["fit"]["rating"], 100.0)
        self.assertEqual(ranked["c"]["breakdown"]["fit"]["rating"], 50.0)

    def test_unscored_listing_is_neutral_on_every_listing_factor(self) -> None:
        f = next(r for r in self.ranker.rank(None, 10, range(6)) if r["job_id"] == "f")
        for factor in ("still_open", "accessibility", "pay"):
            self.assertEqual(f["breakdown"][factor]["rating"], 50.0)
        self.assertEqual(self.ranker.matched, 5)

    def test_limit_feedback_passthrough_and_no_exploration(self) -> None:
        ranked = self.ranker.rank(None, 2, range(6), liked_job_ids=["a"], explore=3)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(self.base.calls[-1], {"liked_job_ids": ["a"]})  # feedback reaches the text match; exploration doesn't
        self.assertFalse(self.ranker.supports_exploration)
        self.assertEqual(self.ranker.rank(None, 5, []), [])

    def test_compass_and_vectors_come_from_the_text_match_ranker(self) -> None:
        self.assertTrue(self.ranker.supports_compass)
        self.assertEqual(self.ranker.query_vector(None), "vector")

    def test_no_tech_role_titles_get_no_fit_and_companies_are_capped(self) -> None:
        from types import SimpleNamespace

        listings = {"a": ("Data Contributor", "gig"), "b": ("Data Scientist", "acme"), "c": ("Data Scientist", "acme"),
                    "d": ("Data Analyst", "acme"), "e": ("Data Analyst", "beta"), "f": ("Data Analyst", "gamma")}
        base = FakeTextMatch({"a": 1.0, "b": 0.9, "c": 0.85, "d": 0.8, "e": 0.7, "f": 0.6})
        base.jobs = [SimpleNamespace(job_id=j, source={"title_raw": t, "company_reference": {"id": c}})
                     for j, (t, c) in listings.items()]
        snap = snapshot().assign(id=["a", "b", "c", "d", "e"], level="entry", likely_ghost=False)
        ranked = MlRankRankingBackend(base, snap, {"weights": WEIGHTS, "hidden_levels": []}).rank(None, 6, range(6))
        by_id = {r["job_id"]: r for r in ranked}
        self.assertEqual(by_id["a"]["breakdown"]["fit"]["rating"], 0.0)  # the best text match, but not a tech role
        top4 = [listings[r["job_id"]][1] for r in ranked[:4]]
        self.assertLessEqual(top4.count("acme"), 2)  # two per company, before the rest

    def test_missing_hand_off_says_to_run_ml_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "ML_Rank section 13"):
                MlRankRankingBackend.from_handoff(self.base, Path(tmp))


if __name__ == "__main__":
    unittest.main()
