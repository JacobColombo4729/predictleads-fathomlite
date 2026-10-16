"""Step 5, the queue (combined/queue.py), on made-up ML_Rank scores; and the whole model (combined/model.py) on a
made-up hand-off, a stand-in BGE and a made-up text matcher.

    python -m unittest discover -s combined/tests -t .
"""
import tempfile
import unittest

import numpy as np
import pandas as pd

import combined._paths  # noqa: F401  puts analysis/ on the path, for `src`
from combined.candidates import Candidate, ListingIndex
from combined.model import CombinedModel
from combined.queue import Queue
from combined.role_fit import RoleFit
from combined.tests.test_role_fit import ANALYST_TO_BE, made_up_handoff
from src import fit_eval as fe
from src import rank_model as rm

WEIGHTS = {"fit": 0.3, "still_open": 0.2, "accessibility": 0.1, "pay": 0.0}


def made_up_scores(n=80, seed=0, roles=None):
    r = np.random.default_rng(seed)
    roles = roles if roles is not None else r.choice(fe.ROLES, n)
    return pd.DataFrame({"id": [f"j{i}" for i in range(n)], "role_bucket": roles, "p_still_open": r.random(n),
                         "accessibility": r.integers(0, 5, n), "pay": np.where(r.random(n) < 0.3, np.nan, r.uniform(20, 80, n)),
                         "level": "entry", "hidden_occupation": False, "likely_ghost": False})


def pool_of(listings, indices, match=None):
    return [Candidate(i, listings.job_ids[i], listings.roles[i], 1.0 if match is None else match[k], overall_rank=k + 1,
                      sources=["overall"]) for k, i in enumerate(indices)]


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.scores = made_up_scores()
        self.listings = ListingIndex.from_scores(list(self.scores.id), self.scores)
        self.queue = Queue.from_scores(self.listings, self.scores, WEIGHTS, fe.ROLES)
        r = np.random.default_rng(1)
        self.ratings = dict(zip(fe.ROLES, r.uniform(10, 100, len(fe.ROLES))))
        self.ratings["data science"] = 100.0

    def test_same_top_10_as_ml_ranks_backtest_over_the_same_pool(self):
        everyone = self.listings.visible()
        got = self.queue.rank(pool_of(self.listings, everyone), self.ratings, limit=10)
        fit_rating = pd.DataFrame([self.ratings])[fe.ROLES]
        pool = self.scores.set_index(self.scores.index)
        expected = rm.queue_top_k(fit_rating, rm.listing_ratings(pool), pool.role_bucket, WEIGHTS, k=10)[0]
        self.assertEqual(list(got["index"]), list(expected))

    def test_parts_add_up_and_use_the_normalised_weights(self):
        t = self.queue.rank(pool_of(self.listings, self.listings.visible()), self.ratings, limit=80)
        parts = t[[c for c in t if c.startswith("part: ")]].sum(axis=1)
        self.assertLess(float((parts - t.score).abs().max()), 1e-9)
        self.assertAlmostEqual(float((t["part: fit"] / t["rating: fit"]).iloc[0]), 0.5)  # 0.3 of 0.6
        self.assertTrue((t["part: pay"] == 0).all())
        self.assertTrue(t.score.is_monotonic_decreasing)

    def test_match_blend_only_moves_the_fit_factor(self):
        idx = self.listings.visible()[:20]
        match = list(np.linspace(1.0, 0.1, len(idx)))
        base = self.queue.rank(pool_of(self.listings, idx, match), self.ratings, limit=20).set_index("job_id")
        blend = self.queue.rank(pool_of(self.listings, idx, match), self.ratings, limit=20, match_weight=1.0).set_index("job_id")
        pd.testing.assert_series_equal(base["rating: still_open"].sort_index(), blend["rating: still_open"].sort_index())
        self.assertLess(float((blend["rating: fit"] - blend["match"]).abs().max()), 1e-9)

    def test_a_role_without_a_fit_model_is_neutral_and_labelled(self):
        scores = made_up_scores(roles=np.array(["technical sales"] * 10 + list(np.resize(fe.ROLES, 70))))
        listings = ListingIndex.from_scores(list(scores.id), scores)
        queue = Queue.from_scores(listings, scores, WEIGHTS, fe.ROLES)
        t = queue.rank(pool_of(listings, listings.visible()), self.ratings, limit=80).set_index("job_id")
        sales = t[t.role == "technical sales"]
        self.assertTrue((sales["rating: fit"] == 50.0).all())
        self.assertTrue(sales["fit source"].str.startswith("no fit model").all())

    def test_an_empty_pool_gives_an_empty_queue(self):
        self.assertTrue(self.queue.rank([], self.ratings).empty)


class WholeModelTests(unittest.TestCase):
    def test_a_resume_to_a_ranked_queue(self):
        with tempfile.TemporaryDirectory() as d:
            bge = made_up_handoff(d)
            role_fit = RoleFit.load(d, bge=bge)
        scores = made_up_scores(n=120, seed=3)
        listings = ListingIndex.from_scores(list(scores.id), scores)
        model = CombinedModel(role_fit, listings, Queue.from_scores(listings, scores, WEIGHTS, fe.ROLES))
        text = np.random.default_rng(4).random(len(listings.job_ids))
        rank = lambda limit, idx: [{"job_id": listings.job_ids[i], "rank": r, "score": float(text[i])}
                                   for r, i in enumerate(sorted(idx, key=lambda i: -text[i])[:limit], 1)]
        result = model.recommend(ANALYST_TO_BE, rank, limit=10, overall=10, per_role=5, as_of="2026-10")
        self.assertEqual(len(result.queue), 10)
        self.assertGreater(result.pool_size, 10)  # the top roles added listings beyond the 10 text matches
        top3 = result.roles.top(3)
        self.assertTrue(set(result.queue.role) <= set(fe.ROLES))
        self.assertGreaterEqual(int(result.queue.role.isin(top3).sum()), 1)
        best = result.queue.iloc[0]
        self.assertAlmostEqual(best["rating: fit"], result.roles.ratings()[best.role])


if __name__ == "__main__":
    unittest.main()
