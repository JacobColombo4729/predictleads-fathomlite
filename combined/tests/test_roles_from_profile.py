"""v2: the roles come from the whole profile, not the trained layer alone (combined/role_fit.py with_evidence,
combined/candidates.py role_mix, combined/model.py). Made-up roles, listings and text matches.

The failure this guards against (2026-10-15): a career changer heading for data science, with data skills and
descriptions, got software development first, because the trained layer reads only titles, programs and dates.

    python -m unittest discover -s combined/tests -t .
"""
import unittest

import numpy as np
import pandas as pd

import combined._paths  # noqa: F401  puts analysis/ on the path
from combined.candidates import Candidate, ListingIndex, role_mix
from combined.model import CombinedModel
from combined.queue import Queue
from combined.role_fit import RoleFitResult, role_of_title
from src import fit_eval as fe

ROLES = fe.ROLES


def model_says_software():
    """What the trained layer gave: software development first, data science well down."""
    p = pd.Series(0.05, index=ROLES)
    p["software development"], p["data science"] = 0.45, 0.10
    p = p / p.sum()
    table = pd.DataFrame({"probability": p, "score": p, "rating": p / p.max() * 100}).sort_values("score", ascending=False)
    return RoleFitResult(table, [], "career change", 0)


def matches(roles):
    return [Candidate(i, f"j{i}", r, 1.0 - i / 100, overall_rank=i + 1, sources=["overall"]) for i, r in enumerate(roles)]


class RoleChoice(unittest.TestCase):
    def test_role_mix_is_the_share_of_text_matches_per_role(self):
        mix = role_mix(matches(["data science"] * 6 + ["database"] * 2 + ["technical sales"] * 2), ROLES)
        self.assertAlmostEqual(mix["data science"], 0.75)
        self.assertAlmostEqual(mix["database"], 0.25)
        self.assertAlmostEqual(sum(mix.values()), 1.0)
        self.assertEqual(sum(role_mix(matches(["technical sales"]), ROLES).values()), 0.0)

    def test_a_profile_that_reads_as_data_gets_data_science_first(self):
        mix = role_mix(matches(["data science"] * 8 + ["database"] * 2), ROLES)
        roles = model_says_software().with_evidence(mix)
        self.assertEqual(roles.top(1), ["data science"])
        self.assertAlmostEqual(roles.table["rating"].iloc[0], 100.0)
        self.assertEqual(roles.table.loc["software development", "model score"], model_says_software().table.loc["software development", "score"])

    def test_desired_roles_always_come_first(self):
        mix = role_mix(matches(["it support"] * 10), ROLES)
        roles = model_says_software().with_evidence(mix, desired_roles=["Junior Data Scientist", "not a tech title"])
        self.assertEqual(roles.top(1), ["data science"])
        self.assertTrue(roles.table.loc["data science", "desired"])
        self.assertEqual(int(roles.table["desired"].sum()), 1)

    def test_text_weight_zero_is_the_trained_layer_alone(self):
        mix = role_mix(matches(["data science"] * 10), ROLES)
        self.assertEqual(model_says_software().with_evidence(mix, text_weight=0.0).top(3), model_says_software().top(3))

    def test_titles_map_to_roles_by_profiles_rules(self):
        self.assertEqual(role_of_title("Data Scientist"), "data science")
        self.assertEqual(role_of_title("Machine Learning Engineer"), "data science")
        self.assertEqual(role_of_title("Help Desk Technician"), "it support")
        self.assertIsNone(role_of_title("Store Manager"))


class ModelEndToEnd(unittest.TestCase):
    def test_the_queue_follows_the_profile_not_the_trained_layer_alone(self):
        roles = np.array(["software development"] * 40 + ["data science"] * 40 + ["it support"] * 40)
        r = np.random.default_rng(0)
        scores = pd.DataFrame({"id": [f"j{i}" for i in range(len(roles))], "role_bucket": roles, "p_still_open": r.random(len(roles)),
                               "accessibility": r.integers(0, 5, len(roles)), "pay": r.uniform(20, 80, len(roles)),
                               "level": "entry", "hidden_occupation": False, "likely_ghost": False})
        listings = ListingIndex.from_scores(list(scores.id), scores)
        weights = {"fit": 0.3, "still_open": 0.2, "accessibility": 0.1, "pay": 0.0}
        model = CombinedModel(role_fit=type("RF", (), {"roles_order": ROLES})(), listings=listings,
                              queue=Queue.from_scores(listings, scores, weights, ROLES))
        # the profile's text matches data-science listings best, as a data career changer's skills would
        text = np.where(roles == "data science", 1.0, 0.1) + r.random(len(roles)) * 0.01
        rank = lambda limit, idx: [{"job_id": listings.job_ids[i], "rank": k, "score": float(text[i])}
                                   for k, i in enumerate(sorted(idx, key=lambda i: -text[i])[:limit], 1)]
        result = model.recommend_from(model_says_software(), rank, limit=10, overall=20, per_role=10)
        self.assertEqual(result.roles.top(1), ["data science"])
        self.assertGreaterEqual(int((result.queue.role == "data science").sum()), 7)


if __name__ == "__main__":
    unittest.main()
