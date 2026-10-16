"""Step 7's scoring (combined/evaluate.py) with made-up people, listings and rankers.

    python -m unittest discover -s combined/tests -t .
"""
import unittest

import numpy as np
import pandas as pd

import combined._paths  # noqa: F401  puts analysis/ on the path
from combined import evaluate as ev
from src import fit_eval as fe

ROLES = ["it support", "data science", "security"]


class RoleScores(unittest.TestCase):
    def test_counts_decide_and_the_best_position_breaks_ties(self):
        s = ev.role_scores([["data science", "it support", "data science", "it support"]], ROLES, n=4)[0]
        self.assertGreater(s[1], s[0])                # two each: data science came first
        self.assertEqual(int(s[1]), 2)
        self.assertEqual(int(s[0]), 2)
        self.assertEqual(s[2], 0.0)
        s = ev.role_scores([["it support", "data science", "data science"]], ROLES, n=3)[0]
        self.assertGreater(s[1], s[0])                # a count always outweighs the tie-break

    def test_only_the_top_n_and_known_roles_count(self):
        s = ev.role_scores([["technical sales", None, "security", "it support"]], ROLES, n=3)[0]
        self.assertEqual(int(s[2]), 1)
        self.assertEqual(s[0], 0.0)                   # 4th listing: outside the top 3

    def test_hit_rates_follow_from_the_role_ranking(self):
        scores = ev.role_scores([["security"] * 3, ["it support", "data science"]], ROLES, n=3)
        order = np.argsort(-scores, axis=1, kind="stable")
        self.assertEqual(ROLES[order[0, 0]], "security")
        self.assertEqual(ROLES[order[1, 0]], "it support")


class TextProfile(unittest.TestCase):
    def test_one_entry_per_title_or_program(self):
        p = ev.text_profile("p1", "Teacher | b.a. english |  | Data Analytics Bootcamp")
        self.assertEqual(p.profile_id, "p1")
        self.assertEqual(p.experience, ("Teacher", "b.a. english", "Data Analytics Bootcamp"))
        self.assertEqual(ev.text_profile("p2", "").experience, ())


class ScoreAllRankers(unittest.TestCase):
    def test_every_ranker_scores_every_person(self):
        people = pd.DataFrame({"text": ["help desk", "analytics", "firewall"],
                               "target_bucket": ["it support", "data science", "security"]}, index=["a", "b", "c"])
        listing_roles = ["it support", "data science", "security", "it support"]
        keyword = {"help desk": "it support", "analytics": "data science", "firewall": "security"}

        def perfect(pid, text, row):
            want = keyword[text]
            return sorted(range(len(listing_roles)), key=lambda i: listing_roles[i] != want)

        def fixed(pid, text, row):
            return [0, 3, 1, 2]

        scores, seconds, tops = ev.score(people, {"perfect": perfect, "fixed": fixed}, listing_roles, ROLES, n=2, progress_every=0)
        self.assertEqual(len(tops["fixed"]), 3)
        self.assertEqual(tops["fixed"][0], ["it support", "it support"])
        self.assertEqual(set(scores), {"perfect", "fixed"})
        self.assertEqual(scores["perfect"].shape, (3, 3))
        top1 = [ROLES[j] for j in np.argmax(scores["perfect"], axis=1)]
        self.assertEqual(top1, list(people.target_bucket))
        self.assertTrue((np.argmax(scores["fixed"], axis=1) == 0).all())
        self.assertTrue(all(v >= 0 for v in seconds.values()))

    def test_it_feeds_ml_fits_evaluation(self):
        roles = fe.ROLES
        people = pd.DataFrame({"text": ["x"] * 4, "target_bucket": [roles[0], roles[1], roles[0], roles[2]],
                               "pool": ["retrained, after working in another field"] * 4, "from_bucket": [None] * 4},
                              index=list("abcd"))
        listing_roles = [roles[0], roles[1], roles[2]]
        scores, _, _ = ev.score(people, {"r": lambda pid, t, row: [0, 1, 2]}, listing_roles, roles, n=3, progress_every=0)
        table = fe.evaluate(scores, people, train_targets=[roles[0]] * 3 + [roles[1]])
        self.assertEqual(set(table.method), {"most common role (baseline)", "r"})
        hit3 = table[(table.method == "r") & (table.k == 3) & (table.pool == "all career changers")].hit_rate.iloc[0]
        self.assertEqual(hit3, 1.0)


if __name__ == "__main__":
    unittest.main()
