"""Step 4, the candidate pool (combined/candidates.py), with a made-up text matcher and made-up ML_Rank scores.

    python -m unittest discover -s combined/tests -t .
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from combined.candidates import ListingIndex, gather

# 12 listings: the text matcher likes data science least, so data science only arrives through its role call
ROLES = ["it support"] * 4 + ["software development"] * 4 + ["data science"] * 3 + [None]
TEXT_SCORE = [0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55, 0.2, 0.15, 0.1, 0.95]
JOB_IDS = [f"j{i}" for i in range(len(ROLES))]


def scores_table():
    rows = [{"id": j, "role_bucket": r, "level": "entry", "hidden_occupation": False, "likely_ghost": False}
            for j, r in zip(JOB_IDS[:-1], ROLES[:-1])]  # the last listing has no ML_Rank score
    rows[1]["likely_ghost"] = True
    rows[5]["level"] = "manager or above"
    rows[9]["hidden_occupation"] = True
    return pd.DataFrame(rows)


def text_match(limit, indices):
    """Best first among the listings it is given, as the RRF ranker returns them."""
    ranked = sorted(indices, key=lambda i: -TEXT_SCORE[i])[:limit]
    return [{"job_id": JOB_IDS[i], "rank": r, "score": TEXT_SCORE[i]} for r, i in enumerate(ranked, 1)]


class CandidatePoolTests(unittest.TestCase):
    def setUp(self):
        self.listings = ListingIndex.from_scores(JOB_IDS, scores_table(), hidden_levels=["manager or above"])

    def test_hidden_listings_never_enter(self):
        self.assertEqual([JOB_IDS[i] for i in range(len(JOB_IDS)) if self.listings.hidden[i]], ["j1", "j5", "j9"])
        pool = gather(text_match, self.listings, ["data science"], overall=20, per_role=20)
        self.assertFalse({"j1", "j5", "j9"} & {c.job_id for c in pool})

    def test_a_top_role_the_text_misses_still_gets_its_listings(self):
        pool = gather(text_match, self.listings, ["data science"], overall=4, per_role=2)
        ids = [c.job_id for c in pool]
        self.assertEqual(ids[:4], ["j11", "j0", "j4", "j2"])  # the overall matches first, in their order
        self.assertEqual(ids[4:], ["j8", "j10"])               # then data science, which the text alone ranked last
        self.assertEqual({c.role for c in pool[4:]}, {"data science"})
        self.assertEqual(pool[4].sources, ["role: data science"])
        self.assertIsNone(pool[4].overall_rank)

    def test_a_listing_found_both_ways_appears_once_with_both_sources(self):
        pool = gather(text_match, self.listings, ["it support"], overall=3, per_role=3)
        j0 = next(c for c in pool if c.job_id == "j0")
        self.assertEqual(j0.sources, ["overall", "role: it support"])
        self.assertEqual((j0.overall_rank, j0.role_rank), (2, 1))
        self.assertEqual(len({c.job_id for c in pool}), len(pool))
        self.assertEqual(j0.match, 0.9)  # the overall call's score

    def test_unscored_listings_can_match_overall_but_belong_to_no_role(self):
        pool = gather(text_match, self.listings, ["it support", "software development"], overall=1, per_role=10)
        self.assertEqual(pool[0].job_id, "j11")
        self.assertIsNone(pool[0].role)
        self.assertNotIn("j11", [c.job_id for c in pool[1:]])

    def test_limits_and_the_given_listings_are_respected(self):
        pool = gather(text_match, self.listings, ["software development"], indices=[0, 2, 3, 4, 6], overall=2, per_role=1)
        self.assertEqual([c.job_id for c in pool], ["j0", "j4"])  # j4 is also software development's best

    def test_a_role_with_no_listings_is_skipped(self):
        pool = gather(text_match, self.listings, ["security"], overall=2, per_role=5)
        self.assertEqual(len(pool), 2)

    def test_missing_hand_off_says_what_to_run(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "ML_Rank section 13"):
                ListingIndex.from_handoff(JOB_IDS, Path(d))


if __name__ == "__main__":
    unittest.main()
