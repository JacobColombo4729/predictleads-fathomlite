"""Role discovery evaluation (src/fit_eval.py) on made-up listings and people. No real data."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import fit_eval as fe

CODES = {"it support": "15-1232.00", "networks and systems": "15-1211.00", "security": "15-1212.00",
         "database": "15-1242.00", "software development": "15-1251.00", "quality assurance": "15-1253.00",
         "data science": "15-1221.00", "it project management": "15-1299.09"}
WORDS = {"it support": "help desk support technician tickets", "networks and systems": "network systems administrator routers",
         "security": "security analyst threat incident", "database": "database administrator sql backups",
         "software development": "software developer python code", "quality assurance": "qa tester test automation",
         "data science": "data scientist machine learning models", "it project management": "it project manager scrum delivery"}


def fake_parts(root, per_role=30):
    rows = [{"id": f"{role[:3]}-{i}", "attributes.title": WORDS[role].split(" ", 2)[0] + " " + WORDS[role].split()[1],
             "attributes.description": f"<p>{WORDS[role]} listing number {i}</p>", "attributes.onet_data.code": code}
            for role, code in CODES.items() for i in range(per_role)]
    rows.append({"id": "other-1", "attributes.title": "nurse", "attributes.description": "patients",
                 "attributes.onet_data.code": "29-1141.00"})  # not a role bucket: left out
    folder = Path(root) / fe.SCOPE_TABLE
    folder.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.iloc[: len(df) // 2].to_pickle(folder / "part_00000.pkl")
    df.iloc[len(df) // 2:].to_pickle(folder / "part_00001.pkl")


class FitEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        fake_parts(self.tmp.name)

    def test_role_listings_caps_each_role_and_is_reproducible(self):
        a = fe.role_listings(n_per_role=10, seed=1, root=self.tmp.name)
        b = fe.role_listings(n_per_role=10, seed=1, root=self.tmp.name)
        self.assertEqual(a.role.value_counts().to_dict(), {r: 10 for r in fe.ROLES})
        pd.testing.assert_frame_equal(a, b)
        self.assertNotIn("<p>", " ".join(a.text))
        self.assertFalse(a.equals(fe.role_listings(n_per_role=10, seed=2, root=self.tmp.name)))

    def test_word_methods_rank_the_matching_role_first(self):
        listings = fe.role_listings(n_per_role=30, root=self.tmp.name)
        people = [WORDS[r] for r in fe.ROLES]
        for scores in (fe.keyword_scores(people, listings), fe.tfidf_scores(people, listings)):
            self.assertEqual(list(np.argmax(scores, axis=1)), list(range(len(fe.ROLES))))
        fused = fe.rrf_scores(fe.keyword_scores(people, listings), fe.tfidf_scores(people, listings))
        self.assertEqual(list(np.argmax(fused, axis=1)), list(range(len(fe.ROLES))))

    def test_top_k_hits_and_excluding_the_current_role(self):
        scores = np.array([[3.0, 2.0, 1.0, 0, 0, 0, 0, 0]])  # it support, then networks, then security
        self.assertEqual(fe.top_k_hits(scores, ["security"], k=2)[0], 0)
        self.assertEqual(fe.top_k_hits(scores, ["security"], exclude=["it support"], k=2)[0], 1)
        self.assertEqual(fe.top_k_hits(scores, ["security"], exclude=[None], k=3)[0], 1)

    def test_ranks_break_ties_by_role_order(self):
        self.assertEqual(list(fe.ranks(np.zeros((1, 8)))[0]), list(range(1, 9)))

    def test_bootstrap_interval_brackets_the_mean(self):
        mean, lo, hi = fe.bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
        self.assertAlmostEqual(mean, 0.7)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)

    def test_split_keeps_test_people_out_of_every_fold(self):
        rng = np.random.default_rng(0)
        people = pd.DataFrame({"target_bucket": rng.choice(fe.ROLES, 400), "pool": rng.choice(["within tech", "retrained"], 400)},
                              index=[f"p{i}" for i in range(400)])
        dev, test, fold = fe.split_people(people)
        self.assertEqual(len(set(dev) & set(test)), 0)
        self.assertEqual(len(dev) + len(test), 400)
        self.assertEqual(sorted(fold.unique()), [0, 1, 2, 3, 4])

    def test_evaluate_reports_every_method_pool_and_k(self):
        listings = fe.role_listings(n_per_role=30, root=self.tmp.name)
        people = pd.DataFrame({"target_bucket": fe.ROLES, "pool": ["a", "b"] * 4,
                               "from_bucket": [None] * 7 + ["it support"]})
        text = [WORDS[r] for r in fe.ROLES]
        table = fe.evaluate({"tf-idf": fe.tfidf_scores(text, listings)}, people, train_targets=fe.ROLES)
        self.assertEqual(set(table.method), {"most common role (baseline)", "tf-idf"})
        self.assertEqual(set(table.pool), {"all career changers", "a", "b"})
        self.assertEqual(set(table.k), {3, 5})
        tfidf_all = table.query("method == 'tf-idf' and k == 3 and pool == 'all career changers'").hit_rate.iloc[0]
        self.assertEqual(tfidf_all, 1.0)


class SameEveryRun(unittest.TestCase):
    """Keyword overlap must not depend on Python's string-hash seed, which changes with every new process (a kernel
    restart): words tied at the top-words cutoff once came out in set (hash) order, and the trained layer's choice flipped."""

    SCRIPT = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); import pandas as pd; from src import fit_eval as fe\n"
        "titles = ['alpha bravo charlie delta echo foxtrot golf hotel india juliet'] * 3\n"  # ten words, all tied
        "listings = pd.DataFrame({'role': fe.ROLES[0], 'title': titles})\n"
        "people = ['alpha', 'bravo charlie', 'hotel india juliet', 'delta golf']\n"  # each has some of the ten words
        "print(json.dumps(fe.keyword_scores(people, listings, top_words=4).tolist()))\n")

    def test_keyword_overlap_is_the_same_whatever_the_hash_seed(self):
        import os
        import subprocess
        root = str(Path(__file__).resolve().parents[1])
        runs = {subprocess.run([sys.executable, "-c", self.SCRIPT, root], capture_output=True, text=True, check=True,
                               env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2", "3")}
        self.assertEqual(len(runs), 1)


if __name__ == "__main__":
    unittest.main()
