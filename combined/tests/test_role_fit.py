"""Step 3, role fit at serving time (combined/role_fit.py), on a hand-off fit to made-up people and listings.
A tiny stand-in replaces BGE; no data/, no model download.

    python -m unittest discover -s combined/tests -t .
"""
import hashlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import combined._paths  # noqa: F401  puts analysis/ on the path, for `src`
from combined.role_fit import RoleFit
from src import fit_eval as fe
from src import fit_handoff as fh
from src import fit_model as fm
from src import rank_model as rm

WORDS = {"it support": "help desk support technician", "networks and systems": "network systems administrator",
         "security": "security analyst threat", "database": "database administrator sql",
         "software development": "software developer python", "quality assurance": "qa tester test automation",
         "data science": "data scientist analyst analytics", "it project management": "it project manager scrum"}
SOC = {"it support": "15-1232.00", "networks and systems": "15-1244.00", "security": "15-1212.00",
       "database": "15-1242.00", "software development": "15-1252.00", "quality assurance": "15-1253.00",
       "data science": "15-2051.00", "it project management": "15-1299.09"}


class FakeBge:
    dim = 16

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for w in str(text).split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 0, n, 1)


def made_up_handoff(directory):
    """A hand-off as ML_Fit 9b writes it, fit to made-up listings and people."""
    rng = np.random.default_rng(0)
    listings = pd.DataFrame([(r, WORDS[r].split()[0], f"{WORDS[r]} listing {i}") for r in fe.ROLES for i in range(20)],
                            columns=["role", "title", "text"])
    skills = ["Programming", "Troubleshooting", "Negotiation", "Instructing"]
    matrix = pd.DataFrame({code: rng.uniform(1, 5, len(skills)) for code in [*SOC.values(), "25-2021.00", "41-2031.00"]},
                          index=skills).T
    n = 240
    target = rng.choice(fe.ROLES, n)
    people = pd.DataFrame({"origin": "other-field job", "from_level": None,
                           "background_area": rng.choice(["education", "sales and retail"], n),
                           "retrain_field": rng.choice(["data science and analytics", "networking"], n), "from_bucket": None,
                           "retrained": True, "study_undated": False, "years_before": rng.uniform(0, 10, n),
                           "n_jobs_before": rng.integers(0, 4, n), "study_level_before": rng.integers(0, 4, n).astype(float),
                           "text": [WORDS[t] for t in target], "target_bucket": target}, index=[f"p{i}" for i in range(n)])
    model = FakeBge()
    scorer = fh.RoleScorer.fit(listings, matrix, model, n_per_role=20)
    X = scorer.features(people, model)
    y = np.array([fe.ROLES.index(t) for t in target])
    from sklearn.base import clone
    layer = clone(fm.candidates()["logistic regression"][0]).set_params(model__C=0.3).fit(X, y)
    fh.save(directory, scorer, layer, X.columns, {"model": "logistic regression"}, explain_background=X)
    return model


def entry(kind, title, dates):
    return {"kind": kind, "title": title, "organization": None, "description": f"{dates}. Did the work."}


ANALYST_TO_BE = [entry("job", "Middle School Teacher", "Aug 2015 - Jun 2023"),
                 entry("education", "Data Analytics Bootcamp", "Jan 2024 - Aug 2024")]


class RoleFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model = made_up_handoff(cls.tmp.name)
        cls.fit = RoleFit.load(cls.tmp.name, bge=model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_role_rated_with_the_best_at_100(self):
        result = self.fit.roles(ANALYST_TO_BE, as_of="2026-10")
        t = result.table
        self.assertEqual(sorted(t.index), sorted(fe.ROLES))
        self.assertAlmostEqual(t.rating.iloc[0], 100.0)
        self.assertTrue((t.rating.diff().dropna() <= 1e-12).all())  # best first
        self.assertAlmostEqual(t.probability.sum(), 1.0)
        self.assertEqual(result.case, "career change")
        self.assertEqual(result.undated_entries, 0)
        self.assertEqual(len(result.top(3)), 3)

    def test_rating_is_ml_ranks_fit_rating(self):
        t = self.fit.roles(ANALYST_TO_BE, as_of="2026-10").table.reindex(fe.ROLES)
        expected = rm.fit_ratings(t.probability.to_numpy()[None, :]).to_numpy()[0]
        self.assertLess(fh.max_difference(t.rating.to_numpy(), expected), 1e-12)

    def test_reasons_are_ml_fits_groups_and_add_up_to_the_log_odds_shift(self):
        result = self.fit.roles(ANALYST_TO_BE, as_of="2026-10")
        self.assertEqual(result.reason_groups, [*fm.GROUPS, "other"])
        why = result.table[[f"why: {g}" for g in result.reason_groups]]
        self.assertTrue(np.isfinite(why.to_numpy()).all())

    def test_growth_moves_scores_only_when_weighted(self):
        base = self.fit.roles(ANALYST_TO_BE, as_of="2026-10").table
        same = self.fit.roles(ANALYST_TO_BE, growth={"security": 0.3}, growth_weight=0.0, as_of="2026-10").table
        pd.testing.assert_frame_equal(base, same)
        moved = self.fit.roles(ANALYST_TO_BE, growth={"security": 1000.0}, growth_weight=1.0, as_of="2026-10").table
        self.assertAlmostEqual(moved.loc["security", "score"], 1001.0 * base.loc["security", "probability"])
        self.assertEqual(moved.index[0], "security")
        self.assertAlmostEqual(moved.loc["security", "probability"], base.loc["security", "probability"])

    def test_missing_hand_off_says_what_to_run(self):
        with self.assertRaisesRegex(FileNotFoundError, "ML_Fit 9b"):
            RoleFit.load(self.tmp.name + "_missing", bge=FakeBge())


if __name__ == "__main__":
    unittest.main()
