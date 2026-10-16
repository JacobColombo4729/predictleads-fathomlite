"""ML_Fit's hand-off for scoring a new person (src/fit_handoff.py), on made-up listings, people and O*NET ratings.
A tiny stand-in replaces BGE, so no model is loaded."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import fit_eval as fe
from src import fit_handoff as fh
from src import fit_model as fm
from tests.test_fit_eval import WORDS, fake_parts
from tests.test_fit_model import made_up_matrix, made_up_people


class FakeBge:
    """Deterministic word-hash embeddings with BGE's encode() signature."""
    dim = 16

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for w in str(text).split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n > 0, n, 1) if normalize_embeddings else out


class FitHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        fake_parts(cls.tmp.name)
        cls.listings = fe.role_listings(n_per_role=30, root=cls.tmp.name)
        cls.matrix = made_up_matrix()
        cls.model = FakeBge()
        people = made_up_people(n=200)
        texts = [WORDS[t] + " and some other words" for t in people.target_bucket]
        cls.people = people.assign(text=texts)
        cls.scorer = fh.RoleScorer.fit(cls.listings, cls.matrix, cls.model, n_per_role=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def notebook_scores(self):
        """The scores as ML_Fit computes them (cells 4a and 5a), with the stand-in BGE."""
        texts = self.people.text.tolist()
        s = {"keyword overlap": fe.keyword_scores(texts, self.listings), "tf-idf": fe.tfidf_scores(texts, self.listings),
             "bge": fe.bge_cosine(texts, self.model, fe.bge_roles(self.listings, self.model, 30))}
        s["rrf (tf-idf + bge)"] = fe.rrf_scores(s["tf-idf"], s["bge"])
        role_prof = fm.role_profiles(self.matrix)
        backgrounds = fm.background_profiles(self.people, self.matrix, role_prof)
        s["skills carry-over"] = fm.carry_over_scores(backgrounds, role_prof)
        s["skills profile cosine"] = fm.profile_cosine_scores(backgrounds, role_prof, self.matrix)
        return s

    def test_scorer_gives_the_notebooks_scores_exactly(self):
        expected, got = self.notebook_scores(), self.scorer.scores(self.people, self.model)
        self.assertEqual(list(got), fh.METHODS)
        self.assertEqual(list(expected), fh.METHODS)
        for method in fh.METHODS:
            self.assertEqual(fh.max_difference(got[method], expected[method]), 0.0, method)

    def test_features_match_the_notebooks_columns_and_values(self):
        expected = fm.features(self.people, self.notebook_scores())
        pd.testing.assert_frame_equal(self.scorer.features(self.people, self.model), expected)

    def test_one_new_person_scores_the_same_alone_as_in_a_batch(self):
        batch = self.scorer.features(self.people, self.model)
        alone = self.scorer.features(self.people.iloc[[7]], self.model)
        numeric = alone.select_dtypes("number").columns
        # a batch and a single row sum in a different order, so the last bits can differ
        self.assertLess(fh.max_difference(alone[numeric], batch.iloc[[7]][numeric]), 1e-12)

    def test_save_and_load_give_the_same_probabilities(self):
        from sklearn.base import clone
        X = self.scorer.features(self.people, self.model)
        y = np.array([fe.ROLES.index(t) for t in self.people.target_bucket])
        layer = clone(fm.candidates()["logistic regression"][0]).set_params(model__C=0.3).fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            manifest = fh.save(d, self.scorer, layer, X.columns, {"model": "logistic regression"})
            scorer, loaded, saved = fh.load(d)
        self.assertEqual(saved, manifest)
        self.assertEqual(saved["feature_columns"], list(X.columns))
        again = fh.probabilities(loaded, scorer.features(self.people, self.model), saved["feature_columns"])
        self.assertEqual(fh.max_difference(again, fm.probabilities(layer, X)), 0.0)
        self.assertTrue(np.allclose(again.sum(axis=1), 1.0))

    def test_load_refuses_a_hand_off_for_other_roles(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            fh.save(d, self.scorer, object(), ["a"], {})
            path = Path(d) / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps({**manifest, "roles": manifest["roles"][:-1]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "roles"):
                fh.load(d)

    def test_parity_helpers_handle_blanks_and_top_roles(self):
        nan = float("nan")
        self.assertEqual(fh.max_difference([[1.0, nan]], [[1.5, nan]]), 0.5)
        self.assertEqual(fh.max_difference([[1.0, nan]], [[1.0, 2.0]]), float("inf"))
        self.assertEqual(fh.max_difference([[nan]], [[nan]]), 0.0)
        a = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(fh.top_k_agreement(a, a + 1e-9, k=2), 1.0)
        self.assertEqual(fh.top_k_agreement(a, np.array([[0.5, 0.3, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]]), k=2), 0.5)

    def test_missing_inputs_are_named(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            fh.probabilities(object(), pd.DataFrame({"a": [1]}), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
