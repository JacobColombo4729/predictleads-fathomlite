"""What the app needs to score a new person with ML_Fit's trained layer (ML_Fit 9b; combined model step 1).

ML_Fit scores the Profiles people once, with every method's role side computed on the fly. A new person needs the role
side kept: RoleScorer holds it (each role's keyword vocabulary, the TF-IDF vectorizer and role centroids, the BGE role
centroids, the O*NET role and background-area skill profiles) and scores people with the same functions ML_Fit uses
(fit_eval, fit_model), so the scores can't drift from the notebook's. The trained layer is the chosen Pipeline fit on
all development people, the same fit that scored the test people in 5c.

    save(directory, scorer, layer, feature_columns, info)   -> role_scorer.joblib, fit_layer.joblib, manifest.json
    load(directory)                                         -> (scorer, layer, manifest); refuses a manifest that
                                                               doesn't match this code's roles or methods
Saved to data/clean/ml_fit/ (git-ignored). Nothing here reads people's text; the scorer holds role-side aggregates
(word lists, centroids, profiles), never a listing or a person.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import fit_eval as fe
from src import fit_model as fm

# The method names, in the order ML_Fit builds `scores` before the trained layer (cells 4a and 5a); the trained layer's
# input columns follow this order (fit_model.features)
METHODS = ["keyword overlap", "tf-idf", "bge", "rrf (tf-idf + bge)", "skills carry-over", "skills profile cosine"]
FORMAT = 1


class RoleScorer:
    """The role side of every ML_Fit method, kept so a new person can be scored as the Profiles people were."""

    def __init__(self, vocab, tfidf_vec, tfidf_centroids, bge_centroids, role_prof, matrix):
        self.vocab, self.tfidf_vec, self.tfidf_centroids = vocab, tfidf_vec, tfidf_centroids
        self.bge_centroids, self.role_prof, self.matrix = bge_centroids, role_prof, matrix

    @classmethod
    def fit(cls, listings, matrix, bge_model, n_per_role=300, role_prof=None):
        """listings: fit_eval.role_listings; matrix: onet.skill_matrix of Transferable Skills; bge_model: fe.bge_model()."""
        vec, centroids = fe.tfidf_roles(listings)
        return cls(fe.keyword_vocab(listings), vec, centroids, fe.bge_roles(listings, bge_model, n_per_role),
                   fm.role_profiles(matrix) if role_prof is None else role_prof, matrix)

    def scores(self, people, bge_model):
        """{method: people x roles} in METHODS order, the trained layer's score inputs.
        people: a DataFrame with `text` (the pre-switch background), `from_bucket` and `background_area`."""
        texts = people["text"].fillna("").tolist()
        out = {"keyword overlap": fe.keyword_overlap(texts, self.vocab),
               "tf-idf": fe.tfidf_cosine(texts, self.tfidf_vec, self.tfidf_centroids),
               "bge": fe.bge_cosine(texts, bge_model, self.bge_centroids)}
        out["rrf (tf-idf + bge)"] = fe.rrf_scores(out["tf-idf"], out["bge"])
        backgrounds = fm.background_profiles(people, self.matrix, self.role_prof)
        out["skills carry-over"] = fm.carry_over_scores(backgrounds, self.role_prof)
        out["skills profile cosine"] = fm.profile_cosine_scores(backgrounds, self.role_prof, self.matrix)
        return out

    def features(self, people, bge_model):
        """The trained layer's inputs for these people: the Profiles feature columns plus every method's scores."""
        return fm.features(people, self.scores(people, bge_model))


def save(directory, scorer, layer, feature_columns, info):
    """Write the scorer, the fitted trained layer and a manifest (roles, methods, columns, versions, `info`)."""
    import joblib
    import sklearn

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, directory / "role_scorer.joblib")
    joblib.dump(layer, directory / "fit_layer.joblib")
    manifest = {"format": FORMAT, "roles": list(fe.ROLES), "methods": METHODS, "feature_columns": list(feature_columns),
                "sklearn_version": sklearn.__version__, **info}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load(directory):
    """(scorer, layer, manifest), after checking the manifest matches this code (roles, methods, format)."""
    import joblib

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    problems = [name for name, saved, now in [("format", manifest.get("format"), FORMAT),
                                              ("roles", manifest.get("roles"), list(fe.ROLES)),
                                              ("methods", manifest.get("methods"), METHODS)] if saved != now]
    if problems:
        raise ValueError(f"ML_Fit's hand-off doesn't match this code ({', '.join(problems)}): rerun ML_Fit 9b.")
    return joblib.load(directory / "role_scorer.joblib"), joblib.load(directory / "fit_layer.joblib"), manifest


def probabilities(layer, X, feature_columns):
    """people x roles probabilities from the saved layer, with the columns in the order it was fit on."""
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(f"{len(missing)} trained-layer inputs are missing, e.g. {missing[0]!r}")
    return fm.probabilities(layer, X[list(feature_columns)])


def max_difference(a, b):
    """The largest absolute difference between two matrices (the hand-off's parity check). Blank in both counts as
    equal (some Profiles inputs are blank and imputed by the layer); blank in only one is an infinite difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    blank_a, blank_b = np.isnan(a), np.isnan(b)
    if (blank_a != blank_b).any():
        return float("inf")
    both = ~blank_a
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def top_k_agreement(a, b, k=3):
    """Share of rows whose top-k roles (as a set) are the same in two score matrices."""
    top = lambda s: [frozenset(row) for row in np.argsort(-np.asarray(s, dtype=float), axis=1, kind="stable")[:, :k]]
    return float(np.mean([x == y for x, y in zip(top(a), top(b))])) if len(a) else 1.0
