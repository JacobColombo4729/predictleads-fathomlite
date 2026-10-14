"""Skills-based matching and the trained layer (ML_Fit section 5, ladder steps 5 and 6).

Step 5, skills: a person's background and each role are both described by O*NET's Transferable Skills importance
ratings (1-5). The background is the O*NET occupations behind the person's background area (career_switch.BACKGROUND_SOC)
or, for a move within tech, their current role; a role is the O*NET occupations that map to it (career_switch.role_bucket).
Two scores per person and role:
  - carry-over: the share of the role's important skills (importance 3+, weighted by importance) that the background also
    rates 3+ (the "skills that carry over" of EDA_SDA_3 6f);
  - profile cosine: cosine between the two profiles after subtracting the average over every O*NET occupation, so the
    score reflects what is distinctive about each (every occupation rates every skill, so raw profiles all look alike).
Nothing here is fit on people: the scores come from O*NET alone, so a test person can't leak into them.

Step 6, the trained layer: a classifier per person over the role buckets, from the Profiles `feature` columns and the
scores of steps 1-5. Candidates are compared on the development folds only (out-of-fold probabilities, the folds of
fit_eval.split_people); the test people are scored once, by the chosen model fit on all development people.
Prints and returns aggregates only.
"""
import numpy as np
import pandas as pd

from src.career_switch import BACKGROUND_SOC, role_bucket
from src.fit_eval import ROLES, top_k_hits

IMPORTANT = 3.0  # O*NET importance at which a skill counts as important (as in onet.carry_over)

# The Profiles modeling table's `feature` columns (Profiles 5a); labels and outcomes never enter
CATEGORICAL = ["origin", "from_level", "background_area", "retrain_field", "from_bucket"]
FLAGS = ["retrained", "study_undated"]
NUMERIC = ["years_before", "n_jobs_before", "study_level_before"]


# ---------- step 5: skills-based matching ----------

def role_profiles(matrix):
    """Each role's skill profile: the mean rating per skill over the O*NET occupations that map to the role.
    matrix: occupation x skill (onet.skill_matrix). Returns a roles x skills DataFrame (NaN row if no occupation maps)."""
    buckets = pd.Series({c: role_bucket(c) for c in matrix.index})
    return pd.DataFrame({r: matrix.loc[buckets.index[buckets.eq(r)]].mean() for r in ROLES}).T.reindex(columns=matrix.columns)


def background_profiles(people, matrix, roles=None):
    """Each person's background profile: their current role's profile for a move within tech (from_bucket), else the
    mean over the O*NET occupations of their background area (BACKGROUND_SOC); NaN for "other" and "unknown"."""
    roles = role_profiles(matrix) if roles is None else roles
    areas = {}
    for area, prefixes in BACKGROUND_SOC.items():
        rows = matrix[matrix.index.str.startswith(tuple(prefixes))]
        areas[area] = rows.mean() if len(rows) else pd.Series(np.nan, index=matrix.columns)
    empty = pd.Series(np.nan, index=matrix.columns)
    out = [roles.loc[fb] if isinstance(fb, str) and fb in roles.index else areas.get(area, empty)
           for fb, area in zip(people.from_bucket, people.background_area)]
    return pd.DataFrame(out, index=people.index, columns=matrix.columns)


def carry_over_scores(backgrounds, roles):
    """People x roles: the importance-weighted share of each role's important skills that the background rates 3+.
    0 for a person with no background profile."""
    t = roles.to_numpy(dtype=float)
    w = np.where(t >= IMPORTANT, t, 0.0)                                   # the role's important skills, by importance
    has = (np.nan_to_num(backgrounds.to_numpy(dtype=float), nan=0.0) >= IMPORTANT).astype(float)
    total = w.sum(axis=1)
    return (has @ w.T) / np.where(total > 0, total, 1.0)


def profile_cosine_scores(backgrounds, roles, matrix):
    """People x roles: cosine between the background and role profiles, each minus the all-occupation average.
    0 for a person with no background profile."""
    mean = matrix.mean()
    b = (backgrounds - mean).fillna(0).to_numpy(dtype=float)
    r = (roles - mean).fillna(0).to_numpy(dtype=float)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    return (b / np.where(bn > 0, bn, 1.0)) @ (r / np.where(rn > 0, rn, 1.0)).T


# ---------- step 6: the trained layer ----------

def features(people, scores):
    """The trained layer's inputs: the Profiles feature columns plus every method's score for every role, both raw and
    relative to the person's own mean over the roles (so a person whose text matches everything isn't read as a fit for
    everything). scores: {method: people x roles array}."""
    X = people[CATEGORICAL + FLAGS + NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(object).where(X[c].notna(), "none").astype(str)
    for c in FLAGS:
        X[c] = X[c].astype("Float64").fillna(0).astype(float)
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    cols = {}
    for m, s in scores.items():
        s = np.asarray(s, dtype=float)
        name = m.split(" (")[0].replace(" ", "_").replace("-", "")
        for j, r in enumerate(ROLES):
            cols[f"{name}: {r}"] = s[:, j]
            cols[f"{name} vs own mean: {r}"] = s[:, j] - s.mean(axis=1)
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)  # one concat, not a column at a time


def _preprocess(scale):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), *([StandardScaler()] if scale else []))
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), CATEGORICAL)],
                             remainder=numeric, verbose_feature_names_out=False)


def candidates(seed=12345):
    """{family: (pipeline, parameter grid)}: the course's classifiers, gradient boosting, and a small neural network.
    Grids are small on purpose: 2,250 development people."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    def pipe(model, scale=False):
        return Pipeline([("prep", _preprocess(scale)), ("model", model)])
    return {
        "logistic regression": (pipe(LogisticRegression(max_iter=3000), scale=True),
                                {"model__C": [0.03, 0.3, 3.0], "model__class_weight": [None, "balanced"]}),
        "random forest": (pipe(RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)),
                          {"model__min_samples_leaf": [3, 10], "model__max_features": ["sqrt", 0.3]}),
        "lightgbm": (pipe(LGBMClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, random_state=seed, verbose=-1)),
                     {"model__num_leaves": [7, 15], "model__min_child_samples": [20, 50]}),
        "xgboost": (pipe(XGBClassifier(n_estimators=300, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                       random_state=seed, n_jobs=-1, verbosity=0)),
                    {"model__max_depth": [2, 4], "model__min_child_weight": [1, 5]}),
        "catboost": (pipe(CatBoostClassifier(iterations=400, learning_rate=0.05, random_seed=seed, verbose=0,
                                             thread_count=-1)),
                     {"model__depth": [3, 5]}),
        "neural network (MLP)": (pipe(MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, max_iter=500,
                                                    random_state=seed), scale=True),
                                 {"model__alpha": [1e-3, 1e-1], "model__hidden_layer_sizes": [(16,), (32,)]}),
    }


def probabilities(model, X):
    """people x roles probabilities in ROLES order (0 for a role the model never saw)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(ROLES)))
    out[:, np.asarray(model.classes_, dtype=int)] = p
    return out


def hit_rate_scorer(k=3):
    """A GridSearchCV scorer: top-k hit rate, the current role left out for moves within tech (as fit_eval.evaluate)."""
    def score(estimator, X, y):
        exclude = X["from_bucket"].where(X["from_bucket"].ne("none"), None).tolist()
        return top_k_hits(probabilities(estimator, X), [ROLES[i] for i in y], exclude, k).mean()
    return score


def compare(X, y, fold, seed=12345, k=3):
    """Grid search of every candidate on the development folds (the folds given, never the test people).
    X, y: development people only (y as ROLES positions); fold: their fold numbers. Returns (table, fitted searches):
    the table has each family's best settings and its mean and std top-k hit rate over the folds."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    split = PredefinedSplit(fold.loc[X.index].to_numpy())
    rows, searches = [], {}
    for family, (pipe, grid) in candidates(seed).items():
        gs = GridSearchCV(clone(pipe), grid, scoring=hit_rate_scorer(k), cv=split, n_jobs=1, refit=True)
        gs.fit(X, y)
        best = gs.best_index_
        rows.append({"model": family, f"top-{k} hit rate (mean)": gs.cv_results_["mean_test_score"][best],
                     "std": gs.cv_results_["std_test_score"][best], "settings": gs.best_params_})
        searches[family] = gs
    table = pd.DataFrame(rows).set_index("model").sort_values(f"top-{k} hit rate (mean)", ascending=False)
    return table, searches


def out_of_fold(estimator, X, y, fold):
    """Development people's probabilities, each from the model fit on the other folds."""
    from sklearn.base import clone
    out = np.zeros((len(X), len(ROLES)))
    f = fold.loc[X.index].to_numpy()
    for k in np.unique(f):
        m = clone(estimator).fit(X[f != k], y[f != k])
        out[f == k] = probabilities(m, X[f == k])
    return out
