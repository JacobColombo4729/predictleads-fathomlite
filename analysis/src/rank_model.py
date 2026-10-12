"""Scaffolding for ML_Rank, the listing model and the priority queue (v1 draft; not yet run on real data).

Targets (closes within N days, days open, survival), the time split with company-grouped folds, leakage-safe counts,
the baselines (newest first, title keyword search), the model list, ranking and survival metrics, and the priority
formula with its per-listing breakdown. Input is a listing table with: id, company_id, title, first_seen, last_seen.
Prints and returns aggregates only.
"""
import itertools
import re

import numpy as np
import pandas as pd

EXPORT_END = pd.Timestamp("2026-09-30")  # the data's end date (STANDARDS.md)
LIVE_TOLERANCE_DAYS = 2  # last seen within this many days of the end = still open (sensitivity check in ML_Rank 11)
SPLIT = {"train": ("2026-03-30", "2026-07-31"), "validation": ("2026-08-01", "2026-08-31"),
         "test": ("2026-09-01", "2026-09-30")}  # by first_seen; the data's first two days (partial_month) join train
SEED = 12345


def _naive(s):
    """Datetimes without a time zone, so comparisons and differences are plain."""
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s


# ---------- targets ----------

def survival_target(df, end=EXPORT_END, tolerance=LIVE_TOLERANCE_DAYS, as_of=None):
    """duration (days from first_seen to last_seen) and event (1 = it closed, 0 = still open, so censored).
    With as_of, only listings first seen by that date are kept, and anything last seen within the tolerance of it
    (or later) is censored there: the view a date-picker model is allowed to train on."""
    end = pd.Timestamp(as_of) if as_of is not None else end
    first, last = _naive(df["first_seen"]), _naive(df["last_seen"])
    keep = first <= end
    last = last.clip(upper=end)
    out = pd.DataFrame({"id": df["id"].values, "duration": ((last - first).dt.total_seconds() / 86400).clip(lower=0).values,
                        "event": (((end - last).dt.total_seconds() / 86400) > tolerance).astype(int).values}, index=df.index)
    return out.loc[keep]


def closes_within(target, n_days):
    """1 = closed within n days, 0 = stayed open past n days, NaN = censored before n days (unknown, not False)."""
    out = pd.Series(np.nan, index=target.index)
    out[target["duration"] > n_days] = 0.0
    out[(target["event"] == 1) & (target["duration"] <= n_days)] = 1.0
    return out


def days_open(target):
    """Days open for closed listings only (regression target); NaN for censored ones."""
    return target["duration"].where(target["event"] == 1)


# ---------- split ----------

def time_split(first_seen):
    """'train' / 'validation' / 'test' by first_seen (April-July / August / September)."""
    t = _naive(first_seen)
    def upto(day):
        return t < pd.Timestamp(day) + pd.Timedelta(days=1)
    out = pd.Series("outside", index=first_seen.index)
    out[upto(SPLIT["train"][1])] = "train"
    out[~upto(SPLIT["train"][1]) & upto(SPLIT["validation"][1])] = "validation"
    out[~upto(SPLIT["validation"][1]) & upto(SPLIT["test"][1])] = "test"
    return out


def company_folds(company_id, n_splits=5):
    """Cross-validation folds grouped by company: a list of (train_positions, valid_positions)."""
    from sklearn.model_selection import GroupKFold
    groups = pd.Series(company_id).astype(str).values
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


# ---------- leakage-safe features ----------

def prior_company_counts(df):
    """For each listing, how many listings its company first posted strictly before it (earlier copies only).
    Ties on first_seen don't count, so a same-moment batch doesn't count itself."""
    order = pd.DataFrame({"c": df["company_id"].values, "t": _naive(df["first_seen"]).values}, index=df.index)
    return (order.groupby("c")["t"].rank(method="min") - 1).astype(int)


def known_at_first_sight(df, flag_cols, has_desc="has_desc"):
    """Leakage audit for has_desc and the text flags: share of listings where a flag is set although has_desc is
    False (should be 0: flags are unknown, not False, without a description)."""
    rows = [{"feature": c, "set_without_description": float(((~df[has_desc].astype(bool)) & df[c].notna()).mean())}
            for c in flag_cols]
    return pd.DataFrame(rows)


# ---------- baselines ----------

def newest_first_score(df):
    """Score for the newest-first baseline: later first_seen = higher."""
    return pd.Series(_naive(df["first_seen"]).astype("int64").values / 1e9, index=df.index)


def title_keyword_score(df, keywords):
    """Score for title keyword search: how many searched words appear in the title (ties broken by newest first)."""
    titles = df["title"].fillna("").str.lower()
    hits = sum(titles.str.contains(r"\b" + re.escape(k.lower()) + r"\b", regex=True).astype(int) for k in keywords)
    tie = newest_first_score(df)
    tie = (tie - tie.min()) / (tie.max() - tie.min() + 1e-9) * 0.5  # under 1, so it only breaks ties
    return hits + tie


def dummy_baseline(strategy="prior"):
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy=strategy, random_state=SEED)


# ---------- models ----------

def build_models(seed=SEED):
    """Classifiers for 'closes within N days'. Put each after the notebook's ColumnTransformer in a Pipeline; the
    scaler inside the logistic regression is part of that model only. Boosting libraries are imported here."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=seed),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=seed, verbose=-1),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, eval_metric="logloss"),
        "catboost": CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced", random_seed=seed, verbose=0),
    }


def cox_model(penalizer=0.1):
    """Cox proportional hazards (lifelines); fit with duration_col='duration', event_col='event', cluster_col='company_id'."""
    from lifelines import CoxPHFitter
    return CoxPHFitter(penalizer=penalizer)


def lambdarank_model(seed=SEED):
    """LightGBM LambdaRank; fit(X, relevance, group=listings_per_query)."""
    from lightgbm import LGBMRanker
    return LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, random_state=seed, verbose=-1)


# ---------- metrics ----------

def precision_at_k(y_true, score, k=10):
    """Share of the k highest-scored items that are positive."""
    y = np.asarray(y_true, dtype=float)
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    return float(y[top].mean())


def ndcg_at_k(relevance, score, k=10):
    """NDCG@k with graded relevance (gain = relevance, log2 discount)."""
    rel = np.asarray(relevance, dtype=float)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    dcg = float((rel[order] / np.log2(np.arange(2, len(order) + 2))).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def c_index(duration, event, predicted_open_days):
    """Harrell's C: does the model order listings by how long they stay open? (higher prediction = stays open longer)"""
    from lifelines.utils import concordance_index
    return float(concordance_index(duration, predicted_open_days, event))


# ---------- the priority queue ----------

FACTORS = ["fit", "still_open", "accessibility", "pay"]
# Placeholder weights: set from EDA_SDA_4's adjusted model, then tune on the validation month (tune_weights).
START_WEIGHTS = {"fit": 0.4, "still_open": 0.3, "accessibility": 0.2, "pay": 0.1}


def rating_0_100(values):
    """Rank-based 0-100 rating of a factor within the listings being ranked (missing = 50, neutral)."""
    return (pd.Series(values, dtype=float).rank(pct=True) * 100).fillna(50.0)


def priority_breakdown(factors, weights=None):
    """factors: DataFrame with one column per FACTORS entry. Returns each factor's 0-100 rating, its part of the
    score (rating x normalised weight; the parts add up to the score) and the total 'score'."""
    w = pd.Series(weights or START_WEIGHTS, dtype=float).reindex(FACTORS).fillna(0.0)
    w = w / w.sum()
    ratings = pd.DataFrame({f: rating_0_100(factors[f]).values for f in FACTORS}, index=factors.index)
    parts = ratings * w
    out = ratings.add_prefix("rating_").join(parts.add_prefix("part_"))
    out["score"] = parts.sum(axis=1)
    return out


def tune_weights(factors, outcome, k=10, grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Coarse grid search of the weights on the validation month: best precision@k of `outcome` (1 = stayed open
    long enough). Returns (best_weights, best_precision). The coarse grid keeps weights interpretable."""
    best, best_p = None, -1.0
    for combo in itertools.product(grid, repeat=len(FACTORS)):
        if sum(combo) == 0:
            continue
        w = dict(zip(FACTORS, combo))
        p = precision_at_k(outcome, priority_breakdown(factors, w)["score"].values, k)
        if p > best_p:
            best, best_p = w, p
    return best, best_p
