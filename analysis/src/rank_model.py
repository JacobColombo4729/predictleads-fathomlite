"""ML_Rank, the listing model and the priority queue (v2 draft; not yet run on real data).

Targets (closes within N days, days open, survival), the time split with company-grouped folds, leakage-safe counts,
the baselines (newest first, title keyword search), preprocessing (ColumnTransformer), class imbalance and threshold
tuning, the model list with grid search, Cox with proportional-hazards checks, AFT models, survival forests,
LambdaRank training, ranking and survival metrics, the company-clustered bootstrap, backtest relevance, and the
priority formula with its per-listing breakdown. Input is a listing table with: id, company_id, title, first_seen,
last_seen. Prints and returns aggregates only.
"""
import itertools
import re

import numpy as np
import pandas as pd

from .as_of import LIVE_TOLERANCE_DAYS  # one definition of "still open" for the targets and the date picker

EXPORT_END = pd.Timestamp("2026-09-30")  # the data's end date (STANDARDS.md)
CLOSING_DAYS = 14  # THE closing window: the main target "closes within N days" and "open long enough" (first seen + N days). Change it here only.
SENSITIVITY_DAYS = (7, 14, 30)  # windows the sensitivity check (ML_Rank 11) repeats the main result for
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


def closes_within(target, n_days=CLOSING_DAYS):
    """1 = closed within n days, 0 = stayed open past n days, NaN = censored before n days (unknown, not False)."""
    out = pd.Series(np.nan, index=target.index)
    out[target["duration"] > n_days] = 0.0
    out[(target["event"] == 1) & (target["duration"] <= n_days)] = 1.0
    return out


def open_long_enough(target, n_days=CLOSING_DAYS):
    """1 = still open n days after the listing was first seen, 0 = closed within n days, NaN = censored before n days.
    The opposite of closes_within, with the same unknown cases."""
    return 1.0 - closes_within(target, n_days)


def sensitivity_windows(target, windows=SENSITIVITY_DAYS):
    """closes_within for every window in one table (columns closes_within_7 ...), for the sensitivity check."""
    return pd.DataFrame({f"closes_within_{n}": closes_within(target, n) for n in windows})


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


# ---------- preprocessing, class imbalance, threshold ----------

def make_preprocessor(numeric, categorical, min_frequency=50):
    """ColumnTransformer: numbers get the median for blanks (plus a was-blank flag); categories get one-hot columns
    with rare levels grouped into one 'infrequent' column and unseen levels handled. Put it first in a Pipeline so it
    is fit on training folds only. Scaling (for the logistic regression) is inside that model; trees need none."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, list(numeric)), ("cat", cat, list(categorical))], remainder="drop")


def with_preprocessing(preprocessor, model):
    """Pipeline(prep, model): the form every model is cross-validated and fit in. The preprocessor is cloned, so
    pipelines built from one shared PREP never refit each other's transformer."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    return Pipeline([("prep", clone(preprocessor)), ("model", model)])


def upsample_minority(X, y, seed=SEED):
    """Repeat minority-class rows (random, with replacement) until both classes are the same size. Training rows only:
    never upsample before splitting, or copies of one row land on both sides."""
    X, y = pd.DataFrame(X).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    counts = y.value_counts()
    if len(counts) < 2 or counts.iloc[0] == counts.iloc[1]:
        return X, y
    extra = y[y == counts.idxmin()].sample(counts.max() - counts.min(), replace=True, random_state=seed).index
    keep = np.concatenate([np.arange(len(y)), extra.values])
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def best_threshold(y_true, proba, metric="f1", grid=None):
    """Decision threshold that maximises `metric` ('f1', 'balanced_accuracy' or 'youden') on the validation month.
    Choose it there, then apply it unchanged to the test month. Returns (threshold, score)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    fn = {"f1": lambda yp: f1_score(y, yp, zero_division=0),
          "balanced_accuracy": lambda yp: balanced_accuracy_score(y, yp),
          "youden": lambda yp: balanced_accuracy_score(y, yp) * 2 - 1}[metric]
    scores = [fn((p >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))  # first best on ties: the lower threshold
    return float(grid[i]), float(scores[i])


# ---------- grid search ----------

PARAM_GRIDS = {  # keys carry the "model__" prefix of with_preprocessing()
    "logistic": {"model__logisticregression__C": [0.01, 0.1, 1.0]},
    "tree": {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [20, 50, 100]},
    "random_forest": {"model__max_depth": [8, 12, None], "model__min_samples_leaf": [10, 20, 50]},
    "lightgbm": {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.05, 0.1]},
    "xgboost": {"model__max_depth": [3, 5, 7], "model__learning_rate": [0.03, 0.05, 0.1]},
    "catboost": {"model__depth": [4, 6, 8], "model__learning_rate": [0.03, 0.05, 0.1]},
}


def grid_search(pipeline, param_grid, X, y, folds, scoring="roc_auc", n_jobs=1):
    """GridSearchCV over the company-grouped folds (a list of (train, valid) positions). Returns the fitted search;
    read best_params_, best_score_ and cv_results_ (mean and std per setting). The notebook times it with %%time."""
    from sklearn.model_selection import GridSearchCV
    return GridSearchCV(pipeline, param_grid, cv=folds, scoring=scoring, n_jobs=n_jobs, refit=True).fit(X, y)


def cv_auc(pipeline, X, y, folds):
    """Cross-validated ROC-AUC per fold (array): report mean and std."""
    from sklearn.model_selection import cross_val_score
    return cross_val_score(pipeline, X, y, cv=folds, scoring="roc_auc")


# ---------- survival models ----------

def cox_model(penalizer=0.1):
    """Cox proportional hazards (lifelines); fit with duration_col='duration', event_col='event', cluster_col='company_id'."""
    from lifelines import CoxPHFitter
    return CoxPHFitter(penalizer=penalizer)


def fit_cox(df, covariates, penalizer=0.1, cluster_col="company_id"):
    """Cox fit with company-clustered (robust) errors. df needs duration, event, cluster_col and the covariates."""
    cols = list(covariates) + ["duration", "event"] + ([cluster_col] if cluster_col else [])
    return cox_model(penalizer).fit(df[cols], "duration", "event", cluster_col=cluster_col, robust=bool(cluster_col))


def cox_ph_check(model, df, covariates, time_transform="rank", cluster_col="company_id"):
    """Proportional-hazards check (scaled Schoenfeld residuals, lifelines): one row per covariate with the test
    statistic and p-value. Small p = that covariate's effect changes over time (its hazard ratio is not constant), so
    report it per period or stratify instead of as one number. Returns a DataFrame."""
    from lifelines.statistics import proportional_hazard_test
    cols = list(covariates) + ["duration", "event"] + ([cluster_col] if getattr(model, "cluster_col", None) else [])
    r = proportional_hazard_test(model, df[cols], time_transform=time_transform)
    return r.summary[["test_statistic", "p"]]


def aft_model(kind="weibull", penalizer=0.1):
    """Accelerated failure time model (lifelines): 'weibull', 'lognormal' or 'loglogistic'. Fit with
    fit(df, 'duration', 'event'); durations must be above 0 (use aft_ready)."""
    from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
    return {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter,
            "loglogistic": LogLogisticAFTFitter}[kind](penalizer=penalizer)


def aft_ready(df, floor=0.5):
    """AFT models need durations above zero: raise any shorter duration to `floor` days (half a day)."""
    out = df.copy()
    out["duration"] = out["duration"].clip(lower=floor)
    return out


def survival_forest(seed=SEED, n_estimators=200, min_samples_leaf=50):
    """Random survival forest (scikit-survival, in requirements.txt). Fit with fit(X, survival_y(duration, event));
    predict(X) is a risk score (higher = closes sooner)."""
    from sksurv.ensemble import RandomSurvivalForest
    return RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1,
                                random_state=seed)


def survival_y(duration, event):
    """Structured array of (event, time) that scikit-survival expects."""
    y = np.empty(len(duration), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = np.asarray(event).astype(bool), np.asarray(duration, dtype=float)
    return y


# ---------- learning to rank ----------

def lambdarank_model(seed=SEED):
    """LightGBM LambdaRank; fit(X, relevance, group=listings_per_query)."""
    from lightgbm import LGBMRanker
    return LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, random_state=seed, verbose=-1)


def ranking_groups(query_ids):
    """LambdaRank wants rows sorted by query with the size of each query's block. Returns (row order, group sizes)."""
    q = pd.Series(np.asarray(query_ids)).astype(str)
    order = np.argsort(q.values, kind="stable")
    sorted_q = q.iloc[order]
    return order, sorted_q.groupby(sorted_q.values, sort=False).size().tolist()


def train_lambdarank(X, relevance, query_ids, seed=SEED):
    """Fit LambdaRank on graded relevance (non-negative integers) per query (one query = one career changer's list)."""
    order, sizes = ranking_groups(query_ids)
    X = X.iloc[order] if hasattr(X, "iloc") else np.asarray(X)[order]
    return lambdarank_model(seed).fit(X, np.asarray(relevance)[order].astype(int), group=sizes)


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


def backtest_relevance(open_long_enough, reachable, in_moved_role):
    """A listing in a career changer's top 10 is relevant (1) only if it is still open long enough, reachable, and in
    the role that person actually moved into. Each input is 0/1 (or bool) per listing; unknown (missing) counts as
    not relevant."""
    parts = [pd.Series(a).fillna(0).astype(float).values for a in (open_long_enough, reachable, in_moved_role)]
    return (parts[0] * parts[1] * parts[2]).astype(int)


def clustered_bootstrap_diff(model_values, baseline_values, clusters, n_boot=1000, seed=SEED):
    """Mean difference (model minus baseline) with a company-clustered 95% bootstrap interval: whole clusters
    (companies) are resampled together, so one company's listings are not counted as independent. A hypothesis counts
    as supported when the model beats the baseline and the interval excludes zero (ci_low above 0).
    Returns dict: difference, ci_low, ci_high, supported, n_clusters."""
    d = pd.DataFrame({"d": np.asarray(model_values, dtype=float) - np.asarray(baseline_values, dtype=float),
                      "c": np.asarray(clusters)}).dropna()
    g = d.groupby("c")["d"].agg(["sum", "count"])
    sums, counts = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"difference": float(sums.sum() / counts.sum()), "ci_low": lo, "ci_high": hi,
            "supported": bool(lo > 0), "n_clusters": int(len(g))}


def clustered_bootstrap_auc_diff(y_true, proba_model, proba_baseline, clusters, n_boot=1000, seed=SEED):
    """ROC-AUC of the model minus that of the baseline, with a company-clustered 95% bootstrap interval (whole
    companies resampled together). AUC only uses the ranking, so models trained with class_weight='balanced' (whose
    probabilities are not calibrated) are compared fairly; a constant baseline such as the prior has AUC 0.5.
    Returns dict: difference, ci_low, ci_high, supported (ci_low above 0), n_clusters."""
    from sklearn.metrics import roc_auc_score
    y, pm, pb = (np.asarray(a, dtype=float) for a in (y_true, proba_model, proba_baseline))
    codes = pd.factorize(np.asarray(clusters))[0]
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    members = [order[a:b] for a, b in zip(starts[:-1], starts[1:])]
    auc = lambda idx, p: roc_auc_score(y[idx], p[idx]) if np.unique(y[idx]).size == 2 else np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        boots.append(auc(idx, pm) - auc(idx, pb))
    lo, hi = (float(v) for v in np.nanpercentile(boots, [2.5, 97.5]))
    every = np.arange(len(y))
    return {"difference": float(auc(every, pm) - auc(every, pb)), "ci_low": lo, "ci_high": hi,
            "supported": bool(lo > 0), "n_clusters": len(members)}


def top_k_overlap(score_a, score_b, k=10):
    """Share of the top k by score_a that is also in the top k by score_b (secondary test: below 80% = they differ)."""
    a = set(np.argsort(-np.asarray(score_a, dtype=float), kind="stable")[:k])
    b = set(np.argsort(-np.asarray(score_b, dtype=float), kind="stable")[:k])
    return len(a & b) / k


def c_index(duration, event, predicted_open_days):
    """Harrell's C: does the model order listings by how long they stay open? (higher prediction = stays open longer)"""
    from lifelines.utils import concordance_index
    return float(concordance_index(duration, predicted_open_days, event))


# ---------- the priority queue ----------

FACTORS = ["fit", "still_open", "accessibility", "pay"]
# Placeholder weights: replaced by EDA_SDA_4's Cox hazard ratios after it runs, then tuned on the validation month (tune_weights).
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
