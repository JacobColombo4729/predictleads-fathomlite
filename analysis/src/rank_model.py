"""ML_Rank, the listing model and the priority queue (v2 draft; not yet run on real data).

Targets (closes within N days, days open, survival), the time split with company-grouped folds, leakage-safe counts,
the baselines (newest first, title keyword search), preprocessing (ColumnTransformer), class imbalance and threshold
tuning, the model list with grid search, Cox with proportional-hazards checks, AFT models, survival forests,
LambdaRank training, ranking and survival metrics, the company-clustered bootstrap, backtest relevance, and the
priority formula with its per-listing breakdown. Input is a listing table with: id, company_id, title, first_seen,
last_seen. Prints and returns aggregates only.
"""
import atexit
import functools
import gc
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


def answer_known_either_way(first_seen, n_days=CLOSING_DAYS, end=EXPORT_END, tolerance=LIVE_TOLERANCE_DAYS):
    """True for listings first seen early enough that their N-day answer is known whichever way it went: more than
    N + tolerance days before the data ends. Later listings have a known answer only if they closed fast (one still open
    hasn't been watched N days yet), so a month's known answers near the end are all early closers; evaluate on these."""
    return (_naive(first_seen) < pd.Timestamp(end) - pd.Timedelta(days=n_days + tolerance)).to_numpy()


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
        "catboost": CatBoostClassifier(iterations=300, learning_rate=0.05, auto_class_weights="Balanced", random_seed=seed, verbose=0,
                                       allow_writing_files=False),  # no catboost_info/ log files written on every fit
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


class PreparedFolds:
    """The preprocessor fit on each training fold, once, with the transformed folds kept in a temporary folder on disk
    (deleted when Python exits). Cross-validating a pipeline with_preprocessing(PREP, model) refits the same
    preprocessor on the same folds for every model and every grid setting (hundreds of times); here that is done once
    and only the model is fit. The scores are the same as GridSearchCV / cross_val_score on the full pipeline (the
    preprocessor is fit on the training fold only, as before). Fold scores are remembered per model setting, so a setting
    scored twice (7a, then the same setting inside 7b's grid) is fit once. Build it with prepare_folds()."""

    def __init__(self, preprocessor, X, y, folds, key=None):
        import shutil
        import tempfile
        from pathlib import Path
        from sklearn.base import clone
        self.key = key
        self.y = np.asarray(y)
        self.folds = [(np.asarray(tr), np.asarray(va)) for tr, va in folds]
        self.dir = Path(tempfile.mkdtemp(prefix="ml_rank_folds_"))
        self._rmtree = shutil.rmtree
        self.scores = {}  # model setting -> ROC-AUC per fold
        rows = (lambda idx: X.iloc[idx]) if hasattr(X, "iloc") else (lambda idx: np.asarray(X)[idx])
        for k, (tr, va) in enumerate(self.folds):
            prep = clone(preprocessor)
            np.save(self.dir / f"train_{k}.npy", np.asarray(prep.fit_transform(rows(tr), self.y[tr])))
            np.save(self.dir / f"valid_{k}.npy", np.asarray(prep.transform(rows(va))))
            if k == 0:  # each fold's files are about the same size: check the disk once, early
                need = sum(f.stat().st_size for f in self.dir.iterdir()) * len(self.folds)
                free = shutil.disk_usage(self.dir).free
                if free < 1.1 * need:
                    self.close()
                    raise RuntimeError(f"prepared folds need about {need / 1e9:.1f} GB of temporary disk space and only "
                                       f"{free / 1e9:.1f} GB is free; call grid_search / cv_auc without prepared=")

    def close(self):
        self._rmtree(self.dir, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def fold_scores(self, models, scoring="roc_auc", fold_jobs=1):
        """ROC-AUC per fold for each model (a list of estimators, fit on the transformed training fold). fold_jobs > 1
        runs (model, fold) fits in that many threads: only for models that use one core themselves (a decision tree)."""
        from joblib import Parallel, delayed
        from sklearn.metrics import get_scorer
        keys = [_setting_key(m) for m in models]
        todo = [(i, k) for i, key in enumerate(keys) if key not in self.scores for k in range(len(self.folds))]

        def one(i, k):
            from sklearn.base import clone
            tr, va = self.folds[k]
            # read-only memory maps: the operating system pages the files in and out, so side-by-side fits don't each hold a copy
            model = clone(models[i]).fit(np.load(self.dir / f"train_{k}.npy", mmap_mode="r"), self.y[tr])
            score = get_scorer(scoring)(model, np.load(self.dir / f"valid_{k}.npy", mmap_mode="r"), self.y[va])
            del model
            gc.collect()  # a fitted model can sit in a reference cycle that keeps the fold's table (up to 1 GB) alive until the next collection
            return score

        if fold_jobs > 1 and len(todo) > 1:
            done = Parallel(n_jobs=fold_jobs, prefer="threads")(delayed(one)(i, k) for i, k in todo)
        else:
            done = [one(i, k) for i, k in todo]
        got = {}
        for (i, k), s in zip(todo, done):
            got.setdefault(keys[i], {})[k] = s
        for key, by_fold in got.items():
            self.scores[key] = np.array([by_fold[k] for k in range(len(self.folds))])
        return [self.scores[key].copy() for key in keys]


def _setting_key(model):
    """Identifies a model and all its settings (two models with the same key give the same fold scores)."""
    params = model.get_params(deep=True)
    return type(model).__module__ + "." + type(model).__name__ + repr(sorted((k, repr(v)) for k, v in params.items()))


_PREPARED = {}  # fingerprint -> PreparedFolds; the two most recent are kept (7a/7b and then 11a's windows)
atexit.register(lambda: [p.close() for p in _PREPARED.values()])  # the temporary fold files go when Python exits


def prepare_folds(preprocessor, X, y, folds):
    """PreparedFolds for these rows, labels and folds. Asked again for the same preprocessor, rows, labels and folds
    (found by a fingerprint of all four), it returns the earlier one with its remembered fold scores: 11a's 14-day
    window is the main target again, so its cross-validation costs nothing."""
    import hashlib
    h = hashlib.sha1()
    h.update(repr(preprocessor).encode())
    h.update(np.ascontiguousarray(np.asarray(y)).tobytes())
    for tr, va in folds:
        h.update(np.asarray(tr, dtype=np.int64).tobytes())
        h.update(np.asarray(va, dtype=np.int64).tobytes())
    frame = X if hasattr(X, "iloc") else pd.DataFrame(np.asarray(X))
    h.update(repr((frame.shape, list(frame.columns))).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    key = h.hexdigest()
    found = _PREPARED.pop(key, None)
    while found is None and len(_PREPARED) >= 2:
        del _PREPARED[next(iter(_PREPARED))]  # the oldest; its temporary files go once nothing else holds it
    found = found or PreparedFolds(preprocessor, X, y, folds, key)
    _PREPARED[key] = found  # most recent last
    return found


class FoldGridSearch:
    """What grid_search() returns with prepared folds: the same attributes as a fitted GridSearchCV that the notebook reads
    (best_params_, best_score_, best_index_, cv_results_, best_estimator_), with the same tie rule (first best setting)."""

    def __init__(self, pipeline, param_grid, X, y, prepared, fold_jobs=1, scoring="roc_auc"):
        from sklearn.base import clone
        from sklearn.model_selection import ParameterGrid
        from scipy.stats import rankdata
        settings = list(ParameterGrid(param_grid))
        models = [clone(pipeline.named_steps["model"]).set_params(**{k.removeprefix("model__"): v for k, v in s.items()})
                  for s in settings]
        scores = np.array(prepared.fold_scores(models, scoring, fold_jobs))  # settings x folds
        mean = scores.mean(axis=1)
        self.cv_results_ = {"params": settings, "mean_test_score": mean,
                            "std_test_score": np.sqrt(np.average((scores - mean[:, None]) ** 2, axis=1)),
                            "rank_test_score": rankdata(-mean, method="min").astype(np.int32),
                            **{f"split{k}_test_score": scores[:, k] for k in range(scores.shape[1])}}
        self.best_index_ = int(self.cv_results_["rank_test_score"].argmin())
        self.best_params_ = settings[self.best_index_]
        self.best_score_ = float(mean[self.best_index_])
        self._refit = (clone(pipeline).set_params(**self.best_params_), X, y)
        self._best_estimator = None

    @property
    def best_estimator_(self):
        """The best setting fit on all training rows (as GridSearchCV's refit=True). Fit when first asked for: a model
        that does not win is never refit, and its forest or boosted trees never take memory."""
        if self._best_estimator is None:
            pipe, X, y = self._refit
            self._best_estimator = pipe.fit(X, y)
        return self._best_estimator


def grid_search(pipeline, param_grid, X, y, folds, scoring="roc_auc", n_jobs=1, prepared=None, fold_jobs=1):
    """GridSearchCV over the company-grouped folds (a list of (train, valid) positions). Returns the fitted search;
    read best_params_, best_score_ and cv_results_ (mean and std per setting). The notebook times it with %%time.
    With prepared=prepare_folds(PREP, X, y, folds) the preprocessing is done once per fold instead of once per fit, and
    the same numbers come out (pipeline must be with_preprocessing(PREP, model)); fold_jobs fits run side by side."""
    if prepared is not None:
        return FoldGridSearch(pipeline, param_grid, X, y, prepared, fold_jobs, scoring)
    from sklearn.model_selection import GridSearchCV
    return GridSearchCV(pipeline, param_grid, cv=folds, scoring=scoring, n_jobs=n_jobs, refit=True).fit(X, y)


def cv_auc(pipeline, X, y, folds, prepared=None, fold_jobs=1):
    """Cross-validated ROC-AUC per fold (array): report mean and std. With prepared folds (see grid_search) only the
    model is fit per fold, and a setting already scored on them is not fit again."""
    if prepared is not None:
        return prepared.fold_scores([pipeline.named_steps["model"]], "roc_auc", fold_jobs)[0]
    from sklearn.model_selection import cross_val_score
    return cross_val_score(pipeline, X, y, cv=folds, scoring="roc_auc")


# ---------- survival models ----------

@functools.lru_cache(maxsize=None)
def _fast_cox_class():
    """lifelines' CoxPHFitter whose inner fitter has one method replaced by an equivalent that is linear in the rows."""
    from lifelines import CoxPHFitter
    from lifelines.fitters.coxph_fitter import SemiParametricPHFitter

    class FastSemiParametricPHFitter(SemiParametricPHFitter):
        def _compute_score_within_strata(self, X, _T, E, weights, entries):
            """Score residuals for the robust (sandwich) variance: lifelines' formula (ties not handled, as there), with
            the sum over earlier rows of its row-by-row loop written as running sums. The loop is quadratic in the number of
            rows (about 7 s at 5,000 rows, 4 minutes at 20,000, days at 600,000); this is the same to about 1e-12."""
            X, E, w = X.to_numpy(float), E.to_numpy(int), weights.to_numpy(float)
            beta = self.params_.to_numpy(float) * self._norm_std.to_numpy(float)  # the betas for the normalised X
            phi = np.exp(X @ beta)
            risk = np.cumsum((w * phi)[::-1])[::-1]
            risk_x = np.cumsum((X * (w * phi)[:, None])[::-1], axis=0)[::-1]
            a = E * w / risk  # the weight of each death in every later row's score
            a_sum, b_sum = np.cumsum(a), np.cumsum(a[:, None] * risk_x / risk[:, None], axis=0)
            score = -phi[:, None] * (X * a_sum[:, None] - b_sum) + E[:, None] * (X - risk_x / risk[:, None])
            return score * w[:, None]

    class FastCoxPHFitter(CoxPHFitter):
        def _fit_model_breslow(self, *args, **kwargs):
            model = FastSemiParametricPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio, strata=self.strata,
                                               alpha=self.alpha, label=self._label)
            model.fit(*args, **kwargs)
            return model

    return FastCoxPHFitter


def cox_model(penalizer=0.1):
    """Cox proportional hazards (lifelines); fit with duration_col='duration', event_col='event', cluster_col='company_id'.
    The company-clustered (robust) errors use the same formula as lifelines, computed in linear time."""
    return _fast_cox_class()(penalizer=penalizer)


def fit_cox(df, covariates, penalizer=0.1, cluster_col="company_id"):
    """Cox fit with company-clustered (robust) errors. df needs duration, event, cluster_col and the covariates."""
    cols = list(covariates) + ["duration", "event"] + ([cluster_col] if cluster_col else [])
    return cox_model(penalizer).fit(df[cols], "duration", "event", cluster_col=cluster_col, robust=bool(cluster_col))


def scaled_schoenfeld(model, df, covariates, cluster_col="company_id"):
    """Scaled Schoenfeld residuals of a fitted CoxPHFitter (Efron ties), the same numbers as
    model.compute_residuals(df, "scaled_schoenfeld") (checked in the tests; agreement to about 1e-12). lifelines grows
    its result one row at a time with np.append, which copies everything so far on every row: hours and many GB of
    copying on 600,000 rows. Here the risk-set sums are reversed cumulative sums and each tied-death group is one
    vectorised step; lifelines' own preprocessing and scaling are reused."""
    cols = list(covariates) + ["duration", "event"] + ([cluster_col] if getattr(model, "cluster_col", None) else [])
    X, T, E, weights, _, original_index, _ = model._preprocess_dataframe(df[cols])  # sorted by duration, as lifelines does
    X, T, E, weights = X.to_numpy(float), T.to_numpy(float), E.to_numpy(bool), weights.to_numpy(float)
    beta = model.params_.to_numpy(float)
    phi = weights * np.exp(X @ beta)
    phi_x = phi[:, None] * X
    risk = np.cumsum(phi[::-1])[::-1]  # at row i: the sum over rows i.. (everyone still at risk at the start of i's time)
    risk_x = np.cumsum(phi_x[::-1], axis=0)[::-1]
    group = np.r_[0, np.cumsum(T[1:] != T[:-1])]  # tied times share a group
    first = np.r_[0, np.flatnonzero(T[1:] != T[:-1]) + 1]  # first row of each group
    ev = np.flatnonzero(E)
    deaths = np.bincount(group[ev], minlength=len(first))  # tied deaths per group
    tie_phi = np.bincount(group[ev], weights=phi[ev], minlength=len(first))
    tie_phi_x = np.zeros((len(first), X.shape[1]))
    np.add.at(tie_phi_x, group[ev], phi_x[ev])
    mean = np.zeros((len(first), X.shape[1]))  # Efron's weighted mean of the covariates in each group's risk set
    single = deaths == 1
    mean[single] = risk_x[first[single]] / risk[first[single]][:, None]
    for g in np.flatnonzero(deaths > 1):
        d = deaths[g]
        step = np.arange(d)[:, None] / d
        mean[g] = ((risk_x[first[g]] - step * tie_phi_x[g]) / ((risk[first[g]] - step[:, 0] * tie_phi[g])[:, None] * d)).sum(axis=0)
    resid = X[ev] - mean[group[ev]]
    scaled = pd.DataFrame(E.sum() * resid @ model.variance_matrix_.to_numpy(float), columns=model.params_.index,
                          index=original_index[E])
    return scaled


def cox_ph_check(model, df, covariates, time_transform="rank", cluster_col="company_id"):
    """Proportional-hazards check (scaled Schoenfeld residuals, lifelines): one row per covariate with the test
    statistic and p-value. Small p = that covariate's effect changes over time (its hazard ratio is not constant), so
    report it per period or stratify instead of as one number. Returns a DataFrame. The residuals come from
    scaled_schoenfeld() (same numbers as lifelines', computed in seconds, not hours)."""
    from lifelines.statistics import proportional_hazard_test
    cols = list(covariates) + ["duration", "event"] + ([cluster_col] if getattr(model, "cluster_col", None) else [])
    r = proportional_hazard_test(model, df[cols], time_transform=time_transform,
                                 precomputed_residuals=scaled_schoenfeld(model, df, covariates, cluster_col))
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


def survival_forest(seed=SEED, n_estimators=200, min_samples_leaf=50, low_memory=True):
    """Random survival forest (scikit-survival, in requirements.txt). Fit with fit(X, survival_y(duration, event));
    predict(X) is a risk score (higher = closes sooner). low_memory keeps each tree's risk score per node instead of a
    cumulative hazard curve per node (one value per distinct duration): predict gives the same scores (to 1e-12) and
    needs a few percent of the memory and time; only predict_cumulative_hazard_function and predict_survival_function
    are not available (pass low_memory=False for them, on a small sample)."""
    from sksurv.ensemble import RandomSurvivalForest
    return RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs=-1,
                                random_state=seed, low_memory=low_memory)


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


LAMBDARANK_MAX_QUERY = 10_000  # LightGBM's limit on rows per query


def split_large_queries(query_ids, max_rows=LAMBDARANK_MAX_QUERY, seed=SEED):
    """Query ids with every query over `max_rows` rows split at random into the fewest equal-sized parts that fit
    (LightGBM refuses longer queries). Each part is a random sample of its query, so it ranks the same kind of list;
    queries that fit keep their id."""
    q = pd.Series(np.asarray(query_ids)).astype(str)
    sizes = q.map(q.value_counts()).values
    parts = np.ceil(sizes / max_rows).astype(int)
    if (parts == 1).all():
        return q.values
    position = pd.Series(np.random.default_rng(seed).permutation(len(q))).groupby(q.values).rank(method="first").values - 1
    part = (position % parts).astype(int).astype(str)
    return np.where(parts > 1, q.values + "#" + part, q.values)


def train_lambdarank(X, relevance, query_ids, seed=SEED):
    """Fit LambdaRank on graded relevance (non-negative integers) per query (one query = one career changer's list).
    Queries longer than LightGBM's limit are split at random into parts (split_large_queries)."""
    order, sizes = ranking_groups(split_large_queries(query_ids, seed=seed))
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


def _cluster_members(clusters):
    """Rows of each cluster (company) as one stable order of row positions plus where each cluster starts and its size."""
    codes = pd.factorize(np.asarray(clusters))[0]
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(codes)]
    return order, starts[:-1], np.diff(starts)


def _resampled_rows(order, starts, sizes, draw):
    """The rows of the drawn clusters, cluster after cluster in the draw's order (what concatenating each cluster's rows gives)."""
    sz, st = sizes[draw], starts[draw]
    end = np.cumsum(sz)
    return order[np.arange(end[-1]) + np.repeat(st - (end - sz), sz)]


def clustered_bootstrap_auc_diff(y_true, proba_model, proba_baseline, clusters, n_boot=1000, seed=SEED, workers=8):
    """ROC-AUC of the model minus that of the baseline, with a company-clustered 95% bootstrap interval (whole
    companies resampled together). AUC only uses the ranking, so models trained with class_weight='balanced' (whose
    probabilities are not calibrated) are compared fairly; a constant baseline such as the prior has AUC 0.5.
    The draws come one after another from one seeded generator; the AUCs of a batch of draws are computed in parallel
    threads (the same numbers as one by one).
    Returns dict: difference, ci_low, ci_high, supported (ci_low above 0), n_clusters."""
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.metrics import roc_auc_score
    y, pm, pb = (np.asarray(a, dtype=float) for a in (y_true, proba_model, proba_baseline))
    order, starts, sizes = _cluster_members(clusters)
    auc = lambda idx, p: roc_auc_score(y[idx], p[idx]) if np.unique(y[idx]).size == 2 else np.nan

    def one(draw):
        idx = _resampled_rows(order, starts, sizes, draw)
        return auc(idx, pm) - auc(idx, pb)

    rng = np.random.default_rng(seed)
    boots = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for first in range(0, n_boot, 4 * max(1, workers)):
            draws = [rng.integers(0, len(sizes), len(sizes)) for _ in range(min(4 * max(1, workers), n_boot - first))]
            boots.extend(pool.map(one, draws))
    lo, hi = (float(v) for v in np.nanpercentile(boots, [2.5, 97.5]))
    every = np.arange(len(y))
    return {"difference": float(auc(every, pm) - auc(every, pb)), "ci_low": lo, "ci_high": hi,
            "supported": bool(lo > 0), "n_clusters": len(sizes)}


def top_k_overlap(score_a, score_b, k=10):
    """Share of the top k by score_a that is also in the top k by score_b (secondary test: below 80% = they differ)."""
    a = set(np.argsort(-np.asarray(score_a, dtype=float), kind="stable")[:k])
    b = set(np.argsort(-np.asarray(score_b, dtype=float), kind="stable")[:k])
    return len(a & b) / k


def c_index(duration, event, predicted_open_days):
    """Harrell's C: does the model order listings by how long they stay open? (higher prediction = stays open longer)"""
    from lifelines.utils import concordance_index
    return float(concordance_index(duration, predicted_open_days, event))


def _c_index_draws(duration, event, risk, rows_list):
    """C-index of each resample (rows_list: positions); run in a worker process."""
    return [c_index(duration[r], event[r], -risk[r]) for r in rows_list]


def clustered_bootstrap_c_index(duration, event, risk, clusters, n_boot=100, seed=SEED, workers=1):
    """Harrell's C of a risk score (higher = closes sooner) with a company-clustered 95% bootstrap interval: whole companies
    are resampled together. The draws come one after another from one seeded generator, so the numbers do not depend on
    `workers`; with workers > 1 the C-index of each draw (about a second each on 150,000 rows) runs in that many parallel
    processes, each of which takes about 1.4 GB (it imports lifelines): raise it only when the memory is free.
    Returns dict: c_index, ci_low, ci_high, n_clusters, boots (every resampled C)."""
    from concurrent.futures import ProcessPoolExecutor
    duration, event, risk = (np.asarray(a, dtype=float) for a in (duration, event, risk))
    order, starts, sizes = _cluster_members(clusters)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(sizes), size=len(sizes)) for _ in range(n_boot)]
    chunks = [draws[i::max(1, workers)] for i in range(max(1, workers))]  # draw i goes to worker i % workers
    if workers <= 1 or n_boot <= 1:
        results = [_c_index_draws(duration, event, risk, [_resampled_rows(order, starts, sizes, d) for d in chunks[0]])]
        boots = results[0]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_c_index_draws, duration, event, risk, [_resampled_rows(order, starts, sizes, d) for d in ch])
                       for ch in chunks if ch]
            results = [f.result() for f in futures]
        boots = [None] * n_boot
        for w, res in enumerate(results):
            for j, v in enumerate(res):
                boots[w + j * workers] = v
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return {"c_index": c_index(duration, event, -risk), "ci_low": lo, "ci_high": hi, "n_clusters": len(sizes), "boots": boots}


# ---------- the priority queue ----------

FACTORS = ["fit", "still_open", "accessibility", "pay"]
# Starting weights, tuned on the validation month (tune_weights). EDA_SDA_4's Cox hazard ratios are about closing speed, so they do not set these four
# weights; they enter through the listing model behind the `still_open` factor.
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


# ---------- the backtest: career changers x one month's listings (H4, H5, H6) ----------
# The pool is one month's listings a career changer can see (in the 8 fit roles, not hidden), with a known N-day answer.
# A listing in a person's top k is relevant when it stayed open long enough, was not a likely ghost, and is in the role
# that person actually moved into. The listing factors don't depend on the person, so each role's k best listings on
# the listing part are the only candidates a person's top k can draw from: the ranking below is exact and fast.

ACCESS_SIGNS = ["entry_seniority", "low_experience_ask", "entry_wording", "changer_wording"]  # career_open's parts (EDA_SDA_2 3.11)


def accessibility_count(df, cols=ACCESS_SIGNS):
    """How many of the career-change signs a listing shows (0-4); a sign that is unknown (no description) counts as absent."""
    return sum(df[c].astype("boolean").fillna(False).astype(int) for c in cols)


def listing_ratings(pool):
    """pool: role_bucket, p_still_open, accessibility, pay per listing. 0-100 ratings of the three listing factors within
    the pool; pay is rated within its role (so a role's pay level doesn't decide it); missing = 50."""
    pay = pool.groupby("role_bucket", observed=True)["pay"].rank(pct=True) * 100
    return pd.DataFrame({"still_open": rating_0_100(pool["p_still_open"]).values,
                         "accessibility": rating_0_100(pool["accessibility"]).values,
                         "pay": pay.fillna(50.0).values}, index=pool.index)


def fit_ratings(fit):
    """fit: people x roles, ML_Fit's probabilities. Rated 0-100 per person: the best-fitting role is 100, the rest in
    proportion, so the rating says how close a role is to the person's best."""
    fit = pd.DataFrame(fit, dtype=float)
    return fit.div(fit.max(axis=1), axis=0) * 100


def _weights(weights):
    w = pd.Series(weights or START_WEIGHTS, dtype=float).reindex(FACTORS).fillna(0.0)
    return w / w.sum()


def queue_top_k(fit_rating, ratings, roles, weights=None, k=10):
    """Each person's k highest-scored pool listings (positions into the pool, best first). score = w_fit x the person's
    fit rating for the listing's role + the weighted listing ratings; the parts add up as in priority_breakdown.
    fit_rating: people x roles (columns = role names); ratings: listing_ratings(); roles: each listing's role."""
    w = _weights(weights)
    part = sum(w[f] * ratings[f].to_numpy(dtype=float) for f in ["still_open", "accessibility", "pay"])
    roles = np.asarray(roles, dtype=object)
    cand, cand_role = [], []
    for j, r in enumerate(fit_rating.columns):
        pos = np.flatnonzero(roles == r)
        best = pos[np.argsort(-part[pos], kind="stable")[:k]]
        cand.append(best)
        cand_role.append(np.full(len(best), j))
    cand, cand_role = np.concatenate(cand), np.concatenate(cand_role)
    order = np.argsort(cand, kind="stable")  # ties go to the earlier pool position, as in one stable sort of the pool
    cand, cand_role = cand[order], cand_role[order]
    scores = w["fit"] * fit_rating.to_numpy(dtype=float)[:, cand_role] + part[cand]
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return cand[top]


def newest_top_k(first_seen, n_people, k=10, roles=None, search_role=None):
    """The newest-first baseline: the k newest pool listings, the same list for everyone. With roles and search_role, a
    title search for that role first (a perfect one: every listing in the role and nothing else), then newest first."""
    t = _naive(pd.Series(first_seen)).astype("int64").to_numpy()
    pos = np.arange(len(t)) if search_role is None else np.flatnonzero(np.asarray(roles, dtype=object) == search_role)
    best = pos[np.argsort(-t[pos], kind="stable")[:k]]
    return np.tile(best, (n_people, 1))


def precision_per_person(top, ok, roles, targets):
    """Share of each person's top k that is relevant: ok (stayed open long enough, not a ghost) and in the person's
    actual role. top: people x k pool positions; ok, roles: per pool listing; targets: per person."""
    ok, roles = np.asarray(ok, dtype=bool), np.asarray(roles, dtype=object)
    hit = ok[top] & (roles[top] == np.asarray(targets, dtype=object)[:, None])
    return hit.mean(axis=1)


def tune_queue_weights(fit_rating, ratings, roles, ok, targets, k=10, grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Coarse grid of the four weights on the validation month and development people: the best mean precision@k.
    Returns (best weights, best mean precision, table of every combination tried). Combinations that are multiples of
    each other give the same ranking; the first one found is kept."""
    rows, best, best_p = [], None, -1.0
    for combo in itertools.product(grid, repeat=len(FACTORS)):
        if sum(combo) == 0:
            continue
        w = dict(zip(FACTORS, combo))
        p = float(precision_per_person(queue_top_k(fit_rating, ratings, roles, w, k), ok, roles, targets).mean())
        rows.append({**w, "precision": p})
        if p > best_p + 1e-12:
            best, best_p = w, p
    return best, best_p, pd.DataFrame(rows)


def mean_overlap(top_a, top_b):
    """Mean share of each person's top k (a) that is also in their top k (b): H6 (below 0.8 = the lists differ)."""
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(top_a, top_b)]))
