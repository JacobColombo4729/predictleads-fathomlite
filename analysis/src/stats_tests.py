"""Statistics helpers for EDA_SDA_4: group summaries, association measures, tests with effect sizes,
company-clustered bootstrap, the adjusted Cox check, and the in-depth feature checks.

Everything takes plain DataFrames/Series and returns small aggregate tables (no row-level output).
Tested on made-up data only (tests/test_stats_tests.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# ---------- 1. memory ----------
def downcast(df, max_categories=200):
    """Smaller dtypes, same values: float64 -> float32, plain (non-nullable) ints -> the smallest int that fits, and
    object columns with at most `max_categories` distinct non-bool values -> category. Bool-like object columns
    (True/False/None) and nullable extension dtypes are left alone. Returns a new frame."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == np.float64:
            out[c] = s.astype(np.float32)
        elif s.dtype.kind in "iu" and not isinstance(s.dtype, pd.api.extensions.ExtensionDtype):
            out[c] = pd.to_numeric(s, downcast="integer" if s.dtype.kind == "i" else "unsigned")
        elif s.dtype == object:
            vals = s.dropna()
            if vals.map(type).eq(bool).all():
                continue
            if vals.nunique() <= max_categories:
                out[c] = s.astype("category")
    return out


# ---------- 2. descriptive statistics ----------
def describe_by_group(df, value, group):
    """Count, mean, median, std, variance, min, Q1, Q3, max of `value` per `group` (course set)."""
    g = df.groupby(group, observed=True)[value]
    out = g.agg(count="count", mean="mean", median="median", std="std", variance="var", min="min", max="max")
    out["q1"], out["q3"] = g.quantile(0.25), g.quantile(0.75)
    return out[["count", "mean", "median", "std", "variance", "min", "q1", "q3", "max"]]


# ---------- 3. association measures ----------
def correlations(df, cols, outcome):
    """Pearson and Spearman of each numeric column with the outcome (rows with both present)."""
    rows = {}
    for c in cols:
        d = df[[c, outcome]].dropna()
        ok = len(d) > 2 and d[c].nunique() > 1 and d[outcome].nunique() > 1
        rows[c] = {"n": len(d), "pearson": d[c].corr(d[outcome]) if ok else np.nan,
                   "spearman": d[c].corr(d[outcome], method="spearman") if ok else np.nan}
    return pd.DataFrame(rows).T


def _contingency(x, y):
    """Counts of each (x, y) pair as an array, rows and columns of levels that occur (rows with either value missing
    dropped). Same table as `pd.crosstab` on the non-missing rows, built from integer codes instead of Python lists."""
    cx, ux = pd.factorize(x, sort=True)
    cy, uy = pd.factorize(y, sort=True)
    ok = (cx >= 0) & (cy >= 0)
    tab = np.bincount(cx[ok].astype(np.int64) * len(uy) + cy[ok], minlength=len(ux) * len(uy)).reshape(len(ux), len(uy))
    return tab[tab.sum(axis=1) > 0][:, tab.sum(axis=0) > 0]


def cramers_v(x, y, bias_correct=True):
    """Cramer's V between two categorical series (Bergsma bias correction by default)."""
    return _cramers_from_table(_contingency(x, y), bias_correct)


def _cramers_from_table(tab, bias_correct=True):
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    phi2 = chi2 / n
    r, k = tab.shape
    if not bias_correct:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def point_biserial(flag, x):
    """Point-biserial correlation of a 0/1 flag with a numeric measure, and its p-value."""
    f, v = pd.Series(flag).astype("float64").to_numpy(), pd.Series(x).astype("float64").to_numpy()
    ok = ~(np.isnan(f) | np.isnan(v))
    f, v = f[ok], v[ok]
    if len(np.unique(f)) < 2 or len(np.unique(v)) < 2:
        return np.nan, np.nan
    r, p = stats.pointbiserialr(f.astype(int), v)
    return float(r), float(p)


def vif(df, cols):
    """Variance inflation factor per numeric column (complete rows; constant added internally)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools import add_constant
    x = add_constant(df[cols].dropna().astype(float), has_constant="add")
    return pd.Series({c: variance_inflation_factor(x.values, i) for i, c in enumerate(x.columns) if c != "const"}, name="vif")


# ---------- 4. group rates, interactions, Simpson's paradox ----------
def rate_by(df, outcome, factors):
    """Mean outcome and count by one or more factors (two-way groupby for interactions)."""
    return df.groupby(factors, observed=True)[outcome].agg(rate="mean", n="size")


def _mean_size_by_flag(d, outcome, flag, group):
    """Mean and count of the outcome per (group, flag) as two tables with one row per group (sorted, observed levels only) and
    columns False and True. `d` has no missing values; the flag is boolean. Counted from integer codes (a plain sum per cell)."""
    codes, levels = pd.factorize(d[group], sort=True)
    k = len(levels)
    cell = codes.astype(np.int64) * 2 + d[flag].to_numpy(bool)
    size = np.bincount(cell, minlength=2 * k).reshape(k, 2)
    total = np.bincount(cell, weights=d[outcome].to_numpy(np.float64), minlength=2 * k).reshape(k, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / size
    index = pd.Index(list(levels))
    return pd.DataFrame(mean, index=index, columns=[False, True]), pd.DataFrame(size.astype(float), index=index, columns=[False, True])


def stratified_gap(df, outcome, flag, strata):
    """Outcome gap (flag=1 minus flag=0) overall and within each stratum.

    Returns (table, summary). summary: crude gap, stratum-size-weighted adjusted gap, and
    `reverses` = True when the crude and adjusted gaps have opposite signs (Simpson's paradox).
    """
    d = df[[outcome, flag, strata]].dropna().copy()
    d[flag] = d[flag].astype(bool)
    mean, size = _mean_size_by_flag(d, outcome, flag, strata)
    tab = pd.DataFrame({"n_flag": size[True].fillna(0).astype(int).to_numpy(), "n_other": size[False].fillna(0).astype(int).to_numpy(),
                        "gap": (mean[True] - mean[False]).to_numpy()}, index=pd.Index(mean.index.tolist(), name="stratum"))
    ok = tab.dropna(subset=["gap"])
    w = ok.n_flag + ok.n_other
    adjusted = float((ok.gap * w).sum() / w.sum()) if len(ok) else np.nan
    crude = float(d.loc[d[flag], outcome].mean() - d.loc[~d[flag], outcome].mean())
    return tab, {"crude_gap": crude, "adjusted_gap": adjusted,
                 "reverses": bool(crude != 0 and adjusted != 0 and np.sign(crude) != np.sign(adjusted))}


# ---------- 5. hypothesis tests with effect sizes ----------
def _numeric(s):
    """A plain float64 Series without missing values, from any numeric-like column: nullable Int64/Float64/boolean
    (pd.NA), or object holding numbers. SciPy can't test nullable or object arrays ("data type object not inexact")."""
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("Float64")
    return pd.to_numeric(s, errors="coerce").dropna().astype("float64")


def cohens_d(a, b):
    a, b = _numeric(a).to_numpy(), _numeric(b).to_numpy()
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def welch_test(a, b):
    """Welch's t-test (unequal variances). Effect size: Cohen's d."""
    a, b = _numeric(a), _numeric(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"test": "welch_t", "n_a": len(a), "n_b": len(b), "statistic": float(t), "p": float(p),
            "effect": cohens_d(a, b), "effect_name": "cohens_d"}


def mann_whitney(a, b):
    """Mann-Whitney U (Wilcoxon rank-sum). Effect size: rank-biserial correlation (-1 to 1).
    Never use it on days open: that measure is censored (use the log-rank test)."""
    a, b = _numeric(a), _numeric(b)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"test": "mann_whitney", "n_a": len(a), "n_b": len(b), "statistic": float(u), "p": float(p),
            "effect": float(2 * u / (len(a) * len(b)) - 1), "effect_name": "rank_biserial"}


def chi_square(x, y):
    """Chi-square test of independence for two categoricals. Effect size: Cramer's V."""
    tab = _contingency(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return {"test": "chi_square", "n_a": int(tab.sum()), "n_b": np.nan, "statistic": float(chi2), "p": float(p), "dof": int(dof),
            "effect": _cramers_from_table(tab, bias_correct=False), "effect_name": "cramers_v"}


def logrank(days_a, done_a, days_b, done_b, weighted=False):
    """Log-rank test between two groups; weighted=True gives Gehan-Breslow-Wilcoxon.
    Effect size: ratio of event rates per listing-day (a crude rate ratio)."""
    from lifelines.statistics import logrank_test
    r = logrank_test(days_a, days_b, done_a, done_b, weightings="wilcoxon" if weighted else None)
    ra = np.sum(done_a) / max(np.sum(days_a), 1e-9)
    rb = np.sum(done_b) / max(np.sum(days_b), 1e-9)
    return {"test": "gehan_breslow_wilcoxon" if weighted else "logrank", "n_a": len(days_a), "n_b": len(days_b),
            "statistic": float(r.test_statistic), "p": float(r.p_value),
            "effect": float(ra / rb) if rb > 0 else np.nan, "effect_name": "event_rate_ratio"}


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (q-values), same order as the input; NaNs stay NaN."""
    p = pd.Series(pvals, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.dropna()
    if ok.empty:
        return out
    order = ok.sort_values().index
    m = len(ok)
    adj = ok[order].values * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out


def tests_table(results, alpha=ALPHA):
    """Stack test dicts (each with a 'name') into a table with BH-adjusted q and a significance column."""
    t = pd.DataFrame(results).set_index("name")
    t["q_bh"] = benjamini_hochberg(t.p)
    t["significant_after_bh"] = t.q_bh < alpha
    return t


# ---------- 6. company-clustered bootstrap ----------
def clustered_bootstrap(df, cluster, stat_fn, n_boot=1000, seed=0, ci=(2.5, 97.5)):
    """Resample whole clusters (companies) with replacement; return the statistic's estimate and interval.
    `stat_fn(frame) -> float`. Listings from one company are not independent, so this is wider than
    the plain bootstrap when companies differ."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster, observed=True)}
    keys = list(groups)
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        reps.append(stat_fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
    lo, hi = np.nanpercentile(reps, ci)
    return {"estimate": float(stat_fn(df)), "ci_low": float(lo), "ci_high": float(hi), "n_clusters": len(keys), "n_boot": n_boot}


def clustered_bootstrap_rate(df, cluster, value, n_boot=1000, seed=0, ci=(2.5, 97.5)):
    """Fast company-clustered bootstrap of a mean (e.g. a close rate): each company is reduced to its sum and
    count once, and each resample reweights companies by how often they are drawn. Same answer as
    `clustered_bootstrap` with `stat_fn = mean`, in seconds instead of hours on ~80K companies."""
    d = df[[cluster, value]].dropna(subset=[value])
    g = d.groupby(cluster, observed=True)[value].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
        reps[b] = (w @ s) / (w @ n)
    lo, hi = np.nanpercentile(reps, ci)
    return {"estimate": float(s.sum() / n.sum()), "ci_low": float(lo), "ci_high": float(hi), "n_clusters": len(g), "n_boot": n_boot}


def plain_bootstrap(df, stat_fn, n_boot=1000, seed=0, ci=(2.5, 97.5)):
    """Row-level bootstrap, for comparing its width with the clustered one (design effect)."""
    rng = np.random.default_rng(seed)
    reps = [stat_fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(reps, ci)
    return {"estimate": float(stat_fn(df)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def plain_bootstrap_mean(values, n_boot=1000, seed=0, ci=(2.5, 97.5)):
    """`plain_bootstrap` for a mean, on the values themselves (same row draws, so the same interval, without rebuilding a frame
    for every resample)."""
    v = pd.Series(values).dropna().to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    reps = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    lo, hi = np.nanpercentile(reps, ci)
    return {"estimate": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


# ---------- adjusted check: Cox regression ----------
def cox_design(df, duration, event, covariates, cluster=None, categorical=()):
    """Complete-case, dummy-coded frame ready for the Cox fit (first level of each categorical dropped)."""
    cols = list(dict.fromkeys([duration, event, *covariates] + ([cluster] if cluster else [])))
    d = df[cols].dropna()
    cats = [c for c in categorical if c in covariates]
    x = pd.get_dummies(d.drop(columns=[cluster] if cluster else []), columns=cats, drop_first=True, dtype=float)
    if cluster:
        x[cluster] = d[cluster].values
    return x.astype({c: float for c in x.columns if x[c].dtype == bool})


def _fast_cox_class():
    """lifelines' CoxPHFitter with two residual computations rewritten as vectorized cumulative sums. The stock versions
    loop over every row and, inside the loop, work on all earlier rows (score residuals, used for the company-clustered
    robust errors) or grow an array one row at a time (Schoenfeld residuals, used for the proportional-hazards test), so
    their time and memory grow with the square of the number of rows (tens of hours at 800K rows). These are the same
    formulas (same tie handling, same row order) in O(rows x covariates); results agree up to float rounding."""
    from lifelines import CoxPHFitter
    from lifelines.fitters.coxph_fitter import SemiParametricPHFitter

    class FastModel(SemiParametricPHFitter):
        def _compute_score_within_strata(self, X, _T, E, weights, entries):
            X, E, w = X.values, E.values.astype(float), weights.values
            beta = self.params_.values * self._norm_std  # X is the normalized design here, as in lifelines
            phi = np.exp(X @ beta)
            risk_phi = np.cumsum((w * phi)[::-1])[::-1]  # risk set of row k = rows k.. (sorted by time)
            s_over_r = np.cumsum((X * (w * phi)[:, None])[::-1], axis=0)[::-1]
            s_over_r /= risk_phi[:, None]  # S_k / R_k
            ew_over_r = E * w / risk_phi
            a = np.cumsum(ew_over_r)  # sum over rows k <= i of E_k w_k / R_k
            b = np.cumsum(s_over_r * ew_over_r[:, None], axis=0)  # ... of E_k w_k S_k / R_k^2
            score = X * a[:, None]
            score -= b
            del b
            score *= -phi[:, None]
            score += E[:, None] * (X - s_over_r)
            score *= w[:, None]
            return score

        def _compute_schoenfeld_within_strata(self, X, T, E, weights, entries):
            X, T, E = X.values, T.values, E.values.astype(bool)
            n, d = X.shape
            out = np.zeros((n, d))
            if not E.any():
                return out
            phi = weights.values * np.exp(X @ self.params_.values)
            starts = np.r_[0, np.flatnonzero(np.diff(T)) + 1]
            ends = np.r_[starts[1:], n]
            phi_x = X * phi[:, None]
            csum = np.add.reduceat(phi, starts)[::-1].cumsum()[::-1]  # risk set at each distinct time: it and everything later
            csum_x = np.add.reduceat(phi_x, starts, axis=0)[::-1].cumsum(axis=0)[::-1]
            for g, (a, b) in enumerate(zip(starts, ends)):
                ev = E[a:b]
                k = int(ev.sum())
                if k == 0:
                    continue
                tie_phi, tie_x = phi[a:b][ev].sum(), phi_x[a:b][ev].sum(axis=0)
                l = np.arange(k)[:, None] / k  # Efron: the tied deaths leave the risk set a fraction at a time
                mean = ((csum_x[g][None, :] - l * tie_x[None, :]) / ((csum[g] - l * tie_phi) * k)).sum(axis=0)
                out[a:b][ev] = X[a:b][ev] - mean
            return out

    class FastCoxPHFitter(CoxPHFitter):  # CoxPHFitter hands the fit to a SemiParametricPHFitter; use the fast one instead
        def _fit_model_breslow(self, *args, **kwargs):
            model = FastModel(penalizer=self.penalizer, l1_ratio=self.l1_ratio, strata=self.strata, alpha=self.alpha, label=self._label)
            model.fit(*args, **kwargs)
            return model

    return FastCoxPHFitter


def cox_adjusted(df, duration, event, covariates, cluster=None, categorical=(), penalizer=0.0):
    """Cox proportional-hazards fit with all covariates at once and company-clustered (robust) errors.

    Returns (fitted model, design frame, hazard-ratio table with 95% intervals and p). The hazard is
    that of the event (here: closing), so a hazard ratio above 1 means closing faster."""
    x = cox_design(df, duration, event, covariates, cluster, categorical)
    cph = _fast_cox_class()(penalizer=penalizer)
    cph.fit(x, duration_col=duration, event_col=event, cluster_col=cluster, robust=bool(cluster))
    s = cph.summary
    table = pd.DataFrame({"hazard_ratio": s["exp(coef)"], "ci_low": s["exp(coef) lower 95%"],
                          "ci_high": s["exp(coef) upper 95%"], "p": s["p"]})
    return cph, x, table


def proportional_hazards_check(cph, design, time_transform="rank"):
    """Schoenfeld-residual test per covariate (`design` from cox_adjusted). Small p flags an effect
    that changes over time."""
    from lifelines.statistics import proportional_hazard_test
    r = proportional_hazard_test(cph, design.copy(), time_transform=time_transform)
    return r.summary[["test_statistic", "p"]]


def joint_wald(params, cov, columns):
    """One chi-square (Wald) test that every dummy column of a multi-level feature is zero, from a fit's coefficients
    and (robust) covariance: W = b' pinv(V) b with df = number of columns. Returns dict: wald, df, p."""
    cols = list(columns)
    b = pd.Series(params).loc[cols].to_numpy(float)
    v = pd.DataFrame(cov).loc[cols, cols].to_numpy(float)
    w = float(b @ np.linalg.pinv(v) @ b)
    return {"wald": w, "df": len(cols), "p": float(stats.chi2.sf(w, len(cols)))}


# ---------- 8. survival checks ----------
def censoring_summary(days, finished, group=None):
    """Share censored (still open at the end) and median days of finished vs censored listings."""
    d = pd.DataFrame({"days": np.asarray(days, float), "finished": np.asarray(finished).astype(bool)})
    d["g"] = np.asarray(group) if group is not None else "all"
    rows = {}
    for k, x in d.groupby("g", observed=True):
        rows[k] = {"n": len(x), "share_censored": 1 - x.finished.mean(),
                   "median_days_finished": x.loc[x.finished, "days"].median(),
                   "median_days_censored": x.loc[~x.finished, "days"].median()}
    return pd.DataFrame(rows).T


def km_with_entry(days, finished, entry=None):
    """Kaplan-Meier fit, optionally with delayed entry (left truncation: listings already open when
    collection started enter the risk set at their age on that date)."""
    from lifelines import KaplanMeierFitter
    return KaplanMeierFitter().fit(days, finished, entry=entry)


def left_truncation_gap(days, finished, entry, at=(7, 14, 30)):
    """Survival at given days, ignoring vs accounting for left truncation; the difference is the bias."""
    naive, adj = km_with_entry(days, finished), km_with_entry(days, finished, entry)
    return pd.DataFrame({"naive": [float(naive.predict(t)) for t in at],
                         "left_truncation_adjusted": [float(adj.predict(t)) for t in at]}, index=list(at))


def cumulative_incidence(days, cause, causes, at=(7, 30, 90)):
    """Aalen-Johansen cumulative incidence per cause (0 = still open/censored). Competing risks,
    e.g. 1 filled, 2 reposted, 3 ghost. Returns the incidence at the given days."""
    from lifelines import AalenJohansenFitter
    out = {}
    for c in causes:
        f = AalenJohansenFitter(calculate_variance=False, seed=0)
        f.fit(np.asarray(days, float), np.asarray(cause), event_of_interest=c)
        ci = f.cumulative_density_
        out[c] = [float(ci[ci.index <= t].iloc[-1, 0]) if (ci.index <= t).any() else 0.0 for t in at]
    return pd.DataFrame(out, index=list(at)).T.rename_axis("cause")


# ---------- 9. in-depth feature analysis ----------
def effect_by_group(df, outcome, feature, group, min_n=30):
    """Effect of a 0/1 feature on the outcome (difference in means) inside each group (month, role bucket),
    with sign agreement with the pooled effect. Groups with fewer than `min_n` per arm give NaN."""
    d = df[[outcome, feature, group]].dropna().copy()
    d[feature] = d[feature].astype(bool)
    pooled = d.loc[d[feature], outcome].mean() - d.loc[~d[feature], outcome].mean()
    mean, size = _mean_size_by_flag(d, outcome, feature, group)
    n_a, n_b = size[True].fillna(0).to_numpy(float), size[False].fillna(0).to_numpy(float)
    enough = (n_a >= min_n) & (n_b >= min_n)
    gap = np.where(enough, (mean[True] - mean[False]).to_numpy(float), np.nan)
    t = pd.DataFrame({"n_feature": n_a, "n_other": n_b, "gap": gap,
                      "same_sign_as_pooled": np.where(enough, (np.sign(gap) == np.sign(pooled)).astype(float), np.nan)},
                     index=pd.Index(mean.index.tolist()))
    t.attrs["pooled_gap"] = float(pooled)
    return t


def stability_over_months(df, outcome, features, month, min_n=30):
    """One row per feature: pooled gap, months with enough listings, share of months with the pooled
    sign, and the smallest and largest monthly gap."""
    rows = {}
    for f in features:
        t = effect_by_group(df, outcome, f, month, min_n)
        ok = t.dropna(subset=["gap"])
        rows[f] = {"pooled_gap": t.attrs["pooled_gap"], "months_tested": len(ok),
                   "share_same_sign": ok.same_sign_as_pooled.mean() if len(ok) else np.nan,
                   "min_gap": ok.gap.min() if len(ok) else np.nan, "max_gap": ok.gap.max() if len(ok) else np.nan}
    return pd.DataFrame(rows).T


def within_company_effect(df, outcome, feature, company):
    """Effect of a 0/1 feature among listings of the same company: per-company difference between the
    feature=1 and feature=0 means, averaged and weighted by the smaller arm. Only companies with both
    kinds of listing count. Removes company-level confounding."""
    d = df[[outcome, feature, company]].dropna().copy()
    d[feature] = d[feature].astype(bool)
    crude = float(d.loc[d[feature], outcome].mean() - d.loc[~d[feature], outcome].mean())
    g = d.groupby([company, feature], observed=True)[outcome].agg(["mean", "size"]).unstack(feature)
    empty = {"companies_used": 0, "within_gap": np.nan, "crude_gap": crude}
    if True not in g["mean"].columns or False not in g["mean"].columns:
        return empty
    both = g.dropna()
    if both.empty:
        return empty
    diff = both["mean"][True] - both["mean"][False]
    w = np.minimum(both["size"][True], both["size"][False])
    return {"companies_used": int(len(both)), "within_gap": float((diff * w).sum() / w.sum()), "crude_gap": crude}


def leakage_audit(features):
    """Table from a dict {feature: (known_at_first_sight: bool or None, why: str)}, with a verdict column (None = to decide).
    The answers are the analyst's; this only lays them out."""
    t = pd.DataFrame(features, index=["known_at_first_sight", "why"]).T
    undecided = t.known_at_first_sight.isna()
    t["verdict"] = np.where(undecided, "to decide", np.where(t.known_at_first_sight.fillna(False).astype(bool), "ok", "not known at first sight: do not use"))
    return t


def drift_over_time(df, feature, month, reference=None):
    """Distribution shift of a numeric or 0/1 feature versus a reference month (default: first):
    count, mean, standardized mean difference and Kolmogorov-Smirnov statistic per month."""
    d = df[[feature, month]].dropna().copy()
    d[feature] = d[feature].astype(float)
    months = sorted(d[month].unique())
    by_month = {m: g for m, g in d.groupby(month, observed=True)[feature]}
    ref = by_month[reference if reference is not None else months[0]]
    ref_var = ref.var(ddof=1)
    rows = {}
    for m in months:
        x = by_month[m]
        sd = np.sqrt((x.var(ddof=1) + ref_var) / 2) if len(x) > 1 and len(ref) > 1 else np.nan
        rows[m] = {"n": len(x), "mean": x.mean(),
                   "std_mean_diff": (x.mean() - ref.mean()) / sd if sd and sd > 0 else np.nan,
                   "ks": float(stats.ks_2samp(x, ref).statistic)}
    return pd.DataFrame(rows).T


def psi(before, after, bins=10, floor=1e-4):
    """Population stability index of `after` against `before`. Numbers are binned at `before`'s quantiles (the first
    and last bins open-ended); categories and flags are compared by share of each level; missing is its own bin;
    shares are floored at `floor`. Read: below 0.1 steady, 0.1 to 0.25 moderate, above 0.25 large."""
    b, a = pd.Series(before), pd.Series(after)
    numeric = pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b)
    if numeric:
        edges = np.unique(np.nanquantile(b.astype(float), np.linspace(0, 1, bins + 1)[1:-1]))
        def cut(s):
            v = s.to_numpy(dtype=float, na_value=np.nan)
            return np.where(np.isnan(v), 0, np.searchsorted(edges, v, side="right") + 1)  # bin codes; 0 = missing
        cb, ca = cut(b), cut(a)
    else:
        codes, _ = pd.factorize(pd.concat([b, a], ignore_index=True))  # one coding for both; missing is -1
        cb, ca = codes[:len(b)] + 1, codes[len(b):] + 1
    k = int(max(cb.max(initial=0), ca.max(initial=0))) + 1
    pb, pa = (np.bincount(c, minlength=k) / len(c) for c in (cb, ca))  # levels in neither sample have share 0 in both: no contribution
    pb, pa = np.clip(pb, floor, None), np.clip(pa, floor, None)
    return float(((pa - pb) * np.log(pa / pb)).sum())


def code_feature(s, bins=8):
    """Integer codes for conditional_mi: flags 0/1, numbers in `bins` quantile bins, categories factorized; missing -1."""
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()) <= {0, 1, True, False}:
        codes = s.astype(float)
    elif pd.api.types.is_numeric_dtype(s):
        codes = pd.Series(pd.qcut(s.astype(float), bins, labels=False, duplicates="drop"), index=s.index)
    else:
        codes = pd.Series(pd.factorize(s)[0], index=s.index).where(s.notna())
    return codes.fillna(-1).astype(int).to_numpy()


def _cmi_setup(codes, cell_codes):
    """Everything conditional_mi needs that does not depend on the outcome: dense cell and (cell, feature) codes and their counts."""
    _, c = np.unique(np.asarray(cell_codes), return_inverse=True)
    _, x = np.unique(np.asarray(codes), return_inverse=True)
    cx_key = c.astype(np.int64) * (int(x.max()) + 1) + x
    cx_u, cx = np.unique(cx_key, return_inverse=True)
    return {"c": c, "cx": cx, "n_c": np.bincount(c), "n_cx": np.bincount(cx), "cx_cell": cx_u // (int(x.max()) + 1), "n": len(c)}


def _cmi_value(st, yc, ny):
    """I(X; Y | cell) in nats for outcome codes `yc` (0..ny-1), from the counts in `st`."""
    ncx, nc = len(st["n_cx"]), len(st["n_c"])
    n_cxy = np.bincount(st["cx"] * ny + yc, minlength=ncx * ny).reshape(ncx, ny)
    n_cy = np.bincount(st["c"] * ny + yc, minlength=nc * ny).reshape(nc, ny)
    r, v = np.nonzero(n_cxy)
    cell = st["cx_cell"][r]
    k = n_cxy[r, v].astype(float)
    return float((k / st["n"] * np.log(k * st["n_c"][cell] / (st["n_cx"][r] * n_cy[cell, v]))).sum())


def conditional_mi(codes, y, cell_codes):
    """Mutual information (nats) between a coded feature and a 0/1 outcome within confounder cells, summed over the
    cells weighted by their share: I(X; Y | cell). Codes from code_feature; cell_codes = one integer per row naming its
    cell (e.g. factorized combinations of the confounders)."""
    _, yc = np.unique(np.asarray(y).astype(int), return_inverse=True)
    return _cmi_value(_cmi_setup(codes, cell_codes), yc, int(yc.max()) + 1)


def shuffle_within(y, cell_codes, rng):
    """`y` permuted inside each confounder cell: the noise floor for conditional_mi (keeps each cell's outcome rate)."""
    y, cells = np.asarray(y).copy(), np.asarray(cell_codes)
    order = np.argsort(cells, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(cells[order])) + 1, len(cells)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        y[idx] = y[rng.permutation(idx)]
    return y


def shuffled_outcomes(y, cell_codes, n_shuffles=20, seed=0):
    """The `n_shuffles` within-cell shuffles of the outcome that `conditional_mi_vs_floor` draws for `seed`, as outcome codes,
    computed once. They depend only on the outcome, the cells and the seed, so every feature of one table can share them
    (pass as `shuffled=`): the same numbers as shuffling again for each feature, without redoing the shuffles."""
    _, yc = np.unique(np.asarray(y).astype(int), return_inverse=True)
    cells = np.asarray(cell_codes)
    order = np.argsort(cells, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(cells[order])) + 1, len(cells)]
    keep = np.diff(starts) > 1  # a cell of one row has nothing to shuffle (and draws no random numbers)
    spans = [(order[a:b]) for a, b in zip(starts[:-1][keep], starts[1:][keep])]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_shuffles):
        y2 = yc.copy()
        for idx in spans:
            y2[idx] = yc[rng.permutation(idx)]
        out.append(y2)
    return out


def conditional_mi_vs_floor(codes, y, cell_codes, n_shuffles=20, seed=0, shuffled=None):
    """Observed conditional MI against the 95th percentile of the same statistic with `y` shuffled within cells.
    `shuffled`: the output of `shuffled_outcomes` for the same y, cells and seed, to reuse across features."""
    _, yc = np.unique(np.asarray(y).astype(int), return_inverse=True)
    ny = int(yc.max()) + 1
    st = _cmi_setup(codes, cell_codes)
    obs = _cmi_value(st, yc, ny)
    if shuffled is None:
        shuffled = shuffled_outcomes(y, cell_codes, n_shuffles, seed)
    p95 = float(np.percentile([_cmi_value(st, y2, ny) for y2 in shuffled], 95))
    return {"cmi": obs, "floor_p95": p95, "above_floor": bool(obs > p95)}


def information_beyond_confounders(df, outcome, feature, confounders, categorical=(), base_cache=None):
    """Likelihood-ratio test: does `feature` improve a logistic model of a 0/1 outcome that already has
    the confounders? Returns the LR statistic, p, the pseudo-R2 gain and the odds ratio.
    `base_cache`: a dict to share between calls; the confounders-only model depends only on which rows are used, so features
    with the same complete rows reuse it instead of refitting."""
    import statsmodels.api as sm
    d = df[[outcome, feature, *confounders]].dropna()
    base = pd.get_dummies(d[confounders], columns=[c for c in categorical if c in confounders], drop_first=True, dtype=float)
    base = sm.add_constant(base.astype(float), has_constant="add")
    full = base.assign(**{feature: d[feature].astype(float)})
    y = d[outcome].astype(float)
    key = (outcome, tuple(confounders), hash(d.index.to_numpy().tobytes())) if base_cache is not None else None
    if key is not None and key in base_cache:
        llf0 = base_cache[key]
    else:
        llf0 = sm.Logit(y, base).fit(disp=0).llf
        if key is not None:
            base_cache[key] = llf0
    m1 = sm.Logit(y, full).fit(disp=0)
    lr = 2 * (m1.llf - llf0)
    p1 = float(y.mean())  # log-likelihood of the constant-only model, in closed form (statsmodels finds the same value by a slow search)
    llnull = len(y) * (p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    return {"n": len(d), "lr_statistic": float(lr), "p": float(stats.chi2.sf(lr, 1)),
            "pseudo_r2_gain": float((llf0 - m1.llf) / llnull), "odds_ratio": float(np.exp(m1.params[feature]))}


def mutual_information(df, outcome, features, seed=0, n_jobs=4):
    """Mutual information of each numeric/0-1 feature with a 0/1 outcome (sklearn; NaN filled with the median).
    `n_jobs` workers split the features (the random tie-breaking noise is drawn once beforehand, so the numbers do not change)."""
    from sklearn.feature_selection import mutual_info_classif
    x = df[features].astype(float)
    x = x.fillna(x.median())
    mi = mutual_info_classif(x, df[outcome].astype(int), random_state=seed, n_jobs=n_jobs)
    return pd.Series(mi, index=features, name="mutual_information")


# ---------- 10-11. decisions and the table ----------
DECISIONS = ("kept", "dropped", "combined")
MODELING_TABLE_NAME = "modeling_ready.parquet"  # data/clean/, handed to TimeSeries, ML_Fit and ML_Rank


def decision_log(rows):
    """Feature decisions as a table. rows: list of (feature, decision, reason, evidence_section).
    The analyst decides; this only checks the vocabulary."""
    t = pd.DataFrame(rows, columns=["feature", "decision", "reason", "evidence"]).set_index("feature")
    bad = set(t.decision) - set(DECISIONS)
    if bad:
        raise ValueError(f"decision must be one of {DECISIONS}, got {sorted(bad)}")
    return t


def modeling_ready(scope, text, decisions, id_col="id", keep_always=()):
    """Join the scope table (EDA_SDA_2) and the text table (EDA_SDA_3) on `id` and keep the columns decided
    'kept' or 'combined', plus `keep_always` (id, company_id, month, days_open, live ...). Left join:
    listings without text keep blank text features."""
    keep = list(dict.fromkeys([id_col, *keep_always, *decisions.index[decisions.decision.isin(["kept", "combined"])]]))
    joined = scope.merge(text, on=id_col, how="left", suffixes=("", "_text"))
    missing = [c for c in keep if c not in joined]
    if missing:
        raise KeyError(f"kept columns not found: {missing}")
    return joined[keep]
