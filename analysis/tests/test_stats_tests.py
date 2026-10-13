"""EDA_SDA_4 statistics helpers (src/stats_tests.py) on made-up data. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import stats_tests as st


def fake(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 60, n)
    flag = rng.random(n) < 0.3
    big = rng.random(n) < 0.4
    month = rng.choice(["2026-04", "2026-05", "2026-06", "2026-07"], n)
    rate = 0.02 + 0.03 * flag + 0.02 * big          # daily hazard of closing
    days = np.minimum(rng.exponential(1 / rate), 60)
    done = days < 60
    return pd.DataFrame({"company_id": comp, "flag": flag, "big": big, "month": month, "days_open": days,
                         "closed": done.astype(int), "x": rng.normal(size=n), "role": rng.choice(["a", "b", "c"], n),
                         "closed7": (days <= 7).astype(int)})


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.df = fake()

    def test_describe_and_correlations(self):
        t = st.describe_by_group(self.df, "x", "role")
        self.assertEqual(set(t.index), {"a", "b", "c"})
        self.assertTrue((t["q1"] <= t["median"]).all() and (t["median"] <= t["q3"]).all())
        c = st.correlations(self.df.assign(y=self.df.x * 2), ["x"], "y")
        self.assertAlmostEqual(c.loc["x", "pearson"], 1.0)
        self.assertAlmostEqual(c.loc["x", "spearman"], 1.0)

    def test_cramers_v_and_point_biserial(self):
        x = pd.Series(list("aabbaabb" * 25))
        self.assertGreater(st.cramers_v(x, x, bias_correct=False), 0.99)
        rng = np.random.default_rng(1)
        self.assertLess(st.cramers_v(rng.choice(list("abc"), 2000), rng.choice(list("xy"), 2000)), 0.1)
        r, p = st.point_biserial(self.df.flag, self.df.flag * 3.0 + self.df.x * 0.1)
        self.assertGreater(r, 0.9)
        self.assertTrue(np.isnan(st.point_biserial([1, 1, 1], [1, 2, 3])[0]))

    def test_vif_flags_collinearity(self):
        d = self.df.assign(x2=self.df.x + np.random.default_rng(2).normal(scale=0.01, size=len(self.df)), z=self.df.x * 0 + np.arange(len(self.df)) % 7)
        v = st.vif(d, ["x", "x2", "z"])
        self.assertGreater(v["x"], 100)
        self.assertLess(v["z"], 5)

    def test_simpson_reverses(self):
        # flag looks bad overall (mostly in the low stratum) but is better within each stratum
        rows = ([("hi", 1, 0.8)] * 20 + [("hi", 0, 0.7)] * 80 + [("lo", 1, 0.3)] * 80 + [("lo", 0, 0.2)] * 20)
        d = pd.DataFrame(rows, columns=["s", "f", "y"])
        tab, summ = st.stratified_gap(d, "y", "f", "s")
        self.assertLess(summ["crude_gap"], 0)
        self.assertGreater(summ["adjusted_gap"], 0)
        self.assertTrue(summ["reverses"])
        self.assertEqual(len(tab), 2)

    def test_tests_and_effect_sizes(self):
        a, b = self.df.x[self.df.flag], self.df.x[~self.df.flag]
        w, m = st.welch_test(a + 1, b), st.mann_whitney(a + 1, b)
        self.assertLess(w["p"], 1e-6)
        self.assertGreater(w["effect"], 0.5)
        self.assertGreater(m["effect"], 0)
        c = st.chi_square(self.df.role, self.df.role)
        self.assertGreater(c["effect"], 0.99)
        f, g = self.df[self.df.flag], self.df[~self.df.flag]
        for weighted in (False, True):
            lr = st.logrank(f.days_open, f.closed, g.days_open, g.closed, weighted=weighted)
            self.assertLess(lr["p"], 0.05)
            self.assertGreater(lr["effect"], 1)

    def test_benjamini_hochberg(self):
        q = st.benjamini_hochberg([0.01, 0.04, 0.03, np.nan, 0.5])
        self.assertAlmostEqual(q[0], 0.04)
        self.assertAlmostEqual(q[1], 0.0533333, places=4)
        self.assertAlmostEqual(q[2], 0.0533333, places=4)
        self.assertTrue(np.isnan(q[3]))
        self.assertTrue((q.dropna().values >= np.array([0.01, 0.04, 0.03, 0.5])).all())
        t = st.tests_table([{"name": "a", **st.welch_test(self.df.x, self.df.x + 5)}, {"name": "b", **st.welch_test(self.df.x, self.df.x[::-1])}])
        self.assertTrue(t.loc["a", "significant_after_bh"])

    def test_clustered_wider_than_plain(self):
        # outcome driven by company: clustering must widen the interval
        rng = np.random.default_rng(3)
        d = pd.DataFrame({"c": np.repeat(np.arange(40), 50)})
        d["y"] = rng.normal(size=40)[d.c] + rng.normal(scale=0.2, size=len(d))
        fn = lambda f: f.y.mean()
        cl, pl = st.clustered_bootstrap(d, "c", fn, 200, seed=1), st.plain_bootstrap(d, fn, 200, seed=1)
        self.assertGreater(cl["ci_high"] - cl["ci_low"], 2 * (pl["ci_high"] - pl["ci_low"]))
        self.assertEqual(cl["n_clusters"], 40)
        self.assertEqual(cl, st.clustered_bootstrap(d, "c", fn, 200, seed=1))

    def test_fast_clustered_rate_matches_general(self):
        # same draws (same seed and cluster order), so the fast version must give the same interval
        rng = np.random.default_rng(4)
        d = pd.DataFrame({"c": np.repeat(np.arange(30), 20), "y": rng.integers(0, 2, 600).astype(float)})
        d.loc[::7, "y"] = np.nan
        slow = st.clustered_bootstrap(d.dropna(), "c", lambda f: f.y.mean(), 200, seed=2)
        fast = st.clustered_bootstrap_rate(d, "c", "y", 200, seed=2)
        for k in ("estimate", "ci_low", "ci_high", "n_clusters"):
            self.assertAlmostEqual(slow[k], fast[k])

    def test_cox_and_ph_check(self):
        cph, x, tab = st.cox_adjusted(self.df, "days_open", "closed", ["flag", "big", "role"], cluster="company_id", categorical=["role"])
        self.assertIn("flag", tab.index)
        self.assertGreater(tab.loc["flag", "hazard_ratio"], 1)
        self.assertTrue((tab.ci_low <= tab.hazard_ratio).all() and (tab.hazard_ratio <= tab.ci_high).all())
        self.assertNotIn("company_id", tab.index)
        ph = st.proportional_hazards_check(cph, x)
        self.assertIn("flag", ph.index)

    def test_survival_checks(self):
        s = st.censoring_summary(self.df.days_open, self.df.closed, self.df.role)
        self.assertTrue(((s.share_censored >= 0) & (s.share_censored <= 1)).all())
        entry = np.where(np.arange(len(self.df)) % 3 == 0, 0.0, 0.0) + np.where(np.arange(len(self.df)) % 3 == 1, self.df.days_open * 0.5, 0.0)
        gap = st.left_truncation_gap(self.df.days_open + 0.01, self.df.closed, entry)
        self.assertEqual(list(gap.columns), ["naive", "left_truncation_adjusted"])
        rng = np.random.default_rng(4)
        days = rng.exponential(20, 600) + 0.01
        cause = rng.choice([0, 1, 2], 600, p=[0.3, 0.5, 0.2])
        ci = st.cumulative_incidence(days, cause, [1, 2])
        self.assertTrue((ci.loc[1] >= ci.loc[2]).all())
        self.assertLessEqual(ci.sum().max(), 1.0)

    def test_in_depth_checks(self):
        by_month = st.stability_over_months(self.df, "closed7", ["flag", "big"], "month")
        self.assertEqual(by_month.loc["flag", "months_tested"], 4)
        self.assertGreaterEqual(by_month.loc["flag", "share_same_sign"], 0.5)
        by_role = st.effect_by_group(self.df, "closed7", "flag", "role")
        self.assertEqual(len(by_role), 3)
        # company confounding: within-company gap is zero, crude is not
        rng = np.random.default_rng(5)
        d = pd.DataFrame({"c": np.repeat(np.arange(50), 20)})
        d["f"] = rng.random(len(d)) < np.where(d.c < 25, 0.8, 0.2)
        d["y"] = (d.c < 25) * 1.0 + rng.normal(scale=0.1, size=len(d))
        w = st.within_company_effect(d, "y", "f", "c")
        self.assertGreater(w["crude_gap"], 0.3)
        self.assertLess(abs(w["within_gap"]), 0.1)
        self.assertGreater(w["companies_used"], 40)

    def test_leakage_drift_information(self):
        a = st.leakage_audit({"level": (True, "title"), "times_listed": (False, "counted later"), "new": (None, "")})
        self.assertEqual(a.loc["times_listed", "verdict"], "not known at first sight: do not use")
        self.assertEqual(a.loc["new", "verdict"], "to decide")
        d = self.df.assign(shifted=self.df.x + (self.df.month >= "2026-06") * 2.0)
        dr = st.drift_over_time(d, "shifted", "month")
        self.assertGreater(dr.loc["2026-06", "std_mean_diff"], 1)
        self.assertLess(abs(st.drift_over_time(d, "x", "month").loc["2026-06", "std_mean_diff"]), 0.3)
        lr = st.information_beyond_confounders(self.df, "closed7", "flag", ["big", "role"], categorical=["role"])
        self.assertLess(lr["p"], 0.05)
        self.assertGreater(lr["odds_ratio"], 1)
        mi = st.mutual_information(self.df, "closed7", ["flag", "x"])
        self.assertTrue((mi >= 0).all() and list(mi.index) == ["flag", "x"])

    def test_decisions_and_final_table(self):
        dec = st.decision_log([("a", "kept", "r", "3"), ("b", "dropped", "r", "9"), ("c", "combined", "r", "4")])
        with self.assertRaises(ValueError):
            st.decision_log([("a", "maybe", "r", "3")])
        scope = pd.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6], "company_id": [7, 7, 8]})
        text = pd.DataFrame({"id": [1, 2], "c": [0.5, 0.6]})
        out = st.modeling_ready(scope, text, dec, keep_always=["company_id"])
        self.assertEqual(list(out.columns), ["id", "company_id", "a", "c"])
        self.assertEqual(len(out), 3)
        self.assertTrue(out.c.isna().iloc[2])
        with self.assertRaises(KeyError):
            st.modeling_ready(scope, text, st.decision_log([("zzz", "kept", "r", "3")]))


class PortedHelpers(unittest.TestCase):
    def test_downcast_keeps_values_and_blanks(self):
        df = pd.DataFrame({"f": [1.5, np.nan, 3.25], "i": np.array([1, 200, 3], dtype=np.int64),
                           "cat": ["a", None, "b"], "flag": [True, None, False], "nullable": pd.array([1, None, 3], dtype="Int64")})
        df["many"] = [f"x{k}" for k in range(3)]
        out = st.downcast(df, max_categories=2)
        self.assertEqual(out.f.dtype, np.float32)
        self.assertEqual(out.i.dtype, np.int16)
        self.assertEqual(str(out.cat.dtype), "category")
        self.assertEqual(out.flag.dtype, object)           # bool-like objects stay
        self.assertEqual(str(out.nullable.dtype), "Int64")  # nullable ints stay
        self.assertEqual(out.many.dtype, object)            # 3 distinct > max_categories
        self.assertTrue(np.allclose(out.f.astype(float), df.f, equal_nan=True))
        self.assertTrue((out.i == df.i).all())
        self.assertEqual(out.cat.isna().tolist(), df.cat.isna().tolist())
        self.assertEqual(out.flag.isna().tolist(), df.flag.isna().tolist())

    def test_psi_steady_vs_shifted(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(size=5000))
        self.assertLess(st.psi(a, pd.Series(rng.normal(size=5000))), 0.1)
        self.assertGreater(st.psi(a, pd.Series(rng.normal(1, 1, 5000))), 0.25)
        cats = pd.Series(rng.choice(list("abc"), 5000, p=[0.6, 0.3, 0.1]))
        self.assertGreater(st.psi(cats, pd.Series(rng.choice(list("abcd"), 5000, p=[0.2, 0.3, 0.1, 0.4]))), 0.25)
        with_missing = a.copy(); with_missing[:1000] = np.nan
        self.assertGreater(st.psi(a, with_missing), 0.1)  # missing is its own bin

    def test_conditional_mi_sees_signal_beyond_cells_and_not_confounding(self):
        rng = np.random.default_rng(1)
        n = 6000
        cell = rng.integers(0, 4, n)
        x_conf = (cell >= 2).astype(int)                            # fully explained by the cell
        y = (rng.random(n) < 0.2 + 0.15 * (cell >= 2)).astype(int)
        r = st.conditional_mi_vs_floor(st.code_feature(pd.Series(x_conf)), y, cell, n_shuffles=10)
        self.assertAlmostEqual(r["cmi"], 0.0, places=10)
        self.assertFalse(r["above_floor"])
        x_real = rng.random(n) < 0.5
        y2 = (rng.random(n) < 0.15 + 0.3 * x_real).astype(int)
        r2 = st.conditional_mi_vs_floor(st.code_feature(pd.Series(x_real)), y2, cell, n_shuffles=10)
        self.assertTrue(r2["above_floor"])

    def test_shuffle_within_keeps_cell_rates(self):
        rng = np.random.default_rng(2)
        cell = rng.integers(0, 5, 500)
        y = rng.integers(0, 2, 500)
        s = st.shuffle_within(y, cell, rng)
        for c in range(5):
            self.assertEqual(s[cell == c].sum(), y[cell == c].sum())

    def test_code_feature(self):
        self.assertEqual(st.code_feature(pd.Series([True, None, False], dtype="boolean")).tolist(), [1, -1, 0])
        self.assertEqual(st.code_feature(pd.Series(["b", None, "a", "b"])).tolist(), [0, -1, 1, 0])
        self.assertEqual(sorted(set(st.code_feature(pd.Series(np.arange(100.0)), bins=8))), list(range(8)))

    def test_joint_wald(self):
        cov = pd.DataFrame(np.eye(3) * 0.01, index=list("abc"), columns=list("abc"))
        null = st.joint_wald(pd.Series({"a": 0.0, "b": 0.01, "c": 5.0}), cov, ["a", "b"])
        self.assertEqual(null["df"], 2)
        self.assertGreater(null["p"], 0.5)
        r = st.joint_wald(pd.Series({"a": 0.5, "b": 0.0, "c": 0.0}), cov, ["a", "b"])
        self.assertAlmostEqual(r["wald"], 25.0)                      # 0.5^2 / 0.01
        self.assertAlmostEqual(r["p"], np.exp(-12.5), places=12)     # chi-square with 2 df: p = exp(-W/2)


class FastPathsMatchReference(unittest.TestCase):
    """The speed-ups give the same numbers as the straightforward versions they replaced."""

    def test_fast_cox_matches_stock_lifelines(self):
        from lifelines import CoxPHFitter
        d = fake(1200, seed=3)
        d["days_open"] = np.ceil(d.days_open).clip(lower=1)  # integer days: many tied times, as in the real table
        x = st.cox_design(d, "days_open", "closed", ["flag", "big", "role"], "company_id", ["role"])
        fast = st._fast_cox_class()(penalizer=0.01).fit(x, "days_open", "closed", cluster_col="company_id", robust=True)
        stock = CoxPHFitter(penalizer=0.01).fit(x, "days_open", "closed", cluster_col="company_id", robust=True)
        np.testing.assert_allclose(fast.standard_errors_, stock.standard_errors_, rtol=1e-8)
        a, b = st.proportional_hazards_check(fast, x), st.proportional_hazards_check(stock, x)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-7, atol=1e-10)

    def test_conditional_mi_and_shared_shuffles_match_the_per_feature_version(self):
        rng = np.random.default_rng(4)
        n = 3000
        cell, x, y = rng.integers(0, 40, n), rng.integers(-1, 3, n), rng.integers(0, 2, n)
        d = pd.DataFrame({"c": cell, "x": x, "y": y})
        ref = 0.0
        cxy = d.groupby(["c", "x", "y"]).size()
        for (c, xv, yv), k in cxy.items():
            ref += k / n * np.log(k * (d.c == c).sum() / (((d.c == c) & (d.x == xv)).sum() * ((d.c == c) & (d.y == yv)).sum()))
        self.assertAlmostEqual(st.conditional_mi(x, y, cell), ref, places=12)
        shared = st.shuffled_outcomes(y, cell, n_shuffles=6, seed=5)
        r1 = st.conditional_mi_vs_floor(x, y, cell, n_shuffles=6, seed=5)
        r2 = st.conditional_mi_vs_floor(x, y, cell, shuffled=shared)
        self.assertEqual(r1, r2)
        gen = np.random.default_rng(5)  # the shuffle stream of the original per-feature loop
        for y2 in shared:
            np.testing.assert_array_equal(y2, st.shuffle_within(y, cell, gen))

    def test_contingency_psi_and_group_gaps_match_pandas(self):
        d = fake(2000, seed=6)
        d.loc[::17, "role"] = None
        tab = pd.crosstab(d.role, d.big)
        np.testing.assert_array_equal(st._contingency(d.role, d.big), tab.to_numpy())
        r, p = st.point_biserial(d.flag, d.x)
        r0, p0 = __import__("scipy.stats", fromlist=["x"]).pointbiserialr(d.flag.astype(int), d.x)
        self.assertAlmostEqual(r, r0, places=12)
        by = st.effect_by_group(d, "closed7", "flag", "role", min_n=1)
        for role, g in d.dropna(subset=["role"]).groupby("role"):
            self.assertAlmostEqual(by.loc[role, "gap"], g.closed7[g.flag].mean() - g.closed7[~g.flag].mean(), places=12)
            self.assertEqual(by.loc[role, "n_feature"], g.flag.sum())
        a = pd.Series(np.random.default_rng(7).normal(size=3000))
        shifted = a.iloc[:1500] + 0.5
        self.assertGreater(st.psi(a, shifted), 0.1)
        flags_b, flags_a = pd.Series([True, False, None, True] * 50, dtype="boolean"), pd.Series([True, True, None, None] * 50, dtype="boolean")
        pb, pa = np.array([2 / 4, 1 / 4, 1 / 4]), np.array([2 / 4, 0, 2 / 4])  # True, False, missing
        pa = np.clip(pa, 1e-4, None)
        self.assertAlmostEqual(st.psi(flags_b, flags_a), float(((pa - pb) * np.log(pa / pb)).sum()), places=12)


class NullableNumbers(unittest.TestCase):
    def test_welch_and_mann_whitney_accept_nullable_and_object_columns(self):
        rng = np.random.default_rng(0)
        a = pd.array(list(rng.integers(0, 50, 200)) + [None] * 5, dtype="Int64")
        b = pd.Series(list(rng.integers(10, 60, 200).astype(float)) + [np.nan], dtype="object")
        w, m = st.welch_test(a, b), st.mann_whitney(a, b)
        self.assertEqual((w["n_a"], w["n_b"]), (200, 200))
        self.assertTrue(np.isfinite(w["p"]) and np.isfinite(m["p"]))
        flags = pd.array([True, False, None] * 40, dtype="boolean")
        self.assertEqual(st.welch_test(flags, flags)["n_a"], 80)


class NestedConfounders(unittest.TestCase):
    def test_lr_test_runs_when_one_confounder_nests_in_another(self):
        rng = np.random.default_rng(1)
        n = 5_000
        platform = rng.choice(["p1", "p2", "p3", "p4"], n)
        source = np.where(np.isin(platform, ["p1", "p2"]), "site", "board")   # each platform belongs to one source type
        flag = rng.random(n) < 0.4
        y = (rng.random(n) < 0.2 + 0.1 * flag).astype(float)
        d = pd.DataFrame({"y": y, "flag": flag.astype(float), "platform": platform, "source": source})
        r = st.information_beyond_confounders(d, "y", "flag", ["source", "platform"], categorical=["source", "platform"])
        self.assertTrue(np.isfinite(r["lr_statistic"]) and r["lr_statistic"] > 10)

    def test_lr_test_runs_with_an_unused_category_level(self):
        # downcast() makes a column categorical; a subset (listings with a description) can lack a level, and its dummy
        # column is then all zeros: the plain logit failed with "Singular matrix" (EDA_SDA_4 9f)
        rng = np.random.default_rng(2)
        n = 5_000
        platform = pd.Categorical(rng.choice(["p1", "p2", "p3"], n), categories=["p1", "p2", "p3", "p4"])
        flag = rng.random(n) < 0.4
        y = (rng.random(n) < 0.2 + 0.1 * flag).astype(float)
        d = pd.DataFrame({"y": y, "flag": flag.astype(float), "platform": platform})
        r = st.information_beyond_confounders(d, "y", "flag", ["platform"], categorical=["platform"])
        self.assertTrue(np.isfinite(r["lr_statistic"]) and r["lr_statistic"] > 10)


if __name__ == "__main__":
    unittest.main()
