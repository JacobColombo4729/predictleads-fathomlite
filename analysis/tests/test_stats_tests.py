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


if __name__ == "__main__":
    unittest.main()
