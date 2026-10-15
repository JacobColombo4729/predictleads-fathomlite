"""ML_Rank scaffolding (src/rank_model.py) on made-up listings. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import rank_model as rm


def listings():
    return pd.DataFrame({
        "id": list("abcdef"),
        "company_id": ["x", "x", "x", "y", "y", "z"],
        "title": ["Python Developer", "Help Desk Technician", "Senior Python Engineer", "Data Analyst", "python tester", None],
        "first_seen": pd.to_datetime(["2026-04-01", "2026-04-10", "2026-04-10", "2026-08-05", "2026-09-10", "2026-09-29"]),
        "last_seen": pd.to_datetime(["2026-04-05", "2026-05-20", "2026-09-30", "2026-08-20", "2026-09-30", "2026-09-30"]),
    })


class Targets(unittest.TestCase):
    def test_survival_censors_listings_still_seen_at_the_end(self):
        t = rm.survival_target(listings())
        self.assertEqual(list(t["event"]), [1, 1, 0, 1, 0, 0])
        self.assertAlmostEqual(t["duration"].iloc[0], 4)

    def test_as_of_date_cuts_future_listings_and_censors_there(self):
        t = rm.survival_target(listings(), as_of="2026-05-01")
        self.assertEqual(list(t["id"]), ["a", "b", "c"])  # d, e, f first seen after the date: unknown then
        self.assertEqual(list(t["event"]), [1, 0, 0])  # b was still seen after May 1: censored at that day
        self.assertAlmostEqual(t["duration"].iloc[1], 21)

    def test_closes_within_marks_censored_before_n_as_unknown(self):
        t = rm.survival_target(listings(), as_of="2026-05-01")
        y = rm.closes_within(t, 10)
        self.assertEqual(y.iloc[0], 1.0)  # closed after 4 days
        self.assertEqual(y.iloc[1], 0.0)  # open 21 days > 10
        t2 = rm.survival_target(listings())
        self.assertTrue(np.isnan(rm.closes_within(t2, 10).iloc[5]))  # censored after 1 day: not "stayed open"

    def test_answer_known_either_way_drops_the_last_n_plus_tolerance_days(self):
        n, tol = 10, rm.LIVE_TOLERANCE_DAYS
        late = rm.EXPORT_END - pd.Timedelta(days=n + tol)
        first = pd.Series([late - pd.Timedelta(hours=1), late, rm.EXPORT_END - pd.Timedelta(days=1)])
        self.assertEqual(list(rm.answer_known_either_way(first, n)), [True, False, False])
        # Every listing kept has a known answer, whether it closed fast or stayed open
        rng = np.random.default_rng(0)
        fs = pd.Series(rm.EXPORT_END - pd.to_timedelta(rng.uniform(0, 60, 500), unit="D"))
        ls = (fs + pd.to_timedelta(rng.uniform(0, 40, 500), unit="D")).clip(upper=rm.EXPORT_END)
        df = pd.DataFrame({"id": range(500), "first_seen": fs, "last_seen": ls})
        y = rm.closes_within(rm.survival_target(df), n)
        keep = rm.answer_known_either_way(df["first_seen"], n)
        self.assertFalse(y[keep].isna().any())
        self.assertTrue(y[~keep].isna().any())

    def test_days_open_only_for_closed(self):
        d = rm.days_open(rm.survival_target(listings()))
        self.assertTrue(d.iloc[2:3].isna().all())
        self.assertAlmostEqual(d.iloc[0], 4)


    def test_one_closing_window_constant_drives_the_defaults(self):
        t = rm.survival_target(listings())
        pd.testing.assert_series_equal(rm.closes_within(t), rm.closes_within(t, rm.CLOSING_DAYS))
        self.assertIn(rm.CLOSING_DAYS, rm.SENSITIVITY_DAYS)

    def test_open_long_enough_is_the_opposite_with_the_same_unknowns(self):
        t = rm.survival_target(listings())
        a, b = rm.closes_within(t, 10), rm.open_long_enough(t, 10)
        self.assertTrue((a.isna() == b.isna()).all())
        self.assertTrue(((a + b).dropna() == 1).all())

    def test_sensitivity_windows_has_a_column_per_window(self):
        w = rm.sensitivity_windows(rm.survival_target(listings()), (7, 14, 30))
        self.assertEqual(list(w.columns), ["closes_within_7", "closes_within_14", "closes_within_30"])
        self.assertEqual(w["closes_within_7"].iloc[0], 1.0)  # a closed after 4 days

    def test_tolerance_is_shared_with_the_date_picker(self):
        from src import as_of
        self.assertEqual(rm.LIVE_TOLERANCE_DAYS, as_of.LIVE_TOLERANCE_DAYS)


class Split(unittest.TestCase):
    def test_time_split_by_month(self):
        s = rm.time_split(listings()["first_seen"])
        self.assertEqual(list(s), ["train", "train", "train", "validation", "test", "test"])

    def test_folds_never_share_a_company(self):
        comp = np.repeat(np.arange(20), 5)
        for tr, va in rm.company_folds(comp, n_splits=4):
            self.assertFalse(set(comp[tr]) & set(comp[va]))


class Leakage(unittest.TestCase):
    def test_prior_counts_use_earlier_listings_only(self):
        c = rm.prior_company_counts(listings())
        self.assertEqual(list(c), [0, 1, 1, 0, 1, 0])  # b and c share a first_seen: neither counts the other

    def test_flags_set_without_description_are_found(self):
        df = pd.DataFrame({"has_desc": [True, False, False], "urgency": [False, pd.NA, True]})
        r = rm.known_at_first_sight(df, ["urgency"])
        self.assertAlmostEqual(r["set_without_description"].iloc[0], 1 / 3)


class Baselines(unittest.TestCase):
    def test_newest_first_orders_by_first_seen(self):
        s = rm.newest_first_score(listings())
        self.assertEqual(s.idxmax(), 5)

    def test_title_keyword_ranks_matches_first_and_handles_missing_titles(self):
        s = rm.title_keyword_score(listings(), ["python"])
        top = s.sort_values(ascending=False).index[:3]
        self.assertEqual(set(top), {0, 2, 4})
        self.assertEqual(s.index[s.idxmax()], 4)  # newest of the matches

    def test_dummy_predicts_prior(self):
        m = rm.dummy_baseline().fit(np.zeros((4, 1)), [0, 0, 0, 1])
        self.assertTrue((m.predict_proba(np.zeros((2, 1)))[:, 1] == 0.25).all())


class Metrics(unittest.TestCase):
    def test_precision_and_ndcg(self):
        y, score = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
        self.assertAlmostEqual(rm.precision_at_k(y, score, 2), 0.5)
        self.assertAlmostEqual(rm.ndcg_at_k([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.0], 2), 1.0)
        self.assertLess(rm.ndcg_at_k([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.0], 2), 1e-9 + 0.0001)

    def test_c_index_perfect_and_reversed(self):
        d, e = [1, 2, 3, 4], [1, 1, 1, 1]
        self.assertAlmostEqual(rm.c_index(d, e, [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(rm.c_index(d, e, [4, 3, 2, 1]), 0.0)


class Models(unittest.TestCase):
    def test_every_model_fits_and_predicts_probabilities(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
        models = rm.build_models()
        self.assertEqual(set(models), {"logistic", "tree", "random_forest", "lightgbm", "xgboost", "catboost"})
        for name, m in models.items():
            p = m.fit(X, y).predict_proba(X)[:, 1]
            self.assertEqual(len(p), 120, name)
            self.assertGreater(((p > 0.5) == y).mean(), 0.7, name)

    def test_cox_fits(self):
        rng = np.random.RandomState(1)
        df = pd.DataFrame({"x": rng.normal(size=80)})
        df["duration"] = rng.exponential(np.exp(-df["x"] * 0.5)) + 0.1
        df["event"] = 1
        self.assertGreater(abs(rm.cox_model().fit(df, "duration", "event").params_["x"]), 0)

    def test_lambdarank_fits(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(40, 3))
        y = (X[:, 0] > 0).astype(int)
        r = rm.lambdarank_model().fit(X, y, group=[20, 20])
        self.assertEqual(len(r.predict(X)), 40)


def synth(n=300, seed=5):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"n": rng.normal(size=n), "m": np.where(rng.rand(n) < 0.1, np.nan, rng.normal(size=n)),
                      "c": rng.choice(["a", "b", "rare1", "rare2"], size=n, p=[0.5, 0.45, 0.025, 0.025])})
    y = (X["n"] + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    return X, y


class Preprocessing(unittest.TestCase):
    def test_pipeline_handles_blanks_and_unseen_levels_and_groups_rare_ones(self):
        X, y = synth()
        prep = rm.make_preprocessor(["n", "m"], ["c"], min_frequency=20)
        pipe = rm.with_preprocessing(prep, rm.build_models()["logistic"]).fit(X, y)
        new = X.head(3).copy()
        new["c"] = "never seen"
        self.assertEqual(len(pipe.predict_proba(new)), 3)
        cols = pipe.named_steps["prep"].get_feature_names_out()
        self.assertFalse(any("rare1" in c or "rare2" in c for c in cols))  # grouped as infrequent

    def test_prep_is_fit_on_training_rows_only(self):
        X, y = synth()
        prep = rm.make_preprocessor(["n", "m"], ["c"]).fit(X.iloc[:100])
        self.assertAlmostEqual(prep.named_transformers_["num"]["impute"].statistics_[0], X["n"].iloc[:100].median())

    def test_upsampling_balances_classes_and_keeps_originals(self):
        X, y = synth()
        X2, y2 = rm.upsample_minority(X, y)
        self.assertEqual(y2.value_counts().nunique(), 1)
        self.assertEqual(len(y2), 2 * (y == 0).sum())
        self.assertTrue(X2.iloc[:len(X)].equals(X.reset_index(drop=True)))

    def test_threshold_tuning_beats_the_default_on_rare_positives(self):
        y = np.array([1] * 10 + [0] * 90)
        p = np.concatenate([np.full(10, 0.3), np.full(90, 0.1)])  # nothing reaches 0.5
        t, f1 = rm.best_threshold(y, p, "f1")
        self.assertTrue(0.1 < t <= 0.3)
        self.assertAlmostEqual(f1, 1.0)

    def test_grid_search_over_company_folds(self):
        X, y = synth()
        comp = np.arange(len(y)) % 30
        folds = rm.company_folds(comp, n_splits=3)
        pipe = rm.with_preprocessing(rm.make_preprocessor(["n", "m"], ["c"]), rm.build_models()["tree"])
        gs = rm.grid_search(pipe, {"model__max_depth": [2, 4]}, X, y, folds)
        self.assertIn(gs.best_params_["model__max_depth"], [2, 4])
        self.assertGreater(gs.best_score_, 0.7)
        self.assertEqual(len(rm.cv_auc(pipe, X, y, folds)), 3)

    def test_prepared_folds_give_the_same_search_as_gridsearchcv(self):
        X, y = synth()
        comp = np.arange(len(y)) % 30
        folds = rm.company_folds(comp, n_splits=3)
        prep = rm.make_preprocessor(["n", "m"], ["c"])
        models = rm.build_models()
        grids = {"tree": {"model__max_depth": [2, 4], "model__min_samples_leaf": [5, 50]},
                 "logistic": {"model__logisticregression__C": [0.1, 1.0]}}
        pf = rm.prepare_folds(prep, X, y, folds)
        self.assertIs(pf, rm.prepare_folds(prep, X, y, folds))  # same rows, labels and folds: the earlier one
        for name, grid in grids.items():
            pipe = rm.with_preprocessing(prep, models[name])
            old = rm.grid_search(pipe, grid, X, y, folds)
            new = rm.grid_search(pipe, grid, X, y, folds, prepared=pf, fold_jobs=2)
            self.assertEqual(old.best_params_, new.best_params_)
            self.assertEqual(old.best_index_, new.best_index_)
            self.assertAlmostEqual(old.best_score_, new.best_score_, places=10)
            for key in ["mean_test_score", "std_test_score", "rank_test_score", "split0_test_score", "split2_test_score"]:
                np.testing.assert_allclose(old.cv_results_[key], new.cv_results_[key], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(old.best_estimator_.predict_proba(X), new.best_estimator_.predict_proba(X), atol=1e-10)
            np.testing.assert_allclose(rm.cv_auc(pipe, X, y, folds), rm.cv_auc(pipe, X, y, folds, prepared=pf), atol=1e-12)
        other = rm.prepare_folds(prep, X, 1 - np.asarray(y), folds)  # other labels: not the earlier one
        self.assertIsNot(other, pf)

    def test_param_grids_name_every_model(self):
        self.assertEqual(set(rm.PARAM_GRIDS), set(rm.build_models()))


def surv(n=200, seed=7):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "company_id": rng.randint(0, 40, size=n)})
    df["duration"] = rng.exponential(np.exp(-0.7 * df["x"])) * 10 + 0.1
    df["event"] = (rng.rand(n) < 0.8).astype(int)
    return df


class Survival(unittest.TestCase):
    def test_cox_with_company_clusters_and_ph_check(self):
        df = surv()
        m = rm.fit_cox(df, ["x", "z"])
        self.assertGreater(m.params_["x"], 0.3)
        ph = rm.cox_ph_check(m, df, ["x", "z"])
        self.assertEqual(list(ph.index), ["x", "z"])
        self.assertTrue(ph["p"].between(0, 1).all())

    def test_fast_scaled_schoenfeld_equals_lifelines_with_and_without_ties(self):
        for round_to in (None, 1.0):  # continuous times, then times rounded to whole days (many ties)
            df = surv(n=400)
            if round_to:
                df["duration"] = np.ceil(df["duration"])
            m = rm.fit_cox(df, ["x", "z"])
            slow = m.compute_residuals(df[["x", "z", "duration", "event", "company_id"]], kind="scaled_schoenfeld")
            fast = rm.scaled_schoenfeld(m, df, ["x", "z"])
            self.assertEqual(list(fast.columns), list(slow.columns))
            self.assertTrue(fast.index.equals(slow.index))
            np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy(), rtol=1e-9, atol=1e-10)

    def test_aft_models_fit_after_flooring_zero_durations(self):
        df = surv()
        df.loc[0, "duration"] = 0.0
        ready = rm.aft_ready(df)
        self.assertGreater(ready["duration"].min(), 0)
        for kind in ["weibull", "lognormal", "loglogistic"]:
            m = rm.aft_model(kind).fit(ready[["x", "z", "duration", "event"]], "duration", "event")
            self.assertEqual(len(m.predict_median(ready[["x", "z"]])), len(df))

    def test_survival_forest_orders_by_the_signal(self):
        try:
            import sksurv  # noqa: F401
            from sksurv.ensemble import RandomSurvivalForest  # noqa: F401
        except ImportError:
            self.skipTest("scikit-survival not installed")
        df = surv()
        rsf = rm.survival_forest(n_estimators=30, min_samples_leaf=10).fit(df[["x", "z"]], rm.survival_y(df["duration"], df["event"]))
        risk = rsf.predict(df[["x", "z"]])
        self.assertGreater(rm.c_index(df["duration"], df["event"], -risk), 0.6)

    def test_low_memory_survival_forest_predicts_the_same_risk(self):
        try:
            import sksurv  # noqa: F401
        except ImportError:
            self.skipTest("scikit-survival not installed")
        df = surv()
        y = rm.survival_y(df["duration"], df["event"])
        full = rm.survival_forest(n_estimators=20, min_samples_leaf=10, low_memory=False).fit(df[["x", "z"]], y)
        lean = rm.survival_forest(n_estimators=20, min_samples_leaf=10).fit(df[["x", "z"]], y)
        np.testing.assert_allclose(lean.predict(df[["x", "z"]]), full.predict(df[["x", "z"]]), rtol=1e-9)

    def test_survival_y_shape(self):
        y = rm.survival_y([3.0, 5.0], [1, 0])
        self.assertEqual(list(y["event"]), [True, False])
        self.assertEqual(list(y["time"]), [3.0, 5.0])


class RankingTraining(unittest.TestCase):
    def test_groups_are_sorted_blocks_and_training_ranks_the_signal(self):
        rng = np.random.RandomState(4)
        q = np.tile(["p1", "p2", "p3", "p4"], 30)  # interleaved on purpose
        X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
        rel = (X["a"] > 0.5).astype(int) + (X["a"] > 1.2).astype(int)
        order, sizes = rm.ranking_groups(q)
        self.assertEqual(sizes, [30, 30, 30, 30])
        self.assertEqual(list(q[order][:30]), ["p1"] * 30)
        model = rm.train_lambdarank(X, rel, q)
        self.assertGreater(rm.ndcg_at_k(rel, model.predict(X), 10), 0.9)

    def test_queries_over_the_limit_are_split_into_equal_random_parts(self):
        q = np.array(["big"] * 25 + ["small"] * 4)
        split = rm.split_large_queries(q, max_rows=10)
        counts = pd.Series(split).value_counts()
        self.assertEqual(counts["small"], 4)
        self.assertEqual(sorted(counts.drop("small").tolist()), [8, 8, 9])
        self.assertTrue(all(s.startswith("big#") for s in split[:25]))
        self.assertNotEqual(list(split[:9]), [split[0]] * 9)  # parts are random samples, not consecutive rows
        self.assertEqual(list(rm.split_large_queries(q[25:], max_rows=10)), ["small"] * 4)


class Hypotheses(unittest.TestCase):
    def test_backtest_relevance_needs_all_three_and_treats_unknown_as_no(self):
        r = rm.backtest_relevance([1, 1, 1, np.nan, 0], [1, 1, 0, 1, 1], [1, 0, 1, 1, 1])
        self.assertEqual(list(r), [1, 0, 0, 0, 0])

    def test_bootstrap_interval_excludes_zero_for_a_clear_win_and_not_for_noise(self):
        rng = np.random.RandomState(8)
        comp = np.repeat(np.arange(60), 5)
        base = rng.rand(300)
        win = rm.clustered_bootstrap_diff(base + 0.3, base, comp)
        self.assertTrue(win["supported"])
        self.assertAlmostEqual(win["difference"], 0.3)
        noise = rm.clustered_bootstrap_diff(rng.rand(300), rng.rand(300), comp)
        self.assertFalse(noise["supported"])
        self.assertEqual(win["n_clusters"], 60)

    def test_bootstrap_resamples_whole_companies_so_it_is_wide_when_companies_differ(self):
        comp = np.repeat(np.arange(10), 30)
        diff = np.repeat(np.random.RandomState(9).normal(size=10), 30)  # one value per company
        r = rm.clustered_bootstrap_diff(diff, np.zeros(300), comp)
        self.assertGreater(r["ci_high"] - r["ci_low"], 0.5)

    def test_top_k_overlap(self):
        a = np.arange(20, dtype=float)
        self.assertEqual(rm.top_k_overlap(a, a, 10), 1.0)
        self.assertEqual(rm.top_k_overlap(a, -a, 10), 0.0)


class Queue(unittest.TestCase):
    def factors(self):
        rng = np.random.RandomState(3)
        return pd.DataFrame(rng.normal(size=(50, 4)), columns=rm.FACTORS)

    def test_parts_add_up_to_the_score(self):
        b = rm.priority_breakdown(self.factors())
        self.assertTrue(np.allclose(b.filter(like="part_").sum(axis=1), b["score"]))
        self.assertTrue(b["score"].between(0, 100).all())

    def test_missing_factor_value_is_neutral(self):
        f = self.factors()
        f.loc[0, "pay"] = np.nan
        self.assertEqual(rm.priority_breakdown(f).loc[0, "rating_pay"], 50.0)

    def test_tuning_finds_the_informative_factor(self):
        f = self.factors()
        outcome = (f["still_open"] > f["still_open"].quantile(0.8)).astype(int)
        w, p = rm.tune_weights(f, outcome, k=5, grid=(0.0, 1.0))
        self.assertEqual(w["still_open"], 1.0)
        self.assertEqual(p, 1.0)


class ReviewFixes(unittest.TestCase):
    def test_auc_diff_ignores_calibration(self):
        # a model that ranks well but with inflated (balanced-weight) probabilities still beats the constant prior
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        good = np.clip(0.6 + 0.3 * y + rng.normal(0, 0.1, 2000), 0, 1)
        r = rm.clustered_bootstrap_auc_diff(y, good, np.full(2000, y.mean()), np.repeat(np.arange(200), 10), n_boot=200)
        self.assertGreater(r["difference"], 0.3)
        self.assertTrue(r["supported"])
        self.assertEqual(r["n_clusters"], 200)

    def test_bootstraps_equal_the_one_draw_at_a_time_loop(self):
        # the original loops: members per company by np.where, one draw at a time from one generator
        from sklearn.metrics import roc_auc_score
        from lifelines.utils import concordance_index
        rng = np.random.default_rng(3)
        n = 600
        comp = rng.integers(0, 40, n).astype(str)
        y = rng.integers(0, 2, n)
        pm, pb = rng.random(n) + 0.3 * y, rng.random(n)
        dur, ev, risk = rng.exponential(10, n), rng.integers(0, 2, n), rng.random(n)
        codes = pd.factorize(comp)[0]
        members = [np.where(codes == k)[0] for k in range(codes.max() + 1)]
        g = np.random.default_rng(rm.SEED)
        auc_loop, c_loop = [], []
        for _ in range(12):
            idx = np.concatenate([members[k] for k in g.integers(0, len(members), len(members))])
            auc_loop.append(roc_auc_score(y[idx], pm[idx]) - roc_auc_score(y[idx], pb[idx]))
        r = rm.clustered_bootstrap_auc_diff(y, pm, pb, comp, n_boot=12, workers=3)
        self.assertAlmostEqual(r["ci_low"], np.percentile(auc_loop, 2.5), places=12)
        self.assertAlmostEqual(r["ci_high"], np.percentile(auc_loop, 97.5), places=12)
        g = np.random.default_rng(rm.SEED)
        for _ in range(7):
            idx = np.concatenate([members[k] for k in g.integers(0, len(members), len(members))])
            c_loop.append(concordance_index(dur[idx], -risk[idx], ev[idx]))
        for workers in (1, 3):
            c = rm.clustered_bootstrap_c_index(dur, ev, risk, comp, n_boot=7, workers=workers)
            np.testing.assert_allclose(c["boots"], c_loop, rtol=0, atol=1e-15)
            self.assertAlmostEqual(c["c_index"], concordance_index(dur, -risk, ev), places=12)

    def test_shared_preprocessor_is_not_refit_by_other_pipelines(self):
        from sklearn.linear_model import LogisticRegression
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0] * 10, "c": list("xxyz") * 10})
        y = np.array([0, 1] * 20)
        prep = rm.make_preprocessor(["a"], ["c"], min_frequency=1)
        p1 = rm.with_preprocessing(prep, LogisticRegression()).fit(X, y)
        rm.with_preprocessing(prep, LogisticRegression()).fit(X.assign(a=X.a * 100), y)
        self.assertIsNot(p1.named_steps["prep"], prep)
        self.assertAlmostEqual(p1.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_[0], 2.0)


class Backtest(unittest.TestCase):
    ROLES = ["a", "b", "c"]

    def pool(self, n=300, seed=4):
        rng = np.random.RandomState(seed)
        pay = rng.normal(30, 5, n)
        pay[rng.rand(n) < 0.3] = np.nan
        return pd.DataFrame({"role_bucket": rng.choice(self.ROLES, n), "p_still_open": rng.rand(n),
                             "accessibility": rng.randint(0, 5, n), "pay": pay,
                             "first_seen": pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.randint(0, 14 * 24, n), unit="h")})

    def fit(self, people=40, seed=5):
        rng = np.random.RandomState(seed)
        f = pd.DataFrame(rng.dirichlet(np.ones(3), people), columns=self.ROLES)
        return rm.fit_ratings(f)

    def test_fast_top_k_matches_scoring_every_listing(self):
        pool, fr = self.pool(), self.fit()
        ratings = rm.listing_ratings(pool)
        w = {"fit": 0.4, "still_open": 0.3, "accessibility": 0.2, "pay": 0.1}
        top = rm.queue_top_k(fr, ratings, pool["role_bucket"], w, k=10)
        role_col = pool["role_bucket"].map({r: j for j, r in enumerate(self.ROLES)}).to_numpy()
        for i in range(len(fr)):
            factors = ratings.assign(fit=fr.to_numpy()[i, role_col])
            score = sum(w[f] * factors[f].to_numpy() for f in rm.FACTORS)
            self.assertEqual(list(top[i]), list(np.argsort(-score, kind="stable")[:10]))

    def test_fit_rating_puts_the_best_role_at_100(self):
        fr = rm.fit_ratings(pd.DataFrame({"a": [0.6, 0.1], "b": [0.3, 0.2], "c": [0.1, 0.7]}))
        self.assertEqual(list(fr.max(axis=1)), [100.0, 100.0])
        self.assertAlmostEqual(fr.loc[0, "b"], 50.0)

    def test_pay_is_rated_within_its_role_and_missing_is_neutral(self):
        pool = pd.DataFrame({"role_bucket": ["a", "a", "b", "b"], "p_still_open": [0.1, 0.2, 0.3, 0.4],
                             "accessibility": [0, 1, 2, 3], "pay": [10.0, 20.0, 100.0, np.nan]})
        r = rm.listing_ratings(pool)
        self.assertEqual(list(r["pay"]), [50.0, 100.0, 100.0, 50.0])

    def test_accessibility_counts_signs_and_unknown_is_absent(self):
        df = pd.DataFrame({c: [True, None] for c in rm.ACCESS_SIGNS})
        self.assertEqual(list(rm.accessibility_count(df)), [4, 0])

    def test_newest_first_and_role_search(self):
        pool = self.pool()
        top = rm.newest_top_k(pool["first_seen"], 3, k=5)
        self.assertEqual(top.shape, (3, 5))
        self.assertTrue((pool["first_seen"].iloc[top[0]].diff().dropna() <= pd.Timedelta(0)).all())
        searched = rm.newest_top_k(pool["first_seen"], 2, k=5, roles=pool["role_bucket"], search_role="b")
        self.assertTrue((pool["role_bucket"].iloc[searched[0]] == "b").all())

    def test_precision_needs_open_and_the_moved_role(self):
        top = np.array([[0, 1], [0, 1]])
        p = rm.precision_per_person(top, ok=[True, False], roles=["a", "a"], targets=["a", "b"])
        self.assertEqual(list(p), [0.5, 0.0])

    def test_tuning_finds_fit_when_relevance_follows_the_moved_role(self):
        pool, fr = self.pool(), self.fit()
        targets = fr.idxmax(axis=1).to_numpy()  # each person moved into their best-fitting role
        ok = np.ones(len(pool), dtype=bool)
        w, p, table = rm.tune_queue_weights(fr, rm.listing_ratings(pool), pool["role_bucket"], ok, targets, k=10, grid=(0.0, 1.0))
        self.assertEqual(p, 1.0)
        self.assertEqual(w["fit"], 1.0)
        self.assertEqual(len(table), 15)

    def test_overlap(self):
        self.assertEqual(rm.mean_overlap(np.array([[1, 2], [3, 4]]), np.array([[2, 9], [3, 4]])), 0.75)


if __name__ == "__main__":
    unittest.main()
