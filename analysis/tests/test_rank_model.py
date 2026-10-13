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

    def test_days_open_only_for_closed(self):
        d = rm.days_open(rm.survival_target(listings()))
        self.assertTrue(d.iloc[2:3].isna().all())
        self.assertAlmostEqual(d.iloc[0], 4)


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


if __name__ == "__main__":
    unittest.main()
