"""Kaplan-Meier (src/kaplan_meier.py), checked against src/eda_helpers.py's km_curve/surv_at/km_median on made-up
listings. No real data. See docs/kaplan_meier.md for the formula each step implements."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import kaplan_meier as km
from src.eda_helpers import km_curve as old_km_curve
from src.eda_helpers import km_median as old_km_median
from src.eda_helpers import surv_at as old_surv_at


class KaplanMeier(unittest.TestCase):
    def test_small_case_with_ties_matches_eda_helpers(self):
        days = pd.Series([1, 1, 2, 2, 2, 3, 4, 5, 5, 5], dtype="float64")
        finished = pd.Series([True, False, True, True, False, True, False, True, True, False])
        old = old_km_curve(days, finished)
        new = km.km_curve(days, finished)
        self.assertTrue((old.index == new.index).all())
        self.assertTrue(np.allclose(old.to_numpy(), new.to_numpy()))
        # hand-checked: day 1 drops to 9/10 (1 of 10 closed, 1 censored); day 2 to 9/10 * 6/8 (2 of 8 at-risk closed)
        self.assertAlmostEqual(new.loc[1], 0.9)
        self.assertAlmostEqual(new.loc[2], 0.9 * 6 / 8)

    def test_no_events_before_the_first_day_stays_at_one(self):
        curve = km.km_curve(pd.Series([5.0, 10.0]), pd.Series([True, True]))
        self.assertEqual(km.survival_at(curve, 0), 1.0)
        self.assertEqual(km.survival_at(curve, 4), 1.0)

    def test_censored_half_gives_no_median(self):
        # 10 listings, 5 close by day 3, 5 are still open (censored) at day 10: curve never reaches 0.5
        days = pd.Series([1, 2, 2, 3, 3] + [10] * 5, dtype="float64")
        finished = pd.Series([True] * 5 + [False] * 5)
        self.assertTrue(pd.isna(km.median_survival(days, finished)))

    def test_matches_eda_helpers_at_scale_with_censoring(self):
        rng = np.random.default_rng(0)
        n = 200_000
        days = pd.Series(rng.integers(0, 180, size=n).astype("float64"))
        finished = pd.Series(rng.random(n) > 0.25)  # 25% still open (censored)

        old_curve = old_km_curve(days, finished)
        new_curve = km.km_curve(days, finished)
        self.assertTrue((old_curve.index == new_curve.index).all())
        self.assertTrue(np.allclose(old_curve.to_numpy(), new_curve.to_numpy()))

        for x in (7, 30, 90, 150):
            self.assertAlmostEqual(old_surv_at(old_curve, x), km.survival_at(new_curve, x))
        self.assertEqual(old_km_median(days, finished), km.median_survival(days, finished))


if __name__ == "__main__":
    unittest.main()
