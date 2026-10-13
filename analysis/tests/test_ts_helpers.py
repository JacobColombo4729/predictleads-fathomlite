"""TimeSeries helpers (src/ts_helpers.py) on made-up listings. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import ts_helpers as ts
from src.eda_helpers import km_curve


class TimeSeriesHelpers(unittest.TestCase):
    def test_posted_reads_central_european_date(self):
        # Midnight CET (winter) and midnight CEST (summer) are 23:00 and 22:00 UTC the day before
        posted = pd.Series(pd.to_datetime(["2026-01-14 23:00", "2026-07-14 22:00"], utc=True))
        self.assertEqual(list(ts.calendar_date(posted, ts.CET)), [pd.Timestamp("2026-01-15"), pd.Timestamp("2026-07-15")])
        self.assertEqual(ts.calendar_date(posted).iloc[0], pd.Timestamp("2026-01-14"))  # the UTC reading is a day early

    def test_daily_counts_fill_zero_and_weekly_flags_partial_weeks(self):
        dates = pd.Series(pd.to_datetime(["2026-04-01"] * 3 + ["2026-04-03", "2026-04-20"]))
        daily = ts.daily_counts(dates, "2026-04-01", "2026-04-28")  # Wed Apr 1 to Tue Apr 28
        self.assertEqual(len(daily), 28)
        self.assertEqual(daily.sum(), 5)
        self.assertEqual(daily["2026-04-02"], 0)
        weekly = ts.weekly_counts(daily)
        self.assertEqual(weekly["listings"].sum(), 5)  # nothing dropped
        self.assertEqual(list(weekly.index[:2]), [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-04-06")])
        self.assertFalse(weekly.loc["2026-03-30", "full_week"])  # starts Apr 1: Mar 30 and 31 are outside the data
        self.assertTrue(weekly.loc["2026-04-06", "full_week"])
        self.assertTrue(weekly.loc["2026-04-13", "full_week"])
        self.assertFalse(weekly.loc["2026-04-27", "full_week"])  # ends Tue Apr 28: partial last week

    def test_features_use_only_the_past(self):
        idx = pd.date_range("2026-04-01", periods=60)
        y = pd.Series(np.arange(60, dtype=float), index=idx, name="n")
        X, target = ts.lag_features(y)
        self.assertEqual(X.index[0], idx[14])  # a 14-day lag needs 14 days of history
        row = X.loc[idx[30]]
        self.assertEqual(row["lag_1"], 29)  # yesterday, not today
        self.assertEqual(row["roll_mean_7"], np.mean(np.arange(23, 30)))  # days 23..29, today excluded
        y2 = y.copy()
        y2.loc[idx[30]] = 1e6  # changing today's value changes nothing in today's features
        X2, _ = ts.lag_features(y2)
        pd.testing.assert_series_equal(X.loc[idx[30]], X2.loc[idx[30]])
        self.assertEqual(target.loc[idx[30]], 30)

    def test_split_is_in_time_order_and_disjoint(self):
        idx = pd.date_range("2026-04-01", "2026-09-30")
        tr, va, te = ts.time_split(idx, "2026-08-01", "2026-09-01")
        self.assertEqual((tr & va).sum() + (va & te).sum() + (tr & te).sum(), 0)
        self.assertEqual(tr.sum() + va.sum() + te.sum(), len(idx))
        self.assertLess(idx[tr].max(), idx[va].min())
        self.assertLess(idx[va].max(), idx[te].min())

    def test_scores(self):
        self.assertAlmostEqual(ts.rmse([1, 2, 3], [1, 2, 5]), np.sqrt(4 / 3))
        self.assertAlmostEqual(ts.mae([1, 2, 3], [1, 2, 5]), 2 / 3)

    def test_catch_rate_behaves_like_a_cadence(self):
        rng = np.random.default_rng(0)
        days = rng.exponential(20, 5000).round()
        curve = km_curve(days, np.ones(len(days), bool))
        rates = [ts.catch_rate(curve, n) for n in (1, 3, 7, 14, 30)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))  # checking less often catches fewer
        self.assertGreater(rates[0], 0.9)
        self.assertLess(ts.catch_rate(curve, 7, days_left=7), ts.catch_rate(curve, 7))  # needing time to apply costs
        longest = ts.longest_cadence(curve, 0.9)
        self.assertGreaterEqual(ts.catch_rate(curve, longest), 0.9)
        self.assertLess(ts.catch_rate(curve, longest + 1), 0.9)
        fixed = km_curve(np.full(100, 4.0), np.ones(100, bool))  # always closes after exactly 4 days
        self.assertGreater(ts.catch_rate(fixed, 4), 0.99)
        self.assertLess(ts.catch_rate(fixed, 8), 0.75)


if __name__ == "__main__":
    unittest.main()
