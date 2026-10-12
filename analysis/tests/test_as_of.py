"""Date picker (src/as_of.py) on made-up listings. No real data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import as_of as ao


def table():
    rows = [("a1", "A", "2026-06-01", "2026-06-20", "x"), ("a2", "A", "2026-06-10", "2026-09-30", "x"),
            ("a3", "A", "2026-06-10 12:00", "2026-07-05", "y"), ("a4", "A", "2026-07-01 08:00", "2026-07-02", "x"),
            ("b1", "B", "2026-06-25", "2026-10-10", "z"), ("b2", "B", "2026-08-15", "2026-09-10", "z"),
            ("c1", "C", "2026-09-05", "2026-09-06", None)]
    df = pd.DataFrame(rows, columns=["id", "company_id", "first_seen", "last_seen", "desc_hash"])
    df[["first_seen", "last_seen"]] = df[["first_seen", "last_seen"]].apply(lambda s: pd.to_datetime(s, format="ISO8601"))
    return df


def sort(df):
    return df.sort_values("id").reset_index(drop=True)


class AsOfTests(unittest.TestCase):
    def test_live_rule(self):
        self.assertEqual(set(ao.live_on(table(), "2026-07-01")["id"]), {"a2", "a3", "a4", "b1"})
        self.assertEqual(set(ao.live_on(table(), "2026-09-01")["id"]), {"a2", "b1", "b2"})

    def test_last_seen_that_day_is_live_next_day_is_not(self):
        self.assertIn("a1", set(ao.live_on(table(), "2026-06-20")["id"]))
        self.assertNotIn("a1", set(ao.live_on(table(), "2026-06-21")["id"]))

    def test_features_count_only_earlier_listings(self):
        live = ao.live_on(table(), "2026-07-01").set_index("id")
        self.assertEqual(live.loc["a2", "company_prior_listings"], 1)  # a1
        self.assertEqual(live.loc["a4", "company_prior_listings"], 3)
        self.assertEqual(live.loc["a4", "repost_prior_copies"], 2)  # a1 and a2 share its text
        self.assertEqual(live.loc["b1", "company_prior_listings"], 0)
        self.assertEqual(live.loc["b1", "company_listings_to_date"], 1)  # b2 is after the date
        self.assertEqual(live.loc["a2", "age_days"], 21)

    def test_listings_after_the_date_change_nothing(self):
        """Dropping everything first seen after the day, or changing its last_seen, leaves the output identical."""
        full = table()
        cut = full[full["first_seen"] < pd.Timestamp("2026-07-02")]
        pd.testing.assert_frame_equal(sort(ao.live_on(full, "2026-07-01")), sort(ao.live_on(cut, "2026-07-01")))
        edited = full.copy()
        edited.loc[edited["first_seen"] >= pd.Timestamp("2026-07-02"), "last_seen"] += pd.Timedelta(days=400)
        pd.testing.assert_frame_equal(sort(ao.training_as_of(full, "2026-07-01")),
                                      sort(ao.training_as_of(edited, "2026-07-01")))

    def test_last_seen_is_not_a_feature(self):
        """Moving when a live listing was last seen (after the day) changes no feature."""
        full = table()
        later = full.copy()
        later.loc[later["id"] == "a2", "last_seen"] = pd.Timestamp("2027-01-01")
        a = sort(ao.live_on(full, "2026-07-01"))
        pd.testing.assert_frame_equal(a, sort(ao.live_on(later, "2026-07-01")))
        self.assertNotIn("last_seen", a.columns)

    def test_training_rows_are_censored_at_the_date(self):
        t = ao.training_as_of(table(), "2026-07-01").set_index("id")
        self.assertEqual(set(t.index), {"a1", "a2", "a3", "a4", "b1"})
        self.assertTrue(t.loc["a1", "finished"])
        self.assertFalse(t.loc["a2", "finished"])  # open that day: censored
        self.assertEqual(t.loc["a2", "days_observed"], 22)  # stops at the end of the day, not 2026-09-30
        self.assertEqual(t.loc["a1", "days_observed"], 19)

    def test_count_check_is_counts_only(self):
        c = ao.count_check(table())
        self.assertEqual(list(c["live"]), [4, 3])
        self.assertTrue(all(pd.api.types.is_numeric_dtype(c[k]) for k in c.columns))

    def test_timezone_aware_data(self):
        df = table()
        df[["first_seen", "last_seen"]] = df[["first_seen", "last_seen"]].apply(lambda s: s.dt.tz_localize("UTC"))
        self.assertEqual(len(ao.live_on(df, "2026-09-01")), 3)


if __name__ == "__main__":
    unittest.main()
