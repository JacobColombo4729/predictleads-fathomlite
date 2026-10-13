"""feature_screen (EDA_SDA_2's screening rule, shared with EDA_SDA_4) on made-up listings. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import feature_screen as fs


def made_up(n=60_000, seed=1):
    r = np.random.default_rng(seed)
    month = r.choice([f"2026-0{m}" for m in range(4, 10)], n)
    cell = r.choice(["a", "b", "c"], n)
    strong = r.random(n) < 0.4                          # raises the 7-day close rate by 12 points
    late = np.isin(month, ["2026-07", "2026-08", "2026-09"])
    flips = r.random(n) < 0.5                           # a strong effect in Apr-Jun, almost none in Jul-Sep
    p = 0.25 + 0.12 * strong + np.where(late, 0.01, 0.14) * flips + np.where(cell == "a", 0.1, 0.0)
    closed7 = r.random(n) < p
    base = pd.DataFrame({"month": month, "source_type": r.choice(["site", "board"], n), "live": ~closed7 & (r.random(n) < 0.5),
                         "days_open": np.where(closed7, 3, 20)})
    values = {"strong": pd.Series(strong, index=base.index), "noise": pd.Series(r.random(n) < 0.5, index=base.index),
              "copy": pd.Series(strong, index=base.index).where(r.random(n) < 0.97, ~strong),   # near-duplicate of strong
              "rare": pd.Series(r.random(n) < 0.02, index=base.index).where(r.random(n) < 0.03),  # mostly blank
              "flips": pd.Series(flips, index=base.index), "late_info": pd.Series(strong, index=base.index)}
    meta = {"strong": ("6 text", "yes"), "noise": ("6 text", "yes"), "copy": ("6 text", "yes"), "rare": ("6 text", "yes"),
            "flips": ("6 text", "yes"), "late_info": ("6 text", "no")}
    return base, values, meta, pd.Series(cell, index=base.index)


class ScreenRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base, values, meta, cell = made_up()
        table, lvls, y = fs.screen(base, values, meta, cell)
        cls.table = fs.redundancy(table, lvls, base.index[:20_000])
        bar = fs.min_info(y.mean())
        sig = cls.table["signal above noise (points)"]
        cls.v = {f: fs.verdict(f, r, sig, bar, hold_text=False) for f, r in cls.table.iterrows()}

    def test_signal_kept_and_noise_dropped(self):
        self.assertEqual(self.v["strong"], "keep")
        self.assertEqual(self.v["noise"], "drop: no signal beyond the confounders")
        self.assertGreater(self.table.loc["strong", "signal above noise (points)"], fs.MIN_SIGNAL)

    def test_rare_late_unstable_and_redundant(self):
        self.assertEqual(self.v["rare"], "drop: too rare (under 5%)")
        self.assertEqual(self.v["late_info"], "drop: not known when the listing is first seen")
        self.assertEqual(self.v["flips"], "keep, watch: unstable across months")
        self.assertTrue(self.v["copy"].startswith("keep one: redundant with"))
        self.assertEqual(fs.to_decision(self.v["copy"]), "dropped")
        self.assertEqual(fs.to_decision("keep, watch: unstable across months"), "kept")

    def test_hold_text_matches_eda_sda_2(self):
        r = self.table.loc["noise"]
        self.assertEqual(fs.verdict("noise", r, self.table["signal above noise (points)"], 0.0), "hold: EDA_SDA_3 validates the text")

    def test_min_info_bar_matches_eda_sda_2_formula(self):
        h = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
        p0 = 0.289
        self.assertAlmostEqual(fs.min_info(p0), 1e4 * (h(p0) - 0.5 * h(p0 + 0.01) - 0.5 * h(p0 - 0.01)) / h(p0))


class RedundantWithAnyStronger(unittest.TestCase):
    """A feature whose single most similar partner is weaker, but which overlaps a stronger one, is still redundant."""

    def test_overlap_with_a_stronger_feature_that_is_not_the_closest(self):
        base, values, meta, cell = made_up()
        r = np.random.default_rng(2)
        strong = values["strong"]
        middle = strong.where(r.random(len(strong)) < 0.9, ~strong)       # V about 0.8 with strong
        twin = middle.where(r.random(len(strong)) < 0.99, ~middle)        # V about 0.98 with middle, weaker than it
        values = {"strong": strong, "middle": middle, "twin": twin}
        meta = {f: ("6 text", "yes") for f in values}
        table, lvls, y = fs.screen(base, values, meta, cell)
        table = fs.redundancy(table, lvls, base.index[:20_000])
        sig = table["signal above noise (points)"]
        self.assertEqual(table.loc["middle", "most similar feature"], "twin")
        self.assertIn("strong", table.loc["middle", f"overlaps (V ≥ {fs.V_REDUNDANT})"])
        self.assertEqual(fs.verdict("middle", table.loc["middle"], sig, 0.0), "keep one: redundant with strong")
        self.assertEqual(fs.verdict("strong", table.loc["strong"], sig, 0.0), "keep")


if __name__ == "__main__":
    unittest.main()
