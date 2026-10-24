"""Plain `Series.map()` on a categorical column, checked against the helper it replaces in EDA_SDA_1_Lean
(src/rows_table.py's cat_map) on made-up labels. No real data.

Why this test exists: `docs/lean_conventions.md` rule 2 says plain pandas already applies a map once per
category, so cat_map/cat_contains can go. That is true about the speed, but `.map()` also *keeps the
categorical dtype* whenever every label maps to a different value - and a categorical can't be multiplied, so
EDA_SDA_1_Lean's 2d-3 (`mid * factor`) failed with "cannot perform the numpy op multiply" on the real
salary_unit column, whose five labels do map one-to-one. The notebook now casts the result, and this test pins
both halves of that: the cast matches cat_map exactly, and the uncast version really does come back
categorical, so the rule is not quietly re-broken later."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src.rows_table import cat_map

PER_YEAR = {"hour": 2080, "day": 260, "week": 52, "month": 12, "year": 1}  # as in EDA_SDA_1_Lean 2d-2


def unit_factor(u):
    """EDA_SDA_1_Lean's 2d-2 mapping: a label such as "per hour" to its number of periods in a year."""
    return next((f for k, f in PER_YEAR.items() if k in u), np.nan) if pd.notna(u) else np.nan


ONE_TO_ONE = ["per hour", "per day", "per week", "per month", "per year"]  # the real salary_unit's shape
MANY_TO_ONE = ["per hour", "per year", "per fortnight", "hourly rate"]     # two map to NaN / the same number


def as_categorical(labels, reps=3):
    return pd.Series(pd.Categorical(labels * reps, categories=labels))


class LeanCatMap(unittest.TestCase):
    def test_cast_map_matches_cat_map(self):
        """The notebook's replacement gives cat_map's exact values, one-to-one mapping or not."""
        for labels in (ONE_TO_ONE, MANY_TO_ONE):
            units = as_categorical(labels)
            lean = units.map(unit_factor).astype("float64")
            self.assertTrue(lean.equals(cat_map(units, unit_factor)), labels)

    def test_missing_labels_stay_missing(self):
        """A missing label, and a label the mapping doesn't recognize, both come back as NaN - not as a number."""
        units = pd.Series(pd.Categorical(["per hour", "per fortnight", None], categories=["per hour", "per fortnight"]))
        lean = units.map(unit_factor).astype("float64")
        self.assertEqual(lean.tolist()[0], 2080)
        self.assertTrue(np.isnan(lean.tolist()[1]))  # unrecognized unit
        self.assertTrue(np.isnan(lean.tolist()[2]))  # no unit given
        self.assertTrue(lean.equals(cat_map(units, unit_factor)))

    def test_uncast_map_keeps_the_categorical_dtype_and_cannot_be_multiplied(self):
        """The bug this guards against: one-to-one mapping -> still categorical -> arithmetic raises."""
        units = as_categorical(ONE_TO_ONE)
        pay = pd.Series([25.0, 200.0, 1200.0, 4000.0, 50000.0] * 3)
        uncast = units.map(unit_factor)
        self.assertIsInstance(uncast.dtype, pd.CategoricalDtype)
        with self.assertRaises(TypeError):
            pay * uncast
        self.assertEqual((pay * uncast.astype("float64")).notna().sum(), len(pay))  # the cast fixes it

    def test_many_to_one_mapping_does_not_hit_the_bug(self):
        """Why the bug hid until the real run: a mapping that isn't one-to-one comes back numeric already."""
        units = as_categorical(MANY_TO_ONE)
        self.assertFalse(isinstance(units.map(unit_factor).dtype, pd.CategoricalDtype))


if __name__ == "__main__":
    unittest.main()
