"""The split section of the ML_Fit leakage audit (src/leakage_audit.py) on made-up people. No real data."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import fit_eval as fe
from src import leakage_audit as la


class SplitSection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        people = pd.DataFrame({"target_bucket": rng.choice(fe.ROLES, 400), "pool": rng.choice(["within tech", "retrained"], 400)},
                              index=[f"person-{i}" for i in range(400)])
        dev, test, fold = fe.split_people(people)
        self.meta = people.assign(split=np.where(people.index.isin(test), "test", "development"),
                                  fold=fold.reindex(people.index).fillna(-1).astype(int))
        self.text = "\n".join(la.split_section(self.meta))
        self.n_dev, self.n_test = len(dev), len(test)

    def test_ratio_and_counts(self):
        self.assertIn(f"**{self.n_dev:,} development ({self.n_dev / 400:.1%})**", self.text)
        self.assertIn(f"**{self.n_test:,} test ({self.n_test / 400:.1%})**", self.text)
        self.assertIn("about 4 to 1", self.text)

    def test_every_development_person_is_scored_in_exactly_one_fold(self):
        rows = [l.split("|") for l in self.text.splitlines() if l[:4] in {f"| {f} " for f in range(5)}]
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(int(r[2]) for r in rows), self.n_dev)
        self.assertTrue(all(int(r[2]) + int(r[3]) == self.n_dev for r in rows))

    def test_every_role_and_pool_has_a_row_and_no_person_is_shown(self):
        for name in [*fe.ROLES, "within tech", "retrained"]:
            self.assertIn(f"| {name} |", self.text)
        self.assertNotIn("person-", self.text)


if __name__ == "__main__":
    unittest.main()
