"""src/differentiators.py on made-up data. No real data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import differentiators as d


class SkillUnlock(unittest.TestCase):
    def test_counts(self):
        df = pd.DataFrame({"skills": [["sql"], ["sql", "python"], ["sql", "aws"], ["aws"], [], ["python", "aws", "sql"]],
                           "hidden": [False, False, False, False, False, True]})
        out = d.skill_unlock_table(df, ["SQL"], hidden_col="hidden").set_index("skill")
        self.assertEqual(out.loc["python", "unlocks"], 1)
        self.assertEqual(out.loc["aws", "unlocks"], 2)   # ["sql","aws"] and ["aws"]
        self.assertEqual(out.loc["aws", "asked_by"], 2)  # the hidden listing is not counted
        self.assertEqual(int(out["reachable_now"].iloc[0]), 2)  # ["sql"] and []
        self.assertEqual(int(out["visible"].iloc[0]), 5)
        self.assertEqual(d.skill_unlock_table(df, ["SQL"], hidden_col="hidden").iloc[0]["skill"], "aws")

    def test_none_skills_ok(self):
        df = pd.DataFrame({"skills": [None, ["x"]]})
        out = d.skill_unlock_table(df, [])
        self.assertEqual(int(out["reachable_now"].iloc[0]), 1)
        self.assertEqual(int(out["unlocks"].iloc[0]), 1)


class Routes(unittest.TestCase):
    def moves(self):
        rows = []
        for p in range(6):  # six people: support > networks > security
            rows += [(p, "it support", "networks", 1), (p, "networks", "security", 2)]
        rows += [(10, "it support", "networks", 1), (11, "networks", "security", 1)]
        rows += [(20, "it support", "networks", 1), (20, "database", "security", 2)]  # not back to back
        return pd.DataFrame(rows, columns=["person", "from_role", "to_role", "step"])

    def test_route(self):
        out = d.stepping_stone_routes(self.moves(), {"security": 40, "networks": 70, "it support": 90}, min_people=5)
        self.assertEqual(list(out["route"]), ["it support > networks > security"])
        r = out.iloc[0]
        self.assertEqual(r["people"], 6)
        self.assertEqual(r["step_people"], 7)  # networks > security: persons 0-5 and 11
        self.assertIn("security: 40", r["openings_at_stops"])
        self.assertEqual(r["last_stop_openings"], 40)

    def test_threshold_and_roles(self):
        self.assertTrue(d.stepping_stone_routes(self.moves(), {}, min_people=7).empty)
        self.assertTrue(d.stepping_stone_routes(self.moves(), {}, min_people=5, roles={"networks", "security"}).empty)


class Backtest(unittest.TestCase):
    def frame(self):
        rows = []
        for g in range(8):
            for i in range(20):
                rows.append({"g": g, "date": pd.Timestamp("2026-09-01") + pd.Timedelta(days=(i * 7 + g) % 20),
                             "score": 1.0 - i / 20, "rel": int(i < 6)})
        return pd.DataFrame(rows)

    def test_model_beats_newest(self):
        r = d.backtest_top_k(self.frame(), "score", "rel", "date", "g", k=10, n_boot=200)
        self.assertAlmostEqual(r["model"], 0.6)
        self.assertLess(r["newest_first"], r["model"])
        self.assertGreater(r["difference"], 0)
        self.assertLessEqual(r["ci_low"], r["difference"] + 1e-9)
        self.assertEqual(r["n_groups"], 8)
        self.assertIn("September", d.claim_sentence(r))

    def test_small_groups_and_split(self):
        with self.assertRaises(ValueError):
            d.backtest_top_k(self.frame(), "score", "rel", "date", "g", k=50)
        a = pd.DataFrame({"date": ["2026-04-03", "2026-07-30", "2026-09-02"]})
        tr, te = d.split_train_test_months(a, "date", ["2026-04", "2026-07"], ["2026-09"])
        self.assertEqual((len(tr), len(te)), (2, 1))
        with self.assertRaises(ValueError):
            d.split_train_test_months(a, "date", ["2026-09"], ["2026-09"])

    def test_precision_at_k(self):
        self.assertEqual(d.precision_at_k([1, 0, 1, 1], 2), 0.5)


if __name__ == "__main__":
    unittest.main()
