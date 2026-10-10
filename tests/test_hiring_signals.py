"""Tests for the hiring-momentum and buying-intent signals. All data is synthetic; day numbers are plain integers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import hiring_signals as hs  # noqa: E402


def make_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"company_name": None, "domain": None, "title": "Data Engineer", "family": "data", "closed": False, "reposts": 0,
                "seniority": None, "has_salary": False, "tags": ()}
    records = []
    for i, row in enumerate(rows):
        record = {**defaults, "job_id": f"j{i}", **row}
        record.setdefault("company_name", record["company_id"])
        records.append(record)
    frame = pd.DataFrame(records)
    frame["company_code"] = pd.factorize(frame["company_id"])[0]
    return frame


def steady_company(company: str, per_month: int, start: int, end: int, **extra) -> list[dict]:
    """per_month openings spread evenly from day `start` to day `end`; each is observed for 25 days."""
    rows = []
    days = np.linspace(start, end, max(1, int(per_month * (end - start) / 30)), endpoint=False).astype(int)
    for d in days:
        rows.append({"company_id": company, "first_day": int(d), "last_day": int(d) + 25, **extra})
    return rows


class LoadingTests(unittest.TestCase):
    def write(self, records: list[dict]) -> Path:
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        path = Path(directory) / "pool.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    @staticmethod
    def record(job_id, company="c1", title="Data Engineer", description="Build pipelines.", first="2026-08-01", last="2026-08-20", **extra):
        return {"job_id": job_id, "company_reference": {"id": company}, "title_searchable": title, "description_text": description,
                "dates_utc": {"first_seen_at": f"{first}T10:00:00Z", "last_seen_at": f"{last}T10:00:00Z"},
                "status_derived": "unknown", "targeted_pool_reasons": ["data_title"], "tags": ["Python", "Snowflake"],
                "location_raw": "Austin", **extra}

    def test_reposts_become_one_opening_spanning_both(self) -> None:
        path = self.write([self.record("a", first="2026-08-01", last="2026-08-10"), self.record("b", first="2026-08-20", last="2026-09-05")])
        frame = hs.load_listings(path)
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual((row["first_day"], row["last_day"]), (hs.day_number("2026-08-01"), hs.day_number("2026-09-05")))
        self.assertEqual(row["reposts"], 1)

    def test_different_description_or_title_is_a_separate_opening(self) -> None:
        path = self.write([self.record("a"), self.record("b", description="Something else."), self.record("c", title="ML Engineer")])
        self.assertEqual(len(hs.load_listings(path)), 3)

    def test_missing_description_falls_back_to_location(self) -> None:
        a = self.record("a", description=None)
        b = self.record("b", description=None)
        c = self.record("c", description=None, location_raw="Boston")
        self.assertEqual(len(hs.load_listings(self.write([a, b, c]))), 2)

    def test_listing_without_dates_or_company_is_skipped(self) -> None:
        bad_dates = self.record("a")
        bad_dates["dates_utc"] = {"first_seen_at": None, "last_seen_at": None}
        no_company = self.record("b")
        no_company["company_reference"] = None
        self.assertEqual(len(hs.load_listings(self.write([bad_dates, no_company, self.record("c")]))), 1)

    def test_closed_only_if_every_copy_is_closed(self) -> None:
        a = self.record("a", status_derived="closed")
        b = self.record("b")
        self.assertFalse(hs.load_listings(self.write([a, b])).iloc[0]["closed"])

    def test_family_mapping(self) -> None:
        self.assertEqual(hs.primary_family(["data_title", "ai_ml_title"]), "ai_ml")
        self.assertEqual(hs.primary_family(["onet:Sales Engineers"]), "technical_sales")
        self.assertEqual(hs.primary_family(["onet:Data Scientists"]), "data")
        self.assertEqual(hs.primary_family([]), "other")

    def test_day_number_handles_strings_dates_and_junk(self) -> None:
        self.assertEqual(hs.day_number("2026-01-02T05:00:00Z"), 1)
        self.assertEqual(hs.day_number(date(2026, 1, 11)), 10)
        self.assertIsNone(hs.day_number("not a date"))
        self.assertIsNone(hs.day_number(None))
        self.assertEqual(hs.day_to_date(10), date(2026, 1, 11))


class ArtifactTests(unittest.TestCase):
    def test_company_discovered_in_one_batch_is_flagged(self) -> None:
        rows = [{"company_id": "batch", "first_day": 100, "last_day": 130} for _ in range(10)]
        rows += [{"company_id": "organic", "first_day": 100 + i * 3, "last_day": 130 + i * 3} for i in range(10)]
        flagged = hs.flag_artifacts(make_frame(rows))
        self.assertTrue(flagged[flagged["company_id"] == "batch"]["batch_start"].all())
        self.assertFalse(flagged[flagged["company_id"] == "organic"]["batch_start"].any())

    def test_big_steady_company_with_a_busy_day_is_not_flagged(self) -> None:
        rows = [{"company_id": "big", "first_day": 100 + i, "last_day": 140 + i} for i in range(60)]
        rows += [{"company_id": "big", "first_day": 150, "last_day": 190} for _ in range(6)]  # busy day, but 6 of 66 listings
        flagged = hs.flag_artifacts(make_frame(rows))
        self.assertFalse(flagged["batch_start"].any())

    def test_mass_disappearance_is_flagged_only_when_clustered(self) -> None:
        rows = [{"company_id": "dropout", "first_day": 100 + i, "last_day": 150} for i in range(8)]
        rows += [{"company_id": "normal", "first_day": 100 + i, "last_day": 140 + i * 2} for i in range(8)]
        flagged = hs.flag_artifacts(make_frame(rows), upto=200)
        self.assertTrue(flagged[flagged["company_id"] == "dropout"]["mass_end"].all())
        self.assertFalse(flagged[flagged["company_id"] == "normal"]["mass_end"].any())

    def test_flags_use_only_data_known_by_the_cutoff(self) -> None:
        early = [{"company_id": "c", "first_day": 100 + i * 4, "last_day": 130 + i * 4} for i in range(8)]
        later_batch = [{"company_id": "c", "first_day": 300, "last_day": 330} for _ in range(30)]
        before = hs.flag_artifacts(make_frame(early), upto=200)["batch_start"].tolist()
        with_future = hs.flag_artifacts(make_frame(early + later_batch), upto=200)
        self.assertEqual(with_future.iloc[:8]["batch_start"].tolist(), before)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = steady_company("steady", 3, 0, 300)
        rows += steady_company("surger", 3, 0, 270) + [{"company_id": "surger", "first_day": 272 + i, "last_day": 300} for i in range(10)]
        rows += [{"company_id": "newcomer", "first_day": 285 + i, "last_day": 300} for i in range(5)]   # no history before day 285
        rows += steady_company("fading", 4, 0, 270)  # inflow stops
        self.frame = make_frame(rows)
        self.m = hs.company_metrics(self.frame, 300).set_index("company_code")
        self.code = {c: int(self.frame[self.frame.company_id == c].company_code.iloc[0]) for c in ("steady", "surger", "newcomer", "fading")}

    def test_surge_is_flagged_for_the_company_that_accelerated(self) -> None:
        self.assertTrue(self.m.loc[self.code["surger"], "surging"])
        self.assertGreater(self.m.loc[self.code["surger"], "surge_z"], hs.SURGE_Z)

    def test_steady_company_is_not_surging(self) -> None:
        self.assertFalse(self.m.loc[self.code["steady"], "surging"])
        self.assertLess(abs(self.m.loc[self.code["steady"], "surge_z"]), 2.5)

    def test_company_new_to_the_crawl_is_never_called_surging(self) -> None:
        row = self.m.loc[self.code["newcomer"]]
        self.assertFalse(row["enough_history"])
        self.assertFalse(row["surging"])

    def test_company_whose_inflow_stopped_has_negative_momentum(self) -> None:
        row = self.m.loc[self.code["fading"]]
        self.assertEqual(row["new_30"], 0)
        self.assertLess(row["surge_z"], 0)

    def test_active_counts_listings_observed_through_the_cutoff(self) -> None:
        rows = [{"company_id": "c", "first_day": 100, "last_day": 300}, {"company_id": "c", "first_day": 100, "last_day": 299},
                {"company_id": "c", "first_day": 301, "last_day": 320}]
        m = hs.company_metrics(make_frame(rows), 300)
        self.assertEqual(int(m.iloc[0]["active"]), 1)   # the second ended before; the third is not known yet
        self.assertEqual(int(m.iloc[0]["seen"]), 2)

    def test_recent_batch_blocks_the_surge_flag(self) -> None:
        rows = steady_company("c", 3, 0, 270) + [{"company_id": "c", "first_day": 290, "last_day": 300} for _ in range(30)]
        m = hs.company_metrics(make_frame(rows), 300).iloc[0]
        self.assertTrue(m["batch_recent"])
        self.assertFalse(m["surging"])

    def test_family_filter(self) -> None:
        rows = steady_company("c", 3, 0, 270, family="data") + [{"company_id": "c", "first_day": 285 + i, "last_day": 300, "family": "ai_ml"} for i in range(4)]
        frame = make_frame(rows)
        only_ai = hs.company_metrics(frame, 300, family="ai_ml").iloc[0]
        self.assertEqual(int(only_ai["new_30"]), 4)
        self.assertEqual(int(only_ai["active"]), 4)


class NoLookAheadTests(unittest.TestCase):
    def test_metrics_at_a_cutoff_do_not_change_when_the_future_changes(self) -> None:
        base = steady_company("a", 3, 0, 300) + steady_company("b", 5, 0, 300)
        frame = make_frame(base)
        before = hs.company_metrics(frame, 200).sort_values("company_code").reset_index(drop=True)

        future_rows = base + [{"company_id": "a", "first_day": 230 + i, "last_day": 260} for i in range(40)] \
            + [{"company_id": "late", "first_day": 250, "last_day": 280} for _ in range(12)]
        altered = make_frame(future_rows)
        # also change what happens later to a listing that was already known at the cutoff
        mask = (altered["first_day"] <= 200) & (altered["last_day"] > 200)
        altered.loc[mask, "last_day"] = altered.loc[mask, "last_day"] + 90
        after = hs.company_metrics(altered, 200)
        after = after[after["company_code"].isin(before["company_code"])].sort_values("company_code").reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after, check_dtype=False)


class OutputsTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = steady_company("hot", 3, 0, 270, family="ai_ml", tags=("Python", "Internship"), company_name="Hot Co", domain="hot.example")
        rows += [{"company_id": "hot", "first_day": 275 + i, "last_day": 300, "family": "ai_ml", "title": f"ML Engineer {i}",
                  "tags": ("PyTorch", "Python", "Internship"), "company_name": "Hot Co", "domain": "hot.example"} for i in range(8)]
        rows += steady_company("calm", 3, 0, 300, family="data", company_name="Calm Co", tags=("SQL",))
        self.frame = make_frame(rows)

    def test_intent_signal_has_evidence_and_filters_non_technology_tags(self) -> None:
        signals = hs.intent_signals(self.frame, 300)
        self.assertEqual(list(signals["company_name"]), ["Hot Co"])
        row = signals.iloc[0]
        self.assertEqual(row["signal"], "AI/ML buildout")
        self.assertNotIn("Internship", row["top_tech"])
        self.assertIn("PyTorch", row["top_tech"])
        days = [e["first_seen"] for e in row["evidence"]]
        self.assertEqual(days, sorted(days, reverse=True))
        self.assertEqual(len(row["evidence"]), 3)

    def test_technology_leads_count_only_recent_non_batch_openings(self) -> None:
        leads = hs.technology_leads(self.frame, 300)
        pytorch = leads[leads["technology"] == "PyTorch"]
        self.assertEqual(int(pytorch.iloc[0]["new_30"]), 8)
        self.assertNotIn("Internship", set(leads["technology"]))

    def test_weekly_series_counts_match_the_listings(self) -> None:
        weekly = hs.weekly_series(self.frame, 300, weeks=4)
        hot = int(self.frame[self.frame.company_id == "hot"].company_code.iloc[0])
        recent = weekly[(weekly["company_code"] == hot)]
        self.assertEqual(int(recent["new"].sum()), int(((self.frame.company_code == hot) & (self.frame.first_day > 272) & (self.frame.first_day <= 300)).sum()))


class BacktestTests(unittest.TestCase):
    def test_backtest_runs_and_reports_baselines(self) -> None:
        rng = np.random.default_rng(1)
        rows = []
        for i in range(120):
            rate = int(rng.integers(2, 12))
            rows += steady_company(f"c{i}", rate, 0, 330)
        result = hs.backtest(make_frame(rows), [150, 165, 180], 220)
        names = {r["method"] for r in result["results"]}
        self.assertIn("baseline: new openings in the last 30 days", names)
        self.assertIn("model: momentum features (gradient boosting)", names)
        self.assertGreater(result["test_rows"], 50)
        for r in result["results"]:
            self.assertTrue(-1 <= r["spearman"] <= 1)

    def test_overlapping_training_window_is_rejected(self) -> None:
        frame = make_frame(steady_company("a", 3, 0, 300))
        with self.assertRaises(ValueError):
            hs.backtest(frame, [190], 200)


if __name__ == "__main__":
    unittest.main()
