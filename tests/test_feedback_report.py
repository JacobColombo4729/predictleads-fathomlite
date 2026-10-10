"""The feedback report: final ratings, the share of good ratings, and the top-10 share. Offline, synthetic events only."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

import feedback_report as fr  # noqa: E402
from backend.events import EventStore  # noqa: E402

A = "aaaaaaaa-0000-0000-0000-000000000001"
B = "bbbbbbbb-0000-0000-0000-000000000002"


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = EventStore(Path(self.temp.name) / "e.sqlite3")
        self.store.initialize()

    def rate(self, who, job, signal, rank=None):
        payload = {"signal": signal, **({"rank": rank} if rank else {})}
        self.store.log(who, "rating", job_id=job, payload=payload)

    def test_last_rating_wins_and_cleared_ratings_drop_out(self) -> None:
        self.rate(A, "j1", "bad", 1)
        self.rate(A, "j1", "good", 1)
        self.rate(A, "j2", "good", 2)
        self.store.log(A, "rating_cleared", job_id="j2")
        final = fr.final_ratings(self.store.events())
        self.assertEqual([(r["job_id"], r["payload"]["signal"]) for r in final], [("j1", "good")])

    def test_summary_numbers(self) -> None:
        for job, signal, rank in [("a", "good", 1), ("b", "good", 2), ("c", "bad", 3), ("d", "bad", 15)]:
            self.rate(A, job, signal, rank)
        self.rate(B, "e", "good")  # no known rank
        summary = fr.summarize(self.store.events())
        self.assertEqual(summary["participants"], 2)
        self.assertEqual(summary["ratings_final"], 5)
        self.assertAlmostEqual(summary["share_good"], 0.6)
        self.assertEqual(summary["rated_in_top_10"], 3)
        self.assertAlmostEqual(summary["share_good_in_top_10"], 0.667, places=3)

    def test_thumbs_down_reasons_are_counted_from_the_final_rating(self) -> None:
        self.store.log(A, "rating", job_id="a", payload={"signal": "bad", "rank": 1})
        self.store.log(A, "rating", job_id="a", payload={"signal": "bad", "rank": 1, "reason": "location", "note": "too far"})
        self.store.log(A, "rating", job_id="b", payload={"signal": "bad", "rank": 2, "reason": "wrong_role"})
        self.store.log(A, "rating", job_id="c", payload={"signal": "bad", "rank": 3})
        self.store.log(A, "rating", job_id="d", payload={"signal": "good", "rank": 4})
        summary = fr.summarize(self.store.events())
        self.assertEqual(summary["thumbs_down_reasons"], {"location": 1, "wrong_role": 1, "none given": 1})
        self.assertEqual(summary["thumbs_down_notes"], 1)

    def test_empty_log_does_not_crash(self) -> None:
        summary = fr.summarize([])
        self.assertEqual(summary["participants"], 0)
        self.assertIsNone(summary["share_good"])

    def test_export_writes_both_files(self) -> None:
        self.rate(A, "a", "good", 1)
        out = Path(self.temp.name) / "out"
        fr.export(self.store.events(), out)
        self.assertTrue((out / "events.csv").exists() and (out / "ratings.csv").exists())
        self.assertIn("good", (out / "ratings.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
