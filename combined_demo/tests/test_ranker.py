"""The combined demo's ranker adapter (combined_demo/ranker.py): made-up model, base ranker and session database.

    python -m unittest discover -s combined_demo/tests -t .
"""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import combined._paths  # noqa: F401  puts analysis/ on the path
from combined_demo.ranker import FACTORS, CombinedRankingBackend, sqlite_profiles

ENTRIES = [{"id": "e1", "kind": "job", "title": "Store Lead", "organization": None, "description": "Mar 2016 - Present. Ran the store."}]


class FakeBase:
    job_ids = ["a", "b", "c"]

    def __init__(self):
        self.calls = []

    def rank(self, profile, limit, indices, **kwargs):
        self.calls.append((limit, list(indices), kwargs))
        return [{"job_id": self.job_ids[i], "rank": r, "score": 1.0 / r} for r, i in enumerate(indices[:limit], 1)]

    def most_similar_liked(self, job_ids, liked):
        return {"from": "base"}


class FakeModel:
    def __init__(self):
        self.seen = None

    def recommend(self, entries, rank, limit, indices):
        self.seen = (entries, indices, rank(5, indices))
        rows = [{"job_id": "b", "role": "it support", "score": 80.0}, {"job_id": "a", "role": "data science", "score": 60.0}]
        for row in rows:
            for f in FACTORS:
                row[f"rating: {f}"], row[f"part: {f}"] = 70.0, 17.5
        return SimpleNamespace(queue=pd.DataFrame(rows))


def profile(pid="prf_1", experience=("Store Lead Ran the store.",)):
    return SimpleNamespace(profile_id=pid, experience=experience)


def session_db(directory, profiles):
    path = Path(directory) / "sessions.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE profiles (profile_id TEXT PRIMARY KEY, session_hash TEXT, version INTEGER, profile_json TEXT, updated_at TEXT)")
        for pid, data in profiles.items():
            db.execute("INSERT INTO profiles VALUES (?, 'h', 1, ?, 'now')", (pid, json.dumps(data)))
        db.commit()
    return path


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = session_db(self.tmp.name, {"prf_1": {"name": "X", "experience": ENTRIES}})
        self.base, self.model = FakeBase(), FakeModel()
        self.ranker = CombinedRankingBackend(self.base, self.model, sqlite_profiles(self.db))

    def test_results_in_the_apps_format_with_a_breakdown(self):
        out = self.ranker.rank(profile(), 10, [0, 1, 2])
        self.assertEqual([r["job_id"] for r in out], ["b", "a"])
        self.assertEqual([r["rank"] for r in out], [1, 2])
        self.assertEqual(out[0]["role"], "it support")
        self.assertEqual(set(out[0]["breakdown"]), set(FACTORS))
        self.assertEqual(out[0]["breakdown"]["fit"], {"rating": 70.0, "part": 17.5})
        self.assertEqual(self.ranker.ranker_id, "combined")

    def test_saved_entries_reach_the_model(self):
        self.ranker.rank(profile(), 10, [0, 2])
        entries, indices, _ = self.model.seen
        self.assertEqual(entries, ENTRIES)
        self.assertEqual(indices, [0, 2])

    def test_feedback_reaches_the_text_match_and_exploration_is_dropped(self):
        self.ranker.rank(profile(), 10, [0, 1], liked_job_ids=["a"], disliked_job_ids=["c"], explore=3, rng=object())
        _, _, kwargs = self.base.calls[0]
        self.assertEqual(kwargs, {"liked_job_ids": ["a"], "disliked_job_ids": ["c"]})

    def test_an_unknown_profile_falls_back_to_its_summary(self):
        with self.assertLogs("predictleads.combined", level="WARNING"):
            self.ranker.rank(profile("prf_missing"), 10, [0])
        entries, _, _ = self.model.seen
        self.assertEqual(entries, [{"kind": "job", "title": "Store Lead Ran the store.", "description": ""}])

    def test_everything_else_comes_from_the_base_ranker(self):
        self.assertEqual(self.ranker.most_similar_liked(["a"], ["b"]), {"from": "base"})
        self.assertEqual(self.ranker.job_ids, ["a", "b", "c"])

    def test_profile_lookup_is_read_only_and_tolerates_no_database(self):
        self.assertIsNone(sqlite_profiles(Path(self.tmp.name) / "none.sqlite3")("prf_1"))
        lookup = sqlite_profiles(self.db)
        self.assertIsNone(lookup("nobody"))
        with self.assertRaises(sqlite3.OperationalError):  # mode=ro: the adapter can't write
            with closing(sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)) as db:
                db.execute("DELETE FROM profiles")


if __name__ == "__main__":
    unittest.main()
