"""Exploration: listings the keyword and embedding signals disagree about (plus one random) are mixed into results."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import job_matching_engine as matching  # noqa: E402
from backend.api import Settings, create_app  # noqa: E402
from backend.events import EventStore  # noqa: E402
from backend.rrf_ranker import RrfRankingBackend  # noqa: E402
from backend.tests.test_api import SNAPSHOT_ID, source_job  # noqa: E402
from backend.tests.test_search import PROFILE  # noqa: E402

PID = "11111111-2222-3333-4444-555555555555"
QUERY_VECTOR = np.array([0, 1, 0], dtype=np.float32)


def build(root: Path):
    """24 listings. K*: strong keyword match to the profile, embedding far from the query. E*: the reverse. N*: neither."""
    rows, vectors = [], []
    keyword_text = "Run product demos for customer accounts and manage customer accounts with product demos."
    for i in range(8):
        rows.append(source_job(f"K{i}", "Account Manager", keyword_text + f" Note {i}.", company_id=f"ck{i}"))
        vectors.append([1.0, 0.05 * i, 0.0])
    for i in range(8):
        rows.append(source_job(f"E{i}", "Quantum Welder", f"Fuse titanium panels in vacuum chambers, shift {i}.", company_id=f"ce{i}"))
        vectors.append([0.0, 1.0 - 0.02 * i, 0.1])
    for i in range(8):
        rows.append(source_job(f"N{i}", "Forklift Operator", f"Move pallets around the warehouse, shift {i}.", company_id=f"cn{i}"))
        vectors.append([0.1, 0.0, 1.0])
    path = root / "jobs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    records, _ = matching.load_prepared_jobs(path)
    matrix = np.array(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(root / "bge_embeddings.npy", matrix)
    ids_hash = hashlib.sha256("\0".join(j.job_id for j in records).encode()).hexdigest()
    (root / "bge_embeddings_manifest.json").write_text(json.dumps({"ordered_job_ids_sha256": ids_hash}))
    return path, list(records)


class RankerExplorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.records = build(self.root)
        self.backend = RrfRankingBackend(self.records, self.root, encoder=lambda text: QUERY_VECTOR)
        self.profile = matching.CandidateProfile(profile_id="p", experience=("customer accounts product demos",), evidenced_skills=(),
                                                 desired_roles=(), preferences={})
        self.indices = list(range(len(self.records)))

    def rank(self, explore: int, **kwargs):
        return self.backend.rank(self.profile, 6, self.indices, explore=explore, rng=random.Random(1), **kwargs)

    def test_no_exploration_by_default(self) -> None:
        ranked = self.backend.rank(self.profile, 6, self.indices)
        self.assertEqual(len(ranked), 6)
        self.assertFalse(any("exploration" in r for r in ranked))

    def test_extras_follow_the_ranked_list_and_are_not_in_it(self) -> None:
        out = self.rank(3)
        ranked, extras = [r for r in out if "exploration" not in r], [r for r in out if "exploration" in r]
        self.assertEqual(len(ranked), 6)
        self.assertEqual(len(extras), 3)
        self.assertFalse({r["job_id"] for r in ranked} & {r["job_id"] for r in extras})
        self.assertTrue(all(r["rank"] is None for r in extras))

    def test_disagreement_listings_come_from_both_sides_plus_one_random(self) -> None:
        extras = {r["exploration"]: r["job_id"] for r in self.rank(3) if "exploration" in r}
        self.assertEqual(set(extras), {"keyword", "embedding", "random"})
        self.assertTrue(extras["keyword"].startswith("K"))     # strong on keywords, weak on meaning
        self.assertTrue(extras["embedding"].startswith("E"))   # strong on meaning, weak on keywords

    def test_two_slots_are_one_of_each_side_and_no_random(self) -> None:
        sources = sorted(r["exploration"] for r in self.rank(2) if "exploration" in r)
        self.assertEqual(sources, ["embedding", "keyword"])

    def test_a_single_slot_is_a_disagreement_listing(self) -> None:
        sources = [r["exploration"] for r in self.rank(1) if "exploration" in r]
        self.assertEqual(len(sources), 1)
        self.assertIn(sources[0], {"keyword", "embedding"})

    def test_rated_and_excluded_listings_are_never_explored(self) -> None:
        blocked = [f"K{i}" for i in range(8)] + [f"E{i}" for i in range(8)]
        out = self.rank(3, exclude_job_ids=blocked)
        self.assertFalse({r["job_id"] for r in out} & set(blocked))


class ApiExplorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        path, records = build(root)
        backend = RrfRankingBackend(records, root, encoder=lambda text: QUERY_VECTOR)
        self.events_path = root / "events.sqlite3"
        settings = Settings(db_path=root / "d.sqlite3", sample_path=path, session_ttl_seconds=3600, secure_cookie=False,
                            allowed_origins=("http://127.0.0.1:5173",), events_db_path=self.events_path)
        self.client = TestClient(create_app(settings=settings, ranker=backend))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")
        self.pid = self.client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]

    def recommend(self, limit: int = 10, explore: bool = True):
        r = self.client.post(f"/api/v1/profiles/{self.pid}/recommendations", headers={"X-Participant-Id": PID},
                             json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": limit, "filters": {}, "explore": explore})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_off_unless_asked_and_off_for_small_pages(self) -> None:
        self.assertFalse(any("exploration" in i for i in self.recommend(10, explore=False)["items"]))
        small = self.recommend(6)
        self.assertFalse(any("exploration" in i for i in small["items"]))
        self.assertEqual(small["ranking"]["exploration"]["slots"], 0)

    def test_page_keeps_its_size_and_exploration_is_never_first(self) -> None:
        for _ in range(12):  # positions are random: check many draws
            body = self.recommend(10)
            items = body["items"]
            self.assertEqual(len(items), 10)
            self.assertEqual([i["rank"] for i in items], list(range(1, 11)))
            self.assertNotIn("exploration", items[0])
            marked = [i for i in items if "exploration" in i]
            self.assertEqual(len(marked), 2)  # round(10 * 0.15) = 2
            self.assertEqual(body["ranking"]["exploration"]["slots"], 2)

    def test_the_log_remembers_which_listing_was_exploration_and_ratings_carry_it(self) -> None:
        items = self.recommend(10)["items"]
        marked = next(i for i in items if "exploration" in i)
        self.client.put("/api/v1/feedback", headers={"X-Participant-Id": PID},
                        json={"job_id": marked["job"]["job_id"], "snapshot_id": SNAPSHOT_ID, "signal": "good"})
        store = EventStore(self.events_path)
        impression = store.events("impression")[0]["payload"]
        flagged = [row for row in impression["items"] if row[3]]
        self.assertEqual(len(flagged), 2)
        rating = store.events("rating")[0]["payload"]
        self.assertEqual(rating["source"], marked["exploration"]["source"])
        self.assertEqual(rating["rank"], marked["rank"])


if __name__ == "__main__":
    unittest.main()
