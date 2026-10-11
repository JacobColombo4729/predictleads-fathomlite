"""Thumbs-down reasons: which ones push results away, which only hide, and that company rejections hide the whole company."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.api import PUSH_AWAY_REASONS, Settings, create_app
from backend.tests.test_api import SNAPSHOT_ID, source_job
from backend.tests.test_search import PROFILE


class ReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        text = "Demo the product to prospects and support the sales team."
        rows = [source_job(f"se-{i}", "Sales Engineer", text, company_id=f"c{i}") for i in range(5)]
        rows.append(source_job("acme-1", "Sales Engineer", text, company_id="acme"))
        rows.append(source_job("acme-2", "Solutions Engineer", text, company_id="acme"))
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        self.client = TestClient(create_app(settings=settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")
        self.pid = self.client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]

    def rate(self, job_id: str, signal: str = "bad", **extra):
        return self.client.put("/api/v1/feedback", json={"job_id": job_id, "snapshot_id": SNAPSHOT_ID, "signal": signal, **extra})

    def recommend(self):
        response = self.client.post(f"/api/v1/profiles/{self.pid}/recommendations",
                                    json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": {}})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def ranker_call(self):
        ranker = self.client.app.state.ranker
        return mock.patch.object(type(ranker), "rank", autospec=True, side_effect=type(ranker).rank)

    def test_wrong_role_or_no_reason_pushes_away(self) -> None:
        self.rate("se-0")                     # no reason
        self.rate("se-1", reason="wrong_role")
        self.rate("se-2", reason="other", note="not sure")
        with self.ranker_call() as spy:
            self.recommend()
        kwargs = spy.call_args.kwargs
        self.assertEqual(sorted(kwargs["disliked_job_ids"]), ["se-0", "se-1", "se-2"])
        self.assertNotIn("exclude_job_ids", kwargs)

    def test_other_reasons_hide_the_job_without_pushing(self) -> None:
        for job, reason in (("se-0", "location"), ("se-1", "pay"), ("se-2", "seniority"), ("se-3", "seen")):
            self.rate(job, reason=reason)
        with self.ranker_call() as spy:
            body = self.recommend()
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["disliked_job_ids"], [])
        self.assertEqual(sorted(kwargs["exclude_job_ids"]), ["se-0", "se-1", "se-2", "se-3"])
        shown = {item["job"]["job_id"] for item in body["items"]}
        self.assertFalse(shown & {"se-0", "se-1", "se-2", "se-3"})  # still hidden
        self.assertEqual(body["feedback"]["bad"], 4)
        self.assertEqual(body["feedback"]["bad_pushing"], 0)

    def test_not_this_company_hides_the_companys_other_jobs(self) -> None:
        self.assertIn("acme-1", {i["job"]["job_id"] for i in self.recommend()["items"]})
        self.rate("acme-1", reason="company")
        shown = {i["job"]["job_id"] for i in self.recommend()["items"]}
        self.assertNotIn("acme-1", shown)
        self.assertNotIn("acme-2", shown)
        self.assertIn("se-0", shown)

    def test_a_reason_can_be_added_after_the_thumbs_down(self) -> None:
        self.rate("se-0")
        updated = self.rate("se-0", reason="location", note="  too far  ").json()
        self.assertEqual(updated["reason"], "location")
        listed = self.client.get(f"/api/v1/feedback?snapshot_id={SNAPSHOT_ID}").json()["items"]
        self.assertEqual([(i["job_id"], i["reason"]) for i in listed], [("se-0", "location")])

    def test_validation(self) -> None:
        self.assertEqual(self.rate("se-0", reason="because").status_code, 422)                     # unknown reason
        self.assertEqual(self.rate("se-0", signal="good", reason="pay").status_code, 422)           # reasons only on thumbs-down
        self.assertEqual(self.rate("se-0", note="x" * 201).status_code, 422)                        # note too long
        self.assertEqual(self.rate("se-0", note="   ").status_code, 200)                           # blank note is dropped

    def test_reasons_are_the_documented_set(self) -> None:
        self.assertEqual(PUSH_AWAY_REASONS, frozenset({None, "wrong_role", "other"}))

    def test_old_databases_get_the_new_columns(self) -> None:
        import sqlite3
        from backend.api import SessionStore
        path = Path(self.temp.name) / "old.sqlite3"
        db = sqlite3.connect(path)
        db.executescript("CREATE TABLE job_feedback (session_hash TEXT, snapshot_id TEXT, job_id TEXT, signal TEXT, created_at TEXT, PRIMARY KEY(session_hash, snapshot_id, job_id));")
        db.close()
        store = SessionStore(path, 3600)
        store.initialize()
        db = sqlite3.connect(path)
        columns = {row[1] for row in db.execute("PRAGMA table_info(job_feedback)")}
        db.close()
        self.assertTrue({"reason", "note"} <= columns)


if __name__ == "__main__":
    unittest.main()
