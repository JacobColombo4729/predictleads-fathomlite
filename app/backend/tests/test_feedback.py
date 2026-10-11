"""Offline tests for relevance feedback: endpoints, session isolation, and ranking behaviour."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.api import SESSION_COOKIE, Settings, TfIdfRankingBackend, create_app
from backend.feedback_ranking import adjusted_dense_query
from backend.tests.test_api import SNAPSHOT_ID, source_job
import job_matching_engine as matching


def _jobs():
    return [
        source_job("a", "Solutions Engineer", "Run customer demos and technical discovery for software products."),
        source_job("b", "Presales Solutions Engineer", "Customer demos, technical discovery, proof of concept for software."),
        source_job("c", "Forklift Operator", "Move pallets and load trucks in a warehouse."),
        source_job("d", "Data Analyst", "Build SQL reports and dashboards for business teams."),
    ]


class FeedbackAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        sample = root / "jobs.jsonl"
        sample.write_text("".join(json.dumps(job) + "\n" for job in _jobs()), encoding="utf-8")
        settings = Settings(db_path=root / "demo.sqlite3", sample_path=sample, session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        self.client = TestClient(create_app(settings=settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.assertEqual(self.client.post("/api/v1/demo-sessions").status_code, 201)
        created = self.client.post("/api/v1/profiles", json={
            "name": "Test", "headline": "Technical sales moving into AI",
            "experience": [{"id": "e1", "title": "Technical sales specialist", "organization": None,
                            "description": "Ran customer demos and technical discovery."}],
            "skills": [{"id": "s1", "name": "Demos", "evidence": "Product demonstrations for customers."}],
            "desired_roles": ["solutions engineer"], "location": None, "arrangements": [], "constraints": None,
        })
        self.profile_id = created.json()["profile_id"]

    def recommend(self, client=None, limit: int = 4) -> dict:
        response = (client or self.client).post(f"/api/v1/profiles/{self.profile_id}/recommendations",
                                                json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": limit})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_current_snapshot_route_needs_no_session(self) -> None:
        anonymous = TestClient(self.client.app)
        anonymous.__enter__()
        self.addCleanup(anonymous.__exit__, None, None, None)
        body = anonymous.get("/api/v1/snapshots/current").json()
        self.assertEqual(body["snapshot_id"], SNAPSHOT_ID)
        self.assertEqual(body["job_count"], 4)
        self.assertEqual(body["availability"], "unverified")

    def test_current_profile_route_returns_the_session_profile(self) -> None:
        current = self.client.get("/api/v1/profiles/current")
        self.assertEqual(current.status_code, 200, current.text)
        self.assertEqual(current.json()["profile_id"], self.profile_id)
        other = TestClient(self.client.app)
        other.__enter__()
        self.addCleanup(other.__exit__, None, None, None)
        other.post("/api/v1/demo-sessions")
        self.assertEqual(other.get("/api/v1/profiles/current").status_code, 404)

    def test_put_list_overwrite_and_delete_feedback(self) -> None:
        put = self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": SNAPSHOT_ID, "signal": "good"})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(put.json()["signal"], "good")
        self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": SNAPSHOT_ID, "signal": "bad"})
        items = self.client.get("/api/v1/feedback", params={"snapshot_id": SNAPSHOT_ID}).json()["items"]
        self.assertEqual([(i["job_id"], i["signal"]) for i in items], [("a", "bad")])
        self.assertEqual(items[0]["job"]["title"], "Solutions Engineer")
        self.assertEqual(self.client.delete(f"/api/v1/feedback/{SNAPSHOT_ID}/a").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/v1/feedback/{SNAPSHOT_ID}/a").status_code, 404)

    def test_validation_and_unknown_job(self) -> None:
        self.assertEqual(self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": SNAPSHOT_ID, "signal": "meh"}).status_code, 422)
        self.assertEqual(self.client.put("/api/v1/feedback", json={"job_id": "zzz", "snapshot_id": SNAPSHOT_ID, "signal": "good"}).status_code, 404)
        self.assertEqual(self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": "other", "signal": "good"}).status_code, 404)

    def test_feedback_is_private_to_the_session(self) -> None:
        self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": SNAPSHOT_ID, "signal": "good"})
        other = TestClient(self.client.app)
        other.__enter__()
        self.addCleanup(other.__exit__, None, None, None)
        other.post("/api/v1/demo-sessions")
        self.assertEqual(other.get("/api/v1/feedback").json()["items"], [])

    def test_rated_jobs_leave_results_and_response_reports_feedback(self) -> None:
        before = self.recommend()
        self.assertFalse(before["feedback"]["applied"])
        self.client.put("/api/v1/feedback", json={"job_id": "a", "snapshot_id": SNAPSHOT_ID, "signal": "good"})
        self.client.put("/api/v1/feedback", json={"job_id": "c", "snapshot_id": SNAPSHOT_ID, "signal": "bad"})
        after = self.recommend()
        ids = [item["job"]["job_id"] for item in after["items"]]
        self.assertNotIn("a", ids)
        self.assertNotIn("c", ids)
        self.assertEqual((after["feedback"]["good"], after["feedback"]["bad"], after["feedback"]["applied"]), (1, 1, True))
        self.assertIn("does not retrain", after["feedback"]["note"])

    def test_liked_job_pulls_similar_job_up_and_disliked_pushes_down(self) -> None:
        records, _ = matching.load_prepared_jobs(self._write_jobs())
        backend = TfIdfRankingBackend(records)
        profile = matching.CandidateProfile(profile_id="p", experience=("customer work",), evidenced_skills=(),
                                            desired_roles=(), preferences={})
        indices = list(range(len(records)))
        liked_rank = [r["job_id"] for r in backend.rank(profile, 4, indices, liked_job_ids=["a"])]
        self.assertEqual(liked_rank[0], "b")  # the job most like the liked one comes first
        disliked_rank = [r["job_id"] for r in backend.rank(profile, 4, indices, disliked_job_ids=["a"])]
        self.assertNotIn("a", disliked_rank)
        self.assertEqual(disliked_rank[-1], "b")  # the job most like the disliked one sinks

    def _write_jobs(self) -> Path:
        path = Path(self.temp.name) / "rank_jobs.jsonl"
        path.write_text("".join(json.dumps(job) + "\n" for job in _jobs()), encoding="utf-8")
        return path

    def test_dense_query_adjustment_moves_toward_liked_and_stays_unit_length(self) -> None:
        query = np.array([1, 0, 0], dtype=np.float32)
        liked = np.array([[0, 1, 0]], dtype=np.float32)
        disliked = np.array([[0, 0, 1]], dtype=np.float32)
        adjusted = adjusted_dense_query(query, liked, disliked)
        self.assertGreater(adjusted[1], 0)
        self.assertLess(adjusted[2], 0)
        self.assertAlmostEqual(float(np.linalg.norm(adjusted)), 1.0, places=5)
        np.testing.assert_allclose(adjusted_dense_query(query, None, None), query)


if __name__ == "__main__":
    unittest.main()
