"""The Compass skills gap analysis endpoint: one provider request for all picked jobs, cached per session."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import Settings, create_app
from backend.tests.test_api import SNAPSHOT_ID, source_job
from backend.tests.test_search import PROFILE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from grounded_match_explanations import MockProvider  # noqa: E402

DESCRIPTION = "Responsibilities\n- Demo the product to prospects.\n- Own the technical evaluation."


def good(entry: dict, profile_id: str) -> dict:
    return {
        "profile_id": profile_id, "job_id": entry["job_id"], "evidence_sufficiency": "sufficient", "insufficiency_reason": None,
        "transferable_experience": [],
        "qualifications": [{"qualification_id": "q1", "qualification": "Demo the product", "classification": "required", "status": "supported",
                            "candidate_evidence_ids": ["candidate.experience[0].description"], "job_evidence_ids": [entry["evidence"][0]["evidence_id"]],
                            "rationale": "Your demos match."}],
        "next_steps": [],
    }


class GapBatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rows = [source_job(f"j{i}", "Sales Engineer", DESCRIPTION, company_id=f"c{i}") for i in range(3)]
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                                 secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))

    def start(self, provider=None):
        self.client = TestClient(create_app(settings=self.settings, provider=provider))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")
        self.pid = self.client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]
        self.path = f"/api/v1/profiles/{self.pid}/compass/gap-analysis"

    def body(self, ids=("j0", "j1", "j2"), version=1):
        return {"snapshot_id": SNAPSHOT_ID, "profile_version": version, "job_ids": list(ids)}

    def test_off_by_default_and_nothing_is_sent(self) -> None:
        self.start()
        response = self.client.post(self.path, json=self.body())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["assessment_state"], "unavailable")

    def test_one_request_for_all_jobs_then_cached(self) -> None:
        provider = MockProvider(lambda payload: {"jobs": [good(e, self.pid) for e in payload["jobs"]]})
        self.start(provider)
        first = self.client.post(self.path, json=self.body())
        self.assertEqual(first.status_code, 200, first.text)
        data = first.json()
        self.assertEqual(data["assessment_state"], "completed")
        self.assertEqual(set(data["jobs"]), {"j0", "j1", "j2"})
        self.assertEqual(data["failed"], 0)
        self.assertTrue(any(e["evidence_id"] == "candidate.experience[0].description" for e in data["candidate_evidence"]))
        second = self.client.post(self.path, json=self.body())
        self.assertEqual(second.json()["cache"], "hit")
        self.assertEqual(provider.calls, 1)

    def test_a_partly_failed_run_is_not_cached(self) -> None:
        def respond(payload):
            items = [good(e, self.pid) for e in payload["jobs"]]
            items[0]["qualifications"][0]["job_evidence_ids"] = ["job9.nope"]
            return {"jobs": items}
        provider = MockProvider(respond)
        self.start(provider)
        first = self.client.post(self.path, json=self.body()).json()
        self.assertEqual(first["failed"], 1)
        self.client.post(self.path, json=self.body())
        self.assertEqual(provider.calls, 2)

    def test_stale_profile_unknown_job_and_limits(self) -> None:
        self.start(MockProvider(lambda p: {"jobs": []}))
        self.assertEqual(self.client.post(self.path, json=self.body(version=9)).status_code, 409)
        self.assertEqual(self.client.post(self.path, json=self.body(ids=("nope",))).status_code, 404)
        self.assertEqual(self.client.post(self.path, json=self.body(ids=[f"j{i}" for i in range(21)])).status_code, 422)

    def test_requires_a_session(self) -> None:
        self.start()
        self.client.cookies.clear()
        self.assertEqual(self.client.post(self.path, json=self.body()).status_code, 401)


if __name__ == "__main__":
    unittest.main()
