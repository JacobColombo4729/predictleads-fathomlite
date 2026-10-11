"""The research event log: what was shown and how people reacted. Demo mode logs by default; production mode needs opt-in."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.api import Settings, create_app
from backend.events import EventStore
from backend.tests.test_api import SNAPSHOT_ID, source_job
from backend.tests.test_search import PROFILE

PID = "11111111-2222-3333-4444-555555555555"
OTHER = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class EventLogTests(unittest.TestCase):
    def start(self, require_consent: bool = False) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rows = [source_job(f"se-{i}", "Sales Engineer", "Demo the product to prospects and support the sales team.", company_id=f"c{i}") for i in range(6)]
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.events_path = root / "events.sqlite3"
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600, secure_cookie=False,
                            allowed_origins=("http://127.0.0.1:5173",), events_db_path=self.events_path, events_require_consent=require_consent)
        self.client = TestClient(create_app(settings=settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")
        self.pid = self.client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]
        self.store = EventStore(self.events_path, require_consent)

    def as_person(self, person: str | None):
        return {"X-Participant-Id": person} if person else {}

    def recommend(self, person: str | None = PID):
        response = self.client.post(f"/api/v1/profiles/{self.pid}/recommendations", headers=self.as_person(person),
                                    json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 5, "filters": {}})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_nothing_is_logged_without_a_participant_id(self) -> None:
        self.start()
        self.recommend(None)
        self.assertEqual(self.store.events(), [])

    def test_invalid_ids_are_ignored(self) -> None:
        self.start()
        self.recommend("short")
        self.recommend("has spaces and is long enough to pass length ok")
        self.assertEqual(self.store.events(), [])

    def test_demo_mode_logs_what_was_shown_with_ranks(self) -> None:
        self.start()
        shown = self.recommend()
        impressions = self.store.events("impression")
        self.assertEqual(len(impressions), 1)
        items = impressions[0]["payload"]["items"]
        self.assertEqual([i[0] for i in items], [r["job"]["job_id"] for r in shown["items"]])
        self.assertEqual([i[1] for i in items], list(range(1, len(items) + 1)))
        self.assertEqual(impressions[0]["participant_id"], PID)

    def test_a_rating_remembers_where_the_job_was_shown(self) -> None:
        self.start()
        shown = self.recommend()
        target = shown["items"][2]["job"]["job_id"]
        headers = self.as_person(PID)
        self.client.put("/api/v1/feedback", headers=headers, json={"job_id": target, "snapshot_id": SNAPSHOT_ID, "signal": "good"})
        rating = self.store.events("rating")[0]
        self.assertEqual(rating["job_id"], target)
        self.assertEqual(rating["payload"]["signal"], "good")
        self.assertEqual(rating["payload"]["rank"], 3)
        self.client.delete(f"/api/v1/feedback/{SNAPSHOT_ID}/{target}", headers=headers)
        self.assertEqual(len(self.store.events("rating_cleared")), 1)

    def test_saves_and_unsaves_are_logged(self) -> None:
        self.start()
        headers = self.as_person(PID)
        self.client.post("/api/v1/saved-jobs", headers=headers, json={"job_id": "se-1", "snapshot_id": SNAPSHOT_ID})
        self.client.delete(f"/api/v1/saved-jobs/{SNAPSHOT_ID}/se-1", headers=headers)
        self.assertEqual([e["type"] for e in self.store.events()], ["save", "unsave"])

    def test_no_text_from_the_profile_is_stored(self) -> None:
        self.start()
        self.recommend()
        dump = json.dumps(self.store.events())
        for fragment in ("Account Manager", "Managed customer accounts", "Product demos", "Test"):
            self.assertNotIn(fragment, dump)

    def test_people_are_kept_apart(self) -> None:
        self.start()
        self.recommend(PID)
        self.recommend(OTHER)
        self.assertEqual(sorted({e["participant_id"] for e in self.store.events()}), sorted([PID, OTHER]))
        self.assertEqual(self.store.count(PID), 1)

    def test_a_logging_failure_never_breaks_the_request(self) -> None:
        self.start()
        with mock.patch.object(EventStore, "log", side_effect=RuntimeError("disk full")):
            self.recommend()
            response = self.client.put("/api/v1/feedback", headers=self.as_person(PID),
                                       json={"job_id": "se-1", "snapshot_id": SNAPSHOT_ID, "signal": "good"})
            self.assertEqual(response.status_code, 200)

    def test_production_mode_logs_nothing_until_the_person_opts_in(self) -> None:
        self.start(require_consent=True)
        self.recommend()
        self.assertEqual(self.store.events(), [])
        status = self.client.get("/api/v1/research/status", headers=self.as_person(PID)).json()
        self.assertEqual((status["consented"], status["consent_required"]), (False, True))
        self.assertEqual(self.client.put("/api/v1/research/consent", headers=self.as_person(PID), json={"consent": True}).json()["consented"], True)
        self.recommend()
        self.assertEqual(len(self.store.events("impression")), 1)

    def test_withdrawing_deletes_everything_held_for_the_person(self) -> None:
        self.start(require_consent=True)
        self.client.put("/api/v1/research/consent", headers=self.as_person(PID), json={"consent": True})
        self.client.put("/api/v1/research/consent", headers=self.as_person(OTHER), json={"consent": True})
        self.recommend(PID)
        self.recommend(OTHER)
        result = self.client.put("/api/v1/research/consent", headers=self.as_person(PID), json={"consent": False}).json()
        self.assertEqual(result["deleted"], 1)
        self.assertEqual({e["participant_id"] for e in self.store.events()}, {OTHER})
        self.recommend(PID)  # no longer logged
        self.assertEqual(self.store.count(PID), 0)

    def test_delete_endpoint_removes_data_in_demo_mode_too(self) -> None:
        self.start()
        self.recommend()
        self.assertEqual(self.client.delete("/api/v1/research/data", headers=self.as_person(PID)).json()["deleted"], 1)
        self.assertEqual(self.store.events(), [])

    def test_switched_off_when_no_events_database_is_configured(self) -> None:
        self.start()
        self.client.app.state.events = None
        self.recommend()
        self.assertEqual(self.client.get("/api/v1/research/status", headers=self.as_person(PID)).json()["available"], False)


if __name__ == "__main__":
    unittest.main()
