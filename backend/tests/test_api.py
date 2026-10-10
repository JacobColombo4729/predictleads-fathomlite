from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from backend.api import SESSION_COOKIE, Settings, create_app
import grounded_match_explanations as grounded


SNAPSHOT_ID = "predictleads-2026-09-30"


def source_job(
    job_id: str,
    title: str,
    description: str | None,
    *,
    location: str | None = "Austin, TX",
    category: str = "operations",
    status: str | None = None,
    company_id: str = "company-1",
    company_name: str | None = "Example Components",
) -> dict[str, Any]:
    present_description = bool(description and description.strip())
    return {
        "job_id": job_id,
        "job_type": "job_opening",
        "title_raw": title,
        "title_normalized": title,
        "title_translated": None,
        "title_searchable": title,
        "title_searchable_source": "normalized_title",
        "onet_data": {"code": "13-1023.00", "family": "Business and Financial Operations", "occupation_name": "Purchasing Agents"},
        "company_reference": {"id": company_id, "type": "company"},
        "company_match_state": "matched",
        "company_name_included": company_name,
        "company_domain_included": "example.test",
        "location_raw": location,
        "location_data_raw": [],
        "salary_raw": None,
        "salary_data_raw": {"salary_low": None, "salary_high": None, "salary_currency": None, "salary_time_unit": None},
        "description_raw": description,
        "description_text": description,
        "description_usable": present_description,
        "url": f"https://jobs.example.test/{job_id}",
        "status_raw": status,
        "status_derived": status or "unknown",
        "historical_snapshot": True,
        "snapshot_date": "2026-09-30",
        "source_key": "2026/09/30/job_openings_2026_09_30.jsonl.zst",
        "source_line": 100 + ord(job_id[-1]) if len(job_id) == 1 else 100,
        "dates_raw": {"first_seen_at": "2026-09-01T00:00:00Z", "last_seen_at": "2026-09-30T00:00:00Z", "last_processed_at": "2026-09-30T00:01:00Z", "posted_at": None},
        "dates_utc": {"first_seen_at": "2026-09-01T00:00:00Z", "last_seen_at": "2026-09-30T00:00:00Z", "last_processed_at": "2026-09-30T00:01:00Z", "posted_at": None},
        "quality_flags": {
            "title": "present",
            "description": "present" if present_description else "null",
            "location": "present" if location is not None else "null",
            "salary": "null",
            "status": "present" if status is not None else "null",
            "posted_at": "null",
        },
        "categories": [category],
        "tags": ["SQL"] if "sql" in (description or "").casefold() else [],
        "seniority": "junior",
        "contract_types": [],
    }


class GoodProvider:
    model = "offline-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        candidate = payload["candidate"]
        job = payload["job"]
        candidate_ids = {item["evidence_id"] for item in candidate["evidence"]}
        job_ids = {item["evidence_id"] for item in job["evidence"]}
        candidate_skill = next(value for value in candidate_ids if value.endswith("skills[0].name"))
        job_requirement = next(value for value in job_ids if value.endswith("description_text[1]"))
        candidate_experience = next(value for value in candidate_ids if value.endswith("experience[0].description"))
        job_duty = next(value for value in job_ids if value.endswith("description_text[0]"))
        return {
            "profile_id": candidate["profile_id"],
            "job_id": job["job_id"],
            "evidence_sufficiency": "sufficient",
            "insufficiency_reason": None,
            "transferable_experience": [{
                "job_duty": "Investigate customer reporting issues",
                "candidate_evidence_ids": [candidate_experience],
                "job_evidence_ids": [job_duty],
                "rationale": "The profile and listing both support troubleshooting work.",
            }],
            "qualifications": [{
                "qualification_id": "q-sql",
                "qualification": "SQL reporting",
                "classification": "required",
                "status": "supported",
                "candidate_evidence_ids": [candidate_skill],
                "job_evidence_ids": [job_requirement],
                "rationale": "The candidate profile explicitly lists SQL.",
            }],
            "next_steps": [],
        }


class FailingProvider:
    model = "offline-failing-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        raise grounded.ProviderFailure("redacted provider test failure")


class BackendAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample_path = self.root / "jobs.jsonl"
        jobs = [
            source_job("a", "Support Reporting Specialist", "Investigate customer reporting issues.\nSQL reporting experience required.", location="Austin, TX", category="support"),
            source_job("b", "Warehouse Operations Coordinator", "Coordinate shipments and reconcile purchase orders.", location="Dallas, TX", category="operations", status="closed", company_id="company-2", company_name=None),
            source_job("c", "Purchasing Assistant", None, location="Austin, TX", category="operations"),
        ]
        self.sample_path.write_text("".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
        self.settings = Settings(
            db_path=self.root / "demo.sqlite3",
            sample_path=self.sample_path,
            session_ttl_seconds=3600,
            secure_cookie=False,
            allowed_origins=("http://127.0.0.1:5173",),
        )
        self.provider = GoodProvider()
        self.client = TestClient(create_app(settings=self.settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.create_session(self.client)

    @staticmethod
    def create_session(client: TestClient) -> str:
        response = client.post("/api/v1/demo-sessions")
        if response.status_code != 201:
            raise AssertionError(response.text)
        return client.cookies.get(SESSION_COOKIE)

    def create_profile(
        self,
        client: TestClient | None = None,
        *,
        skills: list[dict[str, str]] | None = None,
        include_experience: bool = True,
    ) -> dict[str, Any]:
        client = client or self.client
        response = client.post("/api/v1/profiles", json={
            "name": "Test Candidate",
            "headline": "Support specialist",
            "experience": [{"id": "exp-1", "title": "Support Specialist", "organization": None, "description": "Troubleshoots customer reporting issues."}] if include_experience else [],
            "skills": skills if skills is not None else [{"id": "skill-1", "name": "SQL", "evidence": "Writes account-reporting queries."}],
            "desired_roles": ["Support Reporting Specialist"],
            "location": "Austin, TX",
            "arrangements": ["Remote"],
            "constraints": None,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_profile_create_read_update_and_version_conflict(self) -> None:
        created = self.create_profile()
        profile_id = created["profile_id"]
        self.assertEqual(created["version"], 1)
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_id}").json(), created)

        updated = self.client.patch(
            f"/api/v1/profiles/{profile_id}",
            headers={"If-Match": '"1"'},
            json={"headline": "Updated headline"},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["version"], 2)
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_id}").json()["profile"]["headline"], "Updated headline")
        conflict = self.client.patch(f"/api/v1/profiles/{profile_id}", headers={"If-Match": '"1"'}, json={"headline": "Stale"})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "profile_version_conflict")
        self.assertTrue(conflict.json()["error"]["request_id"].startswith("req_"))
        invalid = self.client.patch(f"/api/v1/profiles/{profile_id}", headers={"If-Match": '"2"'}, json={"desired_roles": [""]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "validation_error")

    def test_session_isolation_and_reset(self) -> None:
        profile_a = self.create_profile()
        cookie_a = self.client.cookies.get(SESSION_COOKIE)
        self.create_session(self.client)
        profile_b = self.create_profile(self.client)
        cookie_b = self.client.cookies.get(SESSION_COOKIE)

        self.client.cookies.set(SESSION_COOKIE, cookie_a, domain="testserver.local", path="/")
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_a['profile_id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_b['profile_id']}").status_code, 404)
        reset = self.client.delete("/api/v1/demo-sessions/current")
        self.assertEqual(reset.status_code, 204)
        self.assertIsNone(self.client.cookies.get(SESSION_COOKIE))
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_a['profile_id']}").status_code, 401)

        self.client.cookies.set(SESSION_COOKIE, cookie_b, domain="testserver.local", path="/")
        self.assertEqual(self.client.get(f"/api/v1/profiles/{profile_b['profile_id']}").status_code, 200)

    def test_expired_session_is_rejected_and_data_is_removed(self) -> None:
        profile = self.create_profile()
        session_hash = self.client.cookies.get(SESSION_COOKIE)
        import hashlib
        hashed = hashlib.sha256(session_hash.encode()).hexdigest()
        with self.client.app.state.store.connect() as db:
            db.execute("UPDATE demo_sessions SET expires_at='2000-01-01T00:00:00Z' WHERE session_hash=?", (hashed,))
        response = self.client.get(f"/api/v1/profiles/{profile['profile_id']}")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "demo_session_expired")
        with self.client.app.state.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM profiles").fetchone()[0], 0)

    def test_search_filters_pagination_and_cursor_binding(self) -> None:
        page1 = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "q": "reporting", "limit": 1})
        self.assertEqual(page1.status_code, 200, page1.text)
        self.assertEqual(page1.json()["items"][0]["job_id"], "a")
        cursor = page1.json()["page"]["next_cursor"]
        self.assertIsNone(cursor)

        location = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "location": "Austin", "description": "without"})
        self.assertEqual([item["job_id"] for item in location.json()["items"]], ["c"])
        category = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "category": "support"})
        self.assertEqual([item["job_id"] for item in category.json()["items"]], ["a"])
        unsupported = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "work_arrangement": "Remote"})
        self.assertEqual(unsupported.status_code, 400)

        all_page1 = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "limit": 1}).json()
        self.assertIsNotNone(all_page1["page"]["next_cursor"])
        next_page = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "limit": 1, "cursor": all_page1["page"]["next_cursor"]})
        self.assertEqual(next_page.status_code, 200)
        self.assertNotEqual(next_page.json()["items"][0]["job_id"], all_page1["items"][0]["job_id"])
        invalid = self.client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "q": "different", "limit": 1, "cursor": all_page1["page"]["next_cursor"]})
        self.assertEqual(invalid.status_code, 400)

    def test_snapshot_identity_details_company_and_availability(self) -> None:
        job = self.client.get("/api/v1/jobs/a", params={"snapshot_id": SNAPSHOT_ID})
        self.assertEqual(job.status_code, 200)
        body = job.json()
        self.assertEqual(body["snapshot"]["snapshot_id"], SNAPSHOT_ID)
        self.assertEqual(body["availability"]["state"], "unverified")
        self.assertTrue(body["description_available"])
        self.assertIn("source_url", body)
        self.assertIn("salary_raw", body)
        self.assertEqual(body["field_states"]["salary_raw"], "null")
        wrong_snapshot = self.client.get("/api/v1/jobs/a", params={"snapshot_id": "predictleads-2026-08-30"})
        self.assertEqual(wrong_snapshot.status_code, 404)

        company = self.client.get("/api/v1/companies/company-2", params={"snapshot_id": SNAPSHOT_ID})
        self.assertEqual(company.status_code, 200)
        self.assertIsNone(company.json()["name"])
        self.assertFalse(company.json()["full_company_record_loaded"])
        self.assertIsNone(company.json()["description"])

    def test_recommendations_use_actual_tfidf_adapter_metadata(self) -> None:
        profile = self.create_profile()
        result = self.client.post(
            f"/api/v1/profiles/{profile['profile_id']}/recommendations",
            json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 2, "filters": {"query": "reporting"}},
        )
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual(body["ranking"]["ranker_id"], "tfidf_cosine")
        self.assertEqual(body["ranking"]["version"], "job_matching_engine.rank_tfidf-v1")
        self.assertEqual(body["ranking"]["score_type"], "cosine_similarity")
        self.assertEqual(body["ranking"]["calibration"], "uncalibrated")
        self.assertEqual(body["snapshot"]["snapshot_id"], SNAPSHOT_ID)
        self.assertEqual(body["items"][0]["job"]["job_id"], "a")
        self.assertIn("company_summary", body["items"][0]["job"])
        self.assertIn("description_excerpt", body["items"][0]["job"])
        bad_snapshot = self.client.post(f"/api/v1/profiles/{profile['profile_id']}/recommendations", json={"snapshot_id": "other", "profile_version": 1})
        self.assertEqual(bad_snapshot.status_code, 404)

    def test_saved_job_idempotency_snapshot_identity_and_persistence(self) -> None:
        create = {"job_id": "a", "snapshot_id": SNAPSHOT_ID}
        first = self.client.post("/api/v1/saved-jobs", json=create)
        second = self.client.post("/api/v1/saved-jobs", json=create)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.json()["saved_at"], first.json()["saved_at"])
        self.assertEqual(second.json()["stage"], "Saved")

        patch = self.client.patch(f"/api/v1/saved-jobs/{SNAPSHOT_ID}/a", json={"stage": "Applied", "note": "Submitted application"})
        self.assertEqual(patch.status_code, 200)
        self.assertEqual(patch.json()["stage"], "Applied")
        listed = self.client.get("/api/v1/saved-jobs").json()["items"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["snapshot_id"], SNAPSHOT_ID)
        self.assertEqual(listed[0]["note"], "Submitted application")
        persisted_cookie = self.client.cookies.get(SESSION_COOKIE)
        with TestClient(create_app(settings=self.settings)) as restarted:
            restarted.cookies.set(SESSION_COOKIE, persisted_cookie)
            persisted = restarted.get("/api/v1/saved-jobs")
            self.assertEqual(persisted.status_code, 200)
            self.assertEqual(persisted.json()["items"][0]["note"], "Submitted application")
        deleted = self.client.delete(f"/api/v1/saved-jobs/{SNAPSHOT_ID}/a")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/api/v1/saved-jobs").json()["items"], [])

        session_hash = hashlib.sha256(self.client.cookies.get(SESSION_COOKIE).encode()).hexdigest()
        self.client.app.state.store.save_job(session_hash, "predictleads-2026-08-30", "retained-job")
        retained = self.client.patch(
            "/api/v1/saved-jobs/predictleads-2026-08-30/retained-job",
            json={"stage": "Considering", "note": "Keep my historical note"},
        )
        self.assertEqual(retained.status_code, 200, retained.text)
        self.assertEqual(retained.json()["snapshot_id"], "predictleads-2026-08-30")
        self.assertEqual(retained.json()["note"], "Keep my historical note")

    def test_assessment_unavailable_insufficient_cached_success_and_provider_error(self) -> None:
        profile = self.create_profile()
        request_body = {"snapshot_id": SNAPSHOT_ID, "profile_version": 1}
        path = f"/api/v1/profiles/{profile['profile_id']}/jobs/a/assessment"

        unavailable = self.client.post(path, json=request_body)
        self.assertEqual(unavailable.status_code, 200, unavailable.text)
        self.assertEqual(unavailable.json()["assessment_state"], "unavailable")
        self.assertIsNone(unavailable.json()["assessment"])
        self.assertEqual(self.provider.calls, 0)
        self.assertEqual(self.client.post(path, json=request_body).json()["cache"], "miss")

        profile_b = self.client.post("/api/v1/demo-sessions")
        self.assertEqual(profile_b.status_code, 201)
        no_evidence = self.create_profile(self.client, skills=[], include_experience=False)
        insufficient = self.client.post(f"/api/v1/profiles/{no_evidence['profile_id']}/jobs/a/assessment", json=request_body)
        self.assertEqual(insufficient.status_code, 200)
        self.assertEqual(insufficient.json()["assessment_state"], "completed")
        self.assertEqual(insufficient.json()["assessment"]["evidence_sufficiency"], "insufficient")
        self.assertEqual(self.provider.calls, 0)
        self.assertEqual(self.client.post(f"/api/v1/profiles/{no_evidence['profile_id']}/jobs/a/assessment", json=request_body).json()["cache"], "hit")

    def test_injected_assessment_provider_cache_and_provider_error_not_cached(self) -> None:
        profile = self.create_profile()
        request_body = {"snapshot_id": SNAPSHOT_ID, "profile_version": 1}
        path = f"/api/v1/profiles/{profile['profile_id']}/jobs/a/assessment"
        self.client.app.state.assessment_provider = self.provider
        first = self.client.post(path, json=request_body)
        second = self.client.post(path, json=request_body)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["assessment_state"], "completed")
        self.assertEqual(first.json()["assessment"]["evidence_sufficiency"], "sufficient")
        qualification = first.json()["assessment"]["qualifications"][0]
        self.assertEqual(qualification["status"], "supported")
        catalog = {item["evidence_id"]: item["text"] for item in first.json()["assessment"]["evidence_catalog"]}
        self.assertEqual(catalog[qualification["candidate_evidence_ids"][0]], "SQL")
        self.assertEqual(second.json()["cache"], "hit")
        self.assertEqual(self.provider.calls, 1)

        failure = FailingProvider()
        self.client.app.state.assessment_provider = failure
        path_b = f"/api/v1/profiles/{profile['profile_id']}/jobs/b/assessment"
        failed1 = self.client.post(path_b, json=request_body)
        failed2 = self.client.post(path_b, json=request_body)
        self.assertEqual(failed1.status_code, 502)
        self.assertEqual(failed2.status_code, 502)
        self.assertEqual(failure.calls, 2)
        self.assertNotIn("redacted provider test failure", failed1.text)
        session_hash = hashlib.sha256(self.client.cookies.get(SESSION_COOKIE).encode()).hexdigest()
        with self.client.app.state.store.connect() as db:
            cached = db.execute("SELECT count(*) FROM assessment_cache WHERE session_hash=?", (session_hash,)).fetchone()[0]
        self.assertEqual(cached, 1)


if __name__ == "__main__":
    unittest.main()