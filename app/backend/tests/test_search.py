"""Search behaviour of the recommendations endpoint: title matches first, full text only as a fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import Settings, create_app, job_level
from backend.tests.test_api import SNAPSHOT_ID, source_job

PROFILE = {
    "name": "Test", "headline": "", "desired_roles": [], "location": None, "arrangements": [], "constraints": None,
    "experience": [{"id": "e1", "kind": "job", "title": "Account Manager", "organization": None, "description": "Managed customer accounts and ran product demos."}],
    "skills": [{"id": "s1", "name": "Product demos", "evidence": "Ran weekly demos."}],
}


class SearchScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rows = []
        for i in range(6):  # six real "Sales Engineer" titles
            rows.append(source_job(f"se-{i}", "Sales Engineer" if i % 2 else "Senior Sales Engineer",
                                   "Demo the product to prospects and support the sales team.", company_id=f"c-se-{i}"))
        for i in range(2):  # titles that do not match, but whose text mentions the phrase
            rows.append(source_job(f"rep-{i}", "Account Representative",
                                   "Partner closely with sales engineers on demos for customer accounts.", company_id=f"c-rep-{i}"))
        for i in range(3):
            rows.append(source_job(f"ops-{i}", "Warehouse Coordinator", "Coordinate shipments and purchase orders.", company_id=f"c-ops-{i}"))
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        self.client = TestClient(create_app(settings=settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/v1/demo-sessions")
        self.pid = self.client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]

    def recommend(self, query: str | None):
        filters = {} if query is None else {"query": query}
        response = self.client.post(f"/api/v1/profiles/{self.pid}/recommendations",
                                    json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": filters})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_title_matches_come_first_and_exclude_description_only_mentions(self) -> None:
        body = self.recommend("sales engineer")
        self.assertEqual(body["search"], {"query": "sales engineer", "scope": "title"})
        titles = {item["job"]["title"] for item in body["items"]}
        self.assertEqual(len(body["items"]), 6)
        self.assertTrue(all("sales engineer" in t.lower() for t in titles), titles)
        self.assertNotIn("Account Representative", titles)

    def test_word_order_does_not_matter(self) -> None:
        body = self.recommend("engineer sales")
        self.assertEqual(body["search"]["scope"], "title")
        self.assertEqual(len(body["items"]), 6)

    def test_few_title_matches_falls_back_to_the_whole_listing(self) -> None:
        body = self.recommend("account representative")
        self.assertEqual(body["search"]["scope"], "anywhere")
        self.assertTrue(body["items"])

    def test_text_only_mention_is_found_by_the_fallback(self) -> None:
        body = self.recommend("shipments purchase")
        self.assertEqual(body["search"]["scope"], "anywhere")
        self.assertEqual({item["job"]["title"] for item in body["items"]}, {"Warehouse Coordinator"})

    def test_no_query_reports_scope_all(self) -> None:
        body = self.recommend(None)
        self.assertEqual(body["search"], {"query": "", "scope": "all"})
        self.assertEqual(len(body["items"]), 11)

    def test_blank_query_is_treated_as_no_query(self) -> None:
        self.assertEqual(self.recommend("   ")["search"]["scope"], "all")


class JobLevelTests(unittest.TestCase):
    """job_level reads PredictLeads' seniority plus the title; the levels filter keeps only the chosen levels."""

    def test_levels_from_seniority_and_title(self) -> None:
        cases = [
            ("Data Scientist", "junior", "entry"),
            ("Senior Data Scientist", "junior", "senior"),  # a senior title overrides a junior label
            ("Data Scientist", "mid_senior", "experienced"),
            ("Lead Data Scientist", "mid_senior", "senior"),
            ("Analytics Manager", "manager", "manager"),
            ("Head of Data", "head", "director"),
            ("VP, Data", "vice_president", "director"),
            ("Junior Data Analyst", None, "entry"),  # no stated seniority: the title decides
            ("Data Science Intern", None, "entry"),
            ("Data Analyst", None, "unstated"),
            ("Staffing Coordinator", None, "unstated"),  # "staff" only counts as a whole word
        ]
        for title, seniority, expected in cases:
            source = dict(source_job("x", title, "Text."), seniority=seniority)
            self.assertEqual(job_level(source), expected, (title, seniority))

    def test_levels_filter_keeps_only_chosen_levels(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        rows = [
            dict(source_job("entry-0", "Data Scientist", "Build models.", company_id="c0"), seniority="junior"),
            dict(source_job("entry-1", "Junior Data Analyst", "Build reports.", company_id="c1"), seniority=None),
            dict(source_job("senior-0", "Senior Data Analyst", "Lead analysis.", company_id="c2"), seniority="mid_senior"),
            dict(source_job("exp-0", "Data Scientist", "Build models.", company_id="c3"), seniority="mid_senior"),
            dict(source_job("mgr-0", "Analytics Manager", "Manage a team.", company_id="c4"), seniority="manager"),
        ]
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        client = TestClient(create_app(settings=settings))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        client.post("/api/v1/demo-sessions")
        pid = client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]

        def ids(levels: list[str]) -> set[str]:
            response = client.post(f"/api/v1/profiles/{pid}/recommendations",
                                   json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": {"levels": levels}})
            self.assertEqual(response.status_code, 200, response.text)
            return {item["job"]["job_id"] for item in response.json()["items"]}

        self.assertEqual(ids(["entry"]), {"entry-0", "entry-1"})
        self.assertEqual(ids(["entry", "experienced"]), {"entry-0", "entry-1", "exp-0"})
        self.assertEqual(len(ids([])), 5)  # no levels chosen keeps everything
        response = client.post(f"/api/v1/profiles/{pid}/recommendations",
                               json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": {"levels": ["intern"]}})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
