"""Search behaviour of the recommendations endpoint: title matches first, full text only as a fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import Settings, create_app, job_level, listed_on, open_on
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
            ("Data Science Intern", None, "internship"),  # internships are their own level, apart from entry
            ("Software Engineering Intern", "junior", "internship"),  # whatever the seniority label says
            ("Summer Internship - Data Analytics", None, "internship"),
            ("Co-op Student, Data Engineering", None, "internship"),
            ("Data Science Graduate Program 2026", None, "program"),  # programs are their own level, apart from entry
            ("Data Engineering Apprentice", "junior", "program"),
            ("IT Trainee", None, "program"),
            ("Leadership Development Program - Analytics", None, "program"),
            ("Data Science Fellowship", None, "program"),
            ("Graduate Data Analyst", None, "entry"),  # "graduate" alone is an entry-level job, not a program
            ("Program Manager, Data", None, "unstated"),  # a program manager is a job
            ("Technical Fellow", "director", "director"),  # a senior title, not a fellowship
            ("Data Contributor - Remote", None, "gig"),  # gig / AI-training data work is its own level too
            ("AI Trainer (Python)", "junior", "gig"),
            ("Senior AI Trainer", "mid_senior", "gig"),
            ("Search Quality Rater", None, "gig"),
            ("Data Annotator", None, "gig"),
            ("Internal Audit Analyst", None, "unstated"),  # "internal" is not "intern"
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
            dict(source_job("intern-0", "Data Analyst Intern", "Build reports.", company_id="c5"), seniority="junior"),
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

        self.assertEqual(ids(["entry"]), {"entry-0", "entry-1"})  # the internship is not entry level
        self.assertEqual(ids(["internship"]), {"intern-0"})
        self.assertEqual(ids(["entry", "experienced"]), {"entry-0", "entry-1", "exp-0"})
        self.assertEqual(len(ids([])), 6)  # no levels chosen keeps everything
        response = client.post(f"/api/v1/profiles/{pid}/recommendations",
                               json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": {"levels": ["intern"]}})
        self.assertEqual(response.status_code, 422)


def dated(job_id: str, posted: str | None, first: str | None, last: str | None, company: str) -> dict:
    row = source_job(job_id, "Data Analyst", "Build reports.", company_id=company)
    row["dates_utc"] = {"posted_at": posted, "first_seen_at": first, "last_seen_at": last, "last_processed_at": last}
    return row


class DateListedTests(unittest.TestCase):
    """The listing date (posted or first seen, whichever is earlier) and the "Open on" filter."""

    def test_listed_on_is_the_earlier_of_posted_and_first_seen(self) -> None:
        cases = [
            ("2026-08-01T09:00:00Z", "2026-08-03T00:00:00Z", ("2026-08-01", "posted")),  # posted before the crawler saw it
            ("2026-08-10T00:00:00Z", "2026-08-03T00:00:00Z", ("2026-08-03", "first_seen")),  # a later posted date is a refresh
            (None, "2026-08-03T00:00:00Z", ("2026-08-03", "first_seen")),
            ("not a date", "2026-08-03T00:00:00Z", ("2026-08-03", "first_seen")),
            (None, None, (None, "unknown")),
            ("1969-12-31T23:00:00Z", "2026-08-03T00:00:00Z", ("2026-08-03", "first_seen")),  # a zero date, not a posting
            ("2024-01-01T00:00:00Z", "2026-08-03T00:00:00Z", ("2026-08-03", "first_seen")),  # more than a year earlier
            ("2026-08-01T00:00:00Z", None, (None, "unknown")),  # a posted date alone can't be checked
        ]
        for posted, first, expected in cases:
            self.assertEqual(listed_on(dated("x", posted, first, None, "c")), expected, (posted, first))

    def test_open_on_needs_listed_by_the_day_and_not_gone_before_it(self) -> None:
        job = dated("x", None, "2026-08-03T00:00:00Z", "2026-08-20T00:00:00Z", "c")
        self.assertFalse(open_on(job, "2026-08-02"))  # not listed yet
        self.assertTrue(open_on(job, "2026-08-03"))
        self.assertTrue(open_on(job, "2026-08-20"))  # last seen that day
        self.assertFalse(open_on(job, "2026-08-21"))
        self.assertTrue(open_on(dated("y", None, "2026-08-03T00:00:00Z", None, "c"), "2026-09-30"))  # never seen gone

    def test_filter_and_range_through_the_api(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        rows = [
            dated("july", "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z", "2026-07-20T00:00:00Z", "c0"),
            dated("august", None, "2026-08-05T00:00:00Z", "2026-08-25T00:00:00Z", "c1"),
            dated("long", "2026-06-15T00:00:00Z", "2026-06-16T00:00:00Z", "2026-09-28T00:00:00Z", "c2"),
        ]
        (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        client = TestClient(create_app(settings=settings))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        client.post("/api/v1/demo-sessions")
        pid = client.post("/api/v1/profiles", json=PROFILE).json()["profile_id"]

        def ids(as_of: str | None) -> set[str]:
            filters = {"as_of": as_of} if as_of else {}
            response = client.post(f"/api/v1/profiles/{pid}/recommendations",
                                   json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": filters})
            self.assertEqual(response.status_code, 200, response.text)
            return {item["job"]["job_id"] for item in response.json()["items"]}

        self.assertEqual(ids(None), {"july", "august", "long"})
        self.assertEqual(ids("2026-07-10"), {"july", "long"})
        self.assertEqual(ids("2026-08-10"), {"august", "long"})
        self.assertEqual(ids("2026-09-29"), set())
        snapshot = client.get("/api/v1/snapshots/current").json()
        self.assertEqual(snapshot["listed_range"], {"first": "2026-06-15", "last": "2026-09-28"})
        first = client.post(f"/api/v1/profiles/{pid}/recommendations",
                            json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "limit": 20, "filters": {}}).json()["items"]
        listed = {i["job"]["job_id"]: (i["job"]["listed_on"], i["job"]["listed_basis"]) for i in first}
        self.assertEqual(listed["july"], ("2026-07-01", "posted"))
        self.assertEqual(listed["august"], ("2026-08-05", "first_seen"))
        browse = client.get("/api/v1/jobs", params={"snapshot_id": SNAPSHOT_ID, "as_of": "2026-07-10"})  # the same filter
        self.assertEqual(browse.status_code, 200, browse.text)
        self.assertEqual({j["job_id"] for j in browse.json()["items"]}, {"july", "long"})


if __name__ == "__main__":
    unittest.main()
