"""Offline tests for the Compass analysis. A deterministic bag-of-words embedder stands in for BGE."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from backend import compass
from backend.api import Settings, create_app
from backend.tests.test_api import SESSION_COOKIE, SNAPSHOT_ID, source_job
import job_matching_engine as matching

DIM = 256
_STOP = {"the", "a", "an", "and", "of", "to", "in", "for", "with", "on", "at", "or", "is", "are", "be", "by", "as"}


def embed(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for token in re.findall(r"[a-z0-9+#]+", text.lower()):
        if token not in _STOP:
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


FAMILIES = {
    "sales": (
        ["Solutions Engineer", "Pre-Sales Consultant", "Sales Engineer", "Technical Account Manager",
         "Implementation Consultant", "Customer Success Engineer", "Demo Specialist", "Solutions Consultant"],
        "Requirements\nExperience delivering technical product demos to prospects.\n"
        "Ability to translate customer needs into technical requirements for engineers.\n"
        "At least 5 years of experience delivering technical demos to prospects.\n"
        "Responsibilities\nPartner with account executives on discovery calls.\nBenefits\nFree snacks and unlimited vacation.",
    ),
    "data": (
        ["Data Analyst", "BI Analyst", "Reporting Analyst", "Analytics Engineer", "Business Analyst",
         "Insights Analyst", "Dashboard Developer", "Operations Analyst"],
        "Requirements\nProficiency in SQL and data querying for reporting.\n"
        "Experience building dashboards with BI tools like Tableau.\n"
        "Responsibilities\nAnalyze business data and present insights to stakeholders.\nBenefits\nFree snacks and unlimited vacation.",
    ),
    "ml": (
        ["ML Engineer", "Machine Learning Scientist", "Applied Scientist", "AI Engineer", "NLP Engineer",
         "Deep Learning Engineer", "Data Scientist", "Research Engineer"],
        "Requirements\nExperience training machine learning models in Python.\n"
        "Knowledge of statistics and model evaluation methods.\n"
        "Responsibilities\nDeploy models to production pipelines.\nBenefits\nFree snacks and unlimited vacation.",
    ),
}


def build_jobs(directory: Path) -> list[matching.JobRecord]:
    rows = []
    for family, (titles, description) in FAMILIES.items():
        for number, title in enumerate(titles):
            job = source_job(f"{family}-{number}", title, description + f"\nTeam note {family} {number}.", company_id=f"co-{family}-{number}")
            job["duplicate_key"] = f"dup-{family}-{number}"
            job["onet_data"] = {"code": "00-0000.00", "family": "Test", "occupation_name": {"sales": "Sales Engineers", "data": "Data Analysts", "ml": "Data Scientists"}[family]}
            rows.append(job)
    path = directory / "jobs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    jobs, _ = matching.load_prepared_jobs(path)
    return list(jobs)


class FakeBackend:
    """Implements the CompassBackend protocol with the bag-of-words embedder (and what the API adapter needs)."""

    ranker_id = "fake_embedding"
    version = "fake-v1"
    score_type = "cosine"
    supports_compass = True

    def __init__(self, jobs: Sequence[matching.JobRecord]):
        self.jobs = list(jobs)
        self.index = {job.job_id: i for i, job in enumerate(self.jobs)}
        self.matrix = np.vstack([embed(f"{job.source['title_searchable']} {job.source['description_text']}") for job in self.jobs])
        self.encode_calls = 0

    def job_vectors(self, job_ids):
        return self.matrix[[self.index[j] for j in job_ids]]

    def query_vector(self, profile):
        return embed(" ".join((*profile.experience, *profile.evidenced_skills)))

    def encode_documents(self, texts):
        self.encode_calls += 1
        return np.vstack([embed(t) for t in texts]) if len(texts) else np.zeros((0, DIM), dtype=np.float32)

    def score_all(self, vector):
        return self.matrix @ np.asarray(vector, dtype=np.float32)

    def rank(self, profile, limit, candidate_indices, **_kwargs):  # not used by these tests
        return []


PROFILE = {
    "name": "Test Candidate", "headline": "Inside sales moving to pre-sales",
    "experience": [
        {"id": "e1", "kind": "job", "title": "Inside Sales Representative", "organization": "Acme",
         "description": "Ran weekly product demos for prospects. Translated customer needs into technical requirements for engineers."},
        {"id": "e2", "kind": "education", "title": "Data Analytics Certificate", "organization": "State College", "description": "Capstone on support-ticket trends."},
    ],
    "skills": [{"id": "s1", "name": "SQL", "evidence": "Wrote queries for account reporting."}],
    "desired_roles": ["solutions engineer"], "location": None, "arrangements": [], "constraints": None,
}

# the fake embedder gives lower cosines than BGE, so the heuristics are scaled to it for the test
FAKE_THRESHOLDS = dict(COVERAGE_LOW=0.15, COVERAGE_HIGH=0.55, WHY_MIN_SIMILARITY=0.3, CLUSTER_DISTANCE=0.6, BUILDS_MIN_SIMILARITY=0.35)


def candidate_from(profile_data: dict[str, Any]) -> matching.CandidateProfile:
    from backend.api import _candidate_from_profile
    return _candidate_from_profile("p", profile_data)


class CompassAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.jobs = build_jobs(Path(self.temp.name))
        self.backend = FakeBackend(self.jobs)
        self.by_id = {j.job_id: j for j in self.jobs}
        patcher = mock.patch.multiple(compass, **FAKE_THRESHOLDS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_analysis(self, pick_ids, profile=PROFILE, excluded=()):
        return compass.analyze(
            self.backend, picks=[self.by_id[i] for i in pick_ids], all_jobs=self.jobs,
            profile=candidate_from(profile), profile_data=profile, excluded_job_ids=excluded,
        )

    PICKS = ["sales-0", "sales-1", "sales-2", "data-0", "data-1", "data-2"]

    def test_shape_and_ranges(self) -> None:
        result = self.run_analysis(self.PICKS)
        self.assertEqual(result["picks_analyzed"], 6)
        self.assertGreaterEqual(len(result["themes"]), 1)
        self.assertLessEqual(len(result["themes"]), compass.MAX_THEMES)
        self.assertEqual(sum(t["count"] for t in result["themes"]), 6)
        self.assertEqual(result["strongest_theme_id"], result["themes"][0]["id"])
        theme_ids = {t["id"] for t in result["themes"]}
        for item in result["jobs"]:
            self.assertTrue(0 <= item["fit"] <= 100)
            self.assertIn(item["theme_id"], theme_ids)
            half = next(t["half_width"] for t in result["themes"] if t["id"] == item["theme_id"])
            self.assertLessEqual(abs(item["offset"]), half)
        self.assertEqual(sum(1 for j in result["jobs"] if j["pick"]), 6)

    def test_requirement_groups_count_jobs_and_ground_evidence_in_the_profile(self) -> None:
        result = self.run_analysis(self.PICKS)
        unit_texts = {u.text for u in compass.profile_units(PROFILE)}
        demos = next(r for r in result["requirements"] if "demos" in r["text"].lower())
        self.assertEqual(demos["n"], 3)
        self.assertIn(demos["status"], {"supported", "partial"})
        self.assertIn(demos["evidence"]["text"], unit_texts)  # evidence is a verbatim profile line
        dashboards = next(r for r in result["requirements"] if "dashboards" in r["text"].lower())
        self.assertEqual(dashboards["status"], "unknown")
        self.assertIsNone(dashboards["evidence"])
        for req in result["requirements"]:
            self.assertTrue(set(req["job_ids"]) <= set(self.PICKS))
            if req["evidence"]:
                self.assertIn(req["evidence"]["text"], unit_texts)

    def test_verify_pattern_matches_years_degrees_and_credentials_only(self) -> None:
        for text in ("5+ years in a technical sales role", "At least 3 years of experience", "Bachelor's degree in Business",
                     "MBA preferred", "Relevant certification is a plus", "Active security clearance"):
            self.assertTrue(compass._VERIFY_YOURSELF.search(text), text)
        for text in ("Run technical product demos for prospects", "Strong SQL skills", "Excellent written communication"):
            self.assertFalse(compass._VERIFY_YOURSELF.search(text), text)

    def test_years_and_degree_requirements_are_never_auto_supported(self) -> None:
        with mock.patch.object(compass, "CLUSTER_DISTANCE", 0.05):  # keep each distinct sentence in its own group
            result = self.run_analysis(self.PICKS)
        years = [r for r in result["requirements"] if "years" in r["text"].lower()]
        self.assertTrue(years)
        for req in years:
            self.assertTrue(req["verify"])
            self.assertNotEqual(req["status"], "supported")

    def test_headline_is_not_an_evidence_line(self) -> None:
        self.assertFalse(any(u.source == "Headline" for u in compass.profile_units(PROFILE)))

    def test_same_company_and_title_counts_once(self) -> None:
        twin = self.by_id["sales-1"]
        twin.source["company_reference"] = dict(self.by_id["sales-0"].source["company_reference"])
        twin.source["title_searchable"] = self.by_id["sales-0"].source["title_searchable"]
        result = self.run_analysis(self.PICKS)
        self.assertEqual(result["duplicates_collapsed"], 1)

    def test_long_occupation_labels_are_shortened_for_theme_names(self) -> None:
        long_name = "Sales Representatives of Services, Except Advertising, Insurance, Financial Services, and Travel"
        name = compass._theme_name([{"onet_data": {"occupation_name": long_name}, "title_searchable": "Sales Executive"}])
        self.assertEqual(name, "Sales Representatives of Services")

    def test_boilerplate_never_becomes_a_requirement(self) -> None:
        result = self.run_analysis(self.PICKS)
        for req in result["requirements"]:
            self.assertNotIn("snacks", req["text"].lower())

    def test_suggestions_exclude_picks_and_rated_jobs(self) -> None:
        result = self.run_analysis(self.PICKS, excluded=["sales-3", "data-3"])
        suggested = [j["job_id"] for j in result["jobs"] if not j["pick"]]
        self.assertLessEqual(len(suggested), compass.SUGGESTION_COUNT)
        self.assertTrue(suggested)
        self.assertFalse(set(suggested) & set(self.PICKS))
        self.assertFalse(set(suggested) & {"sales-3", "data-3"})
        self.assertEqual(len(suggested), len(set(suggested)))

    def test_next_steps_are_the_frequent_unsupported_groups(self) -> None:
        result = self.run_analysis(self.PICKS)
        self.assertTrue(result["next_steps"])
        for step in result["next_steps"]:
            self.assertNotEqual(step["status"], "supported")
        scores = [s["n"] * (1 - s["coverage"] / 100) for s in result["next_steps"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_duplicate_listings_are_collapsed(self) -> None:
        self.by_id["sales-1"].source["duplicate_key"] = self.by_id["sales-0"].source["duplicate_key"]
        result = self.run_analysis(self.PICKS)
        self.assertEqual(result["duplicates_collapsed"], 1)
        self.assertEqual(result["picks_analyzed"], 5)

    def test_errors(self) -> None:
        with self.assertRaises(compass.CompassError) as raised:
            self.run_analysis(["sales-0"])
        self.assertEqual(raised.exception.code, "too_few_distinct_jobs")
        with self.assertRaises(compass.CompassError) as raised:
            self.run_analysis(self.PICKS, profile={**PROFILE, "experience": [], "skills": [], "headline": ""})
        self.assertEqual(raised.exception.code, "profile_too_thin")

    def test_single_encode_call_for_all_text(self) -> None:
        self.run_analysis(self.PICKS)
        self.assertEqual(self.backend.encode_calls, 1)

    def test_two_picks_is_enough_and_uses_one_theme(self) -> None:
        result = self.run_analysis(["sales-0", "sales-1"])
        self.assertEqual(len(result["themes"]), 1)
        self.assertEqual(result["picks_analyzed"], 2)

    def test_scores_are_not_presented_as_probabilities(self) -> None:
        result = self.run_analysis(self.PICKS)
        self.assertIn("not hiring probabilities", " ".join(result["method"]["limits"]).lower())
        self.assertIn("unknown, not a verdict", " ".join(result["method"]["limits"]))


class TextExtractionTests(unittest.TestCase):
    def test_requirement_sentences_skip_benefits_and_boilerplate(self) -> None:
        text = ("About us\nWe are a great company.\nRequirements\nExperience with SQL and reporting tools in a team.\n"
                "Equal opportunity employer statement here for all applicants.\nResponsibilities\nPrepare weekly reports for managers.\n"
                "Benefits\nHealth insurance and paid time off for everyone.")
        sentences = compass.requirement_sentences(text)
        self.assertEqual(sentences, ["Experience with SQL and reporting tools in a team.", "Prepare weekly reports for managers."])

    def test_requirement_sentences_are_capped(self) -> None:
        text = "Requirements\n" + "\n".join(f"Must have strong skill number {i} in depth today." for i in range(60))
        self.assertEqual(len(compass.requirement_sentences(text)), compass.MAX_SENTENCES_PER_JOB)

    def test_profile_units_carry_sources_and_dedupe(self) -> None:
        units = compass.profile_units({**PROFILE, "skills": PROFILE["skills"] + [{"id": "s2", "name": "SQL", "evidence": "Wrote queries for account reporting."}]})
        sources = {u.source for u in units}
        self.assertIn("Experience · Inside Sales Representative", sources)
        self.assertIn("Education · Data Analytics Certificate", sources)
        self.assertIn("Skills · SQL", sources)
        texts = [u.text.casefold() for u in units]
        self.assertEqual(len(texts), len(set(texts)))


class CompassEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.jobs = build_jobs(root)
        self.backend = FakeBackend(self.jobs)
        settings = Settings(db_path=root / "d.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        patcher = mock.patch.multiple(compass, **FAKE_THRESHOLDS)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(settings=settings, ranker=self.backend))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.assertEqual(self.client.post("/api/v1/demo-sessions").status_code, 201)
        created = self.client.post("/api/v1/profiles", json={**PROFILE})
        self.assertEqual(created.status_code, 201, created.text)
        self.profile_id = created.json()["profile_id"]

    def post(self, job_ids, **overrides):
        body = {"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "job_ids": job_ids, **overrides}
        return self.client.post(f"/api/v1/profiles/{self.profile_id}/compass", json=body)

    def test_returns_analysis_with_job_summaries(self) -> None:
        response = self.post(["sales-0", "sales-1", "sales-2", "data-0", "data-1"])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["picks_analyzed"], 5)
        self.assertEqual(body["ranking"]["ranker_id"], "fake_embedding")
        for item in body["jobs"]:
            self.assertEqual(item["job"]["job_id"], item["job_id"])
            self.assertIn("title", item["job"])

    def test_rated_jobs_are_not_suggested(self) -> None:
        for job_id in ("sales-3", "sales-4", "sales-5", "sales-6", "sales-7"):
            self.client.put("/api/v1/feedback", json={"job_id": job_id, "snapshot_id": SNAPSHOT_ID, "signal": "bad"})
        body = self.post(["sales-0", "sales-1", "sales-2", "data-0"]).json()
        suggested = {j["job_id"] for j in body["jobs"] if not j["pick"]}
        self.assertFalse(suggested & {"sales-3", "sales-4", "sales-5", "sales-6", "sales-7"})

    def test_validation_errors(self) -> None:
        self.assertEqual(self.post(["sales-0"]).status_code, 422)                       # fewer than two
        self.assertEqual(self.post(["sales-0", "nope"]).status_code, 404)               # unknown job
        self.assertEqual(self.post(["sales-0", "sales-1"], profile_version=9).status_code, 409)
        self.assertEqual(self.post(["sales-0", "sales-1"], snapshot_id="predictleads-1999-01-01").status_code, 404)
        self.assertEqual(self.post([f"sales-{i % 8}" for i in range(21)]).status_code, 422)  # more than twenty

    def test_same_job_twice_counts_once(self) -> None:
        self.assertEqual(self.post(["sales-0", "sales-0"]).status_code, 422)

    def test_requires_session(self) -> None:
        self.client.cookies.clear()
        self.assertEqual(self.post(["sales-0", "sales-1"]).status_code, 401)

    def test_unavailable_without_the_embedding_ranker(self) -> None:
        root = Path(self.temp.name)
        settings = Settings(db_path=root / "e.sqlite3", sample_path=root / "jobs.jsonl", session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        with TestClient(create_app(settings=settings)) as client:  # default TF-IDF ranker has no compass support
            client.post("/api/v1/demo-sessions")
            pid = client.post("/api/v1/profiles", json={**PROFILE}).json()["profile_id"]
            response = client.post(f"/api/v1/profiles/{pid}/compass", json={"snapshot_id": SNAPSHOT_ID, "profile_version": 1, "job_ids": ["sales-0", "sales-1"]})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"]["code"], "compass_unavailable")


if __name__ == "__main__":
    unittest.main()
