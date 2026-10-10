"""Offline tests for grounded resume extraction. No provider or network calls."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

from backend.api import Settings, create_app
import grounded_match_explanations as grounded
import resume_profile_extraction as rpe

RESUME = (
    "Maya Okafor\nAustin, TX\n\n"
    "Experience\n"
    "Tier 2 Support Specialist, Fictional SaaS vendor, 2021-2025\n"
    "- Handled escalated tickets in Zendesk and filed reproducible bug reports with engineering.\n"
    "- Wrote help-center articles and onboarded new accounts.\n\n"
    "Skills\nSQL: wrote queries to pull account data for reporting.\nPython: small automation scripts.\n"
    "Education\nData Analytics Certificate, 2025, capstone on support-ticket trends.\n"
) * 1


def make_pdf(lines: list[str]) -> bytes:
    """A minimal one-page text PDF (no external library) for tests; empty `lines` gives a page with no text."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream = "BT /F1 10 Tf 12 TL 40 760 Td " + " ".join(f"({escape(line)}) Tj T*" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


PDF_LINES = [line for line in RESUME.splitlines() if line.strip()] + ["Email: maya.okafor@example.com", "Phone: (512) 555-0142", "https://www.linkedin.com/in/mayaokafor"]


def good_response() -> dict[str, Any]:
    return {
        "name": "Maya Okafor", "name_quote": "Maya Okafor",
        "headline": None, "headline_quote": None,
        "location": "Austin, TX", "location_quote": "Austin, TX",
        "experience": [{
            "kind": "job", "title": "Tier 2 Support Specialist", "organization": "Fictional SaaS vendor",
            "description": "Handled escalated tickets in Zendesk and filed bug reports.",
            "source_quote": "Handled escalated tickets in Zendesk and filed reproducible bug reports with engineering.",
        }],
        "skills": [
            {"name": "SQL", "evidence": "Wrote queries for account reporting.", "source_quote": "SQL: wrote queries to pull account data for reporting."},
            {"name": "Python", "evidence": "Small automation scripts.", "source_quote": "Python: small automation scripts."},
        ],
    }


class FakeProvider:
    model = "offline-fake"

    def __init__(self, response: Mapping[str, Any] | Exception):
        self.response, self.calls, self.payloads = response, 0, []

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        self.payloads.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ExtractProfileTests(unittest.TestCase):
    def test_grounded_items_pass_through(self) -> None:
        result = rpe.extract_profile(RESUME, FakeProvider(good_response()))
        profile = result["profile"]
        self.assertEqual(result["dropped"], 0)
        self.assertEqual(profile["name"], "Maya Okafor")
        self.assertEqual(profile["location"], "Austin, TX")
        self.assertEqual(profile["headline"], "")
        self.assertEqual([s["name"] for s in profile["skills"]], ["SQL", "Python"])
        self.assertEqual(profile["experience"][0]["organization"], "Fictional SaaS vendor")

    def test_invented_items_are_dropped_and_counted(self) -> None:
        response = good_response()
        response["skills"].append({"name": "Kubernetes", "evidence": "Ran clusters.", "source_quote": "Administered Kubernetes clusters in production."})
        response["experience"].append({"kind": "job", "title": "Staff Engineer", "organization": "Google", "description": "Led a team.",
                                       "source_quote": "Led a team of engineers at Google."})
        response["headline"], response["headline_quote"] = "Seasoned engineer", "Seasoned engineer with ten years"
        result = rpe.extract_profile(RESUME, FakeProvider(response))
        self.assertEqual(result["dropped"], 3)
        self.assertEqual([s["name"] for s in result["profile"]["skills"]], ["SQL", "Python"])
        self.assertEqual(len(result["profile"]["experience"]), 1)
        self.assertEqual(result["profile"]["headline"], "")

    def test_organization_not_in_resume_is_blanked_not_kept(self) -> None:
        response = good_response()
        response["experience"][0]["organization"] = "Acme Corp"
        result = rpe.extract_profile(RESUME, FakeProvider(response))
        self.assertEqual(result["profile"]["experience"][0]["organization"], "")
        self.assertEqual(result["dropped"], 1)

    def test_quote_matching_ignores_case_whitespace_and_bullets(self) -> None:
        response = good_response()
        response["experience"][0]["source_quote"] = "- handled  escalated tickets in zendesk and filed reproducible bug reports with engineering."
        result = rpe.extract_profile(RESUME, FakeProvider(response))
        self.assertEqual(result["dropped"], 0)

    def test_glued_quote_with_stray_separators_is_accepted_but_fabricated_segment_is_not(self) -> None:
        resume = RESUME + "Core Skills\nTechnical Sales • Solution Selling • Python • FastAPI\n"
        response = good_response()
        response["skills"] = [
            {"name": "Python", "evidence": "Listed.", "source_quote": "Technical Sales \t6 Solution Selling \t6 Python \t6 FastAPI"},
            {"name": "Rust", "evidence": "Listed.", "source_quote": "Technical Sales \t6 Rust expert \t6 FastAPI"},
        ]
        result = rpe.extract_profile(resume, FakeProvider(response))
        self.assertEqual([s["name"] for s in result["profile"]["skills"]], ["Python"])
        self.assertEqual(result["dropped"], 1)

    def test_entry_kinds_pass_through_and_unknown_kind_becomes_job(self) -> None:
        response = good_response()
        response["experience"].append({
            "kind": "education", "title": "Data Analytics Certificate", "organization": None,
            "description": "Capstone on support-ticket trends.", "source_quote": "Data Analytics Certificate, 2025",
        })
        response["experience"].append({
            "kind": "volunteering", "title": "Tier 2 Support Specialist", "organization": None,
            "description": "Duplicate title, odd kind.", "source_quote": "Tier 2 Support Specialist",
        })
        kinds = [entry["kind"] for entry in rpe.extract_profile(RESUME, FakeProvider(response))["profile"]["experience"]]
        self.assertEqual(kinds, ["job", "education", "job"])

    def test_desired_roles_are_never_returned(self) -> None:
        response = good_response()
        response["desired_roles"] = ["Solutions Engineer"]
        profile = rpe.extract_profile(RESUME, FakeProvider(response))["profile"]
        self.assertNotIn("desired_roles", profile)

    def test_limits_and_duplicates(self) -> None:
        response = good_response()
        response["skills"] = response["skills"] + [dict(response["skills"][0])]
        self.assertEqual(len(rpe.extract_profile(RESUME, FakeProvider(response))["profile"]["skills"]), 2)

    def test_short_and_long_text_rejected_without_calling_provider(self) -> None:
        provider = FakeProvider(good_response())
        with self.assertRaises(grounded.ResponseValidationError):
            rpe.extract_profile("too short", provider)
        with self.assertRaises(grounded.ResponseValidationError):
            rpe.extract_profile("x" * (rpe.MAX_RESUME_CHARS + 1), provider)
        self.assertEqual(provider.calls, 0)

    def test_provider_failure_propagates(self) -> None:
        with self.assertRaises(grounded.ProviderFailure):
            rpe.extract_profile(RESUME, FakeProvider(grounded.ProviderFailure("down")))

    def test_schema_is_strict(self) -> None:
        def check(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node.get("additionalProperties", True))
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(rpe.RESPONSE_SCHEMA)


class RedactionAndPdfTests(unittest.TestCase):
    def test_redaction_removes_contact_details_but_keeps_date_ranges(self) -> None:
        text = (
            "Maya maya.okafor@example.com (512) 555-0142 512-555-0142 +1 512 555 0142 "
            "https://linkedin.com/in/maya www.maya.dev Worked 2021-2025 and 2008 - 2012."
        )
        cleaned = rpe.redact_contact_details(text)
        for gone in ("@", "555", "linkedin", "www."):
            self.assertNotIn(gone, cleaned)
        self.assertIn("2021-2025", cleaned)
        self.assertIn("2008 - 2012", cleaned)

    def test_provider_never_sees_contact_details(self) -> None:
        provider = FakeProvider(good_response())
        rpe.extract_profile(RESUME + "\nmaya.okafor@example.com | (512) 555-0142 | https://linkedin.com/in/maya\n", provider)
        sent = provider.payloads[0]["resume_text"]
        for gone in ("@", "555-0142", "linkedin"):
            self.assertNotIn(gone, sent)

    def test_pdf_to_text_reads_text_pdf(self) -> None:
        text = rpe.pdf_to_text(make_pdf(PDF_LINES))
        self.assertIn("Tier 2 Support Specialist", text)

    def test_pdf_to_text_rejects_bad_inputs(self) -> None:
        cases = (
            (b"not a pdf", "not a PDF"),
            (make_pdf([]), "No readable text"),
            (b"%PDF" + b"x" * (rpe.MAX_PDF_BYTES + 1), "too large"),
        )
        for data, fragment in cases:
            with self.assertRaises(grounded.ResponseValidationError) as raised:
                rpe.pdf_to_text(data)
            self.assertIn(fragment, str(raised.exception))

    def test_one_word_per_line_pdf_text_is_rejoined(self) -> None:
        sentence = "Alex Rivera data scientist with a strong analytical foundation developed through experience as an analyst "
        text = rpe.pdf_to_text(make_pdf((sentence * 4).split()))
        self.assertNotIn("\n", text.strip())
        self.assertIn("data scientist with a strong", text)


class ProviderConfigTests(unittest.TestCase):
    def test_requires_key(self) -> None:
        with self.assertRaises(grounded.ProviderUnavailable):
            rpe.OpenAIResumeProvider.from_environment({})
        with self.assertRaises(grounded.ProviderUnavailable):
            rpe.OpenAIResumeProvider("  ")


class ExtractionEndpointTests(unittest.TestCase):
    def make_client(self, provider: Any) -> TestClient:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        job = {"job_id": "a", "job_type": "job_opening", "title_raw": "T", "title_normalized": "T", "title_translated": None,
               "title_searchable": "T", "title_searchable_source": "normalized_title", "description_text": "Do things.",
               "description_usable": True, "snapshot_date": "2026-09-30", "status_derived": "unknown",
               "company_reference": {"id": "c", "type": "company"}, "categories": [], "tags": [], "onet_data": None,
               "dates_utc": {}, "quality_flags": {}, "language": "en", "seniority": None, "contract_types": []}
        jobs = root / "jobs.jsonl"
        jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
        settings = Settings(db_path=root / "d.sqlite3", sample_path=jobs, session_ttl_seconds=3600,
                            secure_cookie=False, allowed_origins=("http://127.0.0.1:5173",))
        client = TestClient(create_app(settings=settings, extraction_provider=provider))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        self.assertEqual(client.post("/api/v1/demo-sessions").status_code, 201)
        return client

    def test_completed(self) -> None:
        client = self.make_client(FakeProvider(good_response()))
        response = client.post("/api/v1/profile-extraction", json={"resume_text": RESUME})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["state"], "completed")
        self.assertEqual(body["profile"]["name"], "Maya Okafor")
        self.assertEqual(body["model"], "offline-fake")

    def test_disabled_by_default_and_never_calls_provider(self) -> None:
        client = self.make_client(None)
        body = client.post("/api/v1/profile-extraction", json={"resume_text": RESUME}).json()
        self.assertEqual((body["state"], body["reason"], body["profile"]), ("unavailable", "provider_disabled", None))

    def test_requires_session(self) -> None:
        client = self.make_client(FakeProvider(good_response()))
        client.cookies.clear()
        self.assertEqual(client.post("/api/v1/profile-extraction", json={"resume_text": RESUME}).status_code, 401)

    def test_provider_failure_is_retryable_502_and_text_not_echoed(self) -> None:
        client = self.make_client(FakeProvider(grounded.ProviderFailure("down")))
        response = client.post("/api/v1/profile-extraction", json={"resume_text": RESUME})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("Maya", response.text)

    def test_too_short_is_rejected(self) -> None:
        client = self.make_client(FakeProvider(good_response()))
        self.assertEqual(client.post("/api/v1/profile-extraction", json={"resume_text": "short"}).status_code, 422)

    def test_pdf_upload_completes_and_provider_sees_no_contact_details(self) -> None:
        provider = FakeProvider(good_response())
        client = self.make_client(provider)
        body = {"resume_pdf_base64": base64.b64encode(make_pdf(PDF_LINES)).decode()}
        response = client.post("/api/v1/profile-extraction", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "completed")
        sent = provider.payloads[0]["resume_text"]
        self.assertNotIn("@", sent)
        self.assertNotIn("555-0142", sent)

    def test_pdf_errors_are_422_and_do_not_call_provider(self) -> None:
        provider = FakeProvider(good_response())
        client = self.make_client(provider)
        bodies = (
            {"resume_pdf_base64": "!!!not base64!!!"},
            {"resume_pdf_base64": base64.b64encode(b"hello world").decode()},
            {"resume_pdf_base64": base64.b64encode(make_pdf([])).decode()},
        )
        for body in bodies:
            self.assertEqual(client.post("/api/v1/profile-extraction", json=body).status_code, 422)
        self.assertEqual(provider.calls, 0)

    def test_profile_entries_store_kind_and_default_to_job(self) -> None:
        client = self.make_client(None)
        entries = [
            {"id": "e1", "title": "Support Specialist", "organization": None, "description": "Tickets."},
            {"id": "e2", "title": "Capstone", "organization": None, "description": "CIFAR-10.", "kind": "project"},
            {"id": "e3", "title": "B.A.", "organization": "State U", "description": "", "kind": "education"},
        ]
        response = client.post("/api/v1/profiles", json={"name": "T", "experience": entries})
        self.assertEqual(response.status_code, 201, response.text)
        stored = response.json()["profile"]["experience"]
        self.assertEqual([e["kind"] for e in stored], ["job", "project", "education"])
        bad = client.patch(f"/api/v1/profiles/{response.json()['profile_id']}", headers={"If-Match": '"1"'},
                           json={"experience": [{**entries[0], "kind": "hobby"}]})
        self.assertEqual(bad.status_code, 422)

    def test_exactly_one_source_required(self) -> None:
        client = self.make_client(FakeProvider(good_response()))
        self.assertEqual(client.post("/api/v1/profile-extraction", json={}).status_code, 422)
        both = {"resume_text": RESUME, "resume_pdf_base64": "AAAA"}
        self.assertEqual(client.post("/api/v1/profile-extraction", json=both).status_code, 422)


if __name__ == "__main__":
    unittest.main()
