"""Assessing several jobs with one provider request (the Compass skills gap analysis). Offline: a mock provider only."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import grounded_match_explanations as grounded  # noqa: E402
from grounded_match_explanations import MockProvider, ResponseValidationError, explain_matches  # noqa: E402

CANDIDATE = {
    "profile_id": "p1",
    "experience": ["Ran weekly product demos for enterprise accounts."],
    "skills": ["Python"],
}


def job(job_id: str, description: str | None = "Responsibilities\n- Demo the product to prospects.\n- Own the technical evaluation.") -> dict:
    return {"job_id": job_id, "title": f"Sales Engineer {job_id}", "description_text": description}


def good_item(entry: dict) -> dict:
    """A valid answer for one job in the payload, citing that job's own first line and the candidate's first line."""
    job_line = entry["evidence"][0]["evidence_id"]
    return {
        "profile_id": "p1", "job_id": entry["job_id"], "evidence_sufficiency": "sufficient", "insufficiency_reason": None,
        "transferable_experience": [],
        "qualifications": [{"qualification_id": "q1", "qualification": "Demo the product", "classification": "required", "status": "supported",
                            "candidate_evidence_ids": ["candidate.experience[0]"], "job_evidence_ids": [job_line], "rationale": "Your demos match."}],
        "next_steps": [],
    }


def responder(payload):
    return {"jobs": [good_item(entry) for entry in payload["jobs"]]}


class BatchTests(unittest.TestCase):
    def test_several_jobs_cost_one_request(self) -> None:
        provider = MockProvider(responder)
        out = explain_matches(CANDIDATE, [job("a"), job("b"), job("c")], provider=provider)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(out["failed"], 0)
        self.assertEqual(set(out["jobs"]), {"a", "b", "c"})
        self.assertTrue(all(r["state"] == "completed" for r in out["jobs"].values()))

    def test_candidate_lines_are_returned_once_and_job_lines_stay_separate(self) -> None:
        out = explain_matches(CANDIDATE, [job("a"), job("b")], provider=MockProvider(responder))
        self.assertTrue(any(e["evidence_id"] == "candidate.experience[0]" for e in out["candidate_evidence"]))
        for result in out["jobs"].values():
            ids = [e["evidence_id"] for e in result["assessment"]["evidence_catalog"]]
            self.assertTrue(ids and not any(i.startswith("candidate.") for i in ids))
        a_ids = {e["evidence_id"] for e in out["jobs"]["a"]["assessment"]["evidence_catalog"]}
        b_ids = {e["evidence_id"] for e in out["jobs"]["b"]["assessment"]["evidence_catalog"]}
        self.assertFalse(a_ids & b_ids)  # each job's lines have their own ids

    def test_a_bad_answer_fails_only_that_job(self) -> None:
        def respond(payload):
            items = [good_item(e) for e in payload["jobs"]]
            items[1]["qualifications"][0]["job_evidence_ids"] = ["job0.description_text[0]"]  # a line from a different job
            return {"jobs": items}
        out = explain_matches(CANDIDATE, [job("a"), job("b"), job("c")], provider=MockProvider(respond))
        self.assertEqual(out["jobs"]["b"]["state"], "failed")
        self.assertEqual(out["jobs"]["a"]["state"], "completed")
        self.assertEqual(out["jobs"]["c"]["state"], "completed")
        self.assertEqual(out["failed"], 1)

    def test_a_job_the_provider_skipped_is_reported_failed(self) -> None:
        out = explain_matches(CANDIDATE, [job("a"), job("b")], provider=MockProvider(lambda p: {"jobs": [good_item(p["jobs"][0])]}))
        self.assertEqual(out["jobs"]["b"]["state"], "failed")

    def test_answers_with_changed_or_repeated_ids_are_matched_by_position(self) -> None:
        def respond(payload):
            items = [good_item(e) for e in payload["jobs"]]
            for item in items:
                item["job_id"] = "same-id-for-everyone"
            return {"jobs": items}
        out = explain_matches(CANDIDATE, [job("a"), job("b"), job("c")], provider=MockProvider(respond))
        self.assertEqual(out["failed"], 0)
        self.assertTrue(all(r["state"] == "completed" for r in out["jobs"].values()))

    def test_position_matching_still_rejects_lines_from_another_job(self) -> None:
        def respond(payload):
            items = [good_item(e) for e in payload["jobs"]]
            for item in items:
                item["job_id"] = "x"
            items[1], items[2] = items[2], items[1]  # answers swapped: each now cites the other job's lines
            return {"jobs": items}
        out = explain_matches(CANDIDATE, [job("a"), job("b"), job("c")], provider=MockProvider(respond))
        self.assertEqual(out["jobs"]["a"]["state"], "completed")
        self.assertEqual(out["jobs"]["b"]["state"], "failed")
        self.assertEqual(out["jobs"]["c"]["state"], "failed")

    def test_a_bad_next_step_is_dropped_but_the_assessment_is_kept(self) -> None:
        def respond(payload):
            items = [good_item(e) for e in payload["jobs"]]
            for item in items:
                item["next_steps"] = [
                    {"action": "Brush up", "qualification_ids": ["q1"], "evidence_ids": []},          # empty evidence, and q1 is supported
                    {"action": "Also bad", "qualification_ids": ["q1"], "evidence_ids": ["job0.description_text[0]"]},
                ]
            return {"jobs": items}
        out = explain_matches(CANDIDATE, [job("a"), job("b")], provider=MockProvider(respond))
        self.assertEqual(out["failed"], 0)
        for result in out["jobs"].values():
            self.assertEqual(result["state"], "completed")
            self.assertEqual(result["assessment"]["next_steps"], [])
            self.assertEqual(len(result["assessment"]["qualifications"]), 1)

    def test_a_bad_qualification_still_fails_the_job(self) -> None:
        def respond(payload):
            items = [good_item(e) for e in payload["jobs"]]
            items[0]["qualifications"][0]["candidate_evidence_ids"] = []  # "supported" with no profile line
            return {"jobs": items}
        out = explain_matches(CANDIDATE, [job("a"), job("b")], provider=MockProvider(respond))
        self.assertEqual(out["jobs"]["a"]["state"], "failed")
        self.assertEqual(out["jobs"]["b"]["state"], "completed")

    def test_job_without_a_description_is_not_sent(self) -> None:
        seen = {}
        def respond(payload):
            seen["ids"] = [e["job_id"] for e in payload["jobs"]]
            return responder(payload)
        out = explain_matches(CANDIDATE, [job("a"), job("empty", description=None)], provider=MockProvider(respond))
        self.assertEqual(seen["ids"], ["a"])
        self.assertEqual(out["jobs"]["empty"]["assessment"]["evidence_sufficiency"], "insufficient")

    def test_empty_profile_makes_no_request(self) -> None:
        provider = MockProvider(responder)
        out = explain_matches({"profile_id": "p1", "experience": [], "skills": []}, [job("a"), job("b")], provider=provider)
        self.assertEqual(provider.calls, 0)
        self.assertTrue(all(r["assessment"]["evidence_sufficiency"] == "insufficient" for r in out["jobs"].values()))

    def test_long_descriptions_are_trimmed_before_sending(self) -> None:
        seen = {}
        def respond(payload):
            seen["chars"] = sum(len(e["text"]) for e in payload["jobs"][0]["evidence"])
            return responder(payload)
        long_text = "\n".join(f"- Requirement number {i} with some extra words to fill the line." for i in range(2000))
        explain_matches(CANDIDATE, [job("a", long_text)], provider=MockProvider(respond))
        self.assertLessEqual(seen["chars"], grounded.MAX_DESCRIPTION_CHARS + 200)

    def test_bad_inputs_are_rejected(self) -> None:
        with self.assertRaises(ResponseValidationError):
            explain_matches(CANDIDATE, [], provider=MockProvider(responder))
        with self.assertRaises(ResponseValidationError):
            explain_matches(CANDIDATE, [job("a"), job("a")], provider=MockProvider(responder))
        with self.assertRaises(ResponseValidationError):
            explain_matches(CANDIDATE, [job(str(i)) for i in range(grounded.MAX_BATCH_JOBS + 1)], provider=MockProvider(responder))
        with self.assertRaises(ResponseValidationError):
            explain_matches(CANDIDATE, [job("a")], provider=MockProvider(lambda p: {"not_jobs": []}))

    def test_provider_failure_propagates(self) -> None:
        def boom(payload):
            raise grounded.ProviderFailure("down")
        with self.assertRaises(grounded.ProviderFailure):
            explain_matches(CANDIDATE, [job("a")], provider=MockProvider(boom))


if __name__ == "__main__":
    unittest.main()
