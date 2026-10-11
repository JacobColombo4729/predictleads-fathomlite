import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import evaluate_bge_transition_openai as evaluator


class OpenAIEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.pool_row = {
            "profile_id": "synthetic_profile", "job_id": "stable-job-id",
            "candidate_evidence": {"work_history": ["Recorded field readings on assigned routes."]},
            "job_evidence": {"description": "Record field readings and report access issues."},
        }
        self.prompt = "Frozen rubric text."

    def judgment(self, **overrides):
        row = {
            "profile_id": "synthetic_profile", "job_id": "stable-job-id", "grade": 2,
            "status": "judged", "unknown_reason": None,
            "supporting_candidate_evidence": ["Recorded field readings on assigned routes."],
            "supporting_job_evidence": ["Record field readings and report access issues."],
            "material_gaps": [], "explicit_constraint_conflicts": [],
            "required_qualifications": "unknown",
            "career_transition_relevance": {"level": "high", "rationale": "Field records transfer."},
            "title_outside_stated_targets": False, "evidence_sufficiency": "sufficient",
        }
        row.update(overrides)
        return row

    def id_judgment(self, **overrides):
        row = self.judgment()
        row.pop("supporting_candidate_evidence")
        row.pop("supporting_job_evidence")
        row["supporting_candidate_evidence_ids"] = ["C0001"]
        row["supporting_job_evidence_ids"] = ["J0001"]
        row.update(overrides)
        return row

    def test_frozen_prompt_hash_matches_manifest(self):
        prompt = evaluator.frozen_prompt_text()
        self.assertIn("A qualification missing from the profile is unknown", prompt)
        self.assertEqual(evaluator.FROZEN_PROMPT_SHA256,
                         "2DC3EDDA4FE52C8AD5874D11727EEAC5FFDA49DBE33FEA705EF1FE02A119FD11")

    def test_process_environment_key_precedes_local_dotenv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OPENAI_API_KEY=dotenv-test-value\n", encoding="utf-8")
            with patch.dict("os.environ", {"OPENAI_API_KEY": "process-test-value"}):
                self.assertEqual(evaluator.load_api_key(root), "process-test-value")

    def test_local_dotenv_key_loads_when_process_key_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OPENAI_API_KEY=dotenv-test-value\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(evaluator.load_api_key(root), "dotenv-test-value")

    def test_atomic_checkpoint_retries_transient_windows_replace_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "checkpoint.jsonl"
            destination.write_text("old\n", encoding="utf-8")
            legacy_temp = Path(str(destination) + ".tmp")
            legacy_temp.write_text("preserve stale temporary\n", encoding="utf-8")
            actual_replace = evaluator.os.replace
            attempts = []

            def replace_with_transient_lock(source, target):
                attempts.append((Path(source), Path(target)))
                if len(attempts) == 1:
                    raise PermissionError("transient Windows file lock")
                return actual_replace(source, target)

            with patch.object(evaluator.os, "replace", side_effect=replace_with_transient_lock), \
                    patch.object(evaluator.time, "sleep"):
                evaluator._atomic_write_text(destination, "new\n")

            self.assertEqual(destination.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0][0], attempts[1][0])
            self.assertNotEqual(attempts[0][0], legacy_temp)
            self.assertEqual(legacy_temp.read_text(encoding="utf-8"), "preserve stale temporary\n")

    def test_payload_contains_only_blinded_ids_and_original_evidence(self):
        payload = evaluator.build_response_payload(self.pool_row, self.prompt)
        encoded = json.dumps(payload)
        self.assertIn("stable-job-id", encoded)
        self.assertNotIn("rank", encoded)
        self.assertNotIn("score", encoded)
        self.assertNotIn("method", encoded)
        supplied = json.loads(payload["input"][0]["content"])
        self.assertEqual(supplied["candidate_evidence"][0]["evidence_id"], "C0001")
        self.assertEqual(supplied["candidate_evidence"][0]["source_text"],
                 "Recorded field readings on assigned routes.")
        self.assertEqual(supplied["job_evidence"][0]["evidence_id"], "J0001")
        altered = {**self.pool_row, "rank": 1}
        with self.assertRaises(ValueError):
            evaluator.build_response_payload(altered, self.prompt)

    def test_request_identity_changes_with_evidence(self):
        first = evaluator.request_identity(self.pool_row, self.prompt)
        changed = {**self.pool_row, "job_evidence": {"description": "Different source evidence."}}
        self.assertNotEqual(first, evaluator.request_identity(changed, self.prompt))
        self.assertNotEqual(first, evaluator.request_identity(self.pool_row, self.prompt + " changed"))

    def test_exact_excerpts_are_required_and_unknown_remains_null(self):
        valid, errors = evaluator.validate_judgment(self.judgment(), self.pool_row)
        self.assertEqual(errors, [])
        self.assertIsNotNone(valid)
        invalid, errors = evaluator.validate_judgment(
            self.judgment(supporting_job_evidence=["This paraphrase is not an exact excerpt."]), self.pool_row)
        self.assertIsNone(invalid)
        self.assertIn("job_excerpt_not_exact", errors)
        unknown, errors = evaluator.validate_judgment(self.judgment(
            grade=None, status="unknown", unknown_reason="The source evidence is insufficient.",
            supporting_candidate_evidence=[], supporting_job_evidence=[],
            career_transition_relevance={"level": "unknown", "rationale": "Insufficient evidence."},
            evidence_sufficiency="insufficient"), self.pool_row)
        self.assertEqual(errors, [])
        self.assertIsNone(unknown["grade"])

    def test_case_only_excerpt_is_canonicalized_but_paraphrase_is_rejected(self):
        lower_case = self.judgment(
            supporting_candidate_evidence=["recorded field readings on assigned routes."])
        valid, errors = evaluator.validate_judgment(lower_case, self.pool_row)
        self.assertEqual(errors, [])
        self.assertEqual(valid["supporting_candidate_evidence"],
                         ["Recorded field readings on assigned routes."])
        paraphrase = self.judgment(
            supporting_candidate_evidence=["Collected readings while travelling along assigned routes."])
        invalid, errors = evaluator.validate_judgment(paraphrase, self.pool_row)
        self.assertIsNone(invalid)
        self.assertIn("candidate_excerpt_not_exact", errors)

    def test_schema_requests_source_evidence_ids(self):
        candidate_description = evaluator.JUDGMENT_SCHEMA["properties"]["supporting_candidate_evidence_ids"]["description"]
        job_description = evaluator.JUDGMENT_SCHEMA["properties"]["supporting_job_evidence_ids"]["description"]
        self.assertIn("evidence_id values", candidate_description)
        self.assertIn("Do not invent IDs", candidate_description)
        self.assertIn("evidence_id values", job_description)

    def test_evidence_ids_resolve_exactly_and_unknown_ids_fail(self):
        valid, errors = evaluator.validate_judgment(self.id_judgment(), self.pool_row, require_evidence_ids=True)
        self.assertEqual(errors, [])
        self.assertEqual(valid["supporting_candidate_evidence"],
                         ["Recorded field readings on assigned routes."])
        self.assertEqual(valid["supporting_job_evidence"],
                         ["Record field readings and report access issues."])
        invalid, errors = evaluator.validate_judgment(
            self.id_judgment(supporting_job_evidence_ids=["J9999"]),
            self.pool_row, require_evidence_ids=True)
        self.assertIsNone(invalid)
        self.assertIn("job_excerpt_not_exact", errors)
        legacy, errors = evaluator.validate_judgment(self.judgment(), self.pool_row, require_evidence_ids=True)
        self.assertIsNone(legacy)
        self.assertIn("evidence_id_protocol_required", errors)

    def test_authentication_and_quota_errors_are_fatal(self):
        self.assertTrue(evaluator._must_stop_for_api_error(
            evaluator.OpenAIRequestError(401, "invalid_api_key", "redacted")))
        self.assertTrue(evaluator._must_stop_for_api_error(
            evaluator.OpenAIRequestError(429, "insufficient_quota", "redacted")))
        self.assertFalse(evaluator._must_stop_for_api_error(
            evaluator.OpenAIRequestError(400, "invalid_request_error", "redacted")))

    def test_conflict_claims_require_both_exact_evidence_sides(self):
        invalid, errors = evaluator.validate_judgment(
            self.judgment(explicit_constraint_conflicts=["Mandatory credential conflict."],
                          supporting_candidate_evidence=[], supporting_job_evidence=[]), self.pool_row)
        self.assertIsNone(invalid)
        self.assertIn("conflicts_require_candidate_and_job_excerpts", errors)
        valid, errors = evaluator.validate_judgment(
            self.judgment(explicit_constraint_conflicts=["Mandatory credential conflict."]), self.pool_row)
        self.assertEqual(errors, [])
        self.assertIsNotNone(valid)

    def test_reservations_are_durable_and_cannot_cross_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = evaluator.BudgetLedger(Path(directory) / "ledger.sqlite3", 0.01)
            key = "request-key"
            pair = "pair-key"
            amount = ledger.reserve_pair(key, pair, 1000)
            self.assertEqual(ledger.reserve_pair(key, pair, 1000), amount)
            self.assertEqual(ledger.start_attempt(key, pair), 1)
            reopened = evaluator.BudgetLedger(Path(directory) / "ledger.sqlite3", 0.01)
            self.assertEqual(reopened.attempts(key), 1)
            self.assertEqual(reopened.start_attempt(key, pair), 2)
            with self.assertRaises(RuntimeError):
                reopened.start_attempt(key, pair)
            with self.assertRaises(evaluator.BudgetExceeded):
                reopened.reserve_pair("another-request", "another-pair", 100_000)
            lower_budget = evaluator.BudgetLedger(Path(directory) / "ledger.sqlite3", 0.0001)
            with self.assertRaises(evaluator.BudgetExceeded):
                lower_budget.reserve_pair(key, pair, 1000)

    def test_reservation_accepts_a_larger_audit_output_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = evaluator.BudgetLedger(Path(directory) / "ledger.sqlite3", 0.01)
            amount = ledger.reserve_pair("audit-request", "audit-pair", 1000,
                                         max_output_tokens=1200)
            expected = evaluator._maximum_request_cost(1000, 1200) * evaluator.MAX_ATTEMPTS
            self.assertAlmostEqual(amount, expected)
            connection = sqlite3.connect(ledger.path)
            try:
                saved = connection.execute(
                    "SELECT max_output_tokens, reserve_usd FROM reservations WHERE request_key = ?",
                    ("audit-request",),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(saved[0], 1200)
            self.assertAlmostEqual(saved[1], expected)

    def test_ledger_uses_explicit_alternative_model_rates_for_reserve_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = evaluator.BudgetLedger(Path(directory) / "ledger.sqlite3", 0.1)
            reserved = ledger.reserve_pair(
                "gpt-41-audit", "pair", 2000, max_output_tokens=1800,
                input_usd_per_million=2.0, output_usd_per_million=8.0,
                model_id="gpt-4.1-2025-04-14",
            )
            expected_reserve = evaluator._maximum_request_cost(2000, 1800, 2.0, 8.0) * evaluator.MAX_ATTEMPTS
            self.assertAlmostEqual(reserved, expected_reserve)
            attempt = ledger.start_attempt("gpt-41-audit", "pair")
            actual = ledger.finish_attempt(
                "gpt-41-audit", "pair", attempt, "completed",
                {"model": "gpt-4.1-2025-04-14", "usage": {"input_tokens": 2000, "output_tokens": 900}},
                input_usd_per_million=2.0, output_usd_per_million=8.0,
            )
            self.assertAlmostEqual(actual, evaluator._maximum_request_cost(2000, 900, 2.0, 8.0))

    def test_exact_cache_reuse_rejects_a_different_request_key(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json"
            judgment = self.judgment()
            evaluator._atomic_json(cache_path, {"request_key": "exact-key", "judgment": judgment})
            self.assertEqual(evaluator._cached_judgment(cache_path, "exact-key", self.pool_row), judgment)
            self.assertIsNone(evaluator._cached_judgment(cache_path, "changed-key", self.pool_row))

    def test_metrics_are_withheld_without_numeric_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            evaluator_dir = experiment_dir / "openai_evaluator"
            evaluator_dir.mkdir()
            pending = {"profile_id": "synthetic_profile", "job_id": "stable-job-id",
                       "grade": None, "status": "unknown", "unknown_reason": "Insufficient evidence."}
            (evaluator_dir / "validated_judgments.jsonl").write_text(
                json.dumps(pending) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "metric computation is withheld"):
                evaluator.compute_isolated_metrics(experiment_dir)

    def test_mocked_retry_and_restart_reuse_exact_cached_result(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            evaluator_dir = experiment_dir / "openai_evaluator"
            evaluator_dir.mkdir()
            (evaluator_dir / "evaluator_prompt_frozen.md").write_bytes(
                evaluator.PROMPT_PATH.read_bytes())
            (experiment_dir / "blinded_evaluation_pool.jsonl").write_text(
                json.dumps(self.pool_row) + "\n", encoding="utf-8")
            invalid = self.judgment(supporting_job_evidence=["Not present in the job evidence."])
            valid = self.id_judgment()
            output_values = iter((invalid, valid))
            calls = []

            def fake_http(url, body, api_key, timeout=90.0):
                calls.append(url)
                if url.endswith("/responses/input_tokens"):
                    return {"input_tokens": 1000}
                judgment = next(output_values)
                return {
                    "id": "resp-test-{}".format(len(calls)), "model": evaluator.MODEL,
                    "status": "completed", "usage": {"input_tokens": 1000, "output_tokens": 80},
                    "output": [{"type": "message", "content": [{
                        "type": "output_text", "text": json.dumps(judgment),
                    }]}],
                }

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-mock"}), \
                    patch.object(evaluator, "_http_json", side_effect=fake_http):
                first = evaluator.run_evaluation(experiment_dir, budget_usd=0.01)
                self.assertEqual(first["evaluated_judged"], 1)
                self.assertEqual(first["api_inference_calls_cumulative"], 2)
                first_call_count = len(calls)
                second = evaluator.run_evaluation(experiment_dir, budget_usd=0.01)

            self.assertEqual(second["evaluated_judged"], 1)
            self.assertEqual(len(calls), first_call_count)
            state = evaluator._read_jsonl(evaluator_dir / "pair_status.jsonl")[0]
            self.assertTrue(state["cache_reused"])
            self.assertEqual(state["status"], "evaluated_judged")

    def test_saved_response_recovery_is_reused_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            evaluator_dir = experiment_dir / "openai_evaluator"
            evaluator_dir.mkdir()
            (evaluator_dir / "evaluator_prompt_frozen.md").write_bytes(
                evaluator.PROMPT_PATH.read_bytes())
            (experiment_dir / "blinded_evaluation_pool.jsonl").write_text(
                json.dumps(self.pool_row) + "\n", encoding="utf-8")
            response = {
                "id": "resp-recoverable", "model": evaluator.MODEL, "status": "completed",
                "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": json.dumps(self.judgment()),
                }]}],
            }
            evaluator._atomic_jsonl(evaluator_dir / "raw_responses.jsonl", [{
                "profile_id": self.pool_row["profile_id"], "job_id": self.pool_row["job_id"],
                "request_key": "old-schema-request-key", "attempt": 1,
                "response_id": response["id"], "actual_model": evaluator.MODEL,
                "raw_response": response,
            }])
            report = evaluator.recover_saved_responses(experiment_dir)
            self.assertEqual(report["newly_recovered_judgments"], 1)
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-mock"}), \
                    patch.object(evaluator, "_http_json", side_effect=AssertionError("inference must not run")):
                result = evaluator.run_evaluation(experiment_dir, budget_usd=0.01)
            self.assertEqual(result["evaluated_judged"], 1)
            self.assertEqual(result["api_inference_calls_cumulative"], 0)
            state = evaluator._read_jsonl(evaluator_dir / "pair_status.jsonl")[0]
            self.assertTrue(state["recovered_from_saved_response"])

    def test_insufficient_quota_stops_before_requesting_next_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            evaluator_dir = experiment_dir / "openai_evaluator"
            evaluator_dir.mkdir()
            (evaluator_dir / "evaluator_prompt_frozen.md").write_bytes(
                evaluator.PROMPT_PATH.read_bytes())
            second_row = {**self.pool_row, "job_id": "second-job-id"}
            (experiment_dir / "blinded_evaluation_pool.jsonl").write_text(
                "\n".join(json.dumps(row) for row in (self.pool_row, second_row)) + "\n",
                encoding="utf-8")
            calls = []

            def fake_http(url, body, api_key, timeout=90.0):
                calls.append(url)
                if url.endswith("/responses/input_tokens"):
                    return {"input_tokens": 1000}
                raise evaluator.OpenAIRequestError(429, "insufficient_quota", "credits exhausted")

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-mock"}), \
                    patch.object(evaluator, "_http_json", side_effect=fake_http):
                result = evaluator.run_evaluation(experiment_dir, budget_usd=0.01)
            self.assertEqual(calls, [evaluator.API_BASE + "/responses/input_tokens",
                                     evaluator.API_BASE + "/responses"])
            self.assertEqual(result["stop_reason"], "fatal_api_error_during_inference")
            statuses = evaluator._read_jsonl(evaluator_dir / "pair_status.jsonl")
            self.assertEqual(statuses[0]["status"], "failed")
            self.assertEqual(statuses[1]["status"], "not_evaluated")

    def test_one_pair_gate_does_not_request_second_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            evaluator_dir = experiment_dir / "openai_evaluator"
            evaluator_dir.mkdir()
            (evaluator_dir / "evaluator_prompt_frozen.md").write_bytes(
                evaluator.PROMPT_PATH.read_bytes())
            second_row = {**self.pool_row, "job_id": "second-job-id"}
            (experiment_dir / "blinded_evaluation_pool.jsonl").write_text(
                "\n".join(json.dumps(row) for row in (self.pool_row, second_row)) + "\n",
                encoding="utf-8")
            calls = []

            def fake_http(url, body, api_key, timeout=90.0):
                calls.append(url)
                if url.endswith("/responses/input_tokens"):
                    return {"input_tokens": 1000}
                return {
                    "id": "resp-gate", "model": evaluator.MODEL, "status": "completed",
                    "usage": {"input_tokens": 1000, "output_tokens": 80},
                    "output": [{"type": "message", "content": [{
                        "type": "output_text", "text": json.dumps(self.id_judgment()),
                    }]}],
                }

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-mock"}), \
                    patch.object(evaluator, "_http_json", side_effect=fake_http):
                result = evaluator.run_evaluation(experiment_dir, budget_usd=0.01, max_pairs=1)
            self.assertEqual(result["evaluated_judged"], 1)
            self.assertEqual(result["stop_reason"], "pair_gate_limit")
            self.assertEqual(calls, [evaluator.API_BASE + "/responses/input_tokens",
                                     evaluator.API_BASE + "/responses"])
            statuses = evaluator._read_jsonl(evaluator_dir / "pair_status.jsonl")
            self.assertEqual([row["status"] for row in statuses], ["evaluated_judged", "not_evaluated"])

    def test_dry_run_writes_local_estimates_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory)
            (experiment_dir / "openai_evaluator").mkdir()
            (experiment_dir / "openai_evaluator" / "evaluator_prompt_frozen.md").write_bytes(
                evaluator.PROMPT_PATH.read_bytes())
            pool = [self.pool_row]
            (experiment_dir / "blinded_evaluation_pool.jsonl").write_text(
                json.dumps(pool[0]) + "\n", encoding="utf-8")
            report = evaluator.dry_run(experiment_dir)
            self.assertEqual(report["token_count_source"], "local_tiktoken_estimate_not_authoritative")
            self.assertEqual(report["pool_pairs"], 1)
            self.assertTrue((experiment_dir / "openai_evaluator" / "dry_run_estimate.json").is_file())
            self.assertFalse((experiment_dir / "openai_evaluator" / "budget_ledger.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()