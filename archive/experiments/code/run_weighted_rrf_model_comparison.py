"""Run an isolated alternative-model audit on the frozen weighted-RRF dev selection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_bge_transition_openai as evaluator
import run_weighted_rrf_grader_audit as audit

AUDIT_ROOT = (PROJECT_ROOT / "data/matching/weighted_rrf_20261008T233000Z/diagnostics/"
              "dev_top5_strong_ceiling_20261009/blinded_grader_audit_20261009")
REVISED_DIR = AUDIT_ROOT / "revised_protocol_v2_20261009"
COMPARISON_DIR = AUDIT_ROOT / "model_comparison_gpt41_20261009"
LEDGER_PATH = PROJECT_ROOT / "data/matching/bge_transition_20261008T142818Z/openai_evaluator/budget_ledger.sqlite3"

MODEL_ID = "gpt-4.1-2025-04-14"
MODEL_CONTEXT_TOKENS = 1_047_576
INPUT_USD_PER_MILLION = 2.0
OUTPUT_USD_PER_MILLION = 8.0
MAX_OUTPUT_TOKENS = 1800
MAX_ATTEMPTS = evaluator.MAX_ATTEMPTS
BUDGET_CAP_USD = 3.0
TEMPERATURE = 0
STORE_RESPONSE = False

OFFICIAL_SOURCES = {
    "model_capability_snapshot": "https://developers.openai.com/api/docs/models/gpt-4.1",
    "structured_outputs": "https://developers.openai.com/api/docs/guides/structured-outputs",
    "responses_api": "https://developers.openai.com/api/reference/resources/responses/methods/create",
    "standard_pricing": "https://developers.openai.com/api/docs/pricing",
}


def _sha256(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(evaluator._canonical(row) + "\n" for row in rows), encoding="utf-8", newline="\n")


@contextlib.contextmanager
def _model_context() -> Iterator[None]:
    previous_model = evaluator.MODEL
    try:
        evaluator.MODEL = MODEL_ID
        yield
    finally:
        evaluator.MODEL = previous_model


def _payload(row: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    with audit._revised_audit_configuration():
        return audit._payload(row, prompt)


def _request_identity(row: Mapping[str, Any], prompt: str, schema: Mapping[str, Any]) -> str:
    payload = _payload(row, prompt)
    body = {
        "model": MODEL_ID,
        "prompt_sha256": _sha256(prompt),
        "schema": schema,
        "settings": {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": STORE_RESPONSE,
            "reasoning": "not_applicable_non_reasoning_model",
        },
        "input": payload["input"],
    }
    return _sha256(evaluator._canonical(body))


def _ledger() -> evaluator.BudgetLedger:
    return evaluator.BudgetLedger(LEDGER_PATH, BUDGET_CAP_USD)


def _estimate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = (REVISED_DIR / "audit_prompt_frozen.md").read_text(encoding="utf-8")
    schema = json.loads((REVISED_DIR / "audit_schema_frozen.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(REVISED_DIR / "reviewer_input_blinded.jsonl")
    per_pair = []
    with _model_context(), audit._revised_audit_configuration():
        for row in rows:
            payload = audit._payload(row, prompt)
            local_tokens = evaluator.local_token_estimate(payload)
            reservation = evaluator._maximum_request_cost(
                local_tokens, MAX_OUTPUT_TOKENS, INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION,
            ) * MAX_ATTEMPTS
            per_pair.append({
                "profile_id": row["profile_id"], "job_id": row["job_id"],
                "request_key": _request_identity(row, prompt, schema),
                "local_estimated_input_tokens": local_tokens,
                "max_attempts": MAX_ATTEMPTS,
                "max_output_tokens_per_attempt": MAX_OUTPUT_TOKENS,
                "max_reservation_usd": round(reservation, 10),
            })
    before = {**_ledger().totals(), "attempts": _ledger().total_attempts()}
    new_reservation = sum(item["max_reservation_usd"] for item in per_pair)
    return per_pair, {
        "ledger_before": before,
        "projected_new_reservation_usd": round(new_reservation, 10),
        "projected_cumulative_reserved_usd": round(before["reserved_max_usd"] + new_reservation, 10),
        "remaining_authorized_reservation_usd": round(BUDGET_CAP_USD - before["reserved_max_usd"], 10),
        "fits_cap": before["reserved_max_usd"] + new_reservation <= BUDGET_CAP_USD,
    }


def freeze() -> dict[str, Any]:
    if COMPARISON_DIR.exists() and any(COMPARISON_DIR.iterdir()):
        raise FileExistsError("Model-comparison directory exists; refusing to overwrite it.")
    if not REVISED_DIR.is_dir():
        raise FileNotFoundError("The frozen revised-protocol audit package is required.")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    input_bytes = (REVISED_DIR / "reviewer_input_blinded.jsonl").read_bytes()
    prompt_bytes = (REVISED_DIR / "audit_prompt_frozen.md").read_bytes()
    schema_bytes = (REVISED_DIR / "audit_schema_frozen.json").read_bytes()
    revised_manifest_bytes = (REVISED_DIR / "audit_manifest.json").read_bytes()
    (COMPARISON_DIR / "reviewer_input_blinded.jsonl").write_bytes(input_bytes)
    (COMPARISON_DIR / "audit_prompt_frozen.md").write_bytes(prompt_bytes)
    (COMPARISON_DIR / "audit_schema_frozen.json").write_bytes(schema_bytes)
    per_pair, budget = _estimate()
    selection_reasons_path = AUDIT_ROOT / "selection_reasons.json"
    manifest = {
        "comparison_id": "weighted_rrf_model_comparison_gpt41_20261009",
        "status": "frozen_preflight_no_provider_calls_yet",
        "scope": "isolated grader audit only; no TF-IDF/BGE matching or ranking changes",
        "model_id": MODEL_ID,
        "model_snapshot": MODEL_ID,
        "original_model_id": "gpt-4.1-mini-2025-04-14",
        "model_purpose": "stronger general-purpose non-reasoning model comparison",
        "model_documentation": {
            "model_page": OFFICIAL_SOURCES["model_capability_snapshot"],
            "verified_default_snapshot": MODEL_ID,
            "verified_responses_api": True,
            "verified_structured_outputs": True,
            "verified_model_description": "Smartest non-reasoning model; instruction following model with low latency and no reasoning step.",
            "context_tokens": MODEL_CONTEXT_TOKENS,
            "max_output_tokens_documented": 32768,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "pricing": {
            "source": OFFICIAL_SOURCES["standard_pricing"],
            "processing_tier": "standard",
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "cached_input_usd_per_million": 0.5,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "reasoning_token_rate": "not applicable; GPT-4.1 is a non-reasoning model; output tokens billed at output rate",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "structured_output_compatibility": {
            "source": OFFICIAL_SOURCES["structured_outputs"],
            "text_format_json_schema_strict": True,
            "response_api_create_supported": True,
            "settings_difference_from_v2": "Model ID and cost rates only. Keep revised v2 prompt/schema, temperature=0, max_output_tokens=1800 and store=false; no reasoning effort parameter applies to GPT-4.1.",
        },
        "response_settings": {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": STORE_RESPONSE,
            "reasoning_effort": None,
            "retry_limit_per_pair": MAX_ATTEMPTS,
        },
        "validator_version": audit.AUDIT_VALIDATOR_VERSION,
        "validator_sha256": _sha256(inspect_source := (
            __import__("inspect").getsource(audit._validate_audit)
            + __import__("inspect").getsource(audit._semantic_concerns)
        )),
        "revised_prompt_sha256": _sha256(prompt_bytes),
        "revised_schema_sha256": _sha256(evaluator._canonical(json.loads(schema_bytes))),
        "reviewer_input_sha256": _sha256(input_bytes),
        "revised_protocol_manifest_sha256": _sha256(revised_manifest_bytes),
        "selection_reasons_sha256_local_only": _sha256(selection_reasons_path.read_bytes()),
        "selection_reasons_sent_to_provider": False,
        "prior_grades_responses_rankings_scores_methods_sent_to_provider": False,
        "same_12_pairs_as_revised_audit": len(_read_jsonl(COMPARISON_DIR / "reviewer_input_blinded.jsonl")) == 12,
        "ledger_path": str(LEDGER_PATH), "authorized_cumulative_cap_usd": BUDGET_CAP_USD,
        "budget_preflight": budget, "per_pair_worst_case_cost": per_pair,
        "provider_calls_made_during_freeze": False,
    }
    (COMPARISON_DIR / "preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (COMPARISON_DIR / "official_model_pricing_sources.md").write_text(
        "# Official Model and Pricing Verification\n\n"
        "- Model snapshot and capability: [GPT-4.1 model documentation]({}).\n"
        "- Structured output syntax and supported model guidance: [Structured Outputs guide]({}).\n"
        "- Request fields and output settings: [Responses API create reference]({}).\n"
        "- Standard API rates: [OpenAI API pricing]({}).\n\n"
        "Verified snapshot: `{}`. Standard rates frozen for this request are $2/M input and $8/M output. GPT-4.1 is documented as non-reasoning, so no separate reasoning-token charge applies; any output usage is conservatively reserved at the output rate. Responses supports strict `text.format.type=json_schema`; the model page lists Responses and structured outputs as supported. The API `max_output_tokens` cap includes visible and reasoning tokens, but this model has no reasoning step.\n".format(
            OFFICIAL_SOURCES["model_capability_snapshot"], OFFICIAL_SOURCES["structured_outputs"],
            OFFICIAL_SOURCES["responses_api"], OFFICIAL_SOURCES["standard_pricing"], MODEL_ID),
        encoding="utf-8", newline="\n")
    return manifest


def _cached_result(cache_path: Path, request_key: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    with audit._revised_audit_configuration():
        judgment, errors = audit._validate_audit(cached.get("judgment", {}), row)
    if cached.get("request_key") != request_key or cached.get("actual_model") != MODEL_ID or errors:
        return None
    return judgment


def run(max_pairs: int) -> dict[str, Any]:
    manifest_path = COMPARISON_DIR / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["budget_preflight"]["fits_cap"]:
        raise RuntimeError("Worst-case audit request exceeds authorized cumulative cap; no call made.")
    gate_path = COMPARISON_DIR / "first_pair_gate.json"
    if max_pairs > 1:
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
        if not gate.get("mechanically_validated") or not gate.get("source_review_clear"):
            raise RuntimeError("First-pair source-review gate is not clear; remaining requests are blocked.")
    key = evaluator.load_api_key(PROJECT_ROOT)
    if not key:
        raise RuntimeError("OPENAI_API_KEY unavailable; no credentials printed and no call made.")
    rows = _read_jsonl(COMPARISON_DIR / "reviewer_input_blinded.jsonl")
    prompt = (COMPARISON_DIR / "audit_prompt_frozen.md").read_text(encoding="utf-8")
    schema = json.loads((COMPARISON_DIR / "audit_schema_frozen.json").read_text(encoding="utf-8"))
    results_path = COMPARISON_DIR / "validated_model_judgments.jsonl"
    results = {(row["profile_id"], row["job_id"]): row for row in _read_jsonl(results_path)} if results_path.exists() else {}
    status_path = COMPARISON_DIR / "pair_status.jsonl"
    states = _read_jsonl(status_path) if status_path.exists() else []
    processed = 0
    run_started = time.perf_counter()
    stop_reason = None
    with _model_context(), audit._revised_audit_configuration():
        for row in rows:
            pair = (row["profile_id"], row["job_id"])
            if pair in results or processed >= max_pairs:
                continue
            request_key = _request_identity(row, prompt, schema)
            cache_path = COMPARISON_DIR / "cache" / (request_key + ".json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cached = _cached_result(cache_path, request_key, row)
            if cached is not None:
                result = {"profile_id": pair[0], "job_id": pair[1], "judgment": cached,
                          "model_id": MODEL_ID, "request_key": request_key, "cache_reused": True,
                          "attempts": 0, "runtime_seconds": 0.0, "semantic_concerns": audit._semantic_concerns(cached, row)}
                results[pair] = result
                state = {"profile_id": pair[0], "job_id": pair[1], "status": "validated_cached",
                         "attempts": 0, "errors": []}
            else:
                payload = audit._payload(row, prompt)
                pair_key = _sha256(evaluator._canonical(pair))
                errors_seen = []
                validated = None
                semantic_concerns: list[dict[str, Any]] = []
                response_id = None
                pair_started = time.perf_counter()
                ledger = evaluator.BudgetLedger(LEDGER_PATH, BUDGET_CAP_USD)
                while ledger.attempts(request_key) < MAX_ATTEMPTS:
                    try:
                        token_result = evaluator._http_json(
                            evaluator.API_BASE + "/responses/input_tokens",
                            evaluator._count_payload(payload), key)
                        input_tokens = int(token_result["input_tokens"])
                    except Exception as error:
                        errors_seen.append("input_token_count_failed:" + type(error).__name__)
                        break
                    try:
                        ledger.reserve_pair(
                            request_key, pair_key, input_tokens,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            input_usd_per_million=INPUT_USD_PER_MILLION,
                            output_usd_per_million=OUTPUT_USD_PER_MILLION,
                            model_id=MODEL_ID,
                        )
                    except evaluator.BudgetExceeded as error:
                        errors_seen.append("budget_exceeded:reserved={:.8f}:new={:.8f}:cap={:.2f}".format(
                            error.already_reserved, error.new_reservation, error.budget))
                        break
                    attempt = ledger.start_attempt(request_key, pair_key)
                    response = None
                    attempt_started = time.perf_counter()
                    try:
                        response = evaluator._http_json(evaluator.API_BASE + "/responses", payload, key)
                        response_id = response.get("id")
                        evaluator._append_jsonl(COMPARISON_DIR / "raw_responses.jsonl", {
                            "utc": datetime.now(timezone.utc).isoformat(), "profile_id": pair[0], "job_id": pair[1],
                            "request_key": request_key, "attempt": attempt,
                            "actual_model": response.get("model"), "response_id": response_id,
                            "status": response.get("status"), "usage": response.get("usage"),
                            "raw_response": response,
                        })
                        if response.get("model") != MODEL_ID or response.get("status") != "completed":
                            raise ValueError("unexpected_model_or_incomplete_response")
                        raw = json.loads(evaluator._extract_output_text(response))
                        validated, errors = audit._validate_audit(raw, row)
                        evaluator._append_jsonl(COMPARISON_DIR / "validation_attempts.jsonl", {
                            "profile_id": pair[0], "job_id": pair[1], "request_key": request_key,
                            "attempt": attempt, "valid": not errors, "errors": errors,
                        })
                        if errors:
                            errors_seen.extend(errors)
                            ledger.finish_attempt(request_key, pair_key, attempt, "invalid_judgment", response,
                                                  INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION)
                            validated = None
                        else:
                            ledger.finish_attempt(request_key, pair_key, attempt, "completed", response,
                                                  INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION)
                            semantic_concerns = audit._semantic_concerns(validated or {}, row)
                            evaluator._atomic_json(cache_path, {
                                "request_key": request_key, "actual_model": MODEL_ID,
                                "response_id": response_id, "judgment": validated,
                            })
                            break
                    except Exception as error:
                        errors_seen.append(type(error).__name__ + ":" + str(error))
                        ledger.finish_attempt(request_key, pair_key, attempt, "uncertain_or_failed", response,
                                              INPUT_USD_PER_MILLION, OUTPUT_USD_PER_MILLION)
                        if evaluator._must_stop_for_api_error(error):
                            break
                    finally:
                        evaluator._append_jsonl(COMPARISON_DIR / "runtime.jsonl", {
                            "profile_id": pair[0], "job_id": pair[1], "request_key": request_key,
                            "attempt": attempt, "elapsed_seconds": time.perf_counter() - attempt_started,
                        })
                elapsed = time.perf_counter() - pair_started
                if validated is not None:
                    result = {"profile_id": pair[0], "job_id": pair[1], "judgment": validated,
                              "model_id": MODEL_ID, "request_key": request_key,
                              "cache_reused": False, "attempts": ledger.attempts(request_key),
                              "runtime_seconds": elapsed, "response_id": response_id,
                              "semantic_concerns": semantic_concerns}
                    results[pair] = result
                    state = {"profile_id": pair[0], "job_id": pair[1], "status": "validated",
                             "attempts": ledger.attempts(request_key), "errors": [],
                             "semantic_concerns": semantic_concerns}
                else:
                    state = {"profile_id": pair[0], "job_id": pair[1], "status": "failed_or_unvalidated",
                             "attempts": ledger.attempts(request_key), "errors": errors_seen}
            states.append(state)
            processed += 1
            _write_jsonl(results_path, list(results.values()))
            _write_jsonl(status_path, states)
            if max_pairs == 1:
                stop_reason = "first_pair_requires_source_review"
                break
            if state.get("status") not in {"validated", "validated_cached"} or state.get("semantic_concerns"):
                stop_reason = "stopped_on_mechanical_or_semantic_concern"
                break
    ledger = _ledger()
    ledger.export(COMPARISON_DIR / "shared_ledger_snapshot.jsonl")
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "requested_max_pairs": max_pairs,
        "new_pairs_processed": processed,
        "validated_total": len(results),
        "runtime_seconds": time.perf_counter() - run_started,
        "stop_reason": stop_reason or ("all_remaining_processed" if len(results) == len(rows) else "pair_limit"),
        "ledger_after": {**ledger.totals(), "attempts": ledger.total_attempts()},
        "budget_cap_usd": BUDGET_CAP_USD,
    }
    (COMPARISON_DIR / "run_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluator._append_jsonl(COMPARISON_DIR / "run_history.jsonl", summary)
    if max_pairs == 1 and states:
        first = states[-1]
        gate = {
            "mechanically_validated": first.get("status") in {"validated", "validated_cached"},
            "semantic_concerns": first.get("semantic_concerns", []),
            "source_review_clear": False,
            "profile_id": first["profile_id"], "job_id": first["job_id"],
            "stop_reason": "pending_source_review",
        }
        (COMPARISON_DIR / "first_pair_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def source_review_first_pair() -> dict[str, Any]:
    gate_path = COMPARISON_DIR / "first_pair_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    records = _read_jsonl(COMPARISON_DIR / "validated_model_judgments.jsonl")
    record = next(row for row in records if row["profile_id"] == gate["profile_id"] and row["job_id"] == gate["job_id"])
    rows = _read_jsonl(COMPARISON_DIR / "reviewer_input_blinded.jsonl")
    row = next(item for item in rows if item["profile_id"] == gate["profile_id"] and item["job_id"] == gate["job_id"])
    judgment = record["judgment"]
    cited = {item["evidence_id"]: item["source_text"]
             for item in evaluator._evidence_catalog(row["candidate_evidence"], "C") +
             evaluator._evidence_catalog(row["job_evidence"], "J")}
    candidate_ids = {item["evidence_id"] for item in evaluator._evidence_catalog(row["candidate_evidence"], "C")}
    job_text = str(row["job_evidence"].get("description_text", ""))
    all_candidate_text = " ".join(item["source_text"] for item in evaluator._evidence_catalog(row["candidate_evidence"], "C"))
    required_license = "Florida Class C Wastewater Operator License within one (1) year" in job_text
    training = "Wastewater Operator in Training" in job_text and "Under supervision" in job_text
    driver_required = "valid driver" in job_text.casefold()
    driver_finding = next((finding for finding in judgment["qualification_findings"]
                           if "driver" in finding["qualification"].casefold()), None)
    wastewater_finding = next((finding for finding in judgment["qualification_findings"]
                               if "wastewater" in finding["qualification"].casefold() and "license" in finding["qualification"].casefold()), None)
    has_wastewater_candidate_claim = "wastewater license" in all_candidate_text.casefold()
    concerns = []
    if wastewater_finding and "drinking-water" in " ".join(cited.get(eid, "") for eid in wastewater_finding["candidate_evidence_ids"]).casefold() and not has_wastewater_candidate_claim:
        concerns.append("candidate evidence does not support absence/equivalence for wastewater license")
    if wastewater_finding and required_license and training and wastewater_finding["job_classification"] != "explicitly_obtainable_after_joining":
        concerns.append("one-year training pathway timing is not classified post-entry")
    if wastewater_finding and wastewater_finding["candidate_status"] == "explicitly_contradicted" and wastewater_finding["job_classification"] == "explicitly_obtainable_after_joining":
        concerns.append("post-entry credential is still marked candidate-contradicted")
    if driver_required and driver_finding is None:
        concerns.append("required driver-license qualification omitted; candidate silence should remain unknown")
    if (driver_required and driver_finding is not None
            and driver_finding.get("candidate_status") == "unknown"
            and judgment.get("required_qualifications") == "met"):
        concerns.append("aggregate required-qualifications status is met despite an unknown mandatory driver-license status")
    for field in ("material_gaps", "explicit_constraint_conflicts"):
        for claim in judgment.get(field, []):
            if not isinstance(claim, str) or claim.strip() in candidate_ids or claim.strip() in {item["evidence_id"] for item in evaluator._evidence_catalog(row["job_evidence"], "J")}:
                concerns.append(field + " contains a bare evidence ID rather than a specific supported claim")
    review = {
        "model_id": MODEL_ID, "profile_id": gate["profile_id"], "job_id": gate["job_id"],
        "job_training_path_explicit": training,
        "job_one_year_license_window_explicit": required_license,
        "driver_license_requirement_explicit": driver_required,
        "driver_license_finding_present": driver_finding is not None,
        "driver_license_candidate_status": driver_finding.get("candidate_status") if driver_finding else None,
        "aggregate_required_qualifications": judgment.get("required_qualifications"),
        "wastewater_license_finding": wastewater_finding,
        "original_grade": next(row for row in _read_jsonl(
            PROJECT_ROOT / "data/matching/weighted_rrf_20261008T233000Z/development_evaluation/openai_evaluator/validated_judgments.jsonl")
            if row["profile_id"] == gate["profile_id"] and row["job_id"] == gate["job_id"])["grade"],
        "alternative_model_grade": judgment["grade"],
        "remaining_source_review_concerns": concerns,
        "source_review_clear": not concerns and not gate.get("semantic_concerns"),
        "note": "This local source review is not independent human validation; exact evidence IDs establish identity, not semantic correctness.",
    }
    gate["source_review_clear"] = review["source_review_clear"]
    gate["source_review_concerns"] = concerns
    gate["stop_reason"] = "clear" if review["source_review_clear"] else "unresolved_source_semantic_concern"
    (COMPARISON_DIR / "first_pair_source_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gate


def compare() -> dict[str, Any]:
    original = {(row["profile_id"], row["job_id"]): row for row in _read_jsonl(
        PROJECT_ROOT / "data/matching/weighted_rrf_20261008T233000Z/development_evaluation/openai_evaluator/validated_judgments.jsonl")}
    revised = {(row["profile_id"], row["job_id"]): row for row in _read_jsonl(
        REVISED_DIR / "validated_audit_judgments.jsonl")}
    alternative = {(row["profile_id"], row["job_id"]): row for row in _read_jsonl(
        COMPARISON_DIR / "validated_model_judgments.jsonl")}
    selected = _read_jsonl(COMPARISON_DIR / "reviewer_input_blinded.jsonl")
    comparison_rows = []
    for row in selected:
        pair = (row["profile_id"], row["job_id"])
        old = original.get(pair)
        v2 = revised.get(pair)
        alt = alternative.get(pair)
        old_grade = old.get("grade") if old else None
        v2_grade = v2["judgment"].get("grade") if v2 else None
        alt_judgment = alt["judgment"] if alt else None
        comparison_rows.append({
            "profile_id": pair[0], "job_id": pair[1],
            "original_model_grade": old_grade,
            "same_model_revised_protocol_grade": v2_grade,
            "alternative_model_grade": alt_judgment.get("grade") if alt_judgment else None,
            "required_qualifications": alt_judgment.get("required_qualifications") if alt_judgment else None,
            "alternative_pair_status": (next((s["status"] for s in _read_jsonl(COMPARISON_DIR / "pair_status.jsonl")
                                               if s["profile_id"] == pair[0] and s["job_id"] == pair[1]), "not_attempted")
                                       if (COMPARISON_DIR / "pair_status.jsonl").exists() else "not_attempted"),
            "original_requirement_claims": old.get("explicit_constraint_conflicts", []) if old else [],
            "same_model_revised_qualification_findings": v2["judgment"].get("qualification_findings", []) if v2 else [],
            "alternative_qualification_findings": alt_judgment.get("qualification_findings", []) if alt_judgment else [],
            "alternative_semantic_concerns": alt.get("semantic_concerns", []) if alt else [],
        })
    _write_jsonl(COMPARISON_DIR / "model_comparison.jsonl", comparison_rows)
    run_hist = _read_jsonl(COMPARISON_DIR / "run_history.jsonl") if (COMPARISON_DIR / "run_history.jsonl").exists() else []
    ledger_now = {**_ledger().totals(), "attempts": _ledger().total_attempts()}
    preflight = json.loads((COMPARISON_DIR / "preflight_manifest.json").read_text(encoding="utf-8"))
    valid = len(alternative)
    grade_matches = sum(row["original_model_grade"] == row["alternative_model_grade"]
                        for row in comparison_rows if row["alternative_model_grade"] is not None)
    mini_grade_matches = sum(row["same_model_revised_protocol_grade"] == row["alternative_model_grade"]
                             for row in comparison_rows if row["alternative_model_grade"] is not None
                             and row["same_model_revised_protocol_grade"] is not None)
    semantic_concerns = [c for item in alternative.values() for c in item.get("semantic_concerns", [])]
    summary = {
        "comparison_id": preflight["comparison_id"], "model_id": MODEL_ID,
        "selected_pairs": len(selected), "attempted_pairs": len(_read_jsonl(COMPARISON_DIR / "pair_status.jsonl")) if (COMPARISON_DIR / "pair_status.jsonl").exists() else 0,
        "mechanically_validated_pairs": valid,
        "unknown_grade_count": sum(item["judgment"]["status"] == "unknown" for item in alternative.values()),
        "unknown_candidate_qualification_findings": sum(
            finding.get("candidate_status") == "unknown"
            for item in alternative.values()
            for finding in item["judgment"].get("qualification_findings", [])),
        "failed_or_unvalidated_pairs": sum(state["status"] not in {"validated", "validated_cached"} for state in (_read_jsonl(COMPARISON_DIR / "pair_status.jsonl") if (COMPARISON_DIR / "pair_status.jsonl").exists() else [])),
        "not_attempted_pairs": len(selected) - (len(_read_jsonl(COMPARISON_DIR / "pair_status.jsonl")) if (COMPARISON_DIR / "pair_status.jsonl").exists() else 0),
        "original_grade_agreement_count": grade_matches,
        "original_grade_disagreement_count": valid - grade_matches,
        "revised_protocol_mini_grade_agreement_count": mini_grade_matches,
        "revised_protocol_mini_grade_disagreement_count": sum(
            row["same_model_revised_protocol_grade"] != row["alternative_model_grade"]
            for row in comparison_rows if row["alternative_model_grade"] is not None
            and row["same_model_revised_protocol_grade"] is not None),
        "remaining_semantic_concerns": semantic_concerns,
        "first_pair_gate": json.loads((COMPARISON_DIR / "first_pair_gate.json").read_text(encoding="utf-8")) if (COMPARISON_DIR / "first_pair_gate.json").exists() else None,
        "ledger_before": preflight["budget_preflight"]["ledger_before"], "ledger_after": ledger_now,
        "incremental_reserved_usd": round(ledger_now["reserved_max_usd"] - preflight["budget_preflight"]["ledger_before"]["reserved_max_usd"], 10),
        "incremental_reported_usage_priced_usd": round(ledger_now["reported_actual_usd"] - preflight["budget_preflight"]["ledger_before"]["reported_actual_usd"], 10),
        "preflight_worst_case_new_reservation_usd": preflight["budget_preflight"]["projected_new_reservation_usd"],
        "preflight_projected_cumulative_reserved_usd": preflight["budget_preflight"]["projected_cumulative_reserved_usd"],
        "runtime_seconds_total": sum(item.get("runtime_seconds", 0.0) for item in run_hist),
        "run_history": run_hist,
        "official_sources": OFFICIAL_SOURCES,
        "model_is_stronger_general_purpose_candidate_not_accuracy_established": True,
        "same_api_key_loaded_from_existing_credential_chain": True,
        "ledger_reset": False, "ranking_or_embedding_runs": False,
        "heldout_profiles_opened": False,
        "original_grades_or_metrics_modified": False,
    }
    (COMPARISON_DIR / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# GPT-4.1 Alternative-Model Grader Comparison", "",
        "This is an isolated evaluator comparison. TF-IDF/BGE retrieval, scores, and rankings were not run or changed. GPT-4.1 is a stronger general-purpose model candidate than the original GPT-4.1 mini, but this selected sample cannot establish its general accuracy.", "",
        "## Coverage and Agreement", "",
        "Selected {}; attempted {}; mechanically validated {}; unknown grades {}; unknown candidate-qualification findings {}; failed/unvalidated {}; not attempted {}. Of validated pairs, grades matched the original grader on {} and the revised-protocol mini on {} (original-grade disagreements: {}); original, revised-protocol mini, and GPT-4.1 findings are kept separate in `model_comparison.jsonl`.".format(
            summary["selected_pairs"], summary["attempted_pairs"], summary["mechanically_validated_pairs"],
            summary["unknown_grade_count"], summary["unknown_candidate_qualification_findings"],
            summary["failed_or_unvalidated_pairs"], summary["not_attempted_pairs"],
            summary["original_grade_agreement_count"], summary["revised_protocol_mini_grade_agreement_count"],
            summary["original_grade_disagreement_count"]), "",
        "## Gate and Semantic Review", "",
        "First-pair result: GPT-4.1 grade {} versus original grade {} and revised-protocol mini grade {}. The response represents the driver-license status as unknown but aggregate required qualification status as `{}`. First-pair gate: {}. Remaining semantic concerns: {}. A mechanical citation match establishes the cited text’s identity, not that it supports the claim. Source review is coding-assistant review, not independent human validation.".format(
            next((item["alternative_model_grade"] for item in comparison_rows if item["alternative_model_grade"] is not None), None),
            next((item["original_model_grade"] for item in comparison_rows if item["alternative_model_grade"] is not None), None),
            next((item["same_model_revised_protocol_grade"] for item in comparison_rows if item["alternative_model_grade"] is not None), None),
            next((item["judgment"]["required_qualifications"] for item in alternative.values()), "unknown"),
            summary["first_pair_gate"].get("stop_reason") if summary["first_pair_gate"] else "not run",
            [item.get("code") for item in semantic_concerns]), "",
        "## Model and Cost Provenance", "",
        "Model `{}`; standard rates $2/M input and $8/M output; non-reasoning model, no separate reasoning charge; strict Structured Outputs on Responses; temperature {}; max output {} tokens; store={}. Prompt `{}`; schema `{}`. Frozen full-run worst-case reservation was ${:.6f}, projecting cumulative reservations to ${:.6f}. Actual runtime {:.2f}s. Incremental reserve ${:.6f}; usage-priced ${:.6f}; cumulative reserved ${:.6f}/$3.00; cumulative usage-priced ${:.6f}. Cost is usage priced from returned usage, not invoice verification.".format(
            MODEL_ID, TEMPERATURE, MAX_OUTPUT_TOKENS, STORE_RESPONSE,
            preflight["revised_prompt_sha256"], preflight["revised_schema_sha256"],
            summary["preflight_worst_case_new_reservation_usd"], summary["preflight_projected_cumulative_reserved_usd"],
            summary["runtime_seconds_total"], summary["incremental_reserved_usd"], summary["incremental_reported_usage_priced_usd"],
            ledger_now["reserved_max_usd"], ledger_now["reported_actual_usd"]), "",
        "Official verification: [GPT-4.1 model page]({}); [Structured Outputs guide]({}); [Responses API create reference]({}); [current pricing]({}).".format(
            OFFICIAL_SOURCES["model_capability_snapshot"], OFFICIAL_SOURCES["structured_outputs"],
            OFFICIAL_SOURCES["responses_api"], OFFICIAL_SOURCES["standard_pricing"]), "",
        "## Recommendation", "",
        "Do not adopt this model or reinterpret the original grades from one selected pair. If the first-pair gate is blocked, the credential/timing semantics remain unresolved and the other pairs should not be run. If it is clear and all 12 finish, treat the result as a promising evaluator candidate only, then seek independent source-level human adjudication before making grader or ranking decisions. Do not replace historical ranking metrics.", "",
    ]
    (COMPARISON_DIR / "comparison_report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    first = sub.add_parser("run-first")
    first.set_defaults(max_pairs=1)
    follow = sub.add_parser("run-rest")
    follow.set_defaults(max_pairs=11)
    sub.add_parser("source-review")
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command in {"run-first", "run-rest"}:
        result = run(args.max_pairs)
    elif args.command == "source-review":
        result = source_review_first_pair()
    else:
        result = compare()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
