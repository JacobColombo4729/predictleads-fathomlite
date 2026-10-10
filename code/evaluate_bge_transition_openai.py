"""Run a cost-bounded, blinded OpenAI evaluator for the saved BGE transition pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
EXPERIMENT_ID = "bge_transition_20261008T142818Z"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "matching" / EXPERIMENT_ID
EVALUATOR_DIR = EXPERIMENT_DIR / "openai_evaluator"
PROMPT_PATH = EVALUATOR_DIR / "evaluator_prompt_frozen.md"
FROZEN_PROMPT_SHA256 = "2DC3EDDA4FE52C8AD5874D11727EEAC5FFDA49DBE33FEA705EF1FE02A119FD11"
MODEL = "gpt-4.1-mini-2025-04-14"
MODEL_CONTEXT_TOKENS = 1_047_576
MAX_OUTPUT_TOKENS = 512
MAX_ATTEMPTS = 2
DEFAULT_BUDGET_USD = 2.0
INPUT_USD_PER_MILLION = 0.40
OUTPUT_USD_PER_MILLION = 1.60
API_BASE = "https://api.openai.com/v1"
FORBIDDEN_KEYS = {"method", "rank", "score", "score_type", "similarity"}

JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "profile_id": {"type": "string"},
        "job_id": {"type": "string"},
        "grade": {"type": ["integer", "null"], "enum": [0, 1, 2, 3, None]},
        "status": {"type": "string", "enum": ["judged", "unknown"]},
        "unknown_reason": {"type": ["string", "null"]},
        "supporting_candidate_evidence_ids": {
            "type": "array",
            "description": "Select only evidence_id values from candidate_evidence. Each ID identifies exact source text supplied in this request. Do not invent IDs or cite evidence absent from the supplied catalog.",
            "items": {"type": "string"},
        },
        "supporting_job_evidence_ids": {
            "type": "array",
            "description": "Select only evidence_id values from job_evidence. Each ID identifies exact source text supplied in this request. Do not invent IDs or cite evidence absent from the supplied catalog.",
            "items": {"type": "string"},
        },
        "material_gaps": {"type": "array", "items": {"type": "string"}},
        "explicit_constraint_conflicts": {"type": "array", "items": {"type": "string"}},
        "required_qualifications": {"type": "string", "enum": ["met", "unmet", "unknown"]},
        "career_transition_relevance": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "level": {"type": "string", "enum": ["low", "moderate", "high", "unknown"]},
                "rationale": {"type": "string"},
            },
            "required": ["level", "rationale"],
        },
        "title_outside_stated_targets": {"type": "boolean"},
        "evidence_sufficiency": {"type": "string", "enum": ["sufficient", "insufficient"]},
    },
    "required": [
        "profile_id", "job_id", "grade", "status", "unknown_reason",
        "supporting_candidate_evidence_ids", "supporting_job_evidence_ids", "material_gaps",
        "explicit_constraint_conflicts", "required_qualifications", "career_transition_relevance",
        "title_outside_stated_targets", "evidence_sufficiency",
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_api_key(project_root: Path = PROJECT_ROOT) -> str | None:
    """Use the process environment first, then the repository-local ignored .env."""
    process_key = os.environ.get("OPENAI_API_KEY")
    if process_key:
        return process_key
    load_dotenv(dotenv_path=project_root / ".env", override=False)
    return os.environ.get("OPENAI_API_KEY")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def frozen_prompt_text(path: Path = PROMPT_PATH) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    canonical_crlf = text.replace("\n", "\r\n")
    actual = _sha256(canonical_crlf.encode("utf-8"))
    if actual.upper() != FROZEN_PROMPT_SHA256:
        raise ValueError("Recovered frozen prompt does not match the pre-retrieval SHA-256.")
    return text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, "".join(_canonical(row) + "\n" for row in rows))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(_canonical(row) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _walk_string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _walk_string_leaves(item)]
    if isinstance(value, list):
        return [text for item in value for text in _walk_string_leaves(item)]
    return []


def _evidence_catalog(value: Any, prefix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, str):
            if node.strip():
                rows.append({
                    "evidence_id": "{}{:04d}".format(prefix, len(rows) + 1),
                    "source_path": path,
                    "source_text": node,
                })
        elif isinstance(node, Mapping):
            for key, child in node.items():
                visit(child, "{}/{}".format(path, key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, "{}/{}".format(path, index))

    visit(value, prefix)
    return rows


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).casefold() in FORBIDDEN_KEYS or _contains_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _source_excerpt(reference: str, source: str) -> tuple[str | None, str | None]:
    candidate = reference.strip()
    exact_start = source.find(candidate)
    if exact_start >= 0:
        return source[exact_start:exact_start + len(candidate)], "exact"

    folded_parts: list[str] = []
    source_positions: list[int] = []
    for source_index, character in enumerate(source):
        folded_character = character.casefold()
        folded_parts.append(folded_character)
        source_positions.extend([source_index] * len(folded_character))
    folded_source = "".join(folded_parts)
    folded_candidate = candidate.casefold()
    folded_start = folded_source.find(folded_candidate)
    if folded_start < 0 or not folded_candidate:
        return None, None
    source_start = source_positions[folded_start]
    source_end = source_positions[folded_start + len(folded_candidate) - 1] + 1
    excerpt = source[source_start:source_end]
    if excerpt.casefold() == folded_candidate:
        return excerpt, "case_only"
    return None, None


def build_response_payload(row: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    if _contains_forbidden_key(row):
        raise ValueError("Blinded evaluation input contains a ranking-specific field.")
    content = {
        "profile_id": str(row["profile_id"]),
        "job_id": str(row["job_id"]),
        "candidate_evidence": _evidence_catalog(row["candidate_evidence"], "C"),
        "job_evidence": _evidence_catalog(row["job_evidence"], "J"),
    }
    protocol_addendum = (
        "\n\nEvaluator protocol amendment: supporting_candidate_evidence_ids and "
        "supporting_job_evidence_ids must contain only evidence_id values from the corresponding "
        "supplied evidence catalog. Do not reproduce or paraphrase quotations. IDs resolve to the "
        "exact original source text and are checked locally."
    )
    return {
        "model": MODEL,
        "instructions": prompt + protocol_addendum,
        "input": [{"role": "user", "content": _canonical(content)}],
        "text": {"format": {
            "type": "json_schema", "name": "job_relevance_judgment", "strict": True,
            "schema": JUDGMENT_SCHEMA,
        }},
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


def request_identity(row: Mapping[str, Any], prompt: str) -> str:
    payload = build_response_payload(row, prompt)
    identity = {
        "model": MODEL,
        "prompt_sha256": _sha256(prompt.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"),
        "schema": JUDGMENT_SCHEMA,
        "settings": {"temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS, "store": False},
        "input": payload["input"],
    }
    return _sha256(_canonical(identity))


def local_token_estimate(payload: Mapping[str, Any]) -> int:
    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    material = "\n".join((
        str(payload.get("instructions", "")),
        _canonical(payload.get("input", [])),
        _canonical(payload.get("text", {})),
    ))
    return len(encoding.encode(material, disallowed_special=()))


def _maximum_request_cost(input_tokens: int, output_tokens: int = MAX_OUTPUT_TOKENS,
                          input_usd_per_million: float = INPUT_USD_PER_MILLION,
                          output_usd_per_million: float = OUTPUT_USD_PER_MILLION) -> float:
    return (input_tokens * input_usd_per_million + output_tokens * output_usd_per_million) / 1_000_000


def validate_judgment(raw: Mapping[str, Any], pool_row: Mapping[str, Any],
                      require_evidence_ids: bool = False) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    common = {
        "profile_id", "job_id", "grade", "status", "unknown_reason",
        "material_gaps",
        "explicit_constraint_conflicts", "required_qualifications", "career_transition_relevance",
        "title_outside_stated_targets", "evidence_sufficiency",
    }
    id_fields = {"supporting_candidate_evidence_ids", "supporting_job_evidence_ids"}
    quote_fields = {"supporting_candidate_evidence", "supporting_job_evidence"}
    uses_ids = id_fields.issubset(raw) and not (quote_fields & set(raw))
    uses_quotes = quote_fields.issubset(raw) and not (id_fields & set(raw))
    if require_evidence_ids and not uses_ids:
        errors.append("evidence_id_protocol_required")
    if set(raw) != common | (id_fields if uses_ids else quote_fields if uses_quotes else set()):
        errors.append("schema_fields_mismatch")
    if str(raw.get("profile_id")) != str(pool_row["profile_id"]):
        errors.append("profile_id_mismatch")
    if str(raw.get("job_id")) != str(pool_row["job_id"]):
        errors.append("job_id_mismatch")
    status = raw.get("status")
    grade = raw.get("grade")
    if status not in {"judged", "unknown"}:
        errors.append("invalid_status")
    if status == "judged" and (isinstance(grade, bool) or grade not in {0, 1, 2, 3}):
        errors.append("judged_requires_numeric_grade")
    if status == "unknown" and grade is not None:
        errors.append("unknown_requires_null_grade")
    if status == "unknown" and (not isinstance(raw.get("unknown_reason"), str) or not raw["unknown_reason"].strip()):
        errors.append("unknown_requires_reason")
    if status == "judged" and raw.get("unknown_reason") is not None:
        errors.append("judged_unknown_reason_must_be_null")
    if status == "unknown" and raw.get("evidence_sufficiency") != "insufficient":
        errors.append("unknown_requires_insufficient_evidence")
    if raw.get("required_qualifications") not in {"met", "unmet", "unknown"}:
        errors.append("invalid_required_qualifications")
    relevance = raw.get("career_transition_relevance")
    if not isinstance(relevance, Mapping) or set(relevance) != {"level", "rationale"}:
        errors.append("invalid_transition_relevance")
    elif relevance.get("level") not in {"low", "moderate", "high", "unknown"} or not isinstance(relevance.get("rationale"), str):
        errors.append("invalid_transition_relevance_values")
    if not isinstance(raw.get("title_outside_stated_targets"), bool):
        errors.append("invalid_title_target_flag")
    if raw.get("evidence_sufficiency") not in {"sufficient", "insufficient"}:
        errors.append("invalid_evidence_sufficiency")

    candidate_refs = raw.get("supporting_candidate_evidence_ids" if uses_ids else "supporting_candidate_evidence")
    job_refs = raw.get("supporting_job_evidence_ids" if uses_ids else "supporting_job_evidence")
    gaps = raw.get("material_gaps")
    conflicts = raw.get("explicit_constraint_conflicts")
    normalized = dict(raw)
    for field, values in (("supporting_candidate_evidence", candidate_refs),
                          ("supporting_job_evidence", job_refs),
                          ("material_gaps", gaps), ("explicit_constraint_conflicts", conflicts)):
        if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
            errors.append("invalid_" + field)
    for field, values, evidence_key, error_code, id_field, prefix in (
        ("supporting_candidate_evidence", candidate_refs, "candidate_evidence", "candidate_excerpt_not_exact",
         "supporting_candidate_evidence_ids", "C"),
        ("supporting_job_evidence", job_refs, "job_evidence", "job_excerpt_not_exact",
         "supporting_job_evidence_ids", "J"),
    ):
        if not isinstance(values, list):
            continue
        sources = _walk_string_leaves(pool_row[evidence_key])
        catalog = _evidence_catalog(pool_row[evidence_key], prefix)
        source_by_id = {item["evidence_id"]: item["source_text"] for item in catalog}
        canonical_references: list[str] = []
        unmatched = False
        for reference in values:
            if not isinstance(reference, str):
                canonical_references.append(reference)
                continue
            if uses_ids:
                source_text = source_by_id.get(reference)
                if source_text is None:
                    unmatched = True
                    canonical_references.append(reference)
                else:
                    canonical_references.append(source_text)
                continue
            excerpt = next((match for source in sources
                            for match, match_kind in [_source_excerpt(reference, source)]
                            if match is not None), None)
            if excerpt is None:
                unmatched = True
                canonical_references.append(reference)
            else:
                canonical_references.append(excerpt)
        normalized[field] = canonical_references
        if uses_ids:
            normalized.pop(id_field, None)
        if unmatched:
            errors.append(error_code)
    if isinstance(conflicts, list) and conflicts and (not candidate_refs or not job_refs):
        errors.append("conflicts_require_candidate_and_job_excerpts")
    if status == "judged" and raw.get("evidence_sufficiency") == "sufficient" and (not candidate_refs or not job_refs):
        errors.append("judged_requires_candidate_and_job_excerpts")
    if errors:
        return None, errors
    return normalized, []


def _http_json(url: str, body: Mapping[str, Any], api_key: str, timeout: float = 90.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=_canonical(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        error_code = None
        try:
            error_body = json.loads(detail)
            api_error = error_body.get("error", {})
            error_code = api_error.get("code") or api_error.get("type")
        except (json.JSONDecodeError, AttributeError):
            pass
        raise OpenAIRequestError(error.code, error_code, detail) from None


def _count_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in ("model", "instructions", "input", "text")}


def _extract_output_text(response: Mapping[str, Any]) -> str:
    collected: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                collected.append(str(content.get("text", "")))
            elif isinstance(content, Mapping) and content.get("type") == "refusal":
                raise ValueError("model_refusal")
    return "\n".join(collected)


class OpenAIRequestError(RuntimeError):
    def __init__(self, status_code: int, error_code: str | None, detail: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        super().__init__("OpenAI HTTP {}{}: {}".format(
            status_code, " " + error_code if error_code else "", detail))


def _must_stop_for_api_error(error: BaseException) -> bool:
    if not isinstance(error, OpenAIRequestError):
        return False
    return error.status_code in {401, 403} or error.error_code in {
        "insufficient_quota", "credit_balance_exhausted",
    }


def recover_saved_responses(experiment_dir: Path) -> dict[str, Any]:
    """Recover only complete raw responses whose every citation can be source-grounded."""
    evaluator_dir = experiment_dir / "openai_evaluator"
    pool = _read_jsonl(experiment_dir / "blinded_evaluation_pool.jsonl")
    pool_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in pool}
    response_path = evaluator_dir / "raw_responses.jsonl"
    responses = _read_jsonl(response_path) if response_path.is_file() else []
    syntax = JUDGMENT_SCHEMA
    del syntax
    recovered_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    failures: dict[str, int] = {}
    json_parse_failures = 0
    wrong_model_or_status = 0
    evidence_normalizations = {"candidate_case_only": 0, "job_case_only": 0}
    for response_record in responses:
        pair = (str(response_record.get("profile_id")), str(response_record.get("job_id")))
        pool_row = pool_by_pair.get(pair)
        response = response_record.get("raw_response")
        if pool_row is None or not isinstance(response, Mapping):
            failures["response_identity_or_body_invalid"] = failures.get("response_identity_or_body_invalid", 0) + 1
            continue
        if response.get("model") != MODEL or response.get("status") != "completed":
            wrong_model_or_status += 1
            continue
        try:
            raw_judgment = json.loads(_extract_output_text(response))
        except (ValueError, json.JSONDecodeError):
            json_parse_failures += 1
            continue
        candidate_sources = _walk_string_leaves(pool_row["candidate_evidence"])
        job_sources = _walk_string_leaves(pool_row["job_evidence"])
        for field, sources, count_key in (
            ("supporting_candidate_evidence", candidate_sources, "candidate_case_only"),
            ("supporting_job_evidence", job_sources, "job_case_only"),
        ):
            for reference in raw_judgment.get(field, []) if isinstance(raw_judgment.get(field), list) else []:
                if not isinstance(reference, str):
                    continue
                matches = [_source_excerpt(reference, source) for source in sources]
                if any(kind == "case_only" for _, kind in matches):
                    evidence_normalizations[count_key] += 1
        judgment, errors = validate_judgment(raw_judgment, pool_row)
        if errors:
            for error in errors:
                failures[error] = failures.get(error, 0) + 1
            continue
        recovered_by_pair[pair] = {
            "profile_id": pair[0], "job_id": pair[1],
            "request_key": response_record.get("request_key"),
            "response_id": response_record.get("response_id"),
            "actual_model": response_record.get("actual_model"),
            "judgment": judgment,
        }

    recovery_path = evaluator_dir / "recovered_judgments.jsonl"
    existing = _read_jsonl(recovery_path) if recovery_path.is_file() else []
    existing_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in existing}
    for pair, record in recovered_by_pair.items():
        previous = existing_by_pair.get(pair)
        if previous is not None and previous.get("judgment") != record["judgment"]:
            raise ValueError("Refusing to replace a previously recovered judgment for {} / {}.".format(*pair))
        existing_by_pair[pair] = previous or record
    recovered = list(existing_by_pair.values())
    _atomic_jsonl(recovery_path, recovered)
    report = {
        "created_utc": _utc_now(), "raw_response_records": len(responses),
        "json_parse_failures": json_parse_failures, "wrong_model_or_incomplete_responses": wrong_model_or_status,
        "validation_failure_counts": failures, "case_only_citations_canonicalized": evidence_normalizations,
        "newly_recovered_judgments": len(recovered_by_pair), "total_recovered_judgments": len(recovered),
        "numeric_judgments": sum(row["judgment"].get("status") == "judged" and row["judgment"].get("grade") in {0, 1, 2, 3} for row in recovered),
        "unknown_judgments": sum(row["judgment"].get("status") == "unknown" and row["judgment"].get("grade") is None for row in recovered),
        "raw_responses_modified": False,
        "grounding_rule": "Every evidence reference must be an exact source substring; case-only differences are replaced with the matching exact source substring. No paraphrase, fuzzy match, or unsupported citation is accepted.",
    }
    _atomic_json(evaluator_dir / "response_validation_audit.json", report)
    return report


class BudgetLedger:
    """SQLite-backed append-only reservation and attempt ledger."""

    def __init__(self, path: Path, budget_usd: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.budget_usd = budget_usd
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS reservations (
                    request_key TEXT PRIMARY KEY,
                    pair_key TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    max_output_tokens INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    reserve_usd REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    request_key TEXT NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    started_utc TEXT NOT NULL,
                    finished_utc TEXT,
                    status TEXT NOT NULL,
                    response_id TEXT,
                    actual_model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    actual_cost_usd REAL,
                    PRIMARY KEY (request_key, attempt_no)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    pair_key TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve_pair(self, request_key: str, pair_key: str, input_tokens: int,
                     max_output_tokens: int | None = None,
                     input_usd_per_million: float = INPUT_USD_PER_MILLION,
                     output_usd_per_million: float = OUTPUT_USD_PER_MILLION,
                     model_id: str = MODEL) -> float:
        output_cap = MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
        amount = _maximum_request_cost(input_tokens, output_cap, input_usd_per_million,
                                       output_usd_per_million) * MAX_ATTEMPTS
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT reserve_usd, max_output_tokens FROM reservations WHERE request_key = ?", (request_key,)
            ).fetchone()
            if previous:
                if amount > previous[0] + 1e-12 or output_cap > previous[1]:
                    delta = amount - previous[0]
                    total = connection.execute("SELECT COALESCE(SUM(reserve_usd), 0) FROM reservations").fetchone()[0]
                    if total + delta > self.budget_usd + 1e-12:
                        raise BudgetExceeded(total, delta, self.budget_usd)
                    connection.execute("UPDATE reservations SET reserve_usd = ?, input_tokens = ?, max_output_tokens = ? WHERE request_key = ?",
                                       (amount, input_tokens, output_cap, request_key))
                else:
                    total = connection.execute("SELECT COALESCE(SUM(reserve_usd), 0) FROM reservations").fetchone()[0]
                    if total > self.budget_usd + 1e-12:
                        raise BudgetExceeded(total, 0.0, self.budget_usd)
                    amount = previous[0]
            else:
                total = connection.execute("SELECT COALESCE(SUM(reserve_usd), 0) FROM reservations").fetchone()[0]
                if total + amount > self.budget_usd + 1e-12:
                    raise BudgetExceeded(total, amount, self.budget_usd)
                connection.execute(
                    "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)",
                    (request_key, pair_key, input_tokens, output_cap, MAX_ATTEMPTS, amount),
                )
            connection.execute(
                "INSERT INTO events (utc, event_type, request_key, pair_key, details_json) VALUES (?, ?, ?, ?, ?)",
                (_utc_now(), "pair_budget_reserved", request_key, pair_key,
                 _canonical({"input_tokens": input_tokens, "max_attempts": MAX_ATTEMPTS,
                             "max_output_tokens": output_cap, "reserve_usd": amount,
                             "model_id": model_id, "input_usd_per_million": input_usd_per_million,
                             "output_usd_per_million": output_usd_per_million}))
            )
            return float(amount)

    def start_attempt(self, request_key: str, pair_key: str) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            reservation = connection.execute(
                "SELECT max_attempts FROM reservations WHERE request_key = ?", (request_key,)
            ).fetchone()
            if not reservation:
                raise RuntimeError("Attempt cannot start before its budget reservation.")
            count = connection.execute("SELECT COUNT(*) FROM attempts WHERE request_key = ?", (request_key,)).fetchone()[0]
            if count >= reservation[0]:
                raise RuntimeError("Persistent retry limit has been exhausted for this pair.")
            number = count + 1
            connection.execute(
                "INSERT INTO attempts (request_key, attempt_no, started_utc, status) VALUES (?, ?, ?, ?)",
                (request_key, number, _utc_now(), "started"),
            )
            connection.execute(
                "INSERT INTO events (utc, event_type, request_key, pair_key, details_json) VALUES (?, ?, ?, ?, ?)",
                (_utc_now(), "inference_attempt_started", request_key, pair_key,
                 _canonical({"attempt": number})),
            )
            return number

    def finish_attempt(self, request_key: str, pair_key: str, attempt_no: int, status: str,
                       response: Mapping[str, Any] | None = None,
                       input_usd_per_million: float = INPUT_USD_PER_MILLION,
                       output_usd_per_million: float = OUTPUT_USD_PER_MILLION) -> float | None:
        usage = (response or {}).get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        actual_cost = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            actual_cost = _maximum_request_cost(input_tokens, output_tokens,
                                                input_usd_per_million, output_usd_per_million)
        response_id = (response or {}).get("id")
        actual_model = (response or {}).get("model")
        details = {"attempt": attempt_no, "status": status, "response_id": response_id,
                   "actual_model": actual_model, "input_tokens": input_tokens,
                   "output_tokens": output_tokens, "actual_cost_usd": actual_cost}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE attempts SET finished_utc = ?, status = ?, response_id = ?, actual_model = ?, "
                "input_tokens = ?, output_tokens = ?, actual_cost_usd = ? WHERE request_key = ? AND attempt_no = ?",
                (_utc_now(), status, response_id, actual_model, input_tokens, output_tokens,
                 actual_cost, request_key, attempt_no),
            )
            connection.execute(
                "INSERT INTO events (utc, event_type, request_key, pair_key, details_json) VALUES (?, ?, ?, ?, ?)",
                (_utc_now(), "inference_attempt_finished", request_key, pair_key, _canonical(details)),
            )
        return actual_cost

    def attempts(self, request_key: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM attempts WHERE request_key = ?", (request_key,)).fetchone()[0])

    def total_attempts(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])

    def export(self, path: Path) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT utc, event_type, request_key, pair_key, details_json FROM events ORDER BY id"
            ).fetchall()
        output = [{"utc": utc, "event_type": kind, "request_key": key, "pair_key": pair,
                   "details": json.loads(details)} for utc, kind, key, pair, details in rows]
        _atomic_jsonl(path, output)

    def totals(self) -> dict[str, float]:
        with self._connect() as connection:
            reserved = connection.execute("SELECT COALESCE(SUM(reserve_usd), 0) FROM reservations").fetchone()[0]
            actual = connection.execute("SELECT COALESCE(SUM(actual_cost_usd), 0) FROM attempts").fetchone()[0]
        return {"reserved_max_usd": float(reserved), "reported_actual_usd": float(actual)}


class BudgetExceeded(RuntimeError):
    def __init__(self, already_reserved: float, new_reservation: float, budget: float) -> None:
        self.already_reserved = already_reserved
        self.new_reservation = new_reservation
        self.budget = budget
        super().__init__("Reservation would exceed the hard evaluator budget.")


def dry_run(experiment_dir: Path, budget_usd: float = DEFAULT_BUDGET_USD) -> dict[str, Any]:
    evaluator_dir = experiment_dir / "openai_evaluator"
    prompt = frozen_prompt_text(evaluator_dir / "evaluator_prompt_frozen.md")
    pool = _read_jsonl(experiment_dir / "blinded_evaluation_pool.jsonl")
    estimates: list[dict[str, Any]] = []
    total_reserved = 0.0
    token_limit = MODEL_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS
    for row in pool:
        payload = build_response_payload(row, prompt)
        estimate = local_token_estimate(payload)
        reservation = _maximum_request_cost(estimate) * MAX_ATTEMPTS
        total_reserved += reservation
        estimates.append({
            "profile_id": str(row["profile_id"]), "job_id": str(row["job_id"]),
            "request_key": request_identity(row, prompt), "input_tokens": estimate,
            "token_count_source": "local_tiktoken_estimate_not_authoritative",
            "max_output_tokens_per_attempt": MAX_OUTPUT_TOKENS,
            "attempts_reserved": MAX_ATTEMPTS,
            "max_cost_reservation_usd": round(reservation, 10),
            "too_large_by_local_estimate": estimate > token_limit,
        })
    report = {
        "experiment_id": EXPERIMENT_ID, "created_utc": _utc_now(), "mode": "offline_dry_run",
        "model": MODEL, "standard_pricing_usd_per_million": {
            "input": INPUT_USD_PER_MILLION, "output": OUTPUT_USD_PER_MILLION,
        },
        "budget_usd": budget_usd, "pool_pairs": len(pool), "attempt_limit_per_pair": MAX_ATTEMPTS,
        "output_cap_tokens_per_attempt": MAX_OUTPUT_TOKENS,
        "token_count_source": "local_tiktoken_estimate_not_authoritative",
        "note": "No API request was made. This estimate excludes API framing overhead; run mode uses the official input-token-count endpoint before every inference request.",
        "total_worst_case_reservation_usd": round(total_reserved, 10),
        "within_budget_by_local_estimate": total_reserved <= budget_usd,
        "too_large_pairs": [row for row in estimates if row["too_large_by_local_estimate"]],
        "per_pair": estimates,
    }
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(evaluator_dir / "dry_run_estimate.json", report)
    _atomic_json(evaluator_dir / "request_manifest.json", {
        "model": MODEL, "prompt_sha256": FROZEN_PROMPT_SHA256,
        "schema_sha256": _sha256(_canonical(JUDGMENT_SCHEMA)), "settings": {
            "temperature": 0, "max_output_tokens": MAX_OUTPUT_TOKENS, "store": False,
        },
        "requests": estimates,
    })
    return report


def _cached_judgment(path: Path, request_key: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("request_key") != request_key:
        return None
    valid, errors = validate_judgment(saved.get("judgment", {}), row)
    if errors:
        return None
    return valid


def run_evaluation(experiment_dir: Path, budget_usd: float = DEFAULT_BUDGET_USD,
                   max_pairs: int | None = None,
                   ledger_path: Path | None = None) -> dict[str, Any]:
    api_key = load_api_key(PROJECT_ROOT)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; no API call was made.")
    evaluator_dir = experiment_dir / "openai_evaluator"
    prompt = frozen_prompt_text(evaluator_dir / "evaluator_prompt_frozen.md")
    pool = _read_jsonl(experiment_dir / "blinded_evaluation_pool.jsonl")
    if len({(str(row["profile_id"]), str(row["job_id"])) for row in pool}) != len(pool):
        raise ValueError("Blinded pool contains duplicate profile/job identities.")
    cache_dir = evaluator_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(ledger_path or (evaluator_dir / "budget_ledger.sqlite3"), budget_usd)
    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    pair_states: dict[tuple[str, str], dict[str, Any]] = {}
    recovered_path = evaluator_dir / "recovered_judgments.jsonl"
    recovered_rows = _read_jsonl(recovered_path) if recovered_path.is_file() else []
    recovered_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    pool_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in pool}
    for recovered_row in recovered_rows:
        recovered_pair = (str(recovered_row.get("profile_id")), str(recovered_row.get("job_id")))
        if recovered_row.get("actual_model") != MODEL or recovered_pair not in pool_by_pair:
            continue
        judgment, validation_errors = validate_judgment(recovered_row.get("judgment", {}), pool_by_pair[recovered_pair])
        if not validation_errors:
            recovered_by_pair[recovered_pair] = judgment
    started = time.perf_counter()
    stopped_for_budget = False
    stopped_for_api_error = False
    stop_reason: str | None = None

    processed_pairs = 0
    for index, row in enumerate(pool):
        if max_pairs is not None and processed_pairs >= max_pairs:
            break
        pair = (str(row["profile_id"]), str(row["job_id"]))
        pair_key = _sha256(_canonical(pair))
        request_key = request_identity(row, prompt)
        cache_path = cache_dir / (request_key + ".json")
        cached = _cached_judgment(cache_path, request_key, row)
        if cached is not None:
            judgments[pair] = cached
            pair_states[pair] = {"profile_id": pair[0], "job_id": pair[1],
                                 "status": "evaluated_unknown" if cached["status"] == "unknown" else "evaluated_judged",
                                 "attempts": 0, "cache_reused": True, "request_key": request_key}
            processed_pairs += 1
            continue
        if pair in recovered_by_pair:
            judgment = recovered_by_pair[pair]
            judgments[pair] = judgment
            pair_states[pair] = {
                "profile_id": pair[0], "job_id": pair[1],
                "status": "evaluated_unknown" if judgment["status"] == "unknown" else "evaluated_judged",
                "attempts": 0, "cache_reused": False, "recovered_from_saved_response": True,
                "request_key": request_key,
            }
            processed_pairs += 1
            continue

        pair_errors: list[str] = []
        while ledger.attempts(request_key) < MAX_ATTEMPTS and not stopped_for_api_error:
            payload = build_response_payload(row, prompt)
            count_response: dict[str, Any] | None = None
            try:
                count_response = _http_json(API_BASE + "/responses/input_tokens", _count_payload(payload), api_key)
                input_tokens = int(count_response["input_tokens"])
            except Exception as error:
                pair_errors.append("input_token_count_failed: " + str(error))
                _append_jsonl(evaluator_dir / "errors.jsonl", {
                    "utc": _utc_now(), "profile_id": pair[0], "job_id": pair[1],
                    "request_key": request_key, "stage": "input_token_count",
                    "error_type": type(error).__name__, "error": str(error),
                    "billing_assumption": "no inference request started",
                })
                if _must_stop_for_api_error(error):
                    stopped_for_api_error = True
                    stop_reason = "fatal_api_error_before_inference"
                break
            token_limit = MODEL_CONTEXT_TOKENS - MAX_OUTPUT_TOKENS
            if input_tokens > token_limit:
                pair_errors.append("request_exceeds_model_context_without_truncation")
                break
            try:
                ledger.reserve_pair(request_key, pair_key, input_tokens)
            except BudgetExceeded as error:
                pair_errors.append("budget_stop_reserved={:.8f}_new={:.8f}_limit={:.2f}".format(
                    error.already_reserved, error.new_reservation, error.budget))
                stopped_for_budget = True
                break
            attempt_no = ledger.start_attempt(request_key, pair_key)
            ledger.export(evaluator_dir / "ledger.jsonl")
            request_started = time.perf_counter()
            response_data: dict[str, Any] | None = None
            try:
                response_data = _http_json(API_BASE + "/responses", payload, api_key)
                _append_jsonl(evaluator_dir / "raw_responses.jsonl", {
                    "utc": _utc_now(), "profile_id": pair[0], "job_id": pair[1],
                    "request_key": request_key, "attempt": attempt_no,
                    "response_id": response_data.get("id"), "actual_model": response_data.get("model"),
                    "status": response_data.get("status"), "usage": response_data.get("usage"),
                    "raw_response": response_data,
                })
                if response_data.get("model") != MODEL:
                    raise ValueError("unexpected_response_model")
                if response_data.get("status") != "completed":
                    raise ValueError("response_status_" + str(response_data.get("status", "missing")))
                raw_text = _extract_output_text(response_data)
                _append_jsonl(evaluator_dir / "raw_output_text.jsonl", {
                    "utc": _utc_now(), "profile_id": pair[0], "job_id": pair[1],
                    "request_key": request_key, "attempt": attempt_no,
                    "raw_output_text": raw_text,
                })
                parsed = json.loads(raw_text)
                judgment, errors = validate_judgment(parsed, row, require_evidence_ids=True)
                if errors:
                    pair_errors.extend(errors)
                    ledger.finish_attempt(request_key, pair_key, attempt_no, "invalid_judgment", response_data)
                else:
                    ledger.finish_attempt(request_key, pair_key, attempt_no, "completed", response_data)
                    _atomic_json(cache_path, {
                        "request_key": request_key, "profile_id": pair[0], "job_id": pair[1],
                        "response_id": response_data.get("id"), "actual_model": response_data.get("model"),
                        "judgment": judgment,
                    })
                    judgments[pair] = judgment
                    pair_states[pair] = {
                        "profile_id": pair[0], "job_id": pair[1],
                        "status": "evaluated_unknown" if judgment["status"] == "unknown" else "evaluated_judged",
                        "attempts": attempt_no, "cache_reused": False, "request_key": request_key,
                    }
                    break
            except Exception as error:
                pair_errors.append(type(error).__name__ + ": " + str(error))
                ledger.finish_attempt(request_key, pair_key, attempt_no, "uncertain_or_failed", response_data)
                _append_jsonl(evaluator_dir / "errors.jsonl", {
                    "utc": _utc_now(), "profile_id": pair[0], "job_id": pair[1],
                    "request_key": request_key, "attempt": attempt_no,
                    "error_type": type(error).__name__, "error": str(error),
                    "billing_assumption": "full reserved maximum retained",
                })
                if _must_stop_for_api_error(error):
                    stopped_for_api_error = True
                    stop_reason = "fatal_api_error_during_inference"
            finally:
                _append_jsonl(evaluator_dir / "runtime.jsonl", {
                    "utc": _utc_now(), "profile_id": pair[0], "job_id": pair[1],
                    "request_key": request_key, "attempt": ledger.attempts(request_key),
                    "elapsed_seconds": time.perf_counter() - request_started,
                })
            ledger.export(evaluator_dir / "ledger.jsonl")
        if pair not in pair_states:
            pair_states[pair] = {
                "profile_id": pair[0], "job_id": pair[1],
                "status": "not_evaluated" if ledger.attempts(request_key) == 0 else "failed",
                "attempts": ledger.attempts(request_key), "cache_reused": False,
                "request_key": request_key, "errors": pair_errors,
            }
        processed_pairs += 1
        _atomic_jsonl(evaluator_dir / "validated_judgments.jsonl", list(judgments.values()))
        _atomic_jsonl(evaluator_dir / "pair_status.jsonl", [
            pair_states.get((str(candidate["profile_id"]), str(candidate["job_id"])), {
                "profile_id": str(candidate["profile_id"]), "job_id": str(candidate["job_id"]),
                "status": "not_evaluated", "attempts": 0, "cache_reused": False,
                "request_key": request_identity(candidate, prompt),
            }) for candidate in pool
        ])
        ledger.export(evaluator_dir / "ledger.jsonl")
        if stopped_for_budget or stopped_for_api_error or (max_pairs is not None and processed_pairs >= max_pairs):
            if max_pairs is not None and processed_pairs >= max_pairs and not stopped_for_budget and not stopped_for_api_error:
                stop_reason = "pair_gate_limit"
            break

    for row in pool:
        pair = (str(row["profile_id"]), str(row["job_id"]))
        if pair not in pair_states:
            pair_states[pair] = {"profile_id": pair[0], "job_id": pair[1],
                                 "status": "not_evaluated", "attempts": 0,
                                 "cache_reused": False, "request_key": request_identity(row, prompt)}
    _atomic_jsonl(evaluator_dir / "validated_judgments.jsonl", list(judgments.values()))
    status_rows = [pair_states[(str(row["profile_id"]), str(row["job_id"]))] for row in pool]
    _atomic_jsonl(evaluator_dir / "pair_status.jsonl", status_rows)
    ledger.export(evaluator_dir / "ledger.jsonl")
    totals = ledger.totals()
    provenance = {
        "experiment_id": EXPERIMENT_ID, "status": "complete" if len(judgments) == len(pool) else "partial",
        "stop_reason": stop_reason or ("budget_limit" if stopped_for_budget else None),
        "requested_model": MODEL, "actual_model_ids": sorted({
            row.get("actual_model") for row in _read_jsonl(evaluator_dir / "raw_responses.jsonl")
            if row.get("actual_model")
        }) if (evaluator_dir / "raw_responses.jsonl").is_file() else [],
        "prompt_sha256": FROZEN_PROMPT_SHA256, "schema_sha256": _sha256(_canonical(JUDGMENT_SCHEMA)),
        "citation_protocol": "source_evidence_ids_v1",
        "citation_protocol_addendum_sha256": _sha256(
            "Evaluator protocol amendment: supporting_candidate_evidence_ids and "
            "supporting_job_evidence_ids must contain only evidence_id values from the corresponding "
            "supplied evidence catalog. Do not reproduce or paraphrase quotations. IDs resolve to the "
            "exact original source text and are checked locally."
        ),
        "pool_pairs": len(pool), "evaluated_judged": sum(s["status"] == "evaluated_judged" for s in status_rows),
        "evaluated_unknown": sum(s["status"] == "evaluated_unknown" for s in status_rows),
        "failed": sum(s["status"] == "failed" for s in status_rows),
        "not_evaluated": sum(s["status"] == "not_evaluated" for s in status_rows),
        "api_inference_calls_cumulative": ledger.total_attempts(),
        "budget_usd": budget_usd, **totals,
        "ledger_path": str(ledger.path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "rankings_opened": False,
        "original_pending_judgments_modified": False,
        "prior_results_exposure": "The evaluator received only stable profile/job IDs and original candidate/job evidence; rankings and prior aggregate results were not sent.",
        "token_counting": "official Responses input-token-count endpoint before every inference attempt",
        "billing_assumption": "Every attempt retains its full maximum reservation, including uncertain transport failures.",
    }
    _atomic_json(evaluator_dir / "provenance.json", provenance)
    return provenance


def compute_isolated_metrics(experiment_dir: Path) -> dict[str, Any]:
    evaluator_dir = experiment_dir / "openai_evaluator"
    judgments = _read_jsonl(evaluator_dir / "validated_judgments.jsonl") if (evaluator_dir / "validated_judgments.jsonl").is_file() else []
    if not any(row.get("status") == "judged" and row.get("grade") in {0, 1, 2, 3} for row in judgments):
        raise RuntimeError("No numeric judgments exist; metric computation is withheld.")
    metrics_dir = evaluator_dir / "metrics_inputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name in ("synthetic_profiles.json", "blinded_evaluation_pool.jsonl", "rankings.csv"):
        source = experiment_dir / name
        target = metrics_dir / name
        if not target.exists():
            target.write_bytes(source.read_bytes())
    valid_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    pool = _read_jsonl(experiment_dir / "blinded_evaluation_pool.jsonl")
    adapted = []
    for row in pool:
        key = (str(row["profile_id"]), str(row["job_id"]))
        adapted.append(valid_by_pair.get(key, {
            "profile_id": key[0], "job_id": key[1], "grade": None,
            "status": "pending_evaluator", "unknown_reason": "No validated evaluator judgment was produced.",
        }))
    _atomic_jsonl(metrics_dir / "raw_judgments.jsonl", adapted)
    import run_bge_transition_experiment as experiment

    result = experiment.compute_metrics(metrics_dir)
    conflict_claims = result.pop("confirmed_conflicts_top5_by_profile_and_method", None)
    result["unvalidated_evaluator_conflict_claims_top5_by_profile_and_method"] = conflict_claims
    result["conflict_metric_note"] = "These are evaluator claims, not independently adjudicated or ground-truth conflict rates."
    _atomic_json(evaluator_dir / "metrics_and_coverage.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "recover", "run", "metrics"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
        sub.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
        if command == "run":
            sub.add_argument("--max-pairs", type=int, default=None,
                             help="Stop after this many pool pairs; use 1 for a validation gate.")
            sub.add_argument("--ledger-path", type=Path, default=None,
                             help="Optional existing SQLite budget ledger shared with another experiment.")
    args = parser.parse_args()
    try:
        if args.command == "dry-run":
            result = dry_run(args.experiment_dir, args.budget_usd)
        elif args.command == "recover":
            result = recover_saved_responses(args.experiment_dir)
        elif args.command == "run":
            result = run_evaluation(args.experiment_dir, args.budget_usd, args.max_pairs,
                                    args.ledger_path)
        else:
            result = compute_isolated_metrics(args.experiment_dir)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(2, "error: {}\n".format(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())