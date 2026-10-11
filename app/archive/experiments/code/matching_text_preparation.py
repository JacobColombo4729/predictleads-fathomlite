"""Evidence-preserving extraction and BGE rendering for matching-text experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


DATASET_VERSION = "matching_text_dataset_v1"
REPRESENTATION_ID = "bge_faceted_evidence_priority_v1"
MAX_BGE_TOKENS = 512
CATEGORIES = (
    "duties",
    "required_qualifications",
    "preferred_qualifications",
    "skills_tools",
    "working_conditions",
    "company_background",
    "unclassified",
)
RENDER_ORDER = (
    "required_qualifications",
    "duties",
    "skills_tools",
    "preferred_qualifications",
    "working_conditions",
    "unclassified",
)
LABELS = {
    "duties": "Duty",
    "required_qualifications": "Required qualification",
    "preferred_qualifications": "Preferred qualification",
    "skills_tools": "Skill or tool",
    "working_conditions": "Working condition",
    "unclassified": "Unclassified source evidence",
}

_REQUIRED_HEADING = re.compile(
    r"^(?:requirements?|required qualifications?|minimum qualifications?|mandatory qualifications?|"
    r"required experience|education and experience requirements?)$",
    re.IGNORECASE,
)
_PREFERRED_HEADING = re.compile(
    r"^(?:preferred qualifications?|preferred experience|desired qualifications?|nice to have|preferred)$",
    re.IGNORECASE,
)
_DUTY_HEADING = re.compile(
    r"^(?:responsibilities|duties|essential functions|essential duties(?: and responsibilities)?|"
    r"what you(?:'|’)ll do|what you will do|key responsibilities|key accountabilities|job functions)$",
    re.IGNORECASE,
)
_SKILL_HEADING = re.compile(
    r"^(?:skills|technical skills|skills and abilities|knowledge, skills(?: and abilities)?|"
    r"tools and technologies|technology and tools|software and systems)$",
    re.IGNORECASE,
)
_CONDITION_HEADING = re.compile(
    r"^(?:working conditions|work environment|physical requirements|physical demands|"
    r"schedule and travel|travel requirements|work schedule|work arrangement|compensation and benefits|benefits)$",
    re.IGNORECASE,
)
_COMPANY_HEADING = re.compile(
    r"^(?:about us|about the company|company overview|company description|who we are|our story|"
    r"our mission|our culture|why join us|equal opportunity(?: employer)?|eeo statement|"
    r"diversity and inclusion|careers? at .+)$",
    re.IGNORECASE,
)
_GENERIC_QUALIFICATION_HEADING = re.compile(
    r"^(?:qualifications?|education|experience|education and experience|job requirements)$",
    re.IGNORECASE,
)
_COMPANY_SIGNAL = re.compile(
    r"\b(?:our mission|our vision|founded in \d{4}|about the company|company overview|company description|"
    r"equal opportunity employer|eeo statement|we are proud to be|our culture|our values|"
    r"join our team|vibrant culture|competitive compensation|benefits package)\b",
    re.IGNORECASE,
)
_NEGATED_REQUIRED = re.compile(
    r"\b(?:not required|not mandatory|not necessary|no longer required|"
    r"no\s+[\w-]+\s+(?:is\s+)?required|does not require|do not require)\b",
    re.IGNORECASE,
)
_NEGATED_PREFERRED = re.compile(r"\b(?:not preferred|not a preference)\b", re.IGNORECASE)
_REQUIRED_SIGNAL = re.compile(
    r"\b(?:required|must(?:\s+have|\s+be)?|mandatory|minimum|at least|license required|certification required)\b",
    re.IGNORECASE,
)
_PREFERRED_SIGNAL = re.compile(r"\b(?:preferred|preferably|nice to have|desired qualification)\b", re.IGNORECASE)
_CONDITION_SIGNAL = re.compile(
    r"\b(?:work environment|working conditions|physical demands?|lift(?:ing)?\s+\d+|"
    r"travel up to|travel required|on[- ]site|remote work|hybrid work|rotating shifts?|"
    r"work schedule|weekends?|overnight shifts?|standing for|hours per week|work arrangement)\b",
    re.IGNORECASE,
)
_SKILL_SIGNAL = re.compile(
    r"\b(?:proficient in|proficiency in|experience with|knowledge of|familiar with|"
    r"software|systems?|tools?|equipment|proficiency|skills include|skilled in|operate .{0,50}(?:system|software|tool))\b",
    re.IGNORECASE,
)
_DUTY_SIGNAL = re.compile(
    r"\b(?:manage|perform|support|assist|coordinate|maintain|process|provide|develop|ensure|"
    r"operate|troubleshoot|lead|deliver|handle|monitor|prepare|review|create|conduct|build|"
    r"implement|execute|receive|stock|count|schedule|resolve|communicate|track|reconcile|"
    r"inspect|train|supervise|install|test|design|write|analy[sz]e|record|collect|document)\b",
    re.IGNORECASE,
)
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_heading(value: str) -> str:
    value = _BULLET_PREFIX.sub("", value).strip().strip("#| ").rstrip(":").strip()
    return re.sub(r"\s+", " ", value)


def _heading_kind(value: str) -> tuple[str | None, str, str]:
    heading = _clean_heading(value)
    if _COMPANY_HEADING.fullmatch(heading):
        return "company_background", "unspecified", "explicit_company_heading"
    if _CONDITION_HEADING.fullmatch(heading):
        return "working_conditions", "unspecified", "explicit_working_conditions_heading"
    if _REQUIRED_HEADING.fullmatch(heading):
        return "required_qualifications", "required", "explicit_required_heading"
    if _PREFERRED_HEADING.fullmatch(heading):
        return "preferred_qualifications", "preferred", "explicit_preferred_heading"
    if _DUTY_HEADING.fullmatch(heading):
        return "duties", "unspecified", "explicit_duty_heading"
    if _SKILL_HEADING.fullmatch(heading):
        return "skills_tools", "unspecified", "explicit_skill_heading"
    if _GENERIC_QUALIFICATION_HEADING.fullmatch(heading):
        return "unclassified", "unspecified", "qualification_heading_scope_unspecified"
    return None, "unspecified", "unrecognized_heading"


def _is_heading_line(line: str) -> tuple[bool, str, str, int]:
    bullet_match = _BULLET_PREFIX.match(line)
    prefix_end = bullet_match.end() if bullet_match else len(line) - len(line.lstrip())
    content = line[prefix_end:].rstrip()
    colon = content.find(":")
    if colon > 0 and colon <= 80:
        heading = content[:colon].strip()
        kind, _, _ = _heading_kind(heading)
        if kind is not None or not re.search(r"[.!?]", heading):
            body = content[colon + 1:]
            body_offset = prefix_end + len(content[:colon + 1])
            while body.startswith(" "):
                body = body[1:]
                body_offset += 1
            return True, heading, body, body_offset
    cleaned = _clean_heading(content)
    kind, _, _ = _heading_kind(cleaned)
    if kind is not None:
        return True, cleaned, "", prefix_end
    return False, "", line, 0


def _classify(text: str, heading: str, heading_category: str,
              heading_modality: str, heading_basis: str) -> tuple[str, str, str]:
    if _COMPANY_SIGNAL.search(text):
        return "company_background", "unspecified", "company_background_language"
    if _NEGATED_REQUIRED.search(text):
        return "unclassified", "not_required", "explicit_negated_requirement"
    if _NEGATED_PREFERRED.search(text):
        return "unclassified", "negated_preference", "explicit_negated_preference"
    if heading_category in {"required_qualifications", "preferred_qualifications"}:
        return heading_category, heading_modality, heading_basis
    if _PREFERRED_SIGNAL.search(text):
        return "preferred_qualifications", "preferred", "explicit_preferred_language"
    if _REQUIRED_SIGNAL.search(text):
        return "required_qualifications", "required", "explicit_required_language"
    if heading_category in {"duties", "skills_tools", "working_conditions", "company_background"}:
        return heading_category, heading_modality, heading_basis
    if _CONDITION_SIGNAL.search(text):
        return "working_conditions", "unspecified", "working_conditions_language"
    if _SKILL_SIGNAL.search(text):
        return "skills_tools", "unspecified", "skill_or_tool_language"
    if _DUTY_SIGNAL.search(text):
        return "duties", "unspecified", "action_language"
    return "unclassified", "unspecified", "no_unambiguous_category_signal"


def extract_evidence(record: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    description = record.get("description_text")
    if "description_text" not in record:
        state = "missing_key"
        source_text = ""
    elif description is None:
        state = "null"
        source_text = ""
    elif not isinstance(description, str):
        state = "non_string"
        source_text = ""
    elif not description.strip():
        state = "blank"
        source_text = description
    else:
        state = "present"
        source_text = description

    evidence: list[dict[str, Any]] = []
    headings: list[dict[str, Any]] = []
    state_category = "unclassified"
    state_modality = "unspecified"
    state_basis = "no_unambiguous_category_signal"
    active_heading = ""
    line_offset = 0
    for line_number, line in enumerate(source_text.splitlines(keepends=True), 1):
        raw_line = line.rstrip("\r\n")
        content_line = raw_line
        is_heading, heading, body, body_offset = _is_heading_line(content_line)
        if is_heading:
            category, modality, basis = _heading_kind(heading)
            active_heading = heading
            if category is None:
                state_category, state_modality, state_basis = "unclassified", "unspecified", basis
            else:
                state_category, state_modality, state_basis = category, modality, basis
            headings.append({
                "source_line": line_number,
                "source_char_start": line_offset,
                "source_char_end": line_offset + len(raw_line),
                "source_text": raw_line,
                "heading": heading,
                "recognized_category": category,
            })
            content_line = body
            content_start = line_offset + body_offset
        else:
            content_start = line_offset
        if content_line.strip():
            start_trim = len(content_line) - len(content_line.lstrip())
            end_trim = len(content_line.rstrip())
            exact_text = content_line[start_trim:end_trim]
            start = content_start + start_trim
            category, modality, basis = _classify(
                exact_text, active_heading, state_category, state_modality, state_basis,
            )
            evidence.append({
                "evidence_id": "{}:E{:06d}".format(job_id, len(evidence) + 1),
                "category": category,
                "requirement_modality": modality,
                "extraction_basis": basis,
                "source_heading": active_heading or None,
                "source_line": line_number,
                "source_char_start": start,
                "source_char_end": start + len(exact_text),
                "source_text": exact_text,
                "source_text_sha256": _sha256_text(exact_text),
            })
        line_offset += len(line)

    description_present = state == "present"
    return {
        "description_state": state,
        "description_usable_source_value": record.get("description_usable"),
        "historical_markers": {
            key: record[key] for key in (
                "historical_snapshot", "snapshot_date", "status_raw", "status_derived",
                "current_opening_verified", "dates_raw", "dates_utc",
            ) if key in record
        },
        "description_characters": len(source_text),
        "description_present": description_present,
        "headings": headings,
        "evidence": evidence,
    }


def _token_count(tokenizer: Any, text: str, *, special: bool = False) -> int:
    return len(tokenizer.encode(text, add_special_tokens=special))


def render_bge_document(record: Mapping[str, Any], extraction: Mapping[str, Any],
                        tokenizer: Any, max_tokens: int = MAX_BGE_TOKENS) -> tuple[str, dict[str, Any]]:
    onet = record.get("onet_data")
    occupation = onet.get("occupation_name", "") if isinstance(onet, Mapping) else ""
    title = (record.get("title_searchable") or record.get("title_normalized")
             or record.get("title_translated") or record.get("title_raw") or "")
    header = "Job title: {}. Occupation: {}.".format(title, occupation).strip()
    included: list[tuple[Mapping[str, Any], str]] = []
    omitted: list[Mapping[str, Any]] = []
    used = 2
    header_tokens = _token_count(tokenizer, header)
    if used + header_tokens > max_tokens:
        raise ValueError("Job title and occupation exceed the frozen BGE token budget.")
    used += header_tokens
    for category in RENDER_ORDER:
        for item in extraction.get("evidence", []):
            if item.get("category") != category:
                continue
            label = LABELS[category]
            if item.get("requirement_modality") in {"not_required", "negated_preference"}:
                label += " (source says not required/preferred)"
            sentence = _BULLET_PREFIX.sub("", str(item.get("source_text", ""))).strip()
            piece = "{}: {}".format(label, sentence)
            piece_tokens = _token_count(tokenizer, piece)
            if used + piece_tokens <= max_tokens:
                included.append((item, piece))
                used += piece_tokens
            else:
                omitted.append(item)
    background = [item for item in extraction.get("evidence", [])
                  if item.get("category") == "company_background"]
    rendered = " ".join([header, *(piece for _, piece in included)])
    actual_tokens = _token_count(tokenizer, rendered, special=True)
    if actual_tokens > max_tokens:
        raise AssertionError("Faceted BGE text exceeds its configured token limit.")
    return rendered, {
        "representation_id": REPRESENTATION_ID,
        "max_tokens": max_tokens,
        "encoded_tokens_including_specials": actual_tokens,
        "retained_evidence_ids": [item["evidence_id"] for item, _ in included],
        "omitted_evidence": [
            {"evidence_id": item["evidence_id"], "category": item["category"],
             "source_text_sha256": item["source_text_sha256"]} for item in omitted
        ],
        "company_background_excluded_count": len(background),
        "company_background_excluded_evidence_ids": [item["evidence_id"] for item in background],
        "retained_category_counts": {
            category: sum(item.get("category") == category for item, _ in included)
            for category in CATEGORIES
        },
    }


def build_dataset_row(record: Mapping[str, Any], source_index: int, tokenizer: Any) -> dict[str, Any]:
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("Prepared source record is missing a stable job_id.")
    extraction = extract_evidence(record, job_id)
    rendered, trace = render_bge_document(record, extraction, tokenizer)
    return {
        "dataset_version": DATASET_VERSION,
        "source_record_index": source_index,
        "job_id": job_id,
        "source_record_sha256": _sha256_text(_canonical(record)),
        "source_record": dict(record),
        "matching_text": extraction,
        "bge_document_text": rendered,
        "bge_render_trace": trace,
    }


def validate_dataset_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    source = row.get("source_record")
    if not isinstance(source, Mapping):
        return ["source_record_missing"]
    job_id = str(source.get("job_id", ""))
    extraction = row.get("matching_text")
    if not job_id or row.get("job_id") != job_id:
        errors.append("stable_job_id_mismatch")
    if row.get("source_record_sha256") != _sha256_text(_canonical(source)):
        errors.append("source_record_hash_mismatch")
    if not isinstance(extraction, Mapping):
        return errors + ["matching_text_missing"]
    description = source.get("description_text")
    if not isinstance(description, str):
        description = ""
    ids: set[str] = set()
    for item in extraction.get("evidence", []):
        evidence_id = item.get("evidence_id")
        if evidence_id in ids:
            errors.append("duplicate_evidence_id")
        ids.add(str(evidence_id))
        start, end = item.get("source_char_start"), item.get("source_char_end")
        if not isinstance(start, int) or not isinstance(end, int) or description[start:end] != item.get("source_text"):
            errors.append("source_span_mismatch")
        if item.get("category") not in CATEGORIES:
            errors.append("unknown_category")
        if item.get("requirement_modality") in {"required", "preferred"}:
            if item.get("category") not in {"required_qualifications", "preferred_qualifications"}:
                errors.append("qualification_modality_category_mismatch")
            if not item.get("source_text"):
                errors.append("empty_qualification_evidence")
    return errors