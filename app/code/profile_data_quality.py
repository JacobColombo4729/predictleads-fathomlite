"""Bounded, local-only quality profiling for PredictLeads JSONL.zst files.

Run from the project root with the project's .venv. Outputs are written under
ignored data/discovery/; the script never prints credentials or sends records
to an external service.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import boto3
import botocore
import zstandard as zstd


BUCKET = "tripleten-predictleads-hackathon-109698388873-us-east-2-an"
PREFIX = "2026/09/30/"
REGION = "us-east-2"
PROFILE = "predictleads"
FILES = {
    "jobs": "job_openings_2026_09_30.jsonl.zst",
    "companies": "companies_2026_09_30.jsonl.zst",
}
JOB_FIELDS = (
    "description", "salary", "location", "status", "seniority", "language",
    "posted_at", "first_seen_at", "last_seen_at", "last_processed_at",
)
COMPANY_FIELDS = ("company_name", "domain", "description", "location")
DATE_FIELDS = ("posted_at", "first_seen_at", "last_seen_at", "last_processed_at")
SECTION_SIZE = 10_000
SALARY_CURRENCY_RE = re.compile(
    r"(\b[A-Z]{3}\b|\$|€|£|¥|₹|(?i:USD|EUR|GBP|CAD|AUD))"
)
SALARY_PERIOD_RE = re.compile(
    r"(?i)\b(per\s+hour|hourly|per\s+week|weekly|per\s+month|monthly|"
    r"per\s+year|yearly|annually|annual|\bhr\b|\bmo\b|\byr\b)"
)


class ScanDeadlineExceeded(Exception):
    pass


class _BoundedReader:
    """Enforce a hard compressed-byte cap and a monotonic deadline."""

    def __init__(self, stream: BinaryIO, byte_limit: int, deadline: float):
        self.stream = stream
        self.byte_limit = byte_limit
        self.deadline = deadline
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if time.monotonic() >= self.deadline:
            raise ScanDeadlineExceeded("time_limit")
        remaining = self.byte_limit - self.bytes_read
        if remaining <= 0:
            return b""
        if size is None or size < 0 or size > remaining:
            size = remaining
        data = self.stream.read(size)
        self.bytes_read += len(data)
        return data


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _salary_signature(value: Any) -> tuple[str, str, str]:
    text = str(value)
    if re.search(r"\d+\s*[-–—]\s*\d+", text):
        shape = "range"
    elif re.search(r"\d", text):
        shape = "single_amount_or_text"
    else:
        shape = "non_numeric_text"
    currency_match = SALARY_CURRENCY_RE.search(text)
    period_match = SALARY_PERIOD_RE.search(text)
    return (
        shape,
        currency_match.group(1).upper() if currency_match else "unspecified",
        re.sub(r"\s+", " ", period_match.group(1).lower()) if period_match else "unspecified",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _excerpt(resource: dict[str, Any], cap: int = 500) -> dict[str, Any]:
    attrs = resource.get("attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    relationships = resource.get("relationships")
    relationships = relationships if isinstance(relationships, dict) else {}
    company = relationships.get("company")
    relation_data = company.get("data") if isinstance(company, dict) else None
    company_ref = None
    if isinstance(relation_data, dict):
        company_ref = {"type": relation_data.get("type"), "id": relation_data.get("id")}
    elif isinstance(relation_data, list):
        company_ref = [
            {"type": item.get("type"), "id": item.get("id")}
            for item in relation_data if isinstance(item, dict)
        ]
    excerpt: dict[str, Any] = {
        "resource_type": resource.get("type"),
        "resource_id": resource.get("id"),
    }
    if "title" in attrs:
        excerpt["title"] = attrs.get("title")
    if "company_name" in attrs:
        excerpt["company_name"] = attrs.get("company_name")
    if "description" not in attrs:
        excerpt["description_field_state"] = "absent"
    else:
        description = attrs["description"]
        excerpt["description_field_state"] = (
            "null" if description is None else
            "blank" if isinstance(description, str) and not description.strip() else
            "present"
        )
        excerpt["description_excerpt"] = (
            description[:cap] if isinstance(description, str) else description
        )
    excerpt.update({field: attrs.get(field) for field in ("status", *DATE_FIELDS) if field in attrs})
    if "salary" in attrs:
        excerpt["salary"] = attrs.get("salary")
    if company_ref is not None:
        excerpt["company_reference"] = company_ref
    return excerpt


class DatasetProfiler:
    def __init__(self, name: str, seed: int, reservoir_size: int):
        self.name = name
        self.rng = random.Random(seed)
        self.reservoir_size = reservoir_size
        self.primary_count = 0
        self.wrapper_count = 0
        self.included_count = 0
        self.skipped_resources: Counter[str] = Counter()
        self.malformed: Counter[str] = Counter()
        self.field_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.unexpected_types: dict[str, Counter[str]] = defaultdict(Counter)
        self.categories: dict[str, Counter[str]] = defaultdict(Counter)
        self.category_fields: dict[str, Counter[str]] = defaultdict(Counter)
        self.salary_signatures: Counter[tuple[str, str, str]] = Counter()
        self.salary_examples: dict[tuple[str, str, str], str] = {}
        self.date_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.date_ranges: dict[str, list[datetime | None]] = defaultdict(lambda: [None, None])
        self.date_order: Counter[str] = Counter()
        self.ids: set[tuple[str, str]] = set()
        self.duplicates: Counter[str] = Counter()
        self.relationship_shapes: Counter[str] = Counter()
        self.relationship_match: Counter[str] = Counter()
        self.sections: list[dict[str, Any]] = []
        self._section: dict[str, Any] | None = None
        self.reservoir: list[dict[str, Any]] = []
        self.diagnostics: dict[str, list[dict[str, Any]]] = {
            "null_status": [], "closed_status": [], "missing_description": [],
        }
        self._salary_diagnostics: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _new_section(self, number: int) -> dict[str, Any]:
        return {
            "start_primary_resource": number,
            "end_primary_resource": number - 1,
            "primary_resources": 0,
            "fields": defaultdict(Counter),
        }

    def _quality_fields(self) -> tuple[str, ...]:
        return JOB_FIELDS if self.name == "jobs" else COMPANY_FIELDS

    def _count_fields(self, resource: dict[str, Any], section: dict[str, Any]) -> None:
        attributes = resource.get("attributes")
        if not isinstance(attributes, dict):
            self.skipped_resources["attributes_not_object"] += 1
            attributes = {}
        for field in self._quality_fields():
            count = self.field_counts[field]
            section_field = section["fields"][field]
            if field not in attributes:
                count["absent"] += 1
                section_field["absent"] += 1
                continue
            value = attributes[field]
            if value is None:
                count["null"] += 1
                section_field["null"] += 1
            elif isinstance(value, str) and not value.strip():
                count["blank"] += 1
                section_field["blank"] += 1
            else:
                expected = (str,)
                if not isinstance(value, expected) or isinstance(value, bool):
                    type_name = type(value).__name__
                    count["unexpected_type"] += 1
                    section_field["unexpected_type"] += 1
                    self.unexpected_types[field][type_name] += 1
            if self.name == "jobs" and field == "salary" and value is not None and str(value).strip():
                signature = _salary_signature(value)
                self.salary_signatures[signature] += 1
                self.salary_examples.setdefault(signature, str(value)[:240])
                if signature not in self._salary_diagnostics and len(self._salary_diagnostics) < 8:
                    self._salary_diagnostics[signature] = _excerpt(resource)

    def _count_categories_and_dates(self, resource: dict[str, Any]) -> None:
        attributes = resource.get("attributes")
        if not isinstance(attributes, dict):
            return
        if self.name == "jobs":
            for field in ("status", "seniority", "language"):
                value = attributes.get(field, "<absent>")
                key = "<null>" if value is None else "<absent>" if value == "<absent>" else "<blank>" if isinstance(value, str) and not value.strip() else str(value)
                self.categories[field][key] += 1
            for field, value in attributes.items():
                if "categor" in str(field).lower():
                    vals = value if isinstance(value, list) else [value]
                    for item in vals:
                        self.category_fields[str(field)]["<null>" if item is None else str(item)] += 1
            parsed_dates: dict[str, datetime | None] = {}
            for field in DATE_FIELDS:
                value = attributes.get(field, "<absent>")
                counts = self.date_counts[field]
                if value == "<absent>":
                    counts["absent"] += 1
                    parsed_dates[field] = None
                elif value is None:
                    counts["null"] += 1
                    parsed_dates[field] = None
                elif isinstance(value, str) and not value.strip():
                    counts["blank"] += 1
                    parsed_dates[field] = None
                else:
                    parsed = _parse_timestamp(value)
                    if parsed is None:
                        counts["parse_failure"] += 1
                        parsed_dates[field] = None
                    else:
                        counts["parse_success"] += 1
                        parsed_dates[field] = parsed
                        bounds = self.date_ranges[field]
                        bounds[0] = parsed if bounds[0] is None else min(bounds[0], parsed)
                        bounds[1] = parsed if bounds[1] is None else max(bounds[1], parsed)
            for earlier, later in (("posted_at", "first_seen_at"), ("first_seen_at", "last_seen_at"), ("last_seen_at", "last_processed_at")):
                first, second = parsed_dates.get(earlier), parsed_dates.get(later)
                if first is not None and second is not None and first > second:
                    self.date_order[f"{earlier}_after_{later}"] += 1

    def add_primary(self, resource: dict[str, Any], included_company_keys: set[tuple[str, str]]) -> None:
        next_number = self.primary_count + 1
        section_number = (next_number - 1) // SECTION_SIZE
        if self._section is None or self._section["start_primary_resource"] != section_number * SECTION_SIZE + 1:
            if self._section is not None:
                self._finish_section()
            self._section = self._new_section(section_number * SECTION_SIZE + 1)
        self.primary_count = next_number
        section = self._section
        section["primary_resources"] += 1
        section["end_primary_resource"] = self.primary_count
        self._count_fields(resource, section)
        self._count_categories_and_dates(resource)

        resource_type = str(resource.get("type", "<missing>"))
        resource_id = resource.get("id")
        if resource_id is not None:
            identity = (resource_type, str(resource_id))
            if identity in self.ids:
                self.duplicates[f"{resource_type}:{resource_id}"] += 1
            else:
                self.ids.add(identity)

        if self.name == "jobs":
            self._inspect_relationship(resource, included_company_keys)
        self._diagnostic_examples(resource)

        if self.reservoir_size > 0:
            record = {"resource_type": resource.get("type"), "resource": _json_safe(resource)}
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(record)
            else:
                replacement = self.rng.randrange(self.primary_count)
                if replacement < self.reservoir_size:
                    self.reservoir[replacement] = record

    def _inspect_relationship(self, resource: dict[str, Any], included_company_keys: set[tuple[str, str]]) -> None:
        relationships = resource.get("relationships")
        rel = relationships.get("company", "<absent>") if isinstance(relationships, dict) else "<absent>"
        if rel == "<absent>":
            self.relationship_shapes["relationship_absent"] += 1
            return
        if rel is None:
            self.relationship_shapes["relationship_null"] += 1
            return
        if not isinstance(rel, dict):
            self.relationship_shapes[f"relationship_{type(rel).__name__}"] += 1
            return
        data = rel.get("data", "<absent>")
        if data == "<absent>":
            self.relationship_shapes["data_absent"] += 1
            return
        items = data if isinstance(data, list) else [data]
        shape = "data_list" if isinstance(data, list) else "data_object" if isinstance(data, dict) else "data_null" if data is None else f"data_{type(data).__name__}"
        self.relationship_shapes[shape] += 1
        for item in items:
            if item is None:
                continue
            if not isinstance(item, dict):
                self.skipped_resources["company_relationship_data_not_object"] += 1
                continue
            company_type, company_id = item.get("type"), item.get("id")
            if company_type is None or company_id is None:
                self.relationship_match["incomplete_reference"] += 1
                continue
            key = (str(company_type), str(company_id))
            outcome = "matched_same_wrapper_included_company" if key in included_company_keys else "unmatched_same_wrapper_included_company"
            self.relationship_match[outcome] += 1

    def _diagnostic_examples(self, resource: dict[str, Any]) -> None:
        if self.name != "jobs":
            return
        attrs = resource.get("attributes")
        attrs = attrs if isinstance(attrs, dict) else {}
        if attrs.get("status", object()) is None and len(self.diagnostics["null_status"]) < 5:
            self.diagnostics["null_status"].append(_excerpt(resource))
        if str(attrs.get("status", "")).lower() == "closed" and len(self.diagnostics["closed_status"]) < 5:
            self.diagnostics["closed_status"].append(_excerpt(resource))
        description = attrs.get("description")
        if ("description" not in attrs or description is None
            or isinstance(description, str) and not description.strip()) and len(self.diagnostics["missing_description"]) < 5:
            self.diagnostics["missing_description"].append(_excerpt(resource))

    def _finish_section(self) -> None:
        if self._section is None:
            return
        self.sections.append({
            "start_primary_resource": self._section["start_primary_resource"],
            "end_primary_resource": self._section["end_primary_resource"],
            "primary_resources": self._section["primary_resources"],
            "fields": {field: dict(counts) for field, counts in self._section["fields"].items()},
        })
        self._section = None

    def summary(self) -> dict[str, Any]:
        self._finish_section()
        return {
            "dataset": self.name,
            "primary_resources": self.primary_count,
            "wrappers": self.wrapper_count,
            "included_resources": self.included_count,
            "malformed_records": dict(self.malformed),
            "skipped_resources": dict(self.skipped_resources),
            "fields": {
                field: {
                    "denominator_primary_resources": self.primary_count,
                    "counts": dict(counts),
                    "unexpected_types": dict(self.unexpected_types[field]),
                }
                for field, counts in self.field_counts.items()
            },
            "categories": {field: dict(counts) for field, counts in self.categories.items()},
            "job_category_fields": {field: dict(counts) for field, counts in self.category_fields.items()},
            "salary_formats": [
                {"shape": shape, "currency_observed": currency, "pay_period_observed": period,
                 "count": count, "example": self.salary_examples[(shape, currency, period)]}
                for (shape, currency, period), count in self.salary_signatures.most_common()
            ],
            "dates": {
                field: {
                    **dict(counts),
                    "minimum_utc": self.date_ranges[field][0].isoformat() if self.date_ranges[field][0] else None,
                    "maximum_utc": self.date_ranges[field][1].isoformat() if self.date_ranges[field][1] else None,
                }
                for field, counts in self.date_counts.items()
            },
            "suspicious_date_order": dict(self.date_order),
            "duplicate_primary_type_id_pairs": dict(self.duplicates),
            "company_relationship_shapes": dict(self.relationship_shapes),
            "company_relationship_matches": dict(self.relationship_match),
            "sections_of_10000_primary_resources": self.sections,
            "reservoir_sample": self.reservoir,
            "diagnostic_examples": {
                **self.diagnostics,
                "salary_formats": [
                    {"shape": key[0], "currency_observed": key[1], "pay_period_observed": key[2], **value}
                    for key, value in self._salary_diagnostics.items()
                ],
            },
            "random_sample_note": "Seeded reservoir is random only within the scanned portion unless the file reached EOF.",
            "diagnostic_examples_note": "Deliberately selected diagnostic examples; not representative.",
        }


def profile_stream(
    stream: BinaryIO,
    object_size: int | None,
    profiler: DatasetProfiler,
    *,
    max_primary: int,
    max_bytes: int,
    max_seconds: float,
    max_line_bytes: int,
    progress_every: int,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Profile one compressed object stream; takes ownership and always closes it."""
    started = time.monotonic() if started_at is None else started_at
    deadline = started + max_seconds
    bounded = _BoundedReader(stream, max_bytes, deadline)
    scanned_line_bytes = 0
    pending = bytearray()
    discarding_long_line = False
    stop_reason = "eof"
    eof = False
    last_progress = profiler.primary_count

    def process_line(raw_line: bytes) -> None:
        nonlocal last_progress
        profiler.wrapper_count += 1
        try:
            wrapper = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            profiler.malformed["invalid_json_or_utf8"] += 1
            return
        if not isinstance(wrapper, dict):
            profiler.malformed["wrapper_not_object"] += 1
            return
        primary = wrapper.get("data", "<absent>")
        if primary == "<absent>":
            profiler.malformed["data_absent"] += 1
            return
        if not isinstance(primary, list):
            profiler.malformed["data_not_list"] += 1
            return
        included = wrapper.get("included", [])
        if included is None:
            included = []
        if not isinstance(included, list):
            profiler.malformed["included_not_list"] += 1
            included = []
        included_company_keys: set[tuple[str, str]] = set()
        for item in included:
            if not isinstance(item, dict):
                profiler.skipped_resources["included_resource_not_object"] += 1
                continue
            profiler.included_count += 1
            if item.get("type") is not None and item.get("id") is not None:
                if item.get("type") == "company":
                    included_company_keys.add((str(item["type"]), str(item["id"])))
        for resource in primary:
            if not isinstance(resource, dict):
                profiler.skipped_resources["primary_resource_not_object"] += 1
                continue
            profiler.add_primary(resource, included_company_keys)
        if progress_every > 0 and profiler.primary_count - last_progress >= progress_every:
            last_progress = profiler.primary_count
            elapsed = time.monotonic() - started
            print(
                f"[{profiler.name}] progress: wrappers={profiler.wrapper_count}, "
                f"primary={profiler.primary_count}, included={profiler.included_count}, "
                f"compressed_bytes={bounded.bytes_read}, elapsed_seconds={elapsed:.1f}",
                file=sys.stderr,
                flush=True,
            )

    try:
        decompressor = zstd.ZstdDecompressor().stream_reader(
            bounded, read_across_frames=True, closefd=False,
        )
        try:
            while True:
                if profiler.primary_count >= max_primary:
                    stop_reason = "primary_resource_limit"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "time_limit"
                    break
                try:
                    chunk = decompressor.read(64 * 1024)
                except ScanDeadlineExceeded:
                    stop_reason = "time_limit"
                    break
                except zstd.ZstdError:
                    if bounded.bytes_read >= max_bytes and object_size is not None and object_size > max_bytes:
                        stop_reason = "compressed_byte_limit"
                        break
                    profiler.malformed["zstandard_decompression_error"] += 1
                    stop_reason = "malformed_stream"
                    break
                if not chunk:
                    eof = True
                    if bounded.bytes_read >= max_bytes and object_size is not None and object_size > max_bytes:
                        stop_reason = "compressed_byte_limit"
                    else:
                        stop_reason = "eof"
                    break
                for byte in chunk:
                    scanned_line_bytes += 1
                    if discarding_long_line:
                        if byte == 10:
                            profiler.malformed["line_exceeds_max_line_bytes"] += 1
                            discarding_long_line = False
                            scanned_line_bytes = 0
                        continue
                    if byte == 10:
                        process_line(bytes(pending))
                        pending.clear()
                        scanned_line_bytes = 0
                        if profiler.primary_count >= max_primary:
                            stop_reason = "primary_resource_limit"
                            break
                    elif scanned_line_bytes > max_line_bytes:
                        pending.clear()
                        discarding_long_line = True
                    else:
                        pending.append(byte)
                if stop_reason == "primary_resource_limit":
                    break
                if profiler.primary_count >= max_primary:
                    stop_reason = "primary_resource_limit"
                    break
            if stop_reason == "eof" and pending:
                profiler.malformed["unterminated_final_line"] += 1
        finally:
            decompressor.close()
    except ScanDeadlineExceeded:
        stop_reason = "time_limit"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:
        profiler.malformed[f"stream_error:{type(error).__name__}"] += 1
        stop_reason = "stream_error"
    finally:
        stream.close()

    elapsed = time.monotonic() - started
    profiler._finish_section()
    return {
        "stop_reason": stop_reason,
        "complete_file_scanned": bool(eof and stop_reason == "eof"),
        "object_size_compressed_bytes": object_size,
        "compressed_bytes_read": bounded.bytes_read,
        "elapsed_seconds": round(elapsed, 3),
        "max_line_bytes": max_line_bytes,
        "primary_limit": max_primary,
        "primary_limit_overshoot_by_wrapper": max(0, profiler.primary_count - max_primary),
        "time_limit_seconds": max_seconds,
        "byte_limit": max_bytes,
        "random_sample_scope": "scanned portion only unless complete_file_scanned is true",
    }


def _run_self_test() -> None:
    import io

    class TrackingBytesIO(io.BytesIO):
        def close(self) -> None:
            self.closed_by_profiler = True
            super().close()

    def compressed(records: Iterable[Any]) -> bytes:
        payload = b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records)
        return zstd.ZstdCompressor().compress(payload)

    fixture = [
        {"data": [
            {"id": "j1", "type": "job_opening", "attributes": {
                "title": "A", "description": None, "salary": "$30 per hour", "status": None,
                "location": "NY", "seniority": "junior", "language": "en", "posted_at": "bad-date",
                "first_seen_at": "2026-02-01T00:00:00Z", "last_seen_at": "2026-01-01T00:00:00Z",
            }, "relationships": {"company": {"data": {"type": "company", "id": "c1"}}}},
            {"id": "j2", "type": "job_opening", "attributes": {"status": "closed", "salary": "100 - 120 USD per year"}},
        ], "included": [{"id": "c1", "type": "company"}]},
        {"data": [{"id": "j2", "type": "job_opening", "attributes": {"status": "closed", "description": "ok", "salary": 10}}], "included": []},
        {"data": "bad"},
    ]
    fixture_payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in fixture
    ) + b'{"data":[]}'
    data = zstd.ZstdCompressor().compress(fixture_payload)
    body = TrackingBytesIO(data)
    profiler = DatasetProfiler("jobs", seed=7, reservoir_size=2)
    run = profile_stream(body, len(data), profiler, max_primary=100, max_bytes=len(data) + 5,
                         max_seconds=5, max_line_bytes=4096, progress_every=0)
    assert body.closed_by_profiler
    summary = profiler.summary()
    assert run["stop_reason"] == "eof" and run["complete_file_scanned"]
    assert summary["primary_resources"] == 3 and summary["included_resources"] == 1
    assert summary["fields"]["description"]["counts"]["null"] == 1
    assert summary["fields"]["description"]["counts"]["absent"] == 1
    assert summary["fields"]["salary"]["counts"]["unexpected_type"] == 1
    assert summary["duplicate_primary_type_id_pairs"]["job_opening:j2"] == 1
    assert summary["company_relationship_matches"]["matched_same_wrapper_included_company"] == 1
    assert summary["company_relationship_matches"].get("unmatched_same_wrapper_included_company", 0) == 0
    assert summary["dates"]["posted_at"]["parse_failure"] == 1
    assert summary["suspicious_date_order"]["first_seen_at_after_last_seen_at"] == 1
    assert len(summary["reservoir_sample"]) == 2
    assert summary["malformed_records"]["data_not_list"] == 1
    assert summary["malformed_records"]["unterminated_final_line"] == 1

    cap_body = TrackingBytesIO(compressed(fixture))
    cap_profiler = DatasetProfiler("jobs", seed=7, reservoir_size=1)
    cap_run = profile_stream(cap_body, len(cap_body.getvalue()), cap_profiler, max_primary=1,
                             max_bytes=1_000_000, max_seconds=5, max_line_bytes=4096,
                             progress_every=0)
    assert cap_run["stop_reason"] == "primary_resource_limit"
    assert cap_run["primary_limit_overshoot_by_wrapper"] == 1
    assert cap_body.closed_by_profiler

    long_line_body = TrackingBytesIO(compressed([{"data": [{"id": "large"}], "padding": "x" * 100}]))
    long_line_profiler = DatasetProfiler("jobs", seed=7, reservoir_size=0)
    long_line_run = profile_stream(long_line_body, len(long_line_body.getvalue()), long_line_profiler,
                                   max_primary=100, max_bytes=1_000_000, max_seconds=5,
                                   max_line_bytes=32, progress_every=0)
    assert long_line_run["complete_file_scanned"]
    assert long_line_profiler.malformed["line_exceeds_max_line_bytes"] == 1
    assert long_line_body.closed_by_profiler

    timeout_body = TrackingBytesIO(compressed(fixture))
    timeout_profiler = DatasetProfiler("jobs", seed=7, reservoir_size=0)
    timeout_run = profile_stream(timeout_body, len(timeout_body.getvalue()), timeout_profiler,
                                 max_primary=100, max_bytes=1_000_000, max_seconds=0.001,
                                 max_line_bytes=4096, progress_every=0,
                                 started_at=time.monotonic() - 1)
    assert timeout_run["stop_reason"] == "time_limit"
    assert timeout_body.closed_by_profiler

    byte_body = TrackingBytesIO(compressed(fixture) * 3)
    byte_cap = len(compressed(fixture))
    byte_profiler = DatasetProfiler("jobs", seed=7, reservoir_size=0)
    byte_run = profile_stream(byte_body, len(byte_body.getvalue()), byte_profiler, max_primary=1000,
                              max_bytes=byte_cap, max_seconds=5, max_line_bytes=4096,
                              progress_every=0)
    assert byte_run["compressed_bytes_read"] <= byte_cap
    assert byte_run["stop_reason"] == "compressed_byte_limit"
    assert byte_body.closed_by_profiler

    print("Self-test passed: counters, malformed data, dates, relation matching, duplicate IDs, reservoir, line/byte caps, and stream closure.")


def _counter_percentages(counts: dict[str, int], denominator: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {"count": value, "percent_of_primary_resources": round(100 * value / denominator, 3) if denominator else 0.0}
        for key, value in counts.items()
    }


def _section_comparison(section: dict[str, Any], denominator: int) -> dict[str, Any]:
    return {
        field: _counter_percentages(counts, denominator)
        for field, counts in section["fields"].items()
    }


def _compare_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    metrics = ("absent", "null", "blank", "unexpected_type")
    for previous, current in zip(sections, sections[1:]):
        changes = {}
        fields = set(previous["fields"]) | set(current["fields"])
        for field in sorted(fields):
            left = previous["fields"].get(field, {})
            right = current["fields"].get(field, {})
            field_changes = {}
            for metric in metrics:
                left_count = int(left.get(metric, 0))
                right_count = int(right.get(metric, 0))
                left_rate = 100 * left_count / previous["primary_resources"] if previous["primary_resources"] else 0.0
                right_rate = 100 * right_count / current["primary_resources"] if current["primary_resources"] else 0.0
                field_changes[metric] = {
                    "previous_count": left_count,
                    "previous_percent": round(left_rate, 3),
                    "current_count": right_count,
                    "current_percent": round(right_rate, 3),
                    "percentage_point_change": round(right_rate - left_rate, 3),
                }
            changes[field] = field_changes
        comparisons.append({
            "previous_section": [previous["start_primary_resource"], previous["end_primary_resource"]],
            "current_section": [current["start_primary_resource"], current["end_primary_resource"]],
            "quality_rate_changes": changes,
            "interpretation_scope": "only these successive scanned sections; not representative of the remainder",
        })
    return comparisons


def _scan_one(s3: Any, dataset: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    key = PREFIX + FILES[dataset]
    response = s3.get_object(Bucket=BUCKET, Key=key)
    profiler = DatasetProfiler(dataset, args.seed, args.reservoir_size)
    started = time.monotonic()
    run = profile_stream(
        response["Body"], response.get("ContentLength"), profiler,
        max_primary=args.max_primary, max_bytes=args.max_compressed_mib * 1024 * 1024,
        max_seconds=args.max_seconds, max_line_bytes=args.max_line_bytes,
        progress_every=args.progress_every, started_at=started,
    )
    result = profiler.summary()
    retained_result = {
        "reservoir_sample": result.pop("reservoir_sample"),
        "diagnostic_examples": result.pop("diagnostic_examples"),
        "diagnostic_examples_note": result["diagnostic_examples_note"],
        "random_sample_note": result["random_sample_note"],
    }
    result["run"] = run
    result["fields"] = {
        field: {
            "denominator_primary_resources": profiler.primary_count,
            "counts": _counter_percentages(dict(profiler.field_counts[field]), profiler.primary_count),
            "unexpected_types": dict(profiler.unexpected_types[field]),
        }
        for field in profiler.field_counts
    }
    result["sections_of_10000_primary_resources"] = [
        {
            **section,
            "field_quality_rates": _section_comparison(section, section["primary_resources"]),
        }
        for section in profiler.sections
    ]
    result["successive_section_comparisons"] = _compare_sections(profiler.sections)
    result["category_denominator_primary_resources"] = profiler.primary_count
    result["salary_signature_denominator_nonblank_salary_values"] = sum(profiler.salary_signatures.values())
    result["random_sample_note"] = (
        "Seeded reservoir is random only within the scanned portion unless the file reached EOF."
    )
    result["diagnostic_examples_note"] = "Deliberately selected diagnostic examples; not representative."
    return result, retained_result


def _write_report(results: dict[str, dict[str, Any]], output_path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# PredictLeads Bounded Data-Quality Trial",
        "",
        "This report describes bounded streaming trials, not necessarily complete-file or dataset-wide analysis.",
        "All percentages use the primary-resource denominator stated for each dataset or section.",
        "Reservoir samples are random only within the scanned portion unless the scan reached EOF.",
        "Diagnostic excerpts were deliberately selected and are not representative.",
        "",
    ]
    for dataset, result in results.items():
        run = result["run"]
        denominator = result["primary_resources"]
        lines.extend([
            f"## {dataset.title()}", "",
            f"- Primary resources: {denominator:,}; wrappers: {result['wrappers']:,}; included resources: {result['included_resources']:,}.",
            f"- Stopped: `{run['stop_reason']}` after {run['elapsed_seconds']:.1f}s and {run['compressed_bytes_read']:,} compressed bytes; complete file scanned: `{run['complete_file_scanned']}`.",
            f"- Limits: {run['primary_limit']:,} primary resources, {run['byte_limit']:,} compressed bytes, {run['time_limit_seconds']:.0f}s, {run['max_line_bytes']:,} bytes per decompressed line.",
            f"- Malformed records: {sum(result['malformed_records'].values()):,} total ({result['malformed_records'] or 'none'}); skipped resources: {result['skipped_resources'] or 'none'}.",
            "",
            "### Field quality counts (denominator: primary resources scanned)",
            "| Field | Absent (count, %) | Null (count, %) | Blank (count, %) | Unexpected type (count, %) |",
            "|---|---:|---:|---:|---:|",
        ])
        for field, details in result["fields"].items():
            counts = details["counts"]
            def cell(metric: str) -> str:
                stat = counts.get(metric, {"count": 0, "percent_of_primary_resources": 0.0})
                return f"{stat['count']:,} ({stat['percent_of_primary_resources']:.3f}%)"
            lines.append(
                f"| `{field}` | {cell('absent')} | {cell('null')} | {cell('blank')} | {cell('unexpected_type')} |"
            )
        lines.extend(["", "### Section comparison", ""])
        for section in result["sections_of_10000_primary_resources"]:
            lines.append(
                f"- Primary resources {section['start_primary_resource']:,}–{section['end_primary_resource']:,} "
                f"(n={section['primary_resources']:,}); field quality rates are section-local, not file-wide."
            )
        if len(result["sections_of_10000_primary_resources"]) < 2:
            lines.append("- Only one section was scanned, so section-to-section differences cannot be assessed.")
        for comparison in result["successive_section_comparisons"]:
            changed = []
            for field, metrics in comparison["quality_rate_changes"].items():
                for metric, change in metrics.items():
                    delta = change["percentage_point_change"]
                    if delta:
                        changed.append((abs(delta), f"{field}.{metric} {delta:+.3f} pp"))
            changed.sort(reverse=True)
            change_summary = ", ".join(value for _, value in changed[:3])
            if len(changed) > 3:
                change_summary += f"; {len(changed) - 3} other nonzero rates"
            lines.append(
                f"- Sections {comparison['previous_section'][0]:,}–{comparison['previous_section'][1]:,} "
                f"vs {comparison['current_section'][0]:,}–{comparison['current_section'][1]:,}: "
                + (change_summary if changed else "no absent/null/blank/type-quality rate differences observed")
                + "; scoped to these scanned sections only."
            )
        lines.extend([
            "",
            f"- Observed categories: {json.dumps(result['categories'], ensure_ascii=False)}.",
            f"- Job category assignment counts (multi-valued fields may contribute multiple assignments): {json.dumps(result['job_category_fields'], ensure_ascii=False)}.",
            f"- Salary signatures: {len(result['salary_formats'])} observed combinations among {result.get('salary_signature_denominator_nonblank_salary_values', 0):,} nonblank salary values; top formats only, with no conversions: {json.dumps(result['salary_formats'][:8], ensure_ascii=False)}.",
            f"- Date parse counts/ranges: {json.dumps(result['dates'], ensure_ascii=False)}.",
            f"- Suspicious date ordering flags: {json.dumps(result['suspicious_date_order'], ensure_ascii=False)}. Original values are preserved in samples; these are flags, not corrections.",
            f"- Duplicate primary type/ID pairs: {json.dumps(result['duplicate_primary_type_id_pairs'], ensure_ascii=False)}.",
        ])
        if dataset == "jobs":
            lines.extend([
                f"- Company relationship shapes: {json.dumps(result['company_relationship_shapes'], ensure_ascii=False)}.",
                f"- Company reference matches within each wrapper (resource type and ID): {json.dumps(result['company_relationship_matches'], ensure_ascii=False)}.",
                "- `last_seen_at` is not treated as a closing date, and status is not proof a listing is open today.",
            ])
        lines.append("")
    lines.extend([
        "## Limits and next investigations", "",
        "The trial stops independently for each file at the first reached primary-resource, compressed-byte, or time cap. A resource cap may overshoot within its final wrapper; that amount is recorded. A file is called complete only if decompression reaches EOF.",
        "", "1. Compare broader time-stratified samples and check whether section rates stabilize.",
        "2. Validate lifecycle, status, salary currency/period, and relationship semantics against authoritative documentation.",
        "3. Review deliberately selected excerpts and reservoir samples locally with domain stakeholders before defining data-cleaning rules.", "",
        "No full-file scan or production cleaning rule is performed by this trial.",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(output_dir: Path) -> None:
    command = (
        r'.\.venv\Scripts\python.exe .\code\profile_data_quality.py --dataset both '
        r'--max-primary 100000 --max-compressed-mib 256 --max-seconds 300 '
        r'--max-line-bytes 262144 --reservoir-size 1000 --progress-every 10000 --seed 20261001'
    )
    content = (
        "# Local discovery outputs\n\n"
        "Run from the project root in PowerShell using the project `.venv`.\n\n"
        "```powershell\n" + command + "\n```\n\n"
        "The command streams both S3 files and stops at the first per-file limit; it does not run a full-file scan. "
        "Use `--self-test` to run synthetic fixtures without AWS.\n\n"
        "Outputs are local and ignored by Git: `quality_summary.json`, `quality_report.md`, and `retained_samples.json`. "
        "The reservoir is random only within scanned records unless EOF is reached; diagnostic excerpts are deliberately selected.\n"
    )
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bounded", "audit", "benchmark"), default="bounded")
    parser.add_argument("--dataset", choices=("both", *FILES.keys()), default="both")
    parser.add_argument("--max-primary", type=int, default=100_000)
    parser.add_argument("--max-compressed-mib", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-line-bytes", type=int, default=256 * 1024)
    parser.add_argument("--reservoir-size", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--connect-timeout", type=float, default=10)
    parser.add_argument("--read-timeout", type=float, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("data/discovery"))
    parser.add_argument("--section-size", type=int, default=100_000)
    parser.add_argument("--audit-max-line-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--field-path-cap", type=int, default=20_000)
    parser.add_argument("--category-cap", type=int, default=200)
    parser.add_argument("--sample-record-bytes", type=int, default=16_384)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--self-test", action="store_true", help="run offline synthetic tests; no AWS calls")
    args = parser.parse_args(argv)
    if args.self_test or args.mode == "self-test":
        _run_self_test()
        _run_full_audit_self_test()
        return 0
    if args.mode in {"audit", "benchmark"}:
        return _run_full_audit(args)
    positive = (args.max_primary, args.max_compressed_mib, args.max_seconds,
                args.max_line_bytes, args.reservoir_size, args.connect_timeout, args.read_timeout)
    if any(value <= 0 for value in positive) or args.progress_every < 0:
        parser.error("limits/timeouts/sample size must be positive and progress interval nonnegative")

    datasets = list(FILES) if args.dataset == "both" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    config = botocore.config.Config(
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    s3 = session.client("s3", config=config)
    results: dict[str, dict[str, Any]] = {}
    retained: dict[str, Any] = {}
    try:
        for dataset in datasets:
            try:
                result, retained_result = _scan_one(s3, dataset, args)
            except KeyboardInterrupt:
                print(f"[{dataset}] interrupted; current stream closed. No partial result emitted for this file.", file=sys.stderr)
                raise
            results[dataset] = result
            retained[dataset] = {
                "reservoir_sample": retained_result["reservoir_sample"],
                "diagnostic_examples": retained_result["diagnostic_examples"],
                "diagnostic_examples_note": retained_result["diagnostic_examples_note"],
                "random_sample_note": retained_result["random_sample_note"],
            }
            run = result["run"]
            print(
                f"[{dataset}] stopped={run['stop_reason']} complete_file={run['complete_file_scanned']} "
                f"primary={result['primary_resources']} included={result['included_resources']} "
                f"bytes={run['compressed_bytes_read']} elapsed={run['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        s3.close()

    summary_path = args.output_dir / "quality_summary.json"
    samples_path = args.output_dir / "retained_samples.json"
    report_path = args.output_dir / "quality_report.md"
    summary_path.write_text(json.dumps({"trial_limits": vars(args) | {"output_dir": str(args.output_dir)}, "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    samples_path.write_text(json.dumps(retained, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_report(results, report_path, args)
    _write_readme(args.output_dir)
    print(f"Saved local outputs under {args.output_dir}: {summary_path.name}, {report_path.name}, {samples_path.name}")
    return 0


class _BoundedCounter:
    def __init__(self, cap: int):
        self.cap = cap
        self.values: Counter[str] = Counter()
        self.dropped = 0

    def add(self, value: Any, count: int = 1) -> None:
        key = "<null>" if value is None else str(value)
        if key in self.values or len(self.values) < self.cap:
            self.values[key] += count
        else:
            self.dropped += count

    def snapshot(self) -> dict[str, Any]:
        return {"top_values": dict(self.values.most_common()), "other_capped_count": self.dropped,
                "distinct_values_retained": len(self.values), "distinct_value_cap": self.cap}


def _parent_field_path(path: str) -> str:
    if path.endswith("[]"):
        return path[:-2]
    return path.rsplit(".", 1)[0] if "." in path else ""


def _field_observations(resource: dict[str, Any]) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = defaultdict(list)

    def visit(value: Any, path: str, depth: int) -> None:
        if not path or depth > 32:
            return
        found[path].append(value)
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                visit(child, child_path[:512], depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, f"{path}[]"[:512], depth + 1)

    for key, value in resource.items():
        visit(value, str(key)[:512], 1)
    return found


def _resource_type_name(resource: dict[str, Any]) -> str:
    value = resource.get("type")
    return str(value) if value is not None else "<missing_type>"


def _type_name(value: Any) -> str:
    if value is None:
        return "NoneType"
    if isinstance(value, bool):
        return "bool"
    return type(value).__name__


def _field_state(attributes: dict[str, Any], name: str) -> str:
    if name not in attributes:
        return "absent"
    value = attributes[name]
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank"
    return "present"


class FullAuditProfiler:
    """Full-stream profiler with capped dictionaries and fixed-size samples."""

    QUALITY_FIELDS = ("description", "salary", "location", "status", "seniority", "language", "posted_at")
    COMPANY_QUALITY_FIELDS = ("company_name", "domain", "description", "location")
    DESCRIPTION_BUCKETS = (0, 1, 50, 200, 500, 1000, 5000, 20000, 100000, 1000000)

    def __init__(self, dataset: str, seed: int, reservoir_size: int, *, field_path_cap: int,
                 category_cap: int, sample_record_bytes: int):
        self.dataset = dataset
        self.seed = seed
        self.rng = random.Random(seed)
        self.reservoir_size = reservoir_size
        self.field_path_cap = field_path_cap
        self.category_cap = category_cap
        self.sample_record_bytes = sample_record_bytes
        self.line_count = 0
        self.wrapper_count = 0
        self.primary_count = 0
        self.included_count = 0
        self.primary_types: Counter[str] = Counter()
        self.included_types: Counter[str] = Counter()
        self.resource_type_cap = 100
        self.resource_types_dropped = Counter()
        self.malformed: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()
        self.schema: dict[str, dict[str, Any]] = {}
        self.category_values = {name: _BoundedCounter(category_cap) for name in (
            "status", "language", "seniority", "categories", "contract_types",
            "salary_currency", "salary_pay_period", "location_country", "location_region",
            "location_category",
        )}
        self.salary_format = _BoundedCounter(category_cap)
        self.date_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.date_ranges: dict[str, list[datetime | None]] = defaultdict(lambda: [None, None])
        self.date_order: Counter[str] = Counter()
        self.description_lengths: dict[str, Counter[str]] = defaultdict(Counter)
        self.description_totals: Counter[str] = Counter()
        self.description_max: Counter[str] = Counter()
        self.relationship_shapes: Counter[str] = Counter()
        self.relationship_matches: Counter[str] = Counter()
        self.group_stats: dict[str, dict[str, Any]] = {}
        self.group_values = _BoundedCounter(category_cap)
        self.group_dropped_resources = 0
        self.sections_completed = 0
        self.section_sink: Any = None
        self.section_rate_ranges: dict[str, dict[str, list[float | None]]] = defaultdict(
            lambda: defaultdict(lambda: [None, None])
        )
        self._section: dict[str, Any] | None = None
        self.reservoir: list[dict[str, Any]] = []
        self.last_checkpoint = time.monotonic()
        self.last_progress = self.last_checkpoint

    def _schema_update(self, collection: str, resource: dict[str, Any], resource_type: str | None = None) -> None:
        resource_type = resource_type or _resource_type_name(resource)
        key = f"{collection}:{resource_type}"
        found = _field_observations(resource)
        profile = self.schema.get(key)
        if profile is None:
            profile = {"resource_count": 0, "fields": {}, "field_paths_dropped": 0}
            self.schema[key] = profile
        previous_count = profile["resource_count"]
        fields = profile["fields"]
        for path, stats in fields.items():
            if path in found:
                continue
            parent = _parent_field_path(path)
            parent_present = parent in found if parent else True
            stats["missing_parent_records" if not parent_present else "missing_child_records"] += 1
        for path, values in found.items():
            if path not in fields:
                if len(fields) >= self.field_path_cap:
                    profile["field_paths_dropped"] += 1
                    continue
                parent = _parent_field_path(path)
                parent_seen = fields.get(parent, {}).get("present_records", 0) if parent else previous_count
                absent_parent = max(0, previous_count - parent_seen) if parent else 0
                absent_child = previous_count - absent_parent
                fields[path] = {
                    "present_records": 0,
                    "missing_parent_records": absent_parent,
                    "missing_child_records": absent_child,
                    "value_instances": 0,
                    "null_values": 0,
                    "blank_values": 0,
                    "value_type_counts": Counter(),
                }
            stats = fields.get(path)
            if stats is None:
                continue
            stats["present_records"] += 1
            stats["value_instances"] += len(values)
            for value in values:
                stats["value_type_counts"][_type_name(value)] += 1
                if value is None:
                    stats["null_values"] += 1
                elif isinstance(value, str) and not value.strip():
                    stats["blank_values"] += 1
        profile["resource_count"] += 1

    def _start_section(self) -> dict[str, Any]:
        return {"start_primary_resource": self.primary_count + 1, "end_primary_resource": self.primary_count,
                "primary_resources": 0, "fields": defaultdict(Counter),
                "categories": {name: _BoundedCounter(50) for name in ("status", "seniority", "language")}}

    def _finish_section(self) -> dict[str, Any] | None:
        if self._section is None:
            return None
        section = self._section
        result = {
            "dataset": self.dataset,
            "start_primary_resource": section["start_primary_resource"],
            "end_primary_resource": section["end_primary_resource"],
            "primary_resources": section["primary_resources"],
            "field_states": {key: dict(value) for key, value in section["fields"].items()},
            "categories": {key: value.snapshot() for key, value in section["categories"].items()},
        }
        self.sections_completed += 1
        denominator = section["primary_resources"]
        if denominator:
            fields_for_dataset = self.QUALITY_FIELDS if self.dataset == "jobs" else self.COMPANY_QUALITY_FIELDS
            for field in fields_for_dataset:
                counts = section["fields"].get(field, {})
                for state in ("absent", "null", "blank"):
                    rate = 100 * counts.get(state, 0) / denominator
                    bounds = self.section_rate_ranges[field][state]
                    bounds[0] = rate if bounds[0] is None else min(bounds[0], rate)
                    bounds[1] = rate if bounds[1] is None else max(bounds[1], rate)
        if self.section_sink is not None:
            self.section_sink(result)
        self._section = None
        return result

    def _compact_sample(self, resource: dict[str, Any]) -> dict[str, Any]:
        attributes = resource.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        selected = (
            "title", "translated_title", "normalized_title", "description", "salary", "salary_data",
            "location", "location_data", "status", "seniority", "language", "categories",
            "contract_types", "posted_at", "first_seen_at", "last_seen_at", "last_processed_at",
            "company_name", "domain",
        )
        compact_attrs: dict[str, Any] = {}
        for name in selected:
            if name not in attributes:
                continue
            value = attributes[name]
            if name == "description" and isinstance(value, str):
                value = value[: min(4096, self.sample_record_bytes // 3)]
            compact_attrs[name] = self._bounded_sample_value(value, 512)
        relationships = resource.get("relationships")
        company_ref = None
        if isinstance(relationships, dict):
            rel = relationships.get("company")
            data = rel.get("data") if isinstance(rel, dict) else None
            if isinstance(data, dict):
                company_ref = {"type": data.get("type"), "id": data.get("id")}
        sample = {"type": resource.get("type"), "id": resource.get("id"), "attributes": compact_attrs}
        if company_ref is not None:
            sample["relationships"] = {"company": {"data": company_ref}}
        rendered = json.dumps(sample, ensure_ascii=False, default=str)
        if len(rendered.encode("utf-8")) > self.sample_record_bytes:
            if isinstance(compact_attrs.get("description"), str):
                compact_attrs["description"] = compact_attrs["description"][:256]
            for name, value in list(compact_attrs.items()):
                if isinstance(value, str) and len(value) > 256:
                    compact_attrs[name] = value[:256]
                elif isinstance(value, list):
                    compact_attrs[name] = value[:5]
        return sample

    def _bounded_sample_value(self, value: Any, string_cap: int) -> Any:
        if isinstance(value, str):
            return value[:string_cap]
        if isinstance(value, dict):
            return {str(key)[:128]: self._bounded_sample_value(item, string_cap)
                    for key, item in list(value.items())[:20]}
        if isinstance(value, list):
            return [self._bounded_sample_value(item, string_cap) for item in value[:10]]
        return _json_safe(value)

    def _reservoir_add(self, resource: dict[str, Any]) -> None:
        if not self.reservoir_size:
            return
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append({"resource_type": resource.get("type"), "resource": self._compact_sample(resource)})
            return
        slot = self.rng.randrange(self.primary_count)
        if slot < self.reservoir_size:
            self.reservoir[slot] = {"resource_type": resource.get("type"), "resource": self._compact_sample(resource)}

    def _record_description(self, collection: str, attributes: dict[str, Any]) -> None:
        if "description" not in attributes:
            self.description_lengths[collection]["absent"] += 1
            return
        value = attributes["description"]
        if value is None:
            self.description_lengths[collection]["null"] += 1
        elif isinstance(value, str) and not value.strip():
            self.description_lengths[collection]["blank"] += 1
        elif isinstance(value, str):
            size = len(value)
            bucket = next((f"{edge}+" for edge in reversed(self.DESCRIPTION_BUCKETS) if size >= edge), "0")
            self.description_lengths[collection][bucket] += 1
            self.description_totals[collection] += size
            self.description_max[collection] = max(self.description_max[collection], size)
        else:
            self.description_lengths[collection][f"unexpected_{type(value).__name__}"] += 1

    def _record_job_values(self, resource: dict[str, Any], section: dict[str, Any]) -> None:
        attributes = resource.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        for field in self.QUALITY_FIELDS:
            state = _field_state(attributes, field)
            section["fields"][field][state] += 1
        for field in ("status", "seniority", "language"):
            value = attributes.get(field, "<absent>")
            self.category_values[field].add(value)
            section["categories"][field].add(value)
        for field in ("categories", "contract_types"):
            value = attributes.get(field)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is not None:
                    self.category_values[field].add(item)
        salary = attributes.get("salary")
        if salary is not None and str(salary).strip():
            shape, currency, period = _salary_signature(salary)
            self.salary_format.add(f"{shape}|{currency}|{period}")
        salary_data = attributes.get("salary_data")
        if isinstance(salary_data, dict):
            self.category_values["salary_currency"].add(salary_data.get("salary_currency"))
            self.category_values["salary_pay_period"].add(salary_data.get("salary_time_unit"))
        location_data = attributes.get("location_data")
        if isinstance(location_data, list):
            for item in location_data[:10]:
                if isinstance(item, dict):
                    self.category_values["location_country"].add(item.get("country"))
                    self.category_values["location_region"].add(item.get("region"))
        parsed: dict[str, datetime | None] = {}
        for field in DATE_FIELDS:
            value = attributes.get(field, "<absent>")
            if value == "<absent>":
                self.date_counts[field]["absent"] += 1
                parsed[field] = None
            elif value is None:
                self.date_counts[field]["null"] += 1
                parsed[field] = None
            elif isinstance(value, str) and not value.strip():
                self.date_counts[field]["blank"] += 1
                parsed[field] = None
            else:
                timestamp = _parse_timestamp(value)
                if timestamp is None:
                    self.date_counts[field]["parse_failure"] += 1
                    parsed[field] = None
                else:
                    self.date_counts[field]["parse_success"] += 1
                    parsed[field] = timestamp
                    bounds = self.date_ranges[field]
                    bounds[0] = timestamp if bounds[0] is None else min(bounds[0], timestamp)
                    bounds[1] = timestamp if bounds[1] is None else max(bounds[1], timestamp)
        for first_name, second_name in (("posted_at", "first_seen_at"), ("first_seen_at", "last_seen_at"),
                                        ("last_seen_at", "last_processed_at")):
            first, second = parsed.get(first_name), parsed.get(second_name)
            if first is not None and second is not None and first > second:
                self.date_order[f"{first_name}_after_{second_name}"] += 1

        groups: set[str] = set()
        seniority = attributes.get("seniority")
        if seniority is not None:
            groups.add(f"seniority:{str(seniority)[:256]}")
        categories = attributes.get("categories", [])
        if isinstance(categories, list):
            groups.update(f"category:{str(item)[:256]}" for item in categories[:100] if item is not None)
        for group in groups:
            if group not in self.group_stats and len(self.group_stats) >= self.category_cap:
                self.group_dropped_resources += 1
                continue
            stats = self.group_stats.setdefault(group, {"group_size": 0, "missing": Counter()})
            stats["group_size"] += 1
            for field in ("description", "salary", "location", "status"):
                if _field_state(attributes, field) != "present":
                    stats["missing"][field] += 1

        self._record_description("primary_jobs", attributes)
        section["end_primary_resource"] = self.primary_count
        section["primary_resources"] += 1

    def _record_relationship(self, resource: dict[str, Any], included_keys: set[tuple[str, str]]) -> None:
        relationships = resource.get("relationships")
        if not isinstance(relationships, dict) or "company" not in relationships:
            self.relationship_shapes["relationship_absent"] += 1
            return
        relation = relationships["company"]
        if relation is None:
            self.relationship_shapes["relationship_null"] += 1
            return
        if not isinstance(relation, dict):
            self.relationship_shapes[f"relationship_{type(relation).__name__}"] += 1
            return
        data = relation.get("data", "<absent>")
        if data == "<absent>":
            self.relationship_shapes["data_absent"] += 1
            return
        values = data if isinstance(data, list) else [data]
        shape = "data_list" if isinstance(data, list) else "data_object" if isinstance(data, dict) else "data_null" if data is None else f"data_{type(data).__name__}"
        self.relationship_shapes[shape] += 1
        for item in values:
            if item is None:
                self.relationship_matches["null_reference"] += 1
                continue
            if not isinstance(item, dict):
                self.skipped["company_relationship_item_not_object"] += 1
                continue
            if item.get("type") is None or item.get("id") is None:
                self.relationship_matches["incomplete_reference"] += 1
                continue
            key = (str(item["type"]), str(item["id"]))
            result = "matched_same_wrapper" if key in included_keys else "unresolved_same_wrapper"
            self.relationship_matches[result] += 1

    def add_resource(self, collection: str, resource: Any, included_keys: set[tuple[str, str]], section: dict[str, Any] | None = None) -> None:
        if not isinstance(resource, dict):
            self.skipped[f"{collection}_resource_not_object"] += 1
            return
        resource_type = _resource_type_name(resource)
        counts_by_type = self.primary_types if collection == "primary" else self.included_types
        if resource_type not in counts_by_type and len(counts_by_type) >= self.resource_type_cap:
            self.resource_types_dropped[collection] += 1
            resource_type = "<other_type_capped>"
        if collection == "primary":
            self.primary_count += 1
            self.primary_types[resource_type] += 1
        else:
            self.included_count += 1
            self.included_types[resource_type] += 1
        self._schema_update(collection, resource, resource_type)
        attributes = resource.get("attributes")
        if attributes is None:
            attributes = {}
        elif not isinstance(attributes, dict):
            self.skipped[f"{collection}_attributes_not_object"] += 1
            attributes = {}
        if collection == "primary":
            if self.dataset == "jobs":
                if section is not None:
                    self._record_job_values(resource, section)
                self._record_relationship(resource, included_keys)
            else:
                self._record_description("primary_companies", attributes)
                location_data = attributes.get("location_data")
                if isinstance(location_data, list):
                    for item in location_data[:10]:
                        if isinstance(item, dict):
                            self.category_values["location_country"].add(item.get("country"))
                            self.category_values["location_region"].add(item.get("region"))
                            self.category_values["location_category"].add(item.get("category"))
                if section is not None:
                    for field in self.COMPANY_QUALITY_FIELDS:
                        section["fields"][field][_field_state(attributes, field)] += 1
                    section["end_primary_resource"] = self.primary_count
                    section["primary_resources"] += 1
            self._reservoir_add(resource)
        else:
            self._record_description("included_resources", attributes)

    def _group_summary(self) -> dict[str, Any]:
        return {
            key: {
                "group_size_denominator": value["group_size"],
                "missing_counts": dict(value["missing"]),
                "missing_percent": {
                    field: round(100 * count / value["group_size"], 3) if value["group_size"] else 0.0
                    for field, count in value["missing"].items()
                },
            }
            for key, value in sorted(self.group_stats.items())
        }

    def summary(self) -> dict[str, Any]:
        schema = {}
        for group, profile in self.schema.items():
            fields = {}
            for path, stats in profile["fields"].items():
                denom = profile["resource_count"]
                fields[path] = {
                    "denominator_resources_of_this_type": denom,
                    "present_in_resources": stats["present_records"],
                    "value_instances": stats["value_instances"],
                    "missing_parent_records": stats["missing_parent_records"],
                    "missing_child_records": stats["missing_child_records"],
                    "null_values": stats["null_values"],
                    "blank_values": stats["blank_values"],
                    "value_type_counts": dict(stats["value_type_counts"]),
                }
            schema[group] = {"resource_count": profile["resource_count"], "fields": fields,
                             "field_paths_dropped_at_cap": profile["field_paths_dropped"],
                             "field_path_cap": self.field_path_cap}
        return {
            "dataset": self.dataset,
            "lines_or_wrappers": self.line_count,
            "valid_object_wrappers": self.wrapper_count,
            "primary_resources": self.primary_count,
            "included_resources": self.included_count,
            "primary_resource_types": dict(self.primary_types),
            "included_resource_types": dict(self.included_types),
            "resource_types_dropped_at_cap": dict(self.resource_types_dropped),
            "resource_type_cap_per_collection": self.resource_type_cap,
            "malformed_records": dict(self.malformed),
            "skipped_resources": dict(self.skipped),
            "schema_by_collection_and_type": schema,
            "bounded_categories": {key: counter.snapshot() for key, counter in self.category_values.items()},
            "salary_format_signatures": self.salary_format.snapshot(),
            "date_parsing": {
                field: {**dict(counts),
                        "minimum_utc": self.date_ranges[field][0].isoformat() if self.date_ranges[field][0] else None,
                        "maximum_utc": self.date_ranges[field][1].isoformat() if self.date_ranges[field][1] else None}
                for field, counts in self.date_counts.items()
            },
            "unusual_date_order_flags": dict(self.date_order),
            "description_lengths": {
                key: {"length_buckets": dict(value), "nonblank_characters_total": self.description_totals[key],
                      "maximum_characters": self.description_max[key]}
                for key, value in self.description_lengths.items()
            },
            "job_missingness_by_category_and_seniority": self._group_summary() if self.dataset == "jobs" else {},
            "group_values_dropped_resources": self.group_dropped_resources,
            "company_relationship_shapes": dict(self.relationship_shapes),
            "same_wrapper_company_relationship_matches": dict(self.relationship_matches),
            "sections_completed": self.sections_completed,
            "section_field_rate_ranges_percent": {
                field: {state: {"minimum_percent": bounds[0], "maximum_percent": bounds[1]}
                        for state, bounds in states.items()}
                for field, states in self.section_rate_ranges.items()
            },
            "reservoir_sample_count": len(self.reservoir),
            "reservoir_seed": self.seed,
            "reservoir_scope": "eligible parsed primary resources from the complete stream if and only if EOF was reached; skipped records are not eligible",
            "measurement_limits": {
                "field_path_cap_per_collection_type": self.field_path_cap,
                "category_value_cap_each_dimension": self.category_cap,
                "resource_type_cap_per_collection": self.resource_type_cap,
                "group_cap": self.category_cap,
                "group_category_membership_cap_per_job": 100,
                "location_entries_inspected_per_resource": 10,
                "sample_record_projection_byte_cap": self.sample_record_bytes,
            },
        }


class _AuditReader:
    def __init__(self, stream: BinaryIO, deadline: float | None = None):
        self.stream = stream
        self.deadline = deadline
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise ScanDeadlineExceeded("time_limit")
        data = self.stream.read(size)
        self.bytes_read += len(data)
        return data


def _working_set_bytes() -> int | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        get_info.restype = ctypes.c_int
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        ok = get_info(current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    except Exception:
        return None


def _peak_working_set_bytes() -> int | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        get_info.restype = ctypes.c_int
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        ok = get_info(current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def _audit_sample_document(profiler: FullAuditProfiler, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": profiler.dataset,
        "seed": profiler.seed,
        "sample_count": len(profiler.reservoir),
        "eligibility": "successfully parsed primary resource dictionaries only; malformed wrappers and skipped non-object resources excluded",
        "representativeness": "seeded reservoir is random across the scanned eligible stream; on incomplete scans it covers only the scanned portion",
        "records_are_compact_projections": True,
        "records": profiler.reservoir,
    }


def _audit_process_line(raw_line: bytes, profiler: FullAuditProfiler) -> None:
    profiler.line_count += 1
    try:
        wrapper = json.loads(raw_line.decode("utf-8"))
    except UnicodeDecodeError:
        profiler.malformed["invalid_utf8"] += 1
        return
    except json.JSONDecodeError:
        profiler.malformed["invalid_json"] += 1
        return
    if not isinstance(wrapper, dict):
        profiler.malformed["wrapper_not_object"] += 1
        return
    profiler.wrapper_count += 1
    primary = wrapper.get("data", "<absent>")
    if primary == "<absent>":
        profiler.malformed["data_absent"] += 1
        return
    if not isinstance(primary, list):
        profiler.malformed["data_not_array"] += 1
        return
    included = wrapper.get("included", [])
    if included is None:
        included = []
    elif not isinstance(included, list):
        profiler.malformed["included_not_array"] += 1
        included = []
    included_keys: set[tuple[str, str]] = set()
    for resource in included:
        if not isinstance(resource, dict):
            profiler.skipped["included_resource_not_object"] += 1
            continue
        if resource.get("type") is not None and resource.get("id") is not None:
            included_keys.add((str(resource["type"]), str(resource["id"])))

    section = None
    for resource in primary:
        if not isinstance(resource, dict):
            profiler.skipped["primary_resource_not_object"] += 1
            continue
        if profiler._section is None:
            profiler._section = profiler._start_section()
        section = profiler._section
        profiler.add_resource("primary", resource, included_keys, section)
        if section["primary_resources"] >= getattr(profiler, "section_size", 100_000):
            profiler._finish_section()
    for resource in included:
        profiler.add_resource("included", resource, included_keys)


def _audit_stream(stream: BinaryIO, object_size: int | None, profiler: FullAuditProfiler,
                  *, max_seconds: float | None, max_line_bytes: int, section_size: int,
                  progress_seconds: float, section_path: Path, checkpoint_callback: Any,
                  started_at: float | None = None) -> dict[str, Any]:
    started = time.monotonic() if started_at is None else started_at
    deadline = started + max_seconds if max_seconds is not None else None
    reader = _AuditReader(stream, deadline)
    profiler.section_size = section_size
    pending = bytearray()
    discarding_oversized = False
    complete_eof = False
    stop_reason = "eof"
    stream_error: dict[str, str] | None = None
    last_progress = started
    last_checkpoint = started

    def write_section(section: dict[str, Any] | None) -> None:
        if section is None:
            return
        with section_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(section, ensure_ascii=False, default=str) + "\n")
            output.flush()

    profiler.section_sink = write_section

    def process_line(line: bytes) -> None:
        _audit_process_line(line, profiler)

    try:
        decompressor = zstd.ZstdDecompressor().stream_reader(reader, read_across_frames=True, closefd=False)
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    stop_reason = "benchmark_time_limit"
                    break
                try:
                    chunk = decompressor.read(64 * 1024)
                except ScanDeadlineExceeded:
                    stop_reason = "benchmark_time_limit"
                    break
                except zstd.ZstdError:
                    profiler.malformed["zstandard_decompression_error"] += 1
                    stop_reason = "decompression_error"
                    stream_error = {"type": "ZstdError", "message": "zstandard decompression failed; inspect malformed input and stream logs"}
                    break
                if not chunk:
                    complete_eof = True
                    stop_reason = "eof"
                    if discarding_oversized:
                        profiler.line_count += 1
                        profiler.malformed["oversized_line"] += 1
                    elif pending:
                        profiler.malformed["unterminated_final_line"] += 1
                        process_line(bytes(pending))
                        pending.clear()
                    break

                remaining = chunk
                while remaining:
                    if discarding_oversized:
                        newline = remaining.find(b"\n")
                        if newline < 0:
                            remaining = b""
                            continue
                        profiler.line_count += 1
                        profiler.malformed["oversized_line"] += 1
                        discarding_oversized = False
                        remaining = remaining[newline + 1:]
                        continue
                    newline = remaining.find(b"\n")
                    if newline < 0:
                        if len(pending) + len(remaining) > max_line_bytes:
                            pending.clear()
                            discarding_oversized = True
                        else:
                            pending.extend(remaining)
                        remaining = b""
                        continue
                    segment = remaining[:newline]
                    if len(pending) + len(segment) > max_line_bytes:
                        pending.clear()
                        profiler.line_count += 1
                        profiler.malformed["oversized_line"] += 1
                    else:
                        if pending:
                            pending.extend(segment)
                            process_line(bytes(pending))
                            pending.clear()
                        else:
                            process_line(segment)
                    remaining = remaining[newline + 1:]

                now = time.monotonic()
                if now - last_progress >= progress_seconds:
                    elapsed = now - started
                    rate = profiler.primary_count / elapsed if elapsed else 0
                    memory = _working_set_bytes()
                    print(
                        f"[{profiler.dataset}] progress primary={profiler.primary_count} wrappers={profiler.wrapper_count} "
                        f"included={profiler.included_count} compressed_bytes={reader.bytes_read} "
                        f"elapsed_seconds={elapsed:.1f} primary_per_second={rate:.1f} "
                        f"working_set_bytes={memory}", flush=True,
                    )
                    last_progress = now
                if now - last_checkpoint >= 60:
                    checkpoint_callback(profiler, {
                        "stop_reason": "in_progress", "complete_file_scanned": False,
                        "compressed_bytes_read": reader.bytes_read,
                        "object_size_compressed_bytes": object_size,
                        "elapsed_seconds": round(now - started, 3),
                        "primary_resources": profiler.primary_count,
                        "included_resources": profiler.included_count,
                    })
                    last_checkpoint = now
        finally:
            decompressor.close()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        raise
    except Exception as error:
        profiler.malformed[f"stream_error:{type(error).__name__}"] += 1
        stop_reason = "stream_error"
        message = re.sub(r"(?i)(secret|token|credential|signature|authorization)(\s*[=:]\s*)\S+", r"\1\2<redacted>", str(error))
        stream_error = {"type": type(error).__name__, "message": message[:500]}
    finally:
        stream.close()

    write_section(profiler._finish_section())
    elapsed = time.monotonic() - started
    memory = _working_set_bytes()
    return {
        "stop_reason": stop_reason,
        "error": stream_error,
        "complete_file_scanned": complete_eof and stop_reason == "eof",
        "lines_or_wrappers": profiler.line_count,
        "valid_object_wrappers": profiler.wrapper_count,
        "primary_resources": profiler.primary_count,
        "included_resources": profiler.included_count,
        "compressed_bytes_read": reader.bytes_read,
        "object_size_compressed_bytes": object_size,
        "elapsed_seconds": round(elapsed, 3),
        "primary_resources_per_second": round(profiler.primary_count / elapsed, 3) if elapsed else None,
        "compressed_mib_per_second": round(reader.bytes_read / (1024 * 1024) / elapsed, 3) if elapsed else None,
        "working_set_bytes_at_end": memory,
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "max_decompressed_line_bytes": max_line_bytes,
        "section_size_primary_resources": section_size,
        "uninspected_records": 0 if complete_eof else "unknown remainder; file was not fully scanned",
        "measurement_note": "EOF means every compressed byte reached the decompressor; malformed or oversized lines still count as inspected input and are explicitly recorded.",
    }


def _write_full_report(path: Path, results: dict[str, dict[str, Any]], failure_notes: dict[str, str]) -> None:
    lines = [
        "# PredictLeads Full Streaming Audit",
        "",
        "This report covers each input stream only if its run record says `complete_file_scanned: true`. A partial/error result is never described as a full audit.",
        "Primary and `included` resources are counted/profiled separately. Counts are exact for successfully read input except capped frequency summaries and capped schema paths, which disclose overflow.",
        "",
    ]
    for dataset, result in results.items():
        run = result["run"]
        compressed_bytes = run.get("compressed_bytes_read")
        compressed_text = f"{compressed_bytes:,}" if isinstance(compressed_bytes, int) else "unavailable"
        lines.extend([
            f"## {dataset.title()}", "",
            f"- Source: `s3://{BUCKET}/{PREFIX + FILES[dataset]}`.",
            f"- Object metadata: compressed size={run.get('object_size_compressed_bytes')} bytes; ETag={run.get('etag')}; last modified={run.get('last_modified')}.",
            f"- Result: `{run['stop_reason']}`; EOF reached and complete stream inspected: `{run['complete_file_scanned']}`.",
            f"- Read: {compressed_text} compressed bytes in {run['elapsed_seconds']:.1f}s; primary={result['primary_resources']:,}; wrappers={result['valid_object_wrappers']:,}; included={result['included_resources']:,}; lines={result['lines_or_wrappers']:,}.",
            f"- Uninspected records: {run.get('uninspected_records', 'unknown')}; malformed input lines and oversized records are counted in the malformed totals below, not silently parsed.",
            f"- Primary throughput={run.get('primary_resources_per_second')} resources/s; compressed throughput={run.get('compressed_mib_per_second')} MiB/s; end working set={run.get('working_set_bytes_at_end')} bytes.",
            f"- Malformed input: {json.dumps(result['malformed_records'], ensure_ascii=False)}; skipped resources: {json.dumps(result['skipped_resources'], ensure_ascii=False)}.",
            f"- Resource types: primary={json.dumps(result['primary_resource_types'], ensure_ascii=False)}; included={json.dumps(result['included_resource_types'], ensure_ascii=False)}.",
        ])
        if run.get("error"):
            lines.append(f"- Stream error: `{run['error']['type']}`: {run['error']['message']}")
            lines.append("")
        if dataset == "jobs":
            lines.extend([
                f"- Same-wrapper company references: {json.dumps(result['same_wrapper_company_relationship_matches'], ensure_ascii=False)}; shapes={json.dumps(result['company_relationship_shapes'], ensure_ascii=False)}.",
                f"- Job statuses: {json.dumps(result['bounded_categories']['status'], ensure_ascii=False)}.",
                f"- Date ordering flags: {json.dumps(result['unusual_date_order_flags'], ensure_ascii=False)}; `first_seen_at`, `last_seen_at`, and `last_processed_at` retain documented distinctions and are not closing-date assumptions.",
                f"- Missingness by category/seniority is bounded to {result['measurement_limits']['group_cap']} groups; category membership overlap is allowed, so group denominators overlap.",
            ])
        lines.extend([
            f"- Description lengths/states: {json.dumps(result['description_lengths'], ensure_ascii=False)}.",
            f"- Schema field-path profiling uses normalized `[]` array paths, per-resource-type denominators, and a {result['measurement_limits']['field_path_cap_per_collection_type']:,}-path cap per collection/type; dropped path count={sum(v['field_paths_dropped_at_cap'] for v in result['schema_by_collection_and_type'].values())}.",
            "",
        ])
        key_fields = ("description", "salary", "location", "status", "seniority", "language", "posted_at") if dataset == "jobs" else ("company_name", "domain", "description", "location")
        lines.extend(["### Core field coverage", "", "| Resource type | Field | Denominator | Absent | Null values | Blank values | Value types |", "|---|---|---:|---:|---:|---:|---|"])
        for group, profile in result["schema_by_collection_and_type"].items():
            collection, _, resource_type = group.partition(":")
            if collection != "primary":
                continue
            for field in key_fields:
                path_key = f"attributes.{field}"
                stats = profile["fields"].get(path_key)
                if stats is None:
                    lines.append(f"| `{resource_type}` | `{field}` | {profile['resource_count']:,} | {profile['resource_count']:,} | 0 | 0 | not observed |")
                    continue
                absent = stats["missing_parent_records"] + stats["missing_child_records"]
                types = ", ".join(f"{name}:{count:,}" for name, count in sorted(stats["value_type_counts"].items()))
                lines.append(f"| `{resource_type}` | `{field}` | {stats['denominator_resources_of_this_type']:,} | {absent:,} | {stats['null_values']:,} | {stats['blank_values']:,} | {types} |")
        lines.append("")
        if dataset == "jobs":
            lines.extend([
                f"- Bounded status/language/seniority/category/contract/salary/location summaries: `{json.dumps(result['bounded_categories'], ensure_ascii=False)}`.",
                f"- Date parsing/ranges: `{json.dumps(result['date_parsing'], ensure_ascii=False)}`.",
                f"- Missingness group cap: {result['measurement_limits']['group_cap']}; group sample={json.dumps(list(result['job_missingness_by_category_and_seniority'].items())[:12], ensure_ascii=False)}. Categories can overlap; group denominators overlap, and omitted groups are counted ({result['group_values_dropped_resources']}).",
                f"- Within-section field-state rate spans: `{json.dumps(result['section_field_rate_ranges_percent'], ensure_ascii=False)}`.",
                "",
            ])
    if failure_notes:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {name}: {note}" for name, note in failure_notes.items())
        lines.append("")
    lines.extend([
        "## Interpretation and Follow-up",
        "",
        "Available, promising investigation signals include job title/description/tags/categories, structured location, seniority/language, company UUID/domain, and salary data where populated. Reliability varies by field; missingness, source-derived classifications, salary conversions, and temporal semantics need to remain visible.",
        "",
        "Three narrow product-problem proposals (not conclusions):",
        "1. Evidence-backed skill search: require a query/profile skill and a source description/tag span for each claimed match; risk: parsing can lose negation or required/preferred context. Evaluate with human-labeled relevance judgments and recall@k/precision@k.",
        "2. Location-aware job discovery: require an explicit requested region and structured location evidence or a clearly labeled unknown; risk: fuzzy/missing locations and remote semantics. Evaluate filter correctness on labeled city/region/remote cases.",
        "3. Compensation-transparent filtering: require parsed amount plus known currency and pay period for numeric comparison, otherwise expose raw text/not provided; risk: ambiguous salary text and conversion methodology. Evaluate parse coverage and exactness against manually labeled salary examples.",
        "",
        "Follow-up: disk-backed duplicate detection and separate companies-file join validation were deliberately not run. No candidate PII was used, no source data was changed, and no model or hiring probability was inferred.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_full_audit_self_test() -> None:
    import io

    class TrackingBytesIO(io.BytesIO):
        def close(self) -> None:
            self.closed_by_audit = True
            super().close()

    def zline(record: Any, terminate: bool = True) -> bytes:
        raw = json.dumps(record, separators=(",", ":")).encode("utf-8")
        return raw + (b"\n" if terminate else b"")

    wrappers = [
        {"data": [
            {"id": "j1", "type": "job_opening", "attributes": {
                "description": None, "salary": None, "location": "NY", "status": None,
                "seniority": "junior", "language": "en", "categories": ["ops", "data"],
                "contract_types": ["full time"], "posted_at": "bad", "first_seen_at": "2026-02-02T00:00:00Z",
                "last_seen_at": "2026-02-01T00:00:00Z", "last_processed_at": "2026-02-03T00:00:00Z",
                "location_data": [{"country": "US", "nested": [{"city": "X"}]}],
            }, "relationships": {"company": {"data": {"type": "company", "id": "c1"}}}},
            {"id": "j2", "type": "job_opening", "attributes": {
                "description": "  ", "salary": "35 USD per hour", "status": "closed", "seniority": "manager",
            }},
        ], "included": [{"id": "c1", "type": "company", "attributes": {"domain": "d.test"}}]},
        {"data": [{"id": "j3", "type": "job_opening", "attributes": {"description": "text", "salary": "10"}}], "included": []},
    ]
    raw = b"".join(zline(wrapper) for wrapper in wrappers)
    raw += b"{bad-json}\n"
    raw += b"x" * 2500 + b"\n"
    raw += b'{"data":[]}'
    compressed = zstd.ZstdCompressor().compress(raw)
    profiler = FullAuditProfiler("jobs", 20261005, 2, field_path_cap=100, category_cap=20, sample_record_bytes=2048)
    with io.BytesIO() as sections:
        section_path = Path("_selftest_sections.jsonl")
        try:
            stream = TrackingBytesIO(compressed)
            run = _audit_stream(stream, len(compressed), profiler, max_seconds=None, max_line_bytes=2048,
                                section_size=2, progress_seconds=60, section_path=section_path,
                                checkpoint_callback=lambda *_: None)
            assert stream.closed_by_audit
            summary = profiler.summary()
            assert run["complete_file_scanned"] and run["stop_reason"] == "eof"
            assert profiler.primary_count == 3 and profiler.included_count == 1
            assert summary["malformed_records"]["invalid_json"] == 1
            assert summary["malformed_records"]["oversized_line"] == 1
            assert summary["malformed_records"]["unterminated_final_line"] == 1
            job_schema = summary["schema_by_collection_and_type"]["primary:job_opening"]
            assert "attributes.location_data[].nested[].city" in job_schema["fields"]
            assert profiler.sections_completed == 2
            assert len(profiler.reservoir) == 2
            assert summary["same_wrapper_company_relationship_matches"]["matched_same_wrapper"] == 1
            assert summary["same_wrapper_company_relationship_matches"].get("unresolved_same_wrapper", 0) == 0
            assert summary["bounded_categories"]["status"]["top_values"]["<null>"] == 1
            assert summary["bounded_categories"]["status"]["top_values"]["closed"] == 1
        finally:
            section_path.unlink(missing_ok=True)

    company_wrapper = {"data": [
        {"id": "c1", "type": "company", "attributes": {
            "company_name": "A", "domain": "a.test", "description": None, "location": "X",
        }},
        {"id": "c2", "type": "company", "attributes": {
            "company_name": None, "domain": "b.test", "description": "  ",
        }},
    ], "included": []}
    company_profiler = FullAuditProfiler("companies", 20261005, 2, field_path_cap=100,
                                        category_cap=20, sample_record_bytes=2048)
    section_path = Path("_selftest_company_sections.jsonl")
    try:
        company_payload = zline(company_wrapper)
        company_stream = TrackingBytesIO(zstd.ZstdCompressor().compress(company_payload))
        company_run = _audit_stream(company_stream, None, company_profiler, max_seconds=None,
                                    max_line_bytes=4096, section_size=1, progress_seconds=60,
                                    section_path=section_path, checkpoint_callback=lambda *_: None)
        company_sections = [json.loads(line) for line in section_path.read_text(encoding="utf-8").splitlines()]
        assert company_run["complete_file_scanned"]
        assert company_profiler.sections_completed == 2
        assert company_sections[0]["field_states"]["company_name"]["present"] == 1
        assert company_sections[1]["field_states"]["company_name"]["null"] == 1
        assert company_sections[1]["field_states"]["description"]["blank"] == 1
    finally:
        section_path.unlink(missing_ok=True)

    partial = FullAuditProfiler("jobs", 20261005, 3, field_path_cap=100, category_cap=20, sample_record_bytes=2048)
    section_path = Path("_selftest_sections.jsonl")
    try:
        partial_stream = TrackingBytesIO(zstd.ZstdCompressor().compress(zline(wrappers[0]) * 3))
        partial_run = _audit_stream(partial_stream, None, partial, max_seconds=0.001, max_line_bytes=4096,
                                    section_size=100, progress_seconds=60, section_path=section_path,
                                    checkpoint_callback=lambda *_: None, started_at=time.monotonic() - 1)
        assert not partial_run["complete_file_scanned"]
        assert partial_run["stop_reason"] == "benchmark_time_limit"
        assert partial_stream.closed_by_audit
    finally:
        section_path.unlink(missing_ok=True)

    class ReadFailure:
        def __init__(self):
            self.closed_by_audit = False

        def read(self, size: int = -1) -> bytes:
            raise OSError("synthetic stream read failed")

        def close(self) -> None:
            self.closed_by_audit = True

    failed_profiler = FullAuditProfiler("jobs", 20261005, 1, field_path_cap=100,
                                        category_cap=20, sample_record_bytes=2048)
    failed_stream = ReadFailure()
    section_path = Path("_selftest_error_sections.jsonl")
    try:
        failed_run = _audit_stream(failed_stream, 100, failed_profiler, max_seconds=None,
                                   max_line_bytes=4096, section_size=100, progress_seconds=60,
                                   section_path=section_path, checkpoint_callback=lambda *_: None)
        assert failed_run["stop_reason"] == "stream_error"
        assert failed_run["complete_file_scanned"] is False
        assert failed_run["error"]["type"] == "OSError"
        assert failed_stream.closed_by_audit
        assert _audit_sample_document(failed_profiler, failed_run)["seed"] == 20261005
        failed_result = {**failed_profiler.summary(), "run": failed_run}
        report_path = Path("_selftest_partial_report.md")
        try:
            _write_full_report(report_path, {"jobs": failed_result}, {"jobs": "OSError: synthetic stream read failed"})
            report_text = report_path.read_text(encoding="utf-8")
            assert "complete stream inspected: `False`" in report_text
            assert "Read: 0 compressed bytes" in report_text
            assert "OSError" in report_text
        finally:
            report_path.unlink(missing_ok=True)
    finally:
        section_path.unlink(missing_ok=True)
    print("Full-audit self-test passed: nested arrays, multiple resources, missing/null/blank, malformed/oversized input, sections, bounded reservoir, relation matching, and partial-run reporting.")


def _run_full_audit(args: argparse.Namespace) -> int:
    output_dir = args.output_dir / "full_audit"
    if args.mode == "benchmark":
        output_dir = output_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    section_path = output_dir / "section_metrics.jsonl"
    if section_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit sections: {section_path}")
    if args.mode == "benchmark" and (output_dir / "benchmark.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_dir / 'benchmark.json'}")
    if args.mode == "audit":
        expected = ("full_quality_summary.json", "full_quality_report.md", "retained_samples.json")
        existing = [name for name in expected if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing full-audit outputs: {', '.join(existing)}")
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    config = botocore.config.Config(
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    s3 = session.client("s3", config=config)
    datasets = ["jobs"] if args.mode == "benchmark" else list(FILES) if args.dataset == "both" else [args.dataset]
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    sample_documents: dict[str, Any] = {}
    run_started = datetime.now(timezone.utc).isoformat()
    checkpoint_path = output_dir / "full_quality_summary.checkpoint.json"

    def write_checkpoint(current_profiler: FullAuditProfiler, current_run: dict[str, Any]) -> None:
        checkpoint = {
            "mode": args.mode,
            "started_at_utc": run_started,
            "status": "in_progress",
            "current_dataset": current_profiler.dataset,
            "run": current_run,
            "current_counts": {
                "lines": current_profiler.line_count,
                "wrappers": current_profiler.wrapper_count,
                "primary": current_profiler.primary_count,
                "included": current_profiler.included_count,
            },
            "malformed": dict(current_profiler.malformed),
            "field_quality_counts_by_path": {
                group: {
                    path: {
                        "present_records": stats["present_records"],
                        "missing_parent_records": stats["missing_parent_records"],
                        "missing_child_records": stats["missing_child_records"],
                        "null_values": stats["null_values"],
                        "blank_values": stats["blank_values"],
                        "value_instances": stats["value_instances"],
                    }
                    for path, stats in profile["fields"].items()
                }
                for group, profile in current_profiler.schema.items()
            },
            "bounded_categories": {
                key: counter.snapshot() for key, counter in current_profiler.category_values.items()
            },
            "same_wrapper_company_relationships": dict(current_profiler.relationship_matches),
        }
        temp = checkpoint_path.with_suffix(".tmp")
        temp.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        temp.replace(checkpoint_path)

    try:
        for dataset in datasets:
            profiler = FullAuditProfiler(
                dataset, args.seed, 0 if args.mode == "benchmark" else args.reservoir_size,
                field_path_cap=args.field_path_cap, category_cap=args.category_cap,
                sample_record_bytes=args.sample_record_bytes,
            )
            started_at = time.monotonic()
            try:
                response = s3.get_object(Bucket=BUCKET, Key=PREFIX + FILES[dataset])
                object_size = response.get("ContentLength")
                metadata = {
                    "etag": response.get("ETag"),
                    "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
                    "version_id": response.get("VersionId"),
                    "content_type": response.get("ContentType"),
                }
                max_seconds = args.max_seconds if args.mode == "benchmark" else None
                run = _audit_stream(
                    response["Body"], object_size, profiler, max_seconds=max_seconds,
                    max_line_bytes=args.audit_max_line_bytes, section_size=args.section_size,
                    progress_seconds=args.progress_seconds, section_path=section_path,
                    checkpoint_callback=write_checkpoint, started_at=started_at,
                )
                run.update(metadata)
                if args.mode == "benchmark" and run["compressed_mib_per_second"]:
                    compressed_rate = run["compressed_mib_per_second"] * 1024 * 1024
                    run["linear_estimated_full_runtime_seconds"] = round(object_size / compressed_rate, 1) if object_size else None
                    run["linear_estimated_remaining_seconds"] = round(max(0, object_size - run["compressed_bytes_read"]) / compressed_rate, 1) if object_size else None
                    run["linear_estimate_assumptions"] = "Assumes measured prefix throughput continues unchanged; compression ratio, record complexity, source mix, network throughput, and later-stream variation may differ."
                summary = profiler.summary()
                result = {**summary, "run": run}
                results[dataset] = result
                if args.mode != "benchmark":
                    sample_documents[dataset] = _audit_sample_document(profiler, run)
                if run.get("error"):
                    failures[dataset] = f"{run['error']['type']}: {run['error']['message']}"
                    print(f"[{dataset}] STREAM ERROR: {run['error']['type']}: {run['error']['message']}", file=sys.stderr, flush=True)
                print(
                    f"[{dataset}] stopped={run['stop_reason']} complete_file={run['complete_file_scanned']} "
                    f"primary={run['primary_resources']} included={run['included_resources']} "
                    f"compressed_bytes={run['compressed_bytes_read']} elapsed={run['elapsed_seconds']:.1f}s",
                    flush=True,
                )
                if run["stop_reason"] == "interrupted":
                    failures[dataset] = "interrupted by user; current stream closed; partial counts retained"
                    break
            except KeyboardInterrupt:
                failures[dataset] = "interrupted by user; stream closed; checkpoint preserves last periodic counts"
                results[dataset] = {**profiler.summary(), "run": {
                    "stop_reason": "interrupted_before_or_during_request", "complete_file_scanned": False,
                    "compressed_bytes_read": None, "object_size_compressed_bytes": None,
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    "primary_resources": profiler.primary_count, "included_resources": profiler.included_count,
                    "lines_or_wrappers": profiler.line_count,
                    "uninspected_records": "unknown remainder; object scan did not reach EOF",
                    "source_metadata": None,
                }}
                if args.mode != "benchmark":
                    sample_documents[dataset] = _audit_sample_document(profiler, results[dataset]["run"])
                break
            except Exception as error:
                safe_message = re.sub(r"(?i)(secret|token|credential|signature|authorization)(\s*[=:]\s*)\S+", r"\1\2<redacted>", str(error))[:500]
                failures[dataset] = f"{type(error).__name__}: {safe_message}"
                print(f"[{dataset}] FAILED: {type(error).__name__}: {safe_message}; partial metrics retained where available.", file=sys.stderr, flush=True)
                results[dataset] = {**profiler.summary(), "run": {
                    "stop_reason": "request_or_stream_error", "complete_file_scanned": False,
                    "error": {"type": type(error).__name__, "message": safe_message},
                    "compressed_bytes_read": None, "object_size_compressed_bytes": None,
                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    "primary_resources": profiler.primary_count, "included_resources": profiler.included_count,
                    "lines_or_wrappers": profiler.line_count,
                    "uninspected_records": "unknown remainder; object scan did not reach EOF",
                    "source_metadata": None,
                }}
                if args.mode != "benchmark":
                    sample_documents[dataset] = _audit_sample_document(profiler, results[dataset]["run"])
    finally:
        try:
            s3.close()
        except Exception as error:
            safe_message = re.sub(r"(?i)(secret|token|credential|signature|authorization)(\s*[=:]\s*)\S+", r"\1\2<redacted>", str(error))[:500]
            failures["s3_client_close"] = f"{type(error).__name__}: {safe_message}"

    if args.mode == "benchmark":
        benchmark = {
            "mode": "five_minute_benchmark",
            "assumptions": "Measured only the first configured max_seconds of the jobs object; runtime extrapolation assumes remaining bytes and decompression throughput match the measured prefix, which may be inaccurate.",
            "results": results,
            "failures": failures,
        }
        (output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Benchmark summary: {output_dir / 'benchmark.json'}", flush=True)
        return 1 if failures else 0

    summary_document = {
        "mode": "full_streaming_audit",
        "started_at_utc": run_started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_seed": args.seed,
        "limitations": [
            "No exact dataset-wide duplicate detection; use a disk-backed strategy as follow-up.",
            "No separate companies-file join audit; only same-wrapper type/ID references are counted.",
            "Field paths are capped per collection/type and frequency dimensions are capped; caps and overflow are reported.",
            "Reservoir includes successfully parsed primary resource dictionaries only; malformed wrappers and skipped resources are excluded.",
            "Reservoirs on incomplete scans cover only the scanned portion.",
        ],
        "results": results,
        "failures": failures,
    }
    summary_path = output_dir / "full_quality_summary.json"
    summary_path.write_text(json.dumps(summary_document, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (output_dir / "retained_samples.json").write_text(json.dumps(sample_documents, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_full_report(output_dir / "full_quality_report.md", results, failures)
    checkpoint_path.unlink(missing_ok=True)
    print(f"Saved full-audit artifacts under {output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())