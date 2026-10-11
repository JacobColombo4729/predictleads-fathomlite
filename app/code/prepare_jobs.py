"""Prepare a bounded PredictLeads jobs development slice, preserving source values."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO

import boto3
import botocore
import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parents[2]  # data/, models/ and .env are shared with analysis/ at the top

BUCKET = "tripleten-predictleads-hackathon-109698388873-us-east-2-an"
KEY = "2026/09/30/job_openings_2026_09_30.jsonl.zst"
PROFILE = "predictleads"
REGION = "us-east-2"
SNAPSHOT_DATE = "2026-09-30"
DEFAULT_LIMIT = 10_000
MAX_LINE_BYTES = 4 * 1024 * 1024
PROGRESS_EVERY = 1_000
DATE_FIELDS = ("posted_at", "first_seen_at", "last_seen_at", "last_processed_at")
TITLE_FIELDS = ("title", "normalized_title", "translated_title")


class DescriptionTextParser(HTMLParser):
    """Convert HTML to readable text while preserving block/list boundaries."""

    BLOCK_TAGS = {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt",
        "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "ol", "p",
        "pre", "section", "table", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.hidden_depth += 1
        if self.hidden_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self.hidden_depth:
            self.hidden_depth -= 1
        if self.hidden_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


def clean_description(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parser = DescriptionTextParser()
    parser.feed(value)
    parser.close()
    text = parser.text()
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "- ", text)
    text = re.sub(r"(?m)^\s*(\d+)[.)]\s+", r"\1. ", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return "\n".join(line for line in text.splitlines() if line.strip()).strip()


def field_state(attributes: dict[str, Any], field: str) -> str:
    if field not in attributes:
        return "missing"
    value = attributes[field]
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank"
    return "present"


def parse_utc(value: Any) -> tuple[str | None, str | None, bool]:
    if value is None or isinstance(value, str) and not value.strip():
        return None, None, False
    if not isinstance(value, str):
        return None, "invalid_type", False
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None, "invalid", False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    suspicious_year = parsed.year < 1900 or parsed.year > 2027
    return parsed.isoformat().replace("+00:00", "Z"), None, suspicious_year


def match_included_company(resource: dict[str, Any], included: Any) -> dict[str, Any]:
    relationships = resource.get("relationships")
    company_rel = relationships.get("company") if isinstance(relationships, dict) else None
    ref_data = company_rel.get("data") if isinstance(company_rel, dict) else None
    if not isinstance(ref_data, dict):
        return {
            "reference_type": None,
            "reference_id": None,
            "match_state": "missing_reference",
            "company_name": None,
            "domain": None,
            "match_count": 0,
        }
    ref_type, ref_id = ref_data.get("type"), ref_data.get("id")
    if ref_type is None or ref_id is None:
        return {
            "reference_type": ref_type,
            "reference_id": ref_id,
            "match_state": "incomplete_reference",
            "company_name": None,
            "domain": None,
            "match_count": 0,
        }
    candidates = included if isinstance(included, list) else []
    matches = [
        item for item in candidates
        if isinstance(item, dict)
        and item.get("type") == ref_type
        and item.get("id") == ref_id
    ]
    if len(matches) != 1:
        return {
            "reference_type": ref_type,
            "reference_id": ref_id,
            "match_state": "missing_match" if not matches else "ambiguous_match",
            "company_name": None,
            "domain": None,
            "match_count": len(matches),
        }
    company_attributes = matches[0].get("attributes")
    company_attributes = company_attributes if isinstance(company_attributes, dict) else {}
    return {
        "reference_type": ref_type,
        "reference_id": ref_id,
        "match_state": "matched",
        "company_name": company_attributes.get("company_name"),
        "domain": company_attributes.get("domain"),
        "match_count": 1,
    }


def prepare_resource(
    resource: dict[str, Any],
    included: Any,
    *,
    source_line: int,
    source_key: str = KEY,
) -> dict[str, Any]:
    attributes = resource.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}

    normalized_title = attributes.get("normalized_title")
    translated_title = attributes.get("translated_title")
    raw_title = attributes.get("title")
    if isinstance(normalized_title, str) and normalized_title.strip():
        searchable_title = normalized_title
        searchable_title_source = "normalized_title"
    elif isinstance(translated_title, str) and translated_title.strip():
        searchable_title = translated_title
        searchable_title_source = "translated_title"
    elif isinstance(raw_title, str) and raw_title.strip():
        searchable_title = raw_title
        searchable_title_source = "title"
    else:
        searchable_title = None
        searchable_title_source = None

    raw_description = attributes.get("description")
    description_text = clean_description(raw_description)
    raw_status = attributes.get("status")
    derived_status = "closed" if isinstance(raw_status, str) and raw_status.strip().casefold() == "closed" else "unknown"

    parsed_dates: dict[str, str | None] = {}
    date_flags: list[str] = []
    parsed_datetimes: dict[str, datetime] = {}
    for field in DATE_FIELDS:
        parsed, error, suspicious_year = parse_utc(attributes.get(field))
        parsed_dates[field] = parsed
        if error:
            date_flags.append(f"{field}_{error}")
        if suspicious_year:
            date_flags.append(f"{field}_suspicious_year")
        if parsed:
            parsed_datetimes[field] = datetime.fromisoformat(parsed.replace("Z", "+00:00"))
    for first, second in (
        ("posted_at", "first_seen_at"),
        ("first_seen_at", "last_seen_at"),
        ("last_seen_at", "last_processed_at"),
    ):
        if first in parsed_datetimes and second in parsed_datetimes and parsed_datetimes[first] > parsed_datetimes[second]:
            date_flags.append(f"{first}_after_{second}")

    flags: dict[str, str] = {}
    for field in (*TITLE_FIELDS, "description", "url", "language", "seniority", "categories", "tags",
                  "onet_data", "contract_types", "location", "location_data", "salary", "salary_data",
                  "status", *DATE_FIELDS):
        flags[field] = field_state(attributes, field)
    flags["description_usable"] = "yes" if description_text else "no"
    company = match_included_company(resource, included)
    flags["company_relationship"] = company["match_state"]
    flags["date_anomalies"] = ";".join(date_flags) if date_flags else "none"

    onet_data = attributes.get("onet_data")
    return {
        "job_id": resource.get("id"),
        "job_type": resource.get("type"),
        "source_key": source_key,
        "snapshot_date": SNAPSHOT_DATE,
        "source_line": source_line,
        "title_raw": raw_title,
        "title_normalized": normalized_title,
        "title_translated": translated_title,
        "title_searchable": searchable_title,
        "title_searchable_source": searchable_title_source,
        "description_raw": raw_description,
        "description_text": description_text,
        "description_usable": bool(description_text),
        "url": attributes.get("url"),
        "language": attributes.get("language"),
        "seniority": attributes.get("seniority"),
        "categories": attributes.get("categories"),
        "tags": attributes.get("tags"),
        "onet_data": onet_data,
        "contract_types": attributes.get("contract_types"),
        "location_raw": attributes.get("location"),
        "location_data_raw": attributes.get("location_data"),
        "salary_raw": attributes.get("salary"),
        "salary_data_raw": attributes.get("salary_data"),
        "status_raw": raw_status,
        "status_derived": derived_status,
        "historical_snapshot": True,
        "current_opening_verified": False,
        "dates_raw": {field: attributes.get(field) for field in DATE_FIELDS},
        "dates_utc": parsed_dates,
        "company_reference": {
            "type": company["reference_type"],
            "id": company["reference_id"],
        },
        "company_match_state": company["match_state"],
        "company_match_count": company["match_count"],
        "company_name_included": company["company_name"],
        "company_domain_included": company["domain"],
        "quality_flags": flags,
    }


def _add_quality_counts(summary: dict[str, Any], prepared: dict[str, Any]) -> None:
    for field, state in prepared["quality_flags"].items():
        summary["quality_flag_counts"][field][state] += 1
    summary["company_match_counts"][prepared["company_match_state"]] += 1


def _process_wrapper(
    wrapper: Any,
    *,
    source_line: int,
    remaining_primary: int,
    output: Any,
    summary: dict[str, Any],
    source_key: str,
) -> tuple[int, bool]:
    if not isinstance(wrapper, dict):
        summary["malformed_records"]["wrapper_not_object"] += 1
        return 0, False
    data = wrapper.get("data", "<missing>")
    if not isinstance(data, list):
        summary["malformed_records"]["data_not_array"] += 1
        return 0, False
    included = wrapper.get("included", [])
    if included is None:
        included = []
    elif not isinstance(included, list):
        summary["malformed_records"]["included_not_array"] += 1
        included = []
    summary["included_resources_inspected"] += sum(isinstance(item, dict) for item in included)

    written_here = 0
    for resource in data:
        if written_here >= remaining_primary:
            break
        if not isinstance(resource, dict):
            summary["skipped_resources"]["primary_not_object"] += 1
            continue
        prepared = prepare_resource(resource, included, source_line=source_line, source_key=source_key)
        if prepared["job_id"] is not None:
            identity = (str(prepared["job_type"]), str(prepared["job_id"]))
            summary["primary_identity_counts"][identity] += 1
            if summary["primary_identity_counts"][identity] > 1:
                summary["duplicate_identity_counts"][f"{identity[0]}:{identity[1]}"] += 1
        _add_quality_counts(summary, prepared)
        output.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")
        written_here += 1
        summary["primary_resources_written"] += 1
    reached_limit = summary["primary_resources_written"] >= summary["primary_resource_limit"]
    return written_here, reached_limit


def _new_summary(output_file: Path, source_label: str, limit: int) -> dict[str, Any]:
    return {
        "source": source_label,
        "snapshot_date": SNAPSHOT_DATE,
        "primary_resource_limit": limit,
        "wrappers_or_lines_inspected": 0,
        "primary_resources_processed": 0,
        "primary_resources_written": 0,
        "included_resources_inspected": 0,
        "stop_reason": "not_started",
        "complete_file_scanned": False,
        "uninspected_records": "not started",
        "output_file": str(output_file),
        "quality_flag_counts": CounterMap(),
        "malformed_records": Counter(),
        "skipped_resources": Counter(),
        "company_match_counts": Counter(),
        "primary_identity_counts": Counter(),
        "duplicate_identity_counts": Counter(),
        "failures": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": 0.0,
    }


class CounterMap(dict):
    def __missing__(self, key: str) -> Counter:
        value: Counter = Counter()
        self[key] = value
        return value


def _summary_json(summary: dict[str, Any]) -> dict[str, Any]:
    clean = dict(summary)
    for name in ("quality_flag_counts", "malformed_records", "skipped_resources", "company_match_counts"):
        value = clean[name]
        clean[name] = {key: dict(counts) for key, counts in value.items()} if name == "quality_flag_counts" else dict(value)
    clean["duplicate_identity_counts"] = dict(clean["duplicate_identity_counts"])
    # The identity counter is maintained for this small slice only; report totals, not its internal lookup.
    clean.pop("primary_identity_counts", None)
    return clean


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(_summary_json(summary), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)


def _discover_local_source(workspace: Path) -> Path | None:
    candidates = (
        workspace / Path(KEY),
        workspace / "data" / Path(KEY).name,
        workspace / "data" / "raw" / Path(KEY).name,
    )
    return next((path for path in candidates if path.is_file()), None)


def _run_self_tests() -> None:
    import io

    assert field_state({}, "description") == "missing"
    assert field_state({"description": None}, "description") == "null"
    assert field_state({"description": "  "}, "description") == "blank"
    cleaned = clean_description("<h2>Must have C++</h2><ul><li>Do not remove .NET</li><li>C#</li></ul>")
    assert cleaned is not None
    assert "Must have C++" in cleaned and "Do not remove .NET" in cleaned and "C#" in cleaned
    assert "\n" in cleaned
    markdown_cleaned = clean_description("### Requirements\n\n- **Must have C++**\n- Do not remove .NET\n1. Learn C#")
    assert markdown_cleaned is not None
    assert "###" not in markdown_cleaned and "**" not in markdown_cleaned
    assert "Requirements" in markdown_cleaned and "- Must have C++" in markdown_cleaned
    assert "Do not remove .NET" in markdown_cleaned and "1. Learn C#" in markdown_cleaned
    assert parse_utc("not-a-date")[1] == "invalid"
    assert parse_utc(None) == (None, None, False)
    assert parse_utc("   ") == (None, None, False)
    assert parse_utc(123)[1] == "invalid_type"
    assert parse_utc("0026-04-03T23:01:56Z")[2] is True

    company = {"id": "c1", "type": "company", "attributes": {"company_name": "Acme", "domain": "acme.test"}}
    job = {"id": "j1", "type": "job_opening", "attributes": {"title": "Raw", "normalized_title": "Norm",
           "description": None, "salary": None, "location": None, "status": None,
           "posted_at": "invalid", "first_seen_at": "2026-01-02T00:00:00Z",
           "last_seen_at": "2026-01-01T00:00:00Z"},
           "relationships": {"company": {"data": {"type": "company", "id": "c1"}}}}
    prepared = prepare_resource(job, [company], source_line=7)
    assert prepared["title_raw"] == "Raw" and prepared["title_normalized"] == "Norm"
    assert prepared["title_searchable"] == "Norm" and prepared["title_searchable_source"] == "normalized_title"
    assert prepared["status_derived"] == "unknown" and prepared["historical_snapshot"] is True
    assert prepared["company_match_state"] == "matched" and prepared["company_name_included"] == "Acme"
    assert "posted_at_invalid" in prepared["quality_flags"]["date_anomalies"]
    assert "first_seen_at_after_last_seen_at" in prepared["quality_flags"]["date_anomalies"]
    translated_fallback = prepare_resource(
        {"id": "j2", "type": "job_opening", "attributes": {
            "title": "Raw title", "normalized_title": "  ", "translated_title": "Translated title",
        }}, [], source_line=9,
    )
    assert translated_fallback["title_searchable"] == "Translated title"
    assert translated_fallback["title_searchable_source"] == "translated_title"
    raw_fallback = prepare_resource(
        {"id": "j3", "type": "job_opening", "attributes": {
            "title": "Raw title", "normalized_title": None, "translated_title": "",
        }}, [], source_line=10,
    )
    assert raw_fallback["title_searchable"] == "Raw title"
    assert raw_fallback["title_searchable_source"] == "title"
    assert raw_fallback["title_normalized"] is None and raw_fallback["title_translated"] == ""
    assert prepared["salary_raw"] is None and prepared["location_data_raw"] is None
    null_date = prepare_resource(
        {"id": "j-date", "type": "job_opening", "attributes": {"posted_at": None}},
        [], source_line=11,
    )
    assert "posted_at_null" not in null_date["quality_flags"]["date_anomalies"]
    assert null_date["quality_flags"]["posted_at"] == "null"
    assert match_included_company(job, [company, company])["match_state"] == "ambiguous_match"
    assert match_included_company(job, [])["match_state"] == "missing_match"
    unusual_onet = {"id": "j-onet", "type": "job_opening", "attributes": {"onet_data": ["source-value"]}}
    assert prepare_resource(unusual_onet, [], source_line=8)["onet_data"] == ["source-value"]

    wrapper = {"data": [job, {**job, "id": "j2"}], "included": [company]}
    class DiscardWriter(io.StringIO):
        pass
    summary = _new_summary(Path("synthetic.jsonl"), KEY, 1)
    written, reached = _process_wrapper(wrapper, source_line=1, remaining_primary=1,
                                        output=DiscardWriter(), summary=summary, source_key=KEY)
    assert written == 1 and reached and summary["primary_resources_written"] == 1
    assert summary["company_match_counts"]["matched"] == 1

    summary = _new_summary(Path("synthetic.jsonl"), KEY, 2)
    output = DiscardWriter()
    _process_wrapper({"data": [job], "included": [company]}, source_line=1,
                     remaining_primary=2, output=output, summary=summary, source_key=KEY)
    _process_wrapper({"data": [job], "included": [company]}, source_line=2,
                     remaining_primary=1, output=output, summary=summary, source_key=KEY)
    assert summary["duplicate_identity_counts"]["job_opening:j1"] == 1
    print("Synthetic preparation checks passed: field states, HTML/token preservation, invalid/suspicious dates, company matching, duplicates, and exact resource limit.")


def prepare_slice(workspace: Path, output_dir: Path, limit: int) -> dict[str, Any]:
    local_source = _discover_local_source(workspace)
    source_path: str | None = None
    body: BinaryIO | None = None
    s3_client = None
    object_metadata: dict[str, Any] = {}
    source_label = f"s3://{BUCKET}/{KEY}"
    if local_source is not None:
        source_path = str(local_source)
        source_label = str(local_source)
        body = local_source.open("rb")
        object_metadata["local_size_bytes"] = local_source.stat().st_size
    else:
        session = boto3.Session(profile_name=PROFILE, region_name=REGION)
        s3_client = session.client("s3", config=botocore.config.Config(
            connect_timeout=10, read_timeout=30, retries={"max_attempts": 3, "mode": "standard"},
        ))
        response = s3_client.get_object(Bucket=BUCKET, Key=KEY)
        body = response["Body"]
        object_metadata = {
            "compressed_size_bytes": response.get("ContentLength"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
        }

    assert body is not None
    output_file = output_dir / "jobs_prepared_sample.jsonl"
    summary_file = output_dir / "preparation_summary.json"
    summary = _new_summary(output_file, source_label, limit)
    summary["source_object_metadata"] = object_metadata
    started = time.monotonic()
    pending = bytearray()
    discarding_oversized = False
    stop_reason = "eof"
    complete_eof = False
    last_progress = started
    written = 0
    try:
        with output_file.open("x", encoding="utf-8", newline="\n") as output:
            with zstd.ZstdDecompressor().stream_reader(body, read_across_frames=True, closefd=False) as reader:
                while True:
                    if written >= limit:
                        stop_reason = "primary_resource_limit"
                        break
                    chunk = reader.read(64 * 1024)
                    if not chunk:
                        complete_eof = True
                        stop_reason = "eof"
                        if pending:
                            summary["malformed_records"]["unterminated_final_line"] += 1
                        break
                    remaining = chunk
                    while remaining:
                        if discarding_oversized:
                            newline = remaining.find(b"\n")
                            if newline < 0:
                                remaining = b""
                                continue
                            summary["malformed_records"]["oversized_line"] += 1
                            summary["wrappers_or_lines_inspected"] += 1
                            discarding_oversized = False
                            remaining = remaining[newline + 1:]
                            continue
                        newline = remaining.find(b"\n")
                        if newline < 0:
                            segment = remaining
                            remaining = b""
                            if len(pending) + len(segment) > MAX_LINE_BYTES:
                                pending.clear()
                                discarding_oversized = True
                            else:
                                pending.extend(segment)
                            continue
                        segment = remaining[:newline]
                        remaining = remaining[newline + 1:]
                        summary["wrappers_or_lines_inspected"] += 1
                        if len(pending) + len(segment) > MAX_LINE_BYTES:
                            pending.clear()
                            summary["malformed_records"]["oversized_line"] += 1
                            continue
                        line = bytes(pending) + segment
                        pending.clear()
                        try:
                            wrapper = json.loads(line.decode("utf-8"))
                        except UnicodeDecodeError:
                            summary["malformed_records"]["invalid_utf8"] += 1
                            continue
                        except json.JSONDecodeError:
                            summary["malformed_records"]["invalid_json"] += 1
                            continue
                        if not isinstance(wrapper, dict):
                            summary["malformed_records"]["wrapper_not_object"] += 1
                            continue
                        remaining_limit = limit - written
                        count, reached = _process_wrapper(
                            wrapper, source_line=summary["wrappers_or_lines_inspected"],
                            remaining_primary=remaining_limit, output=output, summary=summary,
                            source_key=KEY,
                        )
                        written += count
                        output.flush()
                        summary["primary_resources_processed"] = written
                        now = time.monotonic()
                        if written and written % PROGRESS_EVERY < count:
                            print(f"Progress: wrappers={summary['wrappers_or_lines_inspected']} primary_written={written} included_inspected={summary['included_resources_inspected']} elapsed_seconds={now-started:.1f}", flush=True)
                        if reached or written >= limit:
                            stop_reason = "primary_resource_limit"
                            break
                    if stop_reason == "primary_resource_limit":
                        break
                    now = time.monotonic()
                    if now - last_progress >= 30:
                        print(f"Progress: wrappers={summary['wrappers_or_lines_inspected']} primary_written={written} included_inspected={summary['included_resources_inspected']} elapsed_seconds={now-started:.1f}", flush=True)
                        last_progress = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        summary["failures"].append({"type": "KeyboardInterrupt", "message": "interrupted by user; partial outputs retained"})
    except Exception as error:
        stop_reason = "error"
        summary["failures"].append({"type": type(error).__name__, "message": str(error)[:500]})
        print(f"Preparation failed: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr, flush=True)
    finally:
        if body is not None:
            try:
                body.close()
            except Exception as error:
                summary["failures"].append({"type": f"body_close:{type(error).__name__}", "message": str(error)[:300]})
        if s3_client is not None:
            try:
                s3_client.close()
            except Exception as error:
                summary["failures"].append({"type": f"client_close:{type(error).__name__}", "message": str(error)[:300]})
        summary["stop_reason"] = stop_reason
        summary["complete_file_scanned"] = complete_eof and stop_reason == "eof"
        summary["uninspected_records"] = 0 if summary["complete_file_scanned"] else "unknown remainder; slice stopped before EOF"
        summary["primary_resources_written"] = written
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["duplicate_identity_counts"] = dict(summary["duplicate_identity_counts"])
        summary["duplicate_identity_pairs_seen"] = sum(count > 1 for count in summary["primary_identity_counts"].values())
        summary.pop("primary_identity_counts", None)
        _write_summary(summary_file, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "prepared")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    _run_self_tests()
    if args.self_test:
        return 0
    if args.limit < 1:
        parser.error("--limit must be positive")
    workspace = REPO_ROOT
    run_name = args.run_name or datetime.now(timezone.utc).strftime("jobs_sample_%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        summary = prepare_slice(workspace, output_dir, args.limit)
    except KeyboardInterrupt:
        output_file = output_dir / "jobs_prepared_sample.jsonl"
        output_file.touch(exist_ok=True)
        summary = _new_summary(output_file, f"s3://{BUCKET}/{KEY}", args.limit)
        summary["stop_reason"] = "interrupted_during_source_setup"
        summary["uninspected_records"] = "unknown; source setup was interrupted"
        summary["failures"].append({"type": "KeyboardInterrupt", "message": "interrupted before stream setup completed"})
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_summary(output_dir / "preparation_summary.json", summary)
    except Exception as error:
        output_file = output_dir / "jobs_prepared_sample.jsonl"
        output_file.touch(exist_ok=True)
        summary = _new_summary(output_file, f"s3://{BUCKET}/{KEY}", args.limit)
        summary["stop_reason"] = "source_setup_error"
        summary["uninspected_records"] = "unknown; source setup failed"
        summary["failures"].append({"type": type(error).__name__, "message": str(error)[:500]})
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_summary(output_dir / "preparation_summary.json", summary)
        print(f"Preparation failed during source setup: {type(error).__name__}: {str(error)[:300]}", file=sys.stderr, flush=True)
        return 1
    print(f"Preparation finished: stop={summary['stop_reason']} written={summary['primary_resources_written']} complete_file={summary['complete_file_scanned']}", flush=True)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())