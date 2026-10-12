"""Stream the full jobs snapshot once and keep AI/ML, data and technical-sales listings.

Matching is by title (any of the three title fields) or a short list of O*NET
occupations. Matches are prepared with the unchanged prepare_jobs.prepare_resource,
so raw values, flags and provenance are identical to the baseline sample. Nothing
is deduplicated or dropped here; reposts are flagged downstream.

Offline check of the matcher against the saved random sample (no S3):
    python code/extract_targeted_pool.py --check-sample
Full scan (reads ~19 GB from S3 with S3_PATH and AWS_PROFILE from the top-level .env, or the copy in data/raw/ if downloaded):
    python code/extract_targeted_pool.py --scan
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import botocore
import zstandard as zstd

from prepare_jobs import BUCKET, DATA_DIR, KEY, SNAPSHOT_DATE, prepare_resource, raw_file, s3_session

MAX_LINE_BYTES = 4 * 1024 * 1024
PROGRESS_SECONDS = 60
FILTER_VERSION = "targeted_pool_filter_v1"
SAMPLE_PATH = DATA_DIR / "discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl"

TITLE_PATTERNS = {
    "ai_ml_title": re.compile(
        r"machine[- ]learning|\bml\b|\bai\b|\bgen ?ai\b|artificial intelligence|data scien|deep learning|"
        r"\bnlp\b|natural language|computer vision|\bllm\b|generative|prompt engineer|mlops|ai trainer|data annotat",
        re.IGNORECASE),
    "technical_sales_title": re.compile(
        r"pre-?sales|solutions? (?:engineer|consultant|architect|specialist)|sales engineer|technical sales|"
        r"technical account|customer success engineer|(?:implementation|onboarding) (?:specialist|consultant|engineer)|"
        r"applications? engineer|field application|technical (?:solutions|specialist|consultant|product)|"
        r"sales (?:and|&) solutions|forward[- ]deployed|solutions? advisor",
        re.IGNORECASE),
    "data_title": re.compile(
        r"data analyst|data engineer|analytics (?:engineer|analyst|specialist|consultant)|business intelligence|"
        r"\bbi (?:analyst|developer|engineer)|research (?:scientist|engineer)|quantitative analyst|statistician|"
        r"python developer|decision scien",
        re.IGNORECASE),
}
ONET_OCCUPATIONS = {
    "Data Scientists",
    "Sales Engineers",
    "Computer and Information Research Scientists",
    "Business Intelligence Analysts",
    "Statisticians",
    "Computer Occupations, All Other",
}


def match_reasons(titles: list[Any], occupation: Any) -> list[str]:
    text = " | ".join(t for t in titles if isinstance(t, str))
    reasons = [name for name, pattern in TITLE_PATTERNS.items() if text and pattern.search(text)]
    if isinstance(occupation, str) and occupation in ONET_OCCUPATIONS:
        reasons.append("onet:" + occupation)
    return reasons


def resource_reasons(resource: dict[str, Any]) -> list[str]:
    attributes = resource.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    onet = attributes.get("onet_data")
    occupation = onet.get("occupation_name") if isinstance(onet, dict) else None
    return match_reasons([attributes.get("title"), attributes.get("normalized_title"),
                          attributes.get("translated_title")], occupation)


def _keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (not a persistent setting)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
    except Exception:
        pass


def check_sample() -> int:
    counts: Counter[str] = Counter()
    total = hits = usable_hits = 0
    with SAMPLE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            onet = row.get("onet_data")
            reasons = match_reasons([row.get("title_raw"), row.get("title_normalized"), row.get("title_translated")],
                                    onet.get("occupation_name") if isinstance(onet, dict) else None)
            if reasons:
                hits += 1
                usable_hits += bool(row.get("description_usable"))
                counts.update(reasons)
    print(json.dumps({"sample_rows": total, "matched": hits, "matched_with_usable_description": usable_hits,
                      "implied_full_file_matches_approx": round(hits * 13415388 / total),
                      "reason_counts": counts.most_common()}, indent=1))
    return 0


def scan(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=False)
    pool_path = output_dir / "targeted_pool.jsonl"
    summary_path = output_dir / "scan_summary.json"
    (output_dir / "filter_definition.json").write_text(json.dumps({
        "filter_version": FILTER_VERSION,
        "title_patterns": {k: v.pattern for k, v in TITLE_PATTERNS.items()},
        "onet_occupations": sorted(ONET_OCCUPATIONS),
        "note": "A listing matches if any title field matches a pattern or its O*NET occupation is listed. Not deduplicated.",
    }, indent=1), encoding="utf-8")
    _keep_awake()
    started = time.monotonic()
    lines = population = matched = 0
    errors: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    pending = bytearray()
    dropping = False
    stop_reason = "setup"
    source_meta: dict[str, Any] = {}
    client = None
    body = None
    last_progress = started
    out = pool_path.open("x", encoding="utf-8", newline="\n")

    def process(line: bytes, number: int) -> None:
        nonlocal population, matched
        try:
            wrapper = json.loads(line.decode("utf-8"))
        except UnicodeDecodeError:
            errors["invalid_utf8"] += 1
            return
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            return
        data = wrapper.get("data") if isinstance(wrapper, dict) else None
        if not isinstance(data, list):
            errors["data_not_array"] += 1
            return
        included = wrapper.get("included") if isinstance(wrapper.get("included"), list) else []
        for resource_index, resource in enumerate(data, 1):
            if not isinstance(resource, dict) or resource.get("type") != "job_opening":
                errors["skipped_non_job_resource"] += 1
                continue
            population += 1
            reasons = resource_reasons(resource)
            if not reasons:
                continue
            prepared = prepare_resource(resource, included, source_line=number, source_key=KEY)
            prepared["resource_index"] = resource_index
            prepared["population_index"] = population
            prepared["targeted_pool_reasons"] = reasons
            out.write(json.dumps(prepared, ensure_ascii=False, separators=(",", ":")) + "\n")
            matched += 1
            reason_counts.update(reasons)

    local = raw_file(Path(KEY).name)  # a downloaded copy (data_sources.get_predictleads) is read instead of S3
    source = str(local) if local else f"s3://{BUCKET}/{KEY}"
    try:
        if local:
            print(f"reading the local copy: {local.name}", flush=True)
            body = local.open("rb")
            source_meta = {"local_size_bytes": local.stat().st_size}
        else:
            client = s3_session().client(
                "s3", config=botocore.config.Config(connect_timeout=10, read_timeout=60,
                                                      retries={"max_attempts": 5, "mode": "standard"}))
            response = client.get_object(Bucket=BUCKET, Key=KEY)
            body = response["Body"]
            source_meta = {"compressed_size_bytes": response.get("ContentLength"), "etag": response.get("ETag"),
                           "last_modified": response["LastModified"].isoformat() if response.get("LastModified") else None}
        stop_reason = "in_progress"
        with zstd.ZstdDecompressor().stream_reader(body, read_across_frames=True, closefd=False) as reader:
            while True:
                chunk = reader.read(64 * 1024)
                if not chunk:
                    if dropping or pending:
                        lines += 1
                        errors["oversized_or_unterminated_final_line"] += 1
                    stop_reason = "eof"
                    break
                remaining = chunk
                while remaining:
                    newline = remaining.find(b"\n")
                    if dropping:
                        if newline < 0:
                            remaining = b""
                            continue
                        lines += 1
                        errors["oversized_line"] += 1
                        dropping = False
                        remaining = remaining[newline + 1:]
                        continue
                    if newline < 0:
                        if len(pending) + len(remaining) > MAX_LINE_BYTES:
                            pending.clear()
                            dropping = True
                        else:
                            pending.extend(remaining)
                        remaining = b""
                        continue
                    segment, remaining = remaining[:newline], remaining[newline + 1:]
                    lines += 1
                    if len(pending) + len(segment) > MAX_LINE_BYTES:
                        pending.clear()
                        errors["oversized_line"] += 1
                        continue
                    line = bytes(pending) + segment
                    pending.clear()
                    process(line, lines)
                now = time.monotonic()
                if now - last_progress >= PROGRESS_SECONDS:
                    out.flush()
                    print(f"progress lines={lines} jobs={population} matched={matched} "
                          f"elapsed_s={now - started:.0f}", flush=True)
                    last_progress = now
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:  # keep the partial pool and say so
        stop_reason = "error"
        errors[f"{type(error).__name__}:{str(error)[:200]}"] += 1
        print(f"scan error: {type(error).__name__}: {str(error)[:250]}", file=sys.stderr, flush=True)
    finally:
        out.close()
        for closer in (body, client):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
    summary = {
        "filter_version": FILTER_VERSION, "source": source, "source_metadata": source_meta,
        "snapshot_date": SNAPSHOT_DATE, "stop_reason": stop_reason,
        "complete_file_scanned": stop_reason == "eof", "lines_read": lines,
        "job_opening_records_seen": population, "matched_records": matched,
        "reason_counts": dict(reason_counts.most_common()), "errors": dict(errors),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Matches are records, not unique jobs; reposts are not removed.",
                        "Title/O*NET matching can miss roles with unusual titles and include false hits.",
                        "Historical snapshot; availability is not verified."],
    }
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if stop_reason == "eof" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-sample", action="store_true")
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args(argv)
    if args.check_sample:
        return check_sample()
    if args.scan:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return scan(DATA_DIR / "matching" / f"targeted_pool_{stamp}")
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
