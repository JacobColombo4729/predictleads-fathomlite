"""Persistence helpers for the local retail recommendation review app."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROFILE_ID = "synthetic_retail_inventory"
VALID_RATINGS = {"0", "1", "2"}
REVIEW_FIELDS = (
    "profile_id", "job_id", "relevance_score", "review_status",
    "insufficient_evidence_reason", "reviewer", "notes",
)


def load_retail_jobs(blinded_path: Path) -> list[dict[str, Any]]:
    """Load the requested profile only and verify stable identifiers."""
    path = Path(blinded_path)
    if not path.is_file():
        raise FileNotFoundError(f"Blinded review file not found: {path}")
    required = {"profile_id", "job_id", "candidate_evidence", "description_full_prepared"}
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Blinded review CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("profile_id") != PROFILE_ID:
                continue
            job_id = (row.get("job_id") or "").strip()
            if not job_id:
                raise ValueError("Retail review row is missing job_id")
            if job_id in seen:
                raise ValueError(f"Duplicate stable job_id in review source: {job_id}")
            seen.add(job_id)
            jobs.append(row)
    return jobs


def load_saved_judgments(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read saved decisions without discarding any profile's existing judgments."""
    path = Path(path)
    if not path.exists():
        return {}
    saved: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"profile_id", "job_id", "relevance_score"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Saved review file is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            key = ((row.get("profile_id") or "").strip(), (row.get("job_id") or "").strip())
            if not all(key):
                raise ValueError("Saved review file contains an empty profile_id or job_id")
            if key in saved:
                raise ValueError(f"Saved review file contains duplicate judgment key: {key}")
            rating = (row.get("relevance_score") or "").strip()
            if rating and rating not in VALID_RATINGS:
                raise ValueError(f"Invalid saved relevance_score {rating!r} for {key}")
            status = (row.get("review_status") or "").strip()
            if not status:
                status = "rated" if rating else (
                    "insufficient_evidence" if (row.get("insufficient_evidence_reason") or "").strip()
                    else "unreviewed"
                )
            if status not in {"rated", "insufficient_evidence", "unreviewed"}:
                raise ValueError(f"Invalid review_status {status!r} for {key}")
            if status == "rated" and not rating:
                raise ValueError(f"Rated review is missing relevance_score for {key}")
            if status != "rated" and rating:
                raise ValueError(f"Non-rated review has relevance_score for {key}")
            saved[key] = {
                "profile_id": key[0],
                "job_id": key[1],
                "relevance_score": rating,
                "review_status": status,
                "insufficient_evidence_reason": (row.get("insufficient_evidence_reason") or "").strip(),
                "reviewer": (row.get("reviewer") or "").strip(),
                "notes": row.get("notes") or "",
            }
    return saved


def write_saved_judgment(
    path: Path,
    profile_id: str,
    job_id: str,
    *,
    relevance_score: str | None,
    review_status: str,
    insufficient_evidence_reason: str = "",
    reviewer: str = "",
    notes: str = "",
) -> None:
    """Atomically upsert one human decision while preserving all other decisions."""
    if not profile_id.strip() or not job_id.strip():
        raise ValueError("profile_id and job_id must both be nonblank")
    if relevance_score is not None and relevance_score not in VALID_RATINGS:
        raise ValueError("relevance_score must be blank, 0, 1, or 2")
    if review_status not in {"rated", "insufficient_evidence", "unreviewed"}:
        raise ValueError("review_status must be rated, insufficient_evidence, or unreviewed")
    reason = insufficient_evidence_reason.strip()
    rating = relevance_score or ""
    if review_status == "rated" and not rating:
        raise ValueError("A rated review must include a relevance score")
    if review_status != "rated" and rating:
        raise ValueError("Only rated reviews may contain relevance_score")
    if review_status == "insufficient_evidence" and not reason:
        raise ValueError("Insufficient-evidence review requires a reason")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_saved_judgments(path)
    key = (profile_id.strip(), job_id.strip())
    rows[key] = {
        "profile_id": key[0],
        "job_id": key[1],
        "relevance_score": rating,
        "review_status": review_status,
        "insufficient_evidence_reason": reason,
        "reviewer": reviewer.strip(),
        "notes": notes,
    }
    _atomic_csv(path, list(rows.values()))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in REVIEW_FIELDS})
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_progress(path: Path, job_ids: Sequence[str]) -> int:
    if not job_ids or not Path(path).is_file():
        return 0
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    job_id = data.get("job_id") if isinstance(data, dict) else None
    try:
        return list(job_ids).index(job_id)
    except ValueError:
        try:
            return min(max(int(data.get("index", 0)), 0), len(job_ids) - 1)
        except (TypeError, ValueError):
            return 0


def save_progress(path: Path, index: int, job_id: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"profile_id": PROFILE_ID, "index": index, "job_id": job_id}, indent=2)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def review_counts(job_ids: Sequence[str], judgments: Mapping[tuple[str, str], Mapping[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter({"rated": 0, "insufficient_evidence": 0, "unreviewed": 0})
    for job_id in job_ids:
        row = judgments.get((PROFILE_ID, job_id))
        status = row.get("review_status", "unreviewed") if row else "unreviewed"
        counts[status if status in counts else "unreviewed"] += 1
    return counts


def rating_from_status(judgment: Mapping[str, str] | None) -> str | None:
    if not judgment or judgment.get("review_status") != "rated":
        return None
    value = judgment.get("relevance_score", "")
    return value if value in VALID_RATINGS else None
