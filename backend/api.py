"""Anonymous demo API over a read-only prepared historical job sample."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import secrets
import sqlite3
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from fastapi import Cookie, Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import grounded_match_explanations as grounded  # noqa: E402
import job_matching_engine as matching  # noqa: E402
import resume_profile_extraction as resume_extraction  # noqa: E402
from . import compass as compass_analysis  # noqa: E402
from . import feedback_ranking  # noqa: E402
from . import signals as hiring_signals  # noqa: E402


SNAPSHOT_ID_PREFIX = "predictleads-"
SESSION_COOKIE = "pl_demo_session"
DEFAULT_SAMPLE = PROJECT_ROOT / "data/matching/ai_ml_serving_v1/jobs_prepared_corpus.jsonl"
DEFAULT_DB = PROJECT_ROOT / "backend/data/predictleads-demo.sqlite3"
RANKER_ID = "tfidf_cosine"
RANKER_VERSION = "job_matching_engine.rank_tfidf-v1"
NOTE_LIMIT = 500
STAGES = ("Saved", "Considering", "Applied", "Interviewing", "Archived")
MAX_PAGE_SIZE = 100


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def source_snapshot_id(record: Mapping[str, Any]) -> str:
    snapshot_date = record.get("snapshot_date")
    if not isinstance(snapshot_date, str) or not snapshot_date:
        raise ValueError("Prepared job is missing snapshot_date")
    return SNAPSHOT_ID_PREFIX + snapshot_date


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class Settings:
    db_path: Path
    sample_path: Path
    session_ttl_seconds: int
    secure_cookie: bool
    allowed_origins: tuple[str, ...]
    max_page_size: int = MAX_PAGE_SIZE

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = os.environ.get(
            "PREDICTLEADS_API_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
        ttl = int(os.environ.get("PREDICTLEADS_API_SESSION_TTL_SECONDS", "86400"))
        if ttl < 60:
            raise ValueError("PREDICTLEADS_API_SESSION_TTL_SECONDS must be at least 60")
        return cls(
            db_path=Path(os.environ.get("PREDICTLEADS_API_DB_PATH", str(DEFAULT_DB))).expanduser(),
            sample_path=Path(os.environ.get("PREDICTLEADS_API_JOBS_PATH", str(DEFAULT_SAMPLE))).expanduser(),
            session_ttl_seconds=ttl,
            secure_cookie=os.environ.get("PREDICTLEADS_API_COOKIE_SECURE", "0").casefold() in {"1", "true", "yes"},
            allowed_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )


class ExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    organization: str | None = Field(default=None, max_length=160)
    description: str = Field(default="", max_length=4000)
    # How the entry is shown (job / project / education). Matching and explanations still read every entry the same way.
    kind: Literal["job", "project", "education"] = "job"


class SkillEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    evidence: str = Field(default="", max_length=2000)


class ProfileData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    headline: str = Field(default="", max_length=240)
    experience: list[ExperienceEntry] = Field(default_factory=list, max_length=30)
    skills: list[SkillEntry] = Field(default_factory=list, max_length=100)
    desired_roles: list[str] = Field(default_factory=list, max_length=30)
    location: str | None = Field(default=None, max_length=240)
    arrangements: list[Literal["Remote", "Hybrid", "On-site"]] = Field(default_factory=list, max_length=3)
    constraints: str | None = Field(default=None, max_length=1000)

    @field_validator("desired_roles")
    @classmethod
    def validate_roles(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 160 for value in values):
            raise ValueError("desired role values must be nonblank and at most 160 characters")
        return values


class ProfilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=120)
    headline: str | None = Field(default=None, max_length=240)
    experience: list[ExperienceEntry] | None = Field(default=None, max_length=30)
    skills: list[SkillEntry] | None = Field(default=None, max_length=100)
    desired_roles: list[str] | None = Field(default=None, max_length=30)
    location: str | None = Field(default=None, max_length=240)
    arrangements: list[Literal["Remote", "Hybrid", "On-site"]] | None = Field(default=None, max_length=3)
    constraints: str | None = Field(default=None, max_length=1000)


class RecommendationFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, max_length=200)
    location: str | None = Field(default=None, max_length=200)
    description: Literal["any", "with", "without"] = "any"
    category: str | None = Field(default=None, max_length=120)
    seniority: str | None = Field(default=None, max_length=80)
    work_arrangement: str | None = Field(default=None, max_length=80)


class RecommendationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(min_length=1, max_length=80)
    profile_version: int = Field(ge=1)
    limit: int = Field(default=10, ge=1, le=50)
    filters: RecommendationFilters = Field(default_factory=RecommendationFilters)


class CompassRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(min_length=1, max_length=80)
    profile_version: int = Field(ge=1)
    job_ids: list[str] = Field(min_length=compass_analysis.MIN_PICKS, max_length=compass_analysis.MAX_PICKS)


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(min_length=1, max_length=80)
    profile_version: int = Field(ge=1)
    include: list[Literal["transferable_experience", "qualification_assessment", "basic_skills_gap"]] = Field(
        default_factory=lambda: ["transferable_experience", "qualification_assessment", "basic_skills_gap"],
        max_length=3,
    )


class GapBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(min_length=1, max_length=80)
    profile_version: int = Field(ge=1)
    job_ids: list[str] = Field(min_length=1, max_length=grounded.MAX_BATCH_JOBS)


class SavedJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=120)
    snapshot_id: str = Field(min_length=1, max_length=80)


class FeedbackPut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=120)
    snapshot_id: str = Field(min_length=1, max_length=80)
    signal: Literal["good", "bad"]


class SavedJobPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["Saved", "Considering", "Applied", "Interviewing", "Archived"] | None = None
    note: str | None = Field(default=None, max_length=NOTE_LIMIT)


@dataclass(frozen=True)
class JobRow:
    job_id: str
    source: Mapping[str, Any]


class JobRepository:
    """Immutable in-memory view of one prepared snapshot."""

    def __init__(self, jobs: Sequence[matching.JobRecord]):
        if not jobs:
            raise ValueError("Prepared sample contains no searchable jobs")
        self.jobs = tuple(jobs)
        snapshots = {source_snapshot_id(job.source) for job in self.jobs}
        if len(snapshots) != 1:
            raise ValueError("Prepared sample must contain exactly one snapshot")
        self.snapshot_id = snapshots.pop()
        self.snapshot_date = self.snapshot_id.removeprefix(SNAPSHOT_ID_PREFIX)
        self.by_id = {job.job_id: job for job in self.jobs}
        self.company_rows: dict[str, Mapping[str, Any]] = {}
        for job in self.jobs:
            ref = job.source.get("company_reference")
            if isinstance(ref, Mapping) and ref.get("type") == "company" and ref.get("id"):
                company_id = str(ref["id"])
                previous = self.company_rows.get(company_id)
                if previous is None:
                    self.company_rows[company_id] = job.source
                else:
                    # Prefer the first record containing the most complete same-wrapper fields.
                    current_score = sum(bool(job.source.get(key)) for key in ("company_name_included", "company_domain_included"))
                    previous_score = sum(bool(previous.get(key)) for key in ("company_name_included", "company_domain_included"))
                    if current_score > previous_score:
                        self.company_rows[company_id] = job.source

    @classmethod
    def from_jsonl(cls, path: Path) -> "JobRepository":
        jobs, _ = matching.load_prepared_jobs(path)
        return cls(jobs)

    def get(self, job_id: str, snapshot_id: str) -> JobRow:
        self.require_snapshot(snapshot_id)
        job = self.by_id.get(job_id)
        if job is None:
            raise APIError(404, "job_not_found", "The job is not present in the requested snapshot.")
        return JobRow(job_id=job.job_id, source=job.source)

    def require_snapshot(self, snapshot_id: str) -> None:
        if snapshot_id != self.snapshot_id:
            raise APIError(404, "snapshot_not_found", "The requested snapshot is not loaded.")

    def company(self, company_id: str, snapshot_id: str) -> Mapping[str, Any]:
        self.require_snapshot(snapshot_id)
        source = self.company_rows.get(company_id)
        if source is None:
            raise APIError(404, "company_not_in_loaded_snapshot", "No same-wrapper company reference was found in the loaded snapshot.")
        return source


class RankingBackend(Protocol):
    ranker_id: str
    version: str
    score_type: str

    def rank(self, profile: matching.CandidateProfile, limit: int, candidate_indices: Sequence[int]) -> list[dict[str, Any]]: ...


class TfIdfRankingBackend:
    """Adapter for the existing fit_tfidf/rank_tfidf implementation."""

    ranker_id = RANKER_ID
    version = RANKER_VERSION
    score_type = "cosine_similarity"

    feedback_version = feedback_ranking.VERSION

    def __init__(self, jobs: Sequence[matching.JobRecord]):
        self.corpus = matching.fit_tfidf(jobs)
        self.job_ids = [job.job_id for job in self.corpus.jobs]
        self.index_of = {job_id: index for index, job_id in enumerate(self.job_ids)}

    def rank(
        self,
        profile: matching.CandidateProfile,
        limit: int,
        candidate_indices: Sequence[int],
        *,
        liked_job_ids: Sequence[str] = (),
        disliked_job_ids: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        liked = [self.index_of[j] for j in liked_job_ids if j in self.index_of]
        disliked = [self.index_of[j] for j in disliked_job_ids if j in self.index_of]
        if not liked and not disliked:
            return matching.rank_tfidf(
                self.corpus,
                profile,
                top_k=limit,
                candidate_indices=candidate_indices,
                condition="api_filters",
            )
        rated = set(liked) | set(disliked)
        eligible, _ = matching._eligible_jobs(self.corpus, profile, [i for i in candidate_indices if i not in rated])
        if not eligible:
            return []
        query = feedback_ranking.adjusted_sparse_query(
            self.corpus.vectorizer.transform([profile.tfidf_text()]), self.corpus.job_matrix, liked, disliked)
        scores = (self.corpus.job_matrix @ query.T).toarray().ravel()[np.asarray(eligible)]
        top = feedback_ranking.top_ranked(scores, eligible, self.job_ids, limit)
        return [{"job_id": self.job_ids[index], "rank": rank, "score": score} for rank, (index, score) in enumerate(top, 1)]


class SessionStore:
    def __init__(self, path: Path, ttl_seconds: int):
        self.path = path
        self.ttl_seconds = ttl_seconds

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_sessions (
                    session_hash TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_activity_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS profiles (
                    profile_id TEXT PRIMARY KEY,
                    session_hash TEXT NOT NULL UNIQUE REFERENCES demo_sessions(session_hash) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS saved_jobs (
                    session_hash TEXT NOT NULL REFERENCES demo_sessions(session_hash) ON DELETE CASCADE,
                    snapshot_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL CHECK(stage IN ('Saved','Considering','Applied','Interviewing','Archived')),
                    note TEXT NOT NULL CHECK(length(note) <= 500),
                    saved_at TEXT NOT NULL,
                    PRIMARY KEY(session_hash, snapshot_id, job_id)
                );
                CREATE TABLE IF NOT EXISTS assessment_cache (
                    cache_key TEXT PRIMARY KEY,
                    session_hash TEXT NOT NULL REFERENCES demo_sessions(session_hash) ON DELETE CASCADE,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_feedback (
                    session_hash TEXT NOT NULL REFERENCES demo_sessions(session_hash) ON DELETE CASCADE,
                    snapshot_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    signal TEXT NOT NULL CHECK(signal IN ('good','bad')),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(session_hash, snapshot_id, job_id)
                );
                CREATE INDEX IF NOT EXISTS saved_jobs_by_session ON saved_jobs(session_hash, saved_at DESC);
                """
            )

    def cleanup_expired(self) -> None:
        now = utc_text()
        with self.connect() as db:
            db.execute("DELETE FROM demo_sessions WHERE expires_at <= ?", (now,))

    def create_session(self) -> tuple[str, str, str]:
        token = secrets.token_urlsafe(32)
        session_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now_dt = utc_now()
        now = utc_text(now_dt)
        expires = utc_text(now_dt + timedelta(seconds=self.ttl_seconds))
        with self.connect() as db:
            db.execute(
                "INSERT INTO demo_sessions(session_hash, created_at, last_activity_at, expires_at) VALUES(?,?,?,?)",
                (session_hash, now, now, expires),
            )
        return token, session_hash, expires

    def validate_and_refresh(self, session_hash: str) -> str | None:
        now_dt = utc_now()
        now = utc_text(now_dt)
        expires = utc_text(now_dt + timedelta(seconds=self.ttl_seconds))
        with self.connect() as db:
            row = db.execute("SELECT expires_at FROM demo_sessions WHERE session_hash = ?", (session_hash,)).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                db.execute("DELETE FROM demo_sessions WHERE session_hash = ?", (session_hash,))
                return None
            db.execute(
                "UPDATE demo_sessions SET last_activity_at = ?, expires_at = ? WHERE session_hash = ?",
                (now, expires, session_hash),
            )
        return expires

    def delete_session(self, session_hash: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM demo_sessions WHERE session_hash = ?", (session_hash,))

    def session_is_active(self, session_hash: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM demo_sessions WHERE session_hash=? AND expires_at>?",
                (session_hash, utc_text()),
            ).fetchone()
        return row is not None

    def get_profile(self, session_hash: str) -> sqlite3.Row | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM profiles WHERE session_hash = ?", (session_hash,)).fetchone()

    def create_profile(self, session_hash: str, profile: Mapping[str, Any]) -> sqlite3.Row:
        profile_id = "prf_" + uuid.uuid4().hex
        now = utc_text()
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO profiles(profile_id, session_hash, version, profile_json, updated_at) VALUES(?,?,?,?,?)",
                    (profile_id, session_hash, 1, json.dumps(profile, ensure_ascii=False, separators=(",", ":")), now),
                )
        except sqlite3.IntegrityError as exc:
            raise APIError(409, "profile_exists", "This demo session already has a profile.") from exc
        return self.get_profile(session_hash)  # type: ignore[return-value]

    def patch_profile(self, session_hash: str, expected_version: int, profile: Mapping[str, Any]) -> sqlite3.Row:
        now = utc_text()
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE profiles SET version = version + 1, profile_json = ?, updated_at = ? WHERE session_hash = ? AND version = ?",
                (json.dumps(profile, ensure_ascii=False, separators=(",", ":")), now, session_hash, expected_version),
            )
            if cursor.rowcount != 1:
                exists = db.execute("SELECT 1 FROM profiles WHERE session_hash = ?", (session_hash,)).fetchone()
                if exists is None:
                    raise APIError(404, "profile_not_found", "No candidate profile exists in this demo session.")
                raise APIError(409, "profile_version_conflict", "Reload the profile before saving changes.")
        return self.get_profile(session_hash)  # type: ignore[return-value]

    def save_job(self, session_hash: str, snapshot_id: str, job_id: str) -> sqlite3.Row:
        now = utc_text()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO saved_jobs(session_hash,snapshot_id,job_id,stage,note,saved_at) VALUES(?,?,?,?,?,?)",
                (session_hash, snapshot_id, job_id, "Saved", "", now),
            )
            row = db.execute(
                "SELECT * FROM saved_jobs WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            ).fetchone()
        return row

    def list_saved_jobs(self, session_hash: str, snapshot_id: str | None) -> list[sqlite3.Row]:
        with self.connect() as db:
            if snapshot_id is None:
                return db.execute("SELECT * FROM saved_jobs WHERE session_hash=? ORDER BY saved_at DESC, job_id", (session_hash,)).fetchall()
            return db.execute(
                "SELECT * FROM saved_jobs WHERE session_hash=? AND snapshot_id=? ORDER BY saved_at DESC, job_id",
                (session_hash, snapshot_id),
            ).fetchall()

    def patch_saved_job(self, session_hash: str, snapshot_id: str, job_id: str, patch: Mapping[str, Any]) -> sqlite3.Row:
        if not patch:
            raise APIError(400, "empty_update", "Provide a stage or note to update.")
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM saved_jobs WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            ).fetchone()
            if row is None:
                raise APIError(404, "saved_job_not_found", "The saved job does not exist in this session and snapshot.")
            stage = patch.get("stage", row["stage"])
            note = patch.get("note", row["note"])
            db.execute(
                "UPDATE saved_jobs SET stage=?, note=? WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (stage, note, session_hash, snapshot_id, job_id),
            )
            return db.execute(
                "SELECT * FROM saved_jobs WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            ).fetchone()

    def delete_saved_job(self, session_hash: str, snapshot_id: str, job_id: str) -> None:
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM saved_jobs WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            )
            if cursor.rowcount != 1:
                raise APIError(404, "saved_job_not_found", "The saved job does not exist in this session and snapshot.")

    def set_feedback(self, session_hash: str, snapshot_id: str, job_id: str, signal: str) -> sqlite3.Row:
        with self.connect() as db:
            db.execute(
                "INSERT INTO job_feedback(session_hash,snapshot_id,job_id,signal,created_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(session_hash,snapshot_id,job_id) DO UPDATE SET signal=excluded.signal, created_at=excluded.created_at",
                (session_hash, snapshot_id, job_id, signal, utc_text()),
            )
            return db.execute(
                "SELECT * FROM job_feedback WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            ).fetchone()

    def list_feedback(self, session_hash: str, snapshot_id: str | None) -> list[sqlite3.Row]:
        with self.connect() as db:
            if snapshot_id is None:
                return db.execute("SELECT * FROM job_feedback WHERE session_hash=? ORDER BY created_at, job_id", (session_hash,)).fetchall()
            return db.execute(
                "SELECT * FROM job_feedback WHERE session_hash=? AND snapshot_id=? ORDER BY created_at, job_id",
                (session_hash, snapshot_id),
            ).fetchall()

    def delete_feedback(self, session_hash: str, snapshot_id: str, job_id: str) -> None:
        with self.connect() as db:
            cursor = db.execute(
                "DELETE FROM job_feedback WHERE session_hash=? AND snapshot_id=? AND job_id=?",
                (session_hash, snapshot_id, job_id),
            )
            if cursor.rowcount != 1:
                raise APIError(404, "feedback_not_found", "No feedback exists for that job in this session and snapshot.")

    def get_assessment(self, cache_key: str, session_hash: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT response_json FROM assessment_cache WHERE cache_key=? AND session_hash=?",
                (cache_key, session_hash),
            ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def cache_assessment(self, cache_key: str, session_hash: str, response: Mapping[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO assessment_cache(cache_key,session_hash,response_json,created_at) VALUES(?,?,?,?)",
                (cache_key, session_hash, json.dumps(response, ensure_ascii=False, separators=(",", ":")), utc_text()),
            )


def _field_state(record: Mapping[str, Any], field: str) -> str:
    flags = record.get("quality_flags")
    if isinstance(flags, Mapping) and isinstance(flags.get(field), str):
        return str(flags[field])
    if field not in record:
        return "absent"
    value = record[field]
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank"
    return "present"


def _company_summary(record: Mapping[str, Any]) -> dict[str, Any] | None:
    ref = record.get("company_reference")
    if not isinstance(ref, Mapping) or ref.get("type") != "company" or not ref.get("id"):
        return None
    if record.get("company_match_state") != "matched":
        return {"company_id": str(ref["id"]), "name": None, "domain": None, "match_state": record.get("company_match_state", "unknown")}
    return {
        "company_id": str(ref["id"]),
        "name": record.get("company_name_included"),
        "domain": record.get("company_domain_included"),
        "match_state": "same_wrapper_match",
    }


def _job_summary(row: JobRow, snapshot_id: str, snapshot_date: str) -> dict[str, Any]:
    source = row.source
    title = (
        source.get("title_searchable")
        or source.get("title_normalized")
        or source.get("title_translated")
        or source.get("title_raw")
        or "Unknown title"
    )
    description = source.get("description_text")
    if not isinstance(description, str):
        description = None
    status_raw = source.get("status_raw")
    status_state = status_raw if status_raw == "closed" else "unknown"
    dates = source.get("dates_utc") if isinstance(source.get("dates_utc"), Mapping) else {}
    description_excerpt = description[:240] if description else None
    location_data = source.get("location_data_raw")
    return {
        "job_id": row.job_id,
        "title": str(title),
        "normalized_title": source.get("title_normalized"),
        "occupation": {
            "code": (source.get("onet_data") or {}).get("code") if isinstance(source.get("onet_data"), Mapping) else None,
            "name": (source.get("onet_data") or {}).get("occupation_name") if isinstance(source.get("onet_data"), Mapping) else None,
            "validated": False,
        },
        "company_ref": source.get("company_reference"),
        "company_summary": _company_summary(source),
        "location": source.get("location_raw"),
        "location_data": location_data,
        "work_arrangement": {"value": None, "state": "unknown"},
        "salary_raw": source.get("salary_raw"),
        "description_available": bool(source.get("description_usable") and description),
        "description_excerpt": description_excerpt,
        "source_url": source.get("url"),
        "observed_on": snapshot_date,
        "source_status": {"raw": status_raw, "state": status_state},
        "availability": {"state": "unverified", "verified_at": None},
        "observed": {
            "first_seen_at": dates.get("first_seen_at"),
            "last_seen_at": dates.get("last_seen_at"),
            "last_processed_at": dates.get("last_processed_at"),
            "posted_at": dates.get("posted_at"),
        },
        "snapshot": {"snapshot_id": snapshot_id, "date": snapshot_date, "source_key": source.get("source_key"), "source_line": source.get("source_line")},
        "field_states": {
            "title": _field_state(source, "title_raw"),
            "description": _field_state(source, "description_raw"),
            "location": _field_state(source, "location_raw"),
            "salary_raw": _field_state(source, "salary_raw"),
            "status": _field_state(source, "status_raw"),
            "posted_at": _field_state(dates, "posted_at"),
        },
        "categories": source.get("categories", []),
        "seniority": source.get("seniority"),
    }


def _job_details(row: JobRow, snapshot_id: str, snapshot_date: str) -> dict[str, Any]:
    summary = _job_summary(row, snapshot_id, snapshot_date)
    source = row.source
    summary.update({
        "description": source.get("description_text"),
        "description_raw": source.get("description_raw"),
        "tags": source.get("tags", []),
        "contract_types": source.get("contract_types", []),
        "salary_data": source.get("salary_data_raw"),
        "quality_flags": source.get("quality_flags", {}),
        "historical_snapshot": True,
    })
    return summary


def _normalize_locations(raw: Mapping[str, Any]) -> str:
    values: list[str] = []
    location = raw.get("location_raw")
    if isinstance(location, str):
        values.append(location)
    structured = raw.get("location_data_raw")
    if isinstance(structured, Sequence) and not isinstance(structured, (str, bytes)):
        for entry in structured:
            if isinstance(entry, Mapping):
                values.extend(str(value) for value in entry.values() if isinstance(value, (str, int)) and value)
    elif isinstance(structured, Mapping):
        values.extend(str(value) for value in structured.values() if isinstance(value, (str, int)) and value)
    return " ".join(values).casefold()


TITLE_FIRST_MIN_RESULTS = 5  # with fewer title matches than this, a search falls back to the whole listing text


def _search_indices(repository: JobRepository, filters: RecommendationFilters, *, title_only: bool = False) -> list[int]:
    """Listings matching the filters. Every word of the query must appear (in any order); `title_only` looks at titles only."""
    tokens = (filters.query or "").strip().casefold().split()
    location = (filters.location or "").strip().casefold()
    output: list[int] = []
    for index, job in enumerate(repository.jobs):
        source = job.source
        if tokens:
            if title_only:
                haystack = " ".join(str(source.get(key) or "") for key in ("title_searchable", "title_normalized", "title_raw")).casefold()
            else:
                title, occupation, description = matching._job_search_fields(source)
                haystack = " ".join((title, occupation, description, str(source.get("company_name_included") or ""), " ".join(source.get("tags") or []), " ".join(source.get("categories") or []))).casefold()
            if not all(token in haystack for token in tokens):
                continue
        if location and location not in _normalize_locations(source):
            continue
        has_description = bool(job.usable_description)
        if filters.description == "with" and not has_description:
            continue
        if filters.description == "without" and has_description:
            continue
        if filters.category and filters.category.casefold() not in {str(value).casefold() for value in source.get("categories", [])}:
            continue
        if filters.seniority and str(source.get("seniority") or "").casefold() != filters.seniority.casefold():
            continue
        output.append(index)
    return output


def _search_scoped(repository: JobRepository, filters: RecommendationFilters) -> tuple[list[int], str]:
    """Title matches first. Returns the listings and the scope used: "all" (no query), "title" or "anywhere"."""
    if not (filters.query or "").strip():
        return _search_indices(repository, filters), "all"
    by_title = _search_indices(repository, filters, title_only=True)
    if len(by_title) >= TITLE_FIRST_MIN_RESULTS:
        return by_title, "title"
    return _search_indices(repository, filters), "anywhere"


def _filters_digest(snapshot_id: str, filters: RecommendationFilters) -> str:
    payload = {"snapshot_id": snapshot_id, **filters.model_dump()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _encode_cursor(offset: int, digest: str) -> str:
    raw = json.dumps({"offset": offset, "filters": digest}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None, digest: str) -> int:
    if cursor is None:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        decoded = json.loads(raw)
        offset = decoded["offset"]
        if not isinstance(offset, int) or offset < 0 or decoded["filters"] != digest:
            raise ValueError
        return offset
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise APIError(400, "invalid_cursor", "The cursor is malformed or belongs to different filters.") from None


def _profile_row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "profile_id": row["profile_id"],
        "version": row["version"],
        "updated_at": row["updated_at"],
        "profile": json.loads(row["profile_json"]),
    }


def _saved_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "snapshot_id": row["snapshot_id"],
        "stage": row["stage"],
        "note": row["note"],
        "saved_at": row["saved_at"],
    }


def _candidate_from_profile(profile_id: str, profile_data: Mapping[str, Any]) -> matching.CandidateProfile:
    """The stored API profile as the matching engine's candidate (shared by recommendations and the Compass)."""
    return matching.CandidateProfile(
        profile_id=profile_id,
        experience=tuple(
            " ".join(part for part in (entry.get("title"), entry.get("organization") or "", entry.get("description")) if part)
            for entry in profile_data.get("experience", [])
        ),
        evidenced_skills=tuple(
            ": ".join(part for part in (item.get("name", ""), item.get("evidence", "")) if part)
            for item in profile_data.get("skills", [])
        ),
        desired_roles=tuple(profile_data.get("desired_roles", [])),
        location=profile_data.get("location"),
        preferences={
            "keywords": [text for text in (profile_data.get("headline"),) if text],
            "work_arrangement": list(profile_data.get("arrangements", [])),
        },
    )


def _candidate_mapping(profile_id: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "experience": profile.get("experience", []),
        "skills": profile.get("skills", []),
        "desired_roles": profile.get("desired_roles", []),
        "education": [],
        "training": [],
        "certifications": [],
        "projects": [],
    }


class DisabledAssessmentProvider:
    model = "disabled"

    def generate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise grounded.ProviderUnavailable("Assessment generation is disabled")

    def generate_batch(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise grounded.ProviderUnavailable("Assessment generation is disabled")


def _assessment_cache_key(
    session_hash: str,
    profile_id: str,
    profile_version: int,
    job_id: str,
    snapshot_id: str,
    content_hash: str,
    model_id: str,
) -> str:
    prompt_hash = hashlib.sha256(grounded._SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    material = {
        "session_hash": session_hash,
        "profile_id": profile_id,
        "profile_version": profile_version,
        "job_id": job_id,
        "snapshot_id": snapshot_id,
        "content_hash": content_hash,
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "schema_version": grounded.SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _default_assessment_provider() -> "grounded.ExplanationProvider | None":
    """The AI gap analysis sends profile and job text to OpenAI, so it is off unless PREDICTLEADS_ASSESSMENT=openai."""
    if os.environ.get("PREDICTLEADS_ASSESSMENT", "").strip().lower() != "openai":
        return None
    return grounded.OpenAIProvider.from_environment()


def _default_extraction_provider() -> "resume_extraction.ResumeProvider | None":
    """Resume "magic fill" sends resume text to OpenAI, so it is off unless PREDICTLEADS_RESUME_EXTRACTION=openai."""
    if os.environ.get("PREDICTLEADS_RESUME_EXTRACTION", "").strip().lower() != "openai":
        return None
    return resume_extraction.OpenAIResumeProvider.from_environment()


class ResumeExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume_text: str | None = Field(default=None, min_length=resume_extraction.MIN_RESUME_CHARS, max_length=resume_extraction.MAX_RESUME_CHARS)
    resume_pdf_base64: str | None = Field(default=None, max_length=resume_extraction.MAX_PDF_BYTES * 4 // 3 + 16)

    @model_validator(mode="after")
    def exactly_one_source(self) -> "ResumeExtractionRequest":
        if (self.resume_text is None) == (self.resume_pdf_base64 is None):
            raise ValueError("send exactly one of resume_text or resume_pdf_base64")
        return self


def _default_ranker(jobs: Sequence[matching.JobRecord]) -> "RankingBackend":
    """TF-IDF by default; PREDICTLEADS_API_RANKER=rrf selects TF-IDF+BGE fusion (needs PREDICTLEADS_API_CORPUS_DIR)."""
    if os.environ.get("PREDICTLEADS_API_RANKER", "tfidf").strip().lower() == "rrf":
        from .rrf_ranker import RrfRankingBackend

        corpus_dir = os.environ.get("PREDICTLEADS_API_CORPUS_DIR")
        if not corpus_dir:
            raise ValueError("PREDICTLEADS_API_RANKER=rrf requires PREDICTLEADS_API_CORPUS_DIR.")
        return RrfRankingBackend(jobs, Path(corpus_dir))
    return TfIdfRankingBackend(jobs)


def _load_signals() -> "hiring_signals.SignalStore | None":
    directory = hiring_signals.default_signals_dir()
    if directory is None or not (directory / "company_signals.csv").exists():
        return None
    return hiring_signals.SignalStore.load(directory)


def create_app(
    *,
    settings: Settings | None = None,
    provider: grounded.ExplanationProvider | None = None,
    repository: JobRepository | None = None,
    ranker: RankingBackend | None = None,
    extraction_provider: "resume_extraction.ResumeProvider | None" = None,
    signal_store: "hiring_signals.SignalStore | None" = None,
) -> FastAPI:
    config = settings or Settings.from_environment()
    store = SessionStore(config.db_path, config.session_ttl_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.initialize()
        jobs = repository or JobRepository.from_jsonl(config.sample_path)
        app.state.job_repository = jobs
        app.state.ranker = ranker or _default_ranker(jobs.jobs)
        app.state.store = store
        app.state.assessment_provider = provider if provider is not None else _default_assessment_provider()
        app.state.extraction_provider = extraction_provider or _default_extraction_provider()
        app.state.signals = signal_store if signal_store is not None else _load_signals()
        yield

    app = FastAPI(title="PredictLeads Hackathon Demo API", version="1.0.0", lifespan=lifespan)
    app.state.job_repository = repository
    app.state.ranker = ranker
    app.state.store = store
    app.state.assessment_provider = provider
    app.state.extraction_provider = extraction_provider

    @app.middleware("http")
    async def reject_untrusted_write_origins(request: Request, call_next):
        request.state.request_id = "req_" + uuid.uuid4().hex
        request_token = request.cookies.get(SESSION_COOKIE)
        if request.method in {"POST", "PATCH", "PUT", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin not in config.allowed_origins:
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "origin_not_allowed", "message": "The request origin is not allowed.", "request_id": request.state.request_id, "retryable": False}},
                )
        response = await call_next(request)
        existing_cookie_headers = response.headers.getlist("set-cookie")
        cookie_was_set = any(header.startswith(f"{SESSION_COOKIE}=") for header in existing_cookie_headers)
        is_session_creation = request.method == "POST" and request.url.path == "/api/v1/demo-sessions"
        if request_token and not cookie_was_set and not is_session_creation:
            session_hash = hashlib.sha256(request_token.encode("utf-8")).hexdigest()
            if store.session_is_active(session_hash):
                response.set_cookie(
                    SESSION_COOKIE,
                    request_token,
                    max_age=config.session_ttl_seconds,
                    httponly=True,
                    secure=config.secure_cookie,
                    samesite="lax",
                    path="/",
                )
            else:
                response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "If-Match"],
    )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "request_id": getattr(request.state, "request_id", None), "retryable": exc.retryable}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        fields = [".".join(str(part) for part in error.get("loc", ()) if part != "body") for error in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "One or more request fields are invalid.", "request_id": getattr(request.state, "request_id", None), "fields": fields, "retryable": False}},
        )

    def session_dependency(
        token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    ) -> str:
        if not token:
            raise APIError(401, "demo_session_required", "Create a demo session before using this endpoint.")
        session_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires = store.validate_and_refresh(session_hash)
        if expires is None:
            raise APIError(401, "demo_session_expired", "The demo session is missing or expired.")
        return session_hash

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/snapshots/current")
    def current_snapshot() -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        return {
            "snapshot_id": jobs.snapshot_id,
            "date": jobs.snapshot_date,
            "job_count": len(jobs.jobs),
            "ranker": {"id": app.state.ranker.ranker_id, "version": app.state.ranker.version},
            "availability": "unverified",
            "note": "Historical listings from one snapshot; none are verified as currently open.",
        }

    @app.post("/api/v1/demo-sessions", status_code=201)
    def create_demo_session(response: Response) -> dict[str, Any]:
        store.cleanup_expired()
        token, _, expires = store.create_session()
        response.set_cookie(
            SESSION_COOKIE,
            token,
            max_age=config.session_ttl_seconds,
            httponly=True,
            secure=config.secure_cookie,
            samesite="lax",
            path="/",
        )
        return {"session": {"expires_at": expires, "authentication": "anonymous_demo_session"}}

    @app.delete("/api/v1/demo-sessions/current", status_code=204)
    def reset_demo_session(
        response: Response,
        session_hash: str = Depends(session_dependency),
    ) -> Response:
        store.delete_session(session_hash)
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.status_code = 204
        return response

    @app.post("/api/v1/profiles", status_code=201)
    def create_profile(
        payload: ProfileData,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        row = store.create_profile(session_hash, payload.model_dump())
        return _profile_row_payload(row)

    @app.get("/api/v1/profiles/current")
    def read_current_profile(session_hash: str = Depends(session_dependency)) -> dict[str, Any]:
        row = store.get_profile(session_hash)
        if row is None:
            raise APIError(404, "profile_not_found", "No candidate profile exists in this demo session.")
        return _profile_row_payload(row)

    @app.get("/api/v1/profiles/{profile_id}")
    def read_profile(
        profile_id: str,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        row = store.get_profile(session_hash)
        if row is None or row["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        return _profile_row_payload(row)

    @app.patch("/api/v1/profiles/{profile_id}")
    def update_profile(
        profile_id: str,
        payload: ProfilePatch,
        if_match: str | None = Header(default=None, alias="If-Match"),
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        if if_match is None:
            raise APIError(428, "if_match_required", "Send the current profile version in If-Match.")
        try:
            expected_version = int(if_match.strip().strip('"'))
        except ValueError:
            raise APIError(400, "invalid_if_match", "If-Match must contain the current integer profile version.") from None
        if expected_version < 1:
            raise APIError(400, "invalid_if_match", "If-Match must contain the current integer profile version.")
        current = store.get_profile(session_hash)
        if current is None or current["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        changes = payload.model_dump(exclude_unset=True)
        if not changes:
            raise APIError(400, "empty_update", "Provide at least one profile field to update.")
        existing = json.loads(current["profile_json"])
        for key, value in changes.items():
            if key == "name" and (value is None or not value.strip()):
                raise APIError(422, "validation_error", "Profile name must remain nonblank.")
            if value is None and key in {"headline", "experience", "skills", "desired_roles", "arrangements"}:
                raise APIError(422, "validation_error", f"{key} cannot be null; use an empty value to clear it.")
            existing[key] = value
        try:
            validated = ProfileData.model_validate(existing)
        except PydanticValidationError:
            raise APIError(422, "validation_error", "One or more profile fields are invalid.") from None
        row = store.patch_profile(session_hash, expected_version, validated.model_dump())
        return _profile_row_payload(row)

    @app.get("/api/v1/jobs")
    def search_jobs(
        snapshot_id: str,
        q: str | None = Query(default=None, max_length=200),
        location: str | None = Query(default=None, max_length=200),
        description: Literal["any", "with", "without"] = "any",
        category: str | None = Query(default=None, max_length=120),
        seniority: str | None = Query(default=None, max_length=80),
        work_arrangement: str | None = None,
        limit: int = Query(default=20, ge=1, le=MAX_PAGE_SIZE),
        cursor: str | None = None,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        if work_arrangement is not None:
            raise APIError(400, "unsupported_filter", "Work arrangement is not represented in this prepared sample.")
        filters = RecommendationFilters(query=q, location=location, description=description, category=category, seniority=seniority)
        jobs: JobRepository = app.state.job_repository
        jobs.require_snapshot(snapshot_id)
        matching_indices = _search_indices(jobs, filters)
        digest = _filters_digest(snapshot_id, filters)
        offset = _decode_cursor(cursor, digest)
        page_indices = matching_indices[offset:offset + limit]
        next_offset = offset + len(page_indices)
        next_cursor = _encode_cursor(next_offset, digest) if next_offset < len(matching_indices) else None
        return {
            "items": [_job_summary(JobRow(jobs.jobs[i].job_id, jobs.jobs[i].source), snapshot_id, jobs.snapshot_date) for i in page_indices],
            "page": {"limit": limit, "next_cursor": next_cursor},
            "snapshot": {"snapshot_id": snapshot_id, "date": jobs.snapshot_date},
        }

    @app.get("/api/v1/jobs/{job_id}")
    def read_job(
        job_id: str,
        snapshot_id: str,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        row = jobs.get(job_id, snapshot_id)
        return _job_details(row, snapshot_id, jobs.snapshot_date)

    def _signals() -> "hiring_signals.SignalStore":
        store_ = getattr(app.state, "signals", None)
        if store_ is None:
            raise APIError(503, "signals_unavailable", "Hiring signals have not been built. Run code/hiring_signals.py build.", retryable=False)
        return store_

    @app.get("/api/v1/signals/overview")
    def signals_overview(session_hash: str = Depends(session_dependency)) -> dict[str, Any]:
        return _signals().overview()

    @app.get("/api/v1/signals/patterns")
    def signals_patterns(session_hash: str = Depends(session_dependency)) -> dict[str, Any]:
        return {"patterns": _signals().patterns()}

    @app.get("/api/v1/signals/companies")
    def signals_companies(
        pattern: Literal["burst", "sustained", "cooling", "quiet"] | None = None,
        family: Literal["ai_ml", "data", "technical_sales"] | None = None,
        q: str | None = Query(default=None, max_length=80),
        sort: Literal["new_30", "active", "surge_z"] = "new_30",
        limit: int = Query(default=25, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        return _signals().leaderboard(pattern, family, (q or "").strip() or None, sort, limit, offset)

    @app.get("/api/v1/signals/companies/{company_id}")
    def signals_company(company_id: str, session_hash: str = Depends(session_dependency)) -> dict[str, Any]:
        found = _signals().company(company_id)
        if found is None:
            raise APIError(404, "signals_company_not_found", "No hiring signals for that company.")
        return found

    @app.get("/api/v1/companies/{company_id}")
    def read_company(
        company_id: str,
        snapshot_id: str,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        source = jobs.company(company_id, snapshot_id)
        return {
            "company_id": company_id,
            "name": source.get("company_name_included"),
            "domain": source.get("company_domain_included"),
            "description": None,
            "location": None,
            "snapshot": {"snapshot_id": snapshot_id, "date": jobs.snapshot_date},
            "source_scope": "same_wrapper_company_lite_only",
            "full_company_record_loaded": False,
            "field_states": {
                "name": _field_state(source, "company_name_included"),
                "domain": _field_state(source, "company_domain_included"),
                "description": "absent",
                "location": "absent",
            },
        }

    @app.post("/api/v1/profiles/{profile_id}/recommendations")
    def recommendations(
        profile_id: str,
        payload: RecommendationsRequest,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        jobs.require_snapshot(payload.snapshot_id)
        if payload.filters.work_arrangement is not None:
            raise APIError(400, "unsupported_filter", "Work arrangement is not represented in this prepared sample.")
        profile_row = store.get_profile(session_hash)
        if profile_row is None or profile_row["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        if profile_row["version"] != payload.profile_version:
            raise APIError(409, "profile_version_conflict", "Reload the current profile before requesting recommendations.")
        profile_data = json.loads(profile_row["profile_json"])
        candidate = _candidate_from_profile(profile_id, profile_data)
        indices, search_scope = _search_scoped(jobs, payload.filters)
        feedback_rows = store.list_feedback(session_hash, payload.snapshot_id)
        liked = [row["job_id"] for row in feedback_rows if row["signal"] == "good"]
        disliked = [row["job_id"] for row in feedback_rows if row["signal"] == "bad"]
        ranker = app.state.ranker
        if liked or disliked:
            ranked = ranker.rank(candidate, payload.limit, indices, liked_job_ids=liked, disliked_job_ids=disliked)
        else:
            ranked = ranker.rank(candidate, payload.limit, indices)
        similar = ranker.most_similar_liked([str(r["job_id"]) for r in ranked], liked) if liked and hasattr(ranker, "most_similar_liked") else {}
        items = []
        for result in ranked:
            row = jobs.get(str(result["job_id"]), payload.snapshot_id)
            item = {
                "rank": int(result["rank"]),
                "retrieval_score": float(result["score"]),
                "job": _job_summary(row, payload.snapshot_id, jobs.snapshot_date),
            }
            near = similar.get(str(result["job_id"]))
            if near is not None:
                liked_row = jobs.by_id.get(near["job_id"])
                item["similar_to_liked"] = {
                    "job_id": near["job_id"],
                    "title": _job_summary(JobRow(liked_row.job_id, liked_row.source), payload.snapshot_id, jobs.snapshot_date)["title"] if liked_row else None,
                    "similarity": near["similarity"],
                }
            items.append(item)
        return {
            "profile_id": profile_id,
            "profile_version": profile_row["version"],
            "ranking": {
                "ranker_id": ranker.ranker_id,
                "version": ranker.version,
                "score_type": ranker.score_type,
                "calibration": "uncalibrated",
            },
            "feedback": {
                "good": len(liked),
                "bad": len(disliked),
                "applied": bool(liked or disliked),
                "method": getattr(ranker, "feedback_version", None) if (liked or disliked) else None,
                "note": "Rated jobs are left out of new results. Feedback adapts this session's ranking only; it does not retrain the model.",
            },
            "snapshot": {"snapshot_id": payload.snapshot_id, "date": jobs.snapshot_date},
            "search": {"query": (payload.filters.query or "").strip(), "scope": search_scope},
            "items": items,
        }

    @app.post("/api/v1/profiles/{profile_id}/compass")
    def compass(
        profile_id: str,
        payload: CompassRequest,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        """Compass analysis of 2-20 picked jobs against the saved profile (local embeddings only, no outside AI call)."""
        jobs: JobRepository = app.state.job_repository
        jobs.require_snapshot(payload.snapshot_id)
        ranker = app.state.ranker
        if not getattr(ranker, "supports_compass", False):
            raise APIError(409, "compass_unavailable", "The Compass needs the embedding ranker. Start the backend with the RRF ranker.")
        profile_row = store.get_profile(session_hash)
        if profile_row is None or profile_row["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        if profile_row["version"] != payload.profile_version:
            raise APIError(409, "profile_version_conflict", "Reload the current profile before running the Compass.")
        profile_data = json.loads(profile_row["profile_json"])
        unique_ids = list(dict.fromkeys(payload.job_ids))
        picks = []
        for job_id in unique_ids:
            record = jobs.by_id.get(job_id)
            if record is None:
                raise APIError(404, "job_not_found", "One of the picked jobs is not present in the requested snapshot.")
            picks.append(record)
        rated = [row["job_id"] for row in store.list_feedback(session_hash, payload.snapshot_id)]
        try:
            result = compass_analysis.analyze(
                ranker, picks=picks, all_jobs=jobs.jobs, profile=_candidate_from_profile(profile_id, profile_data),
                profile_data=profile_data, excluded_job_ids=rated,
            )
        except compass_analysis.CompassError as exc:
            raise APIError(422, exc.code, str(exc)) from None
        for item in result["jobs"]:
            record = jobs.by_id[item["job_id"]]
            item["job"] = _job_summary(JobRow(record.job_id, record.source), payload.snapshot_id, jobs.snapshot_date)
        return {
            "profile_id": profile_id,
            "profile_version": profile_row["version"],
            "snapshot": {"snapshot_id": payload.snapshot_id, "date": jobs.snapshot_date},
            "ranking": {"ranker_id": ranker.ranker_id, "version": ranker.version},
            **result,
        }

    @app.post("/api/v1/profiles/{profile_id}/compass/gap-analysis")
    def compass_gap_analysis(
        profile_id: str,
        payload: GapBatchRequest,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        """AI skills gap analysis for the jobs picked in the Compass: ONE provider request covering every job."""
        jobs: JobRepository = app.state.job_repository
        jobs.require_snapshot(payload.snapshot_id)
        profile_row = store.get_profile(session_hash)
        if profile_row is None or profile_row["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        if profile_row["version"] != payload.profile_version:
            raise APIError(409, "profile_version_conflict", "Reload the current profile before requesting the analysis.")
        unique_ids = list(dict.fromkeys(payload.job_ids))
        job_data = []
        for job_id in unique_ids:
            record = jobs.by_id.get(job_id)
            if record is None:
                raise APIError(404, "job_not_found", "One of the picked jobs is not present in the requested snapshot.")
            source = record.source
            job_data.append({
                "job_id": job_id,
                "title": source.get("title_searchable") or source.get("title_raw"),
                "description_text": source.get("description_text"),
            })
        candidate_data = _candidate_mapping(profile_id, json.loads(profile_row["profile_json"]))
        content_hash = hashlib.sha256(json.dumps(
            {"kind": "compass_batch", "candidate": candidate_data, "jobs": job_data,
             "addendum": hashlib.sha256(grounded._BATCH_PROMPT_ADDENDUM.encode("utf-8")).hexdigest(), "max_chars": grounded.MAX_DESCRIPTION_CHARS},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        selected_provider = app.state.assessment_provider or DisabledAssessmentProvider()
        model_id = str(getattr(selected_provider, "model", type(selected_provider).__name__))
        cache_key = _assessment_cache_key(session_hash, profile_id, profile_row["version"], "compass_batch", payload.snapshot_id, content_hash, model_id)
        cached = store.get_assessment(cache_key, session_hash)
        if cached is not None:
            return {"assessment_state": "completed", "cache": "hit", **cached}
        unavailable = {"assessment_state": "unavailable", "reason": "provider_disabled", "cache": "miss", "jobs": {}, "candidate_evidence": [], "failed": 0}
        try:
            result = grounded.explain_matches(candidate_data, job_data, provider=selected_provider)
        except grounded.ProviderUnavailable:
            return unavailable
        except grounded.ProviderFailure:
            if app.state.assessment_provider is None:
                return unavailable
            raise APIError(502, "provider_failed", "The AI service could not be reached or failed. Nothing was kept; try again.", retryable=True) from None
        except grounded.ResponseValidationError:
            raise APIError(502, "assessment_validation_failed", "The AI's answer was not in the expected form. Nothing was kept; try again.", retryable=True) from None
        wrapped = {"profile_id": profile_id, "profile_version": profile_row["version"], "snapshot_id": payload.snapshot_id, **result}
        if result["failed"] == 0:  # a partly failed run is not cached, so running it again retries the failed jobs
            store.cache_assessment(cache_key, session_hash, wrapped)
        return {"assessment_state": "completed", "cache": "miss", **wrapped}

    @app.get("/api/v1/feedback")
    def list_feedback(
        snapshot_id: str | None = None,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        items = []
        for row in store.list_feedback(session_hash, snapshot_id):
            job = jobs.by_id.get(row["job_id"])
            items.append({
                "job_id": row["job_id"], "snapshot_id": row["snapshot_id"], "signal": row["signal"],
                "created_at": row["created_at"],
                "job": _job_summary(JobRow(job.job_id, job.source), row["snapshot_id"], jobs.snapshot_date) if job else None,
            })
        return {"items": items}

    @app.put("/api/v1/feedback")
    def put_feedback(
        payload: FeedbackPut,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        jobs.get(payload.job_id, payload.snapshot_id)
        row = store.set_feedback(session_hash, payload.snapshot_id, payload.job_id, payload.signal)
        return {"job_id": row["job_id"], "snapshot_id": row["snapshot_id"], "signal": row["signal"], "created_at": row["created_at"]}

    @app.delete("/api/v1/feedback/{snapshot_id}/{job_id}", status_code=204)
    def remove_feedback(
        snapshot_id: str,
        job_id: str,
        session_hash: str = Depends(session_dependency),
    ) -> Response:
        store.delete_feedback(session_hash, snapshot_id, job_id)
        return Response(status_code=204)

    @app.post("/api/v1/profiles/{profile_id}/jobs/{job_id}/assessment")
    def assess_job(
        profile_id: str,
        job_id: str,
        payload: AssessmentRequest,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        row = jobs.get(job_id, payload.snapshot_id)
        profile_row = store.get_profile(session_hash)
        if profile_row is None or profile_row["profile_id"] != profile_id:
            raise APIError(404, "profile_not_found", "No matching candidate profile exists in this demo session.")
        if profile_row["version"] != payload.profile_version:
            raise APIError(409, "profile_version_conflict", "Reload the current profile before requesting an assessment.")
        profile_data = json.loads(profile_row["profile_json"])
        candidate_data = _candidate_mapping(profile_id, profile_data)
        source = row.source
        job_data = {
            "job_id": job_id,
            "title": source.get("title_searchable") or source.get("title_raw"),
            "description_text": source.get("description_text"),
        }
        content_hash = hashlib.sha256(
            json.dumps({"candidate": candidate_data, "job": job_data, "include": payload.include}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        selected_provider = app.state.assessment_provider or DisabledAssessmentProvider()
        model_id = str(getattr(selected_provider, "model", type(selected_provider).__name__))
        cache_key = _assessment_cache_key(
            session_hash, profile_id, profile_row["version"], job_id, payload.snapshot_id,
            content_hash, model_id,
        )
        cached = store.get_assessment(cache_key, session_hash)
        if cached is not None:
            return {"assessment_state": "completed", "cache": "hit", **cached}
        try:
            result = grounded.explain_match(candidate_data, job_data, provider=selected_provider)
        except grounded.ProviderUnavailable:
            return {
                "assessment_state": "unavailable",
                "reason": "provider_disabled",
                "profile_id": profile_id,
                "profile_version": profile_row["version"],
                "job_id": job_id,
                "snapshot_id": payload.snapshot_id,
                "requested_analyses": payload.include,
                "schema_version": grounded.SCHEMA_VERSION,
                "assessment": None,
                "cache": "miss",
            }
        except grounded.ProviderFailure:
            if app.state.assessment_provider is None:
                return {
                    "assessment_state": "unavailable",
                    "reason": "provider_disabled",
                    "profile_id": profile_id,
                    "profile_version": profile_row["version"],
                    "job_id": job_id,
                    "snapshot_id": payload.snapshot_id,
                    "requested_analyses": payload.include,
                    "schema_version": grounded.SCHEMA_VERSION,
                    "assessment": None,
                    "cache": "miss",
                }
            raise APIError(502, "provider_invalid_response", "The assessment provider failed; no result was cached.", retryable=True) from None
        except grounded.ResponseValidationError:
            raise APIError(502, "assessment_validation_failed", "The assessment did not pass evidence validation; no result was cached.", retryable=True) from None
        wrapped = {
            "profile_id": profile_id,
            "profile_version": profile_row["version"],
            "job_id": job_id,
            "snapshot_id": payload.snapshot_id,
            "requested_analyses": payload.include,
            "schema_version": grounded.SCHEMA_VERSION,
            "assessment": result,
        }
        store.cache_assessment(cache_key, session_hash, wrapped)
        return {"assessment_state": "completed", "cache": "miss", **wrapped}

    @app.get("/api/v1/saved-jobs")
    def list_saved_jobs(
        snapshot_id: str | None = None,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        rows = store.list_saved_jobs(session_hash, snapshot_id)
        items = []
        for row in rows:
            payload = _saved_payload(row)
            job = jobs.by_id.get(row["job_id"])
            payload["job"] = _job_summary(JobRow(job.job_id, job.source), row["snapshot_id"], jobs.snapshot_date) if job else None
            items.append(payload)
        return {"items": items}

    @app.post("/api/v1/saved-jobs", status_code=201)
    def create_saved_job(
        payload: SavedJobCreate,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        jobs: JobRepository = app.state.job_repository
        jobs.get(payload.job_id, payload.snapshot_id)
        return _saved_payload(store.save_job(session_hash, payload.snapshot_id, payload.job_id))

    @app.patch("/api/v1/saved-jobs/{snapshot_id}/{job_id}")
    def update_saved_job(
        snapshot_id: str,
        job_id: str,
        payload: SavedJobPatch,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        patch = payload.model_dump(exclude_unset=True)
        if any(value is None for value in patch.values()):
            raise APIError(422, "validation_error", "Stage and note cannot be null.")
        return _saved_payload(store.patch_saved_job(session_hash, snapshot_id, job_id, patch))

    @app.delete("/api/v1/saved-jobs/{snapshot_id}/{job_id}", status_code=204)
    def remove_saved_job(
        snapshot_id: str,
        job_id: str,
        session_hash: str = Depends(session_dependency),
    ) -> Response:
        store.delete_saved_job(session_hash, snapshot_id, job_id)
        return Response(status_code=204)

    @app.post("/api/v1/profile-extraction")
    def extract_profile_from_resume(
        payload: ResumeExtractionRequest,
        session_hash: str = Depends(session_dependency),
    ) -> dict[str, Any]:
        """Fill-in suggestions from resume text. Stateless: the text is not stored or logged, and nothing is saved."""
        extraction_provider = app.state.extraction_provider
        if extraction_provider is None:
            return {"state": "unavailable", "reason": "provider_disabled", "profile": None}
        try:
            if payload.resume_pdf_base64 is not None:
                try:
                    pdf_bytes = base64.b64decode(payload.resume_pdf_base64, validate=True)
                except (binascii.Error, ValueError):
                    raise APIError(422, "resume_not_readable", "The PDF upload was not valid.") from None
                resume_text = resume_extraction.pdf_to_text(pdf_bytes)
            else:
                resume_text = payload.resume_text or ""
            result = resume_extraction.extract_profile(resume_text, extraction_provider)
        except grounded.ProviderUnavailable:
            return {"state": "unavailable", "reason": "provider_disabled", "profile": None}
        except grounded.ProviderFailure:
            raise APIError(502, "provider_failed", "The resume reader could not be reached or failed. Nothing was filled.", retryable=True) from None
        except grounded.ResponseValidationError as exc:
            raise APIError(422, "resume_not_readable", str(exc)) from None
        return {"state": "completed", "model": str(getattr(extraction_provider, "model", type(extraction_provider).__name__)), **result}

    return app


app = create_app()