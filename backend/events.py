"""Opt-in research event log: what was shown to a person and how they reacted.

Demo version: one small SQLite file, append-only. Everyone on the demo team has agreed to share their ratings outside the app, so
by default anyone who sends the anonymous participant id is logged. For production, start with require_consent=True
(PREDICTLEADS_EVENTS_REQUIRE_CONSENT=1): then nothing is written until the person opts in through /api/v1/research/consent. It holds no names,
resume text or profile text: a participant is a random id, a profile is a hash plus the ranker's embedding of it, and everything
else is job ids, ranks and reactions. See docs/feedback-data.md for the schema and the path to a production system.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PARTICIPANT_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")
EVENT_TYPES = frozenset({"impression", "rating", "rating_cleared", "save", "unsave", "compass_run"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY,
    consented_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    type TEXT NOT NULL,
    snapshot_id TEXT,
    job_id TEXT,
    profile_hash TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_participant_type ON events(participant_id, type, event_id);
CREATE INDEX IF NOT EXISTS events_job ON events(participant_id, job_id);
CREATE TABLE IF NOT EXISTS profiles (
    participant_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    vector BLOB,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (participant_id, profile_hash)
);
"""


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EventStore:
    def __init__(self, path: Path, require_consent: bool = False):
        self.path = Path(path)
        self.require_consent = require_consent
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            db = sqlite3.connect(self.path, timeout=10)
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            finally:
                db.close()

    # ---- consent -------------------------------------------------------------------------------------------------

    def is_consented(self, participant_id: str | None) -> bool:
        """True when this participant may be logged. In demo mode (consent not required) any valid id is registered on first sight."""
        if not participant_id or not PARTICIPANT_RE.match(participant_id):
            return False
        with self.connect() as db:
            known = db.execute("SELECT 1 FROM participants WHERE participant_id=?", (participant_id,)).fetchone() is not None
            if not known and not self.require_consent:
                db.execute("INSERT INTO participants(participant_id, consented_at) VALUES(?,?)", (participant_id, utc_text()))
                return True
            return known

    def consent(self, participant_id: str) -> None:
        if not PARTICIPANT_RE.match(participant_id):
            raise ValueError("participant id is not valid")
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO participants(participant_id, consented_at) VALUES(?,?)", (participant_id, utc_text()))

    def withdraw(self, participant_id: str) -> int:
        """Stop logging and delete everything held for this participant. Returns the number of events removed."""
        if not PARTICIPANT_RE.match(participant_id):
            return 0
        with self.connect() as db:
            removed = db.execute("SELECT COUNT(*) FROM events WHERE participant_id=?", (participant_id,)).fetchone()[0]
            db.execute("DELETE FROM events WHERE participant_id=?", (participant_id,))
            db.execute("DELETE FROM profiles WHERE participant_id=?", (participant_id,))
            db.execute("DELETE FROM participants WHERE participant_id=?", (participant_id,))
        return int(removed)

    def count(self, participant_id: str) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM events WHERE participant_id=?", (participant_id,)).fetchone()[0])

    # ---- writing -------------------------------------------------------------------------------------------------

    def log(
        self, participant_id: str | None, event_type: str, *, snapshot_id: str | None = None, job_id: str | None = None,
        profile_hash: str | None = None, payload: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append one event; does nothing (returns False) unless the participant may be logged (see is_consented)."""
        if event_type not in EVENT_TYPES or not self.is_consented(participant_id):
            return False
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(ts, participant_id, type, snapshot_id, job_id, profile_hash, payload) VALUES(?,?,?,?,?,?,?)",
                (utc_text(), participant_id, event_type, snapshot_id, job_id, profile_hash, json.dumps(payload or {}, separators=(",", ":"))),
            )
        return True

    def remember_profile(self, participant_id: str, profile_hash: str, vector: Sequence[float] | None) -> None:
        """Keep the ranker's embedding of a profile (not its text), once per profile version."""
        blob = np.asarray(vector, dtype=np.float16).tobytes() if vector is not None else None
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO profiles(participant_id, profile_hash, vector, first_seen) VALUES(?,?,?,?)",
                       (participant_id, profile_hash, blob, utc_text()))

    def has_profile(self, participant_id: str, profile_hash: str) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM profiles WHERE participant_id=? AND profile_hash=?", (participant_id, profile_hash)).fetchone() is not None

    def last_exposure(self, participant_id: str, job_id: str) -> dict[str, Any] | None:
        """Where the job sat the last time it was shown to this participant (rank and the impression it came from)."""
        with self.connect() as db:
            rows = db.execute(
                "SELECT event_id, payload FROM events WHERE participant_id=? AND type='impression' ORDER BY event_id DESC LIMIT 60",
                (participant_id,),
            ).fetchall()
        for row in rows:
            for entry in json.loads(row["payload"]).get("items", []):
                if entry[0] == job_id:
                    found: dict[str, Any] = {"impression_id": int(row["event_id"]), "rank": int(entry[1])}
                    if len(entry) > 3 and entry[3]:
                        found["source"] = entry[3]  # keyword / embedding / random: an exploration listing
                    return found
        return None

    # ---- reading (for the report) --------------------------------------------------------------------------------

    def events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            if event_type:
                rows = db.execute("SELECT * FROM events WHERE type=? ORDER BY event_id", (event_type,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM events ORDER BY event_id").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def participants(self) -> list[str]:
        with self.connect() as db:
            return [r["participant_id"] for r in db.execute("SELECT participant_id FROM participants ORDER BY consented_at").fetchall()]
