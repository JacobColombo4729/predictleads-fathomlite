"""Full-stream job-role counts plus a uniform, disk-backed reservoir sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import boto3
import botocore
import zstandard as zstd

from prepare_jobs import BUCKET, KEY, SNAPSHOT_DATE, prepare_resource, s3_session

SAMPLE_SIZE = 30_000
MAX_LINE_BYTES = 4 * 1024 * 1024
PROGRESS_SECONDS = 30
COMMIT_RECORDS = 5_000
DISK_RESERVE_BYTES = 2 * 1024**3
REPO_ROOT = Path(__file__).resolve().parents[2]  # data/, models/ and .env are shared with analysis/ at the top
OUTPUT_ROOT = REPO_ROOT / "data" / "discovery" / "role_exploration"
STRING_FIELDS = {
    "raw_title", "normalized_title", "onet_occupation", "onet_family", "status", "seniority",
    "language", "categories", "contract_types", "description_availability", "salary_availability",
    "location_availability",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _state(value: Any, exists: bool = True) -> str:
    if not exists:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "blank"
    return "present"


def _field_state(field: str, value: Any, exists: bool = True) -> str:
    state = _state(value, exists)
    if state == "present" and field in STRING_FIELDS and not isinstance(value, str):
        return "unexpected_type"
    return state


def _type(value: Any) -> str:
    if value is None:
        return "NoneType"
    if isinstance(value, bool):
        return "bool"
    return type(value).__name__


def _pack_label(value_json: str) -> bytes:
    raw = value_json.encode("utf-8")
    if len(raw) > 128:
        return b"\x01" + zlib.compress(raw, level=3)
    return b"\x00" + raw


def _unpack_label(payload: Any) -> Any:
    if payload is None or payload == "":
        return None
    if isinstance(payload, str):
        return json.loads(payload)
    raw = bytes(payload)
    if not raw:
        return None
    if raw[0] == 1:
        raw = zlib.decompress(raw[1:])
    else:
        raw = raw[1:]
    return json.loads(raw.decode("utf-8"))


def reservoir_slot(population_index: int, sample_size: int, rng: random.Random) -> int | None:
    """Return a zero-based reservoir slot for a one-based population index."""
    if population_index <= sample_size:
        return population_index - 1
    candidate = rng.randrange(population_index)
    return candidate if candidate < sample_size else None


def _schema(connection: sqlite3.Connection, seed: int, sample_size: int) -> None:
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        CREATE TABLE run_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE value_counts(
            scope TEXT NOT NULL, field TEXT NOT NULL, state TEXT NOT NULL,
            value_hash TEXT NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY(scope,field,state,value_hash)
        ) WITHOUT ROWID;
        CREATE TABLE value_dictionary(
            value_hash TEXT PRIMARY KEY, value_json TEXT NOT NULL, value_type TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE title_occupation_pairs(
            scope TEXT NOT NULL, title_state TEXT NOT NULL, title_hash TEXT NOT NULL,
            occupation_state TEXT NOT NULL, occupation_hash TEXT NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY(scope,title_state,title_hash,occupation_state,occupation_hash)
        ) WITHOUT ROWID;
        CREATE TABLE sample_slots(
            slot INTEGER PRIMARY KEY, population_index INTEGER NOT NULL,
            source_line INTEGER NOT NULL, resource_index INTEGER NOT NULL,
            job_id TEXT, job_type TEXT, prepared_json TEXT NOT NULL
        );
    """)
    connection.executemany(
        "INSERT INTO run_metadata(key,value) VALUES (?,?)",
        [("seed", str(seed)), ("sample_size_target", str(sample_size)),
         ("population_processed", "0"), ("status", "initialized")],
    )


def _upsert_value(db: sqlite3.Connection, scope: str, field: str, value: Any, exists: bool) -> None:
    state = _field_state(field, value, exists)
    value_type = _type(value) if state in ("present", "unexpected_type") else ""
    value_json = _json(value) if state in ("present", "unexpected_type") else ""
    value_hash = ""
    if state in ("present", "unexpected_type"):
        value_hash = hashlib.sha256((value_type + "\0" + value_json).encode("utf-8")).hexdigest()
        db.execute("INSERT OR IGNORE INTO value_dictionary(value_hash,value_json,value_type) VALUES (?,?,?)",
               (value_hash, _pack_label(value_json), value_type))
    db.execute("""INSERT INTO value_counts VALUES (?,?,?,?,1)
        ON CONFLICT(scope,field,state,value_hash) DO UPDATE SET count=count+1""",
        (scope, field, state, value_hash))


def _upsert_shape(db: sqlite3.Connection, scope: str, field: str, value: Any, exists: bool,
                  expected_type: type = dict) -> None:
    """Count a value's state/type without storing its high-cardinality contents."""
    state = _state(value, exists)
    type_name = _type(value) if state == "present" else ""
    if state == "present":
        state = "present" if isinstance(value, expected_type) else "unexpected_type"
    shape_key = state + "\0" + type_name
    value_hash = hashlib.sha256(shape_key.encode("utf-8")).hexdigest() if state != "missing" else ""
    db.execute("INSERT OR IGNORE INTO value_dictionary(value_hash,value_json,value_type) VALUES (?,?,?)",
               (value_hash, "", type_name))
    db.execute("""INSERT INTO value_counts VALUES (?,?,?,?,1)
        ON CONFLICT(scope,field,state,value_hash) DO UPDATE SET count=count+1""",
        (scope, field, state, value_hash))


def _hashed_pair_value(db: sqlite3.Connection, value: Any, state: str) -> str:
    if state not in ("present", "unexpected_type"):
        return ""
    value_type = _type(value)
    value_json = _json(value)
    value_hash = hashlib.sha256((value_type + "\0" + value_json).encode("utf-8")).hexdigest()
    db.execute("INSERT OR IGNORE INTO value_dictionary(value_hash,value_json,value_type) VALUES (?,?,?)",
               (value_hash, _pack_label(value_json), value_type))
    return value_hash


class _CountBuffer:
    """Bounded Python-side batch; exact labels and aggregate rows are persisted in SQLite."""

    def __init__(self, db: sqlite3.Connection, max_rows: int = 50_000):
        self.db = db
        self.max_rows = max_rows
        self.dictionary: dict[str, tuple[bytes, str]] = {}
        self.values: Counter[tuple[str, str, str, str]] = Counter()
        self.pairs: Counter[tuple[str, str, str, str, str]] = Counter()

    def add_value(self, scope: str, field: str, value: Any, exists: bool, *, expected: type | None = None) -> None:
        state = _field_state(field, value, exists) if field in STRING_FIELDS else _state(value, exists)
        if expected is not None and state == "present" and not isinstance(value, expected):
            state = "unexpected_type"
        value_type = _type(value) if state in ("present", "unexpected_type") else ""
        value_json = _json(value) if state in ("present", "unexpected_type") else ""
        value_hash = ""
        if state in ("present", "unexpected_type"):
            value_hash = hashlib.sha256((value_type + "\0" + value_json).encode("utf-8")).hexdigest()
            self.dictionary.setdefault(value_hash, (_pack_label(value_json), value_type))
        self.values[(scope, field, state, value_hash)] += 1

    def add_shape(self, scope: str, field: str, value: Any, exists: bool, expected_type: type) -> None:
        state = _state(value, exists)
        value_type = _type(value) if state == "present" else ""
        if state == "present" and not isinstance(value, expected_type):
            state = "unexpected_type"
        value_hash = hashlib.sha256((state + "\0" + value_type).encode("utf-8")).hexdigest() if state != "missing" else ""
        self.dictionary.setdefault(value_hash, (b"", value_type))
        self.values[(scope, field, state, value_hash)] += 1

    def _pair_component(self, value: Any, state: str) -> str:
        if state not in ("present", "unexpected_type"):
            return ""
        value_type = _type(value)
        value_json = _json(value)
        value_hash = hashlib.sha256((value_type + "\0" + value_json).encode("utf-8")).hexdigest()
        self.dictionary.setdefault(value_hash, (_pack_label(value_json), value_type))
        return value_hash

    def add_pair(self, scope: str, title: Any, title_state: str, occupation: Any, occupation_state: str) -> None:
        self.pairs[(scope, title_state, self._pair_component(title, title_state),
                    occupation_state, self._pair_component(occupation, occupation_state))] += 1

    def flush(self) -> None:
        if self.dictionary:
            self.db.executemany("INSERT OR IGNORE INTO value_dictionary(value_hash,value_json,value_type) VALUES (?,?,?)",
                                [(key, payload, value_type) for key, (payload, value_type) in self.dictionary.items()])
        if self.values:
            self.db.executemany("""INSERT INTO value_counts(scope,field,state,value_hash,count) VALUES (?,?,?,?,?)
                ON CONFLICT(scope,field,state,value_hash) DO UPDATE SET count=count+excluded.count""",
                [(*key, count) for key, count in self.values.items()])
        if self.pairs:
            self.db.executemany("""INSERT INTO title_occupation_pairs(scope,title_state,title_hash,occupation_state,occupation_hash,count)
                VALUES (?,?,?,?,?,?) ON CONFLICT(scope,title_state,title_hash,occupation_state,occupation_hash)
                DO UPDATE SET count=count+excluded.count""",
                [(*key, count) for key, count in self.pairs.items()])
        self.dictionary.clear()
        self.values.clear()
        self.pairs.clear()

    def population_record(self, resource: dict[str, Any]) -> None:
        found, attrs = _values(resource)
        for field, (value, exists) in found.items():
            if field == "onet_data":
                self.add_shape("population", field, value, exists, dict)
            else:
                self.add_value("population", field, value, exists)
        for field in ("status", "seniority", "language"):
            exists = field in attrs
            value = attrs.get(field)
            self.add_value("population", field, value, exists)
        for field in ("categories", "contract_types"):
            exists = field in attrs
            value = attrs.get(field)
            self.add_shape("population", f"{field}_field", value, exists, list)
            if isinstance(value, list):
                for item in value:
                    self.add_value("population", field, item, True)
            elif exists:
                self.add_value("population", field, value, True)
        for field in ("description", "salary", "location"):
            self.add_shape("population", f"{field}_availability", attrs.get(field), field in attrs, str)
        title, title_exists = found["normalized_title"]
        occupation, occupation_exists = found["onet_occupation"]
        self.add_pair("population", title, _field_state("normalized_title", title, title_exists),
                      occupation, _field_state("onet_occupation", occupation, occupation_exists))

    def sample_record(self, prepared: dict[str, Any]) -> None:
        flags = prepared.get("quality_flags", {})
        for field in ("status", "seniority", "language"):
            key = "status_raw" if field == "status" else field
            self.add_value("sample", field, prepared.get(key), flags.get(field, "missing") != "missing")
        for field, key, source_field in (("raw_title", "title_raw", "title"),
                                         ("normalized_title", "title_normalized", "normalized_title")):
            self.add_value("sample", field, prepared.get(key), flags.get(source_field, "missing") != "missing")
        self.add_shape("sample", "description_availability", prepared.get("description_raw"),
                       flags.get("description", "missing") != "missing", str)
        onet = prepared.get("onet_data")
        onet = onet if isinstance(onet, dict) else {}
        for field, key in (("onet_occupation", "occupation_name"), ("onet_family", "family")):
            self.add_value("sample", field, onet.get(key), key in onet)
        title, occupation = prepared.get("title_normalized"), onet.get("occupation_name")
        self.add_pair("sample", title, _field_state("normalized_title", title,
                                                      flags.get("normalized_title", "missing") != "missing"),
                      occupation, _field_state("onet_occupation", occupation, "occupation_name" in onet))


def _count_sample_slots(db: sqlite3.Connection, max_rows: int = 50_000) -> None:
    batch = _CountBuffer(db, max_rows)
    for (prepared_json,) in db.execute("SELECT prepared_json FROM sample_slots ORDER BY slot"):
        batch.sample_record(json.loads(prepared_json))
        if len(batch.values) + len(batch.pairs) >= max_rows:
            batch.flush()
    batch.flush()


def _values(resource: dict[str, Any]) -> tuple[dict[str, tuple[Any, bool]], dict[str, Any]]:
    attrs = resource.get("attributes")
    attrs_ok = isinstance(attrs, dict)
    attrs = attrs if attrs_ok else {}
    onet_value = attrs.get("onet_data")
    onet = onet_value
    onet_ok = isinstance(onet, dict)
    onet = onet if onet_ok else {}
    found = {
        "raw_title": (attrs.get("title"), attrs_ok and "title" in attrs),
        "normalized_title": (attrs.get("normalized_title"), attrs_ok and "normalized_title" in attrs),
        "onet_data": (onet_value, attrs_ok and "onet_data" in attrs),
        "onet_occupation": (onet.get("occupation_name"), onet_ok and "occupation_name" in onet),
        "onet_family": (onet.get("family"), onet_ok and "family" in onet),
    }
    return found, attrs


def _count_record(db: sqlite3.Connection, resource: dict[str, Any]) -> None:
    found, attrs = _values(resource)
    for field, (value, exists) in found.items():
        if field == "onet_data":
            _upsert_shape(db, "population", field, value, exists)
        else:
            _upsert_value(db, "population", field, value, exists)
    for field in ("status", "seniority", "language"):
        exists = field in attrs
        value = attrs.get(field)
        _upsert_value(db, "population", field, value, exists)
    for field in ("categories", "contract_types"):
        exists = field in attrs
        value = attrs.get(field)
        _upsert_shape(db, "population", f"{field}_field", value, exists, expected_type=list)
        if isinstance(value, list):
            for item in value:
                _upsert_value(db, "population", field, item, True)
        elif exists:
            _upsert_value(db, "population", field, value, True)
    for field in ("description", "salary", "location"):
        _upsert_shape(db, "population", f"{field}_availability", attrs.get(field), field in attrs, expected_type=str)
    title, title_exists = found["normalized_title"]
    occupation, occupation_exists = found["onet_occupation"]
    title_state = _field_state("normalized_title", title, title_exists)
    occupation_state = _field_state("onet_occupation", occupation, occupation_exists)
    title_hash = _hashed_pair_value(db, title, title_state)
    occupation_hash = _hashed_pair_value(db, occupation, occupation_state)
    db.execute("""INSERT INTO title_occupation_pairs VALUES ('population',?,?,?,?,1)
        ON CONFLICT(scope,title_state,title_hash,occupation_state,occupation_hash) DO UPDATE SET count=count+1""",
        (title_state, title_hash, occupation_state, occupation_hash))


def _count_sample(db: sqlite3.Connection, prepared: dict[str, Any]) -> None:
    flags = prepared.get("quality_flags", {})
    for field in ("status", "seniority", "language"):
        key = "status_raw" if field == "status" else field
        _upsert_value(db, "sample", field, prepared.get(key), flags.get(field, "missing") != "missing")
    for field, key in (("raw_title", "title_raw"), ("normalized_title", "title_normalized"),
                       ("description_availability", "description_raw")):
        source_field = "title" if field == "raw_title" else "description" if field == "description_availability" else field
        if field == "description_availability":
            _upsert_shape(db, "sample", field, prepared.get(key), flags.get(source_field, "missing") != "missing", expected_type=str)
        else:
            _upsert_value(db, "sample", field, prepared.get(key), flags.get(source_field, "missing") != "missing")
    onet = prepared.get("onet_data")
    onet = onet if isinstance(onet, dict) else {}
    for field, key in (("onet_occupation", "occupation_name"), ("onet_family", "family")):
        _upsert_value(db, "sample", field, onet.get(key), key in onet)
    title, occupation = prepared.get("title_normalized"), onet.get("occupation_name")
    ts = _field_state("normalized_title", title, flags.get("normalized_title", "missing") != "missing")
    os_ = _field_state("onet_occupation", occupation, "occupation_name" in onet)
    title_hash = _hashed_pair_value(db, title, ts)
    occupation_hash = _hashed_pair_value(db, occupation, os_)
    db.execute("""INSERT INTO title_occupation_pairs VALUES ('sample',?,?,?,?,1)
        ON CONFLICT(scope,title_state,title_hash,occupation_state,occupation_hash) DO UPDATE SET count=count+1""",
        (ts, title_hash, os_, occupation_hash))


def _consider_sample(db: sqlite3.Connection, rng: random.Random, population_index: int,
                     sample_size: int, resource: dict[str, Any], included: Any,
                     source_line: int, resource_index: int) -> bool:
    slot = reservoir_slot(population_index, sample_size, rng)
    if slot is None:
        return False
    prepared = prepare_resource(resource, included, source_line=source_line, source_key=KEY)
    prepared["resource_index"] = resource_index
    db.execute(
        """INSERT INTO sample_slots(slot,population_index,source_line,resource_index,job_id,job_type,prepared_json)
           VALUES (?,?,?,?,?,?,?) ON CONFLICT(slot) DO UPDATE SET
           population_index=excluded.population_index,source_line=excluded.source_line,
           resource_index=excluded.resource_index,job_id=excluded.job_id,job_type=excluded.job_type,
           prepared_json=excluded.prepared_json""",
        (slot, population_index, source_line, resource_index,
         None if resource.get("id") is None else str(resource.get("id")),
         None if resource.get("type") is None else str(resource.get("type")), _json(prepared)),
    )
    return True


def _export_sample(db: sqlite3.Connection, path: Path) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for (record_json,) in db.execute("SELECT prepared_json FROM sample_slots ORDER BY slot"):
            output.write(record_json + "\n")
            count += 1
    return count


def _write_meta(db: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    db.executemany("INSERT INTO run_metadata(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   [(k, _json(v)) for k, v in metadata.items()])


def _export_counts(db: sqlite3.Connection, field: str, path: Path) -> int:
    count = 0
    with path.open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("scope", "field", "state", "value", "value_type", "record_frequency"))
        for row in db.execute("""SELECT c.scope,c.field,c.state,d.value_json,d.value_type,c.count
                                 FROM value_counts c LEFT JOIN value_dictionary d ON d.value_hash=c.value_hash
                                 WHERE c.field=? ORDER BY c.scope,c.count DESC,d.value_json""", (field,)):
            writer.writerow((row[0], row[1], row[2], _unpack_label(row[3]) if row[3] else "", row[4] or "", row[5]))
            count += 1
    return count


def _export_pairs(db: sqlite3.Connection, path: Path) -> int:
    count = 0
    with path.open("x", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("scope", "normalized_title_state", "normalized_title", "occupation_state", "occupation_name", "record_frequency"))
        for row in db.execute("""SELECT p.scope,p.title_state,td.value_json,p.occupation_state,od.value_json,p.count
                                 FROM title_occupation_pairs p
                                 LEFT JOIN value_dictionary td ON td.value_hash=p.title_hash
                                 LEFT JOIN value_dictionary od ON od.value_hash=p.occupation_hash
                                 ORDER BY p.scope,p.count DESC,td.value_json,od.value_json"""):
            writer.writerow((row[0], row[1], _unpack_label(row[2]) if row[2] else "", row[3], _unpack_label(row[4]) if row[4] else "", row[5]))
            count += 1
    return count


class _CountingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.bytes_read += len(chunk)
        return chunk


def _available_memory() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                        ("avail_extended", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        return int(status.avail_phys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
    except Exception:
        return None


def _local_source(workspace: Path) -> Path | None:
    candidates = (workspace / KEY, workspace / "data" / Path(KEY).name,
                  workspace / "data" / "raw" / Path(KEY).name)
    return next((path for path in candidates if path.is_file()), None)


def _distribution(db: sqlite3.Connection, scope: str, field: str, limit: int = 12) -> list[dict[str, Any]]:
    rows = db.execute(
        """SELECT c.state,d.value_json,d.value_type,c.count
           FROM value_counts c LEFT JOIN value_dictionary d ON d.value_hash=c.value_hash
           WHERE c.scope=? AND c.field=? ORDER BY c.count DESC,d.value_json LIMIT ?""",
        (scope, field, limit),
    )
    return [{"state": state, "value": _unpack_label(value) if value else None,
             "value_type": value_type, "record_frequency": count}
            for state, value, value_type, count in rows]


def _write_report(path: Path, db: sqlite3.Connection, summary: dict[str, Any], csv_rows: dict[str, int]) -> None:
    dims = ("onet_occupation", "onet_family", "seniority", "language", "description_availability", "status")
    lines = [
        "# Full-File Job Role Exploration",
        "",
        "This report describes record frequencies in the source snapshot, not verified unique jobs or current openings. The reservoir is uniform over eligible primary job records processed when EOF is reached; it is not filtered by title, occupation, seniority, description, or status. Rare roles can still be absent by chance.",
        "",
        f"- Source: `{summary['source']}` (snapshot `{SNAPSHOT_DATE}`).",
        f"- Stop: `{summary['stop_reason']}`; complete file scanned: `{summary['complete_file_scanned']}`.",
        f"- Wrappers/lines: {summary['wrappers_or_lines_inspected']:,}; eligible primary jobs: {summary['population_processed']:,}; sample: {summary['sample_size_actual']:,}/{summary['sample_size_target']:,}; included resources inspected: {summary['included_resources_inspected']:,}.",
        f"- Compressed bytes read: {summary.get('compressed_bytes_read'):,}; object size: {summary.get('source_metadata', {}).get('compressed_size_bytes')}; elapsed: {summary['elapsed_seconds']:.1f}s.",
        f"- Errors: `{json.dumps(summary['errors'], ensure_ascii=False)}`; skipped records: `{json.dumps(summary['skipped'], ensure_ascii=False)}`.",
        "- Counts are SQLite-backed record frequencies. Null, blank, missing, and present/unexpected types remain distinct; CSVs preserve the source labels.",
        "",
        "## Top observed distributions",
    ]
    mapping = (("Raw titles", "raw_title", "raw_titles.csv"),
               ("Normalized titles", "normalized_title", "normalized_titles.csv"),
               ("O*NET occupations", "onet_occupation", "occupations.csv"),
               ("O*NET families", "onet_family", "families.csv"))
    for label, field, filename in mapping:
        top = _distribution(db, "population", field, 10)
        lines.append(f"- {label} (top record frequencies): `{json.dumps(top, ensure_ascii=False)}`. Full counts: `{filename}` ({csv_rows.get(filename, 0):,} state/value rows).")
    lines.append(f"- Normalized-title/O*NET-occupation pair rows: {csv_rows.get('normalized_title_occupation_pairs.csv', 0):,}; full frequency table is in `normalized_title_occupation_pairs.csv`.")
    lines.extend(["", "## Population/sample comparisons", ""])
    for dim in dims:
        population = _distribution(db, "population", dim, 8)
        sample = _distribution(db, "sample", dim, 8)
        lines.extend([
            f"### {dim}",
            f"- Population top values: `{json.dumps(population, ensure_ascii=False)}`",
            f"- Uniform sample top values: `{json.dumps(sample, ensure_ascii=False)}`",
            "- Differences are expected for finite samples; low-frequency occupations or states can be missing by chance.",
        ])
    lines.extend([
        "",
        "## Limitations",
        "",
        "Only dictionary-valued primary `job_opening` resources enter the population denominator and reservoir; malformed wrappers and non-object/unexpected-type primary items are counted as errors/skips. The sample keeps complete prepared records, including descriptions, in ignored local data storage. SQLite aggregations and reservoir slots are stored on disk; aggregation uses bounded Python state. Checkpoint metadata is diagnostic only: restart/resume is not implemented.",
        "Same-wrapper company attributes are attached by both resource type and ID. The separate companies file is not joined. Exact duplicate detection is not performed; frequency counts are record frequencies.",
        "",
        "Role families and career-transition suitability are for team discussion. These distributions do not establish candidate fit or hiring probability.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _self_test() -> None:
    import tempfile

    import io

    # Replacement path and reproducibility use the same seeded population order.
    def run_reservoir(seed: int) -> tuple[list[int | None], int]:
        rng = random.Random(seed)
        slots: list[int | None] = [None] * 3
        replacements = 0
        for population_index in range(1, 101):
            slot = reservoir_slot(population_index, 3, rng)
            if slot is not None:
                if slots[slot] is not None:
                    replacements += 1
                slots[slot] = population_index
        return slots, replacements
    assert run_reservoir(20261005) == run_reservoir(20261005)
    assert run_reservoir(20261005)[1] > 0

    with sqlite3.connect(":memory:") as db:
        _schema(db, 3, 20261005)
        wrapper = {"data": [
            {"id": "j1", "type": "job_opening", "attributes": {
                "title": "Raw title", "normalized_title": None, "translated_title": "Translated",
                "description": None, "salary": None, "location": None, "status": None,
                "seniority": "junior", "language": "en", "categories": ["operations"],
                "onet_data": {"occupation_name": "Example occupation", "family": "Example family"}},
             "relationships": {"company": {"data": {"type": "company", "id": "c1"}}}},
            {"id": "j2", "type": "job_opening", "attributes": {
                "title": "Two", "normalized_title": "Two", "description": "  ", "status": "closed",
                "categories": [],
                "onet_data": {"occupation_name": "Other occupation", "family": "Other family"}}},
        ], "included": [{"id": "c1", "type": "company", "attributes": {"company_name": "Company"}}]}
        eligible = 0
        rng = random.Random(20261005)
        for idx, resource in enumerate(wrapper["data"], 1):
            _count_record(db, resource)
            eligible += 1
            slot = reservoir_slot(eligible, 3, rng)
            if slot is not None:
                prepared = prepare_resource(resource, wrapper["included"], source_line=1)
                prepared["resource_index"] = idx
                _count_sample(db, prepared)
                db.execute("INSERT INTO sample_slots VALUES (?,?,?,?,?,?,?)",
                           (slot, eligible, 1, idx, resource["id"], resource["type"], _json(prepared)))
        db.commit()
        assert eligible == 2
        assert db.execute("SELECT count(*) FROM sample_slots").fetchone()[0] == 2
        assert db.execute(
            "SELECT c.count FROM value_counts c JOIN value_dictionary d ON d.value_hash=c.value_hash "
            "WHERE c.scope='population' AND c.field='raw_title' AND c.state='present' AND d.value_json=?",
            (_pack_label(_json("Raw title")),),
        ).fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='normalized_title' AND state='null'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='description_availability' AND state='blank'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='categories_field' AND state='present'").fetchone()[0] == 2
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='contract_types_field' AND state='missing'").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM value_counts WHERE scope='population' AND field='categories'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='onet_data' AND state='present'").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM value_dictionary WHERE value_type='dict' AND length(value_json)>2").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM title_occupation_pairs WHERE scope='population'").fetchone()[0] == 2
        rows = [json.loads(row[0]) for row in db.execute("SELECT prepared_json FROM sample_slots ORDER BY slot")]
        assert any(row["company_match_state"] == "matched" for row in rows)
        assert any(row["status_derived"] == "unknown" for row in rows)
        odd = {"id": "odd", "type": "job_opening", "attributes": {
            "title": ["not", "a", "string"], "normalized_title": 7,
            "categories": ["ok", {"unexpected": "object"}],
            "description": "A long but bounded description payload that belongs in the sample only.",
            "onet_data": {"occupation_name": 9, "family": None},
        }}
        _count_record(db, odd)
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='raw_title' AND state='unexpected_type'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='normalized_title' AND state='unexpected_type'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='categories' AND state='unexpected_type'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='onet_occupation' AND state='unexpected_type'").fetchone()[0] == 1
        description_json = _json(odd["attributes"]["description"])
        assert db.execute("SELECT count(*) FROM value_dictionary WHERE value_json=?", (_pack_label(description_json),)).fetchone()[0] == 0
        odd_prepared = prepare_resource(odd, [], source_line=3)
        _count_sample(db, odd_prepared)
        assert db.execute("SELECT count FROM value_counts WHERE scope='sample' AND field='description_availability' AND state='present'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM value_dictionary WHERE value_json=?", (_pack_label(description_json),)).fetchone()[0] == 0
        with tempfile.TemporaryDirectory() as temp:
            title_csv = Path(temp) / "titles.csv"
            pair_csv = Path(temp) / "pairs.csv"
            assert _export_counts(db, "raw_title", title_csv) > 0
            assert _export_pairs(db, pair_csv) == 6
            assert "Raw title" in title_csv.read_text(encoding="utf-8-sig")
            assert "Example occupation" in pair_csv.read_text(encoding="utf-8-sig")

    with sqlite3.connect(":memory:") as db:
        _schema(db, 3, 20261005)
        buffer = _CountBuffer(db)
        buffer.population_record({"attributes": {"categories": [], "contract_types": ["permanent"]}})
        buffer.population_record({"attributes": {}})
        buffer.flush()
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='categories_field' AND state='present'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='categories_field' AND state='missing'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM value_counts WHERE scope='population' AND field='categories' AND state='missing'").fetchone()[0] == 0
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='contract_types_field' AND state='present'").fetchone()[0] == 1
        assert db.execute("SELECT count FROM value_counts WHERE scope='population' AND field='contract_types' AND state='present'").fetchone()[0] == 1

    def run_disk_reservoir(seed: int, output: Path) -> list[str]:
        with sqlite3.connect(":memory:") as sample_db:
            _schema(sample_db, seed, 3)
            sample_rng = random.Random(seed)
            for population_index in range(1, 51):
                resource = {"id": f"job-{population_index}", "type": "job_opening",
                            "attributes": {"title": f"Title {population_index}",
                                           "normalized_title": f"Normalized {population_index}",
                                           "description": None, "status": None,
                                           "onet_data": {"occupation_name": f"Occupation {population_index % 4}",
                                                          "family": f"Family {population_index % 2}"}}}
                _count_record(sample_db, resource)
                _consider_sample(sample_db, sample_rng, population_index, 3, resource, [], population_index, 1)
            sample_db.commit()
            assert sample_db.execute("SELECT count(*) FROM sample_slots").fetchone()[0] == 3
            assert sample_db.execute("SELECT count(*) FROM title_occupation_pairs WHERE scope='population'").fetchone()[0] == 50
            _count_sample_slots(sample_db)
            assert sample_db.execute("SELECT sum(count) FROM value_counts WHERE scope='sample' AND field='status'").fetchone()[0] == 3

            _export_sample(sample_db, output)
            return [json.loads(line)["job_id"] for line in output.read_text(encoding="utf-8").splitlines()]

    with tempfile.TemporaryDirectory() as temporary:
        first = run_disk_reservoir(20261005, Path(temporary) / "first.jsonl")
        second = run_disk_reservoir(20261005, Path(temporary) / "second.jsonl")
        assert first == second and len(first) == 3
        assert any(job_id not in {"job-1", "job-2", "job-3"} for job_id in first)
    print("Offline role exploration checks passed: reservoir replacement/reproducibility, multi-resource wrappers, disk-backed title/occupation counts, null/missing/blank fields and prepared company-linked sample.")


def _local_source(workspace: Path) -> Path | None:
    choices = (workspace / KEY, workspace / "data" / Path(KEY).name, workspace / "data" / "raw" / Path(KEY).name)
    return next((path for path in choices if path.is_file()), None)


def _open_source(workspace: Path) -> tuple[BinaryIO, Any, dict[str, Any], str]:
    local = _local_source(workspace)
    if local:
        return local.open("rb"), None, {"compressed_size_bytes": local.stat().st_size}, str(local)
    client = s3_session().client(
        "s3", config=botocore.config.Config(connect_timeout=10, read_timeout=30,
                                              retries={"max_attempts": 3, "mode": "standard"}),
    )
    try:
        response = client.get_object(Bucket=BUCKET, Key=KEY)
    except Exception:
        client.close()
        raise
    return response["Body"], client, {
        "compressed_size_bytes": response.get("ContentLength"), "etag": response.get("ETag"),
        "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else None,
        "version_id": response.get("VersionId"),
    }, f"s3://{BUCKET}/{KEY}"


def _compare_values(db: sqlite3.Connection, scope: str, field: str, limit: int = 10) -> list[dict[str, Any]]:
    return [{"state": row[0], "value": _unpack_label(row[1]),
             "type": row[2], "count": row[3]}
            for row in db.execute("""SELECT c.state,d.value_json,d.value_type,c.count
                                      FROM value_counts c LEFT JOIN value_dictionary d ON d.value_hash=c.value_hash
                                      WHERE c.scope=? AND c.field=? ORDER BY c.count DESC,d.value_json LIMIT ?""",
                                  (scope, field, limit))]


def _summarize_population(db: sqlite3.Connection) -> dict[str, Any]:
    return {
        dim: {"population_top": _compare_values(db, "population", dim),
              "sample_top": _compare_values(db, "sample", dim)}
        for dim in ("onet_occupation", "onet_family", "seniority", "language", "description_availability", "status")
    }


def _run_exploration(output_dir: Path, sample_size: int, seed: int, storage_estimate: dict[str, Any],
                     minimum_free_during_scan: int) -> tuple[int, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=False)
    db_path = output_dir / "role_counts_and_sample.sqlite3"
    sample_path = output_dir / "jobs_prepared_random_sample.jsonl"
    summary_path = output_dir / "run_summary.json"
    report_path = output_dir / "role_exploration_report.md"
    sectionless_csvs = {
        "raw_titles.csv": "raw_title",
        "normalized_titles.csv": "normalized_title",
        "occupations.csv": "onet_occupation",
        "families.csv": "onet_family",
    }
    pair_csv = output_dir / "normalized_title_occupation_pairs.csv"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA busy_timeout=30000")
    _schema(db, seed, sample_size)
    rng = random.Random(seed)
    count_buffer = _CountBuffer(db)
    started = time.monotonic()
    source: BinaryIO | None = None
    client = None
    reader: _CountingReader | None = None
    source_meta: dict[str, Any] = {}
    source_label = f"s3://{BUCKET}/{KEY}"
    lines = wrappers = population = included_count = 0
    errors: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    pending = bytearray()
    dropping_line = False
    stop_reason = "source_setup"
    eof = False
    last_progress = started
    next_commit_at = COMMIT_RECORDS

    def process_line(line: bytes, line_number: int) -> None:
        nonlocal wrappers, population, included_count, next_commit_at
        try:
            wrapper = json.loads(line.decode("utf-8"))
        except UnicodeDecodeError:
            errors["invalid_utf8"] += 1
            return
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            return
        if not isinstance(wrapper, dict):
            errors["wrapper_not_object"] += 1
            return
        wrappers += 1
        data = wrapper.get("data", "<missing>")
        if not isinstance(data, list):
            errors["data_not_array"] += 1
            return
        included = wrapper.get("included", [])
        if included is None:
            included = []
        elif not isinstance(included, list):
            errors["included_not_array"] += 1
            included = []
        included_count += sum(isinstance(record, dict) for record in included)
        for resource_index, resource in enumerate(data, 1):
            if not isinstance(resource, dict):
                skipped["primary_resource_not_object"] += 1
                continue
            if resource.get("type") != "job_opening":
                skipped["primary_unexpected_type"] += 1
                continue
            population += 1
            count_buffer.population_record(resource)
            _consider_sample(db, rng, population, sample_size, resource, included,
                             source_line=line_number, resource_index=resource_index)
            while population >= next_commit_at:
                count_buffer.flush()
                db.commit()
                _write_meta(db, {"population_processed": population, "wrappers": wrappers,
                                 "included_resources": included_count, "checkpoint_utc": datetime.now(timezone.utc).isoformat(),
                                 "status": "in_progress", "resume_supported": False})
                db.commit()
                next_commit_at += COMMIT_RECORDS

    try:
        source, client, source_meta, source_label = _open_source(REPO_ROOT)
        reader = _CountingReader(source)
        with zstd.ZstdDecompressor().stream_reader(reader, read_across_frames=True, closefd=False) as decompressor:
            while True:
                chunk = decompressor.read(64 * 1024)
                if not chunk:
                    eof = True
                    stop_reason = "eof"
                    if dropping_line:
                        lines += 1
                        errors["oversized_line"] += 1
                    elif pending:
                        lines += 1
                        errors["unterminated_final_line"] += 1
                    break
                remaining = chunk
                while remaining:
                    if dropping_line:
                        newline = remaining.find(b"\n")
                        if newline < 0:
                            remaining = b""
                            continue
                        lines += 1
                        errors["oversized_line"] += 1
                        dropping_line = False
                        remaining = remaining[newline + 1:]
                        continue
                    newline = remaining.find(b"\n")
                    if newline < 0:
                        segment = remaining
                        remaining = b""
                        if len(pending) + len(segment) > MAX_LINE_BYTES:
                            pending.clear()
                            dropping_line = True
                        else:
                            pending.extend(segment)
                        continue
                    segment = remaining[:newline]
                    remaining = remaining[newline + 1:]
                    lines += 1
                    if len(pending) + len(segment) > MAX_LINE_BYTES:
                        pending.clear()
                        errors["oversized_line"] += 1
                        continue
                    line = bytes(pending) + segment
                    pending.clear()
                    process_line(line, lines)

                now = time.monotonic()
                if now - last_progress >= PROGRESS_SECONDS:
                    count_buffer.flush()
                    db.commit()
                    sample_n = db.execute("SELECT count(*) FROM sample_slots").fetchone()[0]
                    disk_free = shutil.disk_usage(output_dir).free
                    print(f"Progress wrappers={wrappers} valid_jobs={population} sample_slots={sample_n} elapsed_seconds={now-started:.1f} available_memory_bytes={_available_memory()} available_disk_bytes={disk_free} sqlite_bytes={db_path.stat().st_size if db_path.exists() else 0}", flush=True)
                    _write_meta(db, {"population_processed": population, "wrappers": wrappers,
                                     "included_resources": included_count, "status": "in_progress",
                                     "checkpoint_utc": datetime.now(timezone.utc).isoformat(),
                                     "resume_supported": False})
                    db.commit()
                    last_progress = now
                    if disk_free < minimum_free_during_scan:
                        stop_reason = "disk_safety_stop"
                        errors["disk_reserve_threshold_reached"] += 1
                        break
                if stop_reason == "disk_safety_stop":
                    break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        errors["interrupted"] += 1
    except Exception as error:
        stop_reason = "error"
        errors[f"{type(error).__name__}:{str(error)[:200]}"] += 1
        print(f"Role exploration error: {type(error).__name__}: {str(error)[:250]}", file=sys.stderr, flush=True)
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as error:
                errors[f"source_close:{type(error).__name__}"] += 1
        if client is not None:
            try:
                client.close()
            except Exception as error:
                errors[f"client_close:{type(error).__name__}"] += 1
        count_buffer.flush()
        db.commit()

    _count_sample_slots(db)
    db.commit()
    _export_sample(db, sample_path)
    csv_rows = {filename: _export_counts(db, field, output_dir / filename)
                for filename, field in sectionless_csvs.items()}
    csv_rows[pair_csv.name] = _export_pairs(db, pair_csv)
    sample_actual = db.execute("SELECT count(*) FROM sample_slots").fetchone()[0]
    summary = {
        "source": source_label,
        "source_key": KEY,
        "source_metadata": source_meta,
        "snapshot_date": SNAPSHOT_DATE,
        "seed": seed,
        "sample_size_target": sample_size,
        "sample_size_actual": sample_actual,
        "population_processed": population,
        "wrappers_or_lines_inspected": lines,
        "valid_wrappers": wrappers,
        "included_resources_inspected": included_count,
        "compressed_bytes_read": reader.bytes_read if reader else 0,
        "stop_reason": stop_reason,
        "complete_file_scanned": bool(eof and stop_reason == "eof"),
        "uninspected_records": 0 if eof and stop_reason == "eof" else "unknown remainder; scan did not reach EOF",
        "errors": dict(errors),
        "skipped": dict(skipped),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sample_scope": "Uniform over valid primary job_opening resource objects processed; includes records missing descriptions or known status. Malformed and skipped resources are excluded.",
        "resume_supported": False,
        "storage_estimate_before_run": storage_estimate,
        "sqlite_path": str(db_path),
        "sqlite_bytes": db_path.stat().st_size,
        "sample_jsonl_bytes": sample_path.stat().st_size,
        "csv_frequency_rows": csv_rows,
        "sample_population_comparisons": _summarize_population(db),
        "limitations": [
            "All title/occupation counts are record frequencies, not verified unique-job counts.",
            "The reservoir is uniform only over successfully parsed eligible resources; skipped records are excluded.",
            "Rare occupations may be absent from a finite reservoir even when the whole file was scanned.",
            "No separate companies-file join or exact duplicate detection is performed.",
            "SQLite progress metadata is diagnostic; restart/resume is not implemented.",
        ],
    }
    temp_summary = summary_path.with_suffix(".tmp")
    temp_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_summary.replace(summary_path)
    _write_report(report_path, db, summary, csv_rows)
    _write_meta(db, {"status": "complete" if summary["complete_file_scanned"] else "partial",
                     "stop_reason": stop_reason, "population_processed": population})
    db.commit()
    db.close()
    print(f"Role exploration finished stop={stop_reason} eof={summary['complete_file_scanned']} population={population} sample={sample_actual} output={output_dir}", flush=True)
    return 0 if summary["complete_file_scanned"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    available = shutil.disk_usage(Path.cwd()).free
    prior = REPO_ROOT / "data/prepared/jobs_sample_validation_20261005T205936Z/jobs_prepared_sample.jsonl"
    if not prior.exists():
        prior = REPO_ROOT / "data/prepared/jobs_sample_20261005T205936Z/jobs_prepared_sample.jsonl"
    estimated_sample = int(prior.stat().st_size / 10_000 * args.sample_size) if prior.exists() else args.sample_size * 10_000
    estimated_db_and_csv = max(6 * 1024**3, estimated_sample * 16)
    minimum_free_during_scan = DISK_RESERVE_BYTES + estimated_sample
    required = estimated_sample + estimated_db_and_csv + DISK_RESERVE_BYTES
    storage_estimate = {"estimated_sample_bytes": estimated_sample, "estimated_sqlite_and_csv_budget_bytes": estimated_db_and_csv,
                        "reserve_bytes": DISK_RESERVE_BYTES, "required_free_bytes": required,
                        "minimum_free_during_scan_bytes": minimum_free_during_scan,
                        "available_free_bytes": available,
                        "basis": "sample scales from existing prepared 10k JSONL size; SQLite plus high-cardinality CSV budget is max(6 GiB, 16x sample size); actual distinct-title/pair cardinality is data-dependent"}
    print(f"Storage estimate: sample~{estimated_sample/1024**3:.2f} GiB; SQLite+CSV budget~{estimated_db_and_csv/1024**3:.2f} GiB; reserve=2 GiB; free={available/1024**3:.2f} GiB", flush=True)
    if available < required:
        print("Insufficient disk space for estimated outputs; source will not be opened.", file=sys.stderr, flush=True)
        return 2
    output_dir = args.output_dir or OUTPUT_ROOT / datetime.now(timezone.utc).strftime("roles_%Y%m%dT%H%M%SZ")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    _self_test()
    return _run_exploration(output_dir, args.sample_size, args.seed, storage_estimate, minimum_free_during_scan)


if __name__ == "__main__":
    raise SystemExit(main())
