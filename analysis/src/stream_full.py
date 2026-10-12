"""Stream the full job-openings file from S3 and keep only what the project uses. Nothing is downloaded whole.

Usage, from the project folder (PL or .venv activated):
    python -m src.stream_full --test 1000000   -> a test run on the first 1,000,000 documents, saved separately under
                                                 data/flat/stream_test/, with projections for the full file
    python -m src.stream_full                  -> the full run (resumable: rerun after a crash or a closed laptop)
    python -m src.stream_full --rows           -> a second pass that saves EDA_SDA_1's full cleaned row (src/listing_rows.py)
                                                 for every English US listing, for full-scale EDA_SDA_1 (own progress file)
    python -m src.stream_full --extra          -> a third pass that saves every field for the TripleTen roles outside
                                                 computer and mathematical (EXTRA_ONET_BUCKETS in
                                                 src/career_switch.py), like the scope parts (own progress file)

Saved under data/flat/ (git-ignored; prints aggregates only: counts, sizes, speeds):
    job_openings_full/part_NNNN.pkl            scope listings (English or unlabeled, US or no country, computer and
                                               mathematical by occupation group or a 15- O*NET code, as in EDA_SDA_2 1d), all columns,
                                               the same flattened format as job_openings_200000/
    job_openings_included_full/part_NNNN.pkl   the related objects of those documents, duplicates removed
    all_fields_lean_full/part_NNNN.parquet     every English US (or no-country) listing in any field, lean typed columns
                                               only (no text): company volume and the occupation summary come from it
    all_fields_rows_full/part_NNNN.parquet     (--rows) EDA_SDA_1's cleaned row for every English US (or no-country)
                                               listing in any field: text flags, source, repost fingerprint; no text
    job_openings_extra/part_NNNN.pkl           (--extra) the TripleTen-role listings outside computer and
                                               mathematical (sales engineers, business and market research analysts), all columns, the same format as job_openings_full/
    stream_full_progress.json                  documents done, for resuming (stream_rows_progress.json for --rows,
                                               stream_extra_progress.json for --extra)

Filters match EDA_SDA_1 (English, US or no country) and EDA_SDA_2 1d (the scope); later notebooks apply the rest.
"""
import argparse
import gc
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.eda_helpers import DATA_DIR
from src.pl_data import FLAT_DIR, PART_ROWS, SNAPSHOT

SCOPE_GROUP = "Computer and Mathematical"
DOCS_PER_CHUNK = 20_000
LEAN_COLS = {  # flattened column -> short name (EDA_SDA_1 6.1a names), no text
    "id": "id",
    "relationships.company.data.id": "company_id",
    "attributes.first_seen_at": "first_seen",
    "attributes.last_seen_at": "last_seen",
    "attributes.posted_at": "posted",
    "attributes.seniority": "seniority",
    "attributes.language": "language",
    "attributes.onet_data.code": "onet_code",
    "attributes.onet_data.family": "occupation_group",
    "attributes.onet_data.occupation_name": "occupation",
    "attributes.salary_data.salary_low_usd": "salary_low_usd",
    "attributes.salary_data.salary_high_usd": "salary_high_usd",
    "attributes.salary_data.salary_time_unit": "salary_unit",
}
TAG_RE = re.compile(r"<[^>]+>")


class CountingReader(io.RawIOBase):
    """Wraps the S3 body and counts compressed bytes read, to project the full run from a test run."""
    def __init__(self, raw):
        self.raw, self.bytes_read = raw, 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n


def open_lines(local=None):
    """Text lines of the job-openings file: a given .jsonl.zst (tests), else the local copy in data/raw/
    (python -m fetch.get_predictleads), else straight from S3. Returns (lines, counter, total bytes)."""
    import zstandard as zstd
    from src.pl_data import RAW_FILES, raw_file

    local = local or raw_file(RAW_FILES["job_openings"])
    if local:
        print(f"reading the local copy: {Path(local).name}")
        raw, total = open(local, "rb"), Path(local).stat().st_size
    else:
        from src.pl_data import connect_s3
        s3, bucket, prefix = connect_s3()
        key = prefix + f"job_openings_{SNAPSHOT}.jsonl.zst"
        total = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"]
    counter = CountingReader(raw)
    reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(counter)
    return io.TextIOWrapper(reader, encoding="utf-8"), counter, total


def first_country(locs):
    if isinstance(locs, list) and locs and isinstance(locs[0], dict):
        return locs[0].get("country")
    return None


def lean_rows(flat):
    """Every English (or unlabeled) US (or no-country) listing, lean columns: the all-fields table."""
    out = flat.reindex(columns=list(LEAN_COLS)).rename(columns=LEAN_COLS)
    out["country"] = flat.reindex(columns=["attributes.location_data"])["attributes.location_data"].map(first_country)
    desc = flat.reindex(columns=["attributes.description"])["attributes.description"].fillna("").map(lambda s: TAG_RE.sub(" ", s))
    out["desc_words"] = desc.str.split().str.len().fillna(0).astype("int32")
    keep = out.language.isin(["en"]) | out.language.isna()
    keep &= out.country.eq("United States") | out.country.isna()
    out = out[keep]
    for c in ["first_seen", "last_seen", "posted"]:  # typed, so the 13 million rows stay small in memory
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    for c in ["salary_low_usd", "salary_high_usd"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


LIST_COLS = ["contract_types", "tags", "categories"]


def typed_rows(flat):
    """EDA_SDA_1's cleaned row (lean_part) for every English (or unlabeled) US (or no-country) listing, typed for Parquet."""
    from src.listing_rows import lean_part
    out = lean_part(flat)
    col = lambda c: flat.reindex(columns=[c])[c]
    # Raw fields EDA_SDA_1 doesn't use yet, for the feature screening (EDA_SDA_2). Recruiter names and contacts are
    # personal data: only a yes/no flag is kept. last_processed_at likely moves with last_seen: leakage audit before use.
    # job_title: PredictLeads' normalized title; where it's missing, the posted title lowercased and trimmed (flagged)
    normalized = col("attributes.normalized_title")
    out["job_title"] = normalized.fillna(out.title.astype("string").str.lower().str.split().str.join(" "))
    out["job_title_fallback"] = normalized.isna()
    out["has_recruiter"] = col("attributes.recruiter_data.name").notna() | col("attributes.recruiter_data.contact").notna()
    out["last_processed"] = col("attributes.last_processed_at")
    out["salary_currency"] = col("attributes.salary_data.salary_currency")
    out["salary_low"] = col("attributes.salary_data.salary_low")    # as posted, in salary_currency (any currency;
    out["salary_high"] = col("attributes.salary_data.salary_high")  # the *_usd columns are the USD conversions)
    out["was_translated"] = col("attributes.translated_title").notna()
    keep = (out.language.eq("en") | out.language.isna()) & (out.country.eq("United States") | out.country.isna())
    out = out[keep].copy()
    for c in ["first_seen", "last_seen", "posted", "last_processed"]:
        out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    for c in ["salary_low_usd", "salary_high_usd", "salary_low", "salary_high", "years_required"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out["desc_words"] = out.desc_words.fillna(0).astype("int32")
    out["n_locations"] = out.n_locations.astype("int16")
    for c in LIST_COLS:  # one consistent type for Parquet: a list of strings, or empty
        out[c] = out[c].map(lambda v: [str(x) for x in v] if isinstance(v, (list, tuple)) else [])
    for c in ["id", "company_id", "title", "status", "seniority", "language", "onet_code", "occupation_group", "occupation",
              "salary_unit", "desc_hash", "country", "source_type", "platform", "job_title", "salary_currency"]:
        out[c] = out[c].astype("string")
    return out


def extra_mask(flat):
    """English or unlabeled, US or no country, and an O*NET code in EXTRA_ONET_BUCKETS (the TripleTen roles outside
    computer and mathematical)."""
    from src.career_switch import EXTRA_ONET_BUCKETS
    col = lambda c: flat.reindex(columns=[c])[c]
    lang, code = col("attributes.language"), col("attributes.onet_data.code").fillna("").astype(str).str[:7]
    country = col("attributes.location_data").map(first_country)
    return (lang.eq("en") | lang.isna()) & (country.eq("United States") | country.isna()) & code.isin(list(EXTRA_ONET_BUCKETS))


def scope_mask(flat):
    """English or unlabeled, US or no country, computer and mathematical: the group, or a 15- O*NET code (EDA_SDA_2 1d's rule)."""
    col = lambda c: flat.reindex(columns=[c])[c]
    lang, group, code = col("attributes.language"), col("attributes.onet_data.family"), col("attributes.onet_data.code")
    country = col("attributes.location_data").map(first_country)
    in_group = group.eq(SCOPE_GROUP) | code.fillna("").str.startswith("15-")
    return (lang.eq("en") | lang.isna()) & (country.eq("United States") | country.isna()) & in_group


class PartWriter:
    """Buffers rows and saves numbered parts (pickle, or Parquet for the typed lean table); resumes numbering after a restart."""
    def __init__(self, folder, fmt="pkl"):
        self.folder, self.fmt = folder, fmt
        folder.mkdir(parents=True, exist_ok=True)
        self.part_no = len(list(folder.glob(f"part_*.{fmt}")))
        self.buf, self.rows = [], 0

    def add(self, df):
        if len(df):
            self.buf.append(df)
            self.rows += len(df)
        if sum(len(d) for d in self.buf) >= PART_ROWS:
            self.flush()

    def flush(self):
        if self.buf:
            df = pd.concat(self.buf, ignore_index=True)
            path = self.folder / f"part_{self.part_no:04d}.{self.fmt}"
            df.to_parquet(path, index=False) if self.fmt == "parquet" else df.to_pickle(path)
            self.part_no += 1
            self.buf = []


def folder_size(folder):
    return sum(f.stat().st_size for f in folder.glob("*")) if folder.exists() else 0


def run(max_docs=None, local=None, out_root=None, rows=False, extra=False):
    out_root = Path(out_root) if out_root else (FLAT_DIR / "stream_test" if max_docs else FLAT_DIR)
    if rows:  # second pass: only EDA_SDA_1's cleaned rows, for every field
        names = {"rows": "all_fields_rows_full"}
        progress_file = out_root / "stream_rows_progress.json"
    elif extra:  # third pass: every field for the TripleTen roles outside computer and mathematical
        names = {"scope": "job_openings_extra"}
        progress_file = out_root / "stream_extra_progress.json"
    else:
        names = {"scope": "job_openings_full", "included": "job_openings_included_full", "lean": "all_fields_lean_full"}
        progress_file = out_root / "stream_full_progress.json"
    out_root.mkdir(parents=True, exist_ok=True)
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {"docs_done": 0, "records": 0, "scope": 0, "lean": 0}
    if progress.get("finished"):
        print("already finished:", {k: v for k, v in progress.items() if k != "finished"})
        return progress
    # Parts are saved at each chunk boundary, then the checkpoint; a crash in between can repeat one chunk's rows,
    # which the notebooks remove (they drop duplicate ids).
    writers = {k: PartWriter(out_root / v, "parquet" if k in ("lean", "rows") else "pkl") for k, v in names.items()}
    seen_included = set()
    lines, counter, total_bytes = open_lines(local)
    start, docs, skipped = time.time(), 0, progress["docs_done"]
    chunk = []
    # Progress bar: share of the file read (full run) or of the test's documents, with elapsed time, time left and speed
    if max_docs:
        bar = tqdm(total=max_docs, unit=" docs", unit_scale=True, desc="Test run", dynamic_ncols=True)
    else:
        bar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="Full file", dynamic_ncols=True)
    if skipped:
        bar.set_description("Skipping finished documents")

    def tick(final=False):
        bar.update((docs if max_docs else counter.bytes_read) - bar.n)
        bar.set_postfix_str(f"{docs:,} docs | {progress['records']:,} listings | scope {progress['scope']:,} "
                            f"({progress['scope'] / max(progress['records'], 1):.1%})", refresh=final)

    def process(chunk_lines):
        recs, inc = [], []
        for line in chunk_lines:
            doc = json.loads(line)
            data, included = doc.get("data"), doc.get("included")
            recs += data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
            for r in included if isinstance(included, list) else []:
                key = (r.get("type"), r.get("id"))
                if key not in seen_included:
                    seen_included.add(key)
                    inc.append(r)
        flat = pd.json_normalize(recs)
        mask = extra_mask(flat) if extra else scope_mask(flat)
        if extra:
            writers["scope"].add(flat[mask].reset_index(drop=True))
        elif rows:
            r = typed_rows(flat).reset_index(drop=True)
            writers["rows"].add(r)
            progress["rows"] = progress.get("rows", 0) + len(r)
        else:
            writers["scope"].add(flat[mask].reset_index(drop=True))
            lean = lean_rows(flat).reset_index(drop=True)
            writers["lean"].add(lean)
            progress["lean"] += len(lean)
            if inc:
                writers["included"].add(pd.json_normalize(inc))
        progress["records"] += len(flat)
        progress["scope"] += int(mask.sum())

    for line in lines:
        docs += 1
        if docs % 1_000 == 0:
            tick()
        if docs <= skipped:  # already done in an earlier run: read past it without parsing
            continue
        if docs == skipped + 1 and skipped:
            bar.set_description("Full file" if not max_docs else "Test run")
        chunk.append(line)
        if len(chunk) == DOCS_PER_CHUNK or (max_docs and docs >= max_docs):
            process(chunk)
            chunk = []
            for w in writers.values():
                w.flush()  # parts end on a chunk boundary, so a checkpoint never splits a part
            progress["docs_done"] = docs
            progress_file.write_text(json.dumps(progress))
            tick(final=True)
            gc.collect()
        if max_docs and docs >= max_docs:
            break
    if chunk:
        process(chunk)
        for w in writers.values():
            w.flush()
        progress["docs_done"] = docs
    tick(final=True)
    bar.close()
    elapsed, share = time.time() - start, counter.bytes_read / total_bytes
    if not max_docs:
        progress["finished"] = True
    progress_file.write_text(json.dumps(progress))

    sizes = {k: folder_size(out_root / v) for k, v in names.items()}
    print(f"\ndone: {docs:,} documents, {progress['records']:,} listings in {elapsed / 60:.1f} min; read {share:.1%} of the file")
    if extra:
        print(f"kept: {progress['scope']:,} TripleTen-role listings outside computer and mathematical")
    elif rows:
        print(f"kept: {progress.get('rows', 0):,} cleaned rows in all fields (scope share of listings: {progress['scope']:,})")
    else:
        print(f"kept: scope {progress['scope']:,} listings, all-fields lean {progress['lean']:,} rows")
    print("disk used: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sizes.items()))
    if max_docs and share > 0:
        scale = 1 / share
        print(f"\nprojection for the full file (by compressed bytes read; rough):"
              f"\n  documents about {docs * scale:,.0f}, listings about {progress['records'] * scale:,.0f}, scope about {progress['scope'] * scale:,.0f}"
              f"\n  time about {elapsed * scale / 3600:.1f} hours"
              f"\n  disk about {sum(sizes.values()) * scale / 1e9:.1f} GB ("
              + ", ".join(f"{k} {v * scale / 1e9:.2f} GB" for k, v in sizes.items()) + ")")
    return progress


# ---------- bulk mode: one pass over the file, chunks parsed in parallel (machines with the memory and cores) ----------

TABLES = {"scope": ("job_openings_full", "pkl"), "lean": ("all_fields_lean_full", "parquet"),
          "rows": ("all_fields_rows_full", "parquet"), "extra": ("job_openings_extra", "pkl")}


def machine():
    """(total RAM GB, available RAM GB, CPU threads) of this computer."""
    import os
    import psutil
    mem = psutil.virtual_memory()
    return mem.total / 1e9, mem.available / 1e9, os.cpu_count() or 1


def choose_mode():
    """'bulk' on a machine with at least 16 GB of RAM and 8 threads, else 'stream' (the memory-light route)."""
    total, _, threads = machine()
    return "bulk" if total >= 16 and threads >= 8 else "stream"


def default_workers():
    """Worker processes for bulk mode: about 2 GB of free memory each, leaving two threads for reading the file."""
    _, available, threads = machine()
    return max(1, min(threads - 2, int(available // 2)))


def _bulk_chunk(index, chunk_lines, out_root):
    """One chunk of documents, in a worker process: writes its part of each table (named by the chunk's position,
    so the tables keep the file's order) and returns the counts and its related objects (deduplicated by the caller)."""
    recs, inc, seen = [], [], set()
    for line in chunk_lines:
        doc = json.loads(line)
        data, included = doc.get("data"), doc.get("included")
        recs += data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for r in included if isinstance(included, list) else []:
            key = (r.get("type"), r.get("id"))
            if key not in seen:
                seen.add(key)
                inc.append(r)
    flat = pd.json_normalize(recs)
    scope, extra = scope_mask(flat), extra_mask(flat)
    parts = {"scope": flat[scope], "lean": lean_rows(flat), "rows": typed_rows(flat), "extra": flat[extra]}
    for key, df in parts.items():
        folder, fmt = TABLES[key]
        path = Path(out_root) / folder / f"part_{index:05d}.{fmt}"
        df = df.reset_index(drop=True)
        if len(df):
            df.to_parquet(path, index=False) if fmt == "parquet" else df.to_pickle(path)
        else:
            path.unlink(missing_ok=True)  # a rerun after a crash leaves no stale part behind
    counts = {"records": len(flat), "scope": int(scope.sum()), "extra": int(extra.sum()),
              "lean": len(parts["lean"]), "rows": len(parts["rows"])}
    return counts, inc


def run_bulk(workers=None, max_docs=None, local=None, out_root=None):
    """Every table of the three streaming passes in one pass over the file, chunks parsed in parallel. Same tables and
    row order as stream mode, so the notebooks read them unchanged. Resumable: a rerun redoes only unfinished chunks."""
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    out_root = Path(out_root) if out_root else (FLAT_DIR / "stream_test_bulk" if max_docs else FLAT_DIR)
    workers = workers or default_workers()
    progress_file = out_root / "stream_bulk_progress.json"
    blank = {"docs_done": 0, "chunks_done": 0, "records": 0, "scope": 0, "extra": 0, "lean": 0, "rows": 0}
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else blank
    if progress.get("finished"):
        print("already finished:", {k: v for k, v in progress.items() if k != "finished"})
        return progress
    for folder, _ in TABLES.values():
        (out_root / folder).mkdir(parents=True, exist_ok=True)
    included = PartWriter(out_root / "job_openings_included_full")
    seen_included = set()
    lines, counter, total_bytes = open_lines(local)
    skipped = progress["docs_done"]
    print(f"bulk mode: {workers} worker processes; {skipped:,} documents already done")
    bar = tqdm(total=max_docs or total_bytes, unit=" docs" if max_docs else "B", unit_scale=True,
               unit_divisor=1000 if max_docs else 1024, desc="Bulk", dynamic_ncols=True)
    start, docs, index = time.time(), 0, progress["chunks_done"]
    pending = deque()

    def finish_oldest():
        counts, inc = pending.popleft().result()  # in submission order, so the progress stays contiguous
        for k, v in counts.items():
            progress[k] += v
        new = []
        for r in inc:
            key = (r.get("type"), r.get("id"))
            if key not in seen_included:
                seen_included.add(key)
                new.append(r)
        if new:
            included.add(pd.json_normalize(new))
        included.flush()
        progress["chunks_done"] += 1
        progress["docs_done"] = min(progress["chunks_done"] * DOCS_PER_CHUNK, docs)
        progress_file.write_text(json.dumps(progress))
        bar.update((progress["docs_done"] if max_docs else counter.bytes_read) - bar.n)
        bar.set_postfix_str(f"{progress['records']:,} listings | scope {progress['scope']:,}", refresh=False)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunk = []
        for line in lines:
            docs += 1
            if docs <= skipped:  # already done in an earlier run: read past it without parsing
                continue
            chunk.append(line)
            if len(chunk) == DOCS_PER_CHUNK or (max_docs and docs >= max_docs):
                pending.append(pool.submit(_bulk_chunk, index, chunk, str(out_root)))
                index, chunk = index + 1, []
                while len(pending) > workers + 1:  # bounds the chunks held in memory
                    finish_oldest()
            if max_docs and docs >= max_docs:
                break
        if chunk:
            pending.append(pool.submit(_bulk_chunk, index, chunk, str(out_root)))
        while pending:
            finish_oldest()
    progress["docs_done"] = docs
    bar.close()
    if not max_docs:
        progress["finished"] = True
        # The three streaming passes' progress files, so stream mode and the notebooks see the work as done
        base = {"docs_done": docs, "records": progress["records"], "finished": True}
        for name, extra in (("stream_full_progress.json", {"scope": progress["scope"], "lean": progress["lean"]}),
                            ("stream_rows_progress.json", {"scope": progress["scope"], "rows": progress["rows"]}),
                            ("stream_extra_progress.json", {"scope": progress["extra"]})):
            (out_root / name).write_text(json.dumps({**base, **extra}))
    progress_file.write_text(json.dumps(progress))
    elapsed = time.time() - start
    sizes = {folder: folder_size(out_root / folder) for folder in [*(f for f, _ in TABLES.values()), "job_openings_included_full"]}
    print(f"\ndone: {docs:,} documents, {progress['records']:,} listings in {elapsed / 60:.1f} min with {workers} workers")
    print(f"kept: scope {progress['scope']:,}, extra {progress['extra']:,}, all-fields lean {progress['lean']:,}, "
          f"cleaned rows {progress['rows']:,}")
    print("disk used: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sizes.items()))
    return progress


def build(mode="auto", workers=None):
    """Build every full-file table the notebooks need, skipping what's already done. mode: 'auto' (bulk on a capable
    machine, else stream), 'bulk' (one parallel pass) or 'stream' (three memory-light passes)."""
    mode = choose_mode() if mode == "auto" else mode
    total, available, threads = machine()
    print(f"{mode} mode ({total:.0f} GB RAM, {available:.0f} GB free, {threads} threads)")
    if mode == "bulk":
        return run_bulk(workers=workers)
    run()
    run(rows=True)
    return run(extra=True)


def load_lean(columns, root=None, table="all_fields_lean_full"):
    """An all-fields table, only the given columns, with repeated text as categories (memory-safe for 13M rows).
    table: "all_fields_lean_full" (first pass) or "all_fields_rows_full" (--rows: EDA_SDA_1's cleaned rows)."""
    import pyarrow.dataset as ds
    folder = Path(root or FLAT_DIR) / table
    return ds.dataset(sorted(folder.glob("part_*.parquet")), format="parquet").to_table(columns=columns).to_pandas(
        strings_to_categorical=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, metavar="N", help="test run on the first N documents (saved under data/flat/stream_test/)")
    ap.add_argument("--rows", action="store_true", help="second pass: EDA_SDA_1's cleaned rows for every field")
    ap.add_argument("--extra", action="store_true", help="third pass: every field for the TripleTen roles outside computer and mathematical")
    ap.add_argument("--bulk", action="store_true", help="one parallel pass that builds every table (needs about 16 GB RAM, 8+ threads)")
    ap.add_argument("--auto", action="store_true", help="every table: bulk on a capable machine, else the three stream passes")
    ap.add_argument("--workers", type=int, help="worker processes for --bulk (default: from free memory and threads)")
    ap.add_argument("--local", help=argparse.SUPPRESS)  # a local .jsonl.zst, for tests on synthetic data
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.auto:
        build("auto", workers=args.workers)
    elif args.bulk:
        run_bulk(workers=args.workers, max_docs=args.test, local=args.local, out_root=args.out)
    else:
        run(max_docs=args.test, local=args.local, out_root=args.out, rows=args.rows, extra=args.extra)
