"""Stream the full job-openings file from S3 and keep only what the project uses. Nothing is downloaded whole.

Usage, from the project folder (PL or .venv activated):
    python -m src.stream_full --test 1000000   -> a test run on the first 1,000,000 documents, saved separately under
                                                 data/flat/stream_test/, with projections for the full file
    python -m src.stream_full                  -> the full run (resumable: rerun after a crash or a closed laptop)

Saved under data/flat/ (git-ignored; prints aggregates only: counts, sizes, speeds):
    job_openings_full/part_NNNN.pkl            niche listings (English or unlabeled, US or no country, computer and
                                               mathematical by occupation group or a 15- O*NET code, as in EDA_SDA_2 1d), all columns,
                                               the same flattened format as job_openings_200000/
    job_openings_included_full/part_NNNN.pkl   the related objects of those documents, duplicates removed
    all_fields_lean_full/part_NNNN.pkl         every English US (or no-country) listing in any field, lean columns
                                               only (no text): company volume and the occupation summary come from it
    stream_full_progress.json                  documents done, for resuming

Filters match EDA_SDA_1 (English, US or no country) and EDA_SDA_2 1d (the niche); later notebooks apply the rest.
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

NICHE_GROUP = "Computer and Mathematical"
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
    """Text lines of the job-openings file: from S3, or from a local .jsonl.zst (for tests). Returns (lines, counter, total bytes)."""
    import zstandard as zstd

    if local:
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
    out["desc_words"] = desc.str.split().str.len()
    keep = out.language.isin(["en"]) | out.language.isna()
    keep &= out.country.eq("United States") | out.country.isna()
    return out[keep]


def niche_mask(flat):
    """English or unlabeled, US or no country, computer and mathematical: the group, or a 15- O*NET code (EDA_SDA_2 1d's rule)."""
    col = lambda c: flat.reindex(columns=[c])[c]
    lang, group, code = col("attributes.language"), col("attributes.onet_data.family"), col("attributes.onet_data.code")
    country = col("attributes.location_data").map(first_country)
    in_group = group.eq(NICHE_GROUP) | code.fillna("").str.startswith("15-")
    return (lang.eq("en") | lang.isna()) & (country.eq("United States") | country.isna()) & in_group


class PartWriter:
    """Buffers rows and saves numbered parts of PART_ROWS rows; resumes numbering after a restart."""
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.part_no = len(list(folder.glob("part_*.pkl")))
        self.buf, self.rows = [], 0

    def add(self, df):
        if len(df):
            self.buf.append(df)
            self.rows += len(df)
        if sum(len(d) for d in self.buf) >= PART_ROWS:
            self.flush()

    def flush(self):
        if self.buf:
            pd.concat(self.buf, ignore_index=True).to_pickle(self.folder / f"part_{self.part_no:04d}.pkl")
            self.part_no += 1
            self.buf = []


def folder_size(folder):
    return sum(f.stat().st_size for f in folder.glob("*")) if folder.exists() else 0


def run(max_docs=None, local=None, out_root=None):
    out_root = Path(out_root) if out_root else (FLAT_DIR / "stream_test" if max_docs else FLAT_DIR)
    names = {"niche": "job_openings_full", "included": "job_openings_included_full", "lean": "all_fields_lean_full"}
    progress_file = out_root / "stream_full_progress.json"
    out_root.mkdir(parents=True, exist_ok=True)
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else {"docs_done": 0, "records": 0, "niche": 0, "lean": 0}
    if progress.get("finished"):
        print("already finished:", {k: v for k, v in progress.items() if k != "finished"})
        return progress
    # Parts are saved at each chunk boundary, then the checkpoint; a crash in between can repeat one chunk's rows,
    # which the notebooks remove (they drop duplicate ids).
    writers = {k: PartWriter(out_root / v) for k, v in names.items()}
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
        bar.set_postfix_str(f"{docs:,} docs | {progress['records']:,} listings | niche {progress['niche']:,} "
                            f"({progress['niche'] / max(progress['records'], 1):.1%})", refresh=final)

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
        mask = niche_mask(flat)
        writers["niche"].add(flat[mask].reset_index(drop=True))
        lean = lean_rows(flat).reset_index(drop=True)
        writers["lean"].add(lean)
        progress["lean"] += len(lean)
        if inc:
            writers["included"].add(pd.json_normalize(inc))
        progress["records"] += len(flat)
        progress["niche"] += int(mask.sum())

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
    print(f"kept: niche {progress['niche']:,} listings, all-fields lean {progress['lean']:,} rows")
    print("disk used: " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in sizes.items()))
    if max_docs and share > 0:
        scale = 1 / share
        print(f"\nprojection for the full file (by compressed bytes read; rough):"
              f"\n  documents ≈ {docs * scale:,.0f}, listings ≈ {progress['records'] * scale:,.0f}, niche ≈ {progress['niche'] * scale:,.0f}"
              f"\n  time ≈ {elapsed * scale / 3600:.1f} hours"
              f"\n  disk ≈ {sum(sizes.values()) * scale / 1e9:.1f} GB (niche {sizes['niche'] * scale / 1e9:.1f} GB, "
              f"lean {sizes['lean'] * scale / 1e9:.1f} GB, included {sizes['included'] * scale / 1e9:.2f} GB)")
    return progress


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=int, metavar="N", help="test run on the first N documents (saved under data/flat/stream_test/)")
    ap.add_argument("--local", help=argparse.SUPPRESS)  # a local .jsonl.zst, for tests on synthetic data
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    run(max_docs=args.test, local=args.local, out_root=args.out)
