"""Load the PredictLeads data export of 2026-09-30 (PredictLeads calls each daily export a snapshot): S3 connection, cached raw documents, resumable flattening, and schemas.

Used by notebooks/EDA_SDA_1.ipynb, sections 1-5. Everything prints aggregates only: names, shapes, sizes.

Layout under data/ (git-ignored):
    data/raw/<export file>.jsonl.zst  the export's files, if downloaded (python -m src.get_predictleads); read instead of S3
    data/<file>.parquet               cached raw documents (delete one to rebuild it from data/raw/ or S3)
    data/flat/<table>/part_NNNN.pkl   flattened tables, in parts, with a _done marker when complete
"""
import gc
import io
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda_helpers import DATA_DIR, REPO_ROOT, ROOT

SNAPSHOT = "2026_09_30"  # the export date in the file names: the data's end date
JOBS_N = 200_000      # job-opening documents to read and flatten (the first N in the file)
PART_ROWS = 20_000    # records per saved part when flattening
FLAT_DIR = DATA_DIR / "flat"
FLAT_TABLES = ["companies", "companies_included", f"job_openings_{JOBS_N}", f"job_openings_included_{JOBS_N}"]
SCHEMA_DOC = ROOT / "docs" / "data.md"
RAW_DIR = DATA_DIR / "raw"  # the export's files as downloaded; the app's pipeline (app/code/) looks here too
RAW_FILES = {"companies": f"companies_{SNAPSHOT}.jsonl.zst", "job_openings": f"job_openings_{SNAPSHOT}.jsonl.zst"}


# Everywhere a copy of an export file has been kept: data/raw/ is the standard place; the rest are older habits
# (earlier runs saved companies in data/; hand downloads kept S3's YYYY/MM/DD/ folders; the app had its own data/).
# get_predictleads moves a copy found elsewhere into data/raw/ instead of downloading it again.
COPY_DIRS = [RAW_DIR, DATA_DIR, DATA_DIR / "2026/09/30", REPO_ROOT / "2026/09/30",
             REPO_ROOT / "app" / "data", REPO_ROOT / "app" / "data" / "raw"]


def copies(name):
    """Every local copy of an export file, data/raw/ first."""
    return [d / name for d in COPY_DIRS if (d / name).is_file()]


def raw_file(name):
    """The local copy of an export file to read, or None (then readers stream from S3)."""
    found = copies(name)
    return found[0] if found else None


# ---------- S3 ----------

DATA_HELP = """
How this project reaches the PredictLeads data changed (2026-10-12): nothing about it is written into the code.
Why: one setup for both halves (analysis/ and app/); no bucket names or AWS profiles in committed files (our data
agreement); and no duplicate copies, since the download script reuses files you already have.

Steps:
  1. In the top-level .env (copy .env.example if you don't have one; README step 8), set
       S3_PATH=s3://<bucket>/<YYYY>/<MM>/<DD>/     the S3 folder of the export
       AWS_PROFILE=<your AWS profile name>          the profile you set up with `aws configure` (README step 7)
  2. Optional, to read from disk instead of streaming from S3: from analysis/, run
       python -m src.get_predictleads --check      what's on S3 and what you already have
       python -m src.get_predictleads              downloads into data/raw/ (checks disk space and asks first;
                                                  a copy you downloaded by hand is moved there, not downloaded again)
  3. Rerun your command. Details: README step 9.
"""


def connect_s3():
    """S3 client, bucket and prefix from .env (AWS_PROFILE, S3_PATH). Prints variable names and file sizes only."""
    import boto3
    import botocore.exceptions
    from dotenv import dotenv_values, find_dotenv, load_dotenv

    env_file = find_dotenv(usecwd=True)
    print("variables in .env:", list(dotenv_values(env_file).keys()))
    load_dotenv(env_file, override=True)
    s3_path = os.getenv("S3_PATH")
    if not s3_path:
        raise SystemExit("S3_PATH isn't set in the top-level .env." + DATA_HELP)
    try:
        s3 = boto3.Session(profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION") or "us-east-2").client("s3")
    except botocore.exceptions.ProfileNotFound:
        raise SystemExit(f"The AWS profile in AWS_PROFILE ({os.getenv('AWS_PROFILE')!r}) isn't set up on this machine." + DATA_HELP) from None
    bucket, _, prefix = s3_path.strip().removeprefix("s3://").partition("/")
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        print(Path(obj["Key"]).name, round(obj["Size"] / 1e6, 1), "MB")
    return s3, bucket, prefix


def fetch(s3, bucket, key):
    """The local copy of an S3 object, downloading it to data/raw/ first if there isn't one."""
    local = raw_file(Path(key).name)
    if local is None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        local = RAW_DIR / Path(key).name
        s3.download_file(bucket, key, str(local) + ".part")
        Path(str(local) + ".part").replace(local)
    return local


def stream_jsonl(s3, bucket, key, nrows=None, chunksize=None):
    """Read a .jsonl.zst line by line: the local copy in data/raw/ if there is one, otherwise straight from S3."""
    import zstandard as zstd

    local = raw_file(Path(key).name)
    body = open(local, "rb") if local else s3.get_object(Bucket=bucket, Key=key)["Body"]
    reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(body)
    return pd.read_json(io.TextIOWrapper(reader, encoding="utf-8"), lines=True, nrows=nrows, chunksize=chunksize)


def load_cached(parquet_path, build):
    """Read parquet if cached; otherwise call build(), save it as parquet, and return it."""
    if parquet_path.exists():
        print("loading cached:", parquet_path.name)
        return pd.read_parquet(parquet_path)
    print("building (slow, first run only):", parquet_path.name)
    df = build()
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print("saved:", parquet_path.name, round(parquet_path.stat().st_size / 1e6, 1), "MB")
    return df


def load_raw():
    """The raw documents, or {} when every table is already flattened (they aren't needed then, saving several GB).
    Companies (~252 MB compressed) are loaded in full; job openings (~19 GB) are the first JOBS_N documents."""
    if all(flat_done(n) for n in FLAT_TABLES):
        print("all tables already flattened: skipping the raw documents")
        return {}
    DATA_DIR.mkdir(exist_ok=True)
    conn = {}

    def s3_args(name):
        """(s3, bucket, key) for an export file; connects to S3 only if there's no local copy."""
        if raw_file(name):
            return None, None, name
        if not conn:
            conn["s3"], conn["bucket"], conn["prefix"] = connect_s3()
        return conn["s3"], conn["bucket"], conn["prefix"] + name

    raw = {
        "companies_raw": load_cached(
            DATA_DIR / f"companies_{SNAPSHOT}.parquet",
            lambda: pd.read_json(raw_file(RAW_FILES["companies"]) or fetch(*s3_args(RAW_FILES["companies"])),
                                 compression="zstd", lines=True)),
        "job_openings_sample": load_cached(
            DATA_DIR / f"job_openings_sample_{SNAPSHOT}.parquet",
            lambda: stream_jsonl(*s3_args(RAW_FILES["job_openings"]), nrows=JOBS_N)),
    }
    for name, df in raw.items():
        print(f"{name} shape:", df.shape)
    return raw


def describe_structure(raw):
    """Column names and the Python type of each value in the first row; no values."""
    for name, df in raw.items():
        print(name, list(df.columns))
        print(df.iloc[0].map(lambda v: type(v).__name__).to_dict())


# ---------- flattening ----------

def flat_done(name):
    return (FLAT_DIR / name / "_done").exists()


def flatten_to_parts(df, col, name, dedupe=False):
    """Flatten df[col] into data/flat/<name>/part_NNNN.pkl, then write a _done marker.
    Each part is saved as soon as it's built, so after a crash a rerun skips finished parts.
    dedupe=True keeps the first (type, id) only, *before* flattening, to save memory."""
    out_dir = FLAT_DIR / name
    if flat_done(name):
        print(f"{name}: already flattened")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("part_*.pkl")}
    start, seen, buf, kept, part_no = time.time(), set(), [], 0, 0
    skip = "part_0000.pkl" in existing

    def flush():
        nonlocal buf, part_no, skip
        if buf:
            pd.json_normalize(buf).to_pickle(out_dir / f"part_{part_no:04d}.pkl")
            print(f"  {name}: part {part_no} saved, {kept:,} records so far, {time.time() - start:.0f}s", flush=True)
        buf, part_no = [], part_no + 1
        skip = f"part_{part_no:04d}.pkl" in existing
        gc.collect()

    for batch in df[col]:
        if not isinstance(batch, (list, np.ndarray)):  # lists come back as numpy arrays from the parquet cache
            continue
        for r in batch:
            if dedupe:
                key = (r.get("type"), r.get("id"))
                if key in seen:
                    continue
                seen.add(key)
            kept += 1
            if not skip:
                buf.append(r)
            if kept % PART_ROWS == 0:
                flush()
    flush()
    (out_dir / "_done").touch()
    print(f"{name}: {kept:,} records in {time.time() - start:.0f}s")


def flatten_all(raw):
    """Flatten the records (`data`) and related objects (`included`, duplicates removed) of both files."""
    if not raw:
        return
    companies, jobs = raw["companies_raw"], raw["job_openings_sample"].head(JOBS_N)
    flatten_to_parts(companies, "data", "companies")
    flatten_to_parts(companies, "included", "companies_included", dedupe=True)
    flatten_to_parts(jobs, "data", f"job_openings_{JOBS_N}")
    flatten_to_parts(jobs, "included", f"job_openings_included_{JOBS_N}", dedupe=True)


def map_parts(name, fn):
    """Apply fn to each saved part and concatenate the results, one part in memory at a time."""
    parts = sorted((FLAT_DIR / name).glob("part_*.pkl"))
    return pd.concat([fn(pd.read_pickle(f)) for f in parts], ignore_index=True)


def load_flat(name, columns):
    """Load only the listed columns of a flattened table (missing columns come back empty)."""
    return map_parts(name, lambda d: d.reindex(columns=columns))


# ---------- schemas ----------

def schema_md(name, table, note=""):
    """A df.info()-style table for one flattened table: names, non-null counts and dtypes, never values."""
    n, non_null, dtypes = 0, pd.Series(dtype="int64"), {}
    for f in sorted((FLAT_DIR / table).glob("part_*.pkl")):
        part = pd.read_pickle(f)
        n += len(part)
        non_null = non_null.add(part.notna().sum(), fill_value=0)
        for col, dtype in part.dtypes.items():
            dtypes[col] = dtype if dtypes.get(col, dtype) == dtype else "object"
        del part
    lines = [f"#### `{name}`: {n:,} rows × {len(dtypes)} columns{note}", "",
             "| # | Column | Non-null | Null % | Dtype |", "|---|---|---|---|---|"]
    for i, (col, dtype) in enumerate(dtypes.items()):
        nn = int(non_null.get(col, 0))
        lines.append(f"| {i} | `{col}` | {nn:,} | {100 * (1 - nn / n) if n else 0:.1f}% | {dtype} |")
    return "\n".join(lines)


def write_schemas(doc=SCHEMA_DOC):
    """Write the four schemas between the schema markers in docs/data.md, only if they changed."""
    jobs_note = f" (from the first {JOBS_N:,} job-opening documents)"
    body = "\n\n".join([
        schema_md("companies", "companies"),
        schema_md("companies_included", "companies_included", " (duplicates removed by type and id)"),
        schema_md("job_openings", f"job_openings_{JOBS_N}", jobs_note),
        schema_md("job_openings_included", f"job_openings_included_{JOBS_N}", jobs_note + ", duplicates removed by type and id"),
    ])
    text = doc.read_text()
    pattern = r"(<!-- schema:start -->\n).*?(\n<!-- schema:end -->)"
    current = re.search(pattern, text, flags=re.S)
    if not current:
        raise RuntimeError(f"schema markers not found in {doc.name}")
    if body in current.group(0):
        print(f"schemas unchanged, {doc.name} left as is")
        return
    generated = f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} from the 2026-09-30 data export._\n\n" + body
    doc.write_text(re.sub(pattern, lambda m: m.group(1) + generated + m.group(2), text, flags=re.S))
    print(f"wrote schemas for 4 tables to {doc.name}")
