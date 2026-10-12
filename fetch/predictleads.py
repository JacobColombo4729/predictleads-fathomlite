"""The PredictLeads export: its S3 location (from .env, never written into code) and its local copies (data/raw/).

Shared by analysis/ (src.pl_data, src.stream_full) and app/ (app/code/prepare_jobs and the pipeline scripts). Every
reader asks raw_file() first and streams from S3 only when there's no local copy; fetch.get_predictleads
downloads the files into data/raw/.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from fetch.paths import ENV_FILE, RAW_DIR

SNAPSHOT_DATE = "2026-09-30"  # the export used: the data's end date
SNAPSHOT = SNAPSHOT_DATE.replace("-", "_")  # as written in the file names
RAW_FILES = {"companies": f"companies_{SNAPSHOT}.jsonl.zst", "job_openings": f"job_openings_{SNAPSHOT}.jsonl.zst"}

_DATED = SNAPSHOT_DATE.replace("-", "/")

# The S3 location and AWS profile come only from the top-level .env (README steps 7-8)
load_dotenv(ENV_FILE, override=False)
BUCKET, _, _S3_PREFIX = os.environ.get("S3_PATH", "").strip().removeprefix("s3://").partition("/")
PREFIX = (_S3_PREFIX.rstrip("/") or _DATED) + "/"
KEY = PREFIX + RAW_FILES["job_openings"]  # the job-openings file's S3 key
PROFILE = os.environ.get("AWS_PROFILE") or None
REGION = os.environ.get("AWS_REGION") or "us-east-2"

SETUP = " Set S3_PATH and AWS_PROFILE in the top-level .env: README step 9."


def raw_file(name):
    """The export file in data/raw/, or None (then readers stream from S3)."""
    path = RAW_DIR / name
    return path if path.is_file() else None


def require_bucket():
    """The bucket from S3_PATH, or an explanation of the setup before any S3 call."""
    if not BUCKET:
        raise SystemExit("S3_PATH isn't set, and there's no local copy in data/raw/." + SETUP)
    return BUCKET


def s3_session():
    """The AWS session from AWS_PROFILE; a profile that doesn't exist (e.g. an old hard-coded name) gets the setup steps."""
    import boto3
    import botocore.exceptions

    require_bucket()
    try:
        return boto3.Session(profile_name=PROFILE, region_name=REGION)
    except botocore.exceptions.ProfileNotFound:
        raise SystemExit(f"The AWS profile {PROFILE!r} (AWS_PROFILE) isn't set up on this machine." + SETUP) from None


def connect(list_files=True):
    """(S3 client, bucket, prefix) from .env; lists the export's files with their sizes (names and sizes only)."""
    s3 = s3_session().client("s3")
    if list_files:
        for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX).get("Contents", []):
            print(Path(obj["Key"]).name, round(obj["Size"] / 1e6, 1), "MB")
    return s3, BUCKET, PREFIX
