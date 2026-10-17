"""Download the trained models' hand-offs from S3, so the apps run without rerunning the analysis notebooks.

Usage, from the repository root (.venv activated; needs AWS access and S3_PATH in .env, README steps 7-8):
    python -m fetch.get_handoffs              -> download the pinned version (VERSION) into its three folders
    python -m fetch.get_handoffs --check      -> what's here and what's on S3 (sizes only)
    python -m fetch.get_handoffs --yes        -> don't ask before downloading (the disk-space check still runs)
    python -m fetch.get_handoffs --force      -> also replace folders the notebooks wrote on this machine
    python -m fetch.get_handoffs --upload     -> (the analysis owner, after a rerun) upload this machine's hand-offs as
                                                 VERSION; refuses if VERSION is already on S3: bump it first

The three hand-offs, all from one run (the corpus is built from ML_Rank's scores, and the queue's weights were tuned on
ML_Fit's ratings, so they're always fetched as one set):
    data/clean/ml_fit/                 ML_Fit 9b: the trained role-fit layer
    data/clean/ml_rank/                ML_Rank section 13: listing scores and the queue's weights
    data/clean/combined_v1/corpus/     python -m combined.corpus build, then encode: the listings and their BGE vectors
Together they're everything the combined demo (combined_demo/) and Jacob's app (app/run_jacob.ps1) read from the
analysis. BGE comes from fetch.get_bge.

S3 holds each version under handoffs/<version>/ in S3_PATH's bucket (or under HANDOFF_S3_PATH/<version>/ if .env sets it), with INDEX.json (every file's size and SHA-256) written last,
so a version without it is incomplete and never downloaded. Downloads go to a staging folder, are checked against
INDEX.json, and replace the three folders only when all of them are complete. Each downloaded folder gets HANDOFF.json
(its version), so a rerun skips what's current, and folders without it (the notebooks' own output) are left alone
unless --force. Prints file names and sizes only.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

from fetch.downloads import confirm_space, gb
from fetch.paths import DATA_DIR, REPO_ROOT
from fetch.predictleads import BUCKET as PL_BUCKET, PROFILE, REGION

VERSION = "v1"  # bump after a rerun, then --upload; Jesse's next fetch replaces his copy
CLEAN_DIR = DATA_DIR / "clean"
HANDOFFS = {  # name on S3: (folder here, files the readers need)
    "ml_fit": (CLEAN_DIR / "ml_fit", {"manifest.json", "role_scorer.joblib", "fit_layer.joblib"}),  # combined/role_fit.py
    "ml_rank": (CLEAN_DIR / "ml_rank", {"listing_scores.parquet", "queue_config.json"}),  # combined/_paths.py
    "combined_corpus": (CLEAN_DIR / "combined_v1" / "corpus",  # combined/_paths.py CORPUS_DIR
                        {"jobs_prepared_corpus.jsonl", "bge_embeddings.npy", "bge_embeddings_manifest.json"}),
}
DEFAULT_PREFIX = "handoffs"  # in S3_PATH's bucket, next to the export
INDEX = "INDEX.json"     # at the version's root on S3
MARKER = "HANDOFF.json"  # in each downloaded folder
STAGING = CLEAN_DIR / ".handoff_staging"
SETUP = " Set S3_PATH and AWS_PROFILE in the top-level .env: README step 8."


def location():
    """(bucket, prefix of VERSION): HANDOFF_S3_PATH from .env if set, otherwise handoffs/ in S3_PATH's bucket."""
    bucket, _, prefix = os.environ.get("HANDOFF_S3_PATH", "").strip().removeprefix("s3://").partition("/")
    if not bucket:
        bucket, prefix = PL_BUCKET, DEFAULT_PREFIX
    if not bucket:
        raise SystemExit("Neither HANDOFF_S3_PATH nor S3_PATH is set." + SETUP)
    prefix = prefix.strip("/")
    return bucket, (prefix + "/" if prefix else "") + VERSION + "/"


def client():
    import boto3
    import botocore.exceptions

    try:
        return boto3.Session(profile_name=PROFILE, region_name=REGION).client("s3")
    except botocore.exceptions.ProfileNotFound:
        raise SystemExit(f"The AWS profile {PROFILE!r} (AWS_PROFILE) isn't set up on this machine." + SETUP) from None


def exists(s3, bucket, key):
    return any(o["Key"] == key for o in s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", []))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def local_files(folder):
    """{relative path: size} of a hand-off folder, without the marker and unfinished downloads."""
    if not folder.is_dir():
        return {}
    return {p.relative_to(folder).as_posix(): p.stat().st_size for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != MARKER and p.suffix != ".part"}


def marker_version(folder):
    try:
        return json.loads((folder / MARKER).read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def read_index(s3, bucket, prefix):
    if not exists(s3, bucket, prefix + INDEX):
        raise SystemExit(f"Version {VERSION} isn't on S3 (or its upload didn't finish): ask the analysis owner to run "
                         "python -m fetch.get_handoffs --upload.")
    return json.loads(s3.get_object(Bucket=bucket, Key=prefix + INDEX)["Body"].read())


def current(name, entry):
    """Whether the folder here is this version, complete."""
    folder = HANDOFFS[name][0]
    return (marker_version(folder) == VERSION
            and local_files(folder) == {rel: f["size"] for rel, f in entry["files"].items()})


def upload(s3, bucket, prefix, yes=False):
    """Upload this machine's three hand-offs as VERSION; INDEX.json last, so a broken upload is never downloaded."""
    if exists(s3, bucket, prefix + INDEX):
        raise SystemExit(f"Version {VERSION} is already on S3: bump VERSION in fetch/get_handoffs.py to upload a new one.")
    index = {"version": VERSION, "uploaded": date.today().isoformat(), "handoffs": {}}
    for name, (folder, required) in HANDOFFS.items():
        files = local_files(folder)
        missing = sorted(required - files.keys())
        if missing:
            raise SystemExit(f"{name}: {folder.relative_to(REPO_ROOT).as_posix()} is missing {', '.join(missing)}: "
                             "finish that step first (see this script's docstring).")
        index["handoffs"][name] = {"files": {rel: {"size": size, "sha256": sha256(folder / rel)}
                                             for rel, size in files.items()}}
        print(f"{name}: {len(files)} files, {gb(sum(files.values()))}")
    total = sum(f["size"] for h in index["handoffs"].values() for f in h["files"].values())
    print(f"uploading version {VERSION}: {gb(total)}")
    if not yes and input("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise SystemExit("Cancelled; nothing was uploaded.")
    for name, entry in index["handoffs"].items():
        for rel in entry["files"]:
            s3.upload_file(str(HANDOFFS[name][0] / rel), bucket, f"{prefix}{name}/{rel}")
    s3.put_object(Bucket=bucket, Key=prefix + INDEX, Body=json.dumps(index, indent=1).encode("utf-8"))
    print(f"uploaded version {VERSION}; tell the team to run python -m fetch.get_handoffs")


def download(s3, bucket, prefix, yes=False, force=False):
    """Download VERSION into staging, check every file, then swap all three folders in at once."""
    index = read_index(s3, bucket, prefix)
    todo = {name: entry for name, entry in index["handoffs"].items() if not current(name, entry)}
    for name in index["handoffs"].keys() - todo.keys():
        print(f"{name}: version {VERSION} already here")
    if not todo:
        return
    own = [name for name in todo if local_files(HANDOFFS[name][0]) and marker_version(HANDOFFS[name][0]) is None]
    if own and not force:
        raise SystemExit(f"{', '.join(own)}: these folders were written by the notebooks on this machine, so nothing "
                         "was replaced. Rerun with --force to replace them with the shared version.")
    need = sum(f["size"] for entry in todo.values() for f in entry["files"].values())
    confirm_space(need, CLEAN_DIR, f"downloading the hand-offs, version {VERSION} ({', '.join(todo)})", yes=yes)
    shutil.rmtree(STAGING, ignore_errors=True)
    try:
        for name, entry in todo.items():
            for rel, f in entry["files"].items():
                target = STAGING / name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, f"{prefix}{name}/{rel}", str(target))
                if target.stat().st_size != f["size"] or sha256(target) != f["sha256"]:
                    raise SystemExit(f"{name}/{rel}: the download doesn't match S3's index; run it again")
            print(f"{name}: downloaded {len(entry['files'])} files")
        for name in todo:  # every file checked: now replace the folders
            folder = HANDOFFS[name][0]
            (STAGING / name / MARKER).write_text(json.dumps({"version": VERSION, "downloaded": date.today().isoformat()}),
                                                 encoding="utf-8")
            folder.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(folder, ignore_errors=True)
            (STAGING / name).replace(folder)
            print(f"{name}: version {VERSION} in {folder.relative_to(REPO_ROOT).as_posix()}")
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)


def check(s3, bucket, prefix):
    on_s3 = exists(s3, bucket, prefix + INDEX)
    index = read_index(s3, bucket, prefix)["handoffs"] if on_s3 else {}
    for name, (folder, _) in HANDOFFS.items():
        files = local_files(folder)
        here = (f"version {marker_version(folder)}" if marker_version(folder) else "from the notebooks") if files else "none"
        size = gb(sum(f["size"] for f in index[name]["files"].values())) if name in index else "not uploaded"
        print(f"{name}: here {here}, {gb(sum(files.values()))}; S3 version {VERSION}: {size}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="show what's here and on S3, download nothing")
    parser.add_argument("--upload", action="store_true", help="upload this machine's hand-offs as VERSION")
    parser.add_argument("--yes", action="store_true", help="don't ask first (the disk-space check still runs)")
    parser.add_argument("--force", action="store_true", help="also replace folders the notebooks wrote here")
    args = parser.parse_args()
    bucket, prefix = location()
    s3 = client()
    if args.check:
        check(s3, bucket, prefix)
    elif args.upload:
        upload(s3, bucket, prefix, yes=args.yes)
    else:
        download(s3, bucket, prefix, yes=args.yes, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
