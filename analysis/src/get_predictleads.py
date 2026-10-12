"""Download the PredictLeads export files from S3 into data/raw/, so every script can read them locally.

Usage, from analysis/ (.venv activated; needs AWS access, README steps 7-8):
    python -m src.get_predictleads              -> both files: companies (~240 MB) and job openings (~17.7 GB)
    python -m src.get_predictleads companies    -> one file (companies or job_openings)
    python -m src.get_predictleads --check      -> what's local and what's on S3 (sizes only), no download
    python -m src.get_predictleads --yes        -> skip the question (the disk-space check still runs)

Before downloading it says how much space the download needs and how much is free, and asks to proceed; if it
doesn't fit (with 2 GB to spare), it stops without downloading anything.

Optional: without a local copy, every reader streams from S3 instead (src.pl_data, src.stream_full and the app's
pipeline in app/code/). With one, they read data/raw/ and skip the network.

It never downloads a file you already have: a complete copy anywhere it knows (src.pl_data.COPY_DIRS) is moved to
data/raw/ instead, an interrupted one is resumed, and extra copies are listed with an offer to delete them. A download is
checked against S3's size and lands under its real name only when complete, so a file in data/raw/ is always whole.
data/raw/MANIFEST.json records each file's size and S3 ETag. Prints file names and sizes only. data/ is git-ignored;
count these copies in the end-of-project deletion.
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from tqdm import tqdm

from src.downloads import confirm_space
from src.eda_helpers import REPO_ROOT
from src.pl_data import RAW_DIR, RAW_FILES, connect_s3, copies

MANIFEST = RAW_DIR / "MANIFEST.json"
CHUNK = 8 << 20  # 8 MB


def remote(s3, bucket, prefix, name):
    """S3 size and ETag of one export file."""
    head = s3.head_object(Bucket=bucket, Key=prefix + name)
    return head["ContentLength"], head.get("ETag", "").strip('"')


def download(s3, bucket, prefix, name, size):
    """Stream one file into data/raw/<name>.part, resuming from what's already there, then rename it."""
    target, part = RAW_DIR / name, RAW_DIR / (name + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done > size:  # a stale partial from a different file: start over
        part.unlink()
        done = 0
    if done < size:
        body = s3.get_object(Bucket=bucket, Key=prefix + name, Range=f"bytes={done}-")["Body"]
        with open(part, "ab") as f, tqdm(total=size, initial=done, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in iter(lambda: body.read(CHUNK), b""):
                f.write(chunk)
                bar.update(len(chunk))
    got = part.stat().st_size
    if got != size:
        raise SystemExit(f"{name}: got {got:,} bytes of {size:,}; rerun to resume")
    part.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="companies and/or job_openings (default: both)")
    parser.add_argument("--check", action="store_true", help="show local and S3 sizes, download nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading (the space check still runs)")
    args = parser.parse_args()
    unknown = set(args.files) - set(RAW_FILES)
    if unknown:
        parser.error(f"unknown file {sorted(unknown)}: choose from {list(RAW_FILES)}")
    wanted = args.files or list(RAW_FILES)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s3, bucket, prefix = connect_s3()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    show = lambda path: path.relative_to(REPO_ROOT).as_posix()
    plan = []
    for key in wanted:
        name = RAW_FILES[key]
        size, etag = remote(s3, bucket, prefix, name)
        found = copies(name)
        whole = [c for c in found if c.stat().st_size == size]
        partial = [c for c in found if c.stat().st_size != size]
        if args.check:
            print(f"{name}: S3 {size / 1e9:.2f} GB")
            for c in found:
                print(f"  {'complete' if c in whole else 'incomplete'}: {show(c)} ({c.stat().st_size / 1e9:.2f} GB)")
            if not found:
                print("  no local copy")
            continue
        plan.append((name, size, etag, whole, partial))
    if args.check:
        return

    # Never download what's already here: reuse a complete copy (moved to data/raw/), resume an incomplete one there
    for name, size, etag, whole, partial in plan:
        target, part = RAW_DIR / name, RAW_DIR / (name + ".part")
        if whole and whole[0] != target:
            whole[0].replace(target)
            print(f"{name}: found a complete copy at {show(whole[0])}; moved it to data/raw/ instead of downloading")
            whole[0] = target
        if not whole and target in partial and not part.exists():
            target.replace(part)  # an interrupted copy under the real name: resume it instead of starting over
            partial.remove(target)
    part_bytes = lambda name: (RAW_DIR / (name + ".part")).stat().st_size if (RAW_DIR / (name + ".part")).exists() else 0
    todo = [(name, size) for name, size, _, whole, _ in plan if not whole]
    if todo:
        confirm_space(sum(size - part_bytes(name) for name, size in todo), RAW_DIR,
                      f"downloading {', '.join(name for name, _ in todo)}", yes=args.yes)
    else:
        print("nothing to download: every file is already complete locally")

    extra = []
    for name, size, etag, whole, partial in plan:
        if whole:
            print(f"{name}: already complete in data/raw/ ({size / 1e9:.2f} GB)")
        else:
            download(s3, bucket, prefix, name, size)
            print(f"{name}: downloaded ({size / 1e9:.2f} GB)")
        manifest[name] = {"bytes": size, "etag": etag, "checked": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        extra += [(c, "a duplicate") for c in whole[1:]] + [(c, "an incomplete copy") for c in partial]
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"files in data/raw: {', '.join(sorted(manifest))}")

    # Other copies only take up space: offer to delete them (never without a yes typed by a person)
    if extra:
        total = sum(c.stat().st_size for c, _ in extra)
        for c, kind in extra:
            print(f"  {kind}: {show(c)} ({c.stat().st_size / 1e9:.2f} GB)")
        if args.yes or not sys.stdin.isatty():
            print(f"{len(extra)} extra copies ({total / 1e9:.2f} GB) left in place; rerun in a terminal to delete them")
        elif input(f"Delete these {len(extra)} extra copies ({total / 1e9:.2f} GB)? data/raw/ keeps one. [y/N] ").strip().lower() in {"y", "yes"}:
            for c, _ in extra:
                c.unlink()
            print("deleted")

if __name__ == "__main__":
    sys.exit(main())
