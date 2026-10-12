"""Download the PredictLeads export files from S3 into data/raw/, so every script can read them locally.

Usage, from the repository root (.venv activated; needs AWS access, README steps 7-8):
    python -m data_sources.get_predictleads              -> both files: companies (~0.25 GB) and job openings (~19 GB)
    python -m data_sources.get_predictleads companies    -> one file (companies or job_openings)
    python -m data_sources.get_predictleads --check      -> what's in data/raw/ and what's on S3 (sizes only)
    python -m data_sources.get_predictleads --yes        -> skip the question (the disk-space check still runs)

Optional: without a local copy, every reader streams from S3 instead (analysis/src/pl_data, stream_full and the
app's pipeline in app/code/). With one, they read data/raw/ and skip the network.

Files already complete in data/raw/ are skipped. Before downloading it says how much space is needed and free and asks
to proceed; if it doesn't fit (with 2 GB to spare), it stops. A file gets its real name only once it's complete and
matches S3's size, so a file in data/raw/ is always whole. Prints file names and sizes only.
"""
import argparse
import sys

from tqdm import tqdm

from data_sources.downloads import confirm_space
from data_sources.paths import RAW_DIR
from data_sources.predictleads import RAW_FILES, connect, raw_file

CHUNK = 8 << 20  # 8 MB


def download(s3, bucket, prefix, name, size):
    """Stream one file into data/raw/<name>.part, check its size, then give it its real name."""
    part = RAW_DIR / (name + ".part")
    body = s3.get_object(Bucket=bucket, Key=prefix + name)["Body"]
    with open(part, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=name) as bar:
        for chunk in iter(lambda: body.read(CHUNK), b""):
            f.write(chunk)
            bar.update(len(chunk))
    if part.stat().st_size != size:
        part.unlink()
        raise SystemExit(f"{name}: the download was incomplete; run it again")
    part.replace(RAW_DIR / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="companies and/or job_openings (default: both)")
    parser.add_argument("--check", action="store_true", help="show what's local and on S3, download nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading (the space check still runs)")
    args = parser.parse_args()
    unknown = set(args.files) - set(RAW_FILES)
    if unknown:
        parser.error(f"unknown file {sorted(unknown)}: choose from {list(RAW_FILES)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s3, bucket, prefix = connect(list_files=False)
    todo = []
    for key in args.files or list(RAW_FILES):
        name = RAW_FILES[key]
        size = s3.head_object(Bucket=bucket, Key=prefix + name)["ContentLength"]
        local = raw_file(name)
        done = local is not None and local.stat().st_size == size
        print(f"{name}: S3 {size / 1e9:.2f} GB; data/raw/: {'complete' if done else 'not downloaded'}")
        if not done:
            todo.append((name, size))
    if args.check or not todo:
        return
    confirm_space(sum(size for _, size in todo), RAW_DIR, f"downloading {', '.join(n for n, _ in todo)}", yes=args.yes)
    for name, size in todo:
        download(s3, bucket, prefix, name, size)
        print(f"{name}: downloaded")


if __name__ == "__main__":
    sys.exit(main())
