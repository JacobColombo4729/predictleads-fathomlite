"""Download the public structured-resume dataset from Kaggle into data/external/resume_structured/ (for Profiles.ipynb).

Usage, from the repository root (.venv activated):
    python -m fetch.get_resumes            -> download version 2 and unpack the tables the project uses
    python -m fetch.get_resumes --force    -> download again even if the tables are already there
    python -m fetch.get_resumes --yes      -> don't ask before downloading (the disk-space check still runs)
Profiles.ipynb calls ensure() at the start, so running the notebook is enough.

Kaggle `suriyaganesh/resume-dataset-structured`, version 2 (2024-11-14), MIT license. It's public, so no Kaggle
account or API key is needed. If the tables are already there, it does nothing. Otherwise the zip goes to
data/external/ (git-ignored), only the tables the project uses are unpacked, and the zip is deleted afterwards. The people table (names and contact details) is never unpacked: nothing
reads it. Prints file names and sizes only, never contents.
"""
import argparse
import json
import re
import time
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

from fetch.downloads import confirm_space
from fetch.paths import DATA_DIR, external

DATASET = "suriyaganesh/resume-dataset-structured"
VERSION = 2
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber={VERSION}"
RESUME_DIR = external("resume_structured")
# Tables unpacked (file names in the zip carry a number prefix, e.g. 04_experience.csv). Profiles reads experience and
# education; the skill tables are kept for the skills work in EDA_SDA_3 and ML_Fit.
KEEP = {"experience.csv", "education.csv", "person_skills.csv", "skills.csv", "abilities.csv"}
NEVER = {"people.csv"}  # names and contact details
REQUIRED = {"experience.csv", "education.csv"}


def table_name(path):
    """The table's name without its number prefix: 04_experience.csv -> experience.csv."""
    return re.sub(r"^\d+_", "", PurePosixPath(path).name.lower())


def present():
    """The required tables already unpacked under RESUME_DIR."""
    if not RESUME_DIR.exists():
        return set()
    return {table_name(p) for p in RESUME_DIR.rglob("*.csv")} & REQUIRED


def download(target, yes=False):
    """Stream the zip to `target` after the space check (resumes nothing: a failed download is deleted and must be rerun)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            # The zip, plus the tables it unpacks to (checked exactly before unpacking; about 3x the zip here)
            confirm_space(4 * (total or 200_000_000), target.parent, "downloading the Kaggle resume tables (zip + unpacked)", yes=yes)
            with open(partial, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def unpack(zip_path):
    """Unpack the KEEP tables flat into RESUME_DIR; refuse unsafe paths. Returns {table: bytes}."""
    written = {}
    with zipfile.ZipFile(zip_path) as z:
        size = sum(i.file_size for i in z.infolist() if table_name(i.filename) in KEEP)
        confirm_space(size, RESUME_DIR, "unpacking the resume tables", yes=True)
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.filename.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(f"unsafe path in the zip: {info.filename}")
            name = table_name(info.filename)
            if info.is_dir() or name in NEVER or name not in KEEP:
                continue
            out = RESUME_DIR / PurePosixPath(info.filename).name
            with z.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            written[name] = out.stat().st_size
    return written


def ensure(force=False, yes=False):
    """Download and unpack the tables unless they're already there (Profiles.ipynb calls this first). Returns RESUME_DIR."""
    if present() == REQUIRED and not force:
        print(f"already there: {RESUME_DIR.relative_to(DATA_DIR.parent)} has {', '.join(sorted(REQUIRED))} (use --force to download again)")
        return RESUME_DIR
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    zip_path = RESUME_DIR.parent / "resume_structured.zip"
    download(zip_path, yes=yes)
    try:
        written = unpack(zip_path)
    finally:
        zip_path.unlink(missing_ok=True)
    missing = REQUIRED - written.keys()
    if missing:
        raise SystemExit(f"the download is missing {sorted(missing)}: the dataset may have changed on Kaggle")
    (RESUME_DIR / "SOURCE.json").write_text(json.dumps({
        "dataset": DATASET, "version": VERSION, "license": "MIT", "downloaded": date.today().isoformat(),
        "files": {k: written[k] for k in sorted(written)}, "not_unpacked": sorted(NEVER),
    }, indent=1))
    for name in sorted(written):
        print(f"  {name:<18} {written[name] / 1e6:8.1f} MB")
    print(f"saved {len(written)} tables to {RESUME_DIR.relative_to(DATA_DIR.parent)} in {time.time() - start:.0f}s; "
          f"the people table (names, contacts) was not unpacked")
    return RESUME_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the tables are already there")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading (the space check still runs)")
    args = parser.parse_args()
    ensure(args.force, args.yes)


if __name__ == "__main__":
    main()
