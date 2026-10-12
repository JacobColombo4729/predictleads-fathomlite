"""Download the public structured-resume dataset from Kaggle into data/external/resume_structured/ (for Profiles.ipynb).

Usage, from analysis/ (.venv activated):
    python -m src.get_resumes            -> download version 2 and unpack the tables the project uses
    python -m src.get_resumes --force    -> download again even if the tables are already there
Profiles.ipynb calls ensure() at the start, so running the notebook is enough.

Kaggle `suriyaganesh/resume-dataset-structured`, version 2 (2024-11-14), MIT license. It's public, so no Kaggle
account or API key is needed. The zip goes to data/external/ (git-ignored), only the tables the project uses are
unpacked, and the zip is deleted afterwards. The people table (names and contact details) is never unpacked: nothing
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

from src.eda_helpers import DATA_DIR

DATASET = "suriyaganesh/resume-dataset-structured"
VERSION = 2
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber={VERSION}"
RESUME_DIR = DATA_DIR / "external" / "resume_structured"
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


def download(target):
    """Stream the zip to `target` (resumes nothing: a failed download is deleted and must be rerun)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
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


def ensure(force=False):
    """Download and unpack the tables unless they're already there (Profiles.ipynb calls this first). Returns RESUME_DIR."""
    if present() == REQUIRED and not force:
        print(f"already there: {RESUME_DIR.relative_to(DATA_DIR.parent)} has {', '.join(sorted(REQUIRED))} (use --force to download again)")
        return RESUME_DIR
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "external" / "resume_structured.zip"
    start = time.time()
    download(zip_path)
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
          f"the zip was deleted; the people table (names, contacts) was not unpacked")
    return RESUME_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the tables are already there")
    ensure(parser.parse_args().force)


if __name__ == "__main__":
    main()
