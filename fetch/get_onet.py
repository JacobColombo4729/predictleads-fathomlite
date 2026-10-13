"""Download the O*NET database tables the project uses into data/external/onet/ (for EDA_SDA_3 section 6 and ML_Fit).

Usage, from the repository root (.venv activated):
    python -m fetch.get_onet            -> download release 31.0 and unpack the tables the project uses
    python -m fetch.get_onet --force    -> download again even if the tables are already there
    python -m fetch.get_onet --yes      -> don't ask before downloading (the disk-space check still runs)

O*NET 31.0 Database (August 2026 release), tab-delimited text, from the O*NET Resource Center. It's public, so no
account or key is needed. License: Creative Commons Attribution 4.0 (https://www.onetcenter.org/license_db.html):
credit "the O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration (USDOL/ETA)" and
say what was changed; SOURCE.json in the folder carries the credit. If the tables are already there, it does nothing.
Otherwise the zip goes to data/external/ (git-ignored), only the tables below are unpacked, and the zip is deleted.
Prints file names and sizes only, never contents.

Release 31.0 has no "Skills" or "Career Changers Matrix" file under those names: skills come as Essential Skills,
Transferable Skills and Software Skills, and related occupations (what the Career Changers Matrix described) as
Related Occupations.
"""
import argparse
import json
import time
import zipfile
from datetime import date
from pathlib import PurePosixPath

import requests
from tqdm import tqdm

from fetch.downloads import confirm_space
from fetch.paths import DATA_DIR, external

RELEASE = "31.0"
URL = f"https://www.onetcenter.org/dl_files/database/db_{RELEASE.replace('.', '_')}_text.zip"
ONET_DIR = external("onet")
CREDIT = (f"O*NET {RELEASE} Database, U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), "
          "used under the CC BY 4.0 license; only the listed tables were kept, unchanged")
# Tables unpacked (about 10 MB). Occupations and their titles join listings on the O*NET-SOC code they already carry.
KEEP = {
    "Occupation Data.txt",        # code, title, description
    "Essential Skills.txt",       # skill ratings per occupation (importance, level)
    "Transferable Skills.txt",    # skills that carry across occupations: the career-changer view
    "Software Skills.txt",        # named tools and technologies per occupation (the old Technology Skills)
    "Job Zones.txt",              # education and experience level per occupation: an accessibility prior
    "Job Zone Reference.txt",     # what each job zone means
    "Related Occupations.txt",    # occupations related to each one (what the Career Changers Matrix gave)
    "Content Model Reference.txt",  # names of the skill and element IDs
    "Scales Reference.txt",       # what each rating scale means
    "Read Me.txt",
}
REQUIRED = {"Occupation Data.txt", "Essential Skills.txt", "Transferable Skills.txt", "Job Zones.txt", "Related Occupations.txt"}


def present():
    """The required tables already unpacked under ONET_DIR."""
    return {p.name for p in ONET_DIR.glob("*.txt")} & REQUIRED if ONET_DIR.exists() else set()


def download(target, yes=False):
    """Stream the zip to `target` after the space check (a failed download is deleted and must be rerun)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with requests.get(URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            # The zip, plus the tables it unpacks to (checked exactly before unpacking)
            confirm_space(2 * (total or 20_000_000), target.parent, f"downloading the O*NET {RELEASE} tables (zip + unpacked)", yes=yes)
            with open(partial, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def unpack(zip_path):
    """Unpack the KEEP tables flat into ONET_DIR; refuse unsafe paths. Returns {table: bytes}."""
    written = {}
    with zipfile.ZipFile(zip_path) as z:
        size = sum(i.file_size for i in z.infolist() if PurePosixPath(i.filename).name in KEEP)
        confirm_space(size, ONET_DIR, "unpacking the O*NET tables", yes=True)
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.filename.startswith(("/", "\\")) or ".." in parts:
                raise ValueError(f"unsafe path in the zip: {info.filename}")
            name = PurePosixPath(info.filename).name
            if info.is_dir() or name not in KEEP:
                continue
            out = ONET_DIR / name
            with z.open(info) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            written[name] = out.stat().st_size
    return written


def ensure(force=False, yes=False):
    """Download and unpack the tables unless they're already there. Returns ONET_DIR."""
    if present() == REQUIRED and not force:
        print(f"already there: {ONET_DIR.relative_to(DATA_DIR.parent)} has the O*NET {RELEASE} tables (use --force to download again)")
        return ONET_DIR
    ONET_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    zip_path = ONET_DIR.parent / "onet.zip"
    download(zip_path, yes=yes)
    try:
        written = unpack(zip_path)
    finally:
        zip_path.unlink(missing_ok=True)
    missing = REQUIRED - written.keys()
    if missing:
        raise SystemExit(f"the download is missing {sorted(missing)}: the O*NET file names may have changed in this release")
    (ONET_DIR / "SOURCE.json").write_text(json.dumps({
        "dataset": "O*NET Database", "release": RELEASE, "url": URL, "license": "CC BY 4.0", "credit": CREDIT,
        "downloaded": date.today().isoformat(), "files": {k: written[k] for k in sorted(written)},
    }, indent=1))
    for name in sorted(written):
        print(f"  {name:<28} {written[name] / 1e6:6.2f} MB")
    print(f"saved {len(written)} tables to {ONET_DIR.relative_to(DATA_DIR.parent)} in {time.time() - start:.0f}s")
    return ONET_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the tables are already there")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading (the space check still runs)")
    args = parser.parse_args()
    ensure(args.force, args.yes)


if __name__ == "__main__":
    main()
