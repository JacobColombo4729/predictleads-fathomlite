"""Share the trained models' hand-offs as one zip, so the apps run without rerunning the analysis notebooks.

Usage, from the repository root (.venv activated):
    python -m fetch.get_handoffs --pack           -> (the analysis owner, after a rerun) zip this machine's hand-offs
                                                     as VERSION into data/handoffs/, in parts under 1 GB
                                                     (handoffs_<version>.zip.001, .002, ...; --part-size MB)
    python -m fetch.get_handoffs --unpack         -> install the zip (or all its parts) in data/handoffs/
    python -m fetch.get_handoffs --unpack <zip>   -> install a zip from somewhere else (any one of its parts will do)
    python -m fetch.get_handoffs --check          -> what's installed here and what's in data/handoffs/ (sizes only)
    --yes skips the question (the disk-space check still runs); --force also replaces folders the notebooks wrote here

The three hand-offs, all from one run (the corpus is built from ML_Rank's scores, and the queue's weights were tuned on
ML_Fit's ratings, so they always travel as one set):
    data/clean/ml_fit/                 ML_Fit 9b: the trained role-fit layer
    data/clean/ml_rank/                ML_Rank section 13: listing scores and the queue's weights
    data/clean/combined_v1/corpus/     python -m combined.corpus build, then encode: the listings and their BGE vectors
Together they're everything the combined demo (combined_demo/) and Jacob's app (app/run_jacob.ps1) read from the
analysis. BGE comes from fetch.get_bge.

The project's AWS access is read-only, so the zip is shared by hand, through a private team channel only (the corpus
holds listing text): README step 9. Only the files the apps read are packed (the corpus's encoder chunks stay
behind). Inside the zip, INDEX.json lists every file's size and SHA-256. Unpacking extracts to
a staging folder, checks every file against the index, and replaces the three folders only when all of them are
complete. Each installed folder gets HANDOFF.json (its version), so a rerun skips what's current, and folders without
it (the notebooks' own output) are left alone unless --force. Prints file names and sizes only.
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

from fetch.downloads import confirm_space, gb
from fetch.paths import DATA_DIR, REPO_ROOT

VERSION = "v1"  # bump after a rerun, then --pack; Jesse's next --unpack replaces his copy
CLEAN_DIR = DATA_DIR / "clean"
HANDOFFS = {  # name in the zip: (folder here, files the readers need)
    "ml_fit": (CLEAN_DIR / "ml_fit", {"manifest.json", "role_scorer.joblib", "fit_layer.joblib"}),  # combined/role_fit.py
    "ml_rank": (CLEAN_DIR / "ml_rank", {"listing_scores.parquet", "queue_config.json"}),  # combined/_paths.py
    "combined_corpus": (CLEAN_DIR / "combined_v1" / "corpus",  # combined/_paths.py CORPUS_DIR
                        {"jobs_prepared_corpus.jsonl", "bge_embeddings.npy", "bge_embeddings_manifest.json"}),
}
ONLY_REQUIRED = {"combined_corpus"}  # its folder also keeps the encoder's chunks (bge_chunks/), a second copy of the
                                     # vectors no app reads: only the files the readers need are packed
PART_SIZE = 900_000_000  # the zip is split into parts of at most this (under the 1 GB upload limits); --part-size
ZIP_DIR = DATA_DIR / "handoffs"  # where --pack writes the zip and --unpack looks for it
INDEX = "INDEX.json"     # at the zip's root
MARKER = "HANDOFF.json"  # in each installed folder
STAGING = CLEAN_DIR / ".handoff_staging"


def zip_path(version=VERSION):
    return ZIP_DIR / f"handoffs_{version}.zip"


def shown(path):
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def local_files(folder):
    """{relative path: size} of a hand-off folder, without the marker."""
    if not folder.is_dir():
        return {}
    return {p.relative_to(folder).as_posix(): p.stat().st_size for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != MARKER}


def marker_version(folder):
    try:
        return json.loads((folder / MARKER).read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def current(name, entry, version):
    """Whether the folder here is that version, complete."""
    folder = HANDOFFS[name][0]
    return (marker_version(folder) == version
            and local_files(folder) == {rel: f["size"] for rel, f in entry["files"].items()})


def parts_of(base):
    """The split parts of a zip (base.001, base.002, ...), in order."""
    return sorted(p for p in base.parent.glob(base.name + ".*") if p.suffix[1:].isdigit())


def split(target, part_size):
    """Split the zip into numbered parts (removing the whole one) if it's bigger than part_size."""
    for old in parts_of(target):
        old.unlink()
    if target.stat().st_size <= part_size:
        return [target]
    parts = []
    with open(target, "rb") as src:
        while chunk := src.read(part_size):
            parts.append(target.with_name(f"{target.name}.{len(parts) + 1:03d}"))
            parts[-1].write_bytes(chunk)
    target.unlink()
    return parts


def joined(base):
    """The zip itself, or its parts joined into it; (path, whether it was joined here and should be removed)."""
    if base.is_file():
        return base, False
    parts = parts_of(base)
    if not parts:
        raise SystemExit(f"No zip at {shown(base)}: put the shared handoffs_<version>.zip (or all its .001, .002, ... "
                         f"parts) in {shown(ZIP_DIR)}, or pass its path.")
    numbers = [int(p.suffix[1:]) for p in parts]
    if numbers != list(range(1, len(parts) + 1)):
        raise SystemExit(f"Some parts of {base.name} are missing: there are {', '.join(p.suffix[1:] for p in parts)}.")
    confirm_space(sum(p.stat().st_size for p in parts), base.parent, f"joining the {len(parts)} parts", yes=True)
    partial = base.with_suffix(".joining")
    with open(partial, "wb") as dst:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, dst, 8 << 20)
    partial.replace(base)
    return base, True


def pack(yes=False, part_size=PART_SIZE):
    """Zip this machine's three hand-offs as VERSION, with INDEX.json, split into parts; refuses an incomplete set."""
    index = {"version": VERSION, "packed": date.today().isoformat(), "handoffs": {}}
    for name, (folder, required) in HANDOFFS.items():
        files = local_files(folder)
        missing = sorted(required - files.keys())
        if missing:
            raise SystemExit(f"{name}: {shown(folder)} is missing {', '.join(missing)}: finish that step first "
                             "(see this script's docstring).")
        if name in ONLY_REQUIRED:
            files = {rel: size for rel, size in files.items() if rel in required}
        index["handoffs"][name] = {"files": {rel: {"size": size, "sha256": sha256(folder / rel)}
                                             for rel, size in files.items()}}
        print(f"{name}: {len(files)} files, {gb(sum(files.values()))}")
    total = sum(f["size"] for h in index["handoffs"].values() for f in h["files"].values())
    target = zip_path()
    confirm_space(total, ZIP_DIR, f"packing the hand-offs, version {VERSION}, into {shown(target)}", yes=yes)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name, entry in index["handoffs"].items():
                for rel in entry["files"]:
                    z.write(HANDOFFS[name][0] / rel, f"{name}/{rel}")
            z.writestr(INDEX, json.dumps(index, indent=1))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    parts = split(target, part_size)
    for p in parts:
        print(f"  {shown(p)}  {gb(p.stat().st_size)}")
    print(f"packed version {VERSION} in {len(parts)} file(s): share {'it' if len(parts) == 1 else 'all of them'} "
          "through a private team channel only")
    return parts


def read_index(z):
    try:
        return json.loads(z.read(INDEX))
    except KeyError:
        raise SystemExit(f"This zip has no {INDEX}: it wasn't made by python -m fetch.get_handoffs --pack.") from None


def unpack(path=None, yes=False, force=False):
    """Check the zip against its index in staging, then swap all three folders in at once."""
    base = Path(path) if path else zip_path()
    if base.suffix[1:].isdigit():  # given one of the parts
        base = base.with_suffix("")
    path, temporary = joined(base)
    try:
        install(path, yes, force)
    finally:
        if temporary:
            path.unlink(missing_ok=True)


def install(path, yes, force):
    with zipfile.ZipFile(path) as z:
        index = read_index(z)
        version = index["version"]
        todo = {name: entry for name, entry in index["handoffs"].items() if not current(name, entry, version)}
        for name in index["handoffs"].keys() - todo.keys():
            print(f"{name}: version {version} already here")
        if not todo:
            return
        unknown = sorted(todo.keys() - HANDOFFS.keys())
        if unknown:
            raise SystemExit(f"This zip has hand-offs this code doesn't know ({', '.join(unknown)}): pull the latest code.")
        own = [name for name in todo if local_files(HANDOFFS[name][0]) and marker_version(HANDOFFS[name][0]) is None]
        if own and not force:
            raise SystemExit(f"{', '.join(own)}: these folders were written by the notebooks on this machine, so "
                             "nothing was replaced. Rerun with --force to replace them with the shared version.")
        need = sum(f["size"] for entry in todo.values() for f in entry["files"].values())
        confirm_space(need, CLEAN_DIR, f"unpacking the hand-offs, version {version} ({', '.join(todo)})", yes=yes)
        shutil.rmtree(STAGING, ignore_errors=True)
        try:
            for name, entry in todo.items():
                for rel, f in entry["files"].items():
                    parts = PurePosixPath(rel).parts
                    if rel.startswith(("/", "\\")) or ".." in parts or ":" in rel:
                        raise SystemExit(f"unsafe path in the zip: {name}/{rel}")
                    target = STAGING / name / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with z.open(f"{name}/{rel}") as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst, 8 << 20)
                    except KeyError:
                        raise SystemExit(f"{name}/{rel} is listed in the index but not in the zip: ask for a new one") from None
                    if target.stat().st_size != f["size"] or sha256(target) != f["sha256"]:
                        raise SystemExit(f"{name}/{rel} doesn't match the zip's index: ask for a new zip")
                print(f"{name}: checked {len(entry['files'])} files")
            for name in todo:  # every file checked: now replace the folders
                folder = HANDOFFS[name][0]
                (STAGING / name / MARKER).write_text(
                    json.dumps({"version": version, "installed": date.today().isoformat()}), encoding="utf-8")
                folder.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(folder, ignore_errors=True)
                (STAGING / name).replace(folder)
                print(f"{name}: version {version} in {shown(folder)}")
        finally:
            shutil.rmtree(STAGING, ignore_errors=True)


def check():
    for name, (folder, _) in HANDOFFS.items():
        files = local_files(folder)
        here = (f"version {marker_version(folder)}" if marker_version(folder) else "from the notebooks") if files else "none"
        print(f"{name}: here {here}, {gb(sum(files.values()))}")
    zips = sorted(ZIP_DIR.glob("handoffs_*.zip*")) if ZIP_DIR.is_dir() else []
    for p in zips:
        print(f"{shown(p)}: {gb(p.stat().st_size)}")
    if not zips:
        print(f"no zip in {shown(ZIP_DIR)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pack", action="store_true", help="zip this machine's hand-offs as VERSION")
    action.add_argument("--unpack", nargs="?", const="", metavar="ZIP", help="install a hand-offs zip")
    action.add_argument("--check", action="store_true", help="show what's here, change nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask first (the disk-space check still runs)")
    parser.add_argument("--force", action="store_true", help="also replace folders the notebooks wrote here")
    parser.add_argument("--part-size", type=int, default=PART_SIZE // 1_000_000, metavar="MB",
                        help=f"with --pack: the largest part (default {PART_SIZE // 1_000_000} MB)")
    args = parser.parse_args()
    if args.pack:
        pack(yes=args.yes, part_size=args.part_size * 1_000_000)
    elif args.check:
        check()
    else:
        unpack(args.unpack or None, yes=args.yes, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
