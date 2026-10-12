"""Download the BGE text-embedding model into pretrained/bge-base-en-v1.5/ (for the app's ranker and ML_Fit).

Usage, from the repository root (.venv activated):
    python -m fetch.get_bge            -> download the pinned version unless it's already there
    python -m fetch.get_bge --force    -> download again even if it's already there
    python -m fetch.get_bge --yes      -> don't ask before downloading (the disk-space check still runs)

Hugging Face `BAAI/bge-base-en-v1.5`, MIT license. It's public, so no account or token is needed. The model name and
the pinned revision are read from app/code/bge_text.py, the one place they're defined: the corpus's saved embeddings
were made with that exact revision, and query embeddings only match them if the model is the same. Only the files
sentence-transformers loads are downloaded (safetensors weights, configs, tokenizer, pooling), not the ONNX or
duplicate PyTorch weights. Prints file names and sizes only.
"""
import argparse
import ast
import json
import time
from datetime import date
from pathlib import PurePath

from huggingface_hub import HfApi, snapshot_download

from fetch.downloads import confirm_space
from fetch.paths import PRETRAINED_DIR, REPO_ROOT

BGE_TEXT = REPO_ROOT / "app" / "code" / "bge_text.py"
BGE_DIR = PRETRAINED_DIR / "bge-base-en-v1.5"
ALLOW = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]
REQUIRED = {"config.json", "modules.json", "model.safetensors", "tokenizer.json"}


def pinned():
    """(BGE_ID, BGE_REVISION) as assigned in app/code/bge_text.py, read without importing the app."""
    found = {}
    for node in ast.parse(BGE_TEXT.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BGE_ID", "BGE_REVISION"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["BGE_ID"], found["BGE_REVISION"]


def present():
    """The required model files already in BGE_DIR."""
    return {p.name for p in BGE_DIR.iterdir()} & REQUIRED if BGE_DIR.is_dir() else set()


def ensure(force=False, yes=False):
    """Download the pinned model unless it's already there. Returns BGE_DIR."""
    if present() == REQUIRED and not force:
        print(f"already there: {BGE_DIR.relative_to(REPO_ROOT)} (use --force to download again)")
        return BGE_DIR
    model_id, revision = pinned()
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    files = [s for s in info.siblings if any(PurePath(s.rfilename).match(p) for p in ALLOW)]
    confirm_space(sum(s.size or 0 for s in files), BGE_DIR, f"downloading {model_id}", yes=yes)
    start = time.time()
    snapshot_download(model_id, revision=revision, local_dir=BGE_DIR, allow_patterns=ALLOW)
    missing = REQUIRED - present()
    if missing:
        raise SystemExit(f"the download is missing {sorted(missing)}: the model repository may have changed")
    (BGE_DIR / "SOURCE.json").write_text(json.dumps({
        "model": model_id, "revision": revision, "license": "MIT", "downloaded": date.today().isoformat(),
        "files": {s.rfilename: s.size for s in files},
    }, indent=1))
    for s in sorted(files, key=lambda s: -(s.size or 0))[:5]:
        print(f"  {s.rfilename:<28} {(s.size or 0) / 1e6:8.1f} MB")
    print(f"saved {len(files)} files to {BGE_DIR.relative_to(REPO_ROOT)} (revision {revision[:7]}) in {time.time() - start:.0f}s")
    return BGE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if it's already there")
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading (the space check still runs)")
    args = parser.parse_args()
    ensure(args.force, args.yes)


if __name__ == "__main__":
    main()
