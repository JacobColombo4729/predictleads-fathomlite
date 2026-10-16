"""Where the combined model's inputs and outputs live, and the import paths to the code it reuses.

combined/ imports the analysis code (analysis/src, as `src`) and Jesse's text-match code (app/code) unchanged; nothing
there imports combined/. Data folders come from fetch.paths, like everywhere else.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "analysis", REPO_ROOT / "app" / "code"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))  # appended, so nothing here shadows a module already on the path

from fetch.paths import DATA_DIR  # noqa: E402

BUNDLE_DIR = DATA_DIR / "clean" / "combined_v1"  # the combined model's bundle (git-ignored)
CORPUS_DIR = BUNDLE_DIR / "corpus"
SLICE_DIR = BUNDLE_DIR / "corpus_slice"  # a small corpus for timing the encoder, never served
ML_RANK_DIR = DATA_DIR / "clean" / "ml_rank"  # ML_Rank section 13's hand-off
