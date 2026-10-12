"""The repository's shared folders: one definition for analysis/ and app/.

Code never writes out where data lives: it builds paths from these names (DATA_DIR / "matching" / ..., external("x")).
`PL_DATA_DIR` in the environment points DATA_DIR somewhere else (tests on synthetic data), for both halves.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"                                   # settings and keys, git-ignored
DATA_DIR = Path(os.environ.get("PL_DATA_DIR") or REPO_ROOT / "data")  # git-ignored; never read by Claude
PRETRAINED_DIR = REPO_ROOT / "pretrained"                       # downloaded models (the BGE model), git-ignored

# The standard folders, by where the data comes from
RAW_DIR = DATA_DIR / "raw"            # the PredictLeads export files (fetch.get_predictleads)
EXTERNAL_DIR = DATA_DIR / "external"  # every outside dataset, one folder each: external("<name>")


def external(name):
    """The standard folder of an outside dataset: data/external/<name>/."""
    return EXTERNAL_DIR / name
