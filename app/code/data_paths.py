"""Where the app's git-ignored data lives: app/data, or $PREDICTLEADS_DATA_DIR when set (for example one data folder shared with analysis/)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_ENV = "PREDICTLEADS_DATA_DIR"


def data_dir() -> Path:
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else PROJECT_ROOT / "data"


DATA_DIR = data_dir()
