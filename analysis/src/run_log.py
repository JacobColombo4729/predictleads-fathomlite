"""Run log: settings, seeds and metrics for every run, one CSV row per run. Never listing text.

Dependency-free (MLflow is not in requirements.txt). The file lives under the data folder from
`fetch.paths` (data/runs/run_log.csv, git-ignored with data/). Only short scalars are accepted, so
listing text, DataFrames and Series cannot get in: such values raise ValueError.

    from src.run_log import log_run
    log_run("ML_Fit", params={"model": "logreg", "C": 1.0}, seeds={"random_state": 42},
            metrics={"cv_mean": 0.81, "cv_std": 0.02, "baseline": 0.70})
"""
import csv
import datetime as dt
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from fetch import paths  # noqa: E402

MAX_TEXT = 40          # longest string accepted; listing text is far longer
FIXED = ["time", "run", "notebook"]


def default_path():
    """data/runs/run_log.csv under the data folder (PL_DATA_DIR respected)."""
    return paths.DATA_DIR / "runs" / "run_log.csv"


def _check(group, key, value):
    """Return a CSV-safe scalar, or raise ValueError if the value could be free text or a table."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "item") and getattr(value, "shape", None) == ():   # numpy scalar
        return _check(group, key, value.item())
    if isinstance(value, str):
        if len(value) > MAX_TEXT or "\n" in value or len(value.split()) > 6:
            raise ValueError(f"{group}[{key!r}] looks like free text; the run log takes settings and numbers only")
        return value
    raise ValueError(f"{group}[{key!r}] is a {type(value).__name__}; the run log takes scalars only")


def log_run(notebook, params=None, seeds=None, metrics=None, run=None, path=None):
    """Append one run. Columns are prefixed param_, seed_, metric_; new keys extend the header."""
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"),
           "run": _check("run", "run", run or ""), "notebook": _check("notebook", "notebook", notebook)}
    for group, prefix in ((params, "param_"), (seeds, "seed_"), (metrics, "metric_")):
        for k, v in (group or {}).items():
            if not isinstance(k, str) or len(k) > MAX_TEXT:
                raise ValueError(f"key {k!r} is not a short name")
            row[prefix + k] = _check(prefix.rstrip("_"), k, v)
    path = Path(path) if path else default_path()
    old = read_log(path)
    cols = list(FIXED)
    for r in old + [row]:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:   # rewrite so a widened header stays consistent
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old + [row])
    tmp.replace(path)
    return row


def read_log(path=None):
    """The log as a list of dicts (strings); empty if there is none yet."""
    path = Path(path) if path else default_path()
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
