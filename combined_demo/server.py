"""The combined demo's backend: Jesse's app (app/backend/api.py, unchanged) built with the combined model as its ranker.

    python -m uvicorn combined_demo.server:app --port 8002      (from the repository root; run.ps1 / run.sh do this)

create_app() already takes a repository and a ranker, so nothing in app/ changes: the jobs are the combined model's serving
corpus (combined/corpus.py), the ranker is CombinedRankingBackend over Jesse's RRF ranker, and the session database is
the demo's own (data/combined_demo/sessions.sqlite3 unless PREDICTLEADS_API_DB_PATH says otherwise), so a profile made
here never mixes with the other two apps'. Loading takes a few minutes (the RRF ranker fits TF-IDF over the corpus).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from combined._paths import CORPUS_DIR, DATA_DIR  # noqa: E402

os.environ.setdefault("PREDICTLEADS_API_DB_PATH", str(DATA_DIR / "combined_demo" / "sessions.sqlite3"))
os.environ.setdefault("PREDICTLEADS_EVENTS_DB_PATH", str(DATA_DIR / "combined_demo" / "events.sqlite3"))
os.environ["PREDICTLEADS_API_RANKER"] = "combined"  # what /snapshots/current reports comes from the ranker itself
Path(os.environ["PREDICTLEADS_API_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)

from backend.api import JobRepository, Settings, create_app  # noqa: E402
from backend.rrf_ranker import RrfRankingBackend  # noqa: E402

from combined.model import CombinedModel  # noqa: E402
from combined_demo.ranker import CombinedRankingBackend, sqlite_profiles  # noqa: E402

log = logging.getLogger("predictleads.combined")


def build():
    started = time.time()
    settings = Settings.from_environment()
    jobs = JobRepository.from_jsonl(CORPUS_DIR / "jobs_prepared_corpus.jsonl")
    base = RrfRankingBackend(jobs.jobs, CORPUS_DIR)
    model = CombinedModel.load(base.job_ids)
    ranker = CombinedRankingBackend(base, model, sqlite_profiles(settings.db_path))
    log.warning("combined demo ready: %d listings, %d hidden by default, loaded in %.0fs",
                len(base.job_ids), int(model.listings.hidden.sum()), time.time() - started)
    return create_app(settings=settings, repository=jobs, ranker=ranker)


app = build()
