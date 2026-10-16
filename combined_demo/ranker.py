"""The combined model as a ranker for the app (the same interface as app/backend/ml_rank_ranker.py).

The app hands a ranker a CandidateProfile, a text summary of the profile. The combined model needs the profile's entries
as saved (kind and dates), so the ranker reads them by profile_id from the demo's own session database, read-only. If a
profile can't be found it falls back to the summary's entries, read as undated jobs (and says so in the log).

Feedback (👍/👎, rated listings left out), the Compass and "similar to a job you liked" come from the RRF ranker
underneath, unchanged; ratings also reach the combined model's text match, since it calls that ranker.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from combined import VERSION
from combined.queue import LISTING_FACTORS

log = logging.getLogger("predictleads.combined")
FACTORS = ["fit", *LISTING_FACTORS]

ProfileLookup = Callable[[str], "Mapping[str, Any] | None"]


def sqlite_profiles(db_path: Path) -> ProfileLookup:
    """profile_id -> the saved profile (the app's `profiles` table), read-only."""
    def lookup(profile_id: str) -> Mapping[str, Any] | None:
        if not Path(db_path).exists():
            return None
        # closing(): sqlite3's own `with` only commits; it would leave a connection open on every request
        with closing(sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)) as db:
            row = db.execute("SELECT profile_json FROM profiles WHERE profile_id = ?", (profile_id,)).fetchone()
        return json.loads(row[0]) if row else None
    return lookup


class CombinedRankingBackend:
    ranker_id = "combined"
    version = VERSION
    score_type = "priority_score_0_100"
    supports_exploration = False

    def __init__(self, base: Any, model: Any, profiles: ProfileLookup):
        self.base, self.model, self.profiles = base, model, profiles

    def __getattr__(self, name: str) -> Any:
        # Feedback, Compass and "similar to a liked job" come from the text-match ranker underneath
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def saved(self, profile: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
        """(entries, desired roles) as saved; without a saved profile, its summary's entries as undated jobs."""
        data = self.profiles(profile.profile_id)
        if data is not None:
            return list(data.get("experience", [])), list(data.get("desired_roles", []))
        log.warning("profile %s not in the session database: reading its summary as undated jobs", profile.profile_id)
        return [{"kind": "job", "title": text, "description": ""} for text in profile.experience], list(profile.desired_roles)

    def rank(self, profile: Any, limit: int, candidate_indices: Sequence[int], **kwargs: Any) -> list[dict[str, Any]]:
        """The combined queue for this profile, best first; each result carries its factor breakdown and its role."""
        kwargs.pop("explore", None)
        kwargs.pop("rng", None)
        entries, desired = self.saved(profile)
        result = self.model.recommend(entries, lambda n, idx: self.base.rank(profile, n, idx, **kwargs),
                                      limit=limit, indices=list(candidate_indices), desired_roles=desired)
        return [{"job_id": row.job_id, "rank": rank, "score": float(row.score), "role": row.role,
                 "breakdown": {f: {"rating": round(float(row[f"rating: {f}"]), 1), "part": round(float(row[f"part: {f}"]), 2)}
                               for f in FACTORS}}
                for rank, (_, row) in enumerate(result.queue.iterrows(), 1)]
