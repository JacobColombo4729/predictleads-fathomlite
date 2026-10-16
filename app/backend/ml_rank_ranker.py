"""ML_Rank's priority queue as an app ranker (PREDICTLEADS_API_RANKER=mlrank; started by run_jacob.ps1 / .sh).

The same queue the analysis backtests (analysis/notebooks/ML_Rank.ipynb 8d-8e): four factors, each rated 0-100, times
ML_Rank's tuned weights:
  - fit: how well the listing matches the user. Phase 1 stand-in: the app's text match (the RRF ranker's score), rated so
    the best match is 100, the way ML_Fit's fit rating makes the person's best role 100. ML_Fit's own model replaces it
    in phase 2.
  - still open, accessibility, pay: from ML_Rank's snapshot (data/clean/ml_rank/, written by ML_Rank 13a), rated over
    every app listing once at start-up, as the backtest rates them over the month's listings (pay within its role).
Listings ML_Rank hides by default (hidden occupations, manager levels, likely ghosts) are left out; a listing without an
ML_Rank score is kept with neutral ratings (50). The text match picks the candidates, so the queue reorders the listings
that match the user. Everything else (feedback, Compass, similar-to-liked) comes from the text-match ranker underneath.
"""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # data/, analysis/ and fetch/ are shared at the top
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.append(str(REPO_ROOT / "analysis"))  # last, so nothing in analysis/ shadows the app's own modules
from fetch.paths import DATA_DIR  # noqa: E402  shared with analysis/: one definition of where data lives
from src import rank_model as rm  # noqa: E402  analysis/src: the same ratings and weights as the backtest

HANDOFF_DIR = DATA_DIR / "clean" / "ml_rank"
DEPTH = 200  # how many text-match candidates the queue reorders (the RRF ranker returns at most 100)
LISTING_FACTORS = ["still_open", "accessibility", "pay"]
COMPANY_CAP = 2  # listings per company among the results, before the rest (rm.cap_per_company)

log = logging.getLogger("predictleads.ml_rank")


class MlRankRankingBackend:
    ranker_id = "ml_rank_queue"
    version = "ml_rank-snapshot-v1-fit-stand-in"
    score_type = "priority_score_0_100"
    supports_exploration = False

    def __init__(self, base: Any, scores: pd.DataFrame, config: dict[str, Any]):
        self.base = base
        self.weights = rm._weights(config["weights"])
        hidden_levels = set(config.get("hidden_levels", []))
        job_ids = pd.Index([str(j) for j in base.job_ids])
        scores = scores.assign(id=scores["id"].astype(str)).drop_duplicates("id").set_index("id")
        hide = scores["hidden_occupation"].astype(bool) | scores["likely_ghost"].astype(bool) | scores["level"].isin(hidden_levels)
        self.hidden_ids = set(scores.index[hide]) & set(job_ids)
        app = scores.reindex(job_ids)  # one row per app listing, in the app's order; unscored listings are blank
        pool = pd.DataFrame({"role_bucket": app["role_bucket"].fillna("unscored"), "p_still_open": app["p_still_open"],
                             "accessibility": app["accessibility"], "pay": app["pay"]}, index=job_ids)
        ratings = rm.listing_ratings(pool)
        ratings.loc[app["p_still_open"].isna().to_numpy()] = 50.0  # no ML_Rank score: neutral on every listing factor
        self.listing_ratings = ratings
        # Titles that read as no tech role ("Data Entry", "Data Contributor") get no fit, whatever their occupation code,
        # and at most COMPANY_CAP listings per company come first (src.career_switch, src.rank_model; 2026-10-16)
        from src import career_switch as cs

        jobs = getattr(base, "jobs", None) or []
        sources = {str(j.job_id): j.source for j in jobs}
        self.not_tech = {i for i, s in sources.items()
                         if cs.NOT_A_TECH_ROLE.search(str(s.get("title_searchable") or s.get("title_raw") or ""))}
        self.company = {i: str((s.get("company_reference") or {}).get("id") or "") or None for i, s in sources.items()}
        self.matched = int(app["p_still_open"].notna().sum())
        log.info("ML_Rank scores for %d of %d app listings; %d hidden by default", self.matched, len(job_ids), len(self.hidden_ids))

    @classmethod
    def from_handoff(cls, base: Any, directory: Path = HANDOFF_DIR) -> "MlRankRankingBackend":
        missing = [f for f in ("listing_scores.parquet", "queue_config.json") if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(f"ML_Rank's hand-off is missing {', '.join(missing)}: run ML_Rank section 13 first.")
        scores = pd.read_parquet(directory / "listing_scores.parquet",
                                 columns=["id", "role_bucket", "p_still_open", "accessibility", "pay", "level",
                                          "hidden_occupation", "likely_ghost"])
        config = json.loads((directory / "queue_config.json").read_text(encoding="utf-8"))
        return cls(base, scores, config)

    def __getattr__(self, name: str) -> Any:
        # Feedback, Compass and "similar to a liked job" come from the text-match ranker underneath
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def rank(self, profile: Any, limit: int, candidate_indices: Sequence[int], **kwargs: Any) -> list[dict[str, Any]]:
        """The queue's order of the user's text-match candidates, best first; each result carries its breakdown."""
        kwargs.pop("explore", None)
        indices = [i for i in candidate_indices if str(self.base.job_ids[i]) not in self.hidden_ids]
        candidates = self.base.rank(profile, DEPTH, indices, **kwargs)
        if not candidates:
            return []
        ids = [str(c["job_id"]) for c in candidates]
        match = np.array([float(c["score"]) for c in candidates])
        fit = match / match.max() * 100 if match.max() > 0 else np.full(len(match), 50.0)
        fit = np.where([i in self.not_tech for i in ids], 0.0, fit)  # a title that reads as no tech role: no fit
        listing = self.listing_ratings.loc[ids]
        ratings = listing.assign(fit=fit)[rm.FACTORS]
        parts = ratings * self.weights
        total = parts.sum(axis=1).to_numpy()
        order = rm.cap_per_company(list(np.argsort(-total, kind="stable")), [self.company.get(i) for i in ids],
                                   cap=COMPANY_CAP, limit=limit)
        return [{"job_id": ids[i], "rank": rank, "score": float(total[i]),
                 "breakdown": {f: {"rating": round(float(ratings.iloc[i][f]), 1), "part": round(float(parts.iloc[i][f]), 2)}
                               for f in rm.FACTORS}}
                for rank, i in enumerate(order, 1)]
