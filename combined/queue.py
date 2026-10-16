"""The combined model's step 5: ML_Rank's priority queue over the candidate pool, with ML_Fit's role fit as its fit factor.

Each candidate's score is ML_Rank's sum (rank_model, 8d): four factors rated 0-100, times the tuned weights from
ML_Rank's hand-off (`queue_config.json`, normalised as rm._weights does):
  - fit: the person's fit rating for the listing's role (step 3: best role = 100). That is the fit ML_Rank's weights were
    tuned on. Optionally blended with the listing's text match: fit = (1 - a) x role fit + a x match, match rated so the
    pool's best match is 100; a = 0 by default, a stated choice that can't be tuned (no data links people to listings).
    A listing outside ML_Fit's roles (sales engineers, business and market research analysts) has no fit model: it gets a
    neutral 50 and says so;
  - still open, accessibility, pay: rm.listing_ratings over every visible serving listing, once at load, as the backtest
    rates them over the month's listings (pay within its role; a listing ML_Rank didn't score gets a neutral 50).
Each result carries its breakdown: every factor's rating and its part of the score (the parts add up to the score).

    queue = Queue.from_handoff(listings)                 listings: candidates.ListingIndex
    queue.rank(pool, ratings, limit=20, match_weight=0.0) -> DataFrame, best first
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from combined._paths import ML_RANK_DIR
from combined.candidates import Candidate, ListingIndex

NEUTRAL = 50.0
LISTING_FACTORS = ["still_open", "accessibility", "pay"]


class Queue:
    def __init__(self, listings: ListingIndex, listing_ratings: pd.DataFrame, weights: Mapping[str, float],
                 fit_roles: Sequence[str]):
        from src import rank_model as rm

        self.listings, self.listing_ratings = listings, listing_ratings
        self.weights = rm._weights(dict(weights))
        self.fit_roles = list(fit_roles)

    @classmethod
    def from_handoff(cls, listings: ListingIndex, handoff_dir: Path = ML_RANK_DIR) -> "Queue":
        config = json.loads((handoff_dir / "queue_config.json").read_text(encoding="utf-8"))
        scores = pd.read_parquet(handoff_dir / "listing_scores.parquet",
                                 columns=["id", "role_bucket", "p_still_open", "accessibility", "pay"])
        return cls.from_scores(listings, scores, config["weights"], config["fit_roles"])

    @classmethod
    def from_scores(cls, listings: ListingIndex, scores: pd.DataFrame, weights: Mapping[str, float],
                    fit_roles: Sequence[str]) -> "Queue":
        """Rate the listing factors over the visible listings (rm.listing_ratings); hidden ones are never ranked."""
        from src import rank_model as rm

        s = scores.assign(id=scores["id"].astype(str)).drop_duplicates("id").set_index("id").reindex(listings.job_ids)
        visible = np.asarray(listings.visible())
        pool = pd.DataFrame({"role_bucket": s["role_bucket"].fillna("unscored").to_numpy()[visible],
                             "p_still_open": s["p_still_open"].to_numpy()[visible],
                             "accessibility": s["accessibility"].to_numpy()[visible],
                             "pay": s["pay"].to_numpy()[visible]}, index=visible)
        ratings = rm.listing_ratings(pool)
        ratings.loc[pool["p_still_open"].isna().to_numpy()] = NEUTRAL  # no ML_Rank score: neutral on every listing factor
        return cls(listings, ratings, weights, fit_roles)

    def rank(self, pool: Sequence[Candidate], role_ratings: Mapping[str, float], limit: int = 20,
             match_weight: float = 0.0) -> pd.DataFrame:
        """The pool, best first, with each factor's rating and part. role_ratings: step 3's {role: rating, best = 100}."""
        if not pool:
            return pd.DataFrame(columns=["job_id", "role", "score"])
        idx = np.array([c.index for c in pool])
        match = np.array([c.match for c in pool], dtype=float)
        match_rating = match / match.max() * 100 if match.max() > 0 else np.full(len(pool), NEUTRAL)
        has_fit = np.array([c.role in self.fit_roles for c in pool])
        role_fit = np.array([float(role_ratings.get(c.role, NEUTRAL)) if c.role in self.fit_roles else NEUTRAL for c in pool])
        fit = (1 - match_weight) * role_fit + match_weight * match_rating
        ratings = self.listing_ratings.reindex(idx).fillna(NEUTRAL)
        table = pd.DataFrame({"job_id": [c.job_id for c in pool], "role": [c.role for c in pool],
                              "rating: fit": fit, **{f"rating: {f}": ratings[f].to_numpy() for f in LISTING_FACTORS}})
        for f in ["fit", *LISTING_FACTORS]:
            table[f"part: {f}"] = self.weights[f] * table[f"rating: {f}"]
        table["score"] = table[[f"part: {f}" for f in ["fit", *LISTING_FACTORS]]].sum(axis=1)
        table["fit source"] = np.where(has_fit, "role fit", "no fit model for this role (neutral 50)")
        table["role fit"], table["match"] = role_fit, match_rating
        table["found by"] = [", ".join(c.sources) for c in pool]
        table["index"] = idx
        order = np.argsort(-table["score"].to_numpy(), kind="stable")[:limit]
        return table.iloc[order].reset_index(drop=True)
