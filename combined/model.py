"""The combined model, end to end: a résumé to role fit (step 3), the candidate pool (step 4) and the queue (step 5).

    model = CombinedModel.load(job_ids)                  job_ids: the serving corpus's listings, in its order
    result = model.recommend(entries, rank, limit=20)    entries: the app's profile entries;
                                                         rank(limit, indices): the text matcher (Jesse's RRF ranker)
    result.roles    the person's roles: probability, score, rating, reasons (combined.role_fit)
    result.queue    the listings, best first, each with its factor breakdown (combined.queue)

Nothing here knows about the web app; combined_demo/ranker.py adapts it to the app's ranker interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from combined import VERSION
from combined.candidates import OVERALL, PER_ROLE, TOP_ROLES, ListingIndex, Rank, gather
from combined.queue import Queue
from combined.role_fit import RoleFit, RoleFitResult


@dataclass
class Recommendation:
    roles: RoleFitResult
    queue: pd.DataFrame
    pool_size: int


class CombinedModel:
    version = VERSION

    def __init__(self, role_fit: RoleFit, listings: ListingIndex, queue: Queue):
        self.role_fit, self.listings, self.queue = role_fit, listings, queue

    @classmethod
    def load(cls, job_ids: Sequence[str], bge: Any = None) -> "CombinedModel":
        listings = ListingIndex.from_handoff(job_ids)
        return cls(RoleFit.load(bge=bge), listings, Queue.from_handoff(listings))

    def recommend(self, entries: Sequence[Mapping[str, Any]], rank: Rank, limit: int = 20,
                  indices: Sequence[int] | None = None, top_roles: int = TOP_ROLES, match_weight: float = 0.0,
                  growth: Mapping[str, float] | None = None, growth_weight: float = 0.0,
                  overall: int = OVERALL, per_role: int = PER_ROLE, as_of: Any = None) -> Recommendation:
        roles = self.role_fit.roles(entries, growth=growth, growth_weight=growth_weight, as_of=as_of)
        pool = gather(rank, self.listings, roles.top(top_roles), indices, overall, per_role)
        return Recommendation(roles, self.queue.rank(pool, roles.ratings(), limit, match_weight), len(pool))
