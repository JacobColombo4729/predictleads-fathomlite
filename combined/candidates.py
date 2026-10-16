"""The combined model's step 4: which listings the queue ranks for a person.

A career changer's résumé reads like their old field, so the text match alone can miss the very roles the trained layer
recommends. The pool is the union of:
  - the top text matches overall, and
  - the top text matches within each of the person's top roles (from step 3),
with the listings ML_Rank hides by default left out (hidden occupations, likely ghosts, the hidden levels), as `mlrank` does.

    listings = ListingIndex.from_handoff(job_ids)          each corpus listing's role bucket and hide flag, from ML_Rank
    pool = gather(rank, listings, top_roles)               -> list[Candidate]

`rank(limit, indices)` is any text matcher that returns [{"job_id", "rank", "score"}] best first for the listings it is
given; in the app it is Jesse's RRF ranker, unchanged (`RrfRankingBackend.rank(profile, limit, indices)`, which returns
at most 100). A candidate's `match` is its text-match score from the overall call when it came back there, else from its
role's call (a role call ranks fewer listings, so its scores run higher; step 5 uses `match` only as an optional tie-break).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from combined._paths import ML_RANK_DIR

OVERALL = 100   # text matches taken overall (the RRF ranker returns at most 100 per call)
PER_ROLE = 100  # text matches taken within each top role
TOP_ROLES = 3

Rank = Callable[[int, Sequence[int]], list[dict[str, Any]]]


@dataclass
class ListingIndex:
    """One entry per corpus listing, in the corpus order: its role bucket (None if ML_Rank has no score) and whether
    ML_Rank hides it by default."""
    job_ids: list[str]
    roles: np.ndarray
    hidden: np.ndarray

    @classmethod
    def from_handoff(cls, job_ids: Sequence[str], handoff_dir: Path = ML_RANK_DIR) -> "ListingIndex":
        missing = [f for f in ("listing_scores.parquet", "queue_config.json") if not (handoff_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"ML_Rank's hand-off is missing {', '.join(missing)}: run ML_Rank section 13 first.")
        scores = pd.read_parquet(handoff_dir / "listing_scores.parquet",
                                 columns=["id", "role_bucket", "level", "hidden_occupation", "likely_ghost"])
        hidden_levels = set(json.loads((handoff_dir / "queue_config.json").read_text(encoding="utf-8")).get("hidden_levels", []))
        return cls.from_scores([str(j) for j in job_ids], scores, hidden_levels)

    @classmethod
    def from_scores(cls, job_ids: Sequence[str], scores: pd.DataFrame, hidden_levels: Iterable[str] = ()) -> "ListingIndex":
        s = scores.assign(id=scores["id"].astype(str)).drop_duplicates("id").set_index("id").reindex(list(job_ids))
        hide = (s["hidden_occupation"].eq(True) | s["likely_ghost"].eq(True) | s["level"].isin(set(hidden_levels))).to_numpy()
        roles = s["role_bucket"].astype(object).where(s["role_bucket"].notna(), None).to_numpy()
        return cls(list(job_ids), roles, hide)

    def visible(self, indices: Sequence[int] | None = None) -> list[int]:
        idx = range(len(self.job_ids)) if indices is None else indices
        return [i for i in idx if not self.hidden[i]]

    def in_role(self, role: str, indices: Sequence[int]) -> list[int]:
        return [i for i in indices if self.roles[i] == role]


@dataclass
class Candidate:
    index: int
    job_id: str
    role: str | None
    match: float
    overall_rank: int | None = None
    role_rank: int | None = None
    sources: list[str] = field(default_factory=list)


def gather(rank: Rank, listings: ListingIndex, top_roles: Sequence[str], indices: Sequence[int] | None = None,
           overall: int = OVERALL, per_role: int = PER_ROLE) -> list[Candidate]:
    """The union of the top `overall` text matches and the top `per_role` within each role of `top_roles`, among the
    visible listings of `indices` (all listings if None). Overall matches first, in their order, then each role's."""
    position = {job_id: i for i, job_id in enumerate(listings.job_ids)}
    eligible = listings.visible(indices)
    pool: dict[int, Candidate] = {}
    for item in rank(overall, eligible):
        i = position[str(item["job_id"])]
        pool[i] = Candidate(i, listings.job_ids[i], listings.roles[i], float(item["score"]), overall_rank=int(item["rank"]),
                            sources=["overall"])
    for role in top_roles:
        in_role = listings.in_role(role, eligible)
        if not in_role:
            continue
        for item in rank(per_role, in_role):
            i = position[str(item["job_id"])]
            if i in pool:
                pool[i].role_rank = pool[i].role_rank or int(item["rank"])
                pool[i].sources.append(f"role: {role}")
            else:
                pool[i] = Candidate(i, listings.job_ids[i], listings.roles[i], float(item["score"]), role_rank=int(item["rank"]),
                                    sources=[f"role: {role}"])
    return list(pool.values())
