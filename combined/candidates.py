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
PER_ROLE = 30   # text matches taken within each top role: only good matches, so the queue reorders relevant listings
                # (100 in v1 let the queue promote a role's weak matches above its strong ones)
TOP_ROLES = 3

Rank = Callable[[int, Sequence[int]], list[dict[str, Any]]]


@dataclass
class ListingIndex:
    """One entry per corpus listing, in the corpus order: its role bucket, whether ML_Rank hides it by default, and
    (with titles) whether its title reads as no tech role, and its company. A listing's role is checked against its
    title (career_switch.listing_role): a "Data Entry" or "Data Contributor" title has no role, whatever the occupation
    code says; a title Profiles' rules place in another bucket moves there. None also when ML_Rank has no score."""
    job_ids: list[str]
    roles: np.ndarray
    hidden: np.ndarray
    not_tech: np.ndarray | None = None   # the title reads as no tech role (ranked with no fit)
    companies: np.ndarray | None = None  # the company id, for the per-company cap

    @classmethod
    def from_handoff(cls, job_ids: Sequence[str], handoff_dir: Path = ML_RANK_DIR, titles: Sequence[str] | None = None,
                     companies: Sequence[str | None] | None = None) -> "ListingIndex":
        missing = [f for f in ("listing_scores.parquet", "queue_config.json") if not (handoff_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"ML_Rank's hand-off is missing {', '.join(missing)}: run ML_Rank section 13 first.")
        scores = pd.read_parquet(handoff_dir / "listing_scores.parquet",
                                 columns=["id", "role_bucket", "level", "hidden_occupation", "likely_ghost"])
        hidden_levels = set(json.loads((handoff_dir / "queue_config.json").read_text(encoding="utf-8")).get("hidden_levels", []))
        return cls.from_scores([str(j) for j in job_ids], scores, hidden_levels, titles, companies)

    @classmethod
    def from_scores(cls, job_ids: Sequence[str], scores: pd.DataFrame, hidden_levels: Iterable[str] = (),
                    titles: Sequence[str] | None = None, companies: Sequence[str | None] | None = None) -> "ListingIndex":
        from src import career_switch as cs

        s = scores.assign(id=scores["id"].astype(str)).drop_duplicates("id").set_index("id").reindex(list(job_ids))
        hide = (s["hidden_occupation"].eq(True) | s["likely_ghost"].eq(True) | s["level"].isin(set(hidden_levels))).to_numpy()
        roles = s["role_bucket"].astype(object).where(s["role_bucket"].notna(), None).to_numpy()
        not_tech = None
        if titles is not None:
            not_tech = np.array([bool(cs.NOT_A_TECH_ROLE.search(str(t or ""))) for t in titles])
            roles = np.array([cs.listing_role(t, r) if r is not None else None for t, r in zip(titles, roles)], dtype=object)
        return cls(list(job_ids), roles, hide, not_tech, None if companies is None else np.asarray(companies, dtype=object))

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


def titles_and_companies(jobs: Sequence[Any]) -> tuple[list[str], list[str | None]]:
    """Each prepared listing's title and company id (the app's JobRecord.source), in order."""
    titles, companies = [], []
    for job in jobs:
        source = job.source
        titles.append(str(source.get("title_searchable") or source.get("title_raw") or ""))
        ref = source.get("company_reference") or {}
        companies.append(str(ref.get("id")) if isinstance(ref, dict) and ref.get("id") else None)
    return titles, companies


def overall_matches(rank: Rank, listings: ListingIndex, indices: Sequence[int] | None = None,
                    overall: int = OVERALL) -> list[Candidate]:
    """The top `overall` text matches among the visible listings of `indices`, best first."""
    position = {job_id: i for i, job_id in enumerate(listings.job_ids)}
    out = []
    for item in rank(overall, listings.visible(indices)):
        i = position[str(item["job_id"])]
        out.append(Candidate(i, listings.job_ids[i], listings.roles[i], float(item["score"]), overall_rank=int(item["rank"]),
                             sources=["overall"]))
    return out


def role_mix(matches: Sequence[Candidate], roles: Sequence[str]) -> dict[str, float]:
    """Each role's share of the text matches that fall in `roles` (0 for all if none do): what the person's whole
    profile, read as text, points to."""
    counts = {r: 0 for r in roles}
    for c in matches:
        if c.role in counts:
            counts[c.role] += 1
    total = sum(counts.values())
    return {r: (n / total if total else 0.0) for r, n in counts.items()}


def gather(rank: Rank, listings: ListingIndex, top_roles: Sequence[str], indices: Sequence[int] | None = None,
           overall: int = OVERALL, per_role: int = PER_ROLE, matches: Sequence[Candidate] | None = None) -> list[Candidate]:
    """The union of the top `overall` text matches (or the given `matches`, from overall_matches) and the top
    `per_role` within each role of `top_roles`, among the visible listings of `indices` (all listings if None).
    Overall matches first, in their order, then each role's."""
    position = {job_id: i for i, job_id in enumerate(listings.job_ids)}
    eligible = listings.visible(indices)
    found = overall_matches(rank, listings, indices, overall) if matches is None else matches
    pool: dict[int, Candidate] = {c.index: Candidate(c.index, c.job_id, c.role, c.match, c.overall_rank, c.role_rank,
                                                    list(c.sources)) for c in found}
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
