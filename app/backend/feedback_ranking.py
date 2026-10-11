"""Relevance feedback (Rocchio-style) for the demo rankers.

The user's query vector is moved toward the jobs they marked "good" and away from jobs they marked
"bad", then results are re-scored. This adapts one session's ranking; it does not retrain any model
and it makes no claim of better accuracy. Constants are deliberately visible and untuned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

LIKED_WEIGHT = 1.0       # pull toward the mean of liked jobs, relative to the original query weight of 1.0
DISLIKED_WEIGHT = 0.5    # push away from the mean of disliked jobs
VERSION = "rocchio-v1-liked1.0-disliked0.5"


def adjusted_dense_query(query: np.ndarray, liked: np.ndarray | None, disliked: np.ndarray | None) -> np.ndarray:
    """Dense (BGE) query adjusted by liked/disliked job vectors; result is re-normalized."""
    adjusted = np.asarray(query, dtype=np.float32).copy()
    if liked is not None and len(liked):
        adjusted += LIKED_WEIGHT * np.asarray(liked, dtype=np.float32).mean(axis=0)
    if disliked is not None and len(disliked):
        adjusted -= DISLIKED_WEIGHT * np.asarray(disliked, dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(adjusted))
    return adjusted / norm if norm > 0 else np.asarray(query, dtype=np.float32)


def adjusted_sparse_query(query: Any, job_matrix: Any, liked_rows: Sequence[int], disliked_rows: Sequence[int]) -> Any:
    """Sparse (TF-IDF) query adjusted by the mean TF-IDF vectors of liked/disliked jobs."""
    adjusted = query.copy()
    if liked_rows:
        adjusted = adjusted + LIKED_WEIGHT * sparse.csr_matrix(job_matrix[list(liked_rows)].mean(axis=0))
    if disliked_rows:
        adjusted = adjusted - DISLIKED_WEIGHT * sparse.csr_matrix(job_matrix[list(disliked_rows)].mean(axis=0))
    return sparse.csr_matrix(adjusted)


def top_ranked(scores: np.ndarray, indices: Sequence[int], job_ids: Sequence[str], limit: int) -> list[tuple[int, float]]:
    """(index, score) pairs, best first; ties broken by job id for determinism.

    `scores[i]` belongs to job index `indices[i]`. Only the best few candidates are fully sorted.
    """
    count = len(scores)
    if count == 0:
        return []
    keep = min(count, max(limit * 4, limit + 50))
    if keep < count:
        shortlist = np.argpartition(-scores, keep - 1)[:keep]
        # also keep everything tied with the cutoff score so ties are broken by job id, not by position
        cutoff = scores[shortlist].min()
        shortlist = np.union1d(shortlist, np.flatnonzero(scores >= cutoff))
    else:
        shortlist = np.arange(count)
    order = sorted(((float(scores[i]), int(indices[i])) for i in shortlist), key=lambda pair: (-pair[0], job_ids[pair[1]]))
    return [(index, score) for score, index in order[:limit]]
