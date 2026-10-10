"""Optional TF-IDF + BGE reciprocal-rank-fusion ranker for the demo API.

Enabled with PREDICTLEADS_API_RANKER=rrf and PREDICTLEADS_API_CORPUS_DIR pointing at a
`rank_candidates.py encode` / `export-serving` output directory (jobs_prepared_corpus.jsonl,
bge_embeddings.npy and its manifest). The API's jobs file must be that same corpus JSONL so row
order matches the matrix. torch / sentence-transformers are imported only when this ranker is
built, so the default TF-IDF backend needs neither. Scores are uncalibrated fusion signals.

With user feedback (jobs marked good or bad) both signals use a Rocchio-style adjusted query; see
feedback_ranking.py. Rated jobs are left out of the results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
for _path in (CODE_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import job_matching_engine as matching  # noqa: E402
from backend import feedback_ranking as feedback  # noqa: E402

DEPTH = 50
RRF_K = 60
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _default_encoder(model_dir: Path) -> tuple[Callable[[str], np.ndarray], Callable[[matching.CandidateProfile], str]]:
    import bge_text as bge  # lazy: keeps torch-free startup for the TF-IDF backend
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True, trust_remote_code=False)
    model.max_seq_length = bge.MAX_BGE_TOKENS

    def query_text(profile: matching.CandidateProfile) -> str:
        keywords = list(profile.preferences.get("keywords", [])) if isinstance(profile.preferences, dict) else []
        arrangements = list(profile.preferences.get("work_arrangement", [])) if isinstance(profile.preferences, dict) else []
        fixture = {"work_history": list(profile.experience), "accomplishments": [], "education_certifications": [],
                   "experience_level": "", "career_transition_goal": keywords[0] if keywords else "",
                   "soft_preferences": keywords[1:], "work_arrangement_preferences": arrangements}
        content, _ = bge.build_bge_candidate_text(profile, fixture, model.tokenizer)
        return content

    def encode(text: str) -> np.ndarray:
        return model.encode([QUERY_PREFIX + text], convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)[0]

    return encode, query_text


class RrfRankingBackend:
    ranker_id = "tfidf_bge_rrf_equal"
    version = "rank_candidates-rrf-v1-k60-depth50-equal"
    score_type = "reciprocal_rank_fusion"
    feedback_version = feedback.VERSION

    def __init__(self, jobs: Sequence[matching.JobRecord], corpus_dir: Path, *,
                 encoder: Callable[[str], np.ndarray] | None = None,
                 query_text: Callable[[matching.CandidateProfile], str] | None = None,
                 model_dir: Path | None = None):
        manifest = json.loads((corpus_dir / "bge_embeddings_manifest.json").read_text(encoding="utf-8"))
        ids_hash = hashlib.sha256("\0".join(job.job_id for job in jobs).encode()).hexdigest()
        if ids_hash != manifest["ordered_job_ids_sha256"]:
            raise ValueError("API jobs file does not match the embedding matrix order; use the corpus JSONL.")
        self.matrix = np.load(corpus_dir / "bge_embeddings.npy")  # held in memory: scored in full on every request
        if self.matrix.shape[0] != len(jobs):
            raise ValueError("Embedding row count does not match the jobs file.")
        self.jobs = list(jobs)
        self.job_ids = [job.job_id for job in self.jobs]
        self.index_of = {job_id: index for index, job_id in enumerate(self.job_ids)}
        self.corpus = matching.fit_tfidf(self.jobs)
        if encoder is None:
            import bge_text  # lazy, for the model location

            model_dir = model_dir or bge_text.model_dir()
            encoder, query_text = _default_encoder(model_dir)
        self._encode = encoder
        self._query_text = query_text or (lambda profile: " ".join((*profile.experience, *profile.evidenced_skills)))

    def _indices(self, job_ids: Sequence[str]) -> list[int]:
        return [self.index_of[job_id] for job_id in job_ids if job_id in self.index_of]

    def rank(self, profile: matching.CandidateProfile, limit: int, candidate_indices: Sequence[int], *,
             liked_job_ids: Sequence[str] = (), disliked_job_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
        liked, disliked = self._indices(liked_job_ids), self._indices(disliked_job_ids)
        rated = set(liked) | set(disliked)
        indices = [index for index in candidate_indices if index not in rated] if rated else list(candidate_indices)
        eligible, _ = matching._eligible_jobs(self.corpus, profile, indices)
        if not eligible:
            return []
        depth = min(DEPTH, len(eligible))
        # TF-IDF signal, with the query adjusted by rated jobs when there are any
        query_sparse = self.corpus.vectorizer.transform([profile.tfidf_text()])
        if rated:
            query_sparse = feedback.adjusted_sparse_query(query_sparse, self.corpus.job_matrix, liked, disliked)
        eligible_array = np.asarray(eligible)
        # score the whole matrix once and pick the eligible rows (slicing 80k sparse rows per request is far slower)
        tfidf_scores = (self.corpus.job_matrix @ query_sparse.T).toarray().ravel()[eligible_array]
        tfidf_top = feedback.top_ranked(tfidf_scores, eligible, self.job_ids, depth)
        # embedding signal
        query = self._encode(self._query_text(profile))
        if rated:
            query = feedback.adjusted_dense_query(
                query,
                np.asarray(self.matrix[liked]) if liked else None,
                np.asarray(self.matrix[disliked]) if disliked else None,
            )
        scores = (self.matrix @ query)[eligible_array]
        embedding_top = feedback.top_ranked(scores, eligible, self.job_ids, depth)
        fused: dict[str, float] = defaultdict(float)
        for position, (index, _) in enumerate(tfidf_top, 1):
            fused[self.job_ids[index]] += 1.0 / (RRF_K + position)
        for position, (index, _) in enumerate(embedding_top, 1):
            fused[self.job_ids[index]] += 1.0 / (RRF_K + position)
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
        return [{"job_id": job_id, "rank": rank, "score": score} for rank, (job_id, score) in enumerate(ranked, 1)]

    def most_similar_liked(self, job_ids: Sequence[str], liked_job_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """For each result, the liked job whose embedding is closest (cosine) to it."""
        liked = [job_id for job_id in liked_job_ids if job_id in self.index_of]
        if not liked:
            return {}
        liked_matrix = np.asarray(self.matrix[[self.index_of[job_id] for job_id in liked]])
        result: dict[str, dict[str, Any]] = {}
        for job_id in job_ids:
            index = self.index_of.get(job_id)
            if index is None:
                continue
            similarities = liked_matrix @ np.asarray(self.matrix[index])
            best = int(np.argmax(similarities))
            result[job_id] = {"job_id": liked[best], "similarity": float(similarities[best])}
        return result
