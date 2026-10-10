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
import random
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
EXPLORE_TOP = 30   # a job counts as one signal's favourite when it sits in that signal's top 30
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


DOCUMENT_MAX_TOKENS = 128  # short sentences (requirements, profile lines) for the Compass analysis


def _default_encoder(model_dir: Path) -> tuple[
    Callable[[str], np.ndarray],
    Callable[[matching.CandidateProfile], str],
    Callable[[Sequence[str]], np.ndarray],
]:
    import threading

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

    lock = threading.Lock()  # max_seq_length is model-wide state, so batch encodes take turns

    def encode_many(texts: Sequence[str]) -> np.ndarray:
        """Embed short texts as documents (no query prefix), unit length, float32."""
        with lock:
            previous = model.max_seq_length
            model.max_seq_length = DOCUMENT_MAX_TOKENS
            try:
                vectors = model.encode(list(texts), batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
                                       show_progress_bar=False)
            finally:
                model.max_seq_length = previous
        return np.asarray(vectors, dtype=np.float32)

    return encode, query_text, encode_many


class RrfRankingBackend:
    ranker_id = "tfidf_bge_rrf_equal"
    version = "rank_candidates-rrf-v1-k60-depth50-equal"
    score_type = "reciprocal_rank_fusion"
    feedback_version = feedback.VERSION
    supports_exploration = True  # can add jobs the two signals disagree about (see rank)
    supports_compass = True  # exposes the vector methods the Compass analysis needs (see backend/compass.py)

    def __init__(self, jobs: Sequence[matching.JobRecord], corpus_dir: Path, *,
                 encoder: Callable[[str], np.ndarray] | None = None,
                 query_text: Callable[[matching.CandidateProfile], str] | None = None,
                 document_encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
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
            encoder, query_text, document_encoder = _default_encoder(model_dir)
        self._encode = encoder
        self._encode_many = document_encoder
        self._query_text = query_text or (lambda profile: " ".join((*profile.experience, *profile.evidenced_skills)))

    # ---- vector access for the Compass analysis ----

    def job_vectors(self, job_ids: Sequence[str]) -> np.ndarray:
        """Stored unit-length embeddings for the given listings, in the given order."""
        return np.asarray(self.matrix[[self.index_of[job_id] for job_id in job_ids]], dtype=np.float32)

    def query_vector(self, profile: matching.CandidateProfile) -> np.ndarray:
        """The profile as a search query, in the same space as the listing vectors (the ranker's own query)."""
        return np.asarray(self._encode(self._query_text(profile)), dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self._encode_many is None:
            raise RuntimeError("This ranker was built without a document encoder.")
        return np.asarray(self._encode_many(texts), dtype=np.float32)

    def score_all(self, vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of a unit vector with every listing, aligned to the repository order."""
        return np.asarray(self.matrix @ np.asarray(vector, dtype=np.float32))

    def _indices(self, job_ids: Sequence[str]) -> list[int]:
        return [self.index_of[job_id] for job_id in job_ids if job_id in self.index_of]

    def rank(self, profile: matching.CandidateProfile, limit: int, candidate_indices: Sequence[int], *,
             liked_job_ids: Sequence[str] = (), disliked_job_ids: Sequence[str] = (),
             exclude_job_ids: Sequence[str] = (), explore: int = 0,
             rng: random.Random | None = None) -> list[dict[str, Any]]:
        """Fused ranking of the eligible listings, best first.

        With explore > 0, up to that many extra listings follow the ranked ones, each flagged with `exploration`:
        "keyword" (the keyword signal ranks it high, the embedding signal does not), "embedding" (the reverse), and, when there
        is room for more than two, one "random" listing. They are chosen to teach us something: a rating on a listing the two
        signals disagree about says which signal was right. The caller decides where to show them.
        """
        liked, disliked = self._indices(liked_job_ids), self._indices(disliked_job_ids)
        adjust = bool(liked or disliked)  # excluded jobs are hidden but do not move the query
        rated = set(liked) | set(disliked) | set(self._indices(exclude_job_ids))
        indices = [index for index in candidate_indices if index not in rated] if rated else list(candidate_indices)
        eligible, _ = matching._eligible_jobs(self.corpus, profile, indices)
        if not eligible:
            return []
        depth = min(DEPTH, len(eligible))
        # TF-IDF signal, with the query adjusted by rated jobs when there are any
        query_sparse = self.corpus.vectorizer.transform([profile.tfidf_text()])
        if adjust:
            query_sparse = feedback.adjusted_sparse_query(query_sparse, self.corpus.job_matrix, liked, disliked)
        eligible_array = np.asarray(eligible)
        # score the whole matrix once and pick the eligible rows (slicing 80k sparse rows per request is far slower)
        tfidf_scores = (self.corpus.job_matrix @ query_sparse.T).toarray().ravel()[eligible_array]
        tfidf_top = feedback.top_ranked(tfidf_scores, eligible, self.job_ids, depth)
        # embedding signal
        query = self._encode(self._query_text(profile))
        if adjust:
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
        results = [{"job_id": job_id, "rank": rank, "score": score} for rank, (job_id, score) in enumerate(ranked, 1)]
        if explore > 0:
            results.extend(self._exploration(explore, {r["job_id"] for r in results}, tfidf_top, embedding_top, depth, eligible, rng or random.Random()))
        return results

    def _exploration(self, count: int, shown: set[str], tfidf_top: Sequence[tuple[int, float]], embedding_top: Sequence[tuple[int, float]],
                     depth: int, eligible: Sequence[int], rng: random.Random) -> list[dict[str, Any]]:
        """Listings to add beyond the ranked ones: where the two signals disagree most, balanced between the two sides, plus a random one."""
        position_keyword = {self.job_ids[i]: p for p, (i, _) in enumerate(tfidf_top, 1)}
        position_embedding = {self.job_ids[i]: p for p, (i, _) in enumerate(embedding_top, 1)}
        sides: dict[str, list[tuple[int, str]]] = {"keyword": [], "embedding": []}
        for job_id in set(position_keyword) | set(position_embedding):
            if job_id in shown:
                continue
            k, e = position_keyword.get(job_id, depth + 1), position_embedding.get(job_id, depth + 1)
            if min(k, e) > EXPLORE_TOP:
                continue
            sides["keyword" if k < e else "embedding"].append((abs(k - e), job_id))
        for side in sides.values():
            side.sort(key=lambda pair: (-pair[0], pair[1]))
        want_random = 1 if count >= 3 else 0
        chosen: list[dict[str, Any]] = []
        order = ["keyword", "embedding"]
        taken = {"keyword": 0, "embedding": 0}
        while len(chosen) < count - want_random and any(taken[s_] < len(sides[s_]) for s_ in order):
            for side in order:
                if len(chosen) >= count - want_random:
                    break
                if taken[side] < len(sides[side]):
                    chosen.append({"job_id": sides[side][taken[side]][1], "rank": None, "score": 0.0, "exploration": side})
                    taken[side] += 1
        if want_random:
            used = shown | {c["job_id"] for c in chosen}
            pool = [self.job_ids[i] for i in eligible if self.job_ids[i] not in used]
            if pool:
                chosen.append({"job_id": rng.choice(pool), "rank": None, "score": 0.0, "exploration": "random"})
        return chosen

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
