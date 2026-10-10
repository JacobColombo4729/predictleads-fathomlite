"""Build one frozen TF-IDF + section-aware MiniLM top-10 union for reranking."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import embedding_comparison as comparison
import job_matching_engine as engine
from requirement_aware_reranker import ProviderConfig, ProviderUnavailable, verify_provider
from run_section_aware_embedding_experiment import _cache_manifest_valid, _sha256
from section_aware_embedding import (
    REPRESENTATION_ID,
    section_aware_candidate_text,
    section_aware_job_text,
)

TOP_K = 10


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _profile_records(path: Path) -> tuple[list[dict[str, Any]], list[engine.CandidateProfile]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("The frozen profile file must contain exactly four profiles.")
    ids = [str(row.get("profile_id", "")) for row in records]
    if len(set(ids)) != 4 or any(row.get("synthetic_fixture") is not True for row in records):
        raise ValueError("Profile IDs must be unique and every fixture explicitly synthetic.")
    required = ("work_history", "accomplishments", "experience_level", "skills_with_evidence",
                "education_certifications", "desired_roles", "soft_preferences", "hard_constraints")
    profiles = []
    for row in records:
        if any(not row.get(field) for field in required):
            raise ValueError("Incomplete frozen profile: {}".format(row.get("profile_id")))
        if not all(isinstance(skill, dict) and skill.get("skill") and skill.get("evidence")
                   for skill in row["skills_with_evidence"]):
            raise ValueError("Every listed skill needs supporting evidence.")
        hard = row["hard_constraints"]
        preferences = {"exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
                       "require_location_match": hard.get("require_location_match") is True,
                       "keywords": list(row.get("soft_preferences", [])),
                       "work_arrangement": list(row.get("work_arrangement_preferences", []))}
        experience = [*row["work_history"], *row["accomplishments"], *row["education_certifications"]]
        skills = ["{}: {}".format(skill["skill"], skill["evidence"])
                  for skill in row["skills_with_evidence"]]
        profiles.append(engine.CandidateProfile(
            profile_id=row["profile_id"], experience=tuple(experience), evidenced_skills=tuple(skills),
            desired_roles=tuple(row["desired_roles"]), location=row.get("location"), preferences=preferences))
    return records, profiles


def _eligible_indices(corpus: engine.MatchingCorpus, profile: engine.CandidateProfile,
                       candidates: list[int]) -> list[int]:
    included, _ = engine._eligible_jobs(corpus, profile, candidates)
    return included


def run(plan_path: Path, profiles_path: Path, sample_path: Path, output_dir: Path) -> dict[str, Any]:
    provider = ProviderConfig.from_environment()
    plan_path, profiles_path, sample_path = (path.resolve() for path in (plan_path, profiles_path, sample_path))
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing union output directory: {}".format(output_dir))
    if not all(path.is_file() for path in (plan_path, profiles_path, sample_path)):
        raise FileNotFoundError("Frozen plan, profiles, and prepared sample must exist before retrieval.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("frozen_before_retrieval") is not True:
        raise ValueError("Plan is not marked frozen before retrieval.")
    if plan.get("experiment_id") != "requirement_rerank_20261007T235420Z":
        raise ValueError("Unexpected frozen experiment ID.")
    if plan.get("corpus", {}).get("sha256") != _sha256(sample_path):
        raise ValueError("Prepared sample hash differs from the frozen plan; aborting before retrieval.")
    records, profiles = _profile_records(profiles_path)
    frozen_ids = [str(row["profile_id"]) for row in records]
    expected = plan.get("profiles", {})
    if expected.get("count") != 4 or expected.get("frozen_in_file_order") is not True:
        raise ValueError("Plan does not match the frozen profile contract.")
    if plan.get("retrieval", {}).get("top_k_each") != TOP_K:
        raise ValueError("Plan does not match the frozen top-10 retrieval contract.")
    provider_probe = verify_provider(provider)
    started = time.perf_counter()
    jobs, load_report = engine.load_prepared_jobs(sample_path)
    corpus = engine.fit_tfidf(jobs)
    available = engine.condition_job_indices(corpus, "description_available")
    if len(jobs) != 30_000 or len(available) != 24_514:
        raise ValueError("Prepared sample differs from frozen 30,000 / 24,514 corpus counts.")

    version = importlib.metadata.version("sentence-transformers")
    model_id = comparison.MODEL_ID
    revision = comparison.MODEL_REVISION
    if plan["retrieval"]["section_aware_embedding"].get("model_id") != model_id:
        raise ValueError("Frozen embedding model ID does not match the implementation.")
    if plan["retrieval"]["section_aware_embedding"].get("revision") != revision:
        raise ValueError("Frozen embedding revision does not match the implementation.")
    job_ids = [corpus.jobs[index].job_id for index in available]
    # Load only the pinned local model snapshot; never fetch model files implicitly.
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / revision
    )
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned local MiniLM snapshot is missing: {}".format(snapshot))
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model = SentenceTransformer(str(snapshot), device="cpu")
    model.max_seq_length = int(plan["retrieval"]["section_aware_embedding"]["max_sequence_length"])
    variant_texts = [section_aware_job_text(corpus.jobs[index].source, model.tokenizer,
                                            model.max_seq_length) for index in available]
    variant_candidate_texts = {
        profile.profile_id: section_aware_candidate_text(
            profile, record, model.tokenizer, model.max_seq_length
        ) for profile, record in zip(profiles, records)
    }
    model_version = json.dumps({"id": model_id, "revision": revision, "sentence_transformers": version,
                                "max_seq_length": model.max_seq_length, "normalize": True}, sort_keys=True)
    variant_key = comparison._cache_key(job_ids, variant_texts, model_version + "|" + REPRESENTATION_ID)
    cache_path = PROJECT_ROOT / "data" / "matching" / ".embedding_cache" / (
        "sectionaware_minilm_{}.npy".format(variant_key[:20])
    )
    cache_manifest = cache_path.with_suffix(".json")
    expected_cache = {"cache_key": variant_key, "model_id": model_id, "model_revision": revision,
                      "representation_id": REPRESENTATION_ID, "job_count": len(job_ids),
                      "dimension": int(model.get_sentence_embedding_dimension()),
                      "text_count": len(variant_texts), "normalization": True,
                      "max_sequence_length_tokens": model.max_seq_length}
    matrix = _cache_manifest_valid(cache_path, cache_manifest, expected_cache)
    if matrix is None:
        raise RuntimeError("Frozen section-aware embedding cache failed hash/manifest verification; no rebuild allowed.")
    row_by_index = {index: row for row, index in enumerate(available)}

    retrieval_rows: list[dict[str, Any]] = []
    union: dict[tuple[str, str], dict[str, Any]] = {}
    for profile, record in zip(profiles, records):
        eligible = _eligible_indices(corpus, profile, available)
        tfidf_rows = engine.rank_tfidf(corpus, profile, TOP_K, candidate_indices=available,
                                       condition="description_available")
        query = model.encode([variant_candidate_texts[profile.profile_id]], convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False, batch_size=1)[0]
        matrix_rows = np.fromiter((row_by_index[index] for index in eligible), dtype=np.int64)
        similarities = matrix[matrix_rows] @ query
        ordered = sorted(zip(similarities.tolist(), eligible),
                         key=lambda pair: (-float(pair[0]), corpus.jobs[pair[1]].job_id))[:TOP_K]
        section_rows = [{"profile_id": profile.profile_id, "job_id": corpus.jobs[index].job_id,
                         "rank": rank, "score": round(float(score), 8)}
                        for rank, (score, index) in enumerate(ordered, 1)]
        methods = {"tfidf_cosine": tfidf_rows, REPRESENTATION_ID: section_rows}
        for method, method_rows in methods.items():
            if len(method_rows) != TOP_K:
                raise ValueError("{} returned fewer than ten results for {}".format(method, profile.profile_id))
            for row in method_rows:
                key = (profile.profile_id, str(row["job_id"]))
                retrieval_rows.append({"profile_id": key[0], "job_id": key[1], "method": method,
                                       "rank": int(row["rank"]), "score": float(row["score"])})
                entry = union.get(key)
                if entry is None:
                    job = corpus.jobs[next(index for index in available if corpus.jobs[index].job_id == key[1])]
                    source = job.source
                    candidate_evidence = comparison._candidate_evidence(record)
                    candidate_evidence["profile_id"] = key[0]
                    job_evidence = {"job_id": key[1],
                                    "title": engine._nonblank(source.get("title_raw")) or job.searchable_title,
                                    "occupation": engine._occupation(source),
                                    "location": source.get("location_raw"),
                                    "full_prepared_description": source.get("description_text") or ""}
                    entry = {"profile_id": key[0], "job_id": key[1],
                             "candidate_evidence": candidate_evidence,
                             "job_evidence": job_evidence,
                             "source_ranks": {}, "best_retrieval_rank": int(row["rank"])}
                    union[key] = entry
                entry["source_ranks"][method] = int(row["rank"])
                entry["best_retrieval_rank"] = min(entry["best_retrieval_rank"], int(row["rank"]))

    if len(union) > 80:
        raise AssertionError("Frozen top-10 union exceeded 80 pairs.")
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "retrieval_rankings.jsonl").open("w", encoding="utf-8") as target:
        for row in retrieval_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    ordered_union = [union[key] for key in sorted(union)]
    with (output_dir / "assessment_input_union.jsonl").open("w", encoding="utf-8") as target:
        for row in ordered_union:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    blinded_pool = [{"profile_id": row["profile_id"], "job_id": row["job_id"],
                     "candidate_evidence": row["candidate_evidence"],
                     "job_evidence": row["job_evidence"]} for row in ordered_union]
    random.Random(20261010).shuffle(blinded_pool)
    for index, row in enumerate(blinded_pool, 1):
        row["presentation_id"] = "P{:04d}".format(index)
    with (output_dir / "blinded_evaluation_pool.jsonl").open("w", encoding="utf-8") as target:
        for row in blinded_pool:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"status": "retrieval_complete_assessment_pending", "experiment_id": plan["experiment_id"],
                "reranker_provider_preflight": provider.public_manifest(),
                "provider_probe": provider_probe, "profile_ids": frozen_ids,
                "corpus_sha256": _sha256(sample_path), "loaded_job_count": len(jobs),
                "description_available_count": len(available), "tfidf_fit_count": 1,
                "embedding_model_id": model_id, "embedding_revision": revision,
                "section_aware_cache_path": str(cache_path), "section_aware_cache_sha256": _sha256(cache_path),
                "section_aware_cache_rebuilt": False, "result_count_by_profile_method": {
                    profile.profile_id: {method: sum(row["profile_id"] == profile.profile_id and row["method"] == method
                                                    for row in retrieval_rows)
                                         for method in ("tfidf_cosine", REPRESENTATION_ID)}
                    for profile in profiles},
                "union_pair_count": len(union), "union_pair_cap": 80,
                "retrieval_elapsed_seconds": time.perf_counter() - started,
                "reranker_model": provider.model, "reranker_revision": provider.revision,
                "judgments_status": "not_run"}
    _write_json(output_dir / "retrieval_manifest.json", manifest)
    return {"status": manifest["status"], "profiles": len(profiles),
            "ranking_rows": len(retrieval_rows), "union_pairs": len(union),
            "elapsed_seconds": manifest["retrieval_elapsed_seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=PROJECT_ROOT / engine.DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.plan, args.profiles, args.sample, args.output_dir)
    except ProviderUnavailable as error:
        print(json.dumps({"status": "pending_provider", "error": str(error),
                          "retrieval_executed": False, "api_calls": 0}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
