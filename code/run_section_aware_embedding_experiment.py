"""Run the single frozen section-aware MiniLM text-representation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import job_matching_engine as engine
import embedding_comparison as comparison
from run_fresh_profile_validation import _profile_inputs
from section_aware_embedding import (
    REPRESENTATION_ID,
    section_aware_candidate_text,
    section_aware_job_text,
)


METHODS = ("keyword", "tfidf_cosine", "embedding_cosine", REPRESENTATION_ID)
TOP_K = 5
MODEL_ID = comparison.MODEL_ID
MODEL_REVISION = comparison.MODEL_REVISION
MAX_SEQUENCE_LENGTH = comparison.MODEL_MAX_SEQ_LENGTH
BATCH_SIZE = comparison.ENCODE_BATCH_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    status = git("status", "--short")
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(status), "working_tree_status_lines": status.splitlines()}


def _cache_manifest_valid(
    cache_path: Path,
    manifest_path: Path,
    expected: Mapping[str, Any],
) -> np.ndarray | None:
    if not cache_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        if manifest.get("embedding_file_sha256") != _sha256(cache_path):
            return None
        matrix = np.load(cache_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != expected["job_count"]:
            return None
        if matrix.shape[1] != expected["dimension"]:
            return None
        return matrix
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _embedding_rank(
    corpus: engine.MatchingCorpus,
    candidate_indices: Sequence[int],
    embedding_matrix: np.ndarray,
    embedding_row_by_index: Mapping[int, int],
    query: np.ndarray,
    profile_id: str,
    method: str,
) -> list[dict[str, Any]]:
    included, _ = engine._eligible_jobs(corpus, engine_profile := _ACTIVE_PROFILES[profile_id], candidate_indices)
    if not included:
        return []
    rows = np.fromiter((embedding_row_by_index[index] for index in included), dtype=np.int64)
    scores = embedding_matrix[rows] @ query
    ordered = sorted(zip(scores.tolist(), included), key=lambda pair: (-float(pair[0]), corpus.jobs[pair[1]].job_id))[:TOP_K]
    output = []
    for rank, (score, index) in enumerate(ordered, 1):
        job = corpus.jobs[index]
        output.append({
            "profile_id": profile_id, "method": method,
            "condition": "description_available", "rank": rank,
            "job_id": job.job_id,
            "title": engine._nonblank(job.source.get("title_raw")) or job.searchable_title,
            "score": round(float(score), 8), "score_type": "normalized_embedding_cosine",
        })
    return output


_ACTIVE_PROFILES: dict[str, engine.CandidateProfile] = {}


def run(plan_path: Path, profiles_path: Path, output_dir: Path, sample_path: Path) -> dict[str, Any]:
    global _ACTIVE_PROFILES
    plan_path, profiles_path = plan_path.resolve(), profiles_path.resolve()
    output_dir, sample_path = output_dir.resolve(), sample_path.resolve()
    if not plan_path.is_file() or not profiles_path.is_file() or not output_dir.is_dir():
        raise FileNotFoundError("Frozen plan/profiles and the precreated experiment directory are required.")
    protected_outputs = ("heldout_rankings.csv", "ranking_mappings.csv", "blinded_evaluation_pool.jsonl",
                         "frozen_configuration.json", "retrieval_timings.json")
    if any((output_dir / name).exists() for name in protected_outputs):
        raise FileExistsError("Refusing to overwrite existing experiment outputs.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    frozen = plan["frozen_rankers"]
    if (plan.get("frozen_before_retrieval") is not True or frozen.get("top_k_per_method_profile") != TOP_K
            or frozen.get("baseline_methods") != list(METHODS[:3])
            or frozen.get("variant_method") != REPRESENTATION_ID
            or frozen.get("embedding", {}).get("id") != MODEL_ID
            or frozen.get("embedding", {}).get("revision") != MODEL_REVISION):
        raise ValueError("Plan does not match the frozen four-method experiment contract.")
    fixtures, profiles = _profile_inputs(raw_profiles)
    if plan["profiles"].get("profile_order_is_frozen") is not True or len(profiles) != 4:
        raise ValueError("Frozen plan does not specify the fixed four-profile input order.")
    _ACTIVE_PROFILES = {profile.profile_id: profile for profile in profiles}

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    jobs, load_report = engine.load_prepared_jobs(sample_path)
    corpus_load_seconds = time.perf_counter() - load_started
    available_indices = engine.condition_job_indices(engine.MatchingCorpus(jobs, None, None), "description_available")
    if len(jobs) != 30_000 or len(available_indices) != 24_514:
        raise ValueError("Prepared sample differs from the verified fixed corpus; aborting.")

    tfidf_started = time.perf_counter()
    corpus = engine.fit_tfidf(jobs)
    tfidf_fit_seconds = time.perf_counter() - tfidf_started
    available_indices = engine.condition_job_indices(corpus, "description_available")
    job_ids = [corpus.jobs[index].job_id for index in available_indices]
    base_texts = [engine._weighted_job_text(corpus.jobs[index].source) for index in available_indices]

    try:
        sentence_transformers_version = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Pinned sentence-transformers dependency is unavailable.") from error
    model_version = json.dumps({"id": MODEL_ID, "revision": MODEL_REVISION,
                                "sentence_transformers": sentence_transformers_version,
                                "max_seq_length": MAX_SEQUENCE_LENGTH, "normalize": True}, sort_keys=True)
    base_key = comparison._cache_key(job_ids, base_texts, model_version)
    base_cache = PROJECT_ROOT / "data" / "matching" / ".embedding_cache" / f"all_minilm_{base_key[:20]}.npy"
    base_manifest_path = base_cache.with_suffix(".json")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8")) if base_manifest_path.is_file() else {}
    base_hash_before = _sha256(base_cache) if base_cache.is_file() else None

    model_snapshot = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--sentence-transformers--all-MiniLM-L6-v2" / Path("snapshots") / MODEL_REVISION
    )
    if not model_snapshot.is_dir():
        raise FileNotFoundError(f"Pinned local model snapshot missing: {model_snapshot}")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model_load_started = time.perf_counter()
    model = SentenceTransformer(str(model_snapshot), device="cpu")
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    model_load_seconds = time.perf_counter() - model_load_started
    dimension = int(model.get_sentence_embedding_dimension())

    base_expected = {"cache_key": base_key, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                     "job_count": len(job_ids), "dimension": dimension, "text_count": len(base_texts),
                     "normalization": True, "max_sequence_length_tokens": MAX_SEQUENCE_LENGTH}
    cache_verify_started = time.perf_counter()
    base_embeddings = _cache_manifest_valid(base_cache, base_manifest_path, base_expected)
    if base_embeddings is None:
        raise RuntimeError("Original embedding cache failed its frozen hash/manifest checks; refusing to rebuild it.")
    base_cache_verify_seconds = time.perf_counter() - cache_verify_started

    tokenizer_prep_started = time.perf_counter()
    variant_texts = [section_aware_job_text(corpus.jobs[index].source, model.tokenizer, MAX_SEQUENCE_LENGTH)
                     for index in available_indices]
    variant_candidate_texts = {
        profile.profile_id: section_aware_candidate_text(
            profile, fixtures[profile.profile_id], model.tokenizer, MAX_SEQUENCE_LENGTH
        ) for profile in profiles
    }
    if any(len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) > MAX_SEQUENCE_LENGTH
           for text in variant_texts):
        raise AssertionError("Section-aware document builder exceeded the frozen tokenizer budget.")
    if any(len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) > MAX_SEQUENCE_LENGTH
           for text in variant_candidate_texts.values()):
        raise AssertionError("Section-aware candidate builder exceeded the frozen tokenizer budget.")
    text_prep_seconds = time.perf_counter() - tokenizer_prep_started

    variant_key = comparison._cache_key(job_ids, variant_texts, model_version + "|" + REPRESENTATION_ID)
    cache_dir = PROJECT_ROOT / "data" / "matching" / ".embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    variant_cache = cache_dir / f"sectionaware_minilm_{variant_key[:20]}.npy"
    variant_manifest_path = variant_cache.with_suffix(".json")
    variant_expected = {"cache_key": variant_key, "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION, "representation_id": REPRESENTATION_ID,
                        "job_count": len(job_ids), "dimension": dimension,
                        "text_count": len(variant_texts), "normalization": True,
                        "max_sequence_length_tokens": MAX_SEQUENCE_LENGTH}
    variant_verify_started = time.perf_counter()
    variant_embeddings = _cache_manifest_valid(variant_cache, variant_manifest_path, variant_expected)
    variant_cache_hit = variant_embeddings is not None
    variant_cache_verify_seconds = time.perf_counter() - variant_verify_started
    variant_encode_seconds = 0.0
    if variant_embeddings is None:
        encode_started = time.perf_counter()
        encoded = model.encode(variant_texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
        variant_embeddings = np.asarray(encoded, dtype=np.float32)
        variant_encode_seconds = time.perf_counter() - encode_started
        np.save(variant_cache, variant_embeddings)
        variant_expected["embedding_file_sha256"] = _sha256(variant_cache)
        _write_json(variant_manifest_path, variant_expected)
    base_hash_after = _sha256(base_cache)
    if base_hash_before != base_hash_after:
        raise RuntimeError("Original baseline cache changed during the variant experiment.")
    variant_hash = _sha256(variant_cache)

    row_by_index = {index: row for row, index in enumerate(available_indices)}
    results: list[dict[str, Any]] = []
    per_profile_method_seconds: dict[str, dict[str, float]] = {}
    baseline_query_seconds = 0.0
    variant_query_seconds = 0.0
    comparison._ACTIVE_MODEL = model
    for profile in profiles:
        per_profile_method_seconds[profile.profile_id] = {}
        included = comparison.filtered_indices_for_profile(corpus, profile, available_indices)
        if not included:
            raise ValueError("A fresh profile has no eligible description-available jobs after frozen filters.")
        for method in METHODS:
            started = time.perf_counter()
            if method == "keyword":
                ranked = engine.rank_keyword(corpus, profile, TOP_K,
                                             candidate_indices=available_indices,
                                             condition="description_available")
            elif method == "tfidf_cosine":
                ranked = engine.rank_tfidf(corpus, profile, TOP_K,
                                           candidate_indices=available_indices,
                                           condition="description_available")
            elif method == "embedding_cosine":
                ranked = comparison._embedding_rank(corpus, profile, available_indices,
                                                    base_embeddings, row_by_index, TOP_K)
            else:
                query = model.encode([variant_candidate_texts[profile.profile_id]],
                                     convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=1)[0]
                ranked = _embedding_rank(corpus, available_indices, variant_embeddings,
                                         row_by_index, query, profile.profile_id, REPRESENTATION_ID)
            elapsed = time.perf_counter() - started
            per_profile_method_seconds[profile.profile_id][method] = elapsed
            if method in {"keyword", "tfidf_cosine", "embedding_cosine"}:
                baseline_query_seconds += elapsed
            else:
                variant_query_seconds += elapsed
            if len(ranked) != TOP_K:
                raise ValueError("A ranker returned fewer than the frozen five results for {} / {}".format(
                    profile.profile_id, method))
            for row in ranked:
                results.append({
                    "profile_id": row["profile_id"],
                    "fixture_type": fixtures[row["profile_id"]]["fixture_type"],
                    "method": row["method"], "condition": row["condition"],
                    "rank": int(row["rank"]), "job_id": row["job_id"],
                    "title": row.get("title", ""), "score": row["score"],
                    "score_type": row["score_type"],
                })

    columns = ["profile_id", "fixture_type", "method", "condition", "rank", "job_id", "title", "score", "score_type"]
    comparison._write_csv(output_dir / "heldout_rankings.csv", results, columns)
    mappings = [{key: row[key] for key in ("profile_id", "fixture_type", "method", "condition", "rank", "job_id", "score", "score_type")}
                for row in results]
    comparison._write_csv(output_dir / "ranking_mappings.csv", mappings,
                          ["profile_id", "fixture_type", "method", "condition", "rank", "job_id", "score", "score_type"])
    profile_ids = [profile.profile_id for profile in profiles]
    profile_types = {row["profile_id"]: row["fixture_type"] for row in raw_profiles}
    split = {"heldout_profile_ids": profile_ids,
             "fixture_type_by_profile_id": profile_types,
             "scope": "All four frozen fresh profiles are evaluated; no internal split."}
    _write_json(output_dir / "profile_split_manifest.json", split)
    pool_started = time.perf_counter()
    pool = comparison.build_blinded_pool(results, corpus, fixtures, profile_ids,
                                         int(plan["evaluation"]["pool_shuffle_seed"]))
    if len(pool) > 80:
        raise AssertionError("Four-method top-five union exceeded the frozen 80-pair cap.")
    with (output_dir / "blinded_evaluation_pool.jsonl").open("w", encoding="utf-8") as target:
        for item in pool:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
    pool_build_seconds = time.perf_counter() - pool_started
    evaluator_prompt = (output_dir / plan["evaluation"]["prompt_source"]).resolve()
    (output_dir / "evaluator_prompt.md").write_bytes(evaluator_prompt.read_bytes())

    frozen_config = {
        "experiment_id": plan["experiment_id"], "branch_and_worktree": _git_metadata(),
        "plan_sha256": _sha256(plan_path), "profiles_sha256": _sha256(profiles_path),
        "sample_sha256": _sha256(sample_path), "prompt_sha256": _sha256(output_dir / "evaluator_prompt.md"),
        "source_hashes": {"engine": _sha256(Path(engine.__file__)),
                          "baseline_comparison": _sha256(Path(comparison.__file__)),
                          "section_aware_representation": _sha256(Path(__file__).with_name("section_aware_embedding.py")),
                          "runner": _sha256(Path(__file__)),
                          "evaluator": _sha256(Path(__file__).with_name("evaluate_embedding_section_experiment.py"))},
        "corpus": {"load_report": load_report.to_dict(), "description_available_count": len(available_indices)},
        "models": {"embedding_id": MODEL_ID, "revision": MODEL_REVISION,
                   "device": "cpu", "sentence_transformers_version": sentence_transformers_version,
                   "max_sequence_length": MAX_SEQUENCE_LENGTH},
        "baseline_cache": {"path": str(base_cache), "key": base_key, "verified_hit": True,
                           "sha256_before": base_hash_before, "sha256_after": base_hash_after},
        "variant_cache": {"path": str(variant_cache), "key": variant_key, "representation_id": REPRESENTATION_ID,
                          "verified_hit": variant_cache_hit, "sha256": variant_hash,
                          "job_count": len(job_ids), "job_text_max_tokens": max(
                              len(model.tokenizer(x, add_special_tokens=True, truncation=False)["input_ids"])
                              for x in variant_texts)},
        "candidate_variant_input_max_tokens": max(
            len(model.tokenizer(x, add_special_tokens=True, truncation=False)["input_ids"])
            for x in variant_candidate_texts.values()),
        "result_counts": {pid: {method: sum(r["profile_id"] == pid and r["method"] == method for r in results)
                                for method in METHODS} for pid in profile_ids},
        "shared_pool_count": len(pool), "shared_pool_seed": plan["evaluation"]["pool_shuffle_seed"],
        "blinded_pool_fields": sorted(pool[0].keys()) if pool else [],
    }
    _write_json(output_dir / "frozen_configuration.json", frozen_config)
    timings = {
        "machine": {"python": platform.python_version(), "cpu": platform.processor(),
                    "logical_cpu_count": os.cpu_count(), "device": "cpu"},
        "corpus_load_seconds": corpus_load_seconds,
        "tfidf_fit_seconds": tfidf_fit_seconds,
        "model_load_seconds": model_load_seconds,
        "baseline_embedding_cache_verify_seconds": base_cache_verify_seconds,
        "section_aware_text_preprocessing_seconds": text_prep_seconds,
        "section_aware_cache_verify_seconds": variant_cache_verify_seconds,
        "section_aware_job_encoding_seconds": variant_encode_seconds,
        "per_profile_method_ranking_seconds": per_profile_method_seconds,
        "baseline_method_ranking_sum_seconds": baseline_query_seconds,
        "variant_query_and_ranking_sum_seconds": variant_query_seconds,
        "pool_build_and_shuffle_seconds": pool_build_seconds,
        "total_retrieval_seconds": time.perf_counter() - total_started,
        "baseline_job_embeddings_rebuilt": False,
        "variant_job_embeddings_rebuilt": not variant_cache_hit,
    }
    _write_json(output_dir / "retrieval_timings.json", timings)
    command = (
        "# Reproduction Commands (PowerShell)\n\n"
        "Run from the repository root with `.venv`. This experiment reads the fixed prepared sample locally, verifies/reuses the original cache, and writes the variant cache separately. It does not contact an evaluator.\n\n"
        "```powershell\n"
        f'& ".\\.venv\\Scripts\\python.exe" .\\code\\run_section_aware_embedding_experiment.py --plan ".\\{plan_path.relative_to(PROJECT_ROOT)}" --profiles ".\\{profiles_path.relative_to(PROJECT_ROOT)}" --output-dir ".\\{output_dir.relative_to(PROJECT_ROOT)}"\n'
        f'& ".\\.venv\\Scripts\\python.exe" .\\code\\evaluate_embedding_section_experiment.py --experiment-dir ".\\{output_dir.relative_to(PROJECT_ROOT)}"\n'
        "```\n"
    )
    (output_dir / "reproduce.md").write_text(command, encoding="utf-8")
    return {"experiment_id": plan["experiment_id"], "profiles": len(profiles),
            "result_rows": len(results), "pool_pairs": len(pool), "result_counts": frozen_config["result_counts"],
            "variant_cache_hit": variant_cache_hit,
            "timings_seconds": {key: round(value, 3) for key, value in timings.items()
                                if key.endswith("_seconds") and isinstance(value, (int, float))}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=PROJECT_ROOT / engine.DEFAULT_SAMPLE)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.profiles, args.output_dir, args.sample), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())