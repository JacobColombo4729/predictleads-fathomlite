"""Run one frozen four-profile retrieval validation using existing rankers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import job_matching_engine as engine
import embedding_comparison as comparison
from embedding_comparison import (
    MODEL_ID,
    MODEL_MAX_SEQ_LENGTH,
    MODEL_REVISION,
    _cache_key,
    _embedding_rank,
    _job_documents,
    _profile_record,
    _write_csv,
    filtered_indices_for_profile,
)


METHODS = ("keyword", "tfidf_cosine", "embedding_cosine")
TOP_K = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    status = git("status", "--short")
    return {"branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(status),
            "working_tree_status_lines": status.splitlines()}


def _profile_inputs(raw: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[engine.CandidateProfile]]:
    if len(raw) != 4:
        raise ValueError("The frozen fresh-profile plan requires exactly four profiles.")
    ids = [row.get("profile_id") for row in raw]
    if len(set(ids)) != 4 or any(row.get("synthetic_fixture") is not True for row in raw):
        raise ValueError("Fresh profiles must have unique IDs and be explicitly synthetic.")
    types = [row.get("fixture_type") for row in raw]
    if types.count("conventional_match") != 2 or types.count("career_transition") != 2:
        raise ValueError("The frozen profile mix must be two conventional and two transition fixtures.")

    fixtures: dict[str, dict[str, Any]] = {}
    profiles: list[engine.CandidateProfile] = []
    required = ("work_history", "accomplishments", "experience_level", "skills_with_evidence",
                "education_certifications", "desired_roles", "soft_preferences", "hard_constraints")
    for row in raw:
        if any(not row.get(field) for field in required):
            raise ValueError(f"Incomplete synthetic profile: {row.get('profile_id')}")
        if not all(isinstance(skill, dict) and skill.get("skill") and skill.get("evidence")
                   for skill in row["skills_with_evidence"]):
            raise ValueError(f"Skills require evidence in profile {row['profile_id']}")
        fixtures[row["profile_id"]] = row
        hard = row["hard_constraints"]
        preferences = {
            "exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
            "require_location_match": hard.get("require_location_match") is True,
            "keywords": list(row.get("soft_preferences", [])),
            "work_arrangement": list(row.get("work_arrangement_preferences", [])),
        }
        experience = [*row["work_history"], *row["accomplishments"],
                      *row["education_certifications"]]
        skills = [f"{item['skill']}: {item['evidence']}" for item in row["skills_with_evidence"]]
        profiles.append(engine.CandidateProfile(
            profile_id=row["profile_id"], experience=tuple(experience),
            evidenced_skills=tuple(skills), desired_roles=tuple(row["desired_roles"]),
            location=row.get("location"), preferences=preferences,
        ))
    return fixtures, profiles


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(plan_path: Path, profiles_path: Path, output_dir: Path, sample_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    profiles_path = profiles_path.resolve()
    output_dir = output_dir.resolve()
    sample_path = sample_path.resolve()
    if not plan_path.is_file() or not profiles_path.is_file():
        raise FileNotFoundError("Save the frozen plan and profiles before starting retrieval.")
    if not output_dir.is_dir() or any((output_dir / name).exists() for name in (
            "heldout_rankings.csv", "ranking_mappings.csv", "blinded_evaluation_pool.jsonl",
            "frozen_configuration.json", "retrieval_timings.json")):
        raise FileExistsError("Output must be the frozen-input directory with no retrieval outputs yet.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    if (plan.get("frozen_before_retrieval") is not True
            or plan.get("frozen_retrieval_settings", {}).get("top_k_per_method_profile") != TOP_K):
        raise ValueError("Frozen plan missing or retrieval scope/settings do not match this runner.")
    frozen = plan["frozen_retrieval_settings"]
    if (frozen.get("methods") != list(METHODS)
            or frozen.get("embedding_model", {}).get("id") != MODEL_ID
            or frozen.get("embedding_model", {}).get("revision") != MODEL_REVISION
            or frozen.get("embedding_model", {}).get("device") != "cpu"):
        raise ValueError("The frozen plan does not match the established retrieval configuration.")
    fixtures, profiles = _profile_inputs(raw_profiles)
    planned_ids = [item["profile_id"] for item in plan["profiles"]]
    if planned_ids != [profile.profile_id for profile in profiles]:
        raise ValueError("Profile order/IDs differ from the frozen plan.")

    started = time.perf_counter()
    jobs, load_report = engine.load_prepared_jobs(sample_path)
    load_seconds = time.perf_counter() - started
    if len(jobs) != 30_000 or sum(job.usable_description for job in jobs) != 24_514:
        raise ValueError("Prepared corpus counts differ from the previously verified sample; aborting.")

    tfidf_started = time.perf_counter()
    corpus = engine.fit_tfidf(jobs)
    tfidf_seconds = time.perf_counter() - tfidf_started
    candidate_indices = engine.condition_job_indices(corpus, "description_available")
    job_ids, job_texts = _job_documents(corpus, candidate_indices)
    model_version = json.dumps({"id": MODEL_ID, "revision": MODEL_REVISION,
                                "sentence_transformers": importlib.metadata.version("sentence-transformers"),
                                "max_seq_length": MODEL_MAX_SEQ_LENGTH, "normalize": True}, sort_keys=True)
    expected_key = _cache_key(job_ids, job_texts, model_version)
    cache_path = PROJECT_ROOT / "data" / "matching" / ".embedding_cache" / f"all_minilm_{expected_key[:20]}.npy"
    manifest_path = cache_path.with_suffix(".json")
    cache_verification_started = time.perf_counter()
    cache_hash_before = _sha256(cache_path) if cache_path.is_file() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    cached = np.load(cache_path, mmap_mode="r") if cache_hash_before else None
    expected_hash = manifest.get("embedding_file_sha256")
    if (manifest.get("cache_key") != expected_key
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("job_count") != len(job_ids)
            or manifest.get("text_count") != len(job_texts)
            or manifest.get("normalization") is not True
            or manifest.get("max_sequence_length_tokens") != MODEL_MAX_SEQ_LENGTH
            or expected_hash != cache_hash_before
            or cached is None or cached.ndim != 2 or cached.shape[0] != len(job_ids)):
        raise RuntimeError("Pinned job-embedding cache failed compatibility/hash verification; refusing to rebuild it.")
    cache_verification_seconds = time.perf_counter() - cache_verification_started

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--sentence-transformers--all-MiniLM-L6-v2" / Path("snapshots") / MODEL_REVISION
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned local model snapshot missing: {snapshot}")
    model_started = time.perf_counter()
    model = SentenceTransformer(str(snapshot), device="cpu")
    model.max_seq_length = MODEL_MAX_SEQ_LENGTH
    model_load_seconds = time.perf_counter() - model_started
    comparison._ACTIVE_MODEL = model
    if int(model.get_sentence_embedding_dimension()) != int(cached.shape[1]):
        raise RuntimeError("Pinned model output dimension does not match verified job cache.")

    embedding_rows = {index: row for row, index in enumerate(candidate_indices)}
    profile_filter_counts: dict[str, dict[str, int]] = {}
    all_rows: list[dict[str, Any]] = []
    per_query_seconds: dict[str, dict[str, float]] = {}
    for profile, fixture in zip(profiles, raw_profiles):
        included = filtered_indices_for_profile(corpus, profile, candidate_indices)
        profile_filter_counts[profile.profile_id] = {"description_available": len(candidate_indices),
                                                     "after_hard_filters": len(included)}
        per_query_seconds[profile.profile_id] = {}
        for method in METHODS:
            query_started = time.perf_counter()
            if method == "keyword":
                rows = engine.rank_keyword(corpus, profile, TOP_K,
                                           candidate_indices=candidate_indices,
                                           condition="description_available")
            elif method == "tfidf_cosine":
                rows = engine.rank_tfidf(corpus, profile, TOP_K,
                                         candidate_indices=candidate_indices,
                                         condition="description_available")
            else:
                rows = _embedding_rank(corpus, profile, candidate_indices, cached,
                                        embedding_rows, TOP_K)
            per_query_seconds[profile.profile_id][method] = time.perf_counter() - query_started
            if len(rows) > TOP_K:
                raise AssertionError("Ranker returned more rows than the frozen top-five limit.")
            all_rows.extend(_profile_record(row, fixture) for row in rows)

    ranking_seconds = sum(sum(method_times.values()) for method_times in per_query_seconds.values())
    ranking_fields = ["profile_id", "fixture_type", "method", "condition", "rank", "job_id",
                      "title", "score", "score_type"]
    mapping_fields = ["profile_id", "fixture_type", "method", "condition", "rank", "job_id",
                      "score", "score_type"]
    _write_csv(output_dir / "heldout_rankings.csv", all_rows, ranking_fields)
    _write_csv(output_dir / "ranking_mappings.csv",
               [{key: row[key] for key in mapping_fields} for row in all_rows], mapping_fields)

    pool_started = time.perf_counter()
    presentation_seed = int(plan["frozen_retrieval_settings"]["pool_shuffle_seed"])
    pool = comparison.build_blinded_pool(
        all_rows, corpus, fixtures, [profile.profile_id for profile in profiles], presentation_seed
    )
    if len(pool) > 60:
        raise AssertionError("Top-five shared pool exceeds the frozen 60-pair cap.")
    pool_path = output_dir / "blinded_evaluation_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as destination:
        for item in pool:
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
    pool_seconds = time.perf_counter() - pool_started
    heldout_ids = [profile.profile_id for profile in profiles]
    _write_json(output_dir / "profile_split_manifest.json", {
        "heldout_profile_ids": heldout_ids,
        "fixture_type_by_profile_id": {row["profile_id"]: row["fixture_type"] for row in raw_profiles},
        "scope": "All four frozen fresh profiles are evaluated; there is no development split.",
    })
    prompt_source = (output_dir / plan["frozen_evaluation"]["prompt_source"]).resolve()
    prompt_copy = output_dir / "evaluator_prompt.md"
    prompt_copy.write_bytes(prompt_source.read_bytes())

    cache_hash_after = _sha256(cache_path)
    if cache_hash_after != cache_hash_before:
        raise RuntimeError("Embedding cache changed during validation; investigate cache integrity.")
    config = {
        "validation_id": plan["validation_id"],
        "branch_and_worktree": _git_metadata(),
        "frozen_input_hashes": {"plan_sha256": _sha256(plan_path),
                                "profiles_sha256": _sha256(profiles_path),
                                "sample_sha256": _sha256(sample_path),
                                "evaluator_prompt_sha256": _sha256(prompt_copy)},
        "source_hashes": {"matcher_sha256": _sha256(Path(engine.__file__)),
                          "comparison_sha256": _sha256(Path(__file__)),
                          "comparison_helpers_sha256": _sha256(CODE_DIR / "embedding_comparison.py"),
                          "evaluation_sha256": _sha256(CODE_DIR / "evaluate_embedding_comparison.py")},
        "corpus": {"load_report": load_report.to_dict(),
                   "description_available_count": len(candidate_indices),
                   "profile_hard_filter_counts": profile_filter_counts},
        "settings": {"methods": list(METHODS), "top_k": TOP_K,
                     "embedding_model_id": MODEL_ID, "embedding_model_revision": MODEL_REVISION,
                     "embedding_device": "cpu", "embedding_max_sequence_length": MODEL_MAX_SEQ_LENGTH,
                     "embedding_cache_key": expected_key,
                     "embedding_cache_path": str(cache_path),
                     "embedding_cache_verified_hit": True,
                     "embedding_cache_sha256_before": cache_hash_before,
                     "embedding_cache_sha256_after": cache_hash_after,
                     "tfidf_fit_scope": "30,000 loaded records using existing fixed engine.fit_tfidf; prior run had no saved fitted matrix/vectorizer",
                     "pool_shuffle_seed": presentation_seed,
                     "pool_pair_count": len(pool),
                     "pool_fields": sorted(pool[0].keys()) if pool else [],
                     "review_inputs_blinded": True},
        "evaluation_method": plan["frozen_evaluation"]["actual_method_to_record"],
        "evaluator_model_identity": "unknown; no verifiable session model identifier",
        "independence_disclosure": plan["frozen_evaluation"]["independence_disclosure"],
        "result_rows_per_profile_method": {
            profile.profile_id: {method: sum(row["profile_id"] == profile.profile_id and row["method"] == method
                                             for row in all_rows) for method in METHODS}
            for profile in profiles
        },
    }
    _write_json(output_dir / "frozen_configuration.json", config)
    timings = {
        "machine": {"python": platform.python_version(), "cpu": platform.processor(),
                    "logical_cpu_count": os.cpu_count(), "device": "cpu"},
        "corpus_load_seconds": load_seconds,
        "tfidf_fit_seconds": tfidf_seconds,
        "embedding_cache_verification_seconds": cache_verification_seconds,
        "embedding_model_load_seconds": model_load_seconds,
        "per_profile_method_query_seconds": per_query_seconds,
        "ranking_total_seconds": ranking_seconds,
        "pool_blinding_seconds": pool_seconds,
        "total_run_seconds": time.perf_counter() - started,
        "job_embedding_cache_hit": True,
        "job_embedding_reencoded": False,
        "tfidf_baseline_refit": True,
        "tfidf_refit_reason": "The prior experiment did not save the fitted vectorizer or sparse job matrix.",
        "evaluator_runtime_seconds": None,
        "evaluator_api_calls": 0,
    }
    _write_json(output_dir / "retrieval_timings.json", timings)
    run_command = (
        f'& ".\\.venv\\Scripts\\python.exe" .\\code\\run_fresh_profile_validation.py '
        f'--plan ".\\{plan_path.relative_to(PROJECT_ROOT)}" '
        f'--profiles ".\\{profiles_path.relative_to(PROJECT_ROOT)}" '
        f'--output-dir ".\\{output_dir.relative_to(PROJECT_ROOT)}"'
    )
    reproduce = (
        "# Reproduction Commands (PowerShell)\n\n"
        "Run from the repository root with the configured `.venv`. The retrieval command refuses to overwrite prior rankings. Repeating this experiment requires a new timestamped directory with copies of the frozen plan and profiles.\n\n"
        "```powershell\n" + run_command + "\n"
        f'& ".\\.venv\\Scripts\\python.exe" .\\code\\evaluate_fresh_profile_validation.py '
        f'--experiment-dir ".\\{output_dir.relative_to(PROJECT_ROOT)}"\n'
        "```\n"
    )
    (output_dir / "reproduce.md").write_text(reproduce, encoding="utf-8")
    return {"validation_id": plan["validation_id"], "profile_count": len(profiles),
            "method_count": len(METHODS), "result_row_count": len(all_rows),
            "blinded_pool_pairs": len(pool), "cache_verified": True,
            "result_counts": config["result_rows_per_profile_method"],
            "timings_seconds": {"tfidf_fit": round(tfidf_seconds, 3),
                                "embedding_model_load": round(model_load_seconds, 3),
                                "ranking_total": round(ranking_seconds, 3),
                                "total": round(timings["total_run_seconds"], 3)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=PROJECT_ROOT / engine.DEFAULT_SAMPLE)
    args = parser.parse_args()
    result = run(args.plan, args.profiles, args.output_dir, args.sample)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())