"""Run a frozen TF-IDF/BGE reciprocal-rank fusion and cross-encoder comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_bge_transition_openai as openai_evaluator
import job_matching_engine as engine
import run_bge_transition_experiment as bge_experiment
import section_aware_embedding as sectioned

EXPERIMENT_ID = "hybrid_rrf_crossencoder_20261008T203000Z"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "matching" / EXPERIMENT_ID
SAMPLE_PATH = PROJECT_ROOT / "data/discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl"
BGE_EXPERIMENT_DIR = PROJECT_ROOT / "data/matching/bge_transition_20261008T142818Z"
BGE_CACHE_DIR = PROJECT_ROOT / "data/matching/.embedding_cache"
SHARED_LEDGER = BGE_EXPERIMENT_DIR / "openai_evaluator/budget_ledger.sqlite3"
CROSS_ENCODER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CROSS_ENCODER_LICENSE = "Apache-2.0"
CROSS_ENCODER_MAX_LENGTH = 512
CANDIDATE_TOKEN_BUDGET = 180
RETRIEVAL_DEPTH = 50
RRF_K = 60
SHORTLIST_GRID = (30, 50)
MAX_PROJECTED_HELDOUT_RERANK_SECONDS = 600.0
TOP_K = 10
POOL_SEED = 20261009
METHOD_TFIDF = "tfidf_cosine"
METHOD_RRF = "tfidf_bge_rrf"
METHOD_CE = "tfidf_bge_rrf_cross_encoder"
METHODS = (METHOD_TFIDF, METHOD_RRF, METHOD_CE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_profiles(path: Path, expected_ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], list[engine.CandidateProfile]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or [row.get("profile_id") for row in raw] != list(expected_ids):
        raise ValueError("Profile file IDs/order differ from the frozen experiment plan.")
    fixtures: dict[str, dict[str, Any]] = {}
    profiles: list[engine.CandidateProfile] = []
    required = ("work_history", "accomplishments", "experience_level", "skills_with_evidence",
                "education_certifications", "qualification_disclosures", "desired_roles",
                "career_transition_goal", "soft_preferences", "hard_constraints")
    for row in raw:
        if row.get("synthetic_fixture") is not True or row.get("fixture_type") != "career_transition":
            raise ValueError("Every experiment profile must be a synthetic career transition.")
        if any(not row.get(field) for field in required):
            raise ValueError("Incomplete experiment profile {}".format(row.get("profile_id")))
        if not all(isinstance(skill, dict) and skill.get("skill") and skill.get("evidence")
                   for skill in row["skills_with_evidence"]):
            raise ValueError("Every demonstrated skill needs evidence: {}".format(row["profile_id"]))
        hard = row["hard_constraints"]
        preferences = {
            "exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
            "require_location_match": hard.get("require_location_match") is True,
            "keywords": list(row.get("soft_preferences", [])),
            "work_arrangement": list(row.get("work_arrangement_preferences", [])),
        }
        experience = [*row["work_history"], *row["accomplishments"], *row["education_certifications"]]
        evidenced_skills = ["{}: {}".format(item["skill"], item["evidence"])
                            for item in row["skills_with_evidence"]]
        profile_id = str(row["profile_id"])
        fixtures[profile_id] = dict(row)
        profiles.append(engine.CandidateProfile(
            profile_id=profile_id, experience=tuple(experience),
            evidenced_skills=tuple(evidenced_skills), desired_roles=tuple(row["desired_roles"]),
            location=row.get("location"), preferences=preferences,
        ))
    return fixtures, profiles


def _assert_fresh_profile_ids(plan: Mapping[str, Any], profile_ids: Sequence[str]) -> None:
    prohibited: set[str] = set()
    prior_paths = [
        BGE_EXPERIMENT_DIR / "synthetic_profiles.json",
        PROJECT_ROOT / "data/matching/fresh_profile_validation_20261007T223454Z/synthetic_profiles.json",
        PROJECT_ROOT / "code/embedding_comparison_profiles.json",
        PROJECT_ROOT / "code/synthetic_candidate_profiles.json",
    ]
    for path in prior_paths:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                prohibited.update(str(row.get("profile_id")) for row in value if isinstance(row, Mapping))
    overlap = sorted(set(profile_ids) & prohibited)
    if overlap:
        raise ValueError("Refusing profile leakage from prior experiments: {}".format(overlap))
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("Experiment profile IDs must be unique.")
    if len(plan["development_profiles"]["profile_ids"]) != 3:
        raise ValueError("Expected three frozen development profiles.")
    if len(plan["heldout_profiles"]["profile_ids"]) != 6:
        raise ValueError("Expected six fresh held-out profiles.")


def _load_cross_encoder():
    import torch
    from sentence_transformers import CrossEncoder

    if torch.cuda.is_available():
        raise RuntimeError("This experiment is frozen for CPU inference.")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.perf_counter()
    model = CrossEncoder(
        CROSS_ENCODER_ID,
        revision=CROSS_ENCODER_REVISION,
        device="cpu",
        max_length=CROSS_ENCODER_MAX_LENGTH,
        trust_remote_code=False,
    )
    if str(model.device) != "cpu":
        raise RuntimeError("Cross-encoder loaded on an unexpected device.")
    config = model.model.config
    if int(config.max_position_embeddings) != CROSS_ENCODER_MAX_LENGTH:
        raise ValueError("Cross-encoder model config does not match its frozen 512-token input limit.")
    return model, time.perf_counter() - started


def _load_assets(plan: Mapping[str, Any], profile_path: Path):
    profile_meta = plan["development_profiles"] if profile_path.name == "development_profiles.json" else plan["heldout_profiles"]
    fixtures, profiles = _load_profiles(profile_path, profile_meta["profile_ids"])
    sample_hash = _sha256_file(SAMPLE_PATH)
    if sample_hash.casefold() != plan["corpus"]["sha256"].casefold():
        raise ValueError("Prepared-sample hash changed; aborting without rebuilding any cache.")
    corpus_load_started = time.perf_counter()
    jobs, load_report = engine.load_prepared_jobs(SAMPLE_PATH)
    corpus_load_seconds = time.perf_counter() - corpus_load_started
    if len(jobs) != 30_000:
        raise ValueError("Prepared sample record count changed.")
    tfidf_fit_started = time.perf_counter()
    corpus = engine.fit_tfidf(jobs)
    tfidf_fit_seconds = time.perf_counter() - tfidf_fit_started
    available_indices = [index for index, job in enumerate(jobs) if job.usable_description]
    if len(available_indices) != 24_514:
        raise ValueError("Description-available eligibility count changed.")
    filter_corpus = engine.MatchingCorpus(jobs, None, None)

    bge_snapshot = BGE_EXPERIMENT_DIR / "bge_model_files"
    bge_model, bge_load_seconds = bge_experiment._load_model(bge_snapshot, bge_experiment.MAX_BGE_TOKENS)
    cache_verify_started = time.perf_counter()
    job_ids = [jobs[index].job_id for index in available_indices]
    job_texts, _ = zip(*(bge_experiment.build_bge_job_text(jobs[index].source, bge_model.tokenizer)
                         for index in available_indices))
    job_texts = list(job_texts)
    bge_key = bge_experiment._bge_cache_key(job_ids, job_texts)
    cache_path = BGE_CACHE_DIR / ("bge_{}.npy".format(bge_key[:24]))
    manifest_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Verified BGE cache is missing; refusing to rebuild job embeddings.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered_ids_hash = hashlib.sha256("\0".join(job_ids).encode("utf-8")).hexdigest()
    job_texts_hash = hashlib.sha256("\0".join(job_texts).encode("utf-8")).hexdigest()
    expected = {
        "cache_key": bge_key,
        "model_id": plan["existing_bge_cache"]["model_id"],
        "model_revision": plan["existing_bge_cache"]["revision"],
        "representation_id": plan["existing_bge_cache"]["representation_id"],
        "ordered_job_ids_sha256": ordered_ids_hash,
        "job_texts_sha256": job_texts_hash,
        "job_count": len(job_ids),
        "dimension": 768,
        "normalization": True,
        "max_sequence_length_tokens": 512,
        "embedding_file_sha256": plan["existing_bge_cache"]["sha256"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("BGE cache manifest, ordered IDs, text hash, or revision mismatch; refusing rebuild.")
    if _sha256_file(cache_path).casefold() != expected["embedding_file_sha256"].casefold():
        raise ValueError("BGE cached matrix checksum mismatch; refusing rebuild.")
    bge_matrix = np.load(cache_path, mmap_mode="r")
    if bge_matrix.shape != (24_514, 768):
        raise ValueError("BGE matrix shape mismatch; refusing rebuild.")
    cache_verification_seconds = time.perf_counter() - cache_verify_started
    cross_model, cross_load_seconds = _load_cross_encoder()
    return {
        "jobs": jobs, "corpus": corpus, "filter_corpus": filter_corpus,
        "available_indices": available_indices, "row_for_corpus_index": {
            index: row for row, index in enumerate(available_indices)
        },
        "bge_model": bge_model, "bge_matrix": bge_matrix,
        "cross_model": cross_model, "fixtures": fixtures, "profiles": profiles,
        "load_report": load_report,
        "runtime": {"corpus_load_seconds": corpus_load_seconds,
                "tfidf_fit_seconds": tfidf_fit_seconds,
                "bge_cache_verification_seconds": cache_verification_seconds,
                "bge_cache_rebuild": False,
                    "bge_model_load_seconds": bge_load_seconds,
                    "cross_encoder_model_load_seconds": cross_load_seconds,
                    "cross_encoder_model_revision": CROSS_ENCODER_REVISION,
                    "cross_encoder_max_position_embeddings": int(cross_model.model.config.max_position_embeddings),
                    "cross_encoder_hidden_layers": int(cross_model.model.config.num_hidden_layers),
                    "cross_encoder_hidden_size": int(cross_model.model.config.hidden_size)},
        "cache": {"path": str(cache_path), "sha256": expected["embedding_file_sha256"],
                  "cache_key": bge_key, "ordered_job_ids_sha256": ordered_ids_hash,
                  "job_texts_sha256": job_texts_hash, "rows": len(job_ids), "dimension": 768},
    }


def _pack_sections(parts: Sequence[tuple[str, str]], tokenizer: Any, budget: int) -> tuple[str, dict[str, Any]]:
    retained: list[tuple[str, str]] = []
    omitted: list[tuple[str, str]] = []
    used = 0
    for label, text in parts:
        piece = "{}: {}".format(label, text)
        piece_tokens = len(tokenizer.encode(piece, add_special_tokens=False))
        if used + piece_tokens <= budget:
            retained.append((label, piece))
            used += piece_tokens
        else:
            omitted.append((label, piece))
    result = " ".join(text for _, text in retained)
    return result, {
        "token_budget": budget,
        "tokens_retained_estimate": used,
        "tokens_omitted_estimate": sum(len(tokenizer.encode(text, add_special_tokens=False)) for _, text in omitted),
        "retained_sections": sorted({label for label, _ in retained}),
        "omitted_sections": sorted({label for label, _ in omitted}),
        "retained_items": len(retained),
        "omitted_items": len(omitted),
    }


def _cross_pair_text(profile: engine.CandidateProfile, fixture: Mapping[str, Any],
                     job: engine.JobRecord, tokenizer: Any) -> tuple[str, str, dict[str, Any]]:
    candidate_parts: list[tuple[str, str]] = []
    for label, values in (
        ("Target roles", fixture.get("desired_roles", [])),
        ("Career transition goal", [fixture.get("career_transition_goal", "")]),
        ("Demonstrated transferable skill", ["{}: {}".format(row.get("skill", ""), row.get("evidence", ""))
                                               for row in fixture.get("skills_with_evidence", [])]),
        ("Work history", fixture.get("work_history", [])),
        ("Accomplishments", fixture.get("accomplishments", [])),
        ("Education and qualifications", fixture.get("education_certifications", [])),
    ):
        candidate_parts.extend((label, str(value)) for value in values if str(value).strip())
    candidate_text, candidate_report = _pack_sections(
        candidate_parts, tokenizer, CANDIDATE_TOKEN_BUDGET - 2,
    )

    title, occupation, description = engine._job_search_fields(job.source)
    job_parts: list[tuple[str, str]] = [("Job title", title)]
    if occupation:
        job_parts.append(("Occupation", occupation))
    sections = sectioned._job_sections(description)
    requirements, duties = sections["requirement"], sections["duty"]
    other = sections["other"]
    tagged: list[tuple[str, str]] = []
    for index in range(max(len(requirements), len(duties))):
        if index < len(requirements):
            tagged.append(("Job requirement", requirements[index]))
        if index < len(duties):
            tagged.append(("Job duty", duties[index]))
    tagged.extend(("Job description", sentence) for sentence in other)
    job_parts.extend(tagged)
    candidate_count = len(tokenizer.encode(candidate_text, add_special_tokens=True))
    job_budget = max(2, CROSS_ENCODER_MAX_LENGTH - candidate_count - 1)
    job_text, job_report = _pack_sections(job_parts, tokenizer, job_budget - 2)
    pair_tokens = len(tokenizer(candidate_text, job_text, add_special_tokens=True, truncation=False)["input_ids"])
    if pair_tokens > CROSS_ENCODER_MAX_LENGTH:
        raise AssertionError("Section packer exceeded cross-encoder pair token limit.")
    original_job_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for _, text in job_parts)
    job_report.update({
        "pair_tokens_including_specials": pair_tokens,
        "cross_encoder_pair_limit": CROSS_ENCODER_MAX_LENGTH,
        "job_tokens_omitted_estimate": max(job_report["tokens_omitted_estimate"],
                                             original_job_tokens - job_report["tokens_retained_estimate"]),
    })
    return candidate_text, job_text, {"candidate": candidate_report, "job": job_report}


def _rank_rows(scores: Mapping[str, float], profile_id: str, jobs_by_id: Mapping[str, engine.JobRecord],
               method: str, top_k: int) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda pair: (-float(pair[1]), pair[0]))[:top_k]
    return [{"profile_id": profile_id, "method": method, "rank": rank, "job_id": job_id,
             "title": jobs_by_id[job_id].searchable_title, "score": float(score),
             "score_type": "cross_encoder_raw_logit" if method == METHOD_CE else "reciprocal_rank_fusion" if method == METHOD_RRF else "tfidf_cosine"}
            for rank, (job_id, score) in enumerate(ordered, 1)]


def _rrf_scores(tfidf_rows: Sequence[Mapping[str, Any]], bge_rows: Sequence[Mapping[str, Any]],
                k: int) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for rows in (tfidf_rows, bge_rows):
        for row in rows:
            scores[str(row["job_id"])] += 1.0 / (k + int(row["rank"]))
    return dict(scores)


def _profile_retrieval(assets: Mapping[str, Any], profile: engine.CandidateProfile,
                       fixture: Mapping[str, Any]) -> dict[str, Any]:
    jobs: Sequence[engine.JobRecord] = assets["jobs"]
    jobs_by_id = {job.job_id: job for job in jobs}
    available = assets["available_indices"]
    included = bge_experiment.eligible_indices(assets["filter_corpus"], profile, available)
    if not included:
        raise ValueError("No eligible description-available jobs for profile {}".format(profile.profile_id))
    source_depth = min(RETRIEVAL_DEPTH, len(included))
    timings: dict[str, float] = {}
    started = time.perf_counter()
    tfidf_rows = engine.rank_tfidf(assets["corpus"], profile, source_depth,
                                   candidate_indices=included,
                                   condition="description_available")
    timings["tfidf_retrieval_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bge_query_content, candidate_bge_report = bge_experiment.build_bge_candidate_text(
        profile, fixture, assets["bge_model"].tokenizer,
    )
    bge_query = bge_experiment.BGE_QUERY_PREFIX + bge_query_content
    bge_query_vector = assets["bge_model"].encode([bge_query], convert_to_numpy=True,
                                                    normalize_embeddings=True,
                                                    show_progress_bar=False, batch_size=1)[0]
    timings["bge_query_encode_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    matrix_rows = [assets["row_for_corpus_index"][index] for index in included]
    bge_scores = assets["bge_matrix"][matrix_rows] @ bge_query_vector
    bge_order = sorted(zip(bge_scores.tolist(), included),
                       key=lambda pair: (-float(pair[0]), jobs[pair[1]].job_id))[:source_depth]
    bge_rows = [{"profile_id": profile.profile_id, "job_id": jobs[index].job_id,
                 "rank": rank, "score": float(score)}
                for rank, (score, index) in enumerate(bge_order, 1)]
    timings["bge_cached_vector_scoring_seconds"] = time.perf_counter() - started
    rrf_scores = _rrf_scores(tfidf_rows, bge_rows, RRF_K)
    rrf_rows = _rank_rows(rrf_scores, profile.profile_id, jobs_by_id, METHOD_RRF, RETRIEVAL_DEPTH)
    return {
        "profile": profile,
        "fixture": fixture,
        "included_indices": included,
        "tfidf_rows": tfidf_rows,
        "bge_rows": bge_rows,
        "rrf_scores": rrf_scores,
        "rrf_rows": rrf_rows,
        "jobs_by_id": jobs_by_id,
        "candidate_bge_input": candidate_bge_report,
        "eligible_job_count": len(included),
        "source_retrieval_depth": source_depth,
        "retrieval_timings": timings,
    }


def _score_cross_pairs(model: Any, records: Sequence[Mapping[str, Any]], batch_size: int = 16) -> tuple[dict[str, float], list[dict[str, Any]], float, float]:
    tokenizer = model.tokenizer
    pair_inputs: list[tuple[str, str]] = []
    reports: list[dict[str, Any]] = []
    prep_started = time.perf_counter()
    for record in records:
        profile = record["profile"]
        fixture = record["fixture"]
        candidate_rows = record.get(
            "cross_candidates",
            record.get("rrf_shortlist", record["rrf_rows"][:RETRIEVAL_DEPTH]),
        )
        for row in candidate_rows:
            job_id = str(row["job_id"])
            job = record["jobs_by_id"][job_id]
            candidate_text, job_text, truncation = _cross_pair_text(profile, fixture, job, tokenizer)
            pair_inputs.append((candidate_text, job_text))
            reports.append({"profile_id": profile.profile_id, "job_id": job_id, **truncation})
    prep_seconds = time.perf_counter() - prep_started
    started = time.perf_counter()
    values = np.asarray(model.predict(pair_inputs, batch_size=batch_size, show_progress_bar=False,
                                      convert_to_numpy=True), dtype=np.float32).reshape(-1)
    score_seconds = time.perf_counter() - started
    if len(values) != len(pair_inputs) or not np.isfinite(values).all():
        raise ValueError("Cross-encoder returned invalid scores.")
    return {
        "{}|{}".format(report["profile_id"], report["job_id"]): float(score)
        for report, score in zip(reports, values)
    }, reports, prep_seconds, score_seconds


def _rows_from_scores(profile_id: str, job_ids: Sequence[str], scores: Mapping[str, float],
                      jobs_by_id: Mapping[str, engine.JobRecord], method: str) -> list[dict[str, Any]]:
    ordered = sorted(((job_id, scores["{}|{}".format(profile_id, job_id)]) for job_id in job_ids),
                     key=lambda pair: (-pair[1], pair[0]))
    return [{"profile_id": profile_id, "method": method, "rank": rank, "job_id": job_id,
             "title": jobs_by_id[job_id].searchable_title, "score": float(score),
             "score_type": "cross_encoder_raw_logit"}
            for rank, (job_id, score) in enumerate(ordered[:TOP_K], 1)]


def _load_plan() -> dict[str, Any]:
    plan_path = EXPERIMENT_DIR / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment_id") != EXPERIMENT_ID or plan.get("frozen_before_dev") is not True:
        raise ValueError("Experiment inputs/settings plan is not frozen or has an unexpected ID.")
    dev_path = EXPERIMENT_DIR / plan["development_profiles"]["path"]
    heldout_path = EXPERIMENT_DIR / plan["heldout_profiles"]["path"]
    if _sha256_file(dev_path).casefold() != str(plan["development_profiles"]["sha256"]).casefold():
        raise ValueError("Development profiles changed after the plan was frozen.")
    if _sha256_file(heldout_path).casefold() != str(plan["heldout_profiles"]["sha256"]).casefold():
        raise ValueError("Held-out profiles changed after the plan was frozen.")
    all_ids = plan["development_profiles"]["profile_ids"] + plan["heldout_profiles"]["profile_ids"]
    _assert_fresh_profile_ids(plan, all_ids)
    return plan


def run_development() -> dict[str, Any]:
    plan = _load_plan()
    output_dir = EXPERIMENT_DIR
    forbidden = ("development_diagnostics.json", "frozen_settings.json", "heldout_rankings.csv",
                 "ranking_mappings.csv", "blinded_evaluation_pool.jsonl")
    if any((output_dir / name).exists() for name in forbidden):
        raise FileExistsError("Development/held-out outputs already exist; refusing overwrite.")
    started = time.perf_counter()
    assets = _load_assets(plan, output_dir / plan["development_profiles"]["path"])
    dev_records = [_profile_retrieval(assets, profile, assets["fixtures"][profile.profile_id])
                   for profile in assets["profiles"]]
    for record in dev_records:
        candidate_rows: dict[str, dict[str, Any]] = {}
        for candidate_k in (20, RRF_K, 100):
            candidate_rrf = _rrf_scores(record["tfidf_rows"], record["bge_rows"], candidate_k)
            for row in _rank_rows(candidate_rrf, record["profile"].profile_id,
                                  record["jobs_by_id"], METHOD_RRF, RETRIEVAL_DEPTH):
                candidate_rows.setdefault(str(row["job_id"]), row)
        record["cross_candidates"] = list(candidate_rows.values())
    candidate_scores, truncation, cross_prep_seconds, cross_score_seconds = _score_cross_pairs(
        assets["cross_model"], dev_records, batch_size=16,
    )
    candidate_count = sum(len(row["cross_candidates"]) for row in dev_records)
    seconds_per_pair = cross_score_seconds / max(1, candidate_count)
    projected = {str(size): seconds_per_pair * len(plan["heldout_profiles"]["profile_ids"]) * size
                 for size in SHORTLIST_GRID}
    feasible = [size for size in SHORTLIST_GRID
                if projected[str(size)] <= MAX_PROJECTED_HELDOUT_RERANK_SECONDS]
    if not feasible:
        _write_json(output_dir / "development_diagnostics.json", {
            "status": "runtime_gate_failed", "model_load_seconds": assets["runtime"]["cross_encoder_model_load_seconds"],
            "development_scored_pairs": candidate_count, "cross_encoder_score_seconds": cross_score_seconds,
            "seconds_per_pair": seconds_per_pair, "projected_seconds_by_shortlist": projected,
            "limit_seconds": MAX_PROJECTED_HELDOUT_RERANK_SECONDS,
        })
        raise RuntimeError("No tested cross-encoder shortlist fits the frozen 10-minute CPU projection gate.")
    chosen_shortlist = max(feasible)
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_profile_ids": [profile.profile_id for profile in assets["profiles"]],
        "development_profiles_sha256": plan["development_profiles"]["sha256"],
        "heldout_profiles_sha256": plan["heldout_profiles"]["sha256"],
        "methods": list(METHODS), "top_k": TOP_K, "retrieval_depth_per_source": RETRIEVAL_DEPTH,
        "rrf_k": RRF_K, "cross_encoder_shortlist_size": chosen_shortlist,
        "shortlist_selection_rule": "largest tested shortlist whose measured dev CPU rerank runtime projects to <=600 seconds for all six held-out profiles; no dev relevance labels or held-out outputs used",
        "projected_heldout_cross_encoder_score_seconds": projected[str(chosen_shortlist)],
        "max_projected_heldout_cross_encoder_seconds": MAX_PROJECTED_HELDOUT_RERANK_SECONDS,
        "cross_encoder_model_id": CROSS_ENCODER_ID, "cross_encoder_revision": CROSS_ENCODER_REVISION,
        "cross_encoder_max_pair_tokens": CROSS_ENCODER_MAX_LENGTH,
        "ranking_heldout_started": False,
    }
    dev_rows: list[dict[str, Any]] = []
    for record in dev_records:
        profile = record["profile"]
        dev_rows.extend(record["tfidf_rows"][:TOP_K])
        for rrf_k in (20, RRF_K, 100):
            fused = _rrf_scores(record["tfidf_rows"], record["bge_rows"], rrf_k)
            rrf_rows = _rank_rows(fused, profile.profile_id, record["jobs_by_id"], METHOD_RRF, RETRIEVAL_DEPTH)
            dev_rows.extend(rrf_rows[:TOP_K])
            shortlist = rrf_rows[:chosen_shortlist]
            dev_method = "dev_ce_rrf_k{}_s{}".format(rrf_k, len(shortlist))
            dev_rows.extend(_rows_from_scores(profile.profile_id, [row["job_id"] for row in shortlist],
                                              candidate_scores, record["jobs_by_id"], dev_method))
    _write_csv(output_dir / "development_rankings.csv", dev_rows,
               ["profile_id", "method", "rank", "job_id", "title", "score", "score_type"])
    timings = {
        "corpus_load_seconds": assets["runtime"]["corpus_load_seconds"],
        "tfidf_fit_seconds": assets["runtime"]["tfidf_fit_seconds"],
        "bge_cache_verification_seconds": assets["runtime"]["bge_cache_verification_seconds"],
        "cross_encoder_config": {
            "revision": assets["runtime"]["cross_encoder_model_revision"],
            "max_position_embeddings": assets["runtime"]["cross_encoder_max_position_embeddings"],
            "hidden_layers": assets["runtime"]["cross_encoder_hidden_layers"],
            "hidden_size": assets["runtime"]["cross_encoder_hidden_size"],
        },
        "development_retrieval_seconds_by_profile": {r["profile"].profile_id: r["retrieval_timings"] for r in dev_records},
        "cross_encoder_model_load_seconds": assets["runtime"]["cross_encoder_model_load_seconds"],
        "bge_query_model_load_seconds": assets["runtime"]["bge_model_load_seconds"],
        "bge_cache_rebuild": False,
        "cross_encoder_input_preparation_seconds": cross_prep_seconds,
        "cross_encoder_scored_pairs": candidate_count,
        "cross_encoder_score_seconds": cross_score_seconds,
        "cross_encoder_seconds_per_pair": seconds_per_pair,
        "projected_heldout_seconds_by_shortlist": projected,
        "truncation_summary": {
            "pairs": len(truncation),
            "pairs_with_candidate_omissions": sum(bool(row["candidate"]["omitted_items"]) for row in truncation),
            "pairs_with_job_omissions": sum(bool(row["job"]["omitted_items"]) for row in truncation),
            "max_pair_tokens": max((row["job"]["pair_tokens_including_specials"] for row in truncation), default=0),
        },
        "total_development_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "development_timings.json", timings)
    _write_jsonl(output_dir / "development_truncation_audit.jsonl", truncation)
    diagnostics = {
        "status": "development_complete_settings_frozen",
        "development_profile_count": len(dev_records),
        "development_profile_ids": [r["profile"].profile_id for r in dev_records],
        "settings": frozen,
        "ranking_rows": len(dev_rows),
        "tfidf_bge_top50_union_sizes": {
            r["profile"].profile_id: len({row["job_id"] for row in r["tfidf_rows"] + r["bge_rows"]})
            for r in dev_records
        },
        "heldout_profiles_used": False,
        "development_labels_used_for_selection": False,
        "interpretation": "Development profiles were used for implementation/filter validation, evidence truncation review, and CPU-only shortlist sizing. No held-out ranking or relevance judgment was consulted.",
    }
    _write_json(output_dir / "development_diagnostics.json", diagnostics)
    _write_json(output_dir / "frozen_settings.json", frozen)
    return diagnostics


def run_heldout_ranking() -> dict[str, Any]:
    plan = _load_plan()
    frozen_path = EXPERIMENT_DIR / "frozen_settings.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Held-out ranking is blocked until development settings are frozen.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("ranking_heldout_started") is not False:
        raise ValueError("Unexpected held-out settings state.")
    if frozen.get("heldout_profiles_sha256") != plan["heldout_profiles"]["sha256"]:
        raise ValueError("Frozen held-out profile hash does not match plan.")
    if frozen.get("rrf_k") != RRF_K or frozen.get("cross_encoder_revision") != CROSS_ENCODER_REVISION:
        raise ValueError("Frozen settings differ from the predeclared model/RRF parameters.")
    outputs = ("heldout_rankings.csv", "ranking_mappings.csv", "blinded_evaluation_pool.jsonl")
    if any((EXPERIMENT_DIR / name).exists() for name in outputs):
        raise FileExistsError("Held-out outputs already exist; refusing to overwrite.")
    started = time.perf_counter()
    assets = _load_assets(plan, EXPERIMENT_DIR / plan["heldout_profiles"]["path"])
    records = [_profile_retrieval(assets, profile, assets["fixtures"][profile.profile_id])
               for profile in assets["profiles"]]
    shortlist_size = int(frozen["cross_encoder_shortlist_size"])
    rerank_candidates: list[dict[str, Any]] = []
    for record in records:
        record["rrf_shortlist"] = record["rrf_rows"][:shortlist_size]
        record["cross_candidates"] = record["rrf_shortlist"]
        rerank_candidates.append(record)
    cross_scores, truncation, cross_prep_seconds, cross_score_seconds = _score_cross_pairs(
        assets["cross_model"], rerank_candidates, batch_size=16,
    )
    ranking_rows: list[dict[str, Any]] = []
    retrieval_timing: dict[str, Any] = {}
    jobs_by_id: dict[str, engine.JobRecord] = {}
    for record in records:
        profile = record["profile"]
        jobs_by_id.update(record["jobs_by_id"])
        tfidf_top10 = record["tfidf_rows"][:TOP_K]
        rrf_top10 = _rank_rows(record["rrf_scores"], profile.profile_id,
                               record["jobs_by_id"], METHOD_RRF, TOP_K)
        ce_rows = _rows_from_scores(profile.profile_id,
                                    [row["job_id"] for row in record["rrf_shortlist"]],
                                    cross_scores, record["jobs_by_id"], METHOD_CE)
        ranking_rows.extend(tfidf_top10)
        ranking_rows.extend(rrf_top10)
        ranking_rows.extend(ce_rows)
        retrieval_timing[profile.profile_id] = record["retrieval_timings"]
    ranking_fields = ["profile_id", "method", "rank", "job_id", "title", "score", "score_type"]
    _write_csv(EXPERIMENT_DIR / "heldout_rankings.csv", ranking_rows, ranking_fields)
    _write_csv(EXPERIMENT_DIR / "ranking_mappings.csv", ranking_rows, ranking_fields)
    pool = bge_experiment._build_blinded_pool(
        ranking_rows, assets["fixtures"], jobs_by_id, POOL_SEED,
    )
    if len(pool) > 180:
        raise AssertionError("Six held-out profiles exceed the fixed 180-pair union cap.")
    _write_jsonl(EXPERIMENT_DIR / "blinded_evaluation_pool.jsonl", pool)
    timings = {
        "corpus_load_seconds": assets["runtime"]["corpus_load_seconds"],
        "tfidf_fit_seconds": assets["runtime"]["tfidf_fit_seconds"],
        "bge_cache_verification_seconds": assets["runtime"]["bge_cache_verification_seconds"],
        "cross_encoder_config": {
            "revision": assets["runtime"]["cross_encoder_model_revision"],
            "max_position_embeddings": assets["runtime"]["cross_encoder_max_position_embeddings"],
            "hidden_layers": assets["runtime"]["cross_encoder_hidden_layers"],
            "hidden_size": assets["runtime"]["cross_encoder_hidden_size"],
        },
        "cross_encoder_model_load_seconds": assets["runtime"]["cross_encoder_model_load_seconds"],
        "bge_query_model_load_seconds": assets["runtime"]["bge_model_load_seconds"],
        "bge_cache_verification_seconds": assets["runtime"].get("bge_cache_verification_seconds"),
        "bge_cache_rebuild": False,
        "retrieval_seconds_by_profile": retrieval_timing,
        "cross_encoder_input_preparation_seconds": cross_prep_seconds,
        "cross_encoder_scored_pairs": len(truncation),
        "cross_encoder_score_seconds": cross_score_seconds,
        "cross_encoder_seconds_per_pair": cross_score_seconds / max(1, len(truncation)),
        "ranking_and_pool_total_seconds": time.perf_counter() - started,
    }
    _write_json(EXPERIMENT_DIR / "heldout_timings.json", timings)
    _write_jsonl(EXPERIMENT_DIR / "heldout_truncation_audit.jsonl", truncation)
    _write_json(EXPERIMENT_DIR / "heldout_run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_profile_ids": [r["profile"].profile_id for r in records],
        "heldout_profiles_sha256": plan["heldout_profiles"]["sha256"],
        "frozen_settings_sha256": _sha256_file(frozen_path),
        "ranking_started_after_settings_freeze": True,
        "rankings_opened_during_retrieval": False,
        "ranking_rows": len(ranking_rows), "pool_pairs": len(pool),
        "methods": list(METHODS), "cache_rebuild": False,
    })
    return {"status": "heldout_rankings_complete", "ranking_rows": len(ranking_rows),
            "pool_pairs": len(pool), "timings": timings}


def calculate_metrics() -> dict[str, Any]:
    pool = openai_evaluator._read_jsonl(EXPERIMENT_DIR / "blinded_evaluation_pool.jsonl")
    judgments = openai_evaluator._read_jsonl(EXPERIMENT_DIR / "openai_evaluator/validated_judgments.jsonl")
    valid_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    with (EXPERIMENT_DIR / "heldout_rankings.csv").open(encoding="utf-8-sig", newline="") as source:
        rankings = list(csv.DictReader(source))
    profile_ids = [row["profile_id"] for row in json.loads(
        (EXPERIMENT_DIR / "heldout_profiles.json").read_text(encoding="utf-8"))]
    per_profile = []
    for profile_id in profile_ids:
        pairs = [row for row in pool if str(row["profile_id"]) == profile_id]
        unknown_pairs = [row for row in pairs if (profile_id, str(row["job_id"])) not in valid_by_pair
                         or valid_by_pair[(profile_id, str(row["job_id"]))].get("status") != "judged"]
        grade_by_job = {str(row["job_id"]): valid_by_pair[(profile_id, str(row["job_id"]))]["grade"]
                         for row in pairs if (profile_id, str(row["job_id"])) in valid_by_pair
                         and valid_by_pair[(profile_id, str(row["job_id"]))].get("status") == "judged"}
        methods: dict[str, Any] = {}
        for method in METHODS:
            rows = sorted([row for row in rankings if row["profile_id"] == profile_id and row["method"] == method],
                          key=lambda row: int(row["rank"]))
            top5, top10 = rows[:5], rows[:10]
            top5_grades = [grade_by_job.get(row["job_id"]) for row in top5]
            p5_complete = len(top5) == 5 and all(value in {0, 1, 2, 3} for value in top5_grades)
            p5 = sum(value >= 2 for value in top5_grades if value is not None) / 5 if p5_complete else None
            strong = sum(value == 3 for value in top5_grades if value is not None) / 5 if p5_complete else None
            ndcg = None
            if not unknown_pairs and len(top10) == 10:
                ideal = sorted(grade_by_job.values(), reverse=True)[:10]
                def dcg(values: Sequence[int]) -> float:
                    return sum((2 ** value - 1) / math.log2(rank + 2) for rank, value in enumerate(values))
                denominator = dcg(ideal)
                actual = [grade_by_job[row["job_id"]] for row in top10]
                ndcg = dcg(actual) / denominator if denominator else 0.0
            conflicts = [valid_by_pair.get((profile_id, row["job_id"])) for row in top5]
            conflicts_complete = len(conflicts) == 5 and all(
                item is not None and item.get("status") == "judged" for item in conflicts)
            conflict_claims = sum(bool(item.get("explicit_constraint_conflicts")) for item in conflicts if item) if conflicts_complete else None
            methods[method] = {
                "retrieved_at_5": len(top5), "judged_top5": sum(value in {0, 1, 2, 3} for value in top5_grades),
                "plausible_or_strong_p5": p5, "strong_only_p5": strong,
                "p5_status": "reported" if p5_complete else "incomplete_or_fewer_than_five",
                "retrieved_at_10": len(top10), "ndcg_at_10": ndcg,
                "ndcg_status": "reported" if ndcg is not None else "incomplete_shared_pool_judgments",
                "shared_pool_size": len(pairs), "judged_shared_pool": len(pairs) - len(unknown_pairs),
                "unknown_shared_pool": len(unknown_pairs),
                "evaluator_claimed_explicit_conflicts_top5": conflict_claims,
                "conflict_claims_status": "reported_evaluator_claims" if conflicts_complete else "incomplete_top5_judgments",
            }
        per_profile.append({"profile_id": profile_id, "methods": methods})
    complete_p5 = [profile_id for profile_id in profile_ids if all(
        next(item for item in per_profile if item["profile_id"] == profile_id)["methods"][method]["plausible_or_strong_p5"] is not None
        for method in METHODS)]
    complete_ndcg = [profile_id for profile_id in profile_ids if all(
        next(item for item in per_profile if item["profile_id"] == profile_id)["methods"][method]["ndcg_at_10"] is not None
        for method in METHODS)]
    def mean(method: str, field: str, profiles: Sequence[str]) -> float | None:
        values = [next(item for item in per_profile if item["profile_id"] == profile)["methods"][method][field]
                  for profile in profiles]
        return sum(values) / len(values) if values else None
    numeric_count = sum(row.get("status") == "judged" and row.get("grade") in {0, 1, 2, 3} for row in judgments)
    unknown_count = sum(row.get("status") == "unknown" and row.get("grade") is None for row in judgments)
    result = {
        "experiment_id": EXPERIMENT_ID, "pool_pairs": len(pool), "numeric_judgments": numeric_count,
        "unknown_judgments": unknown_count, "validation_errors": 0,
        "methods": list(METHODS), "per_profile": per_profile,
        "paired_complete_p5_profile_ids": complete_p5, "paired_complete_p5_denominator": len(complete_p5),
        "paired_p5_plausible_or_strong_mean": {method: mean(method, "plausible_or_strong_p5", complete_p5) for method in METHODS},
        "paired_p5_strong_only_mean": {method: mean(method, "strong_only_p5", complete_p5) for method in METHODS},
        "paired_complete_ndcg_profile_ids": complete_ndcg, "paired_complete_ndcg_denominator": len(complete_ndcg),
        "paired_ndcg_at_10_mean": {method: mean(method, "ndcg_at_10", complete_ndcg) for method in METHODS},
        "evaluator_claimed_conflicts_top5_by_method": {
            method: {
                "profiles_with_complete_judged_top5": sum(
                    row["methods"][method]["conflict_claims_status"] == "reported_evaluator_claims"
                    for row in per_profile),
                "claim_count": sum(
                    row["methods"][method]["evaluator_claimed_explicit_conflicts_top5"] or 0
                    for row in per_profile),
            } for method in METHODS
        },
        "metric_rules": {
            "p_at_5": "Withheld unless five rows were returned and all five have numeric grades.",
            "ndcg_at_10": "Withheld unless every pair in the profile's three-method top-ten shared pool has a numeric grade.",
            "conflicts": "Explicit conflict arrays are evaluator claims, not independent ground truth.",
        },
        "predeclared_relevance_conclusion": "Exploratory comparison only; not human ground truth, hiring-outcome evidence, or production recommendation quality.",
        "statistical_significance_tested": False,
    }
    _write_json(EXPERIMENT_DIR / "metrics_and_coverage.json", result)
    _write_evaluation_report(result)
    return result


def _write_evaluation_report(metrics: Mapping[str, Any]) -> None:
    provenance_path = EXPERIMENT_DIR / "openai_evaluator/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    timings = json.loads((EXPERIMENT_DIR / "heldout_timings.json").read_text(encoding="utf-8"))
    settings = json.loads((EXPERIMENT_DIR / "frozen_settings.json").read_text(encoding="utf-8"))
    lines = [
        "# Hybrid TF-IDF/BGE and Cross-Encoder Evaluation",
        "",
        "## Outcome",
        "",
        "This is an exploratory comparison on six fresh synthetic held-out career-transition profiles and the fixed 30,000-job prepared sample. No retrieval/embedding rebuild was performed. The existing 24,514 x 768 BGE cache was hash- and ordered-ID-verified and memory-mapped.",
        "",
        "The evaluator accepted {} numeric grades and {} unknown/null judgments from {} blinded pairs. Failed: {}; not evaluated: {}.".format(
            metrics["numeric_judgments"], metrics["unknown_judgments"], metrics["pool_pairs"],
            provenance.get("failed", "unknown"), provenance.get("not_evaluated", "unknown")),
        "",
        "## Methods and Frozen Model",
        "",
        "1. TF-IDF baseline using the existing fitted job representation and shared hard filters.",
        "2. TF-IDF + cached BGE using reciprocal-rank fusion with k=60, top-50 source lists, and top-10 output.",
        "3. The same hybrid shortlist reordered by the pretrained cross-encoder, top-10 output.",
        "",
        "Cross-encoder: `{}` at revision `{}`; license {}; max pair length {} tokens including pair special tokens; CPU device. Publisher documentation describes MS MARCO passage ranking and reports performance throughput on a V100 GPU, not CPU. This experiment records its own measured CPU load and inference time below. Raw logits are ranking signals, not probabilities."
        .format(CROSS_ENCODER_ID, CROSS_ENCODER_REVISION, CROSS_ENCODER_LICENSE, CROSS_ENCODER_MAX_LENGTH),
        "Official references: [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [Sentence Transformers pretrained cross-encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html), and [cross-encoder usage/efficiency](https://www.sbert.net/docs/cross_encoder/usage/usage.html).",
        "",
        "Input construction keeps candidate targets, transition goal, work history, accomplishments, evidenced transferable skills, and education visible before lower-priority sections; job title, parsed requirements, and duties are packed first. Per-pair token and omitted-section audit: `heldout_truncation_audit.jsonl`.",
        "",
        "RRF k and the shortlist choice were fixed using development profiles and CPU runtime only. Held-out profiles were ranked only after `frozen_settings.json` was written. No relevance labels from prior BGE profiles were used for tuning.",
        "",
        "## Paired Metrics",
        "",
        "Complete paired P@5 denominator: {} profiles. Complete shared-pool nDCG@10 denominator: {} profiles.".format(
            metrics["paired_complete_p5_denominator"], metrics["paired_complete_ndcg_denominator"]),
        "",
        "| Method | Plausible/strong P@5 | Strong-only P@5 | Shared-pool nDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        p5 = metrics["paired_p5_plausible_or_strong_mean"][method]
        strong = metrics["paired_p5_strong_only_mean"][method]
        ndcg = metrics["paired_ndcg_at_10_mean"][method]
        lines.append("| {} | {} | {} | {} |".format(
            method, "withheld" if p5 is None else "{:.3f}".format(p5),
            "withheld" if strong is None else "{:.3f}".format(strong),
            "withheld" if ndcg is None else "{:.3f}".format(ndcg)))
    lines.extend([
        "",
        "Per-profile metrics and unknown coverage are in `metrics_and_coverage.json`. Explicit conflicts are evaluator claims with coverage, not independently adjudicated ground truth. P@5 is withheld unless all five returned rows are judged; nDCG is withheld unless the complete three-method top-ten pool is numerically judged.",
        "",
        "## Runtime and Cost",
        "",
        "- Retrieval/scoring and cross-encoder reranking are timed separately in `heldout_timings.json`.",
        "- Cross-encoder load: {:.3f}s; held-out pair input preparation: {:.3f}s; {} pairs scored in {:.3f}s on CPU.".format(
            timings["cross_encoder_model_load_seconds"], timings["cross_encoder_input_preparation_seconds"],
            timings["cross_encoder_scored_pairs"], timings["cross_encoder_score_seconds"]),
        "- Development-measured CPU projection for the frozen six-profile shortlist: {:.3f}s; selection ceiling was {:.0f}s.".format(
            settings["projected_heldout_cross_encoder_score_seconds"],
            settings["max_projected_heldout_cross_encoder_seconds"]),
        "- Cumulative shared-ledger reserved maximum: ${:.6f}; usage-priced saved responses: ${:.6f}; actual billing dashboard was not independently inspected.".format(
            provenance.get("reserved_max_usd", 0.0), provenance.get("reported_actual_usd", 0.0)),
        "- Two older attempts have no response/usage record; their reservation remains uncertain. Prior quota-rejected attempts are also preserved in the shared ledger.",
        "",
        "## Limitations and Interpretation",
        "",
        "This small synthetic-profile experiment does not establish human judgment quality, statistical significance, hiring outcomes, or production recommendation quality. The MS MARCO cross-encoder is out-of-domain for employment matching. Automated evidence-ID validation proves that citations resolve to supplied text; it does not prove the relevance judgments are semantically correct. No winner should be interpreted as established.",
        "",
    ])
    (EXPERIMENT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(max_pairs: int | None = None) -> dict[str, Any]:
    if not (EXPERIMENT_DIR / "frozen_settings.json").is_file():
        raise FileNotFoundError("Development settings must be frozen before evaluation.")
    if not (EXPERIMENT_DIR / "blinded_evaluation_pool.jsonl").is_file():
        raise FileNotFoundError("Held-out rankings and blinded pool must exist before evaluation.")
    estimate = openai_evaluator.dry_run(EXPERIMENT_DIR, budget_usd=2.0)
    connection = __import__("sqlite3").connect(SHARED_LEDGER.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        reserved = float(connection.execute(
            "SELECT COALESCE(SUM(reserve_usd), 0) FROM reservations").fetchone()[0])
    finally:
        connection.close()
    combined_estimate = reserved + float(estimate["total_worst_case_reservation_usd"])
    if combined_estimate > 2.0:
        raise RuntimeError(
            "Preflight worst-case estimate ${:.6f} plus existing reservations ${:.6f} exceeds the original $2 budget. No API inference request was made."
            .format(estimate["total_worst_case_reservation_usd"], reserved)
        )
    gate = openai_evaluator.run_evaluation(
        EXPERIMENT_DIR, budget_usd=2.0, max_pairs=1, ledger_path=SHARED_LEDGER,
    )
    states = openai_evaluator._read_jsonl(EXPERIMENT_DIR / "openai_evaluator/pair_status.jsonl")
    gate_state = states[0] if states else {}
    if gate_state.get("status") not in {"evaluated_judged", "evaluated_unknown"}:
        return {"status": "grounding_gate_failed", "gate_pair_state": gate_state,
                "gate_provenance": gate, "full_review_started": False}
    full = openai_evaluator.run_evaluation(
        EXPERIMENT_DIR, budget_usd=2.0, max_pairs=max_pairs, ledger_path=SHARED_LEDGER,
    )
    if full.get("status") != "complete":
        return {"status": "partial", "provenance": full, "metrics_generated": False}
    return {"status": "complete", "provenance": full,
            "metrics": calculate_metrics(), "metrics_generated": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dev", help="Run development diagnostics and freeze settings; no evaluator calls.")
    sub.add_parser("holdout", help="Rank the untouched held-out profiles after frozen settings exist.")
    evaluate = sub.add_parser("evaluate", help="Gate on one pair, then run the existing evidence-ID evaluator against the shared ledger.")
    evaluate.add_argument("--max-pairs", type=int, default=None)
    sub.add_parser("metrics", help="Compute paired metrics/report from saved judgments and rankings.")
    args = parser.parse_args()
    if args.command == "dev":
        result = run_development()
    elif args.command == "holdout":
        result = run_heldout_ranking()
    elif args.command == "evaluate":
        result = run_evaluation(max_pairs=args.max_pairs)
    else:
        result = calculate_metrics()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
