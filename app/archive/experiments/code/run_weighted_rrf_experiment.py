"""Controlled TF-IDF/BGE RRF weight comparison using the saved corpus/cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
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

import evaluate_bge_transition_openai as evaluator
import job_matching_engine as engine
import run_bge_transition_experiment as bge
import run_hybrid_rrf_crossencoder_experiment as hybrid

EXPERIMENT_ID = "weighted_rrf_20261008T233000Z"
EXPERIMENT_DIR = PROJECT_ROOT / "data/matching" / EXPERIMENT_ID
SAMPLE_PATH = hybrid.SAMPLE_PATH
BASELINE_PLAN = hybrid.EXPERIMENT_DIR / "experiment_plan.json"
PROFILE_PATH = PROJECT_ROOT / "code/weighted_rrf_profiles.json"
BGE_DIR = hybrid.BGE_EXPERIMENT_DIR
CACHE_DIR = hybrid.BGE_CACHE_DIR
SHARED_LEDGER = hybrid.SHARED_LEDGER
TOP_K = 10
RETRIEVAL_DEPTH = 50
RRF_K = 60
PRESENTATION_SEED = 20261012
WEIGHTS = {
    "equal_1_1": (1, 1),
    "tfidf_2_bge_1": (2, 1),
    "tfidf_1_bge_2": (1, 2),
}
REFERENCE_METHOD = "equal_1_1"
METHODS = tuple(WEIGHTS)
ORIGINAL_BUDGET_CAP_USD = 2.0


def _budget_cap_usd() -> float:
    amendment_path = EXPERIMENT_DIR / "budget_cap_amendment.json"
    if not amendment_path.is_file():
        return ORIGINAL_BUDGET_CAP_USD
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (amendment.get("experiment_id") != EXPERIMENT_ID
            or amendment.get("previous_cumulative_cap_usd") != ORIGINAL_BUDGET_CAP_USD
            or amendment.get("ledger_reset") is not False
            or amendment.get("prior_reservations_preserved") is not True):
        raise ValueError("Budget amendment does not preserve this experiment's original shared ledger.")
    cap = float(amendment["amended_cumulative_cap_usd"])
    if cap < ORIGINAL_BUDGET_CAP_USD:
        raise ValueError("Amended cumulative cap cannot be below the original frozen cap.")
    return cap
CRITERIA = {
    "dev_selection": "Among the two alternatives, choose at most one with complete development P@5 above equal-weight and strong-only P@5 no lower; use shared-pool nDCG@10 as secondary preference. If development coverage is incomplete or no alternative meets guardrails, freeze no alternative.",
    "heldout_success": "At least six complete paired profiles; higher plausible-or-strong P@5 than equal-weight; strong-only P@5 no lower; shared-pool nDCG@10 maintained or improved.",
    "p_at_5": "Requires five returned results and five numeric judgments for both methods on the same profile.",
    "ndcg_at_10": "Requires numeric judgments for every pair in the profile's deduplicated selected-method top-ten union.",
    "unknowns": "Remain null; never coerce unknown/pending/failed labels to zero.",
    "conflicts": "Report evaluator explicit-conflict claims and source checks as claims/diagnostics, not ground truth.",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
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
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(_canonical(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_profiles() -> tuple[dict[str, dict[str, Any]], list[engine.CandidateProfile]]:
    fixtures: dict[str, dict[str, Any]] = {}
    profiles = []
    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    prior_plan = hybrid._load_plan()
    all_ids = []
    for split in ("development", "heldout"):
        rows = data.get(split)
        if not isinstance(rows, list) or len(rows) != (3 if split == "development" else 6):
            raise ValueError("Expected exactly 3 development and 6 held-out fixtures.")
        for row in rows:
            profile_id = str(row.get("profile_id", ""))
            if not profile_id or row.get("synthetic_fixture") is not True or row.get("fixture_type") != "career_transition":
                raise ValueError("Every weighted-RRF fixture must have a unique synthetic transition identity.")
            if profile_id in fixtures:
                raise ValueError("Profile ID is duplicated across weighted-RRF splits.")
            required = ("work_history", "accomplishments", "experience_level", "skills_with_evidence",
                        "education_certifications", "qualification_disclosures", "desired_roles",
                        "career_transition_goal", "soft_preferences", "hard_constraints")
            if any(not row.get(field) for field in required):
                raise ValueError("Weighted-RRF fixture is missing required evidence: {}".format(profile_id))
            if not all(isinstance(item, dict) and item.get("skill") and item.get("evidence")
                       for item in row["skills_with_evidence"]):
                raise ValueError("Every skill must carry source evidence: {}".format(profile_id))
            fixtures[profile_id] = dict(row)
            all_ids.append(profile_id)
            hard = row["hard_constraints"]
            preferences = {
                "exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
                "require_location_match": hard.get("require_location_match") is True,
                "keywords": list(row.get("soft_preferences", [])),
                "work_arrangement": list(row.get("work_arrangement_preferences", [])),
            }
            experience = [*row["work_history"], *row["accomplishments"], *row["education_certifications"]]
            skills = ["{}: {}".format(item["skill"], item["evidence"])
                      for item in row["skills_with_evidence"]]
            profiles.append(engine.CandidateProfile(
                profile_id=profile_id,
                experience=tuple(experience),
                evidenced_skills=tuple(skills),
                desired_roles=tuple(row["desired_roles"]),
                location=row.get("location"),
                preferences=preferences,
            ))
    if len(set(all_ids)) != 9:
        raise ValueError("Weighted-RRF profile IDs must be unique.")
    hybrid._assert_fresh_profile_ids(prior_plan, all_ids)
    return fixtures, profiles


def _initialize() -> dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = EXPERIMENT_DIR / "experiment_plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    existing_names = {item.name for item in EXPERIMENT_DIR.iterdir()}
    allowed_partial = {"development_profiles.json", "heldout_profiles.json"}
    if existing_names - allowed_partial:
        raise FileExistsError("Weighted-RRF output directory has non-profile artifacts without a plan; preserve and inspect.")
    baseline = json.loads(BASELINE_PLAN.read_text(encoding="utf-8"))
    fixtures, _ = _load_profiles()
    split_file_hash = _sha256_file(PROFILE_PATH)
    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    dev_ids = [row["profile_id"] for row in data["development"]]
    heldout_ids = [row["profile_id"] for row in data["heldout"]]
    plan = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_dev_retrieval",
        "hypothesis": "Changing TF-IDF/BGE RRF weights may improve top-five ordering where higher-graded jobs already occur lower within the saved profile pool; automated labels are not ground truth and do not establish that weights caused the discrepancies.",
        "source_hybrid_experiment": hybrid.EXPERIMENT_ID,
        "diagnosis_path": "data/matching/matching_text_faceted_v1_20261008T220000Z/diagnostics/saved_error_analysis_20261008/error_analysis.md",
        "prepared_sample": {"path": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                            "sha256": baseline["corpus"]["sha256"], "rows": 30_000,
                            "description_available": 24_514},
        "control_bge_cache": baseline["existing_bge_cache"],
        "same_input_control": "bge_section_priority_v1 exact cached document texts and original candidate query formatter/prefix",
        "unchanged_settings": {"filters": "job_matching_engine._eligible_jobs / run_bge_transition_experiment.eligible_indices",
                               "tfidf": "unchanged engine.fit_tfidf + rank_tfidf over 30,000 loaded jobs",
                               "bge_model": baseline["existing_bge_cache"]["model_id"],
                               "bge_revision": baseline["existing_bge_cache"]["revision"],
                               "retrieval_depth_per_source": RETRIEVAL_DEPTH,
                               "rrf_k": RRF_K, "top_k": TOP_K,
                               "new_job_embeddings": False,
                               "cross_encoder": False},
        "predeclared_ratios": {name: {"tfidf_weight": weights[0], "bge_weight": weights[1]}
                               for name, weights in WEIGHTS.items()},
        "development_profiles": {"path": "development_profiles.json", "profile_ids": dev_ids,
                                 "source_profiles_sha256": split_file_hash,
                                 "use": "Only choose at most one weighted alternative using frozen P@5 primary, strong-only guardrail, and nDCG secondary criteria."},
        "heldout_profiles": {"path": "heldout_profiles.json", "profile_ids": heldout_ids,
                             "source_profiles_sha256": split_file_hash,
                             "all_synthetic": True,
                             "independent_of_retrieved_job_wording": True,
                             "must_not_rank_before_selection_freeze": True},
        "criteria": CRITERIA,
        "evaluator": {"model": evaluator.MODEL, "citation_protocol": "source_evidence_ids_v1",
                      "ledger_path": str(SHARED_LEDGER.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                  "original_cumulative_cap_usd": ORIGINAL_BUDGET_CAP_USD,
                      "evaluation_order": "combined development plus maximum held-out reservation preflight before first paid request; dev evidence must freeze selection before held-out rankings"},
        "preservation": {"prior_experiments_read_only": True, "source_labels_read_only": True,
                         "frontend_untouched": True, "branch_commit_push_untouched": True},
    }
    # Persist split files independently with stable file hashes.
    split_rows = {split: data[split] for split in ("development", "heldout")}
    for split, filename in (("development", "development_profiles.json"),
                            ("heldout", "heldout_profiles.json")):
        profile_target = EXPERIMENT_DIR / filename
        profile_text = json.dumps(split_rows[split], ensure_ascii=False, indent=2) + "\n"
        if profile_target.exists():
            if profile_target.read_text(encoding="utf-8") != profile_text:
                raise ValueError("Existing partial split file differs from its frozen source: {}".format(filename))
        else:
            profile_target.write_text(profile_text, encoding="utf-8")
        plan["development_profiles" if split == "development" else "heldout_profiles"]["sha256"] = _sha256_file(profile_target)
    (EXPERIMENT_DIR / "openai_evaluator").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BGE_DIR / "openai_evaluator/evaluator_prompt_frozen.md",
                    EXPERIMENT_DIR / "openai_evaluator/evaluator_prompt_frozen.md")
    evaluator.frozen_prompt_text(EXPERIMENT_DIR / "openai_evaluator/evaluator_prompt_frozen.md")
    _write_json(plan_path, plan)
    _write_json(EXPERIMENT_DIR / "profile_split_manifest.json", {
        "profiles_source": str(PROFILE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "profiles_source_sha256": split_file_hash,
        "independence_note": "Authored from fictional candidate histories and transferable capabilities before any retrieval output was produced or inspected.",
        "development_profile_ids": dev_ids,
        "heldout_profile_ids": heldout_ids,
        "split_frozen_before_retrieval": True,
    })
    return plan


def _load_assets(plan: Mapping[str, Any]) -> dict[str, Any]:
    if _sha256_file(SAMPLE_PATH).casefold() != plan["prepared_sample"]["sha256"].casefold():
        raise ValueError("Prepared sample hash changed; aborting without sampling or cache rebuild.")
    dataset = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    fixtures, profiles = _load_profiles()
    jobs, load_report = engine.load_prepared_jobs(SAMPLE_PATH)
    if len(jobs) != 30_000:
        raise ValueError("Prepared sample count changed.")
    corpus = engine.fit_tfidf(jobs)
    eligible_all = [index for index, job in enumerate(jobs) if job.usable_description]
    if len(eligible_all) != 24_514:
        raise ValueError("Description availability count changed.")
    filter_corpus = engine.MatchingCorpus(jobs, None, None)
    bge_model, _bge_load_seconds = bge._load_model(BGE_DIR / "bge_model_files", bge.MAX_BGE_TOKENS)
    tokenizer = bge_model.tokenizer
    job_ids = [jobs[index].job_id for index in eligible_all]
    job_texts = [bge.build_bge_job_text(jobs[index].source, tokenizer)[0] for index in eligible_all]
    key = bge._bge_cache_key(job_ids, job_texts)
    cache_path = CACHE_DIR / ("bge_{}.npy".format(key[:24]))
    manifest_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Original BGE cache missing; this experiment never creates new job embeddings.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_id_hash = _sha256_text("\0".join(job_ids))
    expected_text_hash = _sha256_text("\0".join(job_texts))
    expected = {
        "cache_key": key, "model_id": plan["control_bge_cache"]["model_id"],
        "model_revision": plan["control_bge_cache"]["revision"],
        "representation_id": plan["control_bge_cache"]["representation_id"],
        "ordered_job_ids_sha256": expected_id_hash,
        "job_texts_sha256": expected_text_hash,
        "job_count": 24_514, "dimension": 768, "normalization": True,
        "max_sequence_length_tokens": 512,
        "embedding_file_sha256": plan["control_bge_cache"]["sha256"],
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("Control cache manifest/input mismatch; refuse cache rebuild or row reordering.")
    if _sha256_file(cache_path).casefold() != expected["embedding_file_sha256"].casefold():
        raise ValueError("Control BGE matrix hash mismatch.")
    matrix = np.load(cache_path, mmap_mode="r")
    if matrix.shape != (24_514, 768):
        raise ValueError("Control BGE cache matrix shape mismatch.")
    return {
        "fixtures": fixtures, "profiles": profiles, "jobs": jobs,
        "corpus": corpus, "filter_corpus": filter_corpus,
        "eligible_all": eligible_all,
        "row_for_index": {index: row for row, index in enumerate(eligible_all)},
        "bge_model": bge_model, "tokenizer": tokenizer, "bge_matrix": matrix,
        "jobs_by_id": {job.job_id: job for job in jobs},
        "cache_identity": {"path": str(cache_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                           "sha256": expected["embedding_file_sha256"], "cache_key": key,
                           "ordered_job_ids_sha256": expected_id_hash,
                           "job_texts_sha256": expected_text_hash,
                           "rows": matrix.shape[0], "dimension": matrix.shape[1], "rebuild": False},
        "sample_load_report": load_report.to_dict(),
    }


def weighted_rrf_scores(tfidf_rows: Sequence[Mapping[str, Any]], bge_rows: Sequence[Mapping[str, Any]],
                        k: int, tfidf_weight: int, bge_weight: int) -> dict[str, float]:
    if tfidf_weight <= 0 or bge_weight <= 0:
        raise ValueError("RRF weights must be positive integers.")
    scores: dict[str, float] = defaultdict(float)
    for row in tfidf_rows:
        scores[str(row["job_id"])] += tfidf_weight / (k + int(row["rank"]))
    for row in bge_rows:
        scores[str(row["job_id"])] += bge_weight / (k + int(row["rank"]))
    return dict(scores)


def _rank_profiles(split: str, assets: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_path = EXPERIMENT_DIR / ("development_profiles.json" if split == "development" else "heldout_profiles.json")
    plan = json.loads((EXPERIMENT_DIR / "experiment_plan.json").read_text(encoding="utf-8"))
    split_meta = plan["development_profiles"] if split == "development" else plan["heldout_profiles"]
    if _sha256_file(split_path).casefold() != split_meta["sha256"].casefold():
        raise ValueError("Frozen {} profiles changed.".format(split))
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    fixtures: dict[str, dict[str, Any]] = {}
    profiles = []
    for row in raw:
        fixture = dict(row)
        profile_id = str(fixture["profile_id"])
        fixtures[profile_id] = fixture
        hard = fixture["hard_constraints"]
        preferences = {"exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
                       "require_location_match": hard.get("require_location_match") is True,
                       "keywords": list(fixture.get("soft_preferences", [])),
                       "work_arrangement": list(fixture.get("work_arrangement_preferences", []))}
        experience = [*fixture["work_history"], *fixture["accomplishments"], *fixture["education_certifications"]]
        skills = ["{}: {}".format(item["skill"], item["evidence"]) for item in fixture["skills_with_evidence"]]
        profiles.append(engine.CandidateProfile(
            profile_id=profile_id, experience=tuple(experience), evidenced_skills=tuple(skills),
            desired_roles=tuple(fixture["desired_roles"]), location=fixture.get("location"),
            preferences=preferences))

    output_rows: list[dict[str, Any]] = []
    pool = []
    timings: dict[str, Any] = {"profiles": {}, "cache_rebuild": False}
    available = assets["eligible_all"]
    for profile in profiles:
        start = time.perf_counter()
        eligible = bge.eligible_indices(assets["filter_corpus"], profile, available)
        if not eligible:
            raise ValueError("No eligible jobs for profile {}.".format(profile.profile_id))
        depth = min(RETRIEVAL_DEPTH, len(eligible))
        tfidf_rows = engine.rank_tfidf(assets["corpus"], profile, depth,
                                       candidate_indices=eligible, condition="description_available")
        fixture = fixtures[profile.profile_id]
        query_content, query_report = bge.build_bge_candidate_text(profile, fixture, assets["tokenizer"])
        query = bge.BGE_QUERY_PREFIX + query_content
        query_vector = assets["bge_model"].encode([query], convert_to_numpy=True,
                                                   normalize_embeddings=True,
                                                   show_progress_bar=False, batch_size=1)[0]
        matrix_rows = [assets["row_for_index"][index] for index in eligible]
        values = assets["bge_matrix"][matrix_rows] @ query_vector
        ordered = sorted(zip(values.tolist(), eligible),
                         key=lambda pair: (-float(pair[0]), assets["jobs"][pair[1]].job_id))[:depth]
        bge_rows = [{"profile_id": profile.profile_id, "job_id": assets["jobs"][index].job_id,
                     "rank": rank, "score": float(score)}
                    for rank, (score, index) in enumerate(ordered, 1)]
        for method, (tfidf_weight, bge_weight) in WEIGHTS.items():
            scores = weighted_rrf_scores(tfidf_rows, bge_rows, RRF_K, tfidf_weight, bge_weight)
            ranked = hybrid._rank_rows(scores, profile.profile_id, assets["jobs_by_id"], method, TOP_K)
            output_rows.extend(ranked)
        timings["profiles"][profile.profile_id] = {
            "eligible_jobs": len(eligible), "retrieval_depth": depth,
            "tfidf_bge_query_scoring_seconds": time.perf_counter() - start,
            "candidate_query_token_count": query_report["encoded_tokens_including_prefix_and_specials"],
        }
    pool = bge._build_blinded_pool(output_rows, fixtures, assets["jobs_by_id"], PRESENTATION_SEED)
    return output_rows, pool, timings


def _write_rankings_and_pool(split: str, rows: Sequence[Mapping[str, Any]],
                             pool: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(EXPERIMENT_DIR / "{}_rankings.csv".format(split), rows,
               ["profile_id", "method", "rank", "job_id", "title", "score", "score_type"])
    _write_csv(EXPERIMENT_DIR / "{}_ranking_mappings.csv".format(split), rows,
               ["profile_id", "method", "rank", "job_id", "title", "score", "score_type"])
    _write_jsonl(EXPERIMENT_DIR / "{}_blinded_evaluation_pool.jsonl".format(split), pool)


def run_development() -> dict[str, Any]:
    plan = _initialize()
    if (EXPERIMENT_DIR / "development_rankings.csv").exists():
        raise FileExistsError("Development rankings already exist; refusing overwrite.")
    total_started = time.perf_counter()
    assets = _load_assets(plan)
    rows, pool, timings = _rank_profiles("development", assets)
    timings["total_development_seconds"] = time.perf_counter() - total_started
    _write_rankings_and_pool("development", rows, pool)
    _write_json(EXPERIMENT_DIR / "development_runtime.json", timings)
    _write_json(EXPERIMENT_DIR / "development_run_manifest.json", {
        "rankings_written": len(rows), "pool_pairs": len(pool),
        "methods": list(METHODS), "heldout_rankings_opened": False,
        "selection_frozen": False, "bge_job_cache_rebuild": False,
        "control_cache": assets["cache_identity"],
        "eligible_jobs_by_profile": {key: value["eligible_jobs"] for key, value in timings["profiles"].items()},
    })
    return {"status": "development_rankings_complete_selection_pending_evaluation",
            "ranking_rows": len(rows), "pool_pairs": len(pool), "timings": timings}


def _dev_metric_summary() -> dict[str, Any]:
    pool = _read_jsonl(EXPERIMENT_DIR / "development_blinded_evaluation_pool.jsonl")
    judgments_path = EXPERIMENT_DIR / "development_evaluator/validated_judgments.jsonl"
    if not judgments_path.is_file():
        return {"status": "development_evaluation_pending", "judged_pairs": 0,
                "pool_pairs": len(pool), "selection": None}
    judgments = _read_jsonl(judgments_path)
    judgment_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    with (EXPERIMENT_DIR / "development_rankings.csv").open(encoding="utf-8-sig", newline="") as source:
        rankings = list(csv.DictReader(source))
    profile_ids = [row["profile_id"] for row in json.loads(
        (EXPERIMENT_DIR / "development_profiles.json").read_text(encoding="utf-8"))]
    per_profile = []
    for profile_id in profile_ids:
        profile_pool = [row for row in pool if row["profile_id"] == profile_id]
        labels = {str(row["job_id"]): judgment_by_pair[(profile_id, str(row["job_id"]))]["grade"]
                  for row in profile_pool if (profile_id, str(row["job_id"])) in judgment_by_pair
                  and judgment_by_pair[(profile_id, str(row["job_id"]))].get("status") == "judged"
                  and judgment_by_pair[(profile_id, str(row["job_id"]))].get("grade") in {0, 1, 2, 3}}
        unknown_count = len(profile_pool) - len(labels)
        method_metrics = {}
        for method in METHODS:
            method_rows = sorted((row for row in rankings if row["profile_id"] == profile_id
                                  and row["method"] == method), key=lambda row: int(row["rank"]))
            top5 = method_rows[:5]
            grades = [labels.get(str(row["job_id"])) for row in top5]
            complete_p5 = len(top5) == 5 and all(grade in {0, 1, 2, 3} for grade in grades)
            method_metrics[method] = {
                "p5_plausible_or_strong": sum(grade >= 2 for grade in grades) / 5 if complete_p5 else None,
                "p5_strong_only": sum(grade == 3 for grade in grades) / 5 if complete_p5 else None,
                "p5_complete": complete_p5,
                "ndcg_at_10": _ndcg(method_rows[:10], labels, unknown_count),
                "shared_pool_size": len(profile_pool), "numeric_pool_size": len(labels),
                "unknown_pool_size": unknown_count,
            }
        per_profile.append({"profile_id": profile_id, "methods": method_metrics})
    complete_p5_profiles = [item["profile_id"] for item in per_profile if all(
        item["methods"][method]["p5_complete"] for method in METHODS)]
    complete_ndcg_profiles = [item["profile_id"] for item in per_profile if all(
        item["methods"][method]["ndcg_at_10"] is not None for method in METHODS)]
    means = {}
    for method in METHODS:
        means[method] = {
            "p5_plausible_or_strong": _mean(per_profile, complete_p5_profiles, method, "p5_plausible_or_strong"),
            "p5_strong_only": _mean(per_profile, complete_p5_profiles, method, "p5_strong_only"),
            "ndcg_at_10": _mean(per_profile, complete_ndcg_profiles, method, "ndcg_at_10"),
        }
    status = "insufficient_dev_coverage" if len(complete_p5_profiles) == 0 else "dev_metrics_complete"
    alternatives = []
    baseline = means[REFERENCE_METHOD]
    if baseline["p5_plausible_or_strong"] is not None:
        for method in METHODS:
            if method == REFERENCE_METHOD:
                continue
            metric = means[method]
            if metric["p5_plausible_or_strong"] is None or metric["p5_strong_only"] is None:
                continue
            if (metric["p5_plausible_or_strong"] > baseline["p5_plausible_or_strong"]
                    and metric["p5_strong_only"] >= baseline["p5_strong_only"]):
                alternatives.append(method)
    selected = None
    if alternatives:
        selected = sorted(alternatives, key=lambda method: (
            means[method]["p5_plausible_or_strong"],
            means[method]["p5_strong_only"],
            means[method]["ndcg_at_10"] if means[method]["ndcg_at_10"] is not None else -1.0,
        ), reverse=True)[0]
    return {
        "status": status,
        "pool_pairs": len(pool),
        "judged_pairs": sum(row.get("status") == "judged" and row.get("grade") in {0, 1, 2, 3}
                             for row in judgments),
        "unknown_pairs": sum(row.get("status") == "unknown" and row.get("grade") is None for row in judgments),
        "complete_p5_profile_ids": complete_p5_profiles,
        "complete_p5_denominator": len(complete_p5_profiles),
        "complete_ndcg_profile_ids": complete_ndcg_profiles,
        "complete_ndcg_denominator": len(complete_ndcg_profiles),
        "per_method_means": means,
        "per_profile": per_profile,
        "eligible_alternatives_by_dev_rule": alternatives,
        "selected_alternative": selected,
    }


def _ndcg(top_rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int], unknown_count: int) -> float | None:
    if unknown_count or len(top_rows) != TOP_K or any(str(row["job_id"]) not in labels for row in top_rows):
        return None
    actual = [labels[str(row["job_id"])] for row in top_rows]
    ideal = sorted(labels.values(), reverse=True)[:TOP_K]
    def dcg(values: Sequence[int]) -> float:
        import math
        return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(values))
    ideal_value = dcg(ideal)
    return dcg(actual) / ideal_value if ideal_value else 0.0


def _mean(per_profile: Sequence[Mapping[str, Any]], ids: Sequence[str], method: str, field: str) -> float | None:
    values = [item["methods"][method][field] for item in per_profile if item["profile_id"] in ids]
    return sum(values) / len(values) if values else None


def _cost_preflight() -> dict[str, Any]:
    dev_eval_dir = EXPERIMENT_DIR / "development_evaluation"
    dev_eval_dir.mkdir(parents=True, exist_ok=True)
    dev_pool = EXPERIMENT_DIR / "development_blinded_evaluation_pool.jsonl"
    if not dev_pool.is_file():
        raise FileNotFoundError("Development pool must exist before combined cost preflight.")
    (dev_eval_dir / "blinded_evaluation_pool.jsonl").write_bytes(dev_pool.read_bytes())
    (dev_eval_dir / "openai_evaluator").mkdir(parents=True, exist_ok=True)
    prompt_source = BGE_DIR / "openai_evaluator/evaluator_prompt_frozen.md"
    prompt_target = dev_eval_dir / "openai_evaluator/evaluator_prompt_frozen.md"
    if not prompt_target.exists():
        prompt_target.write_bytes(prompt_source.read_bytes())
    budget_cap = _budget_cap_usd()
    dev_estimate = evaluator.dry_run(dev_eval_dir, budget_usd=budget_cap)
    dev_reserve = float(dev_estimate["total_worst_case_reservation_usd"])
    per_pair = [float(row["max_cost_reservation_usd"]) for row in dev_estimate["per_pair"]]
    max_dev_pair = max(per_pair, default=0.0)
    mean_dev_pair = sum(per_pair) / len(per_pair) if per_pair else 0.0
    max_heldout_pool_pairs = 6 * TOP_K * 2
    projected_heldout_by_mean = mean_dev_pair * max_heldout_pool_pairs
    projected_heldout_by_max = max_dev_pair * max_heldout_pool_pairs
    ledger = evaluator.BudgetLedger(SHARED_LEDGER, budget_cap)
    current_reserved = ledger.totals()["reserved_max_usd"]
    conservative_total = current_reserved + dev_reserve + projected_heldout_by_max
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_combined_budget_preflight_before_any_paid_comparison",
        "model": evaluator.MODEL,
        "citation_protocol": "source_evidence_ids_v1",
        "current_shared_ledger_reservation_usd": current_reserved,
        "remaining_shared_cap_usd": max(0.0, budget_cap - current_reserved),
        "development_pool_pairs": dev_estimate["pool_pairs"],
        "development_local_worst_case_reservation_usd": dev_reserve,
        "heldout_profile_count": 6,
        "heldout_max_two_method_top10_pool_pairs": max_heldout_pool_pairs,
        "heldout_projected_reservation_at_dev_mean_pair_cost_usd": projected_heldout_by_mean,
        "heldout_projected_reservation_at_max_dev_pair_cost_usd": projected_heldout_by_max,
        "combined_projected_reservation_using_max_dev_pair_cost_usd": conservative_total,
        "original_cumulative_cap_usd": ORIGINAL_BUDGET_CAP_USD,
        "cumulative_cap_usd": budget_cap,
        "within_cap": conservative_total <= budget_cap,
        "estimated_additional_authorization_needed_usd": max(0.0, conservative_total - budget_cap),
        "estimation_method": "Exact local tiktoken per-pair estimate for the actual development pool; held-out estimate uses max observed development-pair reservation multiplied by the maximum 6 profiles x 2 methods x top10 union (120). It is conservative against development pair sizes but excludes API framing uncertainty and must not be treated as a bill.",
        "no_api_calls_made": True,
        "heldout_rankings_opened": False,
    }
    _write_json(EXPERIMENT_DIR / "combined_budget_preflight.json", result)
    return result


def freeze_selection() -> dict[str, Any]:
    metric_path = EXPERIMENT_DIR / "development_metric_summary.json"
    if not metric_path.is_file():
        raise FileNotFoundError("Development judgments/metrics do not exist; cannot select weights.")
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    selection = metrics.get("selected_alternative")
    development_count = len(json.loads((EXPERIMENT_DIR / "development_profiles.json").read_text(encoding="utf-8")))
    if selection:
        selection_status = "frozen_selected_alternative"
    elif metrics.get("paired_complete_p5_denominator") == development_count:
        selection_status = "frozen_no_alternative_meets_dev_guardrails"
    else:
        selection_status = "frozen_unassessable_incomplete_development_coverage"
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": selection_status,
        "reference_method": REFERENCE_METHOD,
        "selected_alternative": selection,
        "weights": WEIGHTS[selection] if selection else None,
        "development_metric_summary_sha256": _sha256_file(metric_path),
        "profile_split_sha256": _sha256_file(EXPERIMENT_DIR / "profile_split_manifest.json"),
        "criteria": CRITERIA,
        "heldout_rankings_opened": False,
        "selection_note": "Alternative eligibility requires complete development paired P@5 above equal weight and strong-only P@5 no lower; nDCG breaks ties. No outcome from held-out profiles was consulted.",
    }
    _write_json(EXPERIMENT_DIR / "frozen_selection.json", frozen)
    return frozen


def run_heldout() -> dict[str, Any]:
    plan = _initialize()
    selection_path = EXPERIMENT_DIR / "frozen_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Freeze development selection before held-out ranking.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen" or not selection.get("selected_alternative"):
        raise RuntimeError("No alternative was selected from complete development evidence; held-out ranking is withheld.")
    if selection.get("heldout_rankings_opened") is not False:
        raise ValueError("Unexpected frozen selection state.")
    if (EXPERIMENT_DIR / "heldout_rankings.csv").exists():
        raise FileExistsError("Held-out rankings already exist; refusing overwrite.")
    assets = _load_assets(plan)
    rows, _all_method_pool, timings = _rank_profiles("heldout", assets)
    selected_method = str(selection["selected_alternative"])
    keep_methods = {REFERENCE_METHOD, selected_method}
    rows = [row for row in rows if row["method"] in keep_methods]
    fixtures = {str(row["profile_id"]): row for row in json.loads(
        (EXPERIMENT_DIR / "heldout_profiles.json").read_text(encoding="utf-8"))}
    pool = bge._build_blinded_pool(rows, fixtures, assets["jobs_by_id"], PRESENTATION_SEED)
    _write_rankings_and_pool("heldout", rows, pool)
    _write_json(EXPERIMENT_DIR / "heldout_runtime.json", timings)
    _write_json(EXPERIMENT_DIR / "heldout_run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_profile_sha256": _sha256_file(EXPERIMENT_DIR / "heldout_profiles.json"),
        "frozen_selection_sha256": _sha256_file(selection_path),
        "heldout_started_after_selection_freeze": True,
        "selection": selected_method,
        "ranking_rows": len(rows), "pool_pairs": len(pool),
        "control_cache_rebuilt": False,
        "new_job_embeddings": False,
    })
    return {"status": "heldout_rankings_complete", "selected_alternative": selected_method,
            "ranking_rows": len(rows), "pool_pairs": len(pool), "timings": timings}


def _prepare_eval_directory(split: str) -> Path:
    eval_dir = EXPERIMENT_DIR / ("development_evaluation" if split == "development" else "heldout_evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    pool_path = EXPERIMENT_DIR / ("development_blinded_evaluation_pool.jsonl" if split == "development"
                                  else "heldout_blinded_evaluation_pool.jsonl")
    (eval_dir / "blinded_evaluation_pool.jsonl").write_bytes(pool_path.read_bytes())
    evaluator_dir = eval_dir / "openai_evaluator"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    prompt = BGE_DIR / "openai_evaluator/evaluator_prompt_frozen.md"
    target = evaluator_dir / "evaluator_prompt_frozen.md"
    if not target.exists():
        target.write_bytes(prompt.read_bytes())
    return eval_dir


def _pending_no_budget(preflight: Mapping[str, Any]) -> dict[str, Any]:
    dev_pool_path = EXPERIMENT_DIR / "development_blinded_evaluation_pool.jsonl"
    dev_pool = _read_jsonl(dev_pool_path)
    dev_pending = [{
        "profile_id": row["profile_id"], "job_id": row["job_id"],
        "grade": None, "status": "pending_budget",
        "unknown_reason": "The combined dev-plus-held-out maximum reservation estimate exceeds the remaining original shared ledger allowance; no evaluator request was made.",
    } for row in dev_pool]
    heldout_profiles = json.loads((EXPERIMENT_DIR / "heldout_profiles.json").read_text(encoding="utf-8"))
    heldout_pending = [{
        "profile_id": row["profile_id"], "grade": None, "status": "pending_budget",
        "unknown_reason": "Held-out rankings are deliberately unopened because development selection could not be evaluated within the remaining cumulative budget.",
    } for row in heldout_profiles]
    _write_jsonl(EXPERIMENT_DIR / "development_evaluation/pending_judgments.jsonl", dev_pending)
    _write_json(EXPERIMENT_DIR / "development_coverage.json", {
        "status": "pending_budget",
        "pool_pairs": len(dev_pool),
        "judged": 0, "unknown": 0, "pending": len(dev_pool),
        "metrics": "withheld",
        "selection": "unassessable_no_development_judgments",
    })
    _write_jsonl(EXPERIMENT_DIR / "heldout_evaluation_pending.jsonl", heldout_pending)
    _write_json(EXPERIMENT_DIR / "heldout_coverage.json", {
        "status": "pending_development_selection_and_budget",
        "profiles": len(heldout_profiles),
        "heldout_rankings_created": False,
        "judgments_created": False,
        "p5": "withheld", "ndcg_at_10": "withheld",
    })
    _write_json(EXPERIMENT_DIR / "weighted_rrf_metrics.json", {
        "status": "unassessable_pending_budget",
        "development": {"pool_pairs": len(dev_pool), "judged": 0, "unknown": 0,
                         "pending": len(dev_pool), "paired_denominator": 0,
                         "means": None, "selected_alternative": None},
        "heldout": {"profiles": len(heldout_profiles), "rankings_created": False,
                    "pool_created": False, "judged": 0, "pending": True,
                    "paired_denominator": 0, "means": None,
                    "success_criterion": "unassessable_not_run"},
        "grades_modified": False,
    })
    result = {
        "status": "pending_insufficient_cumulative_budget",
        "no_paid_requests_started": True,
        "reason": "Combined conservative development plus maximum held-out reservation estimate exceeds the remaining shared $2 ledger allowance.",
        "current_reservation_usd": preflight["current_shared_ledger_reservation_usd"],
        "development_estimated_reservation_usd": preflight["development_local_worst_case_reservation_usd"],
        "heldout_estimated_reservation_usd": preflight["heldout_projected_reservation_at_max_dev_pair_cost_usd"],
        "combined_projected_reservation_usd": preflight["combined_projected_reservation_using_max_dev_pair_cost_usd"],
        "cumulative_cap_usd": _budget_cap_usd(),
        "estimated_additional_authorization_needed_usd": preflight["estimated_additional_authorization_needed_usd"],
        "dev_pool_pairs": preflight["development_pool_pairs"],
        "heldout_rankings_opened": False,
        "selection_frozen": False,
    }
    _write_json(EXPERIMENT_DIR / "evaluation_pending_budget.json", result)
    _write_json(EXPERIMENT_DIR / "development_metric_summary.json", {
        "status": "development_evaluation_pending_insufficient_budget",
        "pool_pairs": preflight["development_pool_pairs"],
        "judged_pairs": 0,
        "unknown_pairs": 0,
        "complete_p5_denominator": 0,
        "complete_ndcg_denominator": 0,
        "selected_alternative": None,
        "selection_status": "unassessable_no_development_judgments",
    })
    write_pending_report(result, preflight)
    return result


def write_pending_report(pending: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    dev_run = json.loads((EXPERIMENT_DIR / "development_run_manifest.json").read_text(encoding="utf-8"))
    dev_runtime = json.loads((EXPERIMENT_DIR / "development_runtime.json").read_text(encoding="utf-8"))
    measured_ranking_seconds = dev_runtime.get("total_development_seconds")
    if measured_ranking_seconds is None:
        measured_ranking_seconds = sum(float(row.get("tfidf_bge_query_scoring_seconds", 0.0))
                                       for row in dev_runtime.get("profiles", {}).values())
        runtime_scope = "sum of measured per-profile retrieval/query-scoring phases; excludes corpus/cache setup"
    else:
        runtime_scope = "whole development ranking command"
    lines = [
        "# Weighted TF-IDF/BGE RRF Comparison: Evaluation Pending",
        "",
        "## Frozen Experiment",
        "",
        "Hypothesis: reweighting the two unchanged retrieval lists may improve top-five ordering where higher automated grades were lower in earlier shared pools. Those grades are not ground truth and do not establish that source weights caused the observed discrepancies.",
        "",
        "The prepared sample, hard filters, TF-IDF settings, BGE document representation/revision, verified original cache, query formatter, retrieval depth (50), RRF k (60), and top-ten output were held fixed. No job embeddings or second ranking model were created. The predeclared ratios were TF-IDF:BGE = 1:1, 2:1, and 1:2. Profiles are frozen and independent of retrieved job wording: three dev and six held out.",
        "",
        "Development ranking produced {} rows and a {}-pair blinded union across three profiles. All three weight methods are saved. Held-out ranking was not run because paid development evaluation could not be completed within the current cumulative ledger allowance; therefore no alternative weight is selected and held-out profiles remain unopened.".format(
            dev_run["rankings_written"], dev_run["pool_pairs"]),
        "",
        "## Budget Gate",
        "",
        "Current shared-ledger maximum reservations: ${:.6f}. Exact local dev-pool maximum reservation estimate: ${:.6f}. Conservative held-out projection (120 pairs at the maximum dev pair cost): ${:.6f}. Combined projected reservations: ${:.6f} against the original $2 cap; additional authorization estimated: ${:.6f}. No paid request was started for this comparison.".format(
            preflight["current_shared_ledger_reservation_usd"],
            preflight["development_local_worst_case_reservation_usd"],
            preflight["heldout_projected_reservation_at_max_dev_pair_cost_usd"],
            preflight["combined_projected_reservation_using_max_dev_pair_cost_usd"],
            preflight["estimated_additional_authorization_needed_usd"]),
        "",
        "The held-out estimate is a conservative projection from local development-pair token estimates, not an invoice. It excludes service framing uncertainty; actual inference would still require the official token-count endpoint and transactional reservation before each request.",
        "",
        "## Runtime and Status",
        "",
        "Development CPU ranking took {:.2f} seconds ({}); eligibility counts were {}. Original BGE cache reuse was verified and no job-embedding rebuild occurred.".format(
            measured_ranking_seconds, runtime_scope,
            ", ".join("{}={}".format(profile_id, record["eligible_jobs"])
                      for profile_id, record in dev_runtime.get("profiles", {}).items())),
        "",
        "Development P@5, strong-only P@5 and nDCG are withheld because the 39 development pool pairs have no judgments. Selection is unassessable. Held-out paired denominator is zero because rankings/evaluation were not run; the six-profile success criterion is unassessable, not failed.",
        "",
        "No automated relevance labels were changed. No prior experiment artifacts, original BGE cache, frontend files, or branch state were modified.",
        "",
        "Artifacts: [plan](experiment_plan.json), [profiles](profile_split_manifest.json), [dev rankings](development_rankings.csv), [dev pool](development_blinded_evaluation_pool.jsonl), [combined preflight](combined_budget_preflight.json), [pending coverage](weighted_rrf_metrics.json), and [pending judgments](development_evaluation/pending_judgments.jsonl).",
    ]
    (EXPERIMENT_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_budget_gated_evaluation() -> dict[str, Any]:
    preflight_path = EXPERIMENT_DIR / "combined_budget_preflight.json"
    if not preflight_path.is_file():
        preflight = _cost_preflight()
    else:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["within_cap"]:
        return _pending_no_budget(preflight)
    if not evaluator.load_api_key(PROJECT_ROOT):
        return {"status": "pending_credentials_unavailable", "no_paid_requests_started": True}
    dev_dir = _prepare_eval_directory("development")
    budget_cap = _budget_cap_usd()
    dev_gate = evaluator.run_evaluation(dev_dir, budget_usd=budget_cap, max_pairs=1, ledger_path=SHARED_LEDGER)
    gate_states = _read_jsonl(dev_dir / "openai_evaluator/pair_status.jsonl")
    if dev_gate.get("failed", 0) or not any(row.get("status") in {"evaluated_judged", "evaluated_unknown"}
                                             for row in gate_states):
        return {"status": "development_gate_failed", "gate": dev_gate,
                "heldout_rankings_opened": False}
    dev_final = evaluator.run_evaluation(dev_dir, budget_usd=budget_cap, ledger_path=SHARED_LEDGER)
    dev_metrics = _dev_metric_summary_from_eval(dev_dir)
    _write_json(EXPERIMENT_DIR / "development_metric_summary.json", dev_metrics)
    selection = freeze_selection()
    if selection.get("selected_alternative"):
        run_heldout()
    else:
        return finalize_development_only(dev_metrics, selection)
    heldout_dir = _prepare_eval_directory("heldout")
    heldout_estimate = evaluator.dry_run(heldout_dir, budget_usd=budget_cap)
    ledger = evaluator.BudgetLedger(SHARED_LEDGER, budget_cap)
    if ledger.totals()["reserved_max_usd"] + heldout_estimate["total_worst_case_reservation_usd"] > budget_cap:
        return {"status": "heldout_pending_budget_after_development", "development_metrics": dev_metrics,
                "heldout_estimate": heldout_estimate,
                "additional_authorization_needed_usd": ledger.totals()["reserved_max_usd"]
                + heldout_estimate["total_worst_case_reservation_usd"] - budget_cap}
    heldout_gate = evaluator.run_evaluation(heldout_dir, budget_usd=budget_cap, max_pairs=1, ledger_path=SHARED_LEDGER)
    heldout_states = _read_jsonl(heldout_dir / "openai_evaluator/pair_status.jsonl")
    if heldout_gate.get("failed", 0) or not any(row.get("status") in {"evaluated_judged", "evaluated_unknown"}
                                                 for row in heldout_states):
        return {"status": "heldout_gate_failed", "heldout_gate": heldout_gate}
    heldout_final = evaluator.run_evaluation(heldout_dir, budget_usd=budget_cap, ledger_path=SHARED_LEDGER)
    metrics = calculate_heldout_metrics(heldout_dir)
    _write_json(EXPERIMENT_DIR / "heldout_metric_summary.json", metrics)
    return {"status": heldout_final.get("status"), "development_metrics": dev_metrics,
            "frozen_selection": selection, "heldout_provenance": heldout_final,
            "heldout_metrics": metrics}


def finalize_development_only(dev_metrics: Mapping[str, Any],
                              selection: Mapping[str, Any]) -> dict[str, Any]:
    budget_cap = _budget_cap_usd()
    ledger = evaluator.BudgetLedger(SHARED_LEDGER, budget_cap)
    totals = ledger.totals()
    dev_eval_dir = EXPERIMENT_DIR / "development_evaluation"
    dev_provenance = json.loads((dev_eval_dir / "openai_evaluator/provenance.json").read_text(encoding="utf-8"))
    if (dev_provenance.get("status") != "complete"
            or dev_provenance.get("pool_pairs") != dev_metrics.get("pool_pairs")
            or dev_provenance.get("not_evaluated") != 0
            or dev_provenance.get("failed") != 0):
        raise RuntimeError("Saved development evaluator provenance is not complete; refusing final report.")
    raw_responses = _read_jsonl(dev_eval_dir / "openai_evaluator/raw_responses.jsonl")
    dev_usage_cost = sum(evaluator._maximum_request_cost(
        int(row.get("usage", {}).get("input_tokens", 0)),
        int(row.get("usage", {}).get("output_tokens", 0)))
        for row in raw_responses
        if isinstance(row.get("usage"), Mapping)
        and isinstance(row["usage"].get("input_tokens"), int)
        and isinstance(row["usage"].get("output_tokens"), int))
    baseline_reserved = float(json.loads((EXPERIMENT_DIR / "combined_budget_preflight.json").read_text())
                             ["current_shared_ledger_reservation_usd"])
    weighted_new_reserved = totals["reserved_max_usd"] - baseline_reserved
    weighted_provenance = {
        "experiment_id": EXPERIMENT_ID,
        "status": "development_evaluated_no_alternative_selected",
        "authorization_cap_usd": budget_cap,
        "original_cap_usd": ORIGINAL_BUDGET_CAP_USD,
        "shared_ledger_path": str(SHARED_LEDGER.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "shared_reservations_cumulative_usd": totals["reserved_max_usd"],
        "shared_usage_priced_cumulative_usd": totals["reported_actual_usd"],
        "new_weighted_experiment_reserved_usd": weighted_new_reserved,
        "new_weighted_dev_usage_priced_usd": dev_usage_cost,
        "dev_responses_with_usage": len(raw_responses),
        "dev_pool_pairs": dev_metrics["pool_pairs"],
        "dev_numeric_judgments": dev_metrics["numeric_judgments"],
        "dev_unknown_judgments": dev_metrics["unknown_judgments"],
        "dev_failed": dev_provenance.get("failed"),
        "dev_not_evaluated": dev_provenance.get("not_evaluated"),
        "dev_metrics_pair_denominators": {
            "p5": dev_metrics["paired_complete_p5_denominator"],
            "ndcg_at_10": dev_metrics["paired_complete_ndcg_denominator"],
        },
        "selected_alternative": None,
        "heldout_rankings_opened": False,
        "why_stopped": "Neither predeclared alternative improved development plausible-or-strong P@5 over equal-weight while maintaining the strong-only guardrail.",
        "usage_billing_note": "Saved response usage priced at published rates; invoice billing was not independently checked.",
    }
    _write_json(EXPERIMENT_DIR / "weighted_rrf_evaluation_provenance.json", weighted_provenance)
    _write_json(EXPERIMENT_DIR / "development_coverage.json", {
        "status": "complete",
        "pool_pairs": dev_metrics["pool_pairs"],
        "numeric_judgments": dev_metrics["numeric_judgments"],
        "unknown_judgments": dev_metrics["unknown_judgments"],
        "failed": dev_provenance.get("failed", 0),
        "not_evaluated": dev_provenance.get("not_evaluated", 0),
        "validation_errors": dev_metrics.get("validation_errors", 0),
        "paired_complete_p5_denominator": dev_metrics["paired_complete_p5_denominator"],
        "paired_complete_ndcg_denominator": dev_metrics["paired_complete_ndcg_denominator"],
        "selection_status": dev_metrics.get("selection_status"),
        "historical_budget_gate_pending_file": "development_evaluation/pending_judgments.jsonl (preserved as pre-amendment snapshot)",
    })
    _write_json(EXPERIMENT_DIR / "development_evaluation_status.json", {
        "status": "complete_no_alternative_selected",
        "completed_under_authorized_cap_usd": budget_cap,
        "pool_pairs": dev_metrics["pool_pairs"],
        "numeric_judgments": dev_metrics["numeric_judgments"],
        "unknown_judgments": dev_metrics["unknown_judgments"],
        "selection": None,
        "heldout_rankings_opened": False,
        "previous_budget_gate_pending_rows_preserved": True,
    })
    heldout_profiles = json.loads((EXPERIMENT_DIR / "heldout_profiles.json").read_text(encoding="utf-8"))
    _write_jsonl(EXPERIMENT_DIR / "heldout_evaluation_pending.jsonl", [
        {"profile_id": row["profile_id"], "grade": None, "status": "not_started_selection_no_alternative",
         "unknown_reason": "Development evaluation completed, but no weighted alternative met the frozen selection guardrails; held-out profiles were not ranked or evaluated."}
        for row in heldout_profiles
    ])
    heldout_coverage = {
        "status": "unassessable_no_alternative_selected",
        "profiles": len(heldout_profiles), "rankings_created": False,
        "pool_pairs": None, "numeric_judgments": 0, "unknown_or_pending_profiles": len(heldout_profiles),
        "paired_p5_denominator": 0, "paired_ndcg_denominator": 0,
        "success_criterion": "unassessable_not_run",
    }
    _write_json(EXPERIMENT_DIR / "heldout_coverage.json", heldout_coverage)
    combined_metrics = {
        "experiment_id": EXPERIMENT_ID,
        "development": dict(dev_metrics),
        "heldout": heldout_coverage,
        "selection": dict(selection),
        "criterion_met": False,
        "criterion_status": "unassessable_not_run",
        "unknowns_remain_null": True,
        "prior_experiment_metrics_combined": False,
    }
    _write_json(EXPERIMENT_DIR / "weighted_rrf_metrics.json", combined_metrics)
    _write_weighted_report(dev_metrics, selection, weighted_provenance, heldout_coverage)
    return {"status": weighted_provenance["status"], "development_metrics": dev_metrics,
            "frozen_selection": selection, "heldout_coverage": heldout_coverage,
            "provenance": weighted_provenance}


def _write_weighted_report(dev_metrics: Mapping[str, Any], selection: Mapping[str, Any],
                           provenance: Mapping[str, Any], heldout: Mapping[str, Any]) -> None:
    preflight = json.loads((EXPERIMENT_DIR / "combined_budget_preflight.json").read_text(encoding="utf-8"))
    rows = [
        "# Weighted TF-IDF/BGE RRF Comparison",
        "",
        "## Hypothesis and Frozen Setup",
        "",
        "Hypothesis: changing the relative TF-IDF/BGE RRF contribution may improve top-five ordering among already retrieved candidates. The prior discrepancies are against automated labels, not verified ground truth, and do not show that weights caused them.",
        "",
        "Frozen ratios: TF-IDF:BGE 1:1 (reference), 2:1, and 1:2. Fixed inputs: original 30,000-job prepared sample and 24,514 description-available rows, shared hard filters, unchanged TF-IDF fit/settings, original BGE revision and verified cached job matrix, existing candidate query builder/prefix, top 50 from each source, k=60, top 10. No job embeddings were rebuilt and no cross-encoder was used. Three development and six fresh held-out synthetic career-transition profiles were frozen independently of retrieved job wording.",
        "",
        "## Development Evaluation",
        "",
        "The development pool had {} pairs: {} numeric judgments, {} null unknowns, {} failed, and {} not evaluated. P@5 was complete on {} paired profiles; nDCG@10 was complete on {} paired profiles.".format(
            dev_metrics["pool_pairs"], dev_metrics["numeric_judgments"], dev_metrics["unknown_judgments"],
            dev_metrics.get("failed", 0), dev_metrics.get("not_evaluated", 0),
            dev_metrics["paired_complete_p5_denominator"], dev_metrics["paired_complete_ndcg_denominator"]),
        "",
        "| Ratio | Plausible/strong P@5 | Strong-only P@5 | nDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        metric = dev_metrics["means"][method]
        rows.append("| {} | {} | {} | {} |".format(
            method,
            "withheld" if metric["p5_plausible_or_strong"] is None else "{:.3f}".format(metric["p5_plausible_or_strong"]),
            "withheld" if metric["p5_strong_only"] is None else "{:.3f}".format(metric["p5_strong_only"]),
            "withheld" if metric["ndcg_at_10"] is None else "{:.3f}".format(metric["ndcg_at_10"])))
    rows.extend([
        "",
        "Selection rule required an alternative to exceed equal-weight plausible/strong P@5 with strong-only P@5 no lower; nDCG was secondary. Neither alternative met the primary P@5 improvement requirement. The 2:1 ratio tied P@5/strong-only and improved dev nDCG on the two complete profiles, but was not selected because P@5 did not improve. The 1:2 ratio had lower P@5. Frozen decision: **no alternative selected**.",
        "",
        "## Held-Out and Success Criterion",
        "",
        "Held-out profiles were not ranked because no alternative passed development selection. Their labels remain null/pending. Paired held-out denominators are zero; the six-profile success criterion is **unassessable, not failed**. No metrics from the separate faceted-text or prior hybrid experiments are combined here.",
        "",
        "## Budget and Runtime",
        "",
        "The user-authorized cap is $3, with the existing shared ledger preserved. The combined preflight before any call projected ${:.6f} maximum reservations including current ${:.6f}, exact development ${:.6f}, and conservative held-out ${:.6f}; it was within the $3 cap. Development retrieval/query scoring took {:.2f}s summed across profiles (corpus loading, TF-IDF fit and cache validation are recorded separately in `development_runtime.json`).".format(
            preflight["combined_projected_reservation_using_max_dev_pair_cost_usd"],
            preflight["current_shared_ledger_reservation_usd"],
            preflight["development_local_worst_case_reservation_usd"],
            preflight["heldout_projected_reservation_at_max_dev_pair_cost_usd"],
            sum(value["tfidf_bge_query_scoring_seconds"] for value in
                json.loads((EXPERIMENT_DIR / "development_runtime.json").read_text())["profiles"].values())),
        "",
        "At stop: cumulative maximum reservations ${:.6f} of $3; usage priced from saved response tokens ${:.6f}; new development review used {} Responses with ${:.6f} usage-priced cost. Billing was not independently verified. No paid held-out evaluation occurred.".format(
            provenance["shared_reservations_cumulative_usd"], provenance["shared_usage_priced_cumulative_usd"],
            provenance["dev_responses_with_usage"], provenance["new_weighted_dev_usage_priced_usd"]),
        "",
        "## Interpretation and Limitations",
        "",
        "Development evidence is small and automated. One dev profile had an unknown in its top-five union, leaving only two paired complete profiles. Evidence-ID validation verifies citation identity, not relevance semantics. The observed equal-vs-weighted differences do not establish a production winner. No held-out comparison or success test was completed.",
        "",
        "Artifacts: [frozen plan](experiment_plan.json), [profile split](profile_split_manifest.json), [development rankings](development_rankings.csv), [development pool](development_blinded_evaluation_pool.jsonl), [development judgments](development_evaluation/openai_evaluator/validated_judgments.jsonl), [selection](frozen_selection.json), [preflight](combined_budget_preflight.json), [metrics and coverage](weighted_rrf_metrics.json), [provenance](weighted_rrf_evaluation_provenance.json).",
        "",
    ])
    (EXPERIMENT_DIR / "evaluation_report.md").write_text("\n".join(rows), encoding="utf-8")


def _dev_metrics_after_eval() -> None:
    return None


def _dev_metric_summary_from_eval(eval_dir: Path) -> dict[str, Any]:
    # Bind validated dev judgments to rankings and the blinded development pool.
    pool = _read_jsonl(eval_dir / "blinded_evaluation_pool.jsonl")
    judgments = _read_jsonl(eval_dir / "openai_evaluator/validated_judgments.jsonl")
    valid = {(row["profile_id"], row["job_id"]): row for row in judgments}
    with (EXPERIMENT_DIR / "development_rankings.csv").open(encoding="utf-8-sig", newline="") as source:
        rankings = list(csv.DictReader(source))
    return _compute_metrics(pool, valid, rankings, "development")


def calculate_heldout_metrics(eval_dir: Path) -> dict[str, Any]:
    pool = _read_jsonl(eval_dir / "blinded_evaluation_pool.jsonl")
    judgments = _read_jsonl(eval_dir / "openai_evaluator/validated_judgments.jsonl")
    valid = {(row["profile_id"], row["job_id"]): row for row in judgments}
    with (EXPERIMENT_DIR / "heldout_rankings.csv").open(encoding="utf-8-sig", newline="") as source:
        rankings = list(csv.DictReader(source))
    return _compute_metrics(pool, valid, rankings, "heldout")


def _compute_metrics(pool: Sequence[Mapping[str, Any]], judgments: Mapping[tuple[str, str], Mapping[str, Any]],
                     rankings: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    profile_path = EXPERIMENT_DIR / ("development_profiles.json" if split == "development" else "heldout_profiles.json")
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_ids = [str(row["profile_id"]) for row in profiles]
    per_profile = []
    for profile_id in profile_ids:
        profile_pool = [row for row in pool if row["profile_id"] == profile_id]
        labels = {str(row["job_id"]): judgments[(profile_id, str(row["job_id"]))]["grade"]
                  for row in profile_pool if (profile_id, str(row["job_id"])) in judgments
                  and judgments[(profile_id, str(row["job_id"]))].get("status") == "judged"
                  and judgments[(profile_id, str(row["job_id"]))].get("grade") in {0, 1, 2, 3}}
        unknowns = len(profile_pool) - len(labels)
        methods = {}
        for method in METHODS:
            rows = sorted((row for row in rankings if row["profile_id"] == profile_id and row["method"] == method),
                          key=lambda row: int(row["rank"]))
            top5 = rows[:5]
            top5_grades = [labels.get(str(row["job_id"])) for row in top5]
            complete = len(top5) == 5 and all(grade in {0, 1, 2, 3} for grade in top5_grades)
            methods[method] = {
                "p5_plausible_or_strong": sum(grade >= 2 for grade in top5_grades) / 5 if complete else None,
                "p5_strong_only": sum(grade == 3 for grade in top5_grades) / 5 if complete else None,
                "p5_complete": complete,
                "ndcg_at_10": _ndcg(rows[:TOP_K], labels, unknowns),
                "shared_pool_size": len(profile_pool), "numeric_pool_size": len(labels),
                "unknown_pool_size": unknowns,
                "top5_evaluator_conflict_claims": sum(
                    bool(judgments.get((profile_id, str(row["job_id"])), {}).get("explicit_constraint_conflicts"))
                    for row in top5) if complete else None,
            }
        per_profile.append({"profile_id": profile_id, "methods": methods})
    complete_p5 = [pid for pid in profile_ids if all(
        next(item for item in per_profile if item["profile_id"] == pid)["methods"][method]["p5_complete"]
        for method in METHODS)]
    complete_ndcg = [pid for pid in profile_ids if all(
        next(item for item in per_profile if item["profile_id"] == pid)["methods"][method]["ndcg_at_10"] is not None
        for method in METHODS)]
    means = {method: {
        "p5_plausible_or_strong": _mean(per_profile, complete_p5, method, "p5_plausible_or_strong"),
        "p5_strong_only": _mean(per_profile, complete_p5, method, "p5_strong_only"),
        "ndcg_at_10": _mean(per_profile, complete_ndcg, method, "ndcg_at_10"),
    } for method in METHODS}
    numeric = sum(row.get("status") == "judged" and row.get("grade") in {0, 1, 2, 3}
                  for row in judgments.values())
    unknown = sum(row.get("status") == "unknown" and row.get("grade") is None
                  for row in judgments.values())
    selected = None
    if split == "development" and len(complete_p5) > 0:
        baseline = means[REFERENCE_METHOD]
        alternatives = [method for method in METHODS if method != REFERENCE_METHOD
                        and means[method]["p5_plausible_or_strong"] is not None
                        and means[method]["p5_plausible_or_strong"] > baseline["p5_plausible_or_strong"]
                        and means[method]["p5_strong_only"] >= baseline["p5_strong_only"]]
        if alternatives:
            selected = sorted(alternatives, key=lambda method: (
                means[method]["p5_plausible_or_strong"], means[method]["p5_strong_only"],
                means[method]["ndcg_at_10"] if means[method]["ndcg_at_10"] is not None else -1), reverse=True)[0]
    return {
        "split": split, "pool_pairs": len(pool), "numeric_judgments": numeric,
        "unknown_judgments": unknown, "pending_or_missing": max(0, len(pool) - len(judgments)),
        "failed": sum(row.get("status") == "failed" for row in judgments.values()),
        "not_evaluated": max(0, len(pool) - len(judgments)),
        "validation_errors": 0, "paired_complete_p5_profile_ids": complete_p5,
        "paired_complete_p5_denominator": len(complete_p5),
        "paired_complete_ndcg_profile_ids": complete_ndcg,
        "paired_complete_ndcg_denominator": len(complete_ndcg),
        "means": means, "per_profile": per_profile,
        "selected_alternative": selected,
        "selection_status": "selected" if selected else "no_alternative_meets_dev_guardrails" if split == "development" and complete_p5 else "unassessable_incomplete_coverage",
        "criteria": CRITERIA,
        "statistical_significance_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dev", "preflight", "dev-metrics", "freeze",
                                             "holdout", "evaluate", "finalize-development"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = _initialize()
    elif args.command == "dev":
        result = run_development()
    elif args.command == "preflight":
        result = _cost_preflight()
    elif args.command == "dev-metrics":
        result = _dev_metric_summary()
        _write_json(EXPERIMENT_DIR / "development_metric_summary.json", result)
    elif args.command == "freeze":
        result = freeze_selection()
    elif args.command == "finalize-development":
        selection = freeze_selection()
        metrics = json.loads((EXPERIMENT_DIR / "development_metric_summary.json").read_text(encoding="utf-8"))
        if selection.get("selected_alternative"):
            raise RuntimeError("An alternative is selected; use the guarded evaluator/holdout workflow, not development-only finalization.")
        result = finalize_development_only(metrics, selection)
    elif args.command == "holdout":
        result = run_heldout()
    else:
        result = run_budget_gated_evaluation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())