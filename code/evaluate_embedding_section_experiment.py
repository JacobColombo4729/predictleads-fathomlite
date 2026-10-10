"""Validate blinded judgments and compare four frozen retrieval methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_embedding_comparison as evaluator
from section_aware_embedding import REPRESENTATION_ID

METHODS = ("keyword", "tfidf_cosine", "embedding_cosine", REPRESENTATION_ID)
BASE_METHODS = METHODS[:3]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _split_rankings(rankings: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    baseline = [row for row in rankings if row["method"] in BASE_METHODS]
    variant = [{**row, "method": "embedding_cosine"}
               for row in rankings if row["method"] == REPRESENTATION_ID]
    return baseline, variant


def _calc(rankings: list[dict[str, str]], pool: list[dict[str, Any]],
          judgments: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    module_spec = importlib.util.spec_from_file_location(
        "section_experiment_metric_evaluator", CODE_DIR / "evaluate_embedding_comparison.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("Cannot load existing metric evaluator.")
    metric_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metric_module)
    return metric_module.calculate_metrics(rankings, pool, judgments, split)


def evaluate(experiment_dir: Path) -> dict[str, Any]:
    root = experiment_dir.resolve()
    plan = json.loads((root / "validation_plan.json").read_text(encoding="utf-8"))
    split = json.loads((root / "profile_split_manifest.json").read_text(encoding="utf-8"))
    profiles = json.loads((root / "synthetic_profiles.json").read_text(encoding="utf-8"))
    pool = evaluator._read_jsonl(root / "blinded_evaluation_pool.jsonl")
    raw = evaluator._read_jsonl(root / "raw_judgments.jsonl")
    rankings = _read_csv(root / "heldout_rankings.csv")
    mappings = _read_csv(root / "ranking_mappings.csv")
    if not raw:
        raise ValueError("Save blinded direct-review judgments before metric evaluation.")

    valid, errors = evaluator.validate_judgments(pool, raw)
    pool_keys = {(str(row["profile_id"]), str(row["job_id"])) for row in pool}
    rank_identity_keys = Counter((row["profile_id"], row["method"], row["rank"], row["job_id"])
                                 for row in rankings)
    mapping_identity_keys = Counter((row["profile_id"], row["method"], row["rank"], row["job_id"])
                                    for row in mappings)
    duplicate_rank_result = len({(row["profile_id"], row["method"], row["job_id"])
                                 for row in rankings}) != len(rankings)
    all_ranked_pairs_in_pool = all((str(row["profile_id"]), str(row["job_id"])) in pool_keys
                                   for row in rankings)
    if len(pool) > 80:
        raise ValueError("Blinded judgment pool exceeds the frozen 80-pair cap.")
    ids = split["heldout_profile_ids"]
    result_counts = {profile_id: {method: sum(
        row["profile_id"] == profile_id and row["method"] == method for row in rankings
    ) for method in METHODS} for profile_id in ids}
    if any(result_counts[profile_id][method] != 5 for profile_id in ids for method in METHODS):
        raise ValueError("Saved result counts differ from frozen top-five requirement.")
    if any(key[1] not in METHODS for key in rank_identity_keys):
        raise ValueError("Unexpected ranking method in saved rankings.")

    baseline_rankings, variant_comparison_rankings = _split_rankings(rankings)
    baseline_metrics = _calc(baseline_rankings, pool, valid, split)
    variant_metrics = _calc(variant_comparison_rankings, pool, valid, split)
    baseline_by_key = {(row["profile_id"], row["method"]): row
                       for row in baseline_metrics["per_profile_metrics"]}
    variant_by_key = {(row["profile_id"], row["method"]): row
                      for row in variant_metrics["per_profile_metrics"]}
    per_profile = []
    paired_p5_ids: list[str] = []
    paired_ndcg_ids: list[str] = []
    conflict_counts: dict[str, dict[str, int | None]] = {}
    judgment_by_key = {(str(row["profile_id"]), str(row["job_id"])): row for row in valid}
    for profile_id in ids:
        profile_type = next(row["fixture_type"] for row in profiles if row["profile_id"] == profile_id)
        row_metrics = {"profile_id": profile_id, "fixture_type": profile_type,
                       "result_counts": result_counts[profile_id], "methods": {}}
        conflict_counts[profile_id] = {}
        for method in METHODS:
            source_method = "embedding_cosine" if method == REPRESENTATION_ID else method
            metrics_row = (variant_by_key if method == REPRESENTATION_ID else baseline_by_key)[
                (profile_id, source_method)
            ]
            ranked_top = sorted((row for row in rankings if row["profile_id"] == profile_id
                                 and row["method"] == method), key=lambda row: int(row["rank"]))[:5]
            explicit_conflict_count = 0
            for ranked in ranked_top:
                judgment = judgment_by_key.get((profile_id, str(ranked["job_id"])))
                if judgment and judgment.get("status") == "judged":
                    explicit_conflict_count += int(bool(judgment.get("explicit_constraint_conflicts"))
                                                   or judgment.get("required_qualifications") == "unmet")
            conflict_counts[profile_id][method] = explicit_conflict_count
            row_metrics["methods"][method] = {
                "saved_results": result_counts[profile_id][method],
                "shared_pool_size": metrics_row["common_pool_size"],
                "judged_shared_pool": metrics_row["judged_common_pool_items"],
                "unknown_shared_pool": metrics_row["unknown_common_pool_items"],
                "shared_pool_coverage": metrics_row["common_pool_coverage"],
                "top5_returned": metrics_row["retrieved_at_5"],
                "top5_known": metrics_row["judged_at_5"],
                "unknown_top5_ranks": metrics_row["unknown_top5_ranks"],
                "plausible_or_strong_count": metrics_row["plausible_or_strong_count_at_5"],
                "plausible_or_strong_p5": metrics_row["plausible_or_strong_precision_at_5"],
                "strong_count": metrics_row["strong_count_at_5"],
                "strong_only_p5": metrics_row["strong_only_precision_at_5"],
                "ndcg5": metrics_row["ndcg_at_5"],
                "ndcg_status": metrics_row["ndcg_status"],
                "clear_required_conflict_count_top5": explicit_conflict_count,
            }
        if all(row_metrics["methods"][method]["plausible_or_strong_p5"] is not None for method in METHODS):
            paired_p5_ids.append(profile_id)
        if all(row_metrics["methods"][method]["ndcg5"] is not None for method in METHODS):
            paired_ndcg_ids.append(profile_id)
        per_profile.append(row_metrics)

    p5_means = {method: _mean([row["methods"][method]["plausible_or_strong_p5"]
                               for row in per_profile if row["profile_id"] in paired_p5_ids])
                for method in METHODS}
    strong_means = {method: _mean([row["methods"][method]["strong_only_p5"]
                                   for row in per_profile if row["profile_id"] in paired_p5_ids])
                    for method in METHODS}
    ndcg_means = {method: _mean([row["methods"][method]["ndcg5"]
                                 for row in per_profile if row["profile_id"] in paired_ndcg_ids])
                  for method in METHODS}
    base_embedding = "embedding_cosine"
    variant = REPRESENTATION_ID
    criterion = plan["predeclared_success_criterion"]["promising_for_further_validation"]
    clear_conflicts_baseline = sum(conflict_counts[pid][base_embedding] for pid in paired_p5_ids)
    clear_conflicts_variant = sum(conflict_counts[pid][variant] for pid in paired_p5_ids)
    promising = (
        len(paired_p5_ids) >= 3
        and p5_means[variant] is not None
        and p5_means[base_embedding] is not None
        and p5_means[variant] - p5_means[base_embedding] >= 0.10
        and strong_means[variant] >= strong_means[base_embedding]
        and clear_conflicts_variant <= clear_conflicts_baseline
    )
    outcome = "promising for further validation" if promising else "mixed or insufficient; no evidence to replace the provisional embedding choice"
    paired = {
        "p5_profile_ids": paired_p5_ids, "p5_denominator": len(paired_p5_ids),
        "plausible_or_strong_p5_mean": p5_means,
        "strong_only_p5_mean_same_profiles": strong_means,
        "ndcg5_profile_ids": paired_ndcg_ids, "ndcg5_denominator": len(paired_ndcg_ids),
        "ndcg5_mean": ndcg_means,
        "paired_differences_vs_baseline_embedding": {
            "plausible_or_strong_p5": (p5_means[variant] - p5_means[base_embedding]
                                        if paired_p5_ids else None),
            "strong_only_p5": (strong_means[variant] - strong_means[base_embedding]
                               if paired_p5_ids else None),
            "ndcg5": (ndcg_means[variant] - ndcg_means[base_embedding]
                      if paired_ndcg_ids else None),
        },
        "clear_required_conflicts_on_paired_p5_profiles": {
            "baseline_embedding": clear_conflicts_baseline,
            "section_aware_variant": clear_conflicts_variant,
        },
        "predeclared_success_criterion": criterion,
        "criterion_met": promising,
        "outcome": outcome,
    }
    result = {
        "profile_ids": ids, "methods": METHODS,
        "judgment_coverage": {"pool_pairs": len(pool), "numeric": sum(row.get("status") == "judged" for row in valid),
                              "unknown": sum(row.get("status") == "unknown" for row in valid),
                              "validation_errors": len(errors)},
        "per_profile": per_profile, "paired_comparison": paired,
        "interpretation": "All metrics are relative to the shared four-method retrieved top-five union, not the full corpus. This single synthetic experiment does not establish generalization or user benefit.",
    }
    (root / "validated_judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid), encoding="utf-8")
    _write_json(root / "judgment_validation_errors.json", errors)
    _write_json(root / "metrics_and_coverage.json", result)
    profile_rows = []
    conflict_rows = []
    for profile in per_profile:
        for method in METHODS:
            item = profile["methods"][method]
            profile_rows.append({"profile_id": profile["profile_id"], "fixture_type": profile["fixture_type"],
                                 "method": method, "saved_results": result_counts[profile["profile_id"]][method],
                                 "shared_pool_size": item["shared_pool_size"],
                                 "judged_shared_pool": item["judged_shared_pool"],
                                 "unknown_shared_pool": item["unknown_shared_pool"],
                                 "shared_pool_coverage": item["shared_pool_coverage"],
                                 "top5_returned": item["top5_returned"], "top5_known": item["top5_known"],
                                 "unknown_top5_ranks": json.dumps(item["unknown_top5_ranks"]),
                                 "plausible_or_strong_p5": item["plausible_or_strong_p5"],
                                 "strong_only_p5": item["strong_only_p5"], "ndcg5": item["ndcg5"],
                                 "ndcg_status": item["ndcg_status"]})
            conflict_rows.append({"profile_id": profile["profile_id"], "method": method,
                                  "clear_required_conflicts_top5": item["clear_required_conflict_count_top5"]})
    _write_csv(root / "per_profile_metrics.csv", profile_rows,
               ["profile_id", "fixture_type", "method", "saved_results", "shared_pool_size",
                "judged_shared_pool", "unknown_shared_pool", "shared_pool_coverage", "top5_returned",
                "top5_known", "unknown_top5_ranks", "plausible_or_strong_p5", "strong_only_p5",
                "ndcg5", "ndcg_status"])
    _write_csv(root / "top5_requirement_conflicts.csv", conflict_rows,
               ["profile_id", "method", "clear_required_conflicts_top5"])
    title_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row["job_evidence"].get("title", "")
                     for row in pool}
    transition_rows = []
    for judgment in valid:
        relevance = judgment.get("career_transition_relevance", {})
        if (judgment.get("status") == "judged" and judgment.get("grade") in {2, 3}
                and relevance.get("level") in {"moderate", "high"}):
            transition_rows.append({"profile_id": judgment["profile_id"], "job_id": judgment["job_id"],
                                    "title": title_by_pair[(str(judgment["profile_id"]), str(judgment["job_id"]))],
                                    "grade": judgment["grade"], "transition_relevance": relevance["level"],
                                    "rationale": relevance["rationale"],
                                    "status": "model-assessed possibility; not a validated discovery"})
    _write_csv(root / "transition_assessments.csv", transition_rows,
               ["profile_id", "job_id", "title", "grade", "transition_relevance", "rationale", "status"])
    validation = {
        "raw_rows": len(raw), "valid_rows": len(valid), "errors": len(errors),
        "pool_pairs": len(pool), "unique_pool_pairs": len(pool_keys),
        "rank_rows": len(rankings), "mapping_rows": len(mappings),
        "rank_mapping_identity_match": rank_identity_keys == mapping_identity_keys,
        "ranked_pair_identity_in_pool": all_ranked_pairs_in_pool,
        "duplicate_profile_method_job": duplicate_rank_result,
        "all_four_methods_have_five_per_profile": all(result_counts[p][m] == 5 for p in ids for m in METHODS),
        "pool_blinded_of_ranker_fields": all(not ({"method", "rank", "score", "score_type"} & set(row)) for row in pool),
    }
    validation["all_checks_passed"] = (not errors and validation["rank_mapping_identity_match"]
                                       and all_ranked_pairs_in_pool and not duplicate_rank_result
                                       and validation["all_four_methods_have_five_per_profile"]
                                       and validation["pool_blinded_of_ranker_fields"]
                                       and len(pool) <= 80 and len(pool_keys) == len(pool))
    _write_json(root / "contract_validation.json", validation)
    hashes = {}
    for name in ("heldout_rankings.csv", "ranking_mappings.csv", "blinded_evaluation_pool.jsonl",
                 "raw_judgments.jsonl", "validated_judgments.jsonl", "metrics_and_coverage.json",
                 "per_profile_metrics.csv", "top5_requirement_conflicts.csv", "transition_assessments.csv"):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    _write_json(root / "artifact_hashes.json", hashes)
    _write_report(root, plan, result, validation)
    return {"validation": validation, "paired_comparison": paired}


def _fmt(value: float | None) -> str:
    return "withheld" if value is None else "{:.3f}".format(value)


def _write_report(root: Path, plan: dict[str, Any], result: dict[str, Any], validation: dict[str, Any]) -> None:
    timings_path = root / "retrieval_timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.is_file() else {
        "measurement_status": "unavailable; retrieval timing artifact was not written",
        "corpus_load_seconds": None, "tfidf_fit_seconds": None, "model_load_seconds": None,
        "baseline_embedding_cache_reused": None,
        "section_aware_text_preprocessing_seconds": None,
        "section_aware_job_encoding_seconds_observed_console": None,
        "baseline_method_ranking_seconds": None, "section_aware_query_ranking_seconds": None,
        "pool_build_seconds": None, "total_retrieval_seconds": None,
    }
    def fmt_time(key: str) -> str:
        value = timings.get(key)
        return "not captured" if value is None else "{:.2f}s".format(value)

    variant_encode = timings.get("section_aware_job_encoding_seconds")
    if variant_encode is None:
        variant_encode = timings.get("section_aware_job_encoding_seconds_observed_console")
    variant_encode_note = "not captured" if variant_encode is None else "approximately {:.0f}s (progress display)".format(variant_encode)
    paired = result["paired_comparison"]
    methods = result["methods"]
    lines = [
        "# Section-Aware Embedding Validation", "",
        "## Method", "",
        "One frozen input-representation variant was compared with keyword, TF-IDF, and original MiniLM embeddings. The variant uses the same pinned local model and 256-token maximum, but removes repeated query/title/occupation text and prioritizes labeled candidate evidence plus job requirements/duties before lower-priority preferences and non-boilerplate remainder.", "",
        "Evaluation method: **Direct coding-assistant model review of saved blinded candidate-job evidence.** Model identity is **unknown**; no API call. This session has inspected prior aggregate results and diagnostic rankings, so method/rank blinding does not make the reviewer independent. Judgments are not ground truth or human validation.", "",
        "## Coverage and Runtime", "",
        "Four synthetic profiles produced {} rankings (five per each of four methods/profile) and a deduplicated shared pool of {} pairs. Judgment coverage: {} numeric, {} unknown, {} validation errors. Unknown is null, not grade zero.".format(
            validation["rank_rows"], validation["pool_pairs"], result["judgment_coverage"]["numeric"],
            result["judgment_coverage"]["unknown"], result["judgment_coverage"]["validation_errors"]), "",
        "Runtime: corpus load {}; TF-IDF fit {}; model load {}; baseline cache verification {}; section-aware text preprocessing {}; variant job encoding {}; baseline ranking {}; variant query/ranking {}; pool build/shuffle {}; total retrieval {}. Baseline cache status: {}. Timing completeness: {}.".format(
            fmt_time("corpus_load_seconds"), fmt_time("tfidf_fit_seconds"), fmt_time("model_load_seconds"),
            fmt_time("baseline_embedding_cache_verify_seconds"), fmt_time("section_aware_text_preprocessing_seconds"),
            variant_encode_note, fmt_time("baseline_method_ranking_sum_seconds"),
            fmt_time("variant_query_and_ranking_sum_seconds"), fmt_time("pool_build_and_shuffle_seconds"),
            fmt_time("total_retrieval_seconds"),
            "verified/reused" if timings.get("baseline_embedding_cache_reused") or timings.get("baseline_embedding_cache_verified_hit") else "status recorded in frozen config",
            timings.get("measurement_status", "complete")), "",
        "## Per-Profile Results", "",
        "P@5 requires five returned and five judged jobs. nDCG@5 uses the four-method top-five union and is withheld if any union judgment is unknown.", "",
        "| Profile | Type | Method | Pool known/size | Top 5 known | Plausible P@5 | Strong-only P@5 | nDCG@5 | Hard requirement conflicts in top 5 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile in result["per_profile"]:
        for method in methods:
            item = profile["methods"][method]
            lines.append("| {} | {} | {} | {}/{} | {}/5 | {} | {} | {} | {} |".format(
                profile["profile_id"], profile["fixture_type"], method,
                item["judged_shared_pool"], item["shared_pool_size"], item["top5_known"],
                _fmt(item["plausible_or_strong_p5"]), _fmt(item["strong_only_p5"]),
                _fmt(item["ndcg5"]), item["clear_required_conflict_count_top5"]))
    lines.extend(["", "## Paired Results", "",
                  "P@5 denominator {}: {}. nDCG denominator {}: {}.".format(
                      paired["p5_denominator"], ", ".join(paired["p5_profile_ids"]) or "none",
                      paired["ndcg5_denominator"], ", ".join(paired["ndcg5_profile_ids"]) or "none"), "",
                  "| Method | Plausible-or-strong P@5 | Strong-only P@5 | nDCG@5 |",
                  "|---|---:|---:|---:|"])
    for method in methods:
        lines.append("| {} | {} | {} | {} |".format(
            method, _fmt(paired["plausible_or_strong_p5_mean"][method]),
            _fmt(paired["strong_only_p5_mean_same_profiles"][method]),
            _fmt(paired["ndcg5_mean"][method])))
    lines.extend(["", "Differences versus original embeddings (variant minus baseline): plausible-or-strong P@5 {}, strong-only P@5 {}, nDCG@5 {}.".format(
        _fmt(paired["paired_differences_vs_baseline_embedding"]["plausible_or_strong_p5"]),
        _fmt(paired["paired_differences_vs_baseline_embedding"]["strong_only_p5"]),
        _fmt(paired["paired_differences_vs_baseline_embedding"]["ndcg5"])), "",
        "## Outcome", "", "**{}**. Predeclared criterion met: {}. The experiment is limited to four synthetic profiles and one sampled historical corpus; model review was not independent, and no user benefit/current-job availability is established. Incomplete profiles are shown above and excluded only from paired metrics according to the frozen completeness rules. This top-five union's nDCG must not be compared to earlier top-ten-union nDCG.".format(
            paired["outcome"], paired["criterion_met"]), "",
        "See `validation_plan.json`, `synthetic_profiles.json`, `frozen_configuration.json`, `heldout_rankings.csv`, `ranking_mappings.csv`, `blinded_evaluation_pool.jsonl`, `raw_judgments.jsonl`, `validated_judgments.jsonl`, `per_profile_metrics.csv`, `metrics_and_coverage.json`, `top5_requirement_conflicts.csv`, `transition_assessments.csv`, `retrieval_timings.json`, `contract_validation.json`, `artifact_hashes.json`, and `reproduce.md`.", ""])
    (root / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.experiment_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())