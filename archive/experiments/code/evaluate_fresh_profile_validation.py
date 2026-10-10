"""Validate blinded fresh-profile judgments and report paired retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import evaluate_embedding_comparison as evaluator


METHODS = ("keyword", "tfidf_cosine", "embedding_cosine")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_per_profile_csv(path: Path, metrics: dict[str, Any], fixture_types: dict[str, str]) -> None:
    fields = ["profile_id", "fixture_type", "method", "saved_results", "parsed_results",
              "common_pool_size", "judged_common_pool_items", "unknown_common_pool_items",
              "common_pool_coverage", "top5_returned", "top5_judged", "unknown_top5_ranks",
              "plausible_or_strong_count", "plausible_or_strong_p5", "plausible_or_strong_status",
              "strong_count", "strong_only_p5", "strong_only_status", "ndcg_at_5", "ndcg_status"]
    rows = []
    counts = metrics["result_counts"]
    for result in metrics["per_profile_metrics"]:
        rows.append({
            "profile_id": result["profile_id"], "fixture_type": fixture_types[result["profile_id"]],
            "method": result["method"],
            "saved_results": counts[result["profile_id"]][result["method"]],
            "parsed_results": counts[result["profile_id"]][result["method"]],
            "common_pool_size": result["common_pool_size"],
            "judged_common_pool_items": result["judged_common_pool_items"],
            "unknown_common_pool_items": result["unknown_common_pool_items"],
            "common_pool_coverage": result["common_pool_coverage"],
            "top5_returned": result["retrieved_at_5"], "top5_judged": result["judged_at_5"],
            "unknown_top5_ranks": json.dumps(result["unknown_top5_ranks"]),
            "plausible_or_strong_count": result["plausible_or_strong_count_at_5"],
            "plausible_or_strong_p5": result["plausible_or_strong_precision_at_5"],
            "plausible_or_strong_status": result["plausible_or_strong_precision_status"],
            "strong_count": result["strong_count_at_5"],
            "strong_only_p5": result["strong_only_precision_at_5"],
            "strong_only_status": result["strong_only_precision_status"],
            "ndcg_at_5": result["ndcg_at_5"], "ndcg_status": result["ndcg_status"],
        })
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(items: list[float]) -> float | None:
    return sum(items) / len(items) if items else None


def evaluate(experiment_dir: Path) -> dict[str, Any]:
    root = experiment_dir.resolve()
    plan = json.loads((root / "validation_plan.json").read_text(encoding="utf-8"))
    split = json.loads((root / "profile_split_manifest.json").read_text(encoding="utf-8"))
    pool = evaluator._read_jsonl(root / "blinded_evaluation_pool.jsonl")
    raw = evaluator._read_jsonl(root / "raw_judgments.jsonl")
    rankings = _read_csv(root / "heldout_rankings.csv")
    mappings = _read_csv(root / "ranking_mappings.csv")
    fixture_types = {row["profile_id"]: row["fixture_type"]
                     for row in json.loads((root / "synthetic_profiles.json").read_text(encoding="utf-8"))}
    expected_pairs = {(str(row["profile_id"]), str(row["job_id"])) for row in pool}
    valid, errors = evaluator.validate_judgments(pool, raw)
    if not raw:
        raise ValueError("Save direct-review outputs to raw_judgments.jsonl before evaluation.")

    pool_pair_counts = Counter(str(row["profile_id"]) for row in pool)
    rank_pair_counts = Counter((str(row["profile_id"]), str(row["method"])) for row in rankings)
    mapping_keys = Counter((row["profile_id"], row["method"], row["rank"], row["job_id"]) for row in mappings)
    ranking_keys = Counter((row["profile_id"], row["method"], row["rank"], row["job_id"]) for row in rankings)
    prohibited = {"method", "rank", "score", "score_type"}
    top_level_blinding_violations = sorted(prohibited.intersection(row) for row in pool)
    top_level_blinding_violations = [items for items in top_level_blinding_violations if items]
    identities_in_pool = all((str(row["profile_id"]), str(row["job_id"])) in expected_pairs for row in rankings)
    duplicate_profile_method_job = len({(row["profile_id"], row["method"], row["job_id"]) for row in rankings}) != len(rankings)
    counts = {
        profile_id: {method: rank_pair_counts[(profile_id, method)] for method in METHODS}
        for profile_id in split["heldout_profile_ids"]
    }
    expected_row_count = sum(counts[profile_id][method]
                             for profile_id in counts for method in METHODS)
    if expected_row_count != len(rankings) or any(not (1 <= count <= 5)
                                                   for row in counts.values() for count in row.values()):
        raise ValueError("Saved rankings violate the frozen actual-count/top-five scope.")

    metrics = evaluator.calculate_metrics(rankings, pool, valid, split)
    for result in metrics["per_profile_metrics"]:
        result["fixture_type"] = fixture_types[result["profile_id"]]
    p5_ids = metrics["aggregate_paired_comparison"]["common_evaluable_profile_ids_for_plausible_or_strong_p5"]
    metric_by_key = {(row["profile_id"], row["method"]): row for row in metrics["per_profile_metrics"]}
    strong_ids = [profile_id for profile_id in split["heldout_profile_ids"]
                  if all(metric_by_key[(profile_id, method)]["strong_only_precision_at_5"] is not None
                         for method in METHODS)]
    ndcg_ids = [profile_id for profile_id in split["heldout_profile_ids"]
                if all(metric_by_key[(profile_id, method)]["ndcg_at_5"] is not None for method in METHODS)]
    plausible_means = metrics["aggregate_paired_comparison"]["plausible_or_strong_p5_mean_by_method"]
    strong_means = {method: _mean([metric_by_key[(pid, method)]["strong_only_precision_at_5"]
                                   for pid in strong_ids]) for method in METHODS}
    ndcg_means = {method: _mean([metric_by_key[(pid, method)]["ndcg_at_5"]
                                 for pid in ndcg_ids]) for method in METHODS}
    if p5_ids != strong_ids:
        raise AssertionError("Plausible and strong-only P@5 must use the same complete-top-five profiles.")
    aggregate = {
        "plausible_or_strong_p5_profile_ids": p5_ids,
        "plausible_or_strong_p5_denominator": len(p5_ids),
        "plausible_or_strong_p5_mean_by_method": plausible_means,
        "strong_only_p5_profile_ids": strong_ids,
        "strong_only_p5_denominator": len(strong_ids),
        "strong_only_p5_mean_by_method": strong_means,
        "ndcg5_profile_ids_with_complete_shared_pool": ndcg_ids,
        "ndcg5_denominator": len(ndcg_ids),
        "ndcg5_mean_by_method": ndcg_means,
        "paired_differences_vs_keyword": {
            "plausible_or_strong_p5": {method: plausible_means[method] - plausible_means["keyword"]
                                        for method in METHODS[1:]} if p5_ids else {},
            "strong_only_p5": {method: strong_means[method] - strong_means["keyword"]
                               for method in METHODS[1:]} if strong_ids else {},
            "ndcg5": {method: ndcg_means[method] - ndcg_means["keyword"]
                      for method in METHODS[1:]} if ndcg_ids else {},
        },
    }
    emb = "embedding_cosine"
    refs = ("keyword", "tfidf_cosine")
    if p5_ids and all(plausible_means[emb] > plausible_means[ref]
                      and strong_means[emb] > strong_means[ref] for ref in refs):
        decision = "supports retaining embeddings provisionally"
    elif p5_ids and all(plausible_means[emb] < plausible_means[ref]
                        and strong_means[emb] < strong_means[ref] for ref in refs):
        decision = "challenges the provisional embedding choice"
    else:
        decision = "mixed; retain embeddings provisionally for further validation, not as an established winner"

    metrics["result_counts"] = counts
    metrics["shared_pool_scope"] = "deduplicated union of each profile's three top-five lists"
    metrics["paired_fresh_profile_comparison"] = aggregate
    metrics["frozen_interpretation"] = {"decision": decision,
                                        "rule": plan["frozen_metric_rules"]["decision_rule"]}
    metrics["metric_caveat"] = "Fresh nDCG uses a top-five union; historical comparison nDCG used a top-ten union and is not directly comparable."
    (root / "validated_judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid), encoding="utf-8")
    _write_json(root / "judgment_validation_errors.json", errors)
    _write_json(root / "metrics_and_coverage.json", metrics)
    _write_per_profile_csv(root / "per_profile_metrics.csv", metrics, fixture_types)

    validation = {
        "raw_judgment_count": len(raw), "validated_judgment_count": len(valid),
        "validation_error_count": len(errors), "numeric_judgments": metrics["judged_pair_count"],
        "unknown_or_pending_count": len(pool) - metrics["judged_pair_count"],
        "pool_pair_count": len(pool), "unique_pool_pair_count": len(expected_pairs),
        "ranking_row_count": len(rankings), "mapping_row_count": len(mappings),
        "ranking_mapping_identity_match": mapping_keys == ranking_keys,
        "all_ranking_identities_in_pool": identities_in_pool,
        "duplicate_profile_method_job": duplicate_profile_method_job,
        "blinded_pool_has_no_top_level_ranker_fields": not top_level_blinding_violations,
        "pool_size_per_profile": dict(pool_pair_counts),
        "all_validations_passed": (len(errors) == 0 and mapping_keys == ranking_keys
                                    and identities_in_pool and not duplicate_profile_method_job
                                    and not top_level_blinding_violations
                                    and len(expected_pairs) == len(pool)),
    }
    _write_json(root / "contract_validation.json", validation)
    _write_json(root / "artifact_hashes.json", {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("heldout_rankings.csv", "ranking_mappings.csv", "blinded_evaluation_pool.jsonl",
                     "raw_judgments.jsonl", "validated_judgments.jsonl", "metrics_and_coverage.json")
    })
    _write_report(root, plan, metrics, validation, decision)
    return {"validation": validation, "decision": decision, "paired": aggregate}


def _format(value: float | None) -> str:
    return "withheld" if value is None else f"{value:.3f}"


def _write_report(root: Path, plan: dict[str, Any], metrics: dict[str, Any],
                  validation: dict[str, Any], decision: str) -> None:
    runtime = json.loads((root / "retrieval_timings.json").read_text(encoding="utf-8"))
    rows = []
    for result in metrics["per_profile_metrics"]:
        pid, method = result["profile_id"], result["method"]
        rows.append(
            f"| {pid} | {result['fixture_type'] if 'fixture_type' in result else ''} | {method} | "
            f"{metrics['result_counts'][pid][method]} | {result['common_pool_size']} "
            f"({result['judged_common_pool_items']} known; {result['unknown_common_pool_items']} unknown) | "
            f"{result['retrieved_at_5']} ({result['judged_at_5']} known; "
            f"{len(result['unknown_top5_ranks'])} unknown) | {_format(result['plausible_or_strong_precision_at_5'])} | "
            f"{_format(result['strong_only_precision_at_5'])} | {_format(result['ndcg_at_5'])} |"
        )
    paired = metrics["paired_fresh_profile_comparison"]
    means_table = []
    for method in METHODS:
        means_table.append(
            f"| {method} | {_format(paired['plausible_or_strong_p5_mean_by_method'][method])} | "
            f"{_format(paired['strong_only_p5_mean_by_method'][method])} | "
            f"{_format(paired['ndcg5_mean_by_method'][method])} |"
        )
    report = [
        "# Fresh-Profile Embedding Validation", "",
        f"Validation ID: `{plan['validation_id']}`. Four synthetic profiles (two conventional matches, two transitions) were evaluated once on the existing description-available corpus.", "",
        "## Method and Independence", "",
        f"Method: **{plan['frozen_evaluation']['actual_method_to_record']}** Model identity: **unknown**; this session provides no verifiable identifier. No API call was made. The coding-assistant session has seen the earlier aggregate findings, so blinding method/rank/score fields does not make this review independent of the project or prior results. Judgments are not human validation, ground truth, or user outcomes.", "",
        "The frozen 0–3/null rubric was applied only to saved blinded candidate/job evidence. Reviewer inputs exclude method, rank, score, and mappings. Unknown remains null, never zero. Required versus preferred qualifications, known conflicts versus missing information, and historical listing status were kept distinct.", "",
        "## Retrieval and Runtime", "",
        f"All three methods were run once per profile with top five requested. Saved ranking rows: **{len(_read_csv(root / 'heldout_rankings.csv'))}**; shared deduplicated top-five union: **{len(evaluator._read_jsonl(root / 'blinded_evaluation_pool.jsonl'))} pairs** (cap 60). Actual per-profile/method counts and post-filter eligible corpus sizes are recorded in `per_profile_metrics.csv` and `frozen_configuration.json`.", "",
        f"The existing job-embedding cache was verified and reused (no job re-encoding): SHA-256 `{json.loads((root / 'frozen_configuration.json').read_text(encoding='utf-8'))['settings']['embedding_cache_sha256_after']}`. TF-IDF was fit once over the same 30,000 loader-eligible jobs with the original settings because no fitted vectorizer or matrix artifact was saved previously. Runtime: corpus load {runtime['corpus_load_seconds']:.2f}s; TF-IDF fit {runtime['tfidf_fit_seconds']:.2f}s; model load {runtime['embedding_model_load_seconds']:.2f}s; ranking queries {runtime['ranking_total_seconds']:.2f}s; total run {runtime['total_run_seconds']:.2f}s. Per-query timings are in `retrieval_timings.json`.", "",
        "## Per-Profile Metrics", "",
        "P@5 reports only when five results were returned and all five have numeric grades. Plausible-or-strong counts grades 2–3; strong-only counts grade 3. nDCG@5 uses gain `2^grade - 1` and the shared top-five union, and is withheld when any pool judgment is unknown. `withheld` means the frozen completeness rule was not met.", "",
        "| Profile | Type | Method | Results | Shared pool | Top five (known / unknown) | Plausible P@5 | Strong-only P@5 | nDCG@5 |",
        "|---|---|---|---:|---|---|---:|---:|---:|", *rows, "",
        "## Paired Comparison", "",
        f"Paired P@5 uses **{paired['plausible_or_strong_p5_denominator']} profiles**: {', '.join(paired['plausible_or_strong_p5_profile_ids']) or 'none'}. Strong-only uses the same denominator. Paired nDCG uses **{paired['ndcg5_denominator']} profiles** with complete shared pools: {', '.join(paired['ndcg5_profile_ids_with_complete_shared_pool']) or 'none'}.", "",
        "| Method | Plausible-or-strong P@5 | Strong-only P@5 | nDCG@5 |", "|---|---:|---:|---:|", *means_table, "",
        "Paired mean differences versus keyword (TF-IDF; embedding): "
        + "; ".join(f"{key}: {', '.join(_format(v) for v in values.values())}"
                     for key, values in paired["paired_differences_vs_keyword"].items()) + ".",
        "",
        "## Interpretation", "",
        f"Under the decision rule frozen before retrieval: **{decision}**. No pass threshold was added after viewing results. This is one small synthetic validation and cannot establish generalization, production readiness, or user benefit. No claims are made that historical listings are currently open. Fresh nDCG is based on a top-five union, unlike the earlier top-ten union; their nDCG values must not be compared directly.", "",
        "Check results: "
        f"{validation['validated_judgment_count']}/{validation['pool_pair_count']} validated judgments; "
        f"{validation['numeric_judgments']} numeric, {validation['unknown_or_pending_count']} unknown/pending; "
        f"{validation['validation_error_count']} validation errors; contract checks {'passed' if validation['all_validations_passed'] else 'have failures'}.", "",
        "## Artifacts", "",
        "See `validation_plan.json`, `synthetic_profiles.json`, `frozen_configuration.json`, `heldout_rankings.csv`, `ranking_mappings.csv`, `blinded_evaluation_pool.jsonl`, `raw_judgments.jsonl`, `validated_judgments.jsonl`, `per_profile_metrics.csv`, `metrics_and_coverage.json`, `retrieval_timings.json`, `contract_validation.json`, and `reproduce.md`.", "",
    ]
    (root / "validation_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.experiment_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())