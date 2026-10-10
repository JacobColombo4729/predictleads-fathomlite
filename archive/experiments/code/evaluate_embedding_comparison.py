"""Validate fixed-prompt judgments and score saved embedding-comparison rankings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

GRADE_VALUES = {0, 1, 2, 3}
METHODS = ("keyword", "tfidf_cosine", "embedding_cosine")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _dcg(grades: Sequence[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def validate_judgments(
    pool: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject invalid/duplicate/out-of-pool model outputs without coercion."""
    expected = {(str(row["profile_id"]), str(row["job_id"])): row for row in pool}
    seen: set[tuple[str, str]] = set()
    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    required = {
        "profile_id", "job_id", "grade", "status", "unknown_reason",
        "supporting_candidate_evidence", "supporting_job_evidence", "material_gaps",
        "explicit_constraint_conflicts", "required_qualifications",
        "career_transition_relevance", "title_outside_stated_targets", "evidence_sufficiency",
    }
    for index, response in enumerate(responses, 1):
        key = (str(response.get("profile_id", "")), str(response.get("job_id", "")))
        issue = None
        if key not in expected:
            issue = "response_pair_not_in_blinded_pool"
        elif key in seen:
            issue = "duplicate_response_pair"
        elif response.get("status") == "pending_evaluator":
            if response.get("grade") is not None or not str(response.get("unknown_reason", "")).strip():
                issue = "pending_evaluator_requires_null_grade_and_reason"
            else:
                seen.add(key)
                valid.append(dict(response))
                continue
        else:
            missing = sorted(required - set(response))
            if missing:
                issue = "missing_required_fields:" + ",".join(missing)
            else:
                grade = response["grade"]
                status = response["status"]
                if grade is not None and (isinstance(grade, bool) or not isinstance(grade, int) or grade not in GRADE_VALUES):
                    issue = "grade_must_be_integer_0_to_3_or_null"
                elif status == "judged" and grade is None:
                    issue = "judged_status_requires_numeric_grade"
                elif status == "unknown" and (grade is not None or not isinstance(response["unknown_reason"], str)
                                               or not response["unknown_reason"].strip()):
                    issue = "unknown_status_requires_null_grade_and_reason"
                elif status not in {"judged", "unknown"}:
                    issue = "status_must_be_judged_or_unknown"
                elif response["required_qualifications"] not in {"met", "unmet", "unknown"}:
                    issue = "invalid_required_qualifications_value"
                elif response["evidence_sufficiency"] not in {"sufficient", "insufficient"}:
                    issue = "invalid_evidence_sufficiency_value"
                elif not isinstance(response["title_outside_stated_targets"], bool):
                    issue = "title_outside_stated_targets_must_be_boolean"
                elif not isinstance(response["career_transition_relevance"], dict) or response["career_transition_relevance"].get("level") not in {"low", "moderate", "high", "unknown"} or not isinstance(response["career_transition_relevance"].get("rationale"), str):
                    issue = "invalid_career_transition_relevance_object"
                elif any(not isinstance(response[field], list) or not all(isinstance(item, str) for item in response[field])
                         for field in ("supporting_candidate_evidence", "supporting_job_evidence", "material_gaps", "explicit_constraint_conflicts")):
                    issue = "evidence_and_gap_fields_must_be_string_arrays"
        if issue:
            errors.append({"response_index": index, "profile_id": key[0], "job_id": key[1], "error": issue})
            continue
        seen.add(key)
        valid.append(dict(response))
    for key in expected:
        if key not in seen:
            errors.append({"profile_id": key[0], "job_id": key[1], "error": "missing_response_remains_unknown"})
    return valid, errors


def calculate_metrics(
    rankings: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    profile_split: Mapping[str, Any],
) -> dict[str, Any]:
    heldout = list(profile_split["heldout_profile_ids"])
    pool_ids: dict[str, set[str]] = {profile_id: set() for profile_id in heldout}
    for row in pool:
        pool_ids[str(row["profile_id"])].add(str(row["job_id"]))
    grades = {(str(row["profile_id"]), str(row["job_id"])): row["grade"]
              for row in judgments if row.get("status") == "judged" and row.get("grade") in GRADE_VALUES}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rankings:
        profile_id = str(row["profile_id"])
        if profile_id in pool_ids:
            grouped[(profile_id, str(row["method"]))].append(row)

    per_profile: list[dict[str, Any]] = []
    scalar_metrics: dict[str, dict[str, float | None]] = {method: {} for method in METHODS}
    for profile_id in heldout:
        ids = sorted(pool_ids[profile_id])
        unknown_pool = [job_id for job_id in ids if (profile_id, job_id) not in grades]
        ideal: list[int] = []
        ndcg_pool_complete = not unknown_pool
        if ndcg_pool_complete:
            ideal = sorted((grades[(profile_id, job_id)] for job_id in ids), reverse=True)[:5]
            ideal_dcg = _dcg(ideal)
        else:
            ideal_dcg = 0.0
        for method in METHODS:
            rows = sorted(grouped.get((profile_id, method), ()), key=lambda row: int(row["rank"]))
            top5 = rows[:5]
            judged = [(int(row["rank"]), grades[(profile_id, str(row["job_id"]))])
                      for row in top5 if (profile_id, str(row["job_id"])) in grades]
            unknown_top5 = [int(row["rank"]) for row in top5
                            if (profile_id, str(row["job_id"])) not in grades]
            complete_p5 = len(top5) == 5 and not unknown_top5
            plausible_count = sum(grade >= 2 for _, grade in judged)
            strong_count = sum(grade == 3 for _, grade in judged)
            plausible_p5 = plausible_count / 5 if complete_p5 else None
            strong_p5 = strong_count / 5 if complete_p5 else None
            if complete_p5:
                scalar_metrics[method][profile_id] = plausible_p5
            if not rows:
                p_status = "no_rankings"
            elif len(top5) < 5:
                p_status = "fewer_than_five_results"
            elif unknown_top5:
                p_status = "incomplete_top5_judgments"
            else:
                p_status = "reported"
            ndcg = None
            if not ids:
                ndcg_status = "empty_common_pool"
            elif unknown_pool:
                ndcg_status = "incomplete_common_pool_judgments"
            elif not rows:
                ndcg_status = "no_rankings"
            else:
                actual = [grades[(profile_id, str(row["job_id"]))] for row in top5]
                actual_dcg = _dcg(actual)
                if ideal_dcg == 0:
                    ndcg = 0.0
                    ndcg_status = "reported_zero_ideal_gain"
                else:
                    ndcg = actual_dcg / ideal_dcg
                    ndcg_status = "reported_common_pool_relative"
            per_profile.append({
                "profile_id": profile_id, "method": method, "retrieved_at_5": len(top5),
                "judged_at_5": len(judged), "unknown_top5_ranks": unknown_top5,
                "plausible_or_strong_count_at_5": plausible_count,
                "strong_count_at_5": strong_count,
                "plausible_or_strong_precision_at_5": plausible_p5,
                "plausible_or_strong_precision_status": p_status,
                "strong_only_precision_at_5": strong_p5,
                "strong_only_precision_status": p_status,
                "common_pool_size": len(ids), "judged_common_pool_items": len(ids) - len(unknown_pool),
                "unknown_common_pool_items": len(unknown_pool),
                "common_pool_coverage": (len(ids) - len(unknown_pool)) / len(ids) if ids else 0.0,
                "ndcg_at_5": ndcg, "ndcg_status": ndcg_status,
                "ndcg_gain": "2^grade-1", "ndcg_ideal_pool_size": min(5, len(ids)),
                "ndcg_zero_ideal_gain_handling": "report 0 when the fully judged ideal has zero gain",
                "title_outside_target_count": sum(bool(row.get("title_outside_stated_targets")) for row in judgments
                                                   if row.get("profile_id") == profile_id and row.get("status") == "judged"),
                "transition_relevance_known_count": sum(
                    row.get("career_transition_relevance", {}).get("level") not in (None, "unknown")
                    for row in judgments if row.get("profile_id") == profile_id and row.get("status") == "judged"),
            })

    common_profiles = [profile_id for profile_id in heldout
                       if all(profile_id in scalar_metrics[method] for method in METHODS)]
    aggregate: dict[str, Any] = {
        "common_evaluable_profile_ids_for_plausible_or_strong_p5": common_profiles,
        "common_evaluable_profile_denominator": len(common_profiles),
        "plausible_or_strong_p5_mean_by_method": {
            method: (sum(scalar_metrics[method][profile_id] for profile_id in common_profiles) / len(common_profiles)
                     if common_profiles else None) for method in METHODS
        },
        "paired_differences_vs_keyword": {
            method: ({profile_id: scalar_metrics[method][profile_id] - scalar_metrics["keyword"][profile_id]
                      for profile_id in common_profiles} if common_profiles else {})
            for method in ("tfidf_cosine", "embedding_cosine")
        },
        "statistical_significance_tested": False,
    }
    return {
        "profile_id_scope": heldout,
        "method_scope": METHODS,
        "judgment_grade_scale": {"0": "unsuitable", "1": "weak", "2": "plausible", "3": "strong"},
        "judged_pair_count": len(grades),
        "pending_or_unknown_pair_count": len(pool) - len(grades),
        "heldout_pair_count": len(pool),
        "per_profile_metrics": per_profile,
        "aggregate_paired_comparison": aggregate,
        "interpretation": "Metrics are relative to each held-out profile's union of retrieved top-10 jobs, not the full corpus. Judgments are model-based when supplied and do not represent human outcomes.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--responses", type=Path, default=None,
                        help="JSONL structured outputs from the one approved evaluator model/prompt")
    parser.add_argument("--evaluator-model", default=None,
                        help="exact evaluator model identifier; required with --responses")
    parser.add_argument("--evaluator-settings-json", default='{"temperature":0,"response_format":"jsonl"}')
    parser.add_argument("--evaluator-tokens", type=int, default=None)
    parser.add_argument("--evaluator-cost-usd", type=float, default=None)
    args = parser.parse_args(argv)
    root = args.experiment_dir.resolve()
    if args.responses is not None and not args.evaluator_model:
        parser.error("--evaluator-model is required when supplying model responses")
    pool = _read_jsonl(root / "blinded_evaluation_pool.jsonl")
    rankings = list(csv.DictReader((root / "heldout_rankings.csv").open(encoding="utf-8-sig", newline="")))
    split = json.loads((root / "profile_split_manifest.json").read_text(encoding="utf-8"))
    if args.responses is None:
        responses = _read_jsonl(root / "validated_judgments.jsonl")
        raw_responses = []
    else:
        responses = _read_jsonl(args.responses.resolve())
        raw_responses = responses
        shutil.copyfile(args.responses.resolve(), root / "raw_evaluator_responses.jsonl")
    valid, errors = validate_judgments(pool, responses)
    (root / "validated_judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid), encoding="utf-8")
    (root / "judgment_validation_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    metrics = calculate_metrics(rankings, pool, valid, split)
    (root / "metrics_and_coverage.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    evaluator_state = "model-based judgments supplied" if raw_responses else "pending; no evaluator responses supplied"
    report = [
        "# Embedding Retrieval Comparison: Evaluation Report", "",
        "This bounded comparison covers eight held-out synthetic profiles and one sampled historical job corpus. It does not measure user outcomes or verified job availability. Similarity scores are retrieval signals, not hiring likelihood.", "",
        f"Judgments accepted: {metrics['judged_pair_count']} of {metrics['heldout_pair_count']} blinded pool pairs. The evaluator state is: {evaluator_state}.", "",
        "## Metric Rules", "",
        "Plausible-or-strong Precision@5 counts grades 2 or 3 and requires five returned results with five known grades. Strong-only Precision@5 counts grade 3 under the same completeness rule. Unknown grades remain unknown, never zero. nDCG@5 uses gain 2^grade - 1 and a shared per-profile ideal from the union of every method's retrieved top-10 jobs; it is withheld if any union judgment is unknown. A fully judged zero-gain ideal reports nDCG 0. This is retrieval-pool-relative, not full-corpus recall.", "",
    ]
    if metrics["judged_pair_count"] == 0:
        report.extend(["No relevance metric is reported because there are no approved evaluator judgments. Plausible-or-strong Precision@5, strong-only Precision@5, and nDCG@5 remain withheld.", ""])
    else:
        report.extend(["See `metrics_and_coverage.json` for per-profile metrics, coverage, unknown counts, withholding reasons, and paired comparisons on the common evaluable-profile denominator.", ""])
    report.extend([
        "## Limits and Next Step", "",
        "No broad winner is claimed. The evaluation is model-based if judgments are supplied, imperfect, and limited to one historical sample and eight held-out synthetic profiles. It cannot establish application competition, employer intent, or hiring likelihood. Next, use one approved fixed evaluator model and this frozen prompt once on the blinded pool; inspect disagreement cases before making product claims.", "",
    ])
    (root / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    config_path = root / "frozen_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["evaluation_model"] = {
        "status": evaluator_state,
        "model": args.evaluator_model,
        "settings": json.loads(args.evaluator_settings_json),
        "response_rows": len(raw_responses), "validated_rows": len(valid),
        "validation_error_rows": len(errors),
        "tokens": args.evaluator_tokens,
        "cost_usd": args.evaluator_cost_usd,
        "cost_status": "unknown" if args.evaluator_cost_usd is None else "reported_by_operator",
        "raw_response_sha256": (__import__("hashlib").sha256(args.responses.read_bytes()).hexdigest()
                                 if args.responses is not None else None),
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_path": str(root / "metrics_and_coverage.json"),
                      "valid_judgments": len(valid), "validation_issues": len(errors),
                      "judgment_status": evaluator_state,
                      "report_path": str(root / "evaluation_report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
