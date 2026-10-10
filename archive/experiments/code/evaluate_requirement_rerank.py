"""Compute frozen method-specific P@5 and complete-pool nDCG@5 metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

METHODS = ("tfidf_cosine", "embedding_section_aware_v1", "requirement_reranker")
VALID_GRADES = {0, 1, 2, 3}


def _dcg(grades: Sequence[int]) -> float:
    return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def calculate_metrics(
    pool: Sequence[Mapping[str, Any]],
    retrieval_rankings: Sequence[Mapping[str, Any]],
    reranked_rows: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pool_keys = [(str(row["profile_id"]), str(row["job_id"])) for row in pool]
    if len(set(pool_keys)) != len(pool_keys):
        raise ValueError("Evaluation pool contains duplicate profile/job pairs.")
    expected = set(pool_keys)
    grades: dict[tuple[str, str], int] = {}
    for row in judgments:
        key = (str(row.get("profile_id", "")), str(row.get("job_id", "")))
        if key not in expected:
            raise ValueError("Judgment is outside the frozen pool: {}".format(key))
        if key in grades:
            raise ValueError("Duplicate judgment: {}".format(key))
        status, grade = row.get("status"), row.get("grade")
        if status == "judged":
            if isinstance(grade, bool) or not isinstance(grade, int) or grade not in VALID_GRADES:
                raise ValueError("Judged rows need an integer grade from 0 to 3.")
            grades[key] = grade
        elif status == "unknown":
            if grade is not None:
                raise ValueError("Unknown judgments must have a null grade.")
        else:
            raise ValueError("Judgment status must be judged or unknown.")

    rankings = [dict(row) for row in retrieval_rankings]
    rerank_positions: dict[str, int] = defaultdict(int)
    rerank_rankings = []
    for row in reranked_rows:
        profile_id = str(row["profile_id"])
        rerank_positions[profile_id] += 1
        rerank_rankings.append({"profile_id": profile_id, "job_id": str(row["job_id"]),
                                "method": "requirement_reranker", "rank": rerank_positions[profile_id]})
    rankings.extend(rerank_rankings)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rankings:
        method = str(row["method"])
        key = (str(row["profile_id"]), str(row["job_id"]))
        if method not in METHODS or key not in expected:
            raise ValueError("Ranking is outside the frozen methods/pool: {} / {}".format(method, key))
        grouped[(key[0], method)].append(row)

    assessment_by_pair = {
        (str(row["profile_id"]), str(row["job_id"])): row.get("assessment", row)
        for row in reranked_rows
    }
    if set(assessment_by_pair) != expected:
        raise ValueError("Reranker assessments must cover the entire frozen retrieval union.")

    per_profile: list[dict[str, Any]] = []
    p5_by_method: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    strong_p5_by_method: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    ndcg_by_method: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    for profile_id in dict.fromkeys(key[0] for key in pool_keys):
        profile_keys = [key for key in pool_keys if key[0] == profile_id]
        unknown_pool = [key for key in profile_keys if key not in grades]
        ideal = sorted((grades[key] for key in profile_keys if key in grades), reverse=True)[:5]
        pool_complete = not unknown_pool
        ideal_dcg = _dcg(ideal) if pool_complete else 0.0
        method_metrics = {}
        for method in METHODS:
            rows = sorted(grouped.get((profile_id, method), []), key=lambda row: int(row["rank"]))
            top5 = rows[:5]
            top_grades = [grades.get((profile_id, str(row["job_id"]))) for row in top5]
            p5_complete = len(top5) == 5 and all(grade is not None for grade in top_grades)
            plausible_p5 = sum(grade >= 2 for grade in top_grades if grade is not None) / 5 if p5_complete else None
            strong_p5 = sum(grade == 3 for grade in top_grades if grade is not None) / 5 if p5_complete else None
            if p5_complete:
                p5_by_method[method][profile_id] = plausible_p5
                strong_p5_by_method[method][profile_id] = strong_p5
            ndcg = None
            ndcg_status = "incomplete_common_pool_judgments" if unknown_pool else "reported"
            if pool_complete and top5:
                actual = [grades[(profile_id, str(row["job_id"]))] for row in top5]
                ndcg = _dcg(actual) / ideal_dcg if ideal_dcg else 0.0
                ndcg_by_method[method][profile_id] = ndcg
                if ideal_dcg == 0:
                    ndcg_status = "reported_zero_ideal_gain"
            conflicts = sum(
                any(item.get("classification") == "mandatory"
                    and item.get("status") == "explicitly_contradicted"
                    for item in assessment_by_pair[(profile_id, str(row["job_id"]))].get("requirements", []))
                or any(item.get("status") == "explicitly_violated"
                       for item in assessment_by_pair[(profile_id, str(row["job_id"]))].get("constraints", []))
                for row in top5
            )
            method_metrics[method] = {
                "retrieved_at_5": len(top5),
                "judged_at_5": sum(grade is not None for grade in top_grades),
                "plausible_or_strong_precision_at_5": plausible_p5,
                "strong_only_precision_at_5": strong_p5,
                "p_at_5_status": "reported" if p5_complete else "incomplete_or_fewer_than_five",
                "ndcg_at_5": ndcg,
                "ndcg_status": ndcg_status if top5 else ("no_rankings" if not rows else ndcg_status),
                "confirmed_conflicts_at_5": conflicts,
            }
        per_profile.append({"profile_id": profile_id, "shared_pool_size": len(profile_keys),
                            "judged_shared_pool_items": len(profile_keys) - len(unknown_pool),
                            "unknown_shared_pool_items": len(unknown_pool), "methods": method_metrics})

    paired_p5_profiles = [profile for profile in p5_by_method[METHODS[0]]
                          if all(profile in p5_by_method[method] for method in METHODS)]
    paired_ndcg_profiles = [profile for profile in ndcg_by_method[METHODS[0]]
                            if all(profile in ndcg_by_method[method] for method in METHODS)]
    return {
        "method_scope": list(METHODS),
        "judged_pair_count": len(grades),
        "pool_pair_count": len(pool_keys),
        "per_method_p5_denominators": {method: len(p5_by_method[method]) for method in METHODS},
        "paired_p5_profiles": paired_p5_profiles,
        "paired_p5_denominator": len(paired_p5_profiles),
        "paired_p5_mean_plausible_or_strong": {
            method: (sum(p5_by_method[method][profile] for profile in paired_p5_profiles)
                     / len(paired_p5_profiles) if paired_p5_profiles else None)
            for method in METHODS
        },
        "paired_p5_mean_strong_only": {
            method: (sum(strong_p5_by_method[method][profile] for profile in paired_p5_profiles)
                     / len(paired_p5_profiles) if paired_p5_profiles else None)
            for method in METHODS
        },
        "paired_ndcg_profiles": paired_ndcg_profiles,
        "paired_ndcg_denominator": len(paired_ndcg_profiles),
        "paired_ndcg_mean": {
            method: (sum(ndcg_by_method[method][profile] for profile in paired_ndcg_profiles)
                     / len(paired_ndcg_profiles) if paired_ndcg_profiles else None)
            for method in METHODS
        },
        "per_profile": per_profile,
        "statistical_significance_tested": False,
        "scope_warning": "Metrics are relative to each profile's fixed retrieved union, not full-corpus coverage or user outcomes.",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--retrieval-rankings", type=Path, required=True)
    parser.add_argument("--reranked-union", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate_metrics(_read_jsonl(args.pool), _read_jsonl(args.retrieval_rankings),
                               _read_jsonl(args.reranked_union), _read_jsonl(args.judgments))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pool_pairs": result["pool_pair_count"],
                      "judged_pairs": result["judged_pair_count"],
                      "paired_p5_denominator": result["paired_p5_denominator"],
                      "paired_ndcg_denominator": result["paired_ndcg_denominator"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
