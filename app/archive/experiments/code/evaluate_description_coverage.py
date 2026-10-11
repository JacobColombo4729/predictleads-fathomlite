"""Evaluate saved description-coverage recommendations from human labels only.

This command reads existing recommendation and review CSVs. It does not load the
job sample, fit TF-IDF, rerun retrieval, access AWS, or change recommendations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ENGINE_PATH = Path(__file__).with_name("job_matching_engine.py")
ENGINE_SPEC = importlib.util.spec_from_file_location("job_matching_engine", ENGINE_PATH)
if ENGINE_SPEC is None or ENGINE_SPEC.loader is None:
    raise ImportError(f"Could not load matching evaluation functions from {ENGINE_PATH}")
engine = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = engine
ENGINE_SPEC.loader.exec_module(engine)


def evaluate_saved_experiment(
    recommendations_path: Path,
    labels_path: Path,
    *,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate saved rankings against genuine labels, optionally for one profile."""
    recommendations: list[dict[str, Any]] = []
    with Path(recommendations_path).open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            row["rank"] = int(row["rank"])
            recommendations.append(row)
    if not recommendations:
        raise ValueError(f"No saved recommendation rows found in {recommendations_path}")
    if profile_id is not None:
        recommendations = [row for row in recommendations if row.get("profile_id") == profile_id]
        if not recommendations:
            raise ValueError(f"No saved recommendations found for profile_id={profile_id!r}")
    labels = engine.read_review_labels(labels_path)
    if not labels:
        raise ValueError(f"No genuine relevance labels found in {labels_path}; metrics were not calculated")
    pools = engine.reference_pools_from_recommendations(recommendations)
    return engine.evaluate_reviewed_recommendations(recommendations, labels, reference_pools=pools)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--profile-id", default=None,
                        help="restrict evaluation and shared reference pool to one profile")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to shared_pool_metrics.json beside the recommendations")
    args = parser.parse_args(argv)
    recommendations_path = args.recommendations.resolve()
    labels_path = args.labels.resolve()
    if not recommendations_path.is_file():
        parser.error(f"Saved recommendations not found: {recommendations_path}")
    if not labels_path.is_file():
        parser.error(f"Human review file not found: {labels_path}")
    try:
        metrics = evaluate_saved_experiment(recommendations_path, labels_path, profile_id=args.profile_id)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Evaluation unavailable: {error}", file=sys.stderr)
        return 2
    output_path = args.output.resolve() if args.output else recommendations_path.parent / "shared_pool_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "metrics_path": str(output_path),
        "review_labels_path": str(labels_path),
        "profile_method_condition_rows": len(metrics),
        "metrics": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
