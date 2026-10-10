"""Run model-backed requirement assessment and deterministic ordering on a saved fixed union."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from requirement_aware_reranker import (
    ProviderConfig,
    ProviderUnavailable,
    assess_pair,
    rerank_assessments,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_no, error)) from error
            if not isinstance(item, dict):
                raise ValueError("Expected JSON objects in {}".format(path))
            output.append(item)
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(input_path: Path, output_dir: Path, *, max_pairs: int = 80) -> dict[str, Any]:
    provider = ProviderConfig.from_environment()
    # Provider preflight completes before input loading or any model request.
    inputs = _read_jsonl(input_path.resolve())
    if not inputs or len(inputs) > max_pairs:
        raise ValueError("Input union must contain 1..{} pairs; found {}.".format(max_pairs, len(inputs)))
    seen = set()
    for row in inputs:
        required = {"profile_id", "job_id", "candidate_evidence", "job_evidence", "best_retrieval_rank"}
        missing = required - set(row)
        if missing:
            raise ValueError("Input row missing fields: {}".format(", ".join(sorted(missing))))
        key = (str(row["profile_id"]), str(row["job_id"]))
        if key in seen:
            raise ValueError("Duplicate profile/job input pair: {}".format(key))
        seen.add(key)
        if isinstance(row["best_retrieval_rank"], bool) or not isinstance(row["best_retrieval_rank"], int):
            raise ValueError("best_retrieval_rank must be an integer")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    assessments: list[dict[str, Any]] = []
    responses_meta: list[dict[str, Any]] = []
    partial_path = output_dir / "rerank_assessments_partial.jsonl"
    # Keep successful pair outputs if a later provider call fails; never label a partial run complete.
    with partial_path.open("w", encoding="utf-8") as partial:
        for row in inputs:
            assessment, metadata = assess_pair(
                row["candidate_evidence"], row["job_evidence"], provider
            )
            if assessment["profile_id"] != str(row["profile_id"]) or assessment["job_id"] != str(row["job_id"]):
                raise ValueError("Assessment identity does not match its frozen input pair.")
            assessment["provider_metadata"] = metadata
            assessments.append(assessment)
            responses_meta.append(metadata)
            partial.write(json.dumps(assessment, ensure_ascii=False) + "\n")
            partial.flush()

    if len(assessments) != len(inputs):
        raise AssertionError("Reranker assessment coverage differs from input union.")
    assessment_by_pair = {(row["profile_id"], row["job_id"]): row for row in assessments}
    with_ranks = []
    for row in inputs:
        assessment = assessment_by_pair[(str(row["profile_id"]), str(row["job_id"]))]
        with_ranks.append({"profile_id": str(row["profile_id"]), "job_id": str(row["job_id"]),
                           "assessment": assessment,
                           "best_retrieval_rank": int(row["best_retrieval_rank"]),
                           "source_ranks": row.get("source_ranks", {})})
    ordered = rerank_assessments(with_ranks, {
        (str(row["profile_id"]), str(row["job_id"])): int(row["best_retrieval_rank"])
        for row in inputs
    })
    _write_jsonl(output_dir / "rerank_assessments.jsonl", assessments)
    _write_jsonl(output_dir / "reranked_union.jsonl", ordered)
    (output_dir / "rerank_assessments_partial.jsonl").unlink(missing_ok=True)
    manifest = {
        "status": "completed",
        "provider": provider.public_manifest(),
        "api_calls": len(inputs),
        "response_model_identities": sorted({str(row.get("response_model")) for row in responses_meta
                                               if row.get("response_model")}),
        "responses_with_usage_metadata": sum(row.get("usage") is not None for row in responses_meta),
        "pair_count": len(inputs),
        "profile_count": len({str(row["profile_id"]) for row in inputs}),
        "elapsed_seconds": time.perf_counter() - started,
        "assessment_schema_version": "requirement_assessment_v1",
        "ordering_rule": ["explicit constraint violations ascending",
                           "mandatory contradictions ascending", "duty transfer descending",
                           "best source retrieval rank ascending", "stable job_id ascending"],
        "evidence_validation_failure_count": 0,
        "assessments_completed": len(assessments),
        "unknowns_are_not_penalties": True,
    }
    (output_dir / "reranker_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"status": "completed", "pair_count": len(inputs), "api_calls": len(inputs),
            "elapsed_seconds": manifest["elapsed_seconds"],
            "provider_model": provider.model, "provider_revision": provider.revision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-union", type=Path, required=True,
                        help="Saved JSONL union with original evidence and source ranks; score/method fields are not sent to the model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=80)
    args = parser.parse_args()
    try:
        result = run(args.input_union, args.output_dir, max_pairs=args.max_pairs)
    except ProviderUnavailable as error:
        print(json.dumps({"status": "pending_provider", "error": str(error), "api_calls": 0}, indent=2))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
