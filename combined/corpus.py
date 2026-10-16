"""The combined model's serving corpus: the scope's listings still open when the data ends, in the app's corpus format.

    python -m combined.corpus build                 scan the raw export once and keep the scope's open listings
    python -m combined.corpus build --limit 5000    a timing slice in its own folder (corpus_slice/), never served
    python -m combined.corpus encode [--slice]      BGE vectors with Jesse's encoder, unchanged (--device auto|cpu)

Which listings: every id in ML_Rank's hand-off (`listing_scores.parquet`, the 908,291 scope listings) with
`open_at_end`, i.e. still open when the data ends. Hidden listings stay in; the queue hides them, as `mlrank` does.

How: the raw export is read with the analysis reader (`src.stream_full.open_lines`: the copy in data/raw/, else S3
from .env); each kept listing is prepared with Jesse's `prepare_jobs.prepare_resource` and goes through the same rules
as his `rank_candidates.build_corpus` (a usable description, duplicates collapsed by company + title + description, the
raw HTML left out), so `rrf_ranker.py` and the app read it like their own corpus. `encode` is his
`rank_candidates.encode`: the same BGE model, revision and job text.

Writes to data/clean/combined_v1/corpus/ (git-ignored): jobs_prepared_corpus.jsonl, corpus_report.json, and after
encode bge_embeddings.npy + bge_embeddings_manifest.json. Prints counts only, never listing values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from combined import VERSION
from combined._paths import CORPUS_DIR, ML_RANK_DIR, SLICE_DIR

SERVING_RULE = "ML_Rank listing_scores.parquet: open_at_end (still open when the data ends)"
PROGRESS_DOCS = 200_000


def serving_ids(handoff_dir: Path = ML_RANK_DIR) -> set[str]:
    """The ids of the scope listings still open when the data ends, from ML_Rank's hand-off."""
    import pandas as pd

    path = handoff_dir / "listing_scores.parquet"
    if not path.exists():
        raise FileNotFoundError("ML_Rank's hand-off is missing listing_scores.parquet: run ML_Rank section 13 first.")
    scores = pd.read_parquet(path, columns=["id", "open_at_end"])
    return set(scores.loc[scores["open_at_end"].eq(True), "id"].astype(str))  # blank counts as not open


class CorpusWriter:
    """Jesse's corpus rules (rank_candidates.build_corpus), one prepared listing at a time."""

    def __init__(self, handle):
        from rank_candidates import _dup_key

        self.handle, self._dup_key = handle, _dup_key
        self.seen_ids: set[str] = set()
        self.seen_keys: set[str] = set()
        self.counts: Counter[str] = Counter()

    def consider(self, record: dict[str, Any]) -> bool:
        job_id = str(record.get("job_id"))
        self.counts["records_seen"] += 1
        if job_id in self.seen_ids:
            self.counts["same_job_id_again"] += 1
            return False
        self.seen_ids.add(job_id)
        if not (record.get("description_usable") and str(record.get("description_text") or "").strip()):
            self.counts["excluded_no_usable_description"] += 1
            return False
        key = self._dup_key(record)
        if key in self.seen_keys:
            self.counts["collapsed_duplicate_listing"] += 1
            return False
        self.seen_keys.add(key)
        record["corpus_sources"] = ["combined_scope"]
        record["duplicate_key"] = key
        record.pop("description_raw", None)  # as in build_corpus: the corpus keeps the cleaned text only
        record["description_raw_omitted_from_corpus"] = True
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.counts["representatives_written"] += 1
        return True


def scope_resources(lines: Iterable[str], ids: set[str], errors: Counter) -> Iterable[tuple[int, dict, list]]:
    """(line number, job_opening resource, included) for every resource whose id is in `ids`."""
    for number, line in enumerate(lines, 1):
        try:
            wrapper = json.loads(line)
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            continue
        data = wrapper.get("data") if isinstance(wrapper, dict) else None
        if not isinstance(data, list):
            errors["data_not_array"] += 1
            continue
        included = wrapper.get("included") if isinstance(wrapper.get("included"), list) else []
        for resource in data:
            if isinstance(resource, dict) and resource.get("type") == "job_opening" and str(resource.get("id")) in ids:
                yield number, resource, included


def build(out_dir: Path, ids: set[str], local: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Scan the raw export once and write the corpus of `ids` (or its first `limit` listings) to `out_dir`."""
    from prepare_jobs import KEY, prepare_resource
    from src.stream_full import open_lines

    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a corpus: a new build goes in a new folder
    corpus_path = out_dir / "jobs_prepared_corpus.jsonl"
    errors: Counter[str] = Counter()
    started, stop_reason, last = time.monotonic(), "in_progress", 0
    lines, counter, total = open_lines(local)
    with corpus_path.open("x", encoding="utf-8", newline="\n") as handle:
        writer = CorpusWriter(handle)
        try:
            for number, resource, included in scope_resources(lines, ids, errors):
                writer.consider(prepare_resource(resource, included, source_line=number, source_key=KEY))
                if limit is not None and writer.counts["representatives_written"] >= limit:
                    stop_reason = "limit"
                    break
                if number - last >= PROGRESS_DOCS:
                    last = number
                    print(f"documents {number:,}  read {counter.bytes_read / total:.0%}  "
                          f"kept {writer.counts['representatives_written']:,}  "
                          f"elapsed {time.monotonic() - started:.0f}s", flush=True)
            else:
                stop_reason = "eof"
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        finally:
            lines.close()
            counter.raw.close()  # the counting wrapper doesn't close the file or S3 body it wraps
    report = {
        "corpus_version": VERSION + "-scope", "serving_rule": SERVING_RULE, "serving_ids": len(ids),
        "stop_reason": stop_reason, "complete_file_scanned": stop_reason == "eof", "limit": limit,
        "ids_not_found": len(ids - writer.seen_ids) if stop_reason == "eof" else None,
        "counts": dict(writer.counts), "errors": dict(errors),
        "corpus_sha256": _sha256(corpus_path), "elapsed_seconds": round(time.monotonic() - started, 1),
        "rules": "rank_candidates.build_corpus: usable description; duplicates collapsed by company id + title + "
                 "description (first kept); description_raw omitted",
    }
    (out_dir / "corpus_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def encode(corpus_dir: Path, device: str = "auto", batch_size: int = 64) -> int:
    """BGE vectors for the corpus, with Jesse's encoder unchanged (resumable: finished chunks are kept)."""
    import rank_candidates

    return rank_candidates.encode(corpus_dir, device, batch_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="scan the raw export and write the scope corpus")
    b.add_argument("--limit", type=int, default=None, help="stop after this many listings (writes corpus_slice/)")
    e = sub.add_parser("encode", help="BGE vectors for the corpus")
    e.add_argument("--slice", action="store_true", help="encode corpus_slice/ instead (timing)")
    e.add_argument("--device", default="auto", choices=["auto", "cpu"])
    e.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.command == "build":
        ids = serving_ids()
        print(f"serving ids (scope, still open at the end): {len(ids):,}", flush=True)
        report = build(SLICE_DIR if args.limit else CORPUS_DIR, ids, limit=args.limit)
        print(json.dumps({k: report[k] for k in ("stop_reason", "counts", "errors", "ids_not_found", "elapsed_seconds")}))
        return 0 if report["stop_reason"] in {"eof", "limit"} else 1
    return encode(SLICE_DIR if args.slice else CORPUS_DIR, args.device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
