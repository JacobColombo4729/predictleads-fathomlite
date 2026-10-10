"""Recall check: do the roles a candidate wrote down in advance show up in the rankings?

    python code/recall_check.py --run-dir data/matching/<ranking_run> --profile-id <profile_id> \
        --titles "solutions engineer" "sales engineer" "technical account manager"

Each target is a case-insensitive title substring or regex written BEFORE seeing rankings. For each
target the script reports how many corpus jobs match it, and for each method the best rank (1-50)
at which any matching job appears, or "not in top 50". Existence in the corpus is reported separately
so "the matcher missed it" is distinguished from "the corpus has no such job".
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--titles", nargs="+", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    titles: dict[str, str] = {}
    with (Path(manifest["corpus_dir"]) / "jobs_prepared_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            titles[record["job_id"]] = str(record.get("title_searchable") or "")
    rows = [r for r in csv.DictReader((args.run_dir / "rankings_top50.csv").open(encoding="utf-8"))
            if r["profile_id"] == args.profile_id]
    report = []
    for target in args.titles:
        pattern = re.compile(target, re.IGNORECASE)
        in_corpus = sum(bool(pattern.search(t)) for t in titles.values())
        best = {}
        for method in ("tfidf_cosine", "bge_cosine", "rrf_equal"):
            ranks = [int(r["rank"]) for r in rows if r["method"] == method and pattern.search(r["title"])]
            best[method] = min(ranks) if ranks else "not in top 50"
        report.append({"target": target, "jobs_in_corpus_with_matching_title": in_corpus, "best_rank": best})
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
