"""Compare rankers using human labels from the local review page.

    python code/analyze_review_labels.py --run-dir data/matching/<ranking_run> \
        --labels labels_jesse.json labels_jacob.json

Reads review/answer_key.json (kept hidden until labelling is finished) and the exported label files.
Reports per-profile counts of labels in each method's top five, win/tie/loss between methods, and
reviewer agreement on pairs both people labelled. A profile/method is withheld from P@5 if any of
its top-five pairs is unlabeled or marked "cant_tell". No significance claims are made.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

METHODS = ("tfidf_cosine", "bge_cosine", "rrf_equal")
STRICT = {"worth_exploring"}
LENIENT = {"worth_exploring", "maybe"}


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    key = json.loads((args.run_dir / "review" / "answer_key.json").read_text(encoding="utf-8"))
    by_reviewer: dict[str, dict[str, str]] = {}
    for path in args.labels:
        data = json.loads(path.read_text(encoding="utf-8"))
        by_reviewer[data["reviewer"]] = {pid: row["label"] for pid, row in data["labels"].items()}
    # The team's own résumés are never used (CLAUDE.md): refuse a review that includes one.
    personal = sorted({info["profile_id"] for info in key.values() if info["profile_id"].startswith("real_")})
    if personal:
        raise SystemExit(f"{len(personal)} personal profile(s) in this review: rebuild it from fictional profiles only")
    # one label per pair: the first reviewer who labelled it
    pair_label: dict[str, str] = {}
    disagreements, overlap = [], []
    for pid, info in key.items():
        labels = {r: m[pid] for r, m in by_reviewer.items() if pid in m}
        if len(labels) == 2:
            a, b = list(labels.values())
            overlap.append((a, b))
            if a != b:
                disagreements.append({"presentation_id": pid, "profile_id": info["profile_id"], "labels": labels})
        if labels:
            pair_label[pid] = next(iter(labels.values()))
    by_profile_method: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    for pid, info in key.items():
        for hit in info["returned_by"]:
            if hit["rank"] <= args.top:
                by_profile_method[(info["profile_id"], hit["method"])].append(pair_label.get(pid))
    profiles = sorted({p for p, _ in by_profile_method})
    table, wins = [], defaultdict(Counter)
    for profile in profiles:
        row = {"profile_id": profile}
        for method in METHODS:
            labels = by_profile_method.get((profile, method), [])
            counts = Counter(l or "unlabeled" for l in labels)
            complete = len(labels) == args.top and not (counts["unlabeled"] or counts["cant_tell"])
            row[method] = {"counts": dict(counts), "strict_p": sum(l in STRICT for l in labels) / args.top if complete else None,
                           "lenient_p": sum(l in LENIENT for l in labels) / args.top if complete else None}
        table.append(row)
        for a, b in (("rrf_equal", "tfidf_cosine"), ("rrf_equal", "bge_cosine"), ("bge_cosine", "tfidf_cosine")):
            for measure in ("strict_p", "lenient_p"):
                x, y = row[a][measure], row[b][measure]
                if x is None or y is None:
                    wins[f"{a} vs {b} ({measure})"]["withheld"] += 1
                else:
                    wins[f"{a} vs {b} ({measure})"]["win" if x > y else "loss" if x < y else "tie"] += 1
    binary = [("bad" if a == "not_realistic" else "ok", "bad" if b == "not_realistic" else "ok") for a, b in overlap]
    report = {
        "labeled_pairs": len(pair_label), "pairs_in_key": len(key),
        "per_profile": table, "win_tie_loss_by_profile": {k: dict(v) for k, v in wins.items()},
        "reviewer_overlap": {"pairs": len(overlap),
                             "raw_agreement": (sum(a == b for a, b in overlap) / len(overlap)) if overlap else None,
                             "kappa_4_labels": kappa(overlap), "kappa_not_realistic_vs_rest": kappa(binary),
                             "disagreements": disagreements},
        "caution": "Small, hand-labelled set. Use for a demo go/no-go choice, not as proof of a better method.",
    }
    out = args.run_dir / "review" / "label_analysis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("labeled_pairs", "pairs_in_key", "win_tie_loss_by_profile", "reviewer_overlap")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
