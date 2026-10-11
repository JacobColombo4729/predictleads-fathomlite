"""Summarise the research event log (what was shown and how people reacted) and optionally export it.

    python code/feedback_report.py                      # summary of backend/data/feedback-events.sqlite3
    python code/feedback_report.py --export out_dir     # also write events.csv and ratings.csv

The summary reports what was measured and what it cannot say: people only rate jobs the ranker put in front of them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.events import EventStore  # noqa: E402

TOP_K = 10


def final_ratings(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The last rating per person and job, dropping ratings that were cleared afterwards."""
    latest: dict[tuple[str, str], dict[str, Any] | None] = {}
    for e in events:
        key = (e["participant_id"], e["job_id"])
        if e["type"] == "rating":
            latest[key] = e
        elif e["type"] == "rating_cleared":
            latest[key] = None
    return [e for e in latest.values() if e is not None]


def by_source(ratings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Share of good ratings by where the listing came from: the ranker, or an exploration listing (keyword / embedding / random).

    "keyword" listings were ranked high by the keyword signal and not by the embedding signal; "embedding" the reverse.
    If one side's share of good ratings is clearly higher, that signal was the better judge on the listings they disagree about.
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in ratings:
        groups[r["payload"].get("source", "ranker")].append(r["payload"].get("signal") == "good")
    return {name: {"rated": len(v), "share_good": round(sum(v) / len(v), 3)} for name, v in sorted(groups.items())}


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(e["type"] for e in events)
    people = sorted({e["participant_id"] for e in events})
    ratings = final_ratings(events)
    good = [r for r in ratings if r["payload"].get("signal") == "good"]
    ranked = [r for r in ratings if r["payload"].get("rank") is not None]
    top = [r for r in ranked if r["payload"]["rank"] <= TOP_K]
    per_person: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for e in events:
        per_person[e["participant_id"]][e["type"]] += 1
    for r in ratings:
        per_person[r["participant_id"]]["final_" + r["payload"].get("signal", "?")] += 1
    return {
        "participants": len(people),
        "events_by_type": dict(by_type),
        "ratings_final": len(ratings),
        "share_good": round(len(good) / len(ratings), 3) if ratings else None,
        "rated_with_known_rank": len(ranked),
        f"share_good_in_top_{TOP_K}": round(sum(r["payload"]["signal"] == "good" for r in top) / len(top), 3) if top else None,
        f"rated_in_top_{TOP_K}": len(top),
        "thumbs_down_reasons": dict(Counter(r["payload"].get("reason", "none given") for r in ratings if r["payload"].get("signal") == "bad")),
        "thumbs_down_notes": sum(1 for r in ratings if r["payload"].get("note")),
        "by_source": by_source(ratings),
        "per_participant": {p[:8]: dict(c) for p, c in per_person.items()},
    }


def export(events: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "ts", "participant", "type", "snapshot_id", "job_id", "profile_hash", "payload_json"])
        for e in events:
            writer.writerow([e["event_id"], e["ts"], e["participant_id"][:8], e["type"], e["snapshot_id"], e["job_id"], e["profile_hash"], str(e["payload"])])
    with (directory / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "job_id", "signal", "reason", "note", "rank_when_shown", "profile_hash", "ts"])
        for r in final_ratings(events):
            writer.writerow([r["participant_id"][:8], r["job_id"], r["payload"].get("signal"), r["payload"].get("reason", ""), r["payload"].get("note", ""),
                             r["payload"].get("rank"), r["profile_hash"], r["ts"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=ROOT / "backend" / "data" / "feedback-events.sqlite3")
    parser.add_argument("--export", type=Path, default=None, help="write events.csv and ratings.csv to this folder")
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"No event log at {args.db}. Use the app with the backend running first.")
    events = EventStore(args.db).events()
    summary = summarize(events)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nReading this: ratings only exist for jobs the ranker showed, so the share of good ratings measures what we showed,\n"
          "not what we missed. Comparing rankers on these labels favours the ranker that chose what was shown.")
    if args.export:
        export(events, args.export)
        print(f"Wrote {args.export / 'events.csv'} and {args.export / 'ratings.csv'}")


if __name__ == "__main__":
    main()
