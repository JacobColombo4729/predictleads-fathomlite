# Alignment with `predictleads-hackathon` (Jesse's app)

A read-only summary of Jesse's repository, refreshed on request ("check the hackathon repo"), so the analysis here stays aligned with the app. Nothing in his repo is changed from here: Claude reads the local clone at `~/dev/predictleads-hackathon` as it is (never its `data/`, `models/` or `.env`). To see his newest work, pull that clone first.

**Last checked:** 2026-10-10, branch `feature/mac-support`, commit `5d6d743` ("demo backend for mac"). Target date in his roadmap: about 2026-10-23.

## What his app does now

| Piece | State |
|:--|:--|
| Backend (FastAPI) and frontend (React) | Live on a laptop: profile, ranked jobs, 👍/👎 feedback, saved jobs |
| Ranker | TF-IDF + BGE combined with RRF; not yet tested for quality |
| Listings | About 320,000 in three areas (AI/ML, data, technical sales); about 81,000 served (status not "closed"; marked "historical, not verified as open") |
| Compass tab | Insights about 3–20 saved jobs (themes, fit tiers, evidence lines); local |
| Signals tab | Companies by hiring bursts, steady hiring or cooling |
| Magic fill and grounded explanations | Call OpenAI; off by default (explanations send listing text) |
| Role-family classifier | Built, not used by the app |

## Where the analysis here plugs in

| Analysis piece | Where it lands in the app | Merge-plan decision |
|:--|:--|:--|
| Last-seen rule for "still open" | Which listings the app shows | 2 |
| Scope and role buckets (8, technical sales maybe a ninth) | Filters and labels | 1 |
| Shared stream and cleaning | The app's corpus and embeddings | 5 |
| Timing (apply-by, chance still open), ghost flags, hiding rules | Ranking and the job detail view | 4 |
| Retraining compass (Profiles + 3.12) | A new view next to his Compass tab | 6 |
| Company history, "apply now" alerts | His Signals tab | – |

## Open questions for Jesse

- The six decisions in [`merge_plan.md`](merge_plan.md), and the organization in [`github_org_setup.md`](github_org_setup.md).
- Whether grounded explanations send the whole listing description (his docs say yes in OpenAI mode).

## Changes since the last check

- First version.
