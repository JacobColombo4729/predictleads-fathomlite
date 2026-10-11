# Alignment with `predictleads-hackathon` (Jesse's app)

A read-only summary of Jesse's repository, refreshed on request ("check the hackathon repo"), so the analysis here stays aligned with the app. Nothing in his repo is changed from here: Claude reads the local clone at `~/dev/predictleads-hackathon` as it is (never its `data/`, `models/` or `.env`). To see his newest work, pull that clone first.

**Last checked:** 2026-10-10 (evening), branch `feature/mac-support`, commit `a8196fd` ("Add a strengths radar to the Compass job panel"). Target date in his roadmap: about 2026-10-23. His audience: career changers, starting with TripleTen AI/ML graduates, aiming at AI/ML, data, and technical sales / solutions engineering.

## What his app does now

| Piece | State |
|:--|:--|
| Backend (FastAPI) and frontend (React) | Live on a laptop: profile, ranked jobs, 👍/👎 feedback, saved jobs |
| Ranker | TF-IDF + BGE combined with RRF; not yet tested for quality |
| Listings | About 320,000 in three areas (AI/ML, data, technical sales); about 81,000 served (status not "closed"; marked "historical, not verified as open") |
| Compass tab | Insights about 3–20 saved jobs (themes, fit tiers, evidence lines); local |
| Signals tab | Companies by hiring bursts, steady hiring or cooling |
| Magic fill and grounded explanations | Call OpenAI; off by default (explanations send listing text) |
| Role-family classifier | Built, not used by the app (17 families from O*NET occupations) |
| Ask the market | Questions answered from the listing data (typed or spoken), with read-back chips and optional cited web research; guards against forecasts, advice and invented filters |
| Compass map and strengths radar | A map of picked jobs and suggested roles, and a strengths radar on each job panel (it merged Jacob's Compass map from `feature/mac-support`) |
| Landing page | "Bright Depth" branding (anglerfish mascot, motto) |

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

- **2026-10-10 (evening), at `a8196fd`:** Ask the market; the Compass map (picks vs. suggested roles, legend highlighting) and a strengths radar; a landing page. A new note, `docs/signal-semantics.md`, measures crawl artifacts on his corpus: **batch onboarding** (8.9% of companies with 5+ listings had 60%+ of them first seen on one day: a company added to the crawl, not a hiring surge) and **crawl dropout** (5% of companies have most disappearances on one day). His rules: call them *observed* openings and disappearances; **active = last seen within 7 days and not `closed`**; windows of two weeks or more; judge momentum against the company's own history; validate by backtest. These bear on our timing results (days open, company volume, weekly arrivals) and on merge decision 2.
- First version.
