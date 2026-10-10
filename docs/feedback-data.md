# Feedback data: what the app records and how it would scale

The app records what each person was **shown** and how they **reacted**. That is the raw material for measuring the ranker and, with enough users, for learning a better one. This page describes the demo version and the path to production.

Built 2026-10-10: `backend/events.py`, the hooks in `backend/api.py`, `code/feedback_report.py`, `frontend/src/lib/research.ts`.

## What is recorded

One append-only table of events, in its own SQLite file (`backend/data/feedback-events.sqlite3`, git-ignored; set `PREDICTLEADS_EVENTS_DB_PATH` to move it). It lives apart from the demo session database, so it survives the 24-hour session expiry and Start over.

| Event | When | What it holds |
|---|---|---|
| `impression` | a ranked list is shown on Discover | the job ids with their rank and score, the ranker and version, the search text, how many 👍/👎 were already given |
| `rating` / `rating_cleared` | 👍/👎 pressed or undone | job id, signal, and **where that job sat the last time it was shown** (rank and which impression) |
| `save` / `unsave` | a job is saved or removed | job id |
| `compass_run` | the Compass is built or the stretch slider is released | the picked job ids, the stretch setting, the suggested job ids with their fit |

Each event also carries a time, an anonymous participant id, the snapshot id and a profile hash. A small table keeps the ranker's **embedding of each profile** (a list of numbers, not text), once per profile version, so a later model can learn from "this kind of profile, this kind of job".

**Why log what was shown, not only clicks.** A 👎 only means something if you know the job was in front of the person, and a missing 👍 means nothing at all. Without impressions and ranks, the data is badly biased toward whatever the ranker already prefers.

## What is not recorded

No names, emails, phone numbers, resume text, profile text, notes or job descriptions. The participant id is a random value generated in the browser (`localStorage`); it identifies a browser, not a person. Job ids point to the PredictLeads snapshot, which stays inside our own infrastructure.

## Demo versus production

| | Demo (now) | Production |
|---|---|---|
| Consent | Everyone using the demo is on the team and has agreed outside the app, so logging is on by default for anyone sending the id | **Users must opt in.** Start the backend with `PREDICTLEADS_EVENTS_REQUIRE_CONSENT=1`: nothing is written until the person agrees through `PUT /api/v1/research/consent`. The API, deletion (`PUT` with `consent: false` or `DELETE /api/v1/research/data`) and tests exist; the consent screen does not |
| Storage | One SQLite file | A managed database (for example PostgreSQL) for the live app, with events streamed to a warehouse for analysis |
| Identity | Random id per browser | Accounts, so ratings follow a person across devices; keep the id pseudonymous in the analytics store |
| Volume | A handful of people | Event queue or batch loader; partitioned tables; retention limits |
| Privacy | Local file | Retention period, deletion on request (already supported per participant), access control, a written data policy |
| Bias handling | Ranks and impressions are logged | Also mix in a small random slice of jobs the ranker would not have picked, and log them as such, so the model can learn about jobs it never showed |

The schema is deliberately the shape a production event table would have, so moving to a real database changes the storage layer, not the design.

## Reading it

```
python code/feedback_report.py                    # summary
python code/feedback_report.py --export out_dir   # also events.csv and ratings.csv
```

The summary gives the number of people and events, the final ratings (the last rating per person and job, minus cleared ones), the share rated good, and the share rated good among jobs shown in the top 10.

## What it can and cannot tell us

- It measures **how good what we showed was**: the share of 👍 in the top 10. That is a real, quotable number once there are a few dozen ratings.
- It cannot tell us what we **missed**. People only rate what appears, so comparing rankers on these ratings favours the ranker that chose what was shown. Treat any comparison as weak evidence and say so.
- With a handful of users, a learned model can only be **per-person**, trained on that person's own ratings ("Refine my matches", see the roadmap). A model across users needs far more people and ratings, and the data collected here is what makes that possible later.
