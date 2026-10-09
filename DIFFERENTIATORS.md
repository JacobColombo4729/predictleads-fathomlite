# Differentiators

How the project stands out from other career-changer projects in the hackathon, and how to pitch it. Decided 2026-10-09. Numbers are aggregates from the notebooks.

## What most teams will build

A résumé compared with job descriptions, returning a match score, maybe with an AI coach on top. That is text similarity (Option A in [`docs/fit_analysis.md`](docs/fit_analysis.md)), and on its own it works against career changers: their résumé is written in their **old** field's language, so it ranks old-field jobs highest.

## What only this project has

- **Live 2026 listings with timing:** when each listing appeared and when it closed (PredictLeads), so we can say how fast to act, not only what fits.
- **Real career switches:** people who retrained into computer and mathematical roles, found in a public resume dataset (`notebooks/Profiles.ipynb`): 17,952 unique people, 2,539 career changers under rules version 8.
- **Role buckets that connect the two:** the same buckets on listings (O*NET codes) and on career switches (job titles).

## Five angles

| Angle | What it means | Evidence we already have |
|:--|:--|:--|
| **1. How fast to move** | Every listing gets an apply-by estimate and urgency, not just a fit score | 28% of listings are gone within a week; employer hiring volume and source change the window 2.3–2.5× |
| **2. Honest reachability** | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer | Entry level ranges from 3% to 70% of listings by field; only 9% of computer and mathematical listings are junior |
| **3. Paths real people took** | "People from your background who trained in X moved into these roles," learned from real switches, not guessed | Retrained career changers with their original background, training field and first tech role |
| **4. Retraining compass** | Training → the roles it led to → each role's 2026 openings, pay and closing speed | Training-field groups and role buckets in Profiles; listing counts, pay and windows by bucket (EDA_SDA_2) |
| **5. Where and when to look** | A watch list of employers hiring at volume or posting entry-level roles, plus a check-every-N-days cadence | High-volume employers post 42% of listings; 300–1,100 companies per field cover most openings |

## What we lead with

**Angles 3 and 4 are the story, angle 1 is the hook.** Angle 2 is built into every view. Angle 5 is a supporting feature.

> **Pitch:** "Everyone tells career changers what they match. We tell them what's realistically reachable, which training actually got people like them there, and how fast they have to move, using real career switches and live 2026 listings."

**Rigor is a quiet differentiator:** testing on later months, testing on held-out real career changers, and limits stated plainly. Teams that don't overclaim stand out.

## How we talk about the data

- **Career paths come from real switches spanning two decades; the 2026 listings supply today's openings, pay and timing.** Paths change slowly and hiring conditions change fast, so each dataset does the job it's good at. This is the framing for the pitch and the deck.
- **The exact date range** belongs in the notebooks' and write-up's limitations, next to how we handle it, not in the pitch.
- **Current evidence that the paths still hold:**
  - O*NET's Career Changers Matrix (current release): related occupations should agree with the paths in the resumes.
  - The compass pairs every path with its 2026 openings and pay.
  - Recent volunteer profiles as a test set, if the advisor approves.

## What we never claim

- **Not causation:** "people like you got there through this training", never "this training gets you hired".
- **Not hires:** listings appear and close; we don't see who got the job, so we never say "you'll get this job".

## Changes made (2026-10-09)

| Where | Change |
|:--|:--|
| `PROJECT_OUTLINE.md` | The pitch line under "The idea", and a short "What makes this different" section |
| `APPLICATION.md` | A **Retraining compass** view next to role discovery: training options → roles people like the user moved into (with counts) → each role's 2026 openings, median pay and closing speed; picking a role feeds the priority queue. The Coach agent explains it; part of build stage 2 |
| `notebooks/Profiles.ipynb` | A training field × first tech role table (4a.2b), as shares of people, hiding cells with fewer than 5 people |
| `notebooks/EDA_SDA_2.ipynb` | A new section, 3.12: each role bucket's 2026 open listings, median pay, median days open and entry-level share; joined to the Profiles table by role bucket |
| `notebooks/ML.ipynb` | Check whether knowing the training field predicts the first tech role better than the background alone |
| Deck | The pitch on the title slide; a new "What makes this different" slide (reachable, real paths, how fast to move); slide 11 shows the training → role → 2026 openings chain; the two-decades framing replaces the date range |

The Profiles table and EDA_SDA_2's 3.12 are written and need a run; the app view is a design only.
