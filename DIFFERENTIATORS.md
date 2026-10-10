# Differentiators

How the project stands out from other career-changer projects in the hackathon, and how to pitch it. Decided 2026-10-09. Numbers are aggregates from the notebooks.

## What most teams will build

A résumé compared with job descriptions, returning a match score, maybe with an AI coach on top. That is text similarity (Option A in [`docs/fit_analysis.md`](docs/fit_analysis.md)), and on its own it works against career changers: their résumé is written in their **old** field's language, so it ranks old-field jobs highest.

## What only this project has

- **Live 2026 listings with timing:** when each listing appeared and when it closed (PredictLeads), so we can say how fast to act, not only what fits.
- **Real career switches:** people who retrained into computer and mathematical roles, found in a public resume dataset (`notebooks/Profiles.ipynb`): 17,952 unique people, 2,813 career changers.
- **Role buckets that connect the two:** the same buckets on listings (O*NET codes) and on career switches (job titles).
- **Job growth expectations per role:** expected growth over at most the next 5 years, from official occupational projections (BLS 2025–2035) joined on the same occupation codes, so role recommendations look ahead. Plan: [`docs/job_growth.md`](docs/job_growth.md).

## Angles

| Angle | What it means | Evidence we already have |
|:--|:--|:--|
| **1. How fast to move** | Every listing gets an apply-by estimate and urgency, not just a fit score | 28% of listings are gone within a week; employer hiring volume and source change the window 2.3–2.5× |
| **2. Honest reachability** | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer | Entry level ranges from 3% to 70% of listings by field; only 9% of computer and mathematical listings are junior |
| **3. Paths real people took** | "People from your background who trained in X moved into these roles," learned from real switches, not guessed. Includes **moving across tech roles** within tech or math, and moving up to senior or manager roles | 764 retrained career changers with their prior career, training field and first tech role; 2,049 people who moved between tech roles (Profiles 4a.3 splits sideways moves from moves up) |
| **4. Retraining compass** | Prior career → the training people took → their first tech job → that role's 2026 openings, pay, closing speed and expected job growth. For changing careers (retraining into a new field) and for moving up (training to reach senior or manager roles in the user's own field). Answers "which path should I take?", including whether to retrain directly or step through another tech role | Prior career × first tech job and training × first tech job (Profiles 4a.2b); data and analytics is reached mostly from other tech roles (218) rather than by retraining (19); listing counts, pay and windows by bucket (EDA_SDA_2) |
| **5. Where and when to look** | A watch list of employers hiring at volume or posting entry-level roles, plus a check-every-N-days cadence; a date picker shows how the list changed (July vs. September 2026) | High-volume employers post 42% of listings; 300–1,100 companies per field cover most openings; every listing has first-seen and last-seen dates |
| **6. Ideas tab** | A separate tab built on the compass, from the résumé: skills to build, career options down the line, the pay increase, expected job growth, and stepping-stone roles (to try; time frames kept vague) | Profiles (time to switch, paths, level at the switch), 2026 listing pay by role, job growth expectations |

Every compass idea, with status and where it's built: [`COMPASS.md`](COMPASS.md).

## What we lead with

**Angles 3 and 4 are the story (the compass leads), angle 1 is the hook.** Angle 2 is built into every view. Angle 6 (the Ideas tab) extends the compass into a personal outlook; angle 5 is a supporting feature.

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
- **Not success rates:** the resumes only show people who made a switch, not those who tried and didn't. The compass shows where switchers went, never the odds that a switch works.
- **Not entry-level growth:** job growth projections cover a whole occupation; entry-level openings come from our listings.

## Changes made (2026-10-09)

| Where | Change |
|:--|:--|
| `PROJECT_OUTLINE.md` | The pitch line under "The idea", and a short "What makes this different" section |
| `APPLICATION.md` | A **Retraining compass** view next to role discovery: training options → roles people like the user moved into (with counts) → each role's 2026 openings, median pay and closing speed; picking a role feeds the priority queue. The Coach agent explains it; part of build stage 2 |
| `notebooks/Profiles.ipynb` | A prior career × first tech job table, then a training field × first tech role table (4a.2b), as percents of all retrained people, fields and roles ordered infrastructure → code → data, with the people count and an "all retrained" row; the within-tech table (4a.3) uses the same layout |
| `notebooks/EDA_SDA_2.ipynb` | A new section, 3.12: each role bucket's 2026 open listings, median pay, median days open and entry-level share; joined to the Profiles table by role bucket |
| `notebooks/ML_Fit.ipynb` | Check whether knowing the training field predicts the first tech role better than the background alone |
| Deck | The pitch on the title slide; a new "What makes this different" slide (reachable, real paths, how fast to move); slide 11 shows the training → role → 2026 openings chain; the two-decades framing replaces the date range |

Added after the team meeting: moving across and up within tech, path questions, job growth expectations, the date picker and the Ideas tab (in `APPLICATION.md` and `PROJECT_OUTLINE.md`). EDA_SDA_2's 3.12 is written and needs a run; the app views are a design only.
