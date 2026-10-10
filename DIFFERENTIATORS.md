# Differentiators

How the project stands out from other career-changer projects in the hackathon, and how to pitch it. Decided 2026-10-09. Numbers are aggregates from the notebooks.

## What most teams will build

A résumé compared with job descriptions, returning a match score, maybe with an AI coach on top. That is text similarity (Option A in [`docs/fit_analysis.md`](docs/fit_analysis.md)), and on its own it works against career changers: their résumé is written in their **old** field's language, so it ranks old-field jobs highest.

## What only this project has

- **Live 2026 listings with timing:** when each listing appeared and when it closed (PredictLeads), so we can say how fast to act, not only what fits.
- **Real career switches:** people who retrained into computer and mathematical roles, found in a public resume dataset (`notebooks/Profiles.ipynb`): 2,813 career changers ([`docs/data.md`](docs/data.md)).
- **Role buckets that connect the two:** the same buckets on listings (O*NET codes) and on career switches (job titles).
- **Job growth expectations per role:** expected growth over at most the next 5 years, from official occupational projections (BLS 2025–2035) joined on the same occupation codes, so role recommendations look ahead. Plan: [`docs/job_growth.md`](docs/job_growth.md).

## Angles

| Angle | What it means | Evidence we already have |
|:--|:--|:--|
| **1. How fast to move** | Every listing gets an apply-by estimate and urgency, not just a fit score | 28% of listings are gone within a week; busy employers fill about 1.4–1.5× faster like-for-like, much more at private high-volume companies (EDA_SDA_2 3.3, 3.8) |
| **2. Honest reachability** | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer | Entry level ranges from 3% to 70% of listings by field; only 9% of computer and mathematical listings are junior; the rules in force hide 22% of open scope listings, and reachable listings pay $11–37 an hour less (EDA_SDA_2 3.11) |
| **3. Paths real people took** | "People from your background who trained in X moved into these roles," learned from real switches, not guessed. Includes **moving across tech roles** within tech or math, and moving up to senior or manager roles | 764 retrained career changers with their prior career, training field and first tech role; 2,049 people who moved between tech roles (Profiles 4a.3 splits sideways moves from moves up) |
| **4. Retraining compass** | Prior career → the training people took → their first tech job → that role's 2026 openings, pay, closing speed and expected job growth. For changing careers (retraining into a new field) and for moving up (training to reach senior or manager roles in the user's own field). Answers "which path should I take?", including whether to retrain directly or step through another tech role | Prior career × first tech job and training × first tech job (Profiles 4a.2b); data science is reached mostly from other tech roles (218) rather than by retraining (19); listing counts, pay and windows by bucket (EDA_SDA_2 3.12: software development 1,175 open listings, IT support the entry door at 44% open to career changers) |
| **5. Where and when to look** | A watch list of employers hiring at volume or posting entry-level roles, plus a check-every-N-days cadence; a date picker shows how the list changed (July vs. September 2026) | High-volume employers post 42% of listings; 300–1,100 companies per field cover most openings; every listing has first-seen and last-seen dates |
| **6. Ideas tab** | A separate tab built on the compass, from the résumé: skills to build, career options down the line, the pay increase, expected job growth, and stepping-stone roles (to try; time frames kept vague) | Profiles (time to switch, paths, level at the switch), 2026 listing pay by role, job growth expectations |

Every compass idea, with status and where it's built: [`COMPASS.md`](COMPASS.md).

## What we lead with

**Angles 3 and 4 are the story (the compass leads), angle 1 is the hook.** Angle 2 is built into every view. Angle 6 (the Ideas tab) extends the compass into a personal outlook; angle 5 is a supporting feature.

> **Pitch:** "Everyone tells career changers what they match. We tell them what's realistically reachable, which training actually got people like them there, and how fast they have to move, using real career switches and live 2026 listings."

**Rigor is a quiet differentiator:** testing on later months, testing on held-out real career changers, checking collection effects before calling something a signal (listings with no description close in 3 days, which made missing pay look predictive), and limits stated plainly. Teams that don't overclaim stand out.

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
- **Not a certain future:** growth is "expected (official projections)", never "will grow", with its accuracy stated (moderate for years 1–2, low to moderate for 3–5) and a range if the error band is adopted.
