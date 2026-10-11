# The Retraining Compass

Every compass idea in one place: what it is, what it shows, the ideas still to try, and where each piece is built. The compass is the project's foundation and its lead differentiator (`DIFFERENTIATORS.md`). Status as of 2026-10-10. Decisions behind each piece are dated in [`docs/decisions.md`](docs/decisions.md).

## 1. What it is

**Prior career → the training people took → their first tech job → that role's 2026 openings, pay, closing speed and expected job growth.**

It serves two moves:
- **Changing careers:** retraining from another field into a computer and mathematical role.
- **Moving across or up within tech:** people already in tech moving to a different role, or toward senior and manager roles.

It's built from **2,813 real career switches** in a public resume dataset (`notebooks/Profiles.ipynb`), joined to the **2026 PredictLeads listings** by role bucket. Career paths come from real switches spanning two decades; the 2026 listings supply today's openings, pay and timing.

## 2. What it shows

| Piece | What the user sees | Built in | Status |
|:--|:--|:--|:--|
| **Prior career → first tech job** | Where people from the user's career landed | Profiles 4a.2b | Built |
| **Field trained in → first tech job** | Where each kind of training led | Profiles 4a.2b | Built |
| **Moves within tech** | Which roles people in the user's tech role moved into, sideways or up | Profiles 4a.3 | Built |
| **Each role's 2026 market** | Open listings, new per week, hourly pay, how fast listings close, entry-level share, openness to career changers, remote share | EDA_SDA_2 3.12 | Built (needs a rerun after each change) |
| **Expected job growth** | Growth for each role over **at most the next 5 years**, from official occupational projections (BLS plugs in by occupation code); it drives forward-looking recommendations (section 2b) | EDA_SDA_2 3.12; used in `ML_Fit` | Column ready; BLS 2025–2035 projections released; plan in [`docs/job_growth.md`](docs/job_growth.md) |
| **People behind each path** | "Based on N people"; paths under 10 people marked thin | EDA_SDA_2 3.12 | Built |
| **Suggested roles for this user** | Roles the user's background transfers to | `ML_Fit` | Planned |

### 2b. Why job growth: recommendations based on future estimates

The point of expected job growth is to **recommend roles with the future in mind**, not just to describe them: among the roles a person fits, a role expected to grow ranks higher than one expected to shrink.
- **At most 5 years out.** Projections further ahead are too uncertain to act on; the exact horizon within 5 years is still the team's call.
- **A visible adjustment, not a trained input.** Growth is a future estimate, so it can't be checked against the historical career moves the fit model learns from. `ML_Fit` applies it after the fit model, shows how much it moved each suggestion, and lets the user change its weight.
- **Never alone.** It's occupation-wide, so it's always shown with the role's entry-level share and today's openings: a fast-growing role with no entry-level listings is still a later move.

**Role buckets** (the same on listings and resumes): IT support, networks and systems, security, database, software development, quality assurance, data science, IT project management.

## 3. Path questions it answers

From the resume data; the app's Coach explains them in plain language.
- **Should I retrain for a role directly, or take a first tech role and move from there?** For data science, 218 career changers moved in from another tech role and only 19 retrained straight into it.
- **Which training led people like me to this role, and how many people is that based on?**
- **How long did it take?** People who retrained after working took a median of 15 months from their last old-field job to their first tech job.
- **Which role is the easiest door in?** IT support: the most entry-level listings (23%) and the most openness (44%), but the lowest pay ($31 an hour). Among TripleTen programs, Data Analytics roles are the most open (30%, $47.60 an hour).

## 4. Moving across and up within tech

- **2,049 people moved between tech roles** after 12+ months in the first. The strongest paths: IT support → networks and systems (279), networks and systems → IT support (199), IT support → security (163), developers → databases (142) and → data science (137).
- **Most moves are sideways:** 79% kept the same level, 8.5% moved up, 12.5% moved down (levels read from title words, so rough). "Moving up" needs more evidence before it leads the pitch.
- **IT project management is a later move, not a first step:** 2% of its listings are entry level and 9% show openness, and 190 of its 264 career changers came from another tech role. Business backgrounds also move there (17% of retrained business people). **Decided 2026-10-10:** not suggested as a first tech job unless the user's prior career has a real path into it; shown to users already in tech as a step up; scope-wide openness figures are also shown without it.
- **To try:** a "trained, then moved up in the same role" pool (for example a master's before a senior role). Profiles' rules don't capture it yet; it would reopen Profiles.

## 5. The Ideas tab

A separate tab built on the compass: a personal outlook from the user's résumé. **Ideas to try, not decided. Time frames stay vague** ("down the line", "in the coming years") until the team picks them. Every number is labelled an estimate, worded as "people like you" and "expected", never a promise, and the résumé stays local.

| Idea | What it says | Where the numbers come from |
|:--|:--|:--|
| **Skills to build** | "Based on your current résumé, we think you should build these skills" | Listing skills mapped to O*NET (EDA_SDA_3) vs. the résumé (`ML_Fit`) |
| **Career options down the line** | "Here's what your career options could look like" | Roles similar people reached and how long it took (Profiles), with each role's 2026 openings |
| **Pay increase** | "Here's your pay increase" | Hourly pay in each option (EDA_SDA_2 3.12) vs. the user's current field (EDA_SDA_1 7.2, the occupation summary), or a rate they enter |
| **Job growth** | "Here's the job growth we expect in your career" (within 5 years) | Official projections, next to the role's entry-level share |
| **Stepping-stone roles** | Roles that bridge two functions on the way to a bigger move (for example engineer → pre-sales engineer → director) | Levels at each move (Profiles) |

## 6. How it connects to the rest of the app

- **Picking a role in the compass sets the priority queue's occupation filter;** every other filter (location, work mode, schedule, job type, role level) combines with it.
- **The date picker** shows the queue as of a chosen date (fair from July 1, for example July vs. September).
- **Hide, never remove:** listings out of reach, likely ghosts (over 90 days old and from a company gone quiet) or in an occupation that isn't a tech path (document management specialists) are hidden with a count and a reason, and a "show hidden" toggle brings them back. Other listings over 90 days old stay visible, labelled "open 90+ days" and ranked lower.

## 7. Honesty rules

*The general rules (no causation, hires, success rates, entry-level growth or certain futures) are in [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md), "What we never claim". Compass-specific:*
- **Job growth sits next to the role's entry-level share,** and carries its accuracy (moderate for years 1–2, low to moderate for 3–5; `docs/job_growth.md`); IT project management's figure is a proxy from the 15-1299 catch-all.
- **Show the sample size on every path.** Most paths are thin (78% of prior-career paths rest on fewer than 10 people), so the compass falls back to the bucket-level pattern when a user's exact path is thin.

## 8. Where it's built

| Step | Notebook | Status |
|:--|:--|:--|
| Career paths from real switches | `Profiles.ipynb` | Frozen (2026-10-10) |
| The 2026 market and the joined paths | `EDA_SDA_2.ipynb` 3.12 → `role_market.pkl`, `compass_paths_2026.pkl` | Built |
| Skills mapped to O*NET | `EDA_SDA_3.ipynb` | Planned |
| Suggested roles and skills to build per user | `ML_Fit.ipynb` | Planned |
| **Assembly** into the compass and Ideas tab tables for the app | `ML_Fit.ipynb`'s last section (a separate `Compass.ipynb` is proposed) | Planned |
| The app views | `APPLICATION.md`: Retraining compass, Moving across and up, Ideas tab | Designed |

## 9. Open decisions

- **Time frames** for career options and job growth (team), **with growth capped at 5 years**.
- **A separate `Compass.ipynb`** that assembles everything that doesn't need ML now, with `ML_Fit` adding personal suggestions later, instead of assembling at the end of `ML_Fit`.
- **The "moved up in the same role" pool** (would reopen Profiles).
- **Job growth extensions** (proposed, `docs/job_growth.md` section 4): an error band from BLS's own track record, interest-rate scenarios for years 1–2, listing momentum shown next to growth, a full horizon-weighted blend, and state-level projections. Each would reopen part of the 2026-10-09 external-data decision.
