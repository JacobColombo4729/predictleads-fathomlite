# The cut: keep the work, cut the surface, freeze the scope

**Status:** the scope freeze is decided (Jacob, 2026-10-16, under 2 weeks left): section 4's defer list is cut; the rest of the plan stands. Proposed 2026-10-15. Mainly Jacob's half (`analysis/`, `combined/`); the demo and doc items touch the app too, so agree them with Jesse.

**Why now:** the project is thorough but hard to explain in the minutes a judge has, and the size is starting to cost accuracy. Nothing here deletes finished work: it changes what is shown first and stops new scope.

## The decision in one line

Lead with three numbers, give each notebook a main path and an appendix, demo one app, and build nothing beyond combined-model steps 4–5 and the real-outcome evaluation of the app rankers.

## Both sides

**For cutting**
- **A judge has minutes, not hours.** Jacob's half alone is seven notebooks plus Profiles, about 20 shared modules (`src/`), two apps and a third model in progress. ML_Rank fits six classifiers, Cox, AFT, a random survival forest and LambdaRank. No one can follow all of that in a pitch.
- **Several pieces don't earn a place in the story:**
  - Skills matching alone scores below the baseline (ML_Fit 6b: carry-over 48.3%, profile cosine 55.6% at top 3, against 63.2%).
  - Cox, AFT, the survival forest and LambdaRank aren't used by the queue; XGBoost's chance still open is.
  - Evergreen wording doesn't line up with the ghost flag (EDA_SDA_3 4); cross-post urgency had little support (EDA_SDA_2 3.5c).
  - Growth is blank until BLS projections arrive.
- **Size is costing accuracy.** On 2026-10-15 alone: the trained layer's test figure was stale (82.4% instead of 81.7%) in four documents, a parity check compared blanks wrongly, and the docs contradicted each other on how the ranker is evaluated. More moving parts, more of these.
- **Upkeep:** every notebook rerun needs a Thread pass and new exports; three app versions triple what the demo can break on.

**Against cutting**
- **Rigor is the differentiator** (`DIFFERENTIATORS.md`): the real-outcome evaluation, the temporal backtests and the leakage audit are what set this apart from a typical "we matched résumés to jobs" project.
- **The course standard asks for breadth:** every applicable course method is used, then taken further (`STANDARDS.md`). Some extra models exist for that, and "tested and not chosen" is a finding.
- **Most of it is finished and works.** Cutting done work saves little, and deleting it risks breaking the hand-offs between notebooks.
- **Negative results show honesty:** "we tested evergreen wording as a ghost signal and it didn't hold" is worth saying.

**Weighed:** both are right about different things. The work is an asset; its presentation is the problem. So cut the surface, not the work.

## The three numbers

Everything else is supporting detail.

| | Result | Against | Source |
|:--|:--|:--|:--|
| **Roles** | 81.7% [78.5%, 84.7%] of held-out real career changers have the role they actually moved into in our top 3 | 63.2% for suggesting the most common roles | ML_Fit 6b; the app's combined model reaches 76.4% over its listings (Evaluate) |
| **Listings** | Half (0.500) of each career changer's top 10 is relevant: still open 14 days, not a ghost, in the role they moved into | 0.031 for newest first, 0.071 for a title search | ML_Rank 8e (test month) |
| **Timing** | 29% of scope listings are gone within a week (median 24 days) | | EDA_SDA_1 Q1b |

## Owning the project: one path to explain

Shrink what has to be explained to **one path: résumé → role fit → candidate pool → queue**, plus the three numbers. If each of those four boxes can be explained on a whiteboard without notes, the project is ours. Everything in the appendices only needs to be recognised: "we tested Cox and AFT, the survival forest did X, we chose XGBoost because Y".

| Box | What it does | How we know it works | What could go wrong |
|:--|:--|:--|:--|
| **Résumé** | Reads a résumé on this computer and turns it into the person's background with Profiles' own rules (`fit_serving`) | Gives exactly Profiles' features for people who switched at a known month (tests) | A scrambled résumé gives a scrambled timeline; undated entries can't be placed |
| **Role fit** | The trained layer predicts which role someone moves into from their background, with the reasons | On 563 held-out real career changers, the right role is in its top 3 for 81.7%, against 63.2% for guessing the common roles | Only 8 roles have a fit model; growth is blank until BLS data arrives |
| **Candidate pool** | Text matches overall plus the best text matches within the person's top 3 roles, hidden listings left out | Fictional people: the top roles add about 210 listings the text match alone misses | The text match decides which listings of a role are considered |
| **Queue** | ML_Rank's weighted sum of fit, still open, accessibility and pay | On the test month, half of each career changer's top 10 is relevant, against 0.031 for newest first | Listings outside the 8 roles get a neutral fit; the weights come from one validation month |

For each main-path piece, be able to answer three questions:
1. **What does it do?** For example, "the trained layer predicts which role someone moves into from their background."
2. **How do we know it works?** "On 563 held-out real career changers it gets the right role in the top 3 for 81.7%, against 63.2% for just guessing the common roles."
3. **What could go wrong?** "A scrambled résumé gives a scrambled timeline; listings outside the 8 roles get a neutral fit."

**The concepts worth knowing cold,** because judges and interviewers ask about them:
- **Leakage:** why features only use what's known when a listing first appears (never `last_seen_at`, status or days open).
- **Censoring:** why still-open listings aren't dropped (dropping them makes windows look shorter than they are; survival methods count them as unfinished).
- **Baselines:** why 81.7% only means something next to 63.2%.
- **Held-out test, run once:** and why tuning on it would be cheating (the score would measure how well we fit the test, not how well it generalises).
- **The weighted sum:** what the queue's score is. Each factor is rated 0–100 and multiplied by its tuned weight (fit 0.3, still open 0.2, accessibility 0.1, pay 0; so fit is half the score), and the parts add up to the score shown on each listing.

## What to do

### 1. Lead with the three numbers
- [ ] One short entry page, "How it works and how we know" (`analysis/SUMMARY.md` or the top of `README.md`): the three numbers, the pipeline in five lines, links to the rest.
- [ ] The same three numbers open the pitch deck and each notebook's purpose cell.

### 2. A main path and an appendix in each notebook (nothing deleted)
- [ ] **ML_Rank:** main path = targets, split, XGBoost (chance still open), the queue (8d–8e). Appendix: one table, "alternatives tested and why not chosen" (Cox, AFT, random survival forest, LambdaRank, the other classifiers).
- [ ] **ML_Fit:** main path = the ladder's result (baseline → RRF → trained layer) and the compass. Appendix: skills matching, per-pool detail, current-role pull.
- [ ] **EDA_SDA_1–4, TimeSeries:** main path = the findings that feed the queue (timing, accessibility, pay, company volume). Appendix: features that didn't hold up (evergreen, cross-post urgency) as one "tested, no usable signal" table.

### 3. One app in the demo
- [ ] Demo the combined model only.
- [ ] Jesse's `rrf` and `mlrank` appear as the comparison, one chart on the same career moves: text match alone → plus the queue → plus role fit (needs the evaluation in 4).

### 4. Freeze the scope
**Finish:**
- [x] Combined model steps 4–5: candidates within the person's top roles, then ML_Rank's queue (`combined/`; built 2026-10-15, v2 2026-10-16).
- [x] The real-outcome evaluation of `rrf`, `mlrank` and the combined model (done 2026-10-15, `notebooks/Evaluate.ipynb`: combined 76.4%, `mlrank` 69.8%, `rrf` 66.3%, baseline 63.2% at top 3) (`ANALYSIS.md`; combined-model plan step 7).
- [ ] The per-factor breakdown in the app ("Why this listing": the four factors and the role's top reason), after the evaluation (`ANALYSIS.md`); it meets `STANDARDS.md`'s "every ranking can be explained".
- [x] The plug-in, as its own app: `combined_demo/` (built 2026-10-15; nothing in `app/` changed).

**Cut (decided 2026-10-16; moved to Later in `ANALYSIS.md`, marked in the combined-model plan; nothing deleted):**
- [x] Combined step 6 (the trained layer driving the Compass: the Compass stays as Jesse built it) and step 8 (BGE fine-tuning).
- [x] Growth (BLS) beyond the slider at weight 0.
- [x] ML_Rank sections 9, 10 and 12 beyond short text; H7 (training without ghosts); 7f's extra baselines.
- [x] New differentiator ideas and new features: ideas 2, 4, 6 and 7 are cut; 1, 3 and 5 (already drafted as tested functions) stay.

**Kept, though time is short (decided 2026-10-16):** the notebook main paths and appendices (2), the docs consolidation (5) and the per-factor breakdown (above).

### 5. Consolidate the docs
About ten planning documents today: `PROJECT_OUTLINE.md`, `STANDARDS.md`, `DIFFERENTIATORS.md`, `ANALYSIS.md`, `app/APP.md`, `app/DESIGN.md`, `app/roadmap.md`, `app/docs/ml-inventory.md`, `app/docs/combined-model-plan.md`, and this file.
- [ ] The entry page (1) is the only thing a newcomer reads first; the rest become references it links to.
- [ ] Quoted numbers live in one place each (the notebook that makes them) and are linked, not copied, so they can't go stale in four places again.

### 6. Keep tooling out of the story
- [ ] Eve, Thread, Pace, Foreman and the safety guard stay as engineering hygiene, mentioned in one line at most; they are not part of the pitch.

## How far to go depends on the deadline

Under 2 weeks left (as of 2026-10-16; `CLAUDE.md`), so the short plan applies, with the three kept items above on top.
- **Weeks left:** do 1–4; start 5.
- **Days left:** do 1 and 3, finish step 4–5 of the combined model, and stop building.

## Open
- [ ] Agree the cut with Jesse (the demo and doc items touch the app).
- [x] Set the deadline in `CLAUDE.md` (under 2 weeks as of 2026-10-16; the exact date still to come).
