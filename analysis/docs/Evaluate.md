<!-- Generated from notebooks/Evaluate.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: Evaluate (the app's rankers on real career moves)

PredictLeads · TripleTen externship · Jacob Colombo

## 0. Purpose

Scores the rankers the app actually serves on **real career moves, with no labelling** (`PROJECT.md`, How we keep it honest). `ML_Fit` scored its own methods this way: for Profiles' career changers we know the role each one actually moved into, and a method is scored on whether that role is in its top 3 and top 5 from the person's background alone. This notebook puts the app's three rankers on the same answer key:

| Ranker | What it is | Where |
|:--|:--|:--|
| `rrf` | Jesse's text match: TF-IDF and BGE fused by reciprocal rank | `app/backend/rrf_ranker.py` |
| `mlrank` | Jacob's app: ML_Rank's priority queue over the text match | `app/backend/ml_rank_ranker.py` |
| combined | The combined model (**v2**): ML_Fit's role fit as the fit factor of ML_Rank's queue, over the text match, on the scope's listings. v2 blends the role choice with the role mix of the person's text matches and puts a stated desired role first (`combined/evaluate.py`) | `combined/` |

**How a listing ranker is scored on roles:** each ranker ranks the serving corpus for a person; its top 20 listings (the app's first page) become a role ranking, a role scoring how many of those listings are in it (ties broken by the best position). `fit_eval.evaluate` then scores it exactly as `ML_Fit` 6b did.

**Clarifying questions**
- Do the app's rankers find the role real career changers moved into more often than the most-common-role baseline?
- Does adding ML_Rank's queue (`mlrank`) or the trained layer's role fit (combined) change that?
- How close does each come to `ML_Fit`'s trained layer (81.7% at top 3), the best the role-level evidence offers?

**What it can't answer:** whether a listing is a good one to apply to (that's `ML_Rank`'s backtest, 8d–8e), and the roles Profiles has no moves into (sales engineers, business and market research analysts: no answer key). **It also can't see half of what v2 does:** Profiles' backgrounds are résumé titles, programs and dates — exactly the trained layer's own inputs — and carry no desired role, so v2's two changes (blending in the text match's role mix, putting a desired role first) are scored here with their cost and none of their benefit. A number from this notebook favours v1 by construction; section 5 says what follows from that.

## Contents

- [0. Purpose](#0-purpose)
- [1c. Hypotheses and decision rule (written before any ranker is run)](#1c-hypotheses-and-decision-rule-written-before-any-ranker-is-run)
- [2. The career changers and the split](#2-the-career-changers-and-the-split)
- [3. The rankers and the anchor](#3-the-rankers-and-the-anchor)
- [4. The test (run once)](#4-the-test-run-once)
- [5. Conclusion](#5-conclusion)

## 1c. Hypotheses and decision rule (written before any ranker is run)

The test people are `ML_Fit`'s held-out 20%, scored **once**, at the end; nothing here is tuned. **One rule decides E1 to E4:** the first method beats the second at top 3, and the 95% bootstrap interval for the per-person difference in hits (people resampled; `rm.clustered_bootstrap_diff` with each person as their own cluster) excludes zero. The size of the difference is reported either way.

| # | Hypothesis | Compared with | Supported when |
|:--|:--|:--|:--|
| E1 | `rrf` finds the moved-into role more often than the baseline | Most common role (baseline), top 3 | Difference positive, interval excludes 0 |
| E2 | The combined model beats the baseline | Same | Same |
| E3 | The combined model beats `rrf` | `rrf`, top 3 | Same |
| E4 | The combined model beats `mlrank` | `mlrank`, top 3 | Same |

**Fixed before the run:** the top 20 listings per person; a role's score is its count among them, ties broken by the best position; listings outside `ML_Fit`'s 8 roles don't count; for moves within tech the current role is left out, as in `ML_Fit`. **Secondary, not deciding:** top 5, the pools, and the top 10 listings instead of 20.

**Anchor (a check, not a hypothesis):** the trained layer scored from `ML_Fit`'s saved hand-off must give the same hits as `ML_Fit` 6b on the same people, or the setup is wrong and nothing below is used.

## 2. The career changers and the split

`ML_Fit` 2a's people and split (`fit_eval.split_people`, seed 0): the development people give the baseline, the test people are scored.

## 3. The rankers and the anchor

One RRF ranker over the serving corpus is shared by all three (loading it fits TF-IDF over the corpus: a few minutes).

## 4. The test (run once)

Each ranker ranks the whole serving corpus for each test person, from their pre-switch background (titles and programs, as in `ML_Fit`). About a second or two per person per ranker.

**What the data shows**
- **The anchor holds:** the trained layer from `ML_Fit`'s saved hand-off gives the same hit or miss as `ML_Fit` 6b for 100% of the 563 test people at top 3 and top 5 (81.7% [78.5%, 84.7%] at top 3, 96.3% at top 5), so the setup still reproduces `ML_Fit` exactly and the rankers are the only thing that moved.
- **No hypothesis is supported.** E1 `rrf` over the baseline +4.6 points (95% interval −0.9 to +10.0), E2 combined over the baseline +5.0 (+0.0 to +9.9, touching zero), E3 combined over `rrf` +0.4 (−2.7 to +3.4), E4 combined over `mlrank` **−1.2** (−4.4 to +2.0). By the 1c rule none of the four holds: on this answer key the three rankers are not separated from each other, and only the trained layer is clearly above the baseline.
- **The three rankers land within 1.5 points at top 3:** `mlrank` 69.4% [65.7%, 73.2%], combined 68.2% [64.5%, 72.1%], `rrf` 67.9% [63.9%, 71.6%], against the baseline's 63.2% [59.1%, 67.1%] and the trained layer's 81.7%. **At top 5 none of them beats the baseline** (87.0%): `mlrank` ties it, `rrf` 86.1%, combined 85.4%.
- **v1 scored 76.4% here and v2 scores 68.2%; the drop is the design change, not a regression.** v1 took the roles from the trained layer alone. v2 blends it equally with the role mix of the person's text matches, because on a real résumé the trained layer reads only titles, programs and dates and was leading with the wrong role (`JACOB.md`, 2026-10-16). The trained layer is the strongest role signal in the project (81.7%), so blending it toward the text match's 67.9% gives up most of v1's lead on this key — while v2's compensating half, a stated desired role ranked first, cannot appear at all, because Profiles has no desired-role field (`combined/evaluate.py`). v2 also caps the pool at the 30 best text matches per role, so the two versions differ in more than the blend and the 8.2-point gap has no single clean cause.
- **The ladder no longer rises:** text match alone 67.9% → plus ML_Rank's queue 69.4% → plus v2's role fit 68.2%, against the trained layer's 81.7%. The queue adds 1.5 points and v2's role fit gives back 1.2, neither interval excluding zero. The honest reading is that over whole-corpus listing rankings these three are alike, and the project's measured gain lives in the role model rather than in the ranker stack. The gap from the trained layer is also wider than before — 13.5 points against v1's 5.3 — so turning roles into listings is no longer the only thing between them.
- **By pool (top 3), v2 helps the people the project is for and hurts the ones who dominate the sample.** The combined model is the weakest of the three for moves within tech (66.0%, against `rrf`'s 68.9% and `mlrank`'s 68.7%) and the strongest for both retrained pools: 73.2% for people who retrained after working in another field (`mlrank` 69.5%, `rrf` 64.6%, baseline 57.3%), and 75.0% for new grads from another field, where it beats even the trained layer's 72.2%. Within-tech movers are 409 of the 563 test people, so they set the overall number. These two pools hold 82 and 72 people and this table carries no intervals, so that pattern is a direction worth testing, not a demonstrated difference.
- **Top 10 instead of 20 listings (secondary):** the same flat picture: `mlrank` 67.9%, combined 67.3%, `rrf` 65.9%, baseline 63.2% at top 3.
- **Speed:** the rankers load in 3.6 minutes over 193,043 serving listings (28,249 hidden by default); per person, `rrf` and `mlrank` take 0.4 seconds and the combined model 1.5 (four text-match calls plus the queue).

## 5. Conclusion

- **E1, E2, E3 and E4 all fail.** On real career moves, held-out people, the three rankers the app can serve are not separated from each other at top 3 — `mlrank` 69.4%, combined model (v2) 68.2%, `rrf` 67.9% — and the closest any comes to the 1c rule is combined's +5.0 points over the 63.2% baseline, on an interval that touches zero. At top 5 none of them beats the baseline.
- **The measured advantage of this project is the role model, not the ranker stack.** `ML_Fit`'s trained layer finds the role a career changer actually moved into for 81.7% [78.5%, 84.7%] of the same people. Everything the app does on top of a text match — the queue, v2's role fit — moves this metric by one or two points, inside its own noise.
- **v1 against v2 is a real trade-off, and only one side of it is measured.** v1 scored 76.4% here by taking roles from the trained layer alone; v2 blends in the text match's role mix so a real résumé leads with the right role, and always puts a stated desired role first. This answer key cannot see either fix: Profiles' backgrounds are the trained layer's own inputs and carry no desired role. So the drop to 68.2% is real but partial evidence, and deciding between the versions on this number alone would be deciding on the half that suits v1.
- **What it means for the demo** (`ANALYSIS.md`, Pitch): the "text match → plus the queue → plus role fit" ladder is not a rising staircase on v2 and should not be shown as one. What the evidence carries is the role result (81.7%, `ML_Fit` 6b) and the listing backtest (precision@10 0.500, `ML_Rank` 8e); this notebook's own contribution is the flat comparison and the reason it is flat.
- **Limits:** 8 roles only (Profiles has no moves into sales engineering or business and market research analysis); the backgrounds are résumé titles and programs from the public data, read with Profiles' rules, not an app user's résumé through the local reader, and they carry no desired role, so v2 is scored in its least favourable setting; this measures which roles a ranker points to, not which listings are good to apply to (that's `ML_Rank`'s backtest, 8d–8e); the per-pool pattern rests on 82 and 72 people with no intervals.
- **Next:** decide which version the app ships and what the pitch claims about the rankers (`ANALYSIS.md`, "Decide which combined-model version the app ships", with Jesse — his `app/APP.md` and `app/docs/ml-inventory.md` still quote v1's numbers). A fair test of v2 needs an answer key with desired roles and full résumé text, which Profiles cannot provide; until there is one, v2's advantage rests on the four résumé checks in `combined/tests/test_roles_from_profile.py`, not on held-out people.
