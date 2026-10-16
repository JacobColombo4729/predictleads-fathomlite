<!-- Generated from notebooks/Evaluate.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: Evaluate (the app's rankers on real career moves)

PredictLeads · TripleTen externship · Jacob Colombo

## 0. Purpose

Scores the rankers the app actually serves on **real career moves, with no labelling** (`STANDARDS.md`, Validation design). `ML_Fit` scored its own methods this way: for Profiles' career changers we know the role each one actually moved into, and a method is scored on whether that role is in its top 3 and top 5 from the person's background alone. This notebook puts the app's three rankers on the same answer key:

| Ranker | What it is | Where |
|:--|:--|:--|
| `rrf` | Jesse's text match: TF-IDF and BGE fused by reciprocal rank | `app/backend/rrf_ranker.py` |
| `mlrank` | Jacob's app: ML_Rank's priority queue over the text match | `app/backend/ml_rank_ranker.py` |
| combined | The combined model: ML_Fit's role fit as the fit factor of ML_Rank's queue, over the text match, on the scope's listings | `combined/` |

**How a listing ranker is scored on roles:** each ranker ranks the serving corpus for a person; its top 20 listings (the app's first page) become a role ranking, a role scoring how many of those listings are in it (ties broken by the best position). `fit_eval.evaluate` then scores it exactly as `ML_Fit` 6b did.

**Clarifying questions**
- Do the app's rankers find the role real career changers moved into more often than the most-common-role baseline?
- Does adding ML_Rank's queue (`mlrank`) or the trained layer's role fit (combined) change that?
- How close does each come to `ML_Fit`'s trained layer (81.7% at top 3), the best the role-level evidence offers?

**What it can't answer:** whether a listing is a good one to apply to (that's `ML_Rank`'s backtest, 8d–8e), and the roles Profiles has no moves into (sales engineers, business and market research analysts: no answer key).

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
- **The anchor holds:** the trained layer from `ML_Fit`'s saved hand-off gives the same hit or miss as `ML_Fit` 6b for 100% of the 563 test people at top 3 and top 5 (81.7% [78.5%, 84.7%] at top 3, 96.3% at top 5), so the setup reproduces `ML_Fit`.
- **The combined model is the best ranker the app serves:** 76.4% [73.0%, 79.8%] of test career changers have the role they actually moved into in the top 3 of its listings, and 90.4% in the top 5, against the baseline's 63.2% and 87.0%. It beats the baseline by 13.1 points (95% interval +8.5 to +17.4; E2 supported), `rrf` by 10.1 points (+5.9 to +14.4; E3 supported) and `mlrank` by 6.6 points (+2.3 to +10.7; E4 supported).
- **The text match alone isn't shown to beat the baseline:** `rrf` reaches 66.3% [62.3%, 70.0%] at top 3, 3.0 points over the baseline with an interval from -2.0 to +8.0 (E1 not supported), and 85.4% at top 5, below the baseline's 87.0%. `mlrank` reaches 69.8% at top 3 and 84.7% at top 5: ML_Rank's queue over the text match helps at top 3 but not at top 5.
- **The order follows the plan's ladder:** text match alone (66.3%) → plus ML_Rank's queue (69.8%) → plus the trained layer's role fit (76.4%), below the trained layer itself (81.7%). The gap from the trained layer is the price of turning listings into roles: the queue's top 20 can come mostly from the person's first role, so the second and third roles get few or no listings and their order falls back to the fixed role order (a likely reason, not tested).
- **By pool (top 3):** the combined model reaches 77.0% for moves within tech (baseline 63.1%), 76.8% for people who retrained after working in another field (baseline 57.3%) and 72.2% for new grads from another field (baseline 70.8%; the trained layer is also 72.2% there). The gain is largest for people who retrained after working, and new grads stay the hardest pool, as in `ML_Fit`.
- **Top 10 instead of 20 listings (secondary):** the same picture: combined 76.4%, `mlrank` 67.5%, `rrf` 65.9%, baseline 63.2% at top 3.
- **Speed:** the rankers load in 3.3 minutes over 193,043 serving listings; per person, `rrf` and `mlrank` take 0.4 seconds and the combined model 1.5 (four text-match calls plus the queue).

## 5. Conclusion

- **E2, E3 and E4 hold; E1 doesn't.** On real career moves, held-out people, the combined model finds the role a career changer actually moved into in its top 3 for 76.4%, against 63.2% for the most common roles, 66.3% for the text match alone (`rrf`) and 69.8% for ML_Rank's queue over the text match (`mlrank`). The text match alone is not shown to beat the baseline.
- **What it means for the demo** (`CUT.md`): demo the combined model, and show this chart as the comparison: text match alone → plus the queue → plus role fit, each a measured step on the same answer key.
- **Limits:** 8 roles only (Profiles has no moves into sales engineering or business and market research analysis); the backgrounds are résumé titles and programs from the public data, read with Profiles' rules, not an app user's résumé through the local reader; this scores which roles a ranker points to, not which listings are good to apply to (that's `ML_Rank`'s backtest, 8d-8e).
- **Next:** the per-factor breakdown in the app (`ANALYSIS.md`); converting listings to roles costs the combined model about 5 points against the trained layer, so showing the person's top roles next to the listings (step 3's output) keeps the full 81.7% visible.
