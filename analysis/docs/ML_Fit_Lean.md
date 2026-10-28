<!-- Generated from notebooks/ML_Fit_Lean.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# ML_Fit (Lean): Which Roles a Career Changer Fits

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

**Which roles and listings fit a career changer?** This notebook builds the **fit model**: from a career changer's pre-switch background, it suggests the role buckets their skills transfer to (**role discovery**), scores how well each listing fits them, and **assembles the retraining compass** and the **Ideas tab's** inputs in its last section: the one place the pieces from Profiles, EDA_SDA_2, EDA_SDA_3 and this notebook come together for the app. `ML_Rank` combines its fit scores with the listing model into the priority queue.

**Inputs:** the **Profiles modeling table** (`data/profiles/modeling_table.pkl`: `feature` columns only, `outcome` columns never), each career changer's **pre-switch text** (job titles and programs before the switch, from the Profiles timeline), EDA_SDA_3's cleaned descriptions and **skills mapped to O*NET**, and O*NET's skills and Career Changers Matrix.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, how does tech compare to other fields, and what's in scope? | Scope table, descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which in-scope listings do we keep, tag or hide? | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Skills mapped to O*NET | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| Profiles | Who really changed into or within tech or math, and how? Career-switch rules on the public résumé data (frozen) | The 2,813 career changers, and the table ML_Fit trains on |
| **ML_Fit** | **Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves** | **Fit scores, role discovery, the compass** |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |
| Evaluate | How well do the app's rankers do on real career moves? `rrf`, `mlrank` and the combined model | The ranker comparison |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### The ladder of methods

| Step | Method | Role | Why it's here |
|:-:|:--|:--|:--|
| 1 | **Keyword overlap** (shared title and skill words) | Simplest baseline | What a keyword search would find |
| 2 | **TF-IDF cosine** | Baseline ML model | The course standard for text similarity |
| 3 | **BGE embeddings, cosine** (a small or base BGE model, run locally) | Stronger model | Catches meaning, not just shared words |
| 4 | **RRF hybrid** (reciprocal rank fusion of TF-IDF and BGE ranks) | Hybrid | Combines word overlap and meaning without tuning weights |
| 5 | **Skills-based matching** (résumé and listing both mapped to O*NET skills) | Transferable fit | Steps 1–4 compare words, and a career changer's résumé uses the old field's words, so they favor old-field roles; skills don't |
| 6 | **Trained layer (Option B)**: steps 2–5 scores plus Profiles features (origin, training field, years before, `from_level`, …) in a model trained on real career moves. Candidates: logistic regression, LightGBM / CatBoost / XGBoost, and a **small neural network (scikit-learn `MLPClassifier`)**, kept only if it wins | **The project's fit model** | Turns similarity into a prediction learned from where career changers actually went; explainable with SHAP |
| 7 | **Stretch: fine-tuned BGE** | Not attempted | Step 6 already cleared the baseline by 18.5 points at top 3 (6b); tested, not needed |

**How the steps are compared:** for each career changer, rank the role buckets using only their pre-switch background and check whether the role they **actually moved into** is in the top 3 or top 5 (hit rate), **per pool** (retrained after working, retrained as new grads, within tech) and overall, against a title-keyword baseline. For within-tech moves, the person's current role is excluded from the candidates (a move means a different role). Also measured: how often each method suggests the person's **old** field, the bias steps 5 and 6 are meant to fix. No teammate ratings are used: the team's judgments would put personal bias into the measure (`CLAUDE.md`), so fit is measured only on real career moves. **IT project management** is left out of first-job suggestions unless the person's prior career has a real path into it (decided 2026-10-10); hit rates are reported with this rule, and for the retrained pools also without it, to show what it costs.

### Clarifying questions
- Which step of the ladder finds the role real career changers moved into most often, and by how much over the keyword baseline?
- Does a small neural network beat gradient boosting on this small table (2,813 career changers)? Kept only if it wins on held-out people.
- **Why no LLM matching:** listing text can't go to outside AI services, a local LLM on 8 GB is too small and slow for matching, and its judgments can't learn from or be validated against real career moves. LLM agents stay in the app layer (the Coach), working from model outputs and aggregates.
- Do the similarity methods (1–4) favor the old field, and do skills-based matching and the trained layer fix it?
- Does knowing what someone trained in predict their first tech role better than their background alone? (The evidence behind the retraining compass.)
- Does training on everyone who reached tech (with origin as a feature) predict career changers' roles better than training on career changers alone? Results are always reported on career changers.
- **Results per pool:** within-tech moves are 73% of career changers (2,049 of 2,813), so one overall score mostly measures them; every metric is also reported per pool.
- **Role imbalance** (quality assurance and data science are the smallest roles): class weights first, per-role scores, and probabilities corrected to the real role mix. Synthetic people only for a role still predicted poorly, inside training folds only.
- Which skills should a career changer build for each suggested role (listing skills their background doesn't show)? This feeds the compass ideas.

**Leakage rules** (`PROJECT.md`, How we keep it honest):
- Only **pre-switch** text and `feature` columns. Résumé skills aren't dated, so they may be from after the switch and aren't used.
- TF-IDF vocabularies, embeddings caches used for training, and the trained layer are fit **inside each training fold**; folds are grouped by person.
- Compass and path counts used as features are rebuilt inside each training fold, so a test person never counts toward their own answer.
- **Claims:** results describe where switchers went, never how likely a switch is to work.

**Data rules:** aggregates only. BGE runs **locally** (sentence-transformers); no résumé or listing text goes to outside services. Embeddings are cached under `data/`, keyed by id. About 14,000 scope listings fit a small or base model on an 8 GB laptop. Clear outputs before committing.

### Roadmap

| Section | Course standard | Taken further |
|:--|:--|:--|
| 1. Setup | Load the Profiles modeling table, pre-switch text, EDA_SDA_3's listing text and skills | |
| 2. Targets | The role bucket each career changer moved into | Per pool |
| 3. Text preparation | Normalize, stopwords, lemmatize; TF-IDF (`ngram_range=(1, 2)`, `sublinear_tf`, `max_df≈0.5`, `min_df` 2–5) fit on training folds only | BGE embeddings, chunked and normalized; O*NET skill mapping |
| 4. The ladder, steps 1–5 | Keyword overlap and TF-IDF cosine, with a **DummyClassifier** / majority-role baseline | BGE cosine, RRF hybrid, skills-based matching |
| 5. The trained layer | Logistic regression, random forest, LightGBM / CatBoost / XGBoost and a small neural network (`MLPClassifier`, scaled inputs, early stopping) with `GridSearchCV`, `class_weight`, `Pipeline` + `ColumnTransformer` | LightGBM LambdaRank for listing order; stretch: fine-tuned BGE (step 7) |
| 6. Evaluation | Accuracy, F1, confusion matrix on held-out people | Hit rate@3 / @5 per pool; old-field bias (real career moves only; no team ratings) |
| 7. Interpretation | Feature importance | SHAP, for "why this role" in the app, and **per listing** for the detail view's rating breakdown: the top features behind each fit rating (skills that carry over, similarity of the text, the user's prior career), grouped under plain-language names and handed to `ML_Rank` with the fit scores |
| 7b. **Growth-adjusted recommendations** | | Applied **after** the fit model as a visible, user-adjustable weight (0 = no adjustment), never trained in - a future projection can't be validated against past moves. Waits on `role_growth.pkl` (BLS, pending PredictLeads' consent; job growth tasks in `ANALYSIS.md`); until then the weight stays 0 and 7b says so. |
| 8. **The compass, assembled** | | The single place the compass is put together for the app: Profiles' paths (prior career → training → first tech job; within-tech moves, sideways or up) with people counts and thin paths marked; each role's 2026 openings, hourly pay, closing speed, entry-level share and expected growth (EDA_SDA_2 3.12); the Ideas tab's inputs: skills to build per suggested role, options and time to switch, pay increase against the current field (EDA_SDA_2 1c) and growth. Saved as compass tables under `data/clean/` |
| 9. Solution report | Steps, difficulties, final model and score | Hand-off of fit scores to `ML_Rank` |

## Contents

- [Purpose](#purpose)
  - [The ladder of methods](#the-ladder-of-methods)
  - [Clarifying questions](#clarifying-questions)
  - [Roadmap](#roadmap)
  - [About this Lean draft](#about-this-lean-draft)
- [2. Targets and the split](#2-targets-and-the-split)
- [3. Role descriptions](#3-role-descriptions)
- [4. The methods (ladder steps 1–4)](#4-the-methods-ladder-steps-14)
- [5. Skills matching and the trained layer (ladder steps 5–6)](#5-skills-matching-and-the-trained-layer-ladder-steps-56)
- [6. Evaluation](#6-evaluation)
- [7. Why a role is suggested (SHAP)](#7-why-a-role-is-suggested-shap)
  - [7b Growth-adjusted suggestions](#7b-growth-adjusted-suggestions)
- [8. The compass, assembled](#8-the-compass-assembled)
- [9. Solution report and hand-off](#9-solution-report-and-hand-off)
  - [Solution report](#solution-report)

### About this Lean draft

A **replica** of `ML_Fit`, threaded into the Lean family rather than rewritten from its question (`docs/lean_conventions.md`, Replicas). The formatting helpers are gone (rule 1): 19 `show()` calls are plain `print()` + `display()`, and the unused survival helpers left the import.

**The change that is not cosmetic: everything this notebook saves now goes to `data/clean/lean/`** - the four compass tables, `fit_scores`, `fit_breakdown`, `fit_inputs`, `fit_choice.json` and the `ml_fit/` hand-off. As it arrived it wrote them beside `ML_Fit`'s own, and `ml_fit/` is what `combined/role_fit.py` loads, so a run of this draft would have replaced the model the combined demo serves. It still **reads** the original chain's tables, including the frozen `Profiles` tables; only the writes moved.

**Run 2026-10-27, on the original chain.** It reads `data/clean/` and the frozen `Profiles` tables (`data/profiles/modeling_table.pkl`, `pre_switch_text.pkl`), not `data/clean/lean/` or `data/profiles/lean/`. Every figure in the findings and the solution report reproduces `ML_Fit`'s saved run; only the table formatting differs (the `show()` helper is gone, so some tables print plain numbers). The leakage-audit figures in the solution report are quoted from `reports/ML_Fit_audit.md`, which audits `ML_Fit`'s own saved inputs, not this notebook's.

## 2. Targets and the split

The cases are the **career changers** from Profiles. For each one, the input is their **pre-switch text** (job titles and study programs from before the switch, `data/profiles/pre_switch_text.pkl`, Profiles 5c) and the answer is the **role bucket they actually moved into** (`target_bucket`). For moves within tech, the person's current role (`from_bucket`) is left out of the candidates: a move means a different role.

**20% of the people are held out as the test set and scored once, at the very end** (section 6b). Everything else is developed on 5 folds of the rest. Both are split by person and stratified by role and pool, so every role and pool shows up in each.

**What the data shows**
- **2,813 career changers, split before anything is scored:** 2,250 for development (1,640 changed roles within tech, 325 retrained after working in another field, 285 new grads from another field) and 563 held out for the test (409 / 82 / 72), 20% of each pool. Within-tech movers are 73% of the people (2,049), so every result below is also shown per pool.
- **Everyone has pre-switch text:** 0 of the 2,813 are empty, so no person ties across all methods.
- **One warning from the split (2a), fixed but not silenced:** `fit_eval.split_people` now recomputes which role-by-pool groups are too small ("rare") using development people only, not everyone, for the fold split - the loose end noted here before. On this data it changes nothing: every group that was borderline under the old (all-people) cutoff is still well above it after removing the test share, so the fold assignments, and every number below, are unchanged from the prior run. scikit-learn still warns ("the least populated class has only 2 members"), because the one group genuinely this small - 3 people in the whole dataset, 2 after the split - stays too small for 5 folds no matter which people it is measured on; there simply aren't more of them. It touches a handful of people and no number below depends on it.

## 3. Role descriptions

Each role bucket is described by its **2026 listings**: a reproducible sample of up to 2,000 per role from EDA_SDA_1's scope table, each listing's title and description as plain text. The methods compare a person's background with these. No person is involved here, so nothing about the test people leaks into the methods. Later, EDA_SDA_3's cleaned text can replace these with one setting.

**What the data shows**
- **Every role is described by the same number of listings:** all 8 roles reached the cap of 2,000 sampled listings, including the smallest roles (quality assurance, data science). The sample says nothing about how many listings a role really has; it only gives each role the same amount of text to compare with.

## 4. The methods (ladder steps 1–4)

Each method scores every role for every person; the higher, the better the fit:
1. **Keyword overlap:** how many of the role's most common title words appear in the person's background.
2. **TF-IDF cosine:** word weights learned from the listings; a role is the average of its listings.
3. **BGE cosine:** the pretrained BGE model, run locally (the person as a query, a role as the average of 300 of its listings).
4. **RRF:** TF-IDF and BGE fused by reciprocal rank (k = 60, as in the app).

The **baseline** suggests the roles in order of how common they are among the training people. None of the four methods learns from people, so each is scored once on everyone; only the baseline is refit per fold.

**What the data shows**
- **All four methods scored all 2,813 people against all 8 roles in 6.3 minutes** (BGE on the laptop's CPU included). The scores are computed once per person, so they are the same in the development folds (6a) and the test (6b); only the baseline is refit per fold.

## 5. Skills matching and the trained layer (ladder steps 5–6)

**Step 5, skills-based matching** (`src/fit_model.py`). Steps 1–4 compare words, and a career changer's background uses the old field's words. Here the background and each role are both described by **O*NET's Transferable Skills** (25 skills, importance 1–5; O*NET 31.0, USDOL/ETA, CC BY 4.0):
- **The background:** the O*NET occupations of the person's background area (`BACKGROUND_SOC`, the same mapping as EDA_SDA_3 6f), or their current role for a move within tech. "Other" and "unknown" backgrounds have no profile and score 0 for every role (a tie, which counts against the method, as empty text does in steps 1–4).
- **A role:** the O*NET occupations that map to it.
- **Two scores:** *carry-over*, the importance-weighted share of the role's important skills (3+) that the background also rates 3+; and *profile cosine*, the cosine between the two profiles after subtracting the average over all O*NET occupations (every occupation rates every skill, so raw profiles all look alike).

Like steps 1–4, nothing here learns from people, so the scores are computed once for everyone.

**Step 6, the trained layer.** A classifier per person over the 8 roles, from the Profiles `feature` columns (origin, background area, training field, years and jobs before, study level, starting role and level for within-tech moves) plus every method's score for every role, raw and relative to the person's own average. Candidates: logistic regression, random forest, LightGBM, XGBoost, CatBoost and a small neural network (`MLPClassifier`), each in a `Pipeline` (one-hot encoding, median imputation with was-blank flags, scaling where the model needs it), tuned with `GridSearchCV` on the **5 development folds** and scored by the top-3 hit rate (current role left out, as in section 6). The test people are never used here: the chosen model's development scores are out-of-fold (each fold predicted by a model fit on the other four), and the test people are scored by the same settings fit on all development people. A version on the Profiles features alone shows how much the scores add.

The best settings are chosen on the same folds that score them in 6a, so 6a's figure for the trained layer is slightly optimistic; the test (6b) is the fair comparison.

**What the data shows**
- **Skills matching alone is weak, and below the baseline.** O*NET's 25 Transferable Skills do not separate the tech roles: 910 occupations, 8 of 8 roles have a profile, and every tech role's top skills are the same general ones (Complex Problem Solving, Judgment and Decision Making). Only software development and quality assurance lead with Programming, and IT project management with Coordination and Time Management. Backgrounds are 18 broad areas, and 95.6% of career changers have a background profile (within tech 100%, retrained after working 83.8%, new grads 84.0%; the 123 with an 'other' or 'unknown' background have none and score 0). The result is 46.7% (carry-over) and 56.6% (profile cosine) at top 3 in development, below the baseline's 63.3% (6a).
- **The trained layer is the fit model.** All six candidates were tuned in 1.4 minutes on the development folds (106 inputs per person), and their top-3 hit rates are within 2 points: CatBoost 83.3% ± 1.6%, logistic regression 83.2% ± 1.6%, XGBoost 82.8%, random forest 82.4%, LightGBM 82.0% and the neural network 81.4%. CatBoost is highest, and logistic regression is chosen by the rule set before looking (the simplest candidate within one standard deviation of the best), not because it scored highest; the choice of model matters little. On the test (6b) it reaches 81.7% at top 3 against the baseline's 63.2% (about 18.5 points) and 96.3% at top 5 against 87.0%: the first step to beat the baseline at top 5.
- **Most of the gain is in the Profiles features, and the scores add the rest.** With the Profiles features alone (training field, starting role and the other `feature` columns) the layer reaches 75.8% at top 3 on the test; adding the methods' scores lifts it to 81.7% (5.9 points; 6.6 in development) and, at top 5, from 94.5% to 96.3%.
- **Caveat:** the chosen logistic regression C of 0.03 is the edge of the grid {0.03, 0.3, 3}, so a smaller value might do slightly better. It is left as is, because the test has been seen. Because the settings were chosen on the same folds that score them in 6a, 6a's figure for the trained layer is slightly optimistic; 6b is the fair one.

## 6. Evaluation

**Hit rate @k:** the share of people whose actual role is in the method's top k (3 and 5), overall and per pool. **Development (6a):** mean ± standard deviation over the 5 folds. **Test (6b):** run **once**, on the held-out 20%, with 95% intervals from resampling people (2,000 bootstrap samples). **Current-role pull (6c):** for moves within tech, how often a method would put the person's current role first if it were allowed: the old-field bias word matching tends to have. **Rule:** the test was run once to see steps 1–4 (2026-10-13); every choice in section 5 is made on the development folds only, and 6b was then run with steps 5–6 (2026-10-13). A rerun that day chose a different model (CatBoost instead of logistic regression) because keyword overlap wasn't reproducible: words tied at its 200-word cutoff came out in a different order in every new kernel. That is fixed (ties broken alphabetically, tested), and the model is now chosen by a rule set before looking: the simplest candidate within one standard deviation of the best on the development folds. Neither change used the test. This run of 6b, with the fix and the rule, is the last one for the fit model. Section 6 scores every method in `scores`, so steps 5–6 appear in each table.

**What the data shows (development, 5 folds)**
- **At top 3, the text methods beat the most-common-role baseline only modestly:** RRF 67.1% ± 1.4% and TF-IDF 66.6% ± 1.2% against the baseline's 63.3% ± 0.3%, BGE 65.5% ± 1.3%. Keyword overlap is below the baseline (54.6% ± 1.6%). The best method gains 3.8 points over the baseline; that is larger than the fold-to-fold spread but was not tested as a paired difference.
- **At top 5, the baseline wins:** 87.1% ± 0.3% against 85.2% (RRF), 85.1% (TF-IDF), 85.0% (BGE) and 79.5% (keyword). With 8 roles, the five most common roles already cover most people, so there is little room for a text method to add.
- **By pool (top 3), the gain is in the within-tech movers:** RRF 69.3% against the baseline's 62.9%; for people who retrained after working, RRF 64.9% against 58.8%.
- **For new grads from another field, the baseline beats every method:** 70.9% against 57.6% (TF-IDF), 57.2% (BGE), 56.9% (RRF) and 44.2% (keyword). A likely reason, not tested: a student's pre-switch text is a degree and a few student jobs, which carries little of the vocabulary the roles' listings use, so the text methods have little to go on and the most common roles win.
- **Steps 5–6 change the picture (development).** The trained layer reaches 83.2% ± 1.8% at top 3 and 95.7% ± 1.3% at top 5, the only method above the baseline at top 5 (87.1%). On the Profiles features alone it reaches 76.6% / 93.7%, so the methods' scores add about 6.6 points at top 3. Skills-based matching is below the baseline: carry-over 46.7% / 75.5% and profile cosine 56.6% / 82.3%.
- **By pool (top 3), the trained layer is above the baseline in every pool:** within tech 84.6% (RRF 69.3%, baseline 62.9%), retrained after working 80.9% (baseline 58.8%) and new grads 77.5% against the baseline's 70.9%, the only method above the baseline for new grads. New grads remain the weakest pool. These figures are slightly optimistic (the settings were chosen on these folds); see 6b.

**What the data shows (test, 563 held-out people)**
- **The ranking of the methods is the same as in development.** At top 3: RRF 70.0% [66.1%, 73.7%], TF-IDF 69.8% [66.1%, 73.4%], BGE 67.0% [63.4%, 70.9%], the baseline 63.2% [59.1%, 67.1%] and keyword overlap 60.9% [57.0%, 64.8%]. RRF is 6.8 points above the baseline, but the intervals overlap a little (they are not paired intervals, so this is a rough reading); BGE's gain over the baseline (3.8 points) is within the intervals.
- **At top 5, the baseline is again the best:** 87.0% [84.2%, 89.7%] against RRF 86.7% [83.7%, 89.3%], TF-IDF 84.7%, BGE 84.4% and keyword 82.2%; at top 5 no method is distinguishable from the baseline.
- **Per pool (top 3):** within tech RRF 73.1% [68.7%, 77.3%] against the baseline's 63.1% [58.2%, 67.7%] (the clearest gain). After working in another field (82 people), RRF 64.6% [54.9%, 74.4%] against 57.3% [46.3%, 68.3%]: the same direction, but the intervals overlap widely. New grads (72 people): the baseline 70.8% [59.7%, 80.6%] is above RRF 58.3% [47.2%, 69.4%], TF-IDF 59.7% and BGE 55.6%.
- **Steps 5–6, test (the fair figure for the fit model):** the trained layer reaches 81.7% [78.5%, 84.7%] at top 3 and 96.3% [94.7%, 97.7%] at top 5, clear of the baseline (63.2% and 87.0%) and of RRF (70.0% and 86.7%) at both, with no interval overlap. With the Profiles features alone it is 75.8% [72.3%, 79.2%] and 94.5% [92.5%, 96.3%]. Skills-based matching is below the baseline: profile cosine 55.6% [51.5%, 59.9%] and carry-over 48.3% [44.4%, 52.6%] at top 3.
- **Per pool (top 3), the trained layer:** within tech 83.4% [79.7%, 86.8%], retrained after working 81.7% [73.2%, 89.0%] (baseline 57.3%) and new grads 72.2% [61.1%, 81.9%] against the baseline's 70.8% [59.7%, 80.6%]. For new grads the intervals overlap, so the trained layer is not shown to be better there; they are 72 people.
- **This is the test result for the project's fit model, and its last run.** The results describe where career changers went, not how likely a switch is to work.

**What the data shows (current-role pull, development, within-tech movers)**
- **If allowed, TF-IDF and RRF would put the person's current role first for 73.7% of within-tech movers, and BGE for 69.3%.** That is the old-field pull: a person's pre-switch text is mostly their current role's titles, so they look most like it. In 6a and 6b this role is removed from the candidates, so the hit rates above are not inflated by it; but it shows the methods rank by "looks like where you are", not "where people like you go".
- **Keyword overlap's 42.9% is lower, but that is not a point in its favor:** it is also the weakest of the text methods at hitting the real next role (54.6% at top 3 in development), so its lower pull comes with weaker matching in general.
- **Skills-based matching does not fix the pull:** carry-over would put the current role first for 69.1% of within-tech movers, and profile cosine for 100.0%, which is by construction (a within-tech person's background is their current role's own skill profile, so it is its closest match).
- **The trained layer has no pull:** 0.1% (0.0% with the Profiles features alone). It learns from real moves that a move is never to the current role, so this is learned from the people, not a sign that it matches skills better; in 6a and 6b the current role is removed from the candidates for every method.

## 7. Why a role is suggested (SHAP)

The trained layer is a logistic regression, so its SHAP values are exact and cheap: for each person, role and input, the coefficient times the person's value minus the average development person's, on the role's log-odds scale (what `shap.LinearExplainer` computes; they add up to the person's score minus the average score). They are computed for the **test people**, with the model fit on the development people (the one scored in 6b), and grouped under the plain-language names the app shows: **prior career** (origin, background area, years and jobs before, starting role and level), **training** (whether and what they trained in, study level), **text similarity** (keyword, TF-IDF, BGE, RRF) and **skills that carry over** (O*NET carry-over and profile).

These explain the model, not the world: an input with a large contribution is one the model leans on, not a cause of where people went.

**What the data shows (test people, the logistic regression)**
- **Text similarity and skills that carry over drive the suggestions.** For each test person's first suggestion, the group that pushes it up most is text similarity for 55.4% of people and skills that carry over for 34.5%; prior career (5.2%) and training (5.0%) lead rarely.
- **Per role, the same two groups are the largest** (mean |SHAP|, log-odds): text similarity from 0.48 (quality assurance) to 0.94 (software development), skills that carry over from 0.22 (database) to 0.74 (quality assurance), against 0.08–0.36 for prior career and 0.09–0.17 for training. Text matters most for software development (0.94), security (0.89) and IT project management (0.81); skills most for quality assurance (0.74), IT support (0.72) and networks and systems (0.66).
- **The inputs the model leans on most are the skills carry-over scores relative to the person's own mean**, for security (0.124), networks and systems (0.120) and software development (0.100). Study level before the switch (0.085) is the only Profiles input among the top 12.
- **This doesn't contradict section 5's finding that the Profiles features alone reach 75.8%.** The method scores are built from the same pre-switch background, so they carry much of the same information, and with both present the model credits the scores. A likely reading, not tested. These values explain the model, not why people moved.

### 7b Growth-adjusted suggestions

Expected growth is applied **after** the fit model, never trained in (it's a projection, so past career moves can't validate it): score × (1 + weight × growth), with the weight on a slider in the app (0 = no adjustment). Always shown with the role's entry-level share and today's openings. Source: `role_growth.pkl` (BLS 2025–2035 projections, rolled up to the role buckets), which waits for PredictLeads' consent to the join (job growth tasks in `ANALYSIS.md`); until then this cell says so and the compass leaves growth blank.

## 8. The compass, assembled

The one place the compass is put together for the app, from tables earlier notebooks saved (aggregates only):
- **Paths** (EDA_SDA_2 3.12b, from Profiles): prior career, field trained in or tech job before the move → the role reached, with people counts (thin under 10), joined to each role's **2026 market** (openings, hourly pay, closing speed, entry-level share; EDA_SDA_2 3.12a) and expected growth when it exists.
- **Skills to build** (the Ideas tab): each role's 10 most asked skills (share of its original listings asking each, required or preferred; EDA_SDA_3 section 6), marked as covered when O*NET lists the skill for the background's occupations (tools, and skills rated 3+ of 5), else to build.
- **Pay against the prior field:** each background's typical hourly pay (its occupations' median pay in the 2026 listings of every field, weighted by listings) against the role's median pay in the career-changer view.

Time to switch isn't in the data (Profiles has years in the old field, not how long the training took), so the compass shows paths and people counts, not durations.

**What the data shows (the compass)**
- **Paths:** 223 paths into the 8 roles. 53 start from a tech job (2,049 people, 43% of paths under 10 people), 109 from a prior career outside tech (764 people, 78% thin) and 61 from a field trained in (764 people, 67% thin). Most career-change paths rest on few people, so the app shows their counts and flags the thin ones.
- **Skills to build:** Python is among the 10 most asked skills for all 8 roles, and first for six (30% of data science listings, 23% of software development and quality assurance). IT support leads with troubleshooting (37%), networks and systems narrowly (16%, Python 15%). IT project management listings rarely name a skill (Python 5%, troubleshooting 4%), so its list is thin.
- **Skills by background:** science and research, and engineering outside software, already show most of each role's top 10 (0–3 to build). Food and hospitality, legal and customer service show the fewest (6–10 of 10 to build). These come from O*NET's lists for each background's occupations, not from a person's own skills.
- **Pay:** every occupation name matched an O*NET code (100% of occupations and listings). A background's typical posted hourly pay runs from $18.78 (food and hospitality) and $21.24 (customer service) to $53.27 (healthcare) and $62.29 (legal). Software development pays more than every background (from +$10.8 an hour for legal to +$54.3 for food and hospitality). IT support pays less than nine of the 18 backgrounds, most for legal (−$31.4) and healthcare (−$22.4). These compare typical posted pay in each field, not a person's own pay.
- **Growth is blank** until `role_growth.pkl` exists (7b), and **time to switch isn't in the data**.

## 9. Solution report and hand-off

**Hand-off to `ML_Rank`:** each career changer's fit score for every role (development people out-of-fold, test people from the model fit on the development people), with their split, pool, current and actual role, and, for the test people, each role's fit broken into the four groups (7a). `ML_Rank`'s backtest (8b) scores listings for test people only.

**Hand-off to the app (9b, the combined model's step 1):** what scoring a new person needs, saved to `data/clean/lean/ml_fit/` with `src/fit_handoff.py`: the trained layer fit on all development people (the fit that scored the test people in 5c), the role side of every method (each role's keyword words, the TF-IDF vectorizer and role centroids, the BGE role centroids, the O*NET skill profiles; aggregates only, no listing or person) and a manifest (roles, methods, the 106 input columns, versions). A parity check rescores the test people from the saved pieces: they must match 5c within 1e-4 and keep the same top 3 roles, or the cell stops.

### Solution report

- **Steps:**
  - 2,813 career changers from the public résumé data (Profiles), split once by person into 2,250 development people (5 folds) and a test set of 563, stratified by role and pool.
  - Each role described by a fixed, hashed sample of 2,000 of its 2026 listings.
  - A ladder of methods, each scored on whether the role people actually moved into is in their top 3 and top 5, against the most common roles: keyword overlap, TF-IDF, BGE, their RRF fusion, skills matching on O*NET, and a trained layer (six candidates tuned on the development folds, 106 inputs).
  - SHAP for the chosen model, the compass tables for the app, and the hand-off to `ML_Rank`.
- **Difficulties:**
  - The text methods beat the baseline only at top 3, and they favor the person's current role (TF-IDF and RRF would put it first for 73.7% of within-tech movers).
  - O*NET's transferable skills barely separate the tech roles, so skills matching alone is below the baseline.
  - The six candidates are within 2 points of each other. A rerun once chose CatBoost instead of logistic regression, because keyword overlap's tied words came out in a different order in every kernel. Fixed (ties break alphabetically, tested), and the model is now chosen by a rule set before looking: the simplest candidate within one standard deviation of the best.
  - Growth data waits on PredictLeads' consent, and time to switch isn't recorded.
- **Final model and score:**
  - Logistic regression (C = 0.03, the edge of its grid, left as is). Test top 3: **81.7% [78.5%, 84.7%]**; top 5: **96.3% [94.7%, 97.7%]**. That compares with the baseline's 63.2% and 87.0% and RRF's 70.0% and 86.7%.
  - Per pool (top 3): within tech 83.4% (baseline 63.1%); retrained after working 81.7% (57.3%); new grads 72.2% [61.1%, 81.9%] against 70.8% [59.7%, 80.6%], not shown to be better.
  - The Profiles features alone reach 75.8%.
- **Leakage audit** (`reports/ML_Fit_audit.md`, `python -m src.leakage_audit`): **8 of 8 checks pass.**
  - No one is in both the development and the test set, and no input is known only after the switch.
  - Rebuilding the saved scores from development people only gives the same values (differences under 1e-9).
  - With the training answers shuffled, the test score falls to 58.2%, below the baseline's 63.2%.
  - 5 test people's résumés are likely saved twice, with a near-copy among the development people. With all 14 near-identical texts left out, the score moves 0.1 points (81.7% to 81.6%).
  - People with undated study score lower (80.6%), not higher (82.1% for the rest).
  - No one's job switched into appears as a job or program they held before.
- **Known limits:**
  - The test set was scored three times (2026-10-13): steps 1–4, then steps 5–6, then after the reproducibility fix. No choice was made from test results.
  - Development figures are slightly optimistic, because the settings were chosen on the same folds that score them.
  - Near-copied résumés survive Profiles' exact de-duplication. Profiles is frozen, so they stay.
  - Role descriptions come from 2026 listings, while the moves happened earlier.
- **Hand-off and what's left:**
  - Saved for `ML_Rank`: `fit_scores.parquet` (2,813 people × 8 roles, development out-of-fold) and `fit_breakdown.parquet` (22,520 rows for the 563 test people).
  - Saved for the app: the compass tables (paths, skills to build, pay, market).
  - Left: growth (BLS, waiting on consent); turning a résumé into the trained layer's inputs in the app (the combined model's step 2; the pieces are saved in 9b); a smaller C to try in a future version, with a fresh test set.
