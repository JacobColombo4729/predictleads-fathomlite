<!-- Generated from notebooks/ML_Rank.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: ML_Rank (the priority queue)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

Builds the **listing model** and the **priority queue**. The listing model predicts how long each listing stays open (chance still open, apply-by date) from every feature that `EDA_SDA_4`'s screening rule kept, text included (2a reads them from `modeling_ready.parquet`, so this notebook holds no list of them), plus the `TimeSeries` timing features. The priority queue then **combines** it with `ML_Fit`'s fit scores, career-change accessibility and pay into one score per listing, for a given career changer. Listings a career changer can't get are already hidden (never removed). Role discovery and the fit model are in `ML_Fit`.

**There's no single "priority" label to train on,** so the two models answer separate questions (will it stay open? does it fit this person?) and a formula with evidence-based, tunable weights combines them. Every score shows its parts.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Text features, text rules as labels and opt-in filters (keyword rules, not validated), skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### Clarifying questions
- How well can we predict whether a listing closes within N days, and how long it stays open?
- Which features drive the prediction most?
- **Does the priority queue beat how a career changer searches today** (newest first, title keyword search)? Measured on the test month: of each career changer's top 10, how many are still open long enough to apply, realistically reachable, and in the role they actually moved into (from `ML_Fit`).
- Does adding career-change accessibility and fit change a career changer's top 10, compared with timing factors alone?
- How many of a career changer's top 10 listings are still open when they apply, and realistically reachable?
- **Ghost listings in training** (H7, frozen for the pitch): would the timing model predict the test month better when trained without `likely_ghost` listings (EDA_SDA_2 3.1d)? A ghost's days open isn't a real hiring window. Not tested; ghosts stay in the data and stay hidden in the app either way.
- Is the model fast enough to rank listings live in the app?

**Leakage rules** (`PROJECT.md`, How we keep it honest):
- Listing features use only what's known when a listing first appears. Company hiring volume and reposts count only listings first seen **before** it; `last_seen_at`, `status` and days open are never features.
- **Date picker:** a model that scores a chosen date is trained only on what was known by then; listings still open that day are censored at that day. The main model is evaluated on the test months, and earlier dates need a model retrained as of that date.
- **Role model:** only `feature` columns from the Profiles modeling table. Compass and path counts used as features are rebuilt inside each training fold, so a test person never counts toward their own answer.
- **Claims:** the resumes show only people who made a switch, so role-model results describe where switchers went, never how likely a switch is to work.

**Data rules:** aggregates only. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard (Intro2ML, SupervisedLearning, NumericalMethods, FinalProject, MLinBusiness) | Taken further |
|:--|:--|:--|
| 1. Setup | Load the modeling-ready table | The features are whatever columns `EDA_SDA_4` kept; none is named here |
| 2. Targets | Classification: closes within N days. Regression: days open | **Survival** target (days open with censoring) |
| 3. Preprocessing | Drop identifier columns; one-hot encoding with aligned columns; OrdinalEncoder for trees; **StandardScaler fit on training data only** | `Pipeline` + `ColumnTransformer`; rare levels grouped |
| 4. Split | Train / validation / test **once**, `random_state=12345`, test used once at the end | Split **by time** (train April–July, validate August, test September), folds **grouped by company**; fit scores come from `ML_Fit`, trained on its own person-grouped folds |
| 5. Baselines | **DummyClassifier / DummyRegressor**, majority class, random | Newest-first ranking; title keyword search (how a career changer searches today) |
| 6. Class imbalance | `class_weight='balanced'`, upsampling (training only) | Decision-threshold tuning |
| 7. Models | **LogisticRegression, DecisionTree, RandomForest, LightGBM, CatBoost, XGBoost**, GradientBoosting, each tuned with loops or **GridSearchCV + cross-validation**; **`%%time`** for speed | **Cox regression** (with proportional-hazards checks), accelerated failure time models, survival forests, competing-risks models; **learning-to-rank** (LightGBM LambdaRank), which optimizes the order of an applicant's list directly |
| 8. Evaluation | ROC-AUC, F1, precision–recall and average precision, accuracy, **confusion matrix** (interpreted correctly), classification report; RMSE / MAE / R²; training vs. validation to check overfitting | **C-index** and time-dependent Brier score for survival; ranking metrics **precision@k** and **NDCG**; calibration; **conformal prediction** for apply-by ranges with guaranteed coverage |
| 8d. The priority queue | | **Combine** `ML_Fit`'s fit score with this model's chance still open and apply-by date, career-change accessibility, pay and the TimeSeries features in one score: a formula with weights that start from the evidence (EDA_SDA_4's adjusted model) and are tuned on the validation month, with user sliders in the app. Only features that survived EDA_SDA_4 go in |
| 9. Interpretation | **Feature importance** from the final model | **SHAP** values and partial dependence. **Per-listing breakdown for the app's detail view:** each factor's 0–100 rating and its exact contribution to the priority score (rating × weight, adding up to the total), plus each listing's top SHAP features for chance still open, grouped under plain-language names; check that SHAP values add up to each prediction, and save the breakdown with the scores |
| 10. Business translation | Top-N selection by prediction; **bootstrap** confidence intervals and risk (MLinBusiness) | "Of a career changer's top 10, how many are still open and realistically reachable", with company-clustered intervals |
| 11. Sensitivity | | Do conclusions hold under our choices (2-day live tolerance, scope boundaries, CET `posted`)? |
| 12. Solution report | Steps performed, difficulties and how they were solved, the final model and its score | |

## Contents

- [0. Purpose](#0-purpose)
  - [Clarifying questions](#clarifying-questions)
  - [Roadmap and methods](#roadmap-and-methods)
- [1c. Hypotheses and decision rule (written before any modeling)](#1c-hypotheses-and-decision-rule-written-before-any-modeling)
- [2. Targets](#2-targets)
- [3. Preprocessing and leakage checks](#3-preprocessing-and-leakage-checks)
  - [3b Date picker check: July 1 vs. September 1, 2026](#3b-date-picker-check-july-1-vs-september-1-2026)
  - [3c Preprocessing pipeline](#3c-preprocessing-pipeline)
- [4. Split](#4-split)
- [5. Baselines](#5-baselines)
- [6. Class imbalance](#6-class-imbalance)
- [7. Models](#7-models)
  - [7b Grid search](#7b-grid-search)
  - [7c Cox proportional hazards (company-clustered errors) with the proportional-hazards check](#7c-cox-proportional-hazards-company-clustered-errors-with-the-proportional-hazards-check)
  - [7d Accelerated failure time models](#7d-accelerated-failure-time-models)
  - [7e Random survival forest](#7e-random-survival-forest)
  - [7f LambdaRank](#7f-lambdarank)
- [8. Evaluation](#8-evaluation)
  - [8c Test month (run once)](#8c-test-month-run-once)
  - [8d The priority queue (H4, H5, H6)](#8d-the-priority-queue-h4-h5-h6)
  - [8e The priority queue on the test month (run once)](#8e-the-priority-queue-on-the-test-month-run-once)
- [9. Interpretation](#9-interpretation)
- [10. Business translation](#10-business-translation)
- [11. Sensitivity](#11-sensitivity)
- [12. Solution report](#12-solution-report)
- [13. Hand-off to the app](#13-hand-off-to-the-app)

## 1c. Hypotheses and decision rule (written before any modeling)

The test month (September) is used once, at the end. **One rule decides H1 to H5:** the model beats its baseline, and the company-clustered 95% bootstrap interval for the difference (model minus baseline; whole companies resampled together) excludes zero (`rm.clustered_bootstrap_diff` for per-listing or per-person values, `rm.clustered_bootstrap_auc_diff` for ROC-AUC). No fixed cut-off numbers are set; the size of the difference is reported either way. The top-10 overlap test (H6) is secondary; H7 is frozen for the pitch (`ANALYSIS.md`), not tested.

| # | Hypothesis | Compared with | Supported when |
|:--|:--|:--|:--|
| H1 | Features known at first sighting predict whether a listing closes within N days better than the prior | `DummyClassifier` (prior, AUC 0.5); ROC-AUC difference, company-clustered (AUC, not Brier: the balanced class weights leave the probabilities uncalibrated) | Difference positive, interval excludes 0 |
| H2 | The best model beats the logistic regression | Same, logistic regression as the baseline | Same (otherwise keep the simpler model) |
| H3 | The survival model orders days open better than chance | C-index against 0.5 (the chance baseline), company-clustered bootstrap | Interval for C-index minus 0.5 excludes 0 |
| H4 | The priority queue beats newest first | Share of each career changer's top 10 that is relevant (below), test month | Difference positive, interval excludes 0 |
| H5 | The queue beats title keyword search | Same | Same |
| H6 (secondary) | Adding accessibility and fit changes the top 10 versus timing alone | Overlap of the two top 10s (`rm.top_k_overlap`) | Overlap below 80% |
| H7 | Training without `likely_ghost` listings predicts the test month better | Same rule, trained with ghosts as the baseline | **Frozen for the pitch, not tested** |

**N, the closing window.** One constant, `rm.CLOSING_DAYS`, drives the main target ("closes within N days"), "open long enough" (still open N days after the person first sees the listing) and every hypothesis. It is 14 for now (TimeSeries: 37-45% close within 14 days); 7, 14 and 30 days are repeated as a sensitivity check in section 11 (`rm.SENSITIVITY_DAYS`). Change it in `src/rank_model.py` only.

**Backtest relevance (H4, H5).** For each career changer, a listing in their top 10 is relevant if it is (1) still open long enough and not a likely ghost, (2) reachable, and (3) in the role that person actually moved into (`rm.precision_per_person`). H4 and H5 compare values per person, so the bootstrap resamples whole people.

## 2. Targets

Closes within N days (unknown, not "stayed open", when still open and seen for fewer than N days), days open (closed listings only), and the survival target (days open with censoring, so still-open listings are kept).

**What the data shows**
- **908,291 listings, 45 numeric and 13 categorical features** from EDA_SDA_4; none is named in this notebook.
- **23.7% are still open at the end of the data (censored).** They stay in the survival target and drop out of the 14-day target only when they were watched for fewer than 14 days.
- **39.2% of the listings with a known answer close within 14 days**, so the classes are uneven but not rare (section 6).

## 3. Preprocessing and leakage checks

Features come only from what is known at first sighting. Company counts use earlier listings only (`prior_company_counts`); `has_desc` and the text flags are checked below (EDA_SDA_2 3.9) before use. Preprocessing sits inside a `Pipeline` + `ColumnTransformer`, fit on training data only.

**What the data shows**
- **Company volume (`company_prior_listings`) counts only listings first seen strictly before each one**, so a company's later listings never feed an earlier one's features.
- **Every description feature is blank, not False or 0, when a listing has no description: 0 of the 15 kept features is set without one** (17.6% of listings have no description). A model can't read "no description" out of them; that signal comes only from `has_desc` itself.

### 3b Date picker check: July 1 vs. September 1, 2026

`src/as_of.py` returns the listings live on a date (first seen on or before it, last seen on or after it) with features from earlier listings only, and the training rows as known that day: a listing last seen more than `LIVE_TOLERANCE_DAYS` (2) before the date is finished; the rest are censored at the date. Counts only; no rows are shown. EDA_SDA_1 8b shows dates before July are skewed young, so the demo starts with later dates.

**What the data shows**
- **September 1 has more of everything than July 1:** 204,536 live listings vs. 166,183 (+23%), from 36,031 companies vs. 30,740, and a median live listing aged 35.7 days vs. 29.4.
- **A model as of September 1 trains on 752,572 rows, 28.4% of them censored at the date; as of July 1, on 463,387 rows, 38.0% censored.** The earlier date has fewer and more often unfinished answers, one more reason the picker starts with test-period dates.

Decision to record: which dates the app's picker offers (test-period dates first; earlier dates need a model retrained as of that date).

### 3c Preprocessing pipeline

`ColumnTransformer`: median for blank numbers (with a was-blank flag), "unknown" for blank categories, one-hot with levels under 50 listings grouped as "infrequent" and unseen levels allowed. Scaling sits inside the logistic regression only. The transformer is fit inside each training fold by the `Pipeline`, never on all rows.

**What the data shows**
- **One preprocessor for every model:** the 45 numeric features get the median for blanks plus a was-blank flag; the 13 categorical ones get "unknown" for blanks and one-hot columns, with levels under 50 listings grouped as "infrequent". It shows "Not fitted" here because it is only ever fit inside a training fold.

## 4. Split

Train April–July (plus the data's first two days), validate August, test September. Folds grouped by company. The test month is touched once, in section 8.

**What the data shows**
- **Train 600,114 listings (66%), validation 146,204 (16%), test 161,973 (18%)**, split by the month each was first seen; together the full 908,291.

**What the data shows**
- **Every training and validation listing has a known 14-day answer** (600,114 and 146,204, the same counts as 4a): all were first seen at least a month before the data ends.
- **The closing share rises from 36.3% (April–July) to 40.8% (August).** The validation month closes listings faster than the training months; the models are compared on ranking (AUC), which this shift affects less than a fixed threshold.

## 5. Baselines

`DummyClassifier`, newest first, and title keyword search (the career changer's keywords come from `ML_Fit`'s role queries).

**What the data shows**
- **Newest first barely ranks which listings close within 14 days: AUC 0.509**, next to the prior's 0.500. How recently a listing appeared says almost nothing about whether it closes soon, so recency alone is a weak guide for a career changer.

## 6. Class imbalance

Three ways to handle the imbalance between closes-within-N and stays-open, compared on the validation month with the logistic regression: no handling, `class_weight="balanced"` (what `rm.build_models()` sets), and upsampling the minority class (training rows only). Then the decision threshold is tuned on the validation month and applied unchanged to the test month.

**What the data shows**
- **Imbalance handling doesn't change the ranking: AUC 0.808 with none, 0.809 with balanced weights or upsampling.** It raises F1 at 0.5 from 0.647 to 0.681, by moving more listings above the cut-off.
- **Balanced weights and upsampling give the same result**, so the models keep `class_weight="balanced"` (no extra rows to fit).

**What the data shows**
- **The best-F1 threshold is 0.42: F1 0.686, vs. 0.681 at 0.5**, a small gain.
- **This shows the method on the logistic regression.** 8a chooses the chosen model's own threshold the same way, on its validation probabilities, and 8c applies it unchanged to the test month.

## 7. Models

Logistic regression, decision tree, random forest, LightGBM, CatBoost, XGBoost (`rm.build_models()`), each with GridSearchCV over the company-grouped folds and `%%time`; Cox and survival models (`rm.cox_model()`, proportional-hazards check, company-clustered errors); LightGBM LambdaRank (`rm.lambdarank_model()`).

**What the data shows**
- **The boosted models lead with their default settings: LightGBM 0.797 ± 0.021, XGBoost 0.797 ± 0.020, random forest 0.793 ± 0.017, CatBoost 0.792 ± 0.020**, against logistic regression 0.746 ± 0.028, a decision tree 0.744 ± 0.018 and the dummy's 0.500 (company-grouped five-fold ROC-AUC).
- **The four ensembles are within one standard deviation of each other**; the clear gap is between them and the two simple models (about 0.05).

### 7b Grid search

`GridSearchCV` over the company-grouped folds, one grid per model (`rm.PARAM_GRIDS`), timed with `%%time`. The settings chosen here are fixed before the validation month is scored.

**What the data shows**
- **Tuning moved every model by at most 0.007:** LightGBM and XGBoost 0.801 (± 0.020 and 0.021), CatBoost 0.798, random forest 0.795, the decision tree 0.751 (up from 0.744 with depth 8 and at least 100 listings per leaf), logistic regression 0.746 (C = 0.1).
- **LightGBM and XGBoost tie at three decimals;** XGBoost is chosen in 8a by the higher unrounded score. Either would do.
- **The grid took 40 minutes** (10 hours 42 minutes of CPU time across cores).

### 7c Cox proportional hazards (company-clustered errors) with the proportional-hazards check

The survival target keeps still-open listings (censored). A small p-value in the check means a feature's effect changes over a listing's life, so its hazard ratio is not one constant number. These estimates feed `START_WEIGHTS` for the queue.

**What the data shows**
- **26 of the 45 numeric features have a hazard ratio significantly different from 1 (p < 0.05)** in the training months, with company-clustered errors.
- **Faster closing:** a recruiter on the listing (hazard ratio 1.52), a repost (1.17), career-changer wording (1.16), each extra category (1.12), remote in the title (1.11), entry wording (1.08).
- **Slower closing:** listings first seen in a crawl batch (0.45; a crawl artifact, EDA_SDA_1 6.8: they were already open for an unknown time when the company was added), a description (0.60), the employer's own site (0.88), and each day of the company's past closing time (0.98 per day).
- **The proportional-hazards check fails for 18 of the 45 (p < 0.05)**, among them the strongest: the company's past closing time, the crawl batch, the description and its length. Their effect changes over a listing's life, so each single hazard ratio is an average, not a constant. With about 600,000 rows the test flags even small changes.
- Count features (company volume, market counts) show 1.000 per unit; their effect per listing is too small to read at three decimals.

### 7d Accelerated failure time models

Weibull, log-normal and log-logistic AFT models say how much a feature stretches or shrinks the time open (a ratio on days), without the proportional-hazards assumption. Compared on C-index and AIC.

**What the data shows**
- **The log-logistic AFT fits best (AIC 4,872,898, vs. 4,881,204 for Weibull and 4,882,823 for log-normal)**; all three order listings about equally (C-index 0.647–0.648).
- These C-indexes are on the training months, the forest's (7e) on validation, so the two are not a like-for-like comparison.

### 7e Random survival forest

Non-linear effects and interactions without assuming a hazard shape. Fit on a random sample of 20,000 training listings: its fit time grows faster than the square of the rows when durations differ (about 13 minutes at 20,000, an estimated 10 hours at 100,000; decided 2026-10-14). Scored by C-index on the validation month.

**What the data shows**
- **The survival forest orders the validation month's listings with C-index 0.725**, well above chance (0.5) and the AFT models' 0.648 on their own training data (7d); 8b puts an interval on it.
- **The 20,000-row fit took 1 minute 28 seconds on this machine** (26 minutes 39 seconds of CPU time across cores), well under the 13 minutes expected when the sample size was chosen, so a larger sample is affordable here.

### 7f LambdaRank

Learning to rank optimizes the order of a list directly. A "query" is one list that gets ranked; relevance is graded (0 = closed within N days, 1 = stayed open). Until `ML_Fit`'s career-changer lists are wired in, the query here is the day a listing first appeared (rank that day's new listings), as a stand-in; `rm.backtest_relevance` defines the final relevance. A day with more than 10,000 new listings is split at random into parts, LightGBM's limit per query.

**What the data shows**
- **LambdaRank's mean NDCG@10 over 31 validation days is 0.952.** Relevance here is "stayed open" (59% of August listings), so most lists are full of relevant listings and even a random order would score high. No baseline NDCG is computed, so this number shows the pipeline runs, not that LambdaRank adds anything; the career-changer lists (8d) are the real test.
- The warning about feature names is harmless: the ranker was fit on a table and scored on an array with the same columns.
- **Tested, not used:** the priority queue (8d) ranks by the chosen classifier's chance still open, not this ranker.

## 8. Evaluation

ROC-AUC, F1, precision-recall, confusion matrix and classification report; C-index for survival; precision@k and NDCG for rankings. Each hypothesis is judged by the 1c rule (beats the baseline, company-clustered interval excludes zero). The test month is used once, with the settings and threshold fixed on validation.

**What the data shows**
- **The chosen model is XGBoost** (best grid-search score, 7b): validation AUC 0.839, average precision 0.821 against a 40.8% base rate.
- **Its threshold is 0.35, chosen on its own validation probabilities** (best F1 0.721, vs. 0.701 at 0.5). At 0.35 it finds 72% of the listings that close within 14 days (43,205 of 59,689), and 72% of the listings it flags do close; 16,484 fast closers are missed and 16,896 flagged listings stay open.
- **H1 supported:** AUC 0.339 above the prior, company-clustered 95% interval 0.319 to 0.360 (21,975 companies).
- **H2 supported:** AUC 0.031 above the logistic regression, interval 0.023 to 0.039. The gain is small but clear, so the boosted model is kept.
- **Validation AUCs are higher than cross-validation's for every model** (XGBoost 0.839 vs. 0.801, logistic 0.808 vs. 0.746): August is easier to rank than the training months' company-grouped folds, so 8c's test month is the number to quote.

**What the data shows**
- **H3 supported:** the survival forest's C-index on the validation month is 0.725, company-clustered 95% interval 0.708 to 0.741 (100 resamples of companies), entirely above chance (0.5).

### 8c Test month (run once)

Set `RUN_TEST = True` only after every choice above is final. Same metrics as 8a with the validation threshold, no further tuning. **Only September listings whose N-day answer is known either way** (first seen more than N + 2 days before the data ends, `rm.answer_known_either_way`): a later listing has a known answer only if it closed fast, since one still open hasn't been watched N days yet, so keeping them would fill the end of the month with early closers. The first test run kept them; 8c and 8e were rerun with this cutoff (2026-10-14), with the model, threshold and weights unchanged. 8c ran once more after the threshold was chosen on the chosen model (8a) instead of the logistic regression (6b).

**What the data shows**
- **The listing model holds on the test month: AUC 0.844 and F1 0.733 at the validation threshold (0.35)**, on the 66,844 September listings first seen before September 14 (validation: 0.839 and 0.721).
- **Why the cutoff:** the first test run kept the later September listings, whose answers are known only when they closed fast, and gave AUC 0.831 and F1 0.750. The model, threshold and weights were not changed for the rerun.
- **Then the threshold moved from the logistic regression's 0.42 to XGBoost's own 0.35** (chosen on August, 8a) and 8c ran once more: F1 0.732 became 0.733; AUC doesn't depend on the threshold.

### 8d The priority queue (H4, H5, H6)

Fit score, chance still open, accessibility and pay in one score (`rm.queue_top_k`, the same sum as `rm.priority_breakdown`): each factor rated 0–100, times its weight. EDA_SDA_4's Cox hazard ratios describe closing speed, so they do not set these four weights (they enter through the listing model's chance still open).

**The backtest.** One month's listings that a career changer sees (the 8 fit roles; not a hidden occupation; not manager level or above) with an N-day answer known either way (first seen more than N + 2 days before the data ends; see 8c), ranked for each career changer from `ML_Fit` (`fit_scores.parquet`):
- **Fit:** the person's `ML_Fit` probability for the listing's role, rated so their best role is 100.
- **Chance still open:** 1 minus the chosen listing model's chance of closing within N days (8a).
- **Accessibility:** how many of the career-change signs a listing shows (entry seniority, 2 years or fewer asked, entry wording, career-changer wording; EDA_SDA_2 3.11).
- **Pay:** hourly pay, rated within its role; missing pay is neutral (50).
- **Relevant:** stayed open N days, not a likely ghost, and in the role the person actually moved into.

**Leakage:** the weights are tuned on August with the **development people**, whose fit scores are out-of-fold; September is scored once, with the **test people**, whose fit scores come from a model trained without them. Relevance uses outcomes (closing, ghost), never as an input.

**Baselines:** newest first (the same list for everyone), and a title search for the most common role among the development people (ML_Fit's baseline) sorted newest first. The search is a perfect one (every listing in that role and nothing else), so it flatters the baseline.

**What the data shows**
- **The August pool: 115,170 listings a career changer would see, 58.8% of them open long enough and not a ghost; 2,250 development people.**
- **Tuned weights: fit 0.3, chance still open 0.2, accessibility 0.1, pay 0** (start: 0.4, 0.3, 0.2, 0.1). Precision@10 0.469 tuned vs. 0.449 with the start weights; the tuned row is optimistic, since August is where its weights were chosen.
- **Pay gets weight 0 because the relevance definition can't reward it:** relevance is open long enough, not a ghost, and in the role the person moved into, none of which pay predicts. Pay stays a user slider in the app.
- **The queue's lead comes from the role:** 47.6% of its top 10 is in the role the person moved into, against 16–24% for the other lists, while 98.6% is open long enough. Chance still open alone gets every listing open long enough (100%) but only 16.3% in the right role.
- **On August, newest first (0.154) and the title search (0.166) score about the same as chance still open alone (0.163).**

### 8e The priority queue on the test month (run once)

The weights from 8d, the test people, September's listings. H4 and H5: the tuned queue against newest first and the title search, per person, with a 95% bootstrap interval over people (`rm.clustered_bootstrap_diff`, one cluster per person). H6: how much of the queue's top 10 is also in the top 10 by chance still open alone (`rm.mean_overlap`; below 0.8 = they differ). Runs with `RUN_TEST` (8c).

**What the data shows**
- **On the test month (53,046 September listings first seen before September 14, 563 test people), half of the queue's top 10 is relevant: precision@10 0.500**, with 99.3% open long enough and 50.4% in the role the person moved into. The start weights give 0.486; chance still open alone 0.210.
- **H4 supported:** the queue beats newest first by 0.469 per person (95% interval 0.430 to 0.509; newest first 0.031).
- **H5 supported:** the queue beats a perfect title search for networks and systems by 0.428 (0.390 to 0.466; title search 0.071).
- **H6 supported:** the queue's top 10 shares 0% with the top 10 by chance still open alone, which ignores the person and gives everyone the same list from any role.
- **Caveat on the size of H4 and H5:** newest first and the title search give every person the same 10 listings, so their "open long enough" share rests on those 10 (20% and 30% in September, 90% and 70% in August). The interval resamples people, not those 10 listings, so it understates the uncertainty in the baselines. The robust part is the role: 50.4% of the queue's top 10 is in the role the person moved into, vs. 23.8% for the best possible title search.

## 9. Interpretation

SHAP and partial dependence; the per-listing breakdown for the detail view (each factor's rating and share of the score from `priority_breakdown`, top SHAP features behind chance still open under plain-language names; check SHAP adds up to each prediction); saved with the scores.

**Frozen for the pitch** (`ANALYSIS.md`): not built.

## 10. Business translation

Top-10 results with company-clustered bootstrap intervals.

**Frozen for the pitch** (`ANALYSIS.md`): not built.

## 11. Sensitivity

The 7, 14 and 30 day windows (`rm.SENSITIVITY_DAYS`) for the closing target, the live tolerance (`rm.LIVE_TOLERANCE_DAYS`), scope boundaries, CET `posted`, with and without ghost listings (H7).

**What the data shows**
- **The listing model's ranking holds across closing windows: cross-validated AUC 0.803 ± 0.023 at 7 days, 0.801 ± 0.021 at 14, 0.790 ± 0.013 at 30.**
- **The share closing grows with the window (29.4%, 39.2%, 59.3%) while the known answers shrink (867,140, 843,734, 804,571)**, since a longer window needs a longer watch. 14 days keeps the main target near balance with little loss of answers.

**Frozen for the pitch** (`ANALYSIS.md`): live tolerance, scope boundaries, CET `posted`, and H7 (ghosts) not tested.

## 12. Solution report

Steps performed, difficulties and how they were solved, the final model and its score.

**Frozen for the pitch** (`ANALYSIS.md`): not built.

## 13. Hand-off to the app

A snapshot the app can serve before live scoring exists: every scope listing's factors, scored by the chosen listing model (8a), plus the model itself and the queue's settings. Saved to `data/clean/ml_rank/`:
- `listing_scores.parquet`: one row per listing: `id`, role, first seen, split, chance of closing within N days and chance still open, accessibility (0-4 signs), hourly pay, level, and the flags the app hides by (`hidden_occupation`, `likely_ghost`, manager levels) plus `open_at_end` (still open when the data ends).
- `listing_model.joblib`: the fitted pipeline (preprocessor + model), for scoring new listings once the app computes the same features.
- `queue_config.json`: the tuned weights, the threshold, N, the feature lists, the fit roles and the levels hidden by default.

**How the app ranks with it:** the listing ratings come from `rm.listing_ratings` over the listings shown (pay rated within role), the fit rating from `ML_Fit` (`rm.fit_ratings`), and the score is the weighted sum (`rm.queue_top_k`, `rm.priority_breakdown` for the detail view). Chances for the training months (April-July) are in-sample; the `split` column says which listings those are.
