<!-- Generated from notebooks/ML_Rank.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: ML_Rank (the priority queue)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

Builds the **listing model** and the **priority queue**. The listing model predicts how long each listing stays open (chance still open, apply-by date) from every factor that survived `EDA_SDA_4`, text included, plus the `TimeSeries` timing features. The priority queue then **combines** it with `ML_Fit`'s fit scores, career-change accessibility and pay into one score per listing, for a given career changer. Listings a career changer can't get are already hidden (never removed). Role discovery and the fit model are in `ML_Fit`.

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
- **Ghost listings in training:** does the timing model predict the test month better when trained without `likely_ghost` listings (EDA_SDA_2 3.1d)? A ghost's days open isn't a real hiring window. Keep whichever wins; ghosts stay in the data and stay hidden in the app either way.
- Is the model fast enough to rank listings live in the app?

**Leakage rules** (`STANDARDS.md`):
- Listing features use only what's known when a listing first appears. Company hiring volume and reposts count only listings first seen **before** it; `last_seen_at`, `status` and days open are never features.
- **Date picker:** a model that scores a chosen date is trained only on what was known by then; listings still open that day are censored at that day. The main model is evaluated on the test months, and earlier dates need a model retrained as of that date.
- **Role model:** only `feature` columns from the Profiles modeling table. Compass and path counts used as features are rebuilt inside each training fold, so a test person never counts toward their own answer.
- **Claims:** the resumes show only people who made a switch, so role-model results describe where switchers went, never how likely a switch is to work.

**Data rules:** aggregates only. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard (Intro2ML, SupervisedLearning, NumericalMethods, FinalProject, MLinBusiness) | Taken further |
|:--|:--|:--|
| 1. Setup | Load the modeling-ready table | |
| 2. Targets | Classification: closes within N days. Regression: days open | **Survival** target (days open with censoring) |
| 3. Preprocessing | Drop identifier columns; one-hot encoding with aligned columns; OrdinalEncoder for trees; **StandardScaler fit on training data only** | `Pipeline` + `ColumnTransformer`; rare levels grouped |
| 4. Split | Train / validation / test **once**, `random_state=12345`, test used once at the end | Split **by time** (train April–July, validate August, test September), folds **grouped by company**; fit scores come from `ML_Fit`, trained on its own person-grouped folds |
| 5. Baselines | **DummyClassifier / DummyRegressor**, majority class, random | Newest-first ranking; title keyword search (how a career changer searches today) |
| 6. Class imbalance | `class_weight='balanced'`, upsampling (training only) | Decision-threshold tuning |
| 7. Models | **LogisticRegression, DecisionTree, RandomForest, LightGBM, CatBoost, XGBoost**, GradientBoosting, each tuned with loops or **GridSearchCV + cross-validation**; **`%%time`** for speed | **Cox regression** (with proportional-hazards checks), accelerated failure time models, survival forests, competing-risks models; **learning-to-rank** (LightGBM LambdaRank), which optimizes the order of an applicant's list directly |
| 8. Evaluation | ROC-AUC, F1, precision–recall and average precision, accuracy, **confusion matrix** (interpreted correctly), classification report; RMSE / MAE / R²; training vs. validation to check overfitting | **C-index** and time-dependent Brier score for survival; ranking metrics **precision@k** and **NDCG**; calibration; **conformal prediction** for apply-by ranges with guaranteed coverage |
| 8b. The priority queue | | **Combine** `ML_Fit`'s fit score with this model's chance still open and apply-by date, career-change accessibility, pay and the TimeSeries features in one score: a formula with weights that start from the evidence (EDA_SDA_4's adjusted model) and are tuned on the validation month, with user sliders in the app. Only features that survived EDA_SDA_4 go in |
| 9. Interpretation | **Feature importance** from the final model | **SHAP** values and partial dependence. **Per-listing breakdown for the app's detail view:** each factor's 0–100 rating and its exact contribution to the priority score (rating × weight, adding up to the total), plus each listing's top SHAP features for chance still open, grouped under plain-language names; check that SHAP values add up to each prediction, and save the breakdown with the scores |
| 10. Business translation | Top-N selection by prediction; **bootstrap** confidence intervals and risk (MLinBusiness) | "Of a career changer's top 10, how many are still open and realistically reachable", with company-clustered intervals |
| 11. Sensitivity | | Do conclusions hold under our choices (2-day live tolerance, scope boundaries, CET `posted`)? |
| 12. Solution report | Steps performed, difficulties and how they were solved, the final model and its score | |

## 1c. Hypotheses and decision rule (written before any modeling)

The test month (September) is used once, at the end. **One rule decides H1 to H5 and H7:** the model beats its baseline, and the company-clustered 95% bootstrap interval for the difference (model minus baseline; whole companies resampled together) excludes zero (`rm.clustered_bootstrap_diff` for per-listing or per-person values, `rm.clustered_bootstrap_auc_diff` for ROC-AUC). No fixed cut-off numbers are set; the size of the difference is reported either way. The top-10 overlap test (H6) is secondary.

| # | Hypothesis | Compared with | Supported when |
|:--|:--|:--|:--|
| H1 | Features known at first sighting predict whether a listing closes within N days better than the prior | `DummyClassifier` (prior, AUC 0.5); ROC-AUC difference, company-clustered (AUC, not Brier: the balanced class weights leave the probabilities uncalibrated) | Difference positive, interval excludes 0 |
| H2 | The best model beats the logistic regression | Same, logistic regression as the baseline | Same (otherwise keep the simpler model) |
| H3 | The survival model orders days open better than chance | C-index against 0.5 (the chance baseline), company-clustered bootstrap | Interval for C-index minus 0.5 excludes 0 |
| H4 | The priority queue beats newest first | Share of each career changer's top 10 that is relevant (below), test month | Difference positive, interval excludes 0 |
| H5 | The queue beats title keyword search | Same | Same |
| H6 (secondary) | Adding accessibility and fit changes the top 10 versus timing alone | Overlap of the two top 10s (`rm.top_k_overlap`) | Overlap below 80% |
| H7 | Training without `likely_ghost` listings predicts the test month better | Same rule, trained with ghosts as the baseline | Same |

**N, the closing window.** One constant, `rm.CLOSING_DAYS`, drives the main target ("closes within N days"), "open long enough" (still open N days after the person first sees the listing) and every hypothesis. It is 14 for now (TimeSeries: 37-45% close within 14 days); 7, 14 and 30 days are repeated as a sensitivity check in section 11 (`rm.SENSITIVITY_DAYS`). Change it in `src/rank_model.py` only.

**Backtest relevance (H4, H5).** For each career changer, a listing in their top 10 is relevant if it is (1) still open long enough, (2) reachable, and (3) in the role that person actually moved into (`rm.backtest_relevance`).

## 2. Targets

Closes within N days (unknown, not "stayed open", when still open and seen for fewer than N days), days open (closed listings only), and the survival target (days open with censoring, so still-open listings are kept).

**What the data shows**
- (fill in after the run)

## 3. Preprocessing and leakage checks

Features come only from what is known at first sighting. Company counts use earlier listings only (`prior_company_counts`); `has_desc` and the text flags are checked below (EDA_SDA_2 3.9) before use. Preprocessing sits inside a `Pipeline` + `ColumnTransformer`, fit on training data only.

**What the data shows**
- (fill in after the run)

### 3b Date picker check: July 1 vs. September 1, 2026

`src/as_of.py` returns the listings live on a date (first seen on or before it, last seen on or after it) with features from earlier listings only, and the training rows as known that day: a listing last seen more than `LIVE_TOLERANCE_DAYS` (2) before the date is finished; the rest are censored at the date. Counts only; no rows are shown. EDA_SDA_1 8b shows dates before July are skewed young, so the demo starts with later dates.

**What the data shows**
- (fill in after the run)

Decision to record: which dates the app's picker offers (test-period dates first; earlier dates need a model retrained as of that date).

### 3c Preprocessing pipeline

`ColumnTransformer`: median for blank numbers (with a was-blank flag), "unknown" for blank categories, one-hot with levels under 50 listings grouped as "infrequent" and unseen levels allowed. Scaling sits inside the logistic regression only. The transformer is fit inside each training fold by the `Pipeline`, never on all rows.

**What the data shows**
- (fill in after the run)

## 4. Split

Train April–July (plus the data's first two days), validate August, test September. Folds grouped by company. The test month is touched once, in section 8.

**What the data shows**
- (fill in after the run)

**What the data shows**
- (fill in after the run)

## 5. Baselines

`DummyClassifier`, newest first, and title keyword search (the career changer's keywords come from `ML_Fit`'s role queries).

**What the data shows**
- (fill in after the run)

## 6. Class imbalance

Three ways to handle the imbalance between closes-within-N and stays-open, compared on the validation month with the logistic regression: no handling, `class_weight="balanced"` (what `rm.build_models()` sets), and upsampling the minority class (training rows only). Then the decision threshold is tuned on the validation month and applied unchanged to the test month.

**What the data shows**
- (fill in after the run)

**What the data shows**
- (fill in after the run)

## 7. Models

Logistic regression, decision tree, random forest, LightGBM, CatBoost, XGBoost (`rm.build_models()`), each with GridSearchCV over the company-grouped folds and `%%time`; Cox and survival models (`rm.cox_model()`, proportional-hazards check, company-clustered errors); LightGBM LambdaRank (`rm.lambdarank_model()`).

**What the data shows**
- (fill in after the run)

### 7b Grid search

`GridSearchCV` over the company-grouped folds, one grid per model (`rm.PARAM_GRIDS`), timed with `%%time`. The settings chosen here are fixed before the validation month is scored.

**What the data shows**
- (fill in after the run)

### 7c Cox proportional hazards (company-clustered errors) with the proportional-hazards check

The survival target keeps still-open listings (censored). A small p-value in the check means a feature's effect changes over a listing's life, so its hazard ratio is not one constant number. These estimates feed `START_WEIGHTS` for the queue.

**What the data shows**
- (fill in after the run)

### 7d Accelerated failure time models

Weibull, log-normal and log-logistic AFT models say how much a feature stretches or shrinks the time open (a ratio on days), without the proportional-hazards assumption. Compared on C-index and AIC.

**What the data shows**
- (fill in after the run)

### 7e Random survival forest

Non-linear effects and interactions without assuming a hazard shape. Fit on a random sample of training listings to keep memory in check; scored by C-index on the validation month.

**What the data shows**
- (fill in after the run)

### 7f LambdaRank

Learning to rank optimizes the order of a list directly. A "query" is one list that gets ranked; relevance is graded (0 = closed within N days, 1 = stayed open). Until `ML_Fit`'s career-changer lists are wired in, the query here is the day a listing first appeared (rank that day's new listings), as a stand-in; `rm.backtest_relevance` defines the final relevance.

**What the data shows**
- (fill in after the run)

## 8. Evaluation

ROC-AUC, F1, precision-recall, confusion matrix and classification report; C-index for survival; precision@k and NDCG for rankings. Each hypothesis is judged by the 1c rule (beats the baseline, company-clustered interval excludes zero). The test month is used once, with the settings and threshold fixed on validation.

**What the data shows**
- (fill in after the run)

**What the data shows**
- (fill in after the run)

### 8c Test month (run once)

Set `RUN_TEST = True` only after every choice above is final. Same metrics as 8a with the validation threshold, no further tuning.

**What the data shows**
- (fill in after the run)

## 8b. The priority queue (H4, H5, H6)

Fit score, chance still open and apply-by, accessibility and pay in one formula (`rm.priority_breakdown`). `rm.START_WEIGHTS` are starting values for the four factors, tuned on August with `rm.tune_weights`. EDA_SDA_4's Cox hazard ratios describe closing speed, so they do not set these four weights (they enter through the listing model's chance still open); only features that survived EDA_SDA_4 are used. The queue against newest first and title keywords, per career changer: of the top 10, how many are relevant, meaning still open long enough, reachable, and in the role the person actually moved into (`rm.backtest_relevance`).

_Needs `ML_Fit`'s fit scores and the career changers' moves; the cell below shows the evaluation once they exist._

**What the data shows**
- (fill in after the run)

## 9. Interpretation

SHAP and partial dependence; the per-listing breakdown for the detail view (each factor's rating and share of the score from `priority_breakdown`, top SHAP features behind chance still open under plain-language names; check SHAP adds up to each prediction); saved with the scores.

_To do._

## 10. Business translation

Top-10 results with company-clustered bootstrap intervals.

_To do._

## 11. Sensitivity

The 7, 14 and 30 day windows (`rm.SENSITIVITY_DAYS`) for the closing target, the live tolerance (`rm.LIVE_TOLERANCE_DAYS`), scope boundaries, CET `posted`, with and without ghost listings (H7).

**What the data shows**
- (fill in after the run)

_Still to do: live tolerance, scope boundaries, CET `posted`, and H7 (ghosts)._

## 12. Solution report

Steps performed, difficulties and how they were solved, the final model and its score.

_To do._
