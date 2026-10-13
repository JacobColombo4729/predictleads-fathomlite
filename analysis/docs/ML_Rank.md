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

## 1c. Hypotheses and thresholds (written before any modeling)

Each hypothesis has a number that decides it. The test month (September) is used once, at the end; thresholds are fixed here and not changed after seeing results.

| # | Hypothesis | Decided by | Threshold to count as supported |
|:--|:--|:--|:--|
| H1 | Features known at first sighting predict whether a listing closes within N days better than the prior | Validation ROC-AUC, CV mean ± std over company-grouped folds, vs. `DummyClassifier` (0.5) | Mean AUC at least 0.60 and above the dummy by more than 2 std |
| H2 | The best model beats the logistic regression | Validation ROC-AUC difference | At least +0.01 (otherwise keep the simpler model) |
| H3 | Survival model orders days open better than chance | C-index on validation (company-clustered bootstrap 95% CI) | CI lower bound above 0.50; reported against the best classifier's implied order |
| H4 | The priority queue beats newest-first | Share of each top 10 still open 14+ days (precision@10), test month | Higher by at least 5 points with a company-clustered 95% CI excluding 0 |
| H5 | The queue beats title keyword search | Same, plus the share of the top 10 in the role the person moved into | Higher on both |
| H6 | Adding accessibility and fit changes the top 10 versus timing alone | Overlap of top 10 with and without them | Overlap below 80% |
| H7 | Training without `likely_ghost` listings predicts the test month better | Validation AUC / C-index | Better by 0.01 or more |

_N (the day cut-off) and the 14-day "open long enough" line are placeholders: fix them from EDA_SDA_4's and TimeSeries' closing-speed curves before running (2.1)._

## 2. Targets

Closes within N days (unknown, not "stayed open", when still open and seen for fewer than N days), days open (closed listings only), and the survival target (days open with censoring, so still-open listings are kept).

**What the data shows**
- (fill in after the run)

## 3. Preprocessing and leakage checks

Features come only from what is known at first sighting. Company counts use earlier listings only (`prior_company_counts`); `has_desc` and the text flags are checked below (EDA_SDA_2 3.9) before use. Preprocessing sits inside a `Pipeline` + `ColumnTransformer`, fit on training data only.

**What the data shows**
- (fill in after the run)

## 4. Split

Train April–July (plus the data's first two days), validate August, test September. Folds grouped by company. The test month is touched once, in section 8.

**What the data shows**
- (fill in after the run)

## 5. Baselines

`DummyClassifier`, newest first, and title keyword search (the career changer's keywords come from `ML_Fit`'s role queries).

**What the data shows**
- (fill in after the run)

## 6. Class imbalance

`class_weight="balanced"` is set in `rm.build_models()`; compare with upsampling (training folds only) and decision-threshold tuning on the validation month.

_To do._

## 7. Models

Logistic regression, decision tree, random forest, LightGBM, CatBoost, XGBoost (`rm.build_models()`), each with GridSearchCV over the company-grouped folds and `%%time`; Cox and survival models (`rm.cox_model()`, proportional-hazards check, company-clustered errors); LightGBM LambdaRank (`rm.lambdarank_model()`).

**What the data shows**
- (fill in after the run)

## 8. Evaluation

ROC-AUC, F1, precision-recall, confusion matrix, classification report; C-index for survival; precision@k and NDCG for rankings. The test month is used once here.

_To do._

## 8b. The priority queue

Fit score, chance still open and apply-by, accessibility and pay in one formula (`rm.priority_breakdown`). Weights start from EDA_SDA_4's adjusted model (`rm.START_WEIGHTS` is a placeholder) and are tuned on August with `rm.tune_weights`; only features that survived EDA_SDA_4. The queue against newest first: of each top 10, how many stay open long enough, are reachable and are in the role the person moved into.

_To do, needs `ML_Fit`'s fit scores._

## 9. Interpretation

SHAP and partial dependence; the per-listing breakdown for the detail view (each factor's rating and share of the score from `priority_breakdown`, top SHAP features behind chance still open under plain-language names; check SHAP adds up to each prediction); saved with the scores.

_To do._

## 10. Business translation

Top-10 results with company-clustered bootstrap intervals.

_To do._

## 11. Sensitivity

Live tolerance (`rm.LIVE_TOLERANCE_DAYS`), scope boundaries, CET `posted`, with and without ghost listings (H7).

_To do._

## 12. Solution report

Steps performed, difficulties and how they were solved, the final model and its score.

_To do._
