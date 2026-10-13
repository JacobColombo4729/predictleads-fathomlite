<!-- Generated from notebooks/ML_Fit.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: ML_Fit (fit, role discovery and the compass)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

**Which roles and listings fit a career changer?** This notebook builds the **fit model**: from a career changer's pre-switch background, it suggests the role buckets their skills transfer to (**role discovery**), scores how well each listing fits them, and **assembles the retraining compass** and the **Ideas tab's** inputs in its last section: the one place the pieces from Profiles, EDA_SDA_2, EDA_SDA_3 and this notebook come together for the app. `ML_Rank` combines its fit scores with the listing model into the priority queue.

**Inputs:** the **Profiles modeling table** (`data/profiles/modeling_table.pkl`: `feature` columns only, `outcome` columns never), each career changer's **pre-switch text** (job titles and programs before the switch, from the Profiles timeline), EDA_SDA_3's cleaned descriptions and **skills mapped to O*NET**, and O*NET's skills and Career Changers Matrix.

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

### The ladder of methods

| Step | Method | Role | Why it's here |
|:-:|:--|:--|:--|
| 1 | **Keyword overlap** (shared title and skill words) | Simplest baseline | What a keyword search would find |
| 2 | **TF-IDF cosine** | Baseline ML model | The course standard for text similarity |
| 3 | **BGE embeddings, cosine** (a small or base BGE model, run locally) | Stronger model | Catches meaning, not just shared words |
| 4 | **RRF hybrid** (reciprocal rank fusion of TF-IDF and BGE ranks) | Hybrid | Combines word overlap and meaning without tuning weights |
| 5 | **Skills-based matching** (résumé and listing both mapped to O*NET skills) | Transferable fit | Steps 1–4 compare words, and a career changer's résumé uses the old field's words, so they favor old-field roles; skills don't |
| 6 | **Trained layer (Option B)**: steps 2–5 scores plus Profiles features (origin, training field, years before, `from_level`, …) in a model trained on real career moves. Candidates: logistic regression, LightGBM / CatBoost / XGBoost, and a **small neural network (scikit-learn `MLPClassifier`)**, kept only if it wins | **The project's fit model** | Turns similarity into a prediction learned from where career changers actually went; explainable with SHAP |
| 7 | **Stretch: fine-tuned BGE** (contrastive training so a person's pre-switch background lands near the role they moved into; small model, CPU, a few thousand pairs from the training folds only) | A neural network trained on our own data | Tried only if step 6 plateaus; aimed at the old-field vocabulary problem; scored like every other step |

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

**Leakage rules** (`STANDARDS.md`):
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
| 7b. **Growth-adjusted recommendations** | | Among the roles a person fits, roles expected to grow over the next 5 years (at most) rank higher: expected growth is applied **after** the fit model as a visible adjustment with a user-adjustable weight, never trained in (it's a future estimate, so historical career moves can't validate it), and always shown with the role's entry-level share and openings. Source: `role_growth.pkl` (BLS 2025–2035, rolled up to buckets by 2025 employment; job growth tasks in `ANALYSIS.md`). If the team adopts the proposed error band (job growth tasks in `ANALYSIS.md`), the range is shown too; with the scenario extensions (4b/4d), the baseline scenario sets the adjustment and the others are shown alongside. IT project management uses the 15-1299 catch-all's growth, flagged as a proxy |
| 8. **The compass, assembled** | | The single place the compass is put together for the app: Profiles' paths (prior career → training → first tech job; within-tech moves, sideways or up) with people counts and thin paths marked; each role's 2026 openings, hourly pay, closing speed, entry-level share and expected growth (EDA_SDA_2 3.12); the Ideas tab's inputs: skills to build per suggested role, options and time to switch, pay increase against the current field (EDA_SDA_2 1c) and growth. Saved as compass tables under `data/clean/` |
| 9. Solution report | Steps, difficulties, final model and score | Hand-off of fit scores to `ML_Rank` |

## 2. Targets and the split

The cases are the **career changers** from Profiles. For each one, the input is their **pre-switch text** (job titles and study programs from before the switch, `data/profiles/pre_switch_text.pkl`, Profiles 5c) and the answer is the **role bucket they actually moved into** (`target_bucket`). For moves within tech, the person's current role (`from_bucket`) is left out of the candidates: a move means a different role.

**20% of the people are held out as the test set and scored once, at the very end** (section 6b). Everything else is developed on 5 folds of the rest. Both are split by person and stratified by role and pool, so every role and pool shows up in each.

**What the data shows**
- **2,813 career changers, split before anything is scored:** 2,250 for development (1,640 changed roles within tech, 325 retrained after working in another field, 285 new grads from another field) and 563 held out for the test (409 / 82 / 72), 20% of each pool. Within-tech movers are 73% of the people (2,049), so every result below is also shown per pool.
- **Everyone has pre-switch text:** 0 of the 2,813 are empty, so no person ties across all methods.
- **One warning from the split (2a):** scikit-learn says the least populated class has only 2 members, fewer than the 5 folds. The role-by-pool groups with fewer than 7 people are lumped into one "rare" group before splitting, and the warning says a group has only 2 people in development, so the folds cannot spread it evenly. It touches a handful of people and no number below depends on it, but it is a loose end in `fit_eval.split_people` (the cutoff is counted on all people, not on development people), noted for Jacob rather than changed here, since changing it changes the split.

## 3. Role descriptions

Each role bucket is described by its **2026 listings**: a reproducible sample of up to 2,000 per role from the full-file scope tables (EDA_SDA_1 cell 4b), each listing's title and description as plain text. The methods compare a person's background with these. No person is involved here, so nothing about the test people leaks into the methods. Later, EDA_SDA_3's cleaned text can replace these with one setting.

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
- **All four methods scored all 2,813 people against all 8 roles in 6.4 minutes** (BGE on the laptop's CPU included). The scores are computed once per person, so they are the same in the development folds (6a) and the test (6b); only the baseline is refit per fold.

## 5. The trained layer

_To do: steps 5–6 of the ladder (skills-based matching, and the trained layer on these scores plus the Profiles features)._

## 6. Evaluation

**Hit rate @k:** the share of people whose actual role is in the method's top k (3 and 5), overall and per pool. **Development (6a):** mean ± standard deviation over the 5 folds. **Test (6b):** run **once**, on the held-out 20%, with 95% intervals from resampling people (2,000 bootstrap samples). **Current-role pull (6c):** for moves within tech, how often a method would put the person's current role first if it were allowed: the old-field bias word matching tends to have. **Rule:** the test was run once to see steps 1–4; all choices in section 5 are made on development folds only, and 6b is run once more at the very end, with the trained layer.

**What the data shows (development, 5 folds)**
- **At top 3, the text methods beat the most-common-role baseline only modestly:** RRF 67.1% ± 1.4% and TF-IDF 66.6% ± 1.2% against the baseline's 63.3% ± 0.3%, BGE 65.5% ± 1.3%. Keyword overlap is below the baseline (53.9% ± 1.3%). The best method gains 3.8 points over the baseline; that is larger than the fold-to-fold spread but was not tested as a paired difference.
- **At top 5, the baseline wins:** 87.1% ± 0.3% against 85.2% (RRF), 85.1% (TF-IDF), 85.0% (BGE) and 79.1% (keyword). With 8 roles, the five most common roles already cover most people, so there is little room for a text method to add.
- **By pool (top 3), the gain is in the within-tech movers:** RRF 69.3% against the baseline's 62.9%; for people who retrained after working, RRF 64.9% against 58.8%.
- **For new grads from another field, the baseline beats every method:** 70.9% against 57.6% (TF-IDF), 57.2% (BGE), 56.9% (RRF) and 43.8% (keyword). A likely reason, not tested: a student's pre-switch text is a degree and a few student jobs, which carries little of the vocabulary the roles' listings use, so the text methods have little to go on and the most common roles win.

**What the data shows (test, 563 held-out people)**
- **The ranking of the methods is the same as in development.** At top 3: RRF 70.0% [66.1%, 73.7%], TF-IDF 69.8% [66.1%, 73.4%], BGE 67.0% [63.4%, 70.9%], the baseline 63.2% [59.1%, 67.1%] and keyword overlap 60.2% [56.1%, 64.3%]. RRF is 6.8 points above the baseline, but the intervals overlap a little (they are not paired intervals, so this is a rough reading); BGE's gain over the baseline (3.8 points) is within the intervals.
- **At top 5, the baseline is again the best:** 87.0% [84.2%, 89.7%] against RRF 86.7% [83.7%, 89.3%], TF-IDF 84.7%, BGE 84.4% and keyword 81.5%; at top 5 no method is distinguishable from the baseline.
- **Per pool (top 3):** within tech RRF 73.1% [68.7%, 77.3%] against the baseline's 63.1% [58.2%, 67.7%] (the clearest gain). After working in another field (82 people), RRF 64.6% [54.9%, 74.4%] against 57.3% [46.3%, 68.3%]: the same direction, but the intervals overlap widely. New grads (72 people): the baseline 70.8% [59.7%, 80.6%] is above RRF 58.3% [47.2%, 69.4%], TF-IDF 59.7% and BGE 55.6%.
- **These test numbers are not a final result for the project's fit model:** the trained layer and skills matching (section 5) are not built yet.

**What the data shows (current-role pull, development, within-tech movers)**
- **If allowed, TF-IDF and RRF would put the person's current role first for 73.7% of within-tech movers, and BGE for 69.3%.** That is the old-field pull: a person's pre-switch text is mostly their current role's titles, so they look most like it. In 6a and 6b this role is removed from the candidates, so the hit rates above are not inflated by it; but it shows the methods rank by "looks like where you are", not "where people like you go".
- **Keyword overlap's 42.0% is lower, but that is not a point in its favor:** it is also the weakest method at hitting the real next role (53.9% at top 3), so its lower pull comes with weaker matching in general.
- **The plan for section 5, stated as a plan, not a result:** skills-based matching (the person's and each listing's skills mapped to O*NET, which do not depend on the old field's vocabulary) and a trained layer on real career moves (steps 5 and 6 of the ladder) are meant to beat the baseline at top 5 as well, to do better for new grads, and to reduce this pull. Whether they do is not known yet.

### Where the notebook stands (sections 1–6)
- **Steps 1–4 are built, run on all 2,813 career changers and evaluated on real career moves.** The text methods find the role people moved into modestly more often than "the most common role" at top 3 (development: RRF 67.1% against 63.3%; test: 70.0% against 63.2%) and not at top 5 (baseline 87.1% in development, 87.0% in test, the best). They help most for within-tech movers and not at all for new grads from another field, and they favor the person's current role (TF-IDF and RRF 73.7%).
- **Still to build: section 5 (skills matching, the trained layer) and sections 7–9** (interpretation, growth-adjusted recommendations, the compass, the solution report). No conclusion on the fit model can be drawn until then.
- **Rule for what comes next:** the test (6b) has now been run once, to see how steps 1–4 do. Every choice in section 5 (features, model, settings, whether the neural network is kept) is made on the development folds only, and nothing is tuned to the test numbers above. 6b is run once more at the very end with the trained layer included, and that is the test result the project reports.

## 7.–9.

_To do: interpretation (SHAP), growth-adjusted recommendations, the compass, and the solution report._
