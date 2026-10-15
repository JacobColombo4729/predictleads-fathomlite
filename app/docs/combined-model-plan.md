# Combining the two models

**Status:** plan, not started (Jacob, 2026-10-15). To agree with Jesse before building (`.eve/ASKS.md`).

## Goal

One ranker that uses both halves' work:

- **Jesse's text match** (`app/backend/rrf_ranker.py`): TF-IDF and the pretrained BGE model, fused by reciprocal rank. It answers *which listings read like this résumé*, and it carries the app's feedback loop, Compass and "similar to a job you liked".
- **Jacob's models:** ML_Fit's trained layer answers *which roles people with this background actually move into*. ML_Rank's queue answers *which of those listings are worth applying to now* (still open, accessibility, pay), and hides what a career changer can't get.

## Where things stand

`PREDICTLEADS_API_RANKER=mlrank` (`app/backend/ml_rank_ranker.py`) already stacks the two:

1. Jesse's RRF ranker picks the top 200 text matches.
2. ML_Rank's queue reorders them by four factors × tuned weights: fit, still open, accessibility, pay.

**The gap is the fit factor.** In the app it is a stand-in: the RRF score, rated so that the best match is 100. In ML_Rank's backtest, fit is ML_Fit's role-level fit rating (`fit_scores.parquet`). So the weights were tuned on a different fit from the one the app uses.

| Fit method (ML_Fit test people) | Right role in top 3 |
|:--|:--:|
| Most-common-role baseline | 63.2% |
| Text match (RRF of TF-IDF and BGE) | 70.0% |
| Trained layer (uses the text scores, BGE included, as inputs) | 82.4% |

The trained layer already builds on BGE. Combining the models means bringing it into the app, not replacing Jesse's ranker.

## The combined ranker

```
résumé ──► local résumé reader ──► background features (career_switch rules)
   │                                         │
   │                                         ▼
   │                     ML_Fit trained layer ──► role fit (per role, best = 100)
   │                                         │
   ▼                                         ▼
Jesse's RRF ──► candidates: top text matches overall + top text matches within the person's top roles
                                             │
                                             ▼
             ML_Rank queue: fit (role fit, optional listing-match blend) + still open + accessibility + pay
                                             │
                                             ▼
                     ranked listings, with breakdown; feedback / Compass / similar-to-liked from Jesse's ranker
```

## Steps

### 0. Agree with Jesse
- Add an ask in `.eve/ASKS.md` that links this plan. The ranker lives in Jesse's half: the work goes on `jacob`, is logged in `JACOB.md`, and Jesse reviews before it reaches `main`.
- Agree on the naming: `mlrank` stays Jacob's app; the combined ranker becomes a new version of it (`ml_rank-v2-ml-fit`), not a new switch.

### 1. ML_Fit saves what the app needs to score a new person
ML_Fit saves scores for the Profiles people (`fit_scores.parquet`), not a model that can score someone new. Add a section that saves, to `data/clean/ml_fit/` (git-ignored):
- the chosen trained layer, fit on all development people (the same `Pipeline` that was evaluated), with `joblib`;
- what the per-role method scores need at serving time: each role's keyword set, TF-IDF role documents and vectorizer, BGE role vectors (same `bge-base-en-v1.5`), role skill profiles (`fit_model.role_profiles`);
- a manifest: `ROLES` order, feature column list, model family, sklearn version, ML_Fit commit.

Test: a round trip on synthetic people gives the same probabilities as the in-memory model.

### 2. Résumé → the trained layer's inputs
The trained layer reads ten Profiles columns plus the method scores per role:

| Input | From the app's résumé reader |
|:--|:--|
| `origin`, `from_level`, `from_bucket` | job titles through `career_switch.tag_jobs` / `role_bucket` |
| `background_area` | `career_switch.background_area` on the non-tech jobs |
| `retrained`, `retrain_field`, `study_level_before`, `study_undated` | education entries through `training_field` and `_study_level` |
| `years_before`, `n_jobs_before` | job entries and their dates (`parse_month`) |
| method scores per role | keyword, TF-IDF, BGE, RRF and skills scores of the résumé against each role (step 1's artefacts) |

- New module, shared by both halves: `analysis/src/fit_serving.py` (`person_features(entries) -> one-row DataFrame`), reusing `career_switch` so the rules can't drift from Profiles.
- Use the **local résumé reader** (`app/code/local_resume_reader.py`): no résumé text leaves the computer.
- Missing values go to the same "none" / median imputation the model was trained with; record which inputs were missing in the breakdown.
- Tests on the fictional profiles (`app/code/ai_ml_candidate_profiles.json`) and synthetic timelines: every column present, right dtypes, the rules give the same tags as Profiles on the same synthetic timeline.

### 3. Role fit at serving time
- Probabilities per role from the trained layer; growth applied afterwards with its visible weight (ML_Fit 7b), exactly as in ML_Fit.
- Rate them so that the person's best role is 100, as ML_Fit's fit rating does. That is what ML_Rank's weights were tuned on.
- Show the top roles and their reasons (ML_Fit's grouped SHAP: prior career / training / text similarity / skills) next to the listings.

### 4. Candidates: don't lose the roles the résumé doesn't read like
A career changer's résumé reads like their old field. Text-only candidates can miss the very roles the trained layer recommends. Take the union of:
- the top text matches overall (as now), and
- the top text matches **within each of the person's top 3 roles** (RRF restricted to listings of that role bucket).

Hidden listings stay out, as now.

### 5. The fit factor
- **Default: fit = role fit** of the listing's role bucket. That is what the backtest measured, so the tuned weights still apply.
- **Option: blend in the listing match**, `fit = (1 − a) · role fit + a · listing match`, with `a` a visible, adjustable weight (default 0). It breaks ties within a role using Jesse's text match. No data links people to listings, so `a` can't be tuned or reported as a result. It's a stated choice.
- Each result's breakdown shows role fit and listing match separately.

### 6. Role discovery and the Compass
Check what the app's Compass and role suggestions currently use (`app/code/role_family_classifier.py`, `app/DESIGN.md` section 3). Then decide with Jesse whether the trained layer's top roles drive them or are shown next to them.

### 7. Evaluation (what we can and can't claim)
- **Role level:** ML_Fit's held-out test hit rates stand, as long as the serving model is the evaluated one (step 1 manifest).
- **Serving matches training:** a parity check that the step 2 features from the fictional profiles fall inside the Profiles ranges. This is checked on aggregates only.
- **Listing level:** ML_Rank's backtest against newest-first. Rerun it with the step 4 candidate pool if that changes which listings are eligible.
- **Not a measure:** the team's ratings and logged app feedback are usability feedback only, never a reported result or training data.

### 8. Later, optional: fine-tune BGE on career moves
Contrastive pairs (prior career → the role moved into) from the Profiles development people, with sentence-transformers. Keep it only if, on the same test people, it beats the trained layer, or improves the trained layer as an input. Caveats: 2,813 people is small, it needs the GPU PC, and the app's corpus has to be re-encoded.

## Risks
- **Résumé reader vs Profiles:** Profiles reads structured résumé records, while the app reads free-form résumés. If the reader misses titles or dates, the inputs shift and the role fit gets worse without any warning. Step 2's tests and step 7's parity check guard this.
- **Version drift:** the app must refuse a model whose manifest doesn't match its `ROLES` or feature list.
- **Speed:** scoring one person is one BGE encode plus one model call. Measure it on start-up and per request.

## Who does what
| Step | Owner |
|:--|:--|
| 0 | Jacob (ask), Jesse (agree) |
| 1–3 | Jacob (`analysis/`) |
| 4–5 | Jacob, in `app/backend/ml_rank_ranker.py` and `rrf_ranker.py`, reviewed by Jesse |
| 6 | Jesse, with Jacob |
| 7 | Jacob |
| 8 | either, later |
