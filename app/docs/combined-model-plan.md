# Combining the two models

**Status:** plan, not started (Jacob, 2026-10-15). To agree with Jesse before building (`.eve/ASKS.md`). The combined model is a **third model**, built as its own package and pluggable into either app (see "Modular design"); `rrf` (Jesse's) and `mlrank` (Jacob's) stay as they are, as the baselines it is compared with.

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

**A second gap: the listings.** Both apps rank only Jesse's targeted pool (`app/code/extract_targeted_pool.py`): titles for AI/ML, data and technical sales plus six O*NET occupations, chosen when the audience was the team itself. The agreed scope (EDA_SDA_1 8c: the roles TripleTen trains for plus adjacent tech roles, 908,291 listings) also covers software and web development, QA, cybersecurity, IT support, networks and systems, database, IT project management and market research, which that pool mostly misses. A user whose background points there gets few relevant listings, whatever the ranker. The combined model serves the scope's listings instead (step 1a). Jesse's methods (BGE, TF-IDF, RRF) carry no personal data and are kept.

## Modular design

The combined model is one self-contained package at the top of the repository, shared by both halves like `fetch/`. Neither app nor any notebook is inside it; both plug into it.

```
combined/                       the third model (Python package, no web framework)
  __init__.py                   CombinedModel.load(bundle) -> .roles(person), .rank(person, limit), .manifest
  corpus.py                     the serving corpus from the scope, in the app's corpus format (step 1a)
  person.py                     résumé entries -> ML_Fit's person features, via career_switch (step 2)
  role_fit.py                   ML_Fit's trained layer at serving time: probabilities, growth, rating, reasons (step 3)
  candidates.py                 text matches overall + within the person's top roles (step 4)
  queue.py                      ML_Rank's queue: fit + still open + accessibility + pay (step 5)
  bundle.py                     writes and checks the bundle and its manifest
  build.py                      python -m combined.build: corpus, role-fit artefacts, queue snapshot -> one bundle
  tests/                        synthetic data only
app/backend/combined_ranker.py  thin adapter to the app's ranker interface: PREDICTLEADS_API_RANKER=combined
```

- **One bundle, one folder:** `data/clean/combined_v1/` (git-ignored), written by `python -m combined.build` and nothing else: the serving corpus (`jobs_prepared_corpus.jsonl`, `bge_embeddings.npy` and its manifest, the same format `rrf_ranker.py` reads), ML_Fit's trained layer and role artefacts, ML_Rank's listing scores and queue config, and `manifest.json` (versions, `ROLES`, feature lists, source commits, row counts). `CombinedModel.load` refuses a bundle whose manifest doesn't match the code.
- **Dependencies point one way:** `combined/` imports `analysis/src` (`career_switch`, `fit_model`, `rank_model`, `text_features`) and Jesse's text-match code; nothing in `analysis/src` or `app/` imports `combined/` except the adapter. The rules can't drift from the notebooks because they are the same functions.
- **Plugging in:** either launcher selects it with `-Ranker combined` (`run_demo_backend.ps1`, `run_jacob.ps1` and the `.sh` versions), pointing `PREDICTLEADS_API_CORPUS_DIR` and `PREDICTLEADS_API_JOBS_PATH` at the bundle's corpus. Jesse's app keeps its OpenAI features; Jacob's keeps them off. The analysis notebooks can load the same bundle to evaluate it.
- **The other two stay:** `rrf` and `mlrank` are unchanged, so the three can be compared on the same people. They can also be pointed at the scope corpus, which fixes the listings gap for them too. Run one backend at a time (two already use about 9 GB).
- **Version id:** `combined-v1` (`ranker_id = "combined"`).

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
- Agree on the design above: a third model (`combined`) in its own package, selectable in both apps; `rrf` and `mlrank` unchanged.
- Agree that the combined model serves the scope's listings, not the targeted pool.
- Ask what the Compass thresholds were set on ("a few real examples", `ml-inventory.md` section 3); they stay out of the combined model until that is known.

### 1a. The serving corpus from the scope
- `combined/corpus.py`: take the scope listings still open when the data ends (`open_at_end` in ML_Rank's `listing_scores.parquet`), prepare them with Jesse's unchanged `prepare_jobs.prepare_resource` (so the app reads them like any other listing), and encode them with the same BGE model and text as `rank_candidates.py encode`.
- Reads the raw export through `fetch` (the copy in `data/raw/`, or S3 from `.env`), like every other reader; nothing hard-coded.
- Encoding time grows with the rows: measure it on a slice first, and use the GPU PC if the full set is too slow on the laptop.
- Test: on synthetic listings, the corpus has the app's fields, ids match the embeddings row for row, and the manifest hash checks out.

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
| 1a | Jacob (`combined/corpus.py`), reviewed by Jesse (it reuses his preparation and encoding) |
| 1–3 | Jacob (`analysis/` saves the artefacts; `combined/` serves them) |
| 4–5 | Jacob, in `combined/` plus the adapter `app/backend/combined_ranker.py`, reviewed by Jesse; `ml_rank_ranker.py` and `rrf_ranker.py` unchanged |
| 6 | Jesse, with Jacob |
| 7 | Jacob |
| 8 | either, later |
