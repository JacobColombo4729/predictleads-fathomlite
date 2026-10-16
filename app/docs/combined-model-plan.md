# Combining the two models

**Status:** steps 1a–5 built and running as `combined_demo/` (Jacob, 2026-10-15); step 7's evaluation of the app rankers is next. The design asks for Jesse's agreement (`.eve/ASKS.md`). The combined model is a **third model**, built as its own package and run as its own app (see "Modular design"); `rrf` (Jesse's) and `mlrank` (Jacob's) stay as they are, as the baselines it is compared with.

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
| Trained layer (uses the text scores, BGE included, as inputs) | 81.7% |

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
combined_demo/                  the combined model as its own app: Jesse's app built with create_app(ranker=...)
  ranker.py                     the adapter to the app's ranker interface
  server.py, run.ps1, run.sh    its own backend (port 8002), page (5175) and session database
```

- **One bundle, one folder:** `data/clean/combined_v1/` (git-ignored), written by `python -m combined.build` and nothing else: the serving corpus (`jobs_prepared_corpus.jsonl`, `bge_embeddings.npy` and its manifest, the same format `rrf_ranker.py` reads), ML_Fit's trained layer and role artefacts, ML_Rank's listing scores and queue config, and `manifest.json` (versions, `ROLES`, feature lists, source commits, row counts). `CombinedModel.load` refuses a bundle whose manifest doesn't match the code.
- **Dependencies point one way:** `combined/` imports `analysis/src` (`career_switch`, `fit_model`, `rank_model`, `text_features`) and Jesse's text-match code; nothing in `analysis/src` or `app/` imports `combined/` except the adapter. The rules can't drift from the notebooks because they are the same functions.
- **Running it:** `combined_demo/run.ps1` / `run.sh` (OpenAI off unless `-AI` / `--ai`). Jesse's `create_app()` already takes a repository and a ranker, so the combined demo builds his app with the combined corpus and ranker without changing anything in `app/` (decided 2026-10-15, instead of adding `-Ranker combined` to his `api.py` and launchers: no merge conflicts, and it can't break the other two apps). The analysis notebooks can load the same bundle to evaluate it.
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

**Built (2026-10-15):** `combined/corpus.py` (`python -m combined.corpus build`, then `encode`), tested in `combined/tests/test_corpus.py` on a made-up export. 214,197 scope listings are still open when the data ends. A 5,000-listing slice took 10 seconds to build and about 70 seconds to encode on the GPU (21 s of text, 50 s of encoding), and Jesse's `rrf` backend serves it unchanged: a fictional profile gets its ranked top 10. **Full corpus:** 193,043 listings (of 214,197: 14,989 without a usable description, 6,165 duplicates; every id found), built in 6.4 minutes and encoded in 58 (25 of text, 33 on the GPU); the vectors match the ids row for row, and the `rrf` backend serves all 193,043 (ready in about 2 minutes) and ranks a fictional profile.

- `combined/corpus.py`: take the scope listings still open when the data ends (`open_at_end` in ML_Rank's `listing_scores.parquet`), prepare them with Jesse's unchanged `prepare_jobs.prepare_resource` (so the app reads them like any other listing), and encode them with the same BGE model and text as `rank_candidates.py encode`.
- Reads the raw export through `fetch` (the copy in `data/raw/`, or S3 from `.env`), like every other reader; nothing hard-coded.
- Encoding time grows with the rows: measure it on a slice first, and use the GPU PC if the full set is too slow on the laptop.
- Test: on synthetic listings, the corpus has the app's fields, ids match the embeddings row for row, and the manifest hash checks out.

### 1. ML_Fit saves what the app needs to score a new person

**Built (2026-10-15):** ML_Fit 9b with `analysis/src/fit_handoff.py` (`RoleScorer`, `save`, `load`), tested in `analysis/tests/test_fit_handoff.py`. `fit_eval`'s keyword, TF-IDF and BGE methods are split into a role half and a person half, and the old functions call the same halves, so ML_Fit's results are unchanged (test top 3 still 81.7%; the leakage audit's 8 checks pass). Parity on the 563 test people rescored from the saved pieces: largest input gap 4.2e-07, largest probability gap 1.5e-06 (BGE batching), the same top 3 roles for 100%.

ML_Fit saves scores for the Profiles people (`fit_scores.parquet`), not a model that can score someone new. Add a section that saves, to `data/clean/ml_fit/` (git-ignored):
- the chosen trained layer, fit on all development people (the same `Pipeline` that was evaluated), with `joblib`;
- what the per-role method scores need at serving time: each role's keyword set, TF-IDF role documents and vectorizer, BGE role vectors (same `bge-base-en-v1.5`), role skill profiles (`fit_model.role_profiles`);
- a manifest: `ROLES` order, feature column list, model family, sklearn version, ML_Fit commit.

Test: a round trip on synthetic people gives the same probabilities as the in-memory model.

### 2. Résumé → the trained layer's inputs

**Built (2026-10-15):** `analysis/src/fit_serving.py` (`person_features(entries)`), tested in `analysis/tests/test_fit_serving.py`. The app's entries become Profiles' job and education tables (dates from the start of each description, as the local reader writes them; projects left out) and go through `career_switch` unchanged. A career changer is described at a switch now, with a placeholder tech job this month; someone already in tech moves from their current tech role, with Profiles' rules at their first tech job for origin and training. The pre-switch text is Profiles 5c's (titles and programs, internships and high school left out). Parity: for three made-up people who switched at month M, `find_switches` and Profiles 5c's text rule give exactly the features `person_features` computes as of M. End to end, three fictional résumés go through the local reader, `person_features` and ML_Fit's saved scorer and layer in 0.1–0.4 seconds each, after a one-time load of about 9 seconds.

What it can't know: undated jobs and programs can't be placed on the timeline (counted in `undated_entries`), a single year on a program is read as its start, and a résumé the reader scrambles gives a scrambled timeline. The app should show these, not hide them.
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

**Built (2026-10-15):** `combined/role_fit.py` (`RoleFit.load()`, `.roles(entries, growth, growth_weight)`), tested in `combined/tests/test_role_fit.py`: per role, the trained layer's probability, the growth-adjusted score (weight 0 until BLS growth exists), the fit rating (ML_Rank's `rm.fit_ratings`, best = 100) and ML_Fit 7a's grouped SHAP reasons. For the reasons, ML_Fit 9b also saves 7a's SHAP background as its average preprocessed input only (`fit_model.background_mean`; `fit_handoff.reasons`); tested to give 7a's values exactly.

- Probabilities per role from the trained layer; growth applied afterwards with its visible weight (ML_Fit 7b), exactly as in ML_Fit.
- Rate them so that the person's best role is 100, as ML_Fit's fit rating does. That is what ML_Rank's weights were tuned on.
- Show the top roles and their reasons (ML_Fit's grouped SHAP: prior career / training / text similarity / skills) next to the listings.

### 4. Candidates: don't lose the roles the résumé doesn't read like

**Built (2026-10-15):** `combined/candidates.py` (`ListingIndex.from_handoff`, `gather`), tested in `combined/tests/test_candidates.py` with a made-up text matcher. The text matcher is passed in as a function; in the app it is Jesse's `RrfRankingBackend.rank`, unchanged. On the scope corpus, every one of the 193,043 listings has a role bucket from ML_Rank's hand-off and 28,249 are hidden by default. For two fictional career changers the pool held 316 and 312 listings: 100 from the overall text match and 216 and 212 added by their top 3 roles, none hidden, in about 1.5 seconds each (after a one-time load of about 5.6 minutes, mostly the RRF ranker's TF-IDF fit, as in Jesse's app).
A career changer's résumé reads like their old field. Text-only candidates can miss the very roles the trained layer recommends. Take the union of:
- the top text matches overall (as now), and
- the top text matches **within each of the person's top 3 roles** (RRF restricted to listings of that role bucket). Jesse's ranker already takes the listings it may return (`candidate_indices`), so `combined/candidates.py` calls it once overall and once per role; `rrf_ranker.py` doesn't change.

Hidden listings stay out, as now.

### 5. The fit factor

**Built (2026-10-15):** `combined/queue.py` (`Queue`) and `combined/model.py` (`CombinedModel.load(job_ids)`, `.recommend(entries, rank)`: steps 3–5 end to end), tested in `combined/tests/test_queue.py`. Over the same pool the queue's top 10 equals ML_Rank's backtest (`rm.queue_top_k`); listings outside ML_Fit's 8 roles get a neutral fit of 50 and say so. On the scope corpus with Jesse's RRF ranker, two fictional career changers each got a top 20 in about 1.6 seconds, all 20 in their top 3 roles, and 19 and 16 of the 20 came only through a role call. With fit weighted half and the best role rated 100, the trained layer picks the roles, the text match picks the listings within them, and still open and accessibility order them. That follows the tuned weights; the match blend below is the knob if the text match should count more.
- **Default: fit = role fit** of the listing's role bucket. That is what the backtest measured, so the tuned weights still apply.
- **Option: blend in the listing match**, `fit = (1 − a) · role fit + a · listing match`, with `a` a visible, adjustable weight (default 0). It breaks ties within a role using Jesse's text match. No data links people to listings, so `a` can't be tuned or reported as a result. It's a stated choice.
- Each result's breakdown shows role fit and listing match separately.

### 6. Role discovery and the Compass
Check what the app's Compass and role suggestions currently use (`app/code/role_family_classifier.py`, `app/DESIGN.md` section 3). Then decide with Jesse whether the trained layer's top roles drive them or are shown next to them.

### 7. Evaluation (what we can and can't claim)
Against real outcomes, with no labelling (`analysis/STANDARDS.md`, Validation design).
- **Role level, the trained layer:** ML_Fit's held-out test hit rates stand (81.7% [78.5%, 84.7%] at top 3, against the baseline's 63.2%), as long as the serving model is the evaluated one. ML_Fit 9b's parity check shows it is: inputs, probabilities, top 3 and reasons all match.
- **Serving matches training:** `fit_serving` is tested to give exactly Profiles' features and pre-switch text for people who switched at a given month (`analysis/tests/test_fit_serving.py`).
- **Role level, the app rankers (to build):** each app ranker (`rrf`, `mlrank`, the combined model) is run on every Profiles test person's pre-switch background over the serving corpus. Its top listings are turned into a role ranking (by the role buckets of the listings it returns), and it is scored on hit rate @3 and @5 against the same most-common-role baseline, with 95% intervals from resampling people, per pool, and the test run once. This puts the rankers the app actually serves next to ML_Fit's methods on the same answer key.
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
| 4–5 | Jacob, in `combined/`; it reuses Jesse's code (preparation, encoding, the RRF ranker) without changing it |
| Plug-in | **Built (2026-10-15) as `combined_demo/`**, its own app: `server.py` builds Jesse's app with `create_app(repository=..., ranker=...)`, `ranker.py` adapts the combined model to the app's ranker interface (reading the saved profile entries by id, read-only), `run.ps1` / `run.sh` start it on ports 8002 / 5175 with its own session database. Nothing in `app/` changes, so there's nothing to review there. Not yet shown on the page: the per-factor breakdown (the API keeps only job, rank and score from a ranker; the same for `mlrank`) |
| 6 | Jesse, with Jacob |
| 7 | Jacob |
| 8 | either, later |
