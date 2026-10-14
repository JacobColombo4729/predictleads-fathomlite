# Analysis

> The analysis priority queue. Top = next. Delete a task when it's done. Add new tasks to the tier where they belong. The app has its own queue in [`../app/APP.md`](../app/APP.md).
> **[agent] = Jacob's Foreman may draft it in the background** (code only, tested on synthetic data, on its own branch for review; README step 14). Add or remove [agent] to control what it takes. `[agent: Jesse]` hands a task here to Jesse's Foreman instead.
> Each notebook's full plan is in the notebook itself; how we work is in [`STANDARDS.md`](STANDARDS.md).

---

## Now

- **Full dataset, full scale for every notebook** (decided 2026-10-10; stream test: ≈13.4M listings, ≈1.0M in scope, ≈40 minutes, ≈8.4 GB):
  1. **Claude:** EDA_SDA_3 for full scale (also: a second ghost sign, since "company still posting" no longer separates forgotten listings at big companies (82% of stale listings): the same company and title still open after 90 days while newer copies of that title keep appearing) (reads `scope_table.parquet` and `scope_descriptions.parquet`): text read part by part, regex features over every listing, spaCy lemmas on a large stratified sample (about 150,000), cache in parts. Hold EDA_SDA_3's first run until then.
  2. **Jacob (any time):** delete `data/flat/all_fields_lean_full/` (1.1 GB): nothing reads it any more. After EDA_SDA_3 is adapted, also the sample-era tables in `data/clean/` (`analysis_table.pkl`, `scope_table.pkl`, `job_openings_lean_200000_v2.pkl`).
  3. **New analysis the full data opens** (added to each notebook's plan): real market sizes per role (openings, new per week), role buckets by month × source and by state, a company watch list (employers posting entry-level roles often), rare signals (cross-posts, evergreen wording, licenses, staffing agencies), full-scale daily series for TimeSeries, more training data and a proper test month for ML_Rank.
- **Jacob (from 2026-10-11):** upload the slide deck `reports/Team update 2026-10-11.pptx` (24 slides, EDA_SDA_1 and 2 at full scale with the feature screening and the 2026-10-11 pass; local only) to Google Slides, then delete the 2026-10-09 and 2026-10-10 decks; tell the team (below).
- **Known gaps in `.claude/settings.json`:** `Read(./.env.*)` also blocks `.env.example`; `Read` rules don't stop shell commands such as `cat .env`.
- **Tell the team** to pull, rerun `pip install -r requirements.txt` (and on a Mac `brew install libomp`), run `python -m spacy download en_core_web_sm`, set up the new layout (README steps 4–9: one `.venv`, `.env`, `data/` and `pretrained/` at the top; `S3_PATH` in `.env`), and run `nbstripout --install` once (it's in `requirements.txt`): git then strips notebook outputs on every commit.

## Next

- **Differentiation ideas (2026-10-10), lead with 1, 3 and 5** (1, 3 and 5 drafted as tested functions in `src/differentiators.py`, 2026-10-13; they need notebook cells once their inputs exist): seven planned additions, described in [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md) (Planned additions): (1) which skill unlocks the most jobs; (2) certification payoff; (3) stepping-stone planner; (4) "apply now" alerts; (5) a backtested ranking claim; (6) pay now vs. one move later; (7) a map of reachable openings.
- **Set up the GPU PC before ML_Fit** (or before EDA_SDA_3, if it uses transformer models; spaCy's small model is fine on the Mac): BGE embeddings of the scope's ~750K descriptions take hours on a GPU (Jesse: 244K in about 45 minutes on an RTX 3070) vs. likely a day or more on the Mac's CPU. Steps: clone the repo; Python 3.11 venv from `requirements.txt` with the CUDA build of PyTorch; `nbstripout --install`; set `PL_DATA_DIR` to the PC's data folder; copy only what's needed (`data/clean/analysis_table.parquet`, `scope_descriptions.parquet`) by external drive, or stream from S3 with the PC's own AWS profile; add the PC's data folder to the end-of-project deletion list. EDA_SDA_1 and 2 stay on the Mac (pandas is CPU-only; no gain).
- **EDA_SDA_3 (text features; no matching models), in progress:**
  - **Sections 1–2 written** (setup, text loaded from the saved parts, HTML to structured text, boilerplate removal, lemmas; cached in `data/clean/eda3_text.pkl`): needs a run, then notes. Section 3 (years and degree rules) is next.
  - **Dropped 2026-10-13: checking EDA_SDA_2's two title rules** (they were first-pass keyword rules, decided 2026-10-11): (1) staffing agencies from recruiter titles (`agency_recruiter`, `staffing_by_title`, 2b.1); (2) 13-1161 split into analysts vs. marketing specialists by job title (`marketing_specialist`, 3.0). They won't be hand-checked, so both are labels ("staffing agency, from the title"; marketing specialist) and opt-in filters, not hidden or demoted by default. Marketing specialists were hidden before; `hidden_occupation` (`src/career_switch.py`) now covers document management only, and EDA_SDA_2 was rerun with that (2026-10-13).
  - **Hand-checking:** not done (decided 2026-10-13: the samples are not hand-checked). Text rules are labels and opt-in filters, never hidden by default; section 5's samples stay unmarked, and the text says the rules are unvalidated.
  - ~~Validate the keyword flags by hand~~ (dropped 2026-10-13): the flags stay keyword rules, reported by how often they fire and compared with EDA_SDA_1's rules.
  - Text-blocker labels per role bucket (years-asked threshold, required credentials), shown in the app as labels and opt-in filters, each with a count of what it would hide. The "asks 5+ years" preview would hide 32% of open listings (EDA_SDA_2 3.11): check required vs. preferred and the year counts first. The degree flag fires on only 3.7% of listings: widen and validate its pattern ("Bachelor's in …") before any degree rule.
  - Evergreen wording (feeds the ghost-listing flag), required vs. preferred qualifications, **skills mapped to O*NET** (the input to `ML_Fit`'s skills-based matching): built and run 2026-10-13 (`src/onet.py`, section 6): 1,518,984 matches over 459,912 listings into `eda3_skills.parquet`, with everyday-word tool names and generic skill names excluded; the matches are unchecked keyword hits.
  - **New text rules (added 2026-10-10):** security clearance required, "no visa sponsorship", travel percentage, shift or on-call work, contract-to-hire. Clearance and sponsorship are hard blockers for some career changers (per applicant in the app).
  - Save the scope table with text features for EDA_SDA_4 and `ML_Fit`.
- **EDA_SDA_4 (statistics)** (run at full scale 2026-10-13, findings written, feature decisions from EDA_SDA_2's pre-set rule: 13 of 43 text features kept, 30 dropped, 45 non-text kept and 9 dropped, redundancy checked against every stronger feature; `modeling_ready.parquet` saved, 908,291 x 67; still to do: mark the text rows of 9d after EDA_SDA_2's 3.9 check, decide whether to test `career_open`):
  - Descriptive statistics by group; Pearson + Spearman, Cramér's V, point-biserial; VIF.
  - Group interactions, including by role bucket, and checks for confounding (Simpson's paradox).
  - Adjusted check: a Cox survival regression with every standard confounder (company volume, source and platform, staffing agency, role bucket, seniority, month, has a description; company type is not in the scope table), company-clustered errors, hazard ratios with 95% intervals and a proportional-hazards check.
  - Welch's t-test plus Mann–Whitney (Wilcoxon rank-sum), chi-square and log-rank, with effect sizes and Benjamini–Hochberg; this is where EDA_SDA_1's and EDA_SDA_2's descriptive gaps get significance tests.
  - Company-clustered bootstrap intervals; survival checks (censoring, left truncation, competing risks).
  - **In-depth feature analysis (added 2026-10-10, full data):** (1) **stability over months**: does each feature's effect hold in every month, not just overall (ML_Rank trains on April–July, tests on September); (2) **within-company comparisons**: does it still matter between listings from the same company; (3) **effects by role bucket**; (4) **leakage audit**, one row per feature: known when the listing first appears?; (5) **drift checks**: which distributions shift over time, and do they line up with the collection changes (job boards after May, descriptions from June); (6) **information beyond the confounders** (mutual information or likelihood-ratio tests). Priority: 1, 2 and 4 first.
  - Record each feature kept or dropped by EDA_SDA_2's pre-set rule (3.13); save the modeling-ready table. Result: of 43 text features 13 kept and 30 dropped, the non-text features keep EDA_SDA_2's verdicts (45 kept, 9 dropped); the confounders (volume, source, having a description) matter far more than any text flag.
- **TimeSeries:**
  - Daily and weekly series from `first_seen` and from `posted` (Central European dates), to separate crawler from employer rhythm.
  - Rolling stats, decomposition (STL), ACF/PACF, ADF and KPSS.
  - Lag and rolling features with `.shift()`; a time-ordered split, test used once.
  - Models: `DummyRegressor`, linear regression, random forest, ARIMA / SARIMA; RMSE and MAE.
  - Survival by month first seen, and a "check every N days" cadence.
- **ML_Fit (fit, role discovery, the compass):**
  - **Built (2026-10-12): steps 1–4 and their evaluation** (sections 2–4 and 6, `src/fit_eval.py`): keyword, TF-IDF, BGE and RRF ranked against the most-common-role baseline on real career moves, hit rate @3/@5 per pool, 5 development folds by person, a held-out 20% scored once with bootstrap intervals, and the current-role pull. **Run in full on 2026-10-13** (EDA_SDA_1, Profiles 5c, then ML_Fit; 2,813 career changers: development 2,250, test 563): at top 3 RRF 67.1% and TF-IDF 66.6% against the most-common-role baseline's 63.3% in development (test: RRF 70.0%, baseline 63.2%); at top 5 the baseline is best (87.1% development, 87.0% test); for new grads from another field the baseline beats every method; TF-IDF and RRF would put a within-tech mover's current role first 73.7% of the time. To do: the `split_people` cutoff for rare role-by-pool groups (counted on all people, not on development people; sklearn warns that one group has 2 people in development).
  - **Built and run (2026-10-13): steps 5–6** (section 5, `src/fit_model.py`): skills-based matching on O*NET Transferable Skills (carry-over share, centered profile cosine) and the trained layer (six candidates tuned on the development folds by top-3 hit rate; out-of-fold for development, one fit for the test; a Profiles-features-only version). Results on the 563 held-out people (6b, the last run for the fit model): **the trained layer (logistic regression; all six candidates within 2 points, the neural network lowest) reaches 82.4% [79.4%, 85.6%] at top 3 and 96.8% [95.2%, 98.0%] at top 5** against the baseline's 63.2% and 87.0%; the Profiles features alone give 75.8% / 94.5%, so the scores add about 6.6 points; skills matching alone is below the baseline (profile cosine 55.6%, carry-over 48.3% at top 3); new grads 75.0% against 70.8% (intervals overlap, not shown better); the trained layer has no current-role pull (0.1%). Caveat: the chosen logistic regression C (0.03) is at the edge of its grid, left as is because the test has been seen. **Next: sections 7–9** (SHAP interpretation, growth-adjusted recommendations, the compass, the solution report).
  - **Built (2026-10-13), to run: sections 7–9** (`src/fit_model.py`, tested): SHAP of the trained layer on the test people, grouped as prior career / training / text similarity / skills that carry over (7a); growth applied after the model with a weight (7b; waits for `role_growth.pkl`); the compass assembled (8): paths and the 2026 market (EDA_SDA_2 3.12), each role's 10 most asked skills marked covered or to build per background (EDA_SDA_3 skills, O*NET), pay against the prior field (EDA_SDA_1 7.2), saved as `compass_paths.pkl`, `compass_skills.pkl`, `compass_pay.pkl`, `compass_market.pkl`; the hand-off (9): `fit_scores.parquet` and `fit_breakdown.parquet` for ML_Rank. Run ML_Fit from the top, then Thread (the solution report is written from the run). Not covered: time to switch (not in the data); a scorer the app can call for a new résumé (the trained layer needs the method scores, so the app needs the role listing sample, TF-IDF and BGE centroids too).
  - Build each career changer's pre-switch text from the Profiles timeline (titles and programs before the switch; no undated skills).
  - The ladder: keyword overlap → TF-IDF cosine → BGE embeddings (small or base, local) → RRF hybrid → skills-based matching on O*NET skills → a trained layer (Option B) on real career moves (candidates include a small neural network, `MLPClassifier`); stretch: fine-tune BGE on career moves if the trained layer plateaus.
  - Score each step on hit rate@3/@5 for the role people actually moved into, per pool and overall, against a title-keyword baseline; measure old-field bias; exclude the current role for within-tech moves; folds grouped by person.
  - SHAP, including each listing's top fit features for the detail view's rating breakdown; hand fit scores and their breakdown to `ML_Rank`.
  - **Assemble the compass (last section):** Profiles' paths with people counts, each role's 2026 market and growth (EDA_SDA_2 3.12), and the Ideas tab's inputs (skills to build, options and time to switch, pay increase vs. the current field from 1c, growth); save the compass tables for the app.
- **ML_Rank (the priority queue)** (v1 and v2 scaffolded 2026-10-13, 14-day window: 2a reads its features from `modeling_ready.parquet`, which EDA_SDA_4 has saved; the queue backtest 8d-8e reads `ML_Fit`'s `fit_scores.parquet`, built 2026-10-14; needs a run):
  - Write the hypotheses with thresholds **before** any modeling.
  - Targets: closes within N days, days open, and the survival target.
  - Split by time (train April–July, validate August, test September), folds grouped by company; test used once.
  - Baselines: `DummyClassifier`, newest first, title keyword search.
  - Models: logistic regression, trees, random forest, LightGBM, CatBoost, XGBoost; Cox and survival models; LambdaRank.
  - Leakage: counts from earlier listings only; a date-picker model trained only on what was known by its date; check that `has_desc` and the text flags are known at first sighting (EDA_SDA_2 3.9) before using them.
  - **The priority queue:** combine the fit score, chance still open and apply-by, accessibility and pay in one formula; weights start at 0.4 / 0.3 / 0.2 / 0.1 and are tuned on the validation month (EDA_SDA_4's hazard ratios are about closing speed, so they do not set these four); only features that survived EDA_SDA_4.
  - Evaluation (ROC-AUC, F1, C-index, precision@k, NDCG); the queue against newest-first: of each top 10, how many stay open long enough, are reachable and are in the role the person actually moved into; SHAP and the per-listing rating breakdown for the detail view (each factor's rating and share of the score, top features behind chance still open), sensitivity checks, solution report.
- **Job growth expectations** (one number per role bucket, over at most 5 years, from **BLS Employment Projections 2025–2035**; [data tables](https://www.bls.gov/emp/data.htm)):
  1. **Get PredictLeads' written consent first** to combine their data with BLS data (agreement 4.3), and note it in your change log (`JACOB.md` or `JESSE.md`). Check the BLS terms of use.
  2. Download the 2025–2035 occupation table into `data/external/` (a `fetch/` script).
  3. Join on the 6-digit occupation code (the first part of `onet_code`, e.g. `15-2051`) and roll up to role buckets with `ONET_ROLE_BUCKETS`, weighted by 2025 employment. BLS has no projection for the detailed codes inside 15-1299, so **IT project management (15-1299.09) takes 15-1299's growth as a flagged proxy**.
  4. Convert to the chosen horizon at the same yearly rate: growth over *n* years = (1 + 10-year growth)^(*n*/10) − 1, *n* ≤ 5.
  5. Save `role_growth.pkl` (role bucket, occupation codes, 2025 employment, 10-year growth, yearly rate, *n*-year growth, vintage) and add the expected-growth column to the compass (EDA_SDA_2 3.12), then rerun.
  6. `ML_Fit`: apply growth **after** the fit model as a visible, user-adjustable adjustment (not a trained input), and show how much it moved each suggestion. Always show it next to the role's entry-level share and today's openings.
  7. Write-up: say "expected growth (official projections)", never "will grow"; limitations: national, occupation-wide, assumes full employment, no business cycles.
- **Decide the time frames** for career options and job growth in the compass (team; growth at most 5 years), and which growth extensions to adopt (each reopens the 2026-10-09 decision to drop other external data):
  - **Error band** from BLS's own track record (an earlier projection vs. OEWS actuals), shown as a range: recommended, small.
  - **Listing momentum:** each bucket's new-listings trend from TimeSeries, shown next to growth, never blended in: recommended, small.
  - **Interest-rate scenarios** for years 1–2 (FOMC paths; sensitivity from fed funds vs. Indeed/JOLTS postings), fading to BLS by year 5: stretch.
  - **Horizon-weighted blend** of a near-term postings signal with the BLS trend (weights 0.8 → 0 over years 1–5, tuned in a backtest): later; our six months can't support it.
  - **State level** (Projections Central): only if the app adds location-based growth.
- **O*NET (in scope, small): downloaded 2026-10-13** (`python -m fetch.get_onet`, release 31.0, CC BY 4.0). Release 31.0 renamed the files: skills are **Essential Skills**, **Transferable Skills** and **Software Skills**, and the Career Changers Matrix is now **Related Occupations**; all occupations are kept (filter to the scope's occupations when reading). Also the **Job Zones** (O*NET's level of education and experience per occupation: an accessibility prior for the feature screening). Listings already carry O*NET codes, so they join directly.
- **Write up the analysis** and **choose the final format** (write-up, notebook demo, or app).

## Blocked

- **Team:**
  - Assign owners for the notebooks above, the app, and the pitch.
  - Set a meeting cadence and a place for async discussion.
  - Confirm the final due date and deliverables.
- **Project advisor:** get answers to our questions (scope, deliverables, AI tools and data, applicant profiles, external data, hosting, presenting results, data deletion), then record them in your change log (`JACOB.md` or `JESSE.md`). Get the AI-and-data answers in writing; the agent stage of the app depends on them, and the answer on collecting career-changer profiles blocks the volunteer profiles.
- **Applicant profiles** (after the advisor answers): the plan, the volunteer steps and the blank form are in [`docs/volunteer_profiles.md`](docs/volunteer_profiles.md) (with [`docs/volunteer_profile_template.csv`](docs/volunteer_profile_template.csv)). Assign owners for the tasks, draft the consent form, recruit career changers, convert their exports, and build the synthetic generator in `src/`.
- **PredictLeads:** get answers to our data questions (field definitions, crawl timing, `status`, sources, sampling, duplicates, older daily exports, `seniority`; why job-board listings stop after May and why descriptions go missing more often from June on, from EDA_SDA_2), then record them in your change log (`JACOB.md` or `JESSE.md`).

## Later

- **Backup:** an encrypted Time Machine backup to Jacob's removable drive when it's available, **excluding every `data/` folder** (the data stays in S3 and approved locations only).
- **predictleads-hackathon:** its `.venv` was removed 2026-10-10 to free space; rebuild it (`python -m venv .venv && pip install -r requirements.txt`) before working there.

- **Profiles is frozen** (refrozen 2026-10-10). Reopen only if needed: a pool for moving up within the same role (training, then a senior or manager role; levels are already tagged), or more prior-career keywords ("other" is 115 of 706; 4a.2e lists the common titles). Any change means a rerun, then the ML notebook reloads `modeling_table.pkl`.
- **Run log:** built (`src/run_log.py`, a CSV that refuses listing text); call `log_run` from each modeling notebook.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily exports to observe closures directly; a one-page system card.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, `.env`, or anything in `reports/`.
- After a kernel restart, run the notebook from the top.
- **After any change to `src/` (rules, mappings, helpers), restart the kernel before rerunning:** Python keeps the old version loaded until a restart, so Run All alone uses the old code.
- Follow the working standards in `STANDARDS.md` and the course-template baseline in every notebook.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record every decision, with its reasoning, in your change log (`JACOB.md` or `JESSE.md`).
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.
- Regenerate a notebook's `docs/` export after editing it (`python -m src.export_markdown notebooks/<name>.ipynb`).

## Finish

- **Polish all seven notebooks (EDA_SDA_1–4, TimeSeries, ML_Fit, ML_Rank) and Profiles so they read as one unit:** consistent titles, numbering, terms and formatting; each opening links back to the previous notebook's conclusion; imports at the top (course standard); no repeated code; every course-template requirement checked. EDA_SDA_1 is already trimmed, with its data loading in `src/pl_data.py`.
- Permanently delete every copy of the data, **including the GPU PC's data folder if it was set up**: `data/` (Parquet, pickle, zip and `.jsonl.zst` files, including `data/flat/`, `data/clean/`, `data/external/` and `data/profiles/`), `reports/`, the copy in `~/dev/predictleads-hackathon/data/`, downloads (including `~/Downloads/1_corpus_and_signals.zip` if it's still there), and any S3 copies.
- Confirm the deletion in writing if asked.
