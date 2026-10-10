# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> Each notebook's full plan is in the notebook itself; how we work is in [`STANDARDS.md`](STANDARDS.md).

---

## Now

- **Full dataset, full scale for every notebook** (decided 2026-10-10; stream test: ≈13.4M listings, ≈1.0M in scope, ≈40 minutes, ≈8.4 GB):
  1. **Jacob:** finish the first full stream (`PL/bin/python -m src.stream_full`), paste the summary, delete `data/flat/stream_test/`. Its scope parts (all columns) are kept.
  3. **Jacob:** after step 1, run the second pass: `PL/bin/python -m src.stream_full --rows` (≈40 minutes, resumable): EDA_SDA_1's cleaned row for every English US listing, typed Parquet, no text (`data/flat/all_fields_rows_full/`). Then `data/flat/all_fields_lean_full/` can be deleted once EDA_SDA_2 reads the rows table.
  4. **Claude:** rework EDA_SDA_1 to read columns, not the whole table (13.4M rows in 8 GB); repost matching on compact fingerprints; display cells on a sample. Then Jacob reruns it, and Claude updates notes, reports and the deck.
  5. **Claude:** EDA_SDA_2 starts with a **feature screening** over ten families of candidates, each through the same checks (coverage overall and by source and month; leakage: known at first sighting?; signal on closing speed within the standard confounders; redundancy: Cramér's V, mutual information; stability by month and within companies), giving one keep/drop table that every later notebook starts from. The families: raw listing fields (including the newly saved `normalized_title`, has-recruiter flag, `last_processed_at` for a leakage audit, currency, translated flag), the listing's own history, **company history from earlier listings** (past closing speed of its closed listings, entry-level share, repost and ghost rates, pay-disclosure habit, hiring trend, roles open), **the market at first sighting** (open listings in the same bucket and state that day, new listings that week, day and month first seen), **pay in context** (percentile within bucket and state, range width, hourly vs. salaried), text (EDA_SDA_3), company facts, location (metro area, a pay-transparency-law flag per state), **O*NET Job Zone**, and BLS growth. A gradient-boosting permutation-importance check is a cross-check at the start of ML_Rank, not the decision.
  6. **Claude:** EDA_SDA_2 reads the full scope (≈1M rows, cleaned part by part into a typed table), company volume and the 1c summary from the all-fields table (`load_lean`), memory-safe heavy cells (3.7d location records from the full parts). Jacob reruns; Claude updates notes, reports and the deck.
  7. **Claude:** EDA_SDA_3 for full scale: text read part by part, regex features over every listing, spaCy lemmas on a large stratified sample (about 150,000), cache in parts. Hold EDA_SDA_3's first run until then.
  8. **New analysis the full data opens** (added to each notebook's plan): real market sizes per role (openings, new per week), role buckets by month × source and by state, a company watch list (employers posting entry-level roles often), rare signals (cross-posts, evergreen wording, licenses, staffing agencies), full-scale daily series for TimeSeries, more training data and a proper test month for ML_Rank.
- **Jacob (from 2026-10-10):** upload the slide deck (`reports/Team update 2026-10-10.pptx`, local only, already built; it replaces the 2026-10-09 version) to Google Slides, then delete the old 2026-10-09 deck; tell the team (below). Data housekeeping and laptop cleanup done 2026-10-10 (53+ GB free; raw caches removed, all flattened tables, clean tables, profiles and resume CSVs checked intact).
- **Jacob: two quick checks after the cleanup:** confirm the venv still works (`PL/bin/python -c "import pandas, sklearn, torch"`), and turn on automatic output stripping: `pip install nbstripout`, `nbstripout --install`, pin `nbstripout` in `requirements.txt`, commit, and tell the team (below). The three notebooks show as modified only because of outputs; their code matches the last commit.
- **Tell the team** to pull, rerun `pip install -r requirements.txt` (and on a Mac `brew install libomp`), run `python -m spacy download en_core_web_sm`, switch their `.env` to `S3_PATH` (README section 4), and run `nbstripout --install` once `nbstripout` is in `requirements.txt`.

## Next

- **EDA_SDA_3 (text features; no matching models), in progress:**
  - **Sections 1–2 written** (setup, text loaded from the saved parts, HTML to structured text, boilerplate removal, lemmas; cached in `data/clean/eda3_text.pkl`): needs a run, then notes. Section 3 (years and degree rules) is next.
  - **Hand-checking:** decide who marks the validation samples (section 5), about 100–150 matches per rule, locally.
  - Validate the keyword flags: a teammate checks a random sample of matches locally (career-change flags, `urgency`, `mentions_remote`); precision and recall against EDA_SDA_1's rules.
  - Hard-blocker rules per role bucket (years-asked threshold, required credentials), hidden in the app with a count of what each hides. The "asks 5+ years" preview would hide 31% of open listings (EDA_SDA_2 3.11): check required vs. preferred and the year counts first. The degree flag fires on only 3–6% of listings: widen and validate its pattern ("Bachelor's in …") before any degree rule.
  - Evergreen wording (feeds the ghost-listing flag), required vs. preferred qualifications, **skills mapped to O*NET** (the input to `ML_Fit`'s skills-based matching).
  - **New text rules (added 2026-10-10):** security clearance required, "no visa sponsorship", travel percentage, shift or on-call work, contract-to-hire. Clearance and sponsorship are hard blockers for some career changers (per applicant in the app).
  - Save the scope table with text features for EDA_SDA_4 and `ML_Fit`.
- **EDA_SDA_4 (statistics):**
  - Descriptive statistics by group; Pearson + Spearman, Cramér's V, point-biserial; VIF.
  - Group interactions, including by role bucket, and checks for confounding (Simpson's paradox).
  - Adjusted check: a Cox survival regression with every standard confounder (company volume, source and platform, staffing agency, role bucket, seniority, month, company type, has a description), company-clustered errors, hazard ratios with 95% intervals and a proportional-hazards check.
  - Welch's t-test plus Mann–Whitney (Wilcoxon rank-sum), chi-square and log-rank, with effect sizes and Benjamini–Hochberg; this is where EDA_SDA_1's and EDA_SDA_2's descriptive gaps get significance tests.
  - Company-clustered bootstrap intervals; survival checks (censoring, left truncation, competing risks).
  - **In-depth feature analysis (added 2026-10-10, full data):** (1) **stability over months**: does each feature's effect hold in every month, not just overall (ML_Rank trains on April–July, tests on September); (2) **within-company comparisons**: does it still matter between listings from the same company; (3) **effects by role bucket**; (4) **leakage audit**, one row per feature: known when the listing first appears?; (5) **drift checks**: which distributions shift over time, and do they line up with the collection changes (job boards after May, descriptions from June); (6) **information beyond the confounders** (mutual information or likelihood-ratio tests). Priority: 1, 2 and 4 first.
  - Record each feature kept, dropped or combined; save the modeling-ready table.
- **TimeSeries:**
  - Daily and weekly series from `first_seen` and from `posted` (Central European dates), to separate crawler from employer rhythm.
  - Rolling stats, decomposition (STL), ACF/PACF, ADF and KPSS.
  - Lag and rolling features with `.shift()`; a time-ordered split, test used once.
  - Models: `DummyRegressor`, linear regression, random forest, ARIMA / SARIMA; RMSE and MAE.
  - Survival by month first seen, and a "check every N days" cadence.
- **ML_Fit (fit, role discovery, the compass):**
  - Build each career changer's pre-switch text from the Profiles timeline (titles and programs before the switch; no undated skills).
  - The ladder: keyword overlap → TF-IDF cosine → BGE embeddings (small or base, local) → RRF hybrid → skills-based matching on O*NET skills → a trained layer (Option B) on real career moves (candidates include a small neural network, `MLPClassifier`); stretch: fine-tune BGE on career moves if the trained layer plateaus.
  - Score each step on hit rate@3/@5 for the role people actually moved into, per pool and overall, against a title-keyword baseline; measure old-field bias; exclude the current role for within-tech moves; folds grouped by person.
  - SHAP, including each listing's top fit features for the detail view's rating breakdown; hand fit scores and their breakdown to `ML_Rank`.
  - **Assemble the compass (last section):** Profiles' paths with people counts, each role's 2026 market and growth (EDA_SDA_2 3.12), and the Ideas tab's inputs (skills to build, options and time to switch, pay increase vs. the current field from 1c, growth); save the compass tables for the app.
- **ML_Rank (the priority queue):**
  - Write the hypotheses with thresholds **before** any modeling.
  - Targets: closes within N days, days open, and the survival target.
  - Split by time (train April–July, validate August, test September), folds grouped by company; test used once.
  - Baselines: `DummyClassifier`, newest first, title keyword search.
  - Models: logistic regression, trees, random forest, LightGBM, CatBoost, XGBoost; Cox and survival models; LambdaRank.
  - Leakage: counts from earlier listings only; a date-picker model trained only on what was known by its date; check that `has_desc` and the text flags are known at first sighting (EDA_SDA_2 3.9) before using them.
  - **The priority queue:** combine the fit score, chance still open and apply-by, accessibility and pay in one formula; weights start from EDA_SDA_4's adjusted model and are tuned on the validation month; only features that survived EDA_SDA_4.
  - Evaluation (ROC-AUC, F1, C-index, precision@k, NDCG); the queue against newest-first: of each top 10, how many stay open long enough, are reachable and are in the role the person actually moved into; SHAP and the per-listing rating breakdown for the detail view (each factor's rating and share of the score, top features behind chance still open), sensitivity checks, solution report.
- **Job growth expectations** (plan: [`docs/job_growth.md`](docs/job_growth.md)): get PredictLeads' written consent to combine their data with BLS data (agreement 4.3), check the terms of use, download the **BLS 2025–2035** occupation table into `data/external/`, join on occupation code, roll up to role buckets weighted by 2025 employment (IT project management takes 15-1299's growth as a flagged proxy), and convert to the time frame the team picks (at most 5 years). Add the expected-growth column to the compass (EDA_SDA_2 3.12).
- **Date picker (as-of date):** a helper in `src/` that returns the listings live on a date with features computed as of that date; check it on July 1 vs. September 1, 2026 (counts only; EDA_SDA_1 8b shows dates before July are skewed young). The model for a date is trained only on what was known by then; the demo can start with test-period dates.
- **Ideas tab** (`APPLICATION.md`, a separate tab): skills to build (EDA_SDA_3 skills vs. a résumé, in `ML_Fit`), career options and time to switch (Profiles), pay increase (EDA_SDA_2 1c current-field pay vs. 3.12 role pay, hourly), expected growth, and stepping-stone roles; built from `ML_Fit`'s compass tables. Time frames stay vague until the team decides.
- **Decide the time frames** for career options and job growth in the compass (team; growth at most 5 years), and whether to adopt the proposed growth extensions (`docs/job_growth.md` section 4: error band, rate scenarios, listing momentum, horizon-weighted blend, state level).
- **O*NET (in scope, small):** check the terms of use, then download only the **Career Changers Matrix** (latest release), the **skills** for computer and mathematical occupations plus the team's background occupations, and the **Job Zones** (O*NET's level of education and experience per occupation: an accessibility prior for the feature screening). Listings already carry O*NET codes, so they join directly.
- **Retraining compass app view** (`APPLICATION.md`, `COMPASS.md`); its 2026 side is ready (EDA_SDA_2 3.12).
- **App MVP** (stage 1 in `APPLICATION.md`): list view with search, sort and filters; detail view with apply link; stars.
- **Write up the analysis** and **choose the final format** (write-up, notebook demo, or app).

## Blocked

- **Team:**
  - Each teammate writes down their background (current field and roles; no personal details), to fix the source backgrounds.
  - Assign owners for the notebooks above, the app, and the pitch.
  - Set a meeting cadence and a place for async discussion.
  - Confirm the final due date and deliverables.
- **App:** answer the open questions in `APPLICATION.md`, section 8.
- **Project advisor:** get answers to our questions (scope, deliverables, AI tools and data, applicant profiles, external data, hosting, presenting results, data deletion), then record them in `docs/decisions.md`. Get the AI-and-data answers in writing; the agent stage of the app depends on them, and the answer on collecting career-changer profiles blocks the volunteer profiles.
- **Applicant profiles** (after the advisor answers): follow `docs/fit_analysis.md`. Assign owners for its task table, draft the consent form, recruit career changers, convert their exports, and build the synthetic generator in `src/`.
- **PredictLeads:** get answers to our data questions (field definitions, crawl timing, `status`, sources, sampling, duplicates, older snapshots, `seniority`; why job-board listings stop after May and why descriptions go missing more often from June on, from EDA_SDA_2), then record them in `docs/decisions.md`.

## Later

- **Backup:** an encrypted Time Machine backup to Jacob's removable drive when it's available, **excluding every `data/` folder** (the data stays in S3 and approved locations only).
- **predictleads-hackathon:** its `.venv` was removed 2026-10-10 to free space; rebuild it (`python -m venv .venv && pip install -r requirements.txt`) before working there.

- **Profiles is frozen** (refrozen 2026-10-10). Reopen only if needed: a pool for moving up within the same role (training, then a senior or manager role; levels are already tagged), or more prior-career keywords ("other" is 115 of 706; 4a.2e lists the common titles). Any change means a rerun, then the ML notebook reloads `modeling_table.pkl`.
- **App:** transferable-fit stage (the career changer's background profile), priority model with adjustable weights, MCP servers, then agents (see `APPLICATION.md`).
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never listing text.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily snapshots to observe closures directly; a one-page system card.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, `.env`, or anything in `reports/`.
- After a kernel restart, run the notebook from the top.
- **After any change to `src/` (rules, mappings, helpers), restart the kernel before rerunning:** Python keeps the old version loaded until a restart, so Run All alone uses the old code.
- Follow the working standards in `STANDARDS.md` and the course-template baseline in every notebook.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record every decision, with its reasoning, in `docs/decisions.md`.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.
- Regenerate a notebook's `docs/` export after editing it (`python -m src.export_markdown notebooks/<name>.ipynb`).

## Finish

- **Polish all seven notebooks (EDA_SDA_1–4, TimeSeries, ML_Fit, ML_Rank) and Profiles so they read as one unit:** consistent titles, numbering, terms and formatting; each opening links back to the previous notebook's conclusion; imports at the top (course standard); no repeated code; every course-template requirement checked. EDA_SDA_1 is already trimmed, with its data loading in `src/pl_data.py`.
- Permanently delete every copy of the data: `data/` (Parquet, pickle, zip and `.jsonl.zst` files, including `data/flat/`, `data/clean/`, `data/external/` and `data/profiles/`), `reports/`, the copy in `~/dev/predictleads-hackathon/data/`, downloads (including `~/Downloads/1_corpus_and_signals.zip` if it's still there), and any S3 copies.
- Confirm the deletion in writing if asked.
