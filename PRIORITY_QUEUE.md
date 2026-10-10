# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> Each notebook's full plan is in the notebook itself; how we work is in [`HOW_WE_WORK.md`](HOW_WE_WORK.md).

---

## Now

- **Jacob (from 2026-10-10):** upload the slide deck (`reports/Team update 2026-10-09.pptx`, local only, already built) to Google Slides; do the data housekeeping and tell the team (both below).
- **EDA_SDA_2 (non-text), in progress.** Done (2026-10-10): setup and the niche cut (14,019 listings), 2b cleaning and filter fields, 3.1 ghosts and the 90-day line, 3.2 source, 3.3 freshness and crawl timing, 3.4 pay disclosure, 3.7 blind spots and state, 3.5 company volume and copies, 3.6 role buckets, 3.8 company facts and no-country listings, 3.12 the compass's 2026 side. Ten charts, matching EDA_SDA_1. Still to write:
  - **3.9 Missing data:** written (per-field classes, by group, missing together, by outcome, with a chart); needs a run, then notes.
  - **3.10 Outliers:** IQR and boxplots for days open, words, hourly pay, times listed.
  - **3.11 Hide what a career changer can't get:** manager-level listings per applicant, text blockers after EDA_SDA_3; report what each rule hides.
  - **4 Final table:** defend each decision, apply them at the end, save for EDA_SDA_3. **5 Conclusion.**
- **Data housekeeping:** delete `data/external/resume_structured/archive.zip` once you no longer want a backup (the resume data is unzipped; the notebook re-extracts from it only if the CSVs go missing), the old `data/profiles/*_v9*` files (replaced by `modeling_table.pkl` and `retrained_candidates.csv`), the old lean-table cache (`data/clean/job_openings_lean_*.pkl` without `_v2`), the whole `data/profiles/review/` folder (old round_01 files; nothing reads them), and any leftover `*_flat_*.pkl` files.
- **Tell the team** to pull, rerun `pip install -r requirements.txt` (and on a Mac `brew install libomp`), run `python -m spacy download en_core_web_sm`, and switch their `.env` to `S3_PATH` (README section 4).

## Next

- **EDA_SDA_3 (text features; no matching models):**
  - Description cleanup moved here from EDA_SDA_2 2b (HTML, equal-opportunity and benefits boilerplate → `desc_clean`), loading the text from the saved parts; normalize, stopwords, lemmatize.
  - Validate the keyword flags: a teammate checks a random sample of matches locally (career-change flags, `urgency`, `mentions_remote`); precision and recall against EDA_SDA_1's rules.
  - Hard-blocker rules per role bucket (years-asked threshold, required credentials), hidden in the app with a count of what each hides.
  - Evergreen wording (feeds the ghost-listing flag), required vs. preferred qualifications, **skills mapped to O*NET** (the input to `ML_Fit`'s skills-based matching).
  - Save the niche table with text features for EDA_SDA_4 and `ML_Fit`.
- **EDA_SDA_4 (statistics):**
  - Descriptive statistics by group; Pearson + Spearman, Cramér's V, point-biserial; VIF.
  - Group interactions, including by role bucket, and checks for confounding (Simpson's paradox).
  - Adjusted check: a Cox survival regression with every standard confounder (company volume, source and platform, staffing agency, role bucket, seniority, month), company-clustered errors, hazard ratios with 95% intervals and a proportional-hazards check.
  - Welch's t-test plus Mann–Whitney (Wilcoxon rank-sum), chi-square and log-rank, with effect sizes and Benjamini–Hochberg; this is where EDA_SDA_1's and EDA_SDA_2's descriptive gaps get significance tests.
  - Company-clustered bootstrap intervals; survival checks (censoring, left truncation, competing risks).
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
  - SHAP; hand fit scores to `ML_Rank`.
  - **Assemble the compass (last section):** Profiles' paths with people counts, each role's 2026 market and growth (EDA_SDA_2 3.12), and the Ideas tab's inputs (skills to build, options and time to switch, pay increase vs. the current field from 1c, growth); save the compass tables for the app.
- **ML_Rank (the priority queue):**
  - Write the hypotheses with thresholds **before** any modeling.
  - Targets: closes within N days, days open, and the survival target.
  - Split by time (train April–July, validate August, test September), folds grouped by company; test used once.
  - Baselines: `DummyClassifier`, newest first, title keyword search.
  - Models: logistic regression, trees, random forest, LightGBM, CatBoost, XGBoost; Cox and survival models; LambdaRank.
  - Leakage: counts from earlier listings only; a date-picker model trained only on what was known by its date.
  - **The priority queue:** combine the fit score, chance still open and apply-by, accessibility and pay in one formula; weights start from EDA_SDA_4's adjusted model and are tuned on the validation month; only features that survived EDA_SDA_4.
  - Evaluation (ROC-AUC, F1, C-index, precision@k, NDCG); the queue against newest-first: of each top 10, how many stay open long enough, are reachable and are in the role the person actually moved into; SHAP, sensitivity checks, solution report.
- **Job growth expectations:** check the terms of use, download the official 10-year occupational projections into `data/external/`, map them to role buckets by occupation code, and convert to the time frame the team picks (at most 5 years). Add an expected-growth column to the compass (EDA_SDA_2 3.12).
- **Date picker (as-of date):** a helper in `src/` that returns the listings live on a date with features computed as of that date; check it on July 1 vs. September 1, 2026 (counts only; EDA_SDA_1 8b shows dates before July are skewed young). The model for a date is trained only on what was known by then; the demo can start with test-period dates.
- **Ideas tab** (`APPLICATION.md`, a separate tab): skills to build (EDA_SDA_3 skills vs. a résumé, in `ML_Fit`), career options and time to switch (Profiles), pay increase (EDA_SDA_2 1c current-field pay vs. 3.12 role pay, hourly), expected growth, and stepping-stone roles; built from `ML_Fit`'s compass tables. Time frames stay vague until the team decides.
- **Decide the time frames** for career options and job growth in the compass (team; growth at most 5 years).
- **O*NET (in scope, small):** check the terms of use, then download only the **Career Changers Matrix** (latest release) and the **skills** for computer and mathematical occupations plus the team's background occupations. Listings already carry O*NET codes, so they join directly.
- **Retraining compass app view** (`APPLICATION.md`), after EDA_SDA_2 3.12.
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
- **PredictLeads:** get answers to our data questions (field definitions, crawl timing, `status`, sources, sampling, duplicates, older snapshots, `seniority`), then record them in `docs/decisions.md`.

## Later

- **Profiles is frozen** (2026-10-09). Reopen only if needed: a pool for moving up within the same role (training, then a senior or manager role; levels are already tagged), or more prior-career keywords ("other" is 115 of 706; 4a.2e lists the common titles). Any change means a rerun, then the ML notebook reloads `modeling_table.pkl`.
- **App:** transferable-fit stage (the career changer's background profile), priority model with adjustable weights, MCP servers, then agents (see `APPLICATION.md`).
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never listing text.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily snapshots to observe closures directly; a one-page system card.
- **If the niche is too small or the sample biased,** stream more of the full file with filters and random sampling.
- **Fix the README's venv name:** the README says `.venv` while your venv is `PL`.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, `.env`, or anything in `reports/`.
- After a kernel restart, run the notebook from the top.
- **After any change to `src/` (rules, mappings, helpers), restart the kernel before rerunning:** Python keeps the old version loaded until a restart, so Run All alone uses the old code.
- Follow the working standards in `HOW_WE_WORK.md` and the course-template baseline in every notebook.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record every decision, with its reasoning, in `docs/decisions.md`.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.
- Regenerate a notebook's `docs/` export after editing it (`python -m src.export_markdown notebooks/<name>.ipynb`).

## Finish

- **Polish all seven notebooks (EDA_SDA_1–4, TimeSeries, ML_Fit, ML_Rank) and Profiles so they read as one unit:** consistent titles, numbering, terms and formatting; each opening links back to the previous notebook's conclusion; imports at the top (course standard); no repeated code; every course-template requirement checked. EDA_SDA_1 is already trimmed, with its data loading in `src/pl_data.py`.
- Permanently delete every copy of the data: `data/` (Parquet, pickle, zip and `.jsonl.zst` files, including `data/flat/`, `data/clean/`, `data/external/` and `data/profiles/`), `reports/`, downloads, and any S3 copies.
- Confirm the deletion in writing if asked.
