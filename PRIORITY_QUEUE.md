# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> Each notebook's full plan is in the notebook itself; how we work is in [`HOW_WE_WORK.md`](HOW_WE_WORK.md).

---

## Now

- **Slide deck for the team, due Friday 2026-10-09:** `reports/Team update 2026-10-09.pptx` (local only) is built; upload it to Google Slides.
- **Give Claude instructions for the additional data cleaning** (reminder).
- **EDA_SDA_2 (non-text), the next notebook:**
  - **Setup:** load EDA_SDA_1's final table (182,367 listings).
  - **3.1 Ghost listings vs. struggling employers:** describe the long-open listings, measure each signal (relisted, company still posting, source, `posted` vs. `first_seen`), and combine them into a "likely ghost listing" flag.
  - **3.2 Source:** source within company-volume bands; job-board copies of company-site listings; platform sizes. Add the domain check that refines the 60.7%.
  - **3.3 Freshness:** the late-found tail, survival of late-found vs. prompt listings, with vs. without `posted`.
  - **3.4 Pay disclosed:** by state, hourly vs. salaried and seniority, within company-volume bands.
  - **3.5 Reposts and company volume:** recheck EDA_SDA_1's 14 vs. 32 days with volume counted from earlier listings only (the original used the whole six months, which includes the future); how strongly reposts count, and whether high volume means more competition or more openings (and more entry-level listings).
  - **3.6 The niche, computer and mathematical by role bucket:** map `onet_code` to buckets (report unmapped codes); size each bucket with and without manager-level listings (hidden per applicant, so the niche keeps them); count career changers per bucket from `Profiles.ipynb`, all three pools, split by pool; run the depth check; keep buckets that pass on both sides (30+ career changers in all pools; retrained-only would drop data and analytics).
  - **3.7 Blind spots:** the 2-day live tolerance, part-time annualized pay, the seniority labels, and state from `location_data`.
  - **3.8 Join the companies table** (left join, keys checked).
  - **3.9 Missing data:** follow the per-field plan already in the notebook; by-group missingness and `isnull().corr()`.
  - **3.10 Outliers:** IQR and boxplots for days open, words, salary, times listed.
  - **3.11 Hide what a career changer can't get:** manager-level listings per applicant (only for applicants without management experience), text blockers after EDA_SDA_3; report what each rule hides; accessibility checks within the niche.
  - **3.12 The compass, 2026 side:** each role's openings, pay, closing speed, entry-level share and expected growth, joined to Profiles' three path tables (prior career, training, within tech), with people counts and thin paths marked.
  - **4 Final table:** defend each decision, apply them all at the end, save for EDA_SDA_3. **5 Conclusion.**
- **Rerun EDA_SDA_1 from the top once** to confirm the new loader (`src/pl_data.py`) works with the real data and cache; restart the kernel first.
- **Pin the packages the next notebooks need** in `requirements.txt` (`lifelines`, `statsmodels`, `nltk`, `spacy`, `lightgbm`, `catboost`, `xgboost`, `shap`), then tell the team.
- **Data housekeeping:** delete `data/external/resume_structured/archive.zip` once you no longer want a backup (the resume data is unzipped; the notebook re-extracts from it only if the CSVs go missing), the old `data/profiles/*_v9` files, the old lean-table cache (`data/clean/job_openings_lean_*.pkl` without `_v2`), the blank review files in `data/profiles/review/`, and any leftover `*_flat_*.pkl` files.
- **Tell the team** to pull, rerun `pip install -r requirements.txt`, and switch their `.env` to `S3_PATH`.

## Next

- **EDA_SDA_3 (NLP):**
  - Preprocessing: normalize, stopwords, lemmatize; TF-IDF fit on the training months only; boilerplate removal.
  - Validate the keyword flags: a teammate checks a random sample of matches locally (career-change flags, `urgency`, `mentions_remote`).
  - Hard-blocker rules per role bucket (years-asked threshold, required licenses); apply them in this notebook's final table, with a count of what each hides.
  - Evergreen wording (feeds the ghost-listing flag), required vs. preferred qualifications, skills extraction mapped to O*NET skills.
  - Role similarity: how close each role bucket is to the team's backgrounds (TF-IDF or embeddings, plus O*NET).
  - Course evaluation with a `DummyClassifier` baseline; save the niche table with text features.
- **EDA_SDA_4 (statistics):**
  - Descriptive statistics by group; Pearson + Spearman, Cramér's V, point-biserial; VIF.
  - Group interactions, including by role bucket, and checks for confounding (Simpson's paradox).
  - Welch's t-test plus Mann–Whitney, chi-square and log-rank, with effect sizes and Benjamini–Hochberg.
  - Company-clustered bootstrap intervals; survival checks (censoring, left truncation, competing risks).
  - Record each feature kept, dropped or combined; save the modeling-ready table.
- **TimeSeries:**
  - Daily and weekly series from `first_seen` and from `posted` (Central European dates), to separate crawler from employer rhythm.
  - Rolling stats, decomposition (STL), ACF/PACF, ADF and KPSS.
  - Lag and rolling features with `.shift()`; a time-ordered split, test used once.
  - Models: `DummyRegressor`, linear regression, random forest, ARIMA / SARIMA; RMSE and MAE.
  - Survival by month first seen, and a "check every N days" cadence.
- **ML:**
  - Write the hypotheses with thresholds **before** any modeling.
  - Targets: closes within N days, days open, and the survival target.
  - Split by time (train April–July, validate August, test September), folds grouped by company; test used once.
  - Baselines: `DummyClassifier`, newest first, title keyword search.
  - Models: logistic regression, trees, random forest, LightGBM, CatBoost, XGBoost; Cox and survival models; LambdaRank.
  - Role discovery and transferable fit (Option B) trained on `Profiles.ipynb`'s modeling table, folds grouped by person; role-discovery hit rate in the top 3 and top 5, **reported per pool** (within tech is 72% of career changers) and overall.
  - Leakage: counts from earlier listings only; a date-picker model trained only on what was known by its date; compass counts rebuilt inside each training fold.
  - Evaluation (ROC-AUC, F1, C-index, precision@k, NDCG), SHAP, bootstrap business translation, sensitivity checks, solution report.
- **Job growth expectations:** check the terms of use, download the official 10-year occupational projections into `data/external/`, map them to role buckets by occupation code, and convert to the time frame the team picks. Add an expected-growth column to the compass (EDA_SDA_2 3.12).
- **Date picker (as-of date):** a helper in `src/` that returns the listings live on a date with features computed as of that date; check it on June 1 vs. August 1, 2026 (counts only). The model for a date is trained only on what was known by then; the demo can start with test-period dates.
- **Widen the background-area keywords (Profiles):** "other" is 22% of retrained people (157 of 706) and the biggest row of the prior-career table. List the most common unmatched titles (counts only, shared by 5+ people) and add keywords.
- **Compass for moving up (Profiles):** the rules only count moves into a *different* role, so "trained, then reached a senior or manager role in the same role" isn't captured. Add a pool or label for it (training dated before the level change; the level at each job is already tagged), then a training × level table next to the compass.
- **Compass ideas** (`APPLICATION.md`): skills to build (EDA_SDA_3 skills vs. a résumé), career options and time to switch (Profiles), pay increase (2026 listing pay by occupation), expected growth, and stepping-stone roles (level at the switch in Profiles).
- **Decide the time frames** for career options and job growth in the compass (team).
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

- **App:** transferable-fit stage (the career changer's background profile), priority model with adjustable weights, MCP servers, then agents (see `APPLICATION.md`).
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never listing text.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily snapshots to observe closures directly; a one-page system card.
- **If the niche is too small or the sample biased,** stream more of the full file with filters and random sampling.
- **Fix the README's venv name:** the README says `.venv` while your venv is `PL`.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, `.env`, or anything in `reports/`.
- After a kernel restart, run the notebook from the top.
- Follow the working standards in `HOW_WE_WORK.md` and the course-template baseline in every notebook.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record every decision, with its reasoning, in `docs/decisions.md`.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.
- Regenerate a notebook's `docs/` export after editing it (`python -m src.export_markdown notebooks/<name>.ipynb`).

## Finish

- **Polish all six notebooks (EDA_SDA_1–4, TimeSeries, ML) and Profiles so they read as one unit:** consistent titles, numbering, terms and formatting; each opening links back to the previous notebook's conclusion; imports at the top (course standard); no repeated code; every course-template requirement checked. EDA_SDA_1 is already trimmed, with its data loading in `src/pl_data.py`.
- Permanently delete every copy of the data: `data/` (Parquet, pickle, zip and `.jsonl.zst` files, including `data/flat/`, `data/clean/`, `data/external/` and `data/profiles/`), `reports/`, downloads, and any S3 copies.
- Confirm the deletion in writing if asked.
