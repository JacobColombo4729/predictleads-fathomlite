# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> Each notebook's full plan is in the notebook itself; how we work is in [`HOW_WE_WORK.md`](HOW_WE_WORK.md).

---

## Now

- **Slide deck for the team, due Friday 2026-10-09** (reminder): our findings at each step, how we made the major decisions, and a final pitch, framed around career changers. The 5-minute report (`python -m src.export_markdown --report notebooks/EDA_SDA_1.ipynb`, local only) is a starting point.
- **Fill in EDA_SDA_1's "*To fill in after rerunning*" notes** from the rerun (6.1, Q1a, Q1g, Q3a, 12 and the executive summary's career-change row), and the matching row in `PROJECT_OUTLINE.md`. Then regenerate `docs/EDA_SDA_1.md`.
- **Profiles review round (rules v5):** rerun `notebooks/Profiles.ipynb` (File: Revert File first), then in 4b/4c review 30 people from each retrained pool; fill in `verdict` and `reason`, run section 5 and share its counts. Target: 90%+ good per pool. Then update slide 11 with the new count and the measured precision.
- **Check the seniority labels** (EDA_SDA_1 Q3b) and add any entry-level ones besides `junior` to `ENTRY_LEVELS` in 6.7a. EDA_SDA_2 needs the list to hide manager-level listings.
- **Give Claude instructions for the additional data cleaning** (reminder).
- **Answer Claude's open suggestions:**
  - Reword the 60.7% as "up to 60.7%" in the summaries, and add a domain check to EDA_SDA_2 (3.2) to refine it.
  - Add a caveat that `urgency` and `mentions_remote` are keyword rules with suspected false positives ("urgent care"), and a validation step to EDA_SDA_3.
- **EDA_SDA_2 (non-text), the next notebook:**
  - **Setup:** load EDA_SDA_1's final table (171,143 listings).
  - **3.1 Ghost listings vs. struggling employers:** describe the long-open listings, measure each signal (relisted, company still posting, source, `posted` vs. `first_seen`), and combine them into a "likely ghost listing" flag.
  - **3.2 Source:** source within company-volume bands; job-board copies of company-site listings; platform sizes. Add the domain check that refines the 60.7%.
  - **3.3 Freshness:** the late-found tail, survival of late-found vs. prompt listings, with vs. without `posted`.
  - **3.4 Pay disclosed:** by state, hourly vs. salaried and seniority, within company-volume bands.
  - **3.5 Reposts and company volume:** how strongly reposts count, and whether high volume means more competition or more openings (and more entry-level listings).
  - **3.6 The niche, computer and mathematical by role bucket:** map `onet_code` to buckets (report unmapped codes); size each bucket before and after hiding manager-level listings; count retrained switches per bucket from `Profiles.ipynb`; run the depth check; keep buckets that pass on both sides (proposed: 30+ retrained switches). Fix the threshold before looking at results.
  - **3.7 Blind spots:** the 2-day live tolerance, part-time annualized pay, the seniority labels, and state from `location_data`.
  - **3.8 Join the companies table** (left join, keys checked).
  - **3.9 Missing data:** follow the per-field plan already in the notebook; by-group missingness and `isnull().corr()`.
  - **3.10 Outliers:** IQR and boxplots for days open, words, salary, times listed.
  - **3.11 Hide what a career changer can't get:** manager level and above; report what each rule hides; accessibility checks within the niche.
  - **4 Final table:** defend each decision, apply them all at the end, save for EDA_SDA_3. **5 Conclusion.**
- **Pin the packages the next notebooks need** in `requirements.txt` (`lifelines`, `statsmodels`, `nltk`, `spacy`, `lightgbm`, `catboost`, `xgboost`, `shap`), then tell the team.
- **Data housekeeping:** delete `data/archive.zip` (the resume data is unzipped), the old lean-table cache (`data/clean/job_openings_lean_*.pkl` without `_v2`), the blank review files in `data/profiles/review/`, and any leftover `*_flat_*.pkl` files.
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
  - Role discovery and transferable fit (Option B) trained on `Profiles.ipynb`'s retrained career changers, folds grouped by person; role-discovery hit rate in the top 3 and top 5.
  - Evaluation (ROC-AUC, F1, C-index, precision@k, NDCG), SHAP, bootstrap business translation, sensitivity checks, solution report.
- **Profiles (career switches):**
  - Fix unreadable job dates (37% of people): count date formats without showing dates, then extend the parser.
- **O*NET (in scope, small):** check the terms of use, then download only the **Career Changers Matrix** (latest release) and the **skills** for computer and mathematical occupations plus the team's background occupations. Listings already carry O*NET codes, so they join directly.
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
- **External data:** BLS JOLTS and BLS wage statistics. Check each one's terms of use.
- **Stretch, only if time allows: CPS occupation transitions** (IPUMS CPS ASEC through 2025): a robustness check on the resume dataset's background mix, with reweighting. A mini-project that no feature depends on, and it can't see retraining. Until then, the dataset's age is stated as a limitation.
- **Optional context: BLS occupational separations** (2024–34): how often people leave the team's background occupations, for the pitch. Small; drop if short on time.
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never listing text.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily snapshots to observe closures directly; a one-page system card.
- **If the niche is too small or the sample biased,** stream more of the full file with filters and random sampling.
- **Move the data loaders into `src/`:** `fetch`, `stream_jsonl`, `load_cached`, `flatten_to_parts`, `map_parts`, `load_flat`.
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

- **Polish all six notebooks (EDA_SDA_1–4, TimeSeries, ML) and Profiles so they read as one unit:** consistent titles, numbering, terms and formatting; each opening links back to the previous notebook's conclusion; imports at the top (course standard); no repeated code; every course-template requirement checked. EDA_SDA_1 is already trimmed; its data loading still moves to `src/`.
- Permanently delete every copy of the data: `data/` (Parquet, pickle, zip and `.jsonl.zst` files, including `data/flat/`, `data/clean/`, `data/external/` and `data/profiles/`), `reports/`, downloads, and any S3 copies.
- Confirm the deletion in writing if asked.
