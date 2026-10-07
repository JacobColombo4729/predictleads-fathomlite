# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> The reasoning behind each task is in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## Now

- **Rerun the EDA notebook from step 6.0** to render the new charts and the `contract_types` cleaning fix, then check that each "What the data shows" note and the executive summary still match the outputs. Clear outputs before committing.
- **Decide the segment with the team:** the EDA recommends computer and mathematical (US), with healthcare practitioners as the robustness check (notebook section 9).
- **Run the EDA notebook** (`notebooks/jacob/EDA.ipynb`) from the top: flattening (step 4), cleaning (step 6), then the prioritization EDA (Q1 which factors, Q2 time information, Q3 segment). Share only the printed aggregates, and fill in step 9.
- **Delete any leftover `*_flat_*.pkl` files in `data/`** from the earlier version of the flatten cell, if they exist.
- **Post the updated Slack summary** about the new focus, and ask the team to agree.
- **Switch `.env.example` from `S3_BUCKET` to `S3_PATH`.** Claude can't read this file under the project rules, so do it by hand. It should contain three blank lines: `AWS_PROFILE=`, `AWS_REGION=`, `S3_PATH=`. The README is already updated.
- **Tell the team** to pull, rerun `pip install -r requirements.txt` (adds `pyarrow`, `zstandard`, `sentence-transformers` with PyTorch, `rank-bm25` and `umap-learn`), and switch their `.env` to `S3_PATH`.
- **Open the pull request** from `jacob/eda-and-plan` into `main` on GitHub. From now on, commit only on branches.

## Blocked

- **Team:**
  - Agree on the focus: job listing prioritization for a given applicant, NLP + time series. Blocks P0.
  - Decide which "best chance" factors matter most: still open, less competition, fit, pay, employer intent.
  - Assign owners for time factors (seasonal context, cadence, freshness), still open / apply-by and text models, and the prioritization model and pitch.
  - Set a meeting cadence and a place for async discussion.
  - Confirm the final due date and deliverables.
- **App** (details in `APPLICATION.md`, section 8):
  - Course requirements: what's required for the web app and the multi-agent part (rubric, must-haves), or is it open-ended?
  - Who is the user in the demo: a team member with their own résumé, or a synthetic job seeker?
  - Agent ambition: are agents the centerpiece, or a layer on top of a solid queue? Decides how much of week 3 goes to them.
  - Team split: who owns the app (front end, MCP servers), so the analysis isn't starved?
  - Queue fields: is anything missing, such as a skills-gap column, salary vs. market benchmark, or company information?
  - External data: which datasets are allowed, and do we need permission to combine them with PredictLeads data?
  - Hosting: hosted, or a local demo only?
  - Front end: is Streamlit enough for live search and full-screen detail views, or do we need React?
- **Project advisor:** send email 1 in `EMAILS.md`, then record the answers. Get the AI-and-data answers in writing.
  - Scope check on the new focus
  - Final due date and deliverable format
  - Web app and multi-agent requirements or rubric
  - Whether an agent backed by an outside LLM may process posting text. Blocks the agent stage of the app.
  - Using team members' de-identified résumés for fit
  - Combining public datasets with the PredictLeads data
  - Hosting the app, or a local demo only
  - Showing real postings in the final presentation
  - The data-deletion procedure
- **PredictLeads:** send email 2 in `EMAILS.md`, then record the answers.
  - Project fit for the new focus
  - Schema documentation, especially the date fields and `onet_data`
  - Crawl timing: how precise `first_seen_at` and `last_seen_at` are
  - Daily snapshots: full copies or changes, how far back history goes, and whether older snapshots exist
  - `status` values, and how to tell a posting is still open
  - Sources: platforms and company sites only, or job boards too
  - Whether `normalized_title`, `categories`, `tags` or `seniority` come from O*NET, and what `tags` contain
  - File order, and whether the first 200,000 documents are representative
  - Duplicates and reposts
  - Whether `description` is cleaned, and how `language` is detected
  - Whether real postings can appear in the final presentation

## P0

- **Choose from the EDA:**
  - The segment: occupation group × country (Q3)
  - The time scales the data supports (Q2): seasonal needs two or more years of history; day of week needs the crawler check to pass
  - The prioritization factors (Q1): those with real spread across occupation groups, sources or listing types
- **Review the cleaning report (step 6.7).** Check that removed and flagged counts are plausible, and record the key numbers for the write-up.
- **If the segment is too small or the sample is biased** (Q2b, Q3), stream more of the full file with filters and random sampling.
- **Record the agreed focus, segment and time scales** in the plan's decision log.
- **Write the hypotheses** (H1–H4 in the plan) with the segment and thresholds filled in, before any modeling.
- **Pin `lifelines`** (survival analysis, Cox regression) **and `statsmodels`** (seasonal decomposition) in `requirements.txt`, then tell the team.
- **Still open / apply-by:** Kaplan–Meier curves for the segment, by source and posting type, with bootstrap confidence intervals.
- **Seasonal context:** monthly and weekly new-posting volume; STL seasonal decomposition if history allows; compare with a last-year baseline.
- **Source and freshness:** arrival cadence: new postings per week by source and by company; turn into a "check every N days" rule and measure how many openings it catches.

## P1

- **Text predicts timing:**
  - Cox regression with text features (salary disclosed, remote, urgency, TF-IDF terms)
  - Closes-within-N-days classifier: TF-IDF + logistic regression vs. a non-text baseline, trained on earlier months and tested on later ones, folds grouped by company
  - Features only from what's known when a posting appears (see the plan's leakage rules)
- **Repost analysis:** frequency, time to repost, and what reposted postings have in common.
- **Day-of-week patterns,** only if the crawler check (Q2c) shows the dates reflect employers.
- **Statistics:** bootstrap confidence intervals throughout; Benjamini–Hochberg correction across many features or segments.
- **Reproducibility:** fixed seeds, saved posting-ID list and snapshot date, each run's settings recorded with its results.

- **App MVP** (stage 1 in `APPLICATION.md`): list view with live search, sort and filters (priority, apply-by, pay, freshness, source); detail view with apply link; stars and Starred tab. Target: end of week 2.

## P2

- **Prioritization model:** combine chance still open, competition proxies, pay and employer intent. Compare with a newest-first baseline on a later time period.
- **Combine the layers into one strategy,** with the evidence and uncertainty for each.
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never posting text. If MLflow, pin it and tell the team.
- **Remove boilerplate** (equal-opportunity statements, benefits sections) from descriptions before building TF-IDF features.
- **Write up the analysis:** hypotheses, methods, results with confidence intervals, null results, and limitations (crawler dates, disappearing ≠ filled, sample).
- **Choose the final format:** a write-up, a notebook demo, or a small app. If an app, pin it and tell the team.

- **App fit stage:** résumé upload, fit score and filter, "why it matches" and gaps. Résumés stay local.
- **App priority model:** tuned priority score from the analysis, with adjustable weights, checked against a newest-first baseline.
- **MCP servers:** `postings`, `timing`, `fit`, `priority`, `tracker`, built with the `mcp` Python SDK. Pin it and tell the team.

## P3

- **Polish the pitch.**
- **Fit:** TF-IDF and local embeddings to match a profile to postings, added to the prioritization score.
- **Embeddings vs. TF-IDF** as text features for predicting timing.
- **Change-point detection** (`ruptures`) on demand, to find hiring surges and freezes. Pin it if used.
- **Topic modeling** (NMF or BERTopic) to find sub-roles with different timing.
- **Several daily snapshots,** to observe postings closing directly.
- **Calibration** of the closes-within-N predictions: reliability diagram and Brier score.
- **Robustness:** rerun the main findings on a second occupation group or a different time window.
- **Write a one-page system card:** intended use, data, metrics, limitations, failure modes.
- **Move the shared loaders into `src/`:** `fetch`, `stream_jsonl`, `load_cached`, `flatten_to_parts`, `map_parts`, `load_flat`.
- **Fix the README's inconsistent names.** The kernel is registered as `project_leads`, but the text says "Python (team_project)". The README also says `.venv` while your venv is `PL`.

- **App agents:** Ranker, Coach and Reminder first, then Scout, Résumé and Gap. Follow the agent data rule in `APPLICATION.md`.
- **External data:** BLS JOLTS for seasonality and market tightness, BLS wage statistics for pay benchmarks, O*NET skills for fit. Check each one's terms of use.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, or `.env`.
- After a kernel restart, run the notebook from the top.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record team decisions in the plan's decision log.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.

## Finish

- Permanently delete every copy of the data: `data/` (Parquet, pickle and `.jsonl.zst` files, including `data/flat/` and `data/clean/`), downloads, and any S3 copies.
- Confirm the deletion in writing if asked.
