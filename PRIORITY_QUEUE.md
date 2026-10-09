# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> What each notebook will do is in its own plan; how we work is in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## Now

- **Download the structured resume dataset** into `data/external/resume_structured/` (the command is in the chat, or download it from Kaggle by hand), then run `notebooks/Profiles.ipynb`.
- **Career-switch review rounds:** mark each round's sample in `data/profiles/review/`, share the section 5 counts, adjust `src/career_switch.py`, repeat until candidate precision is 90%+ two rounds in a row. Also review a rejected pool or two.
- **Rerun EDA_SDA_1 from 6.1** (the lean-table cache rebuilds for the new career-change columns), then fill in every "*To fill in after rerunning*" note (6.1h, Q1a, Q1g, Q3a, 12b, the executive summary and the conclusion) and the matching rows in `PROJECT_OUTLINE.md`.
- **Check the seniority labels** (EDA_SDA_1 Q3b) and add any entry-level ones besides `junior` to `ENTRY_LEVELS` in 6.7a.
- **Slide deck for the team, due Friday 2026-10-09** (reminder): our findings at each step, how we made the major decisions, and a final pitch, framed around career changers.
- **Give Claude instructions for the additional data cleaning** (reminder).
- **Answer Claude's open suggestions:**
  - Reword the 60.7% as "up to 60.7%" in the summaries, and add a domain check to EDA_SDA_2 (3.2) to refine it.
  - Add a caveat that `urgency` and `mentions_remote` are keyword rules with suspected false positives ("urgent care"), and a validation step to EDA_SDA_3.
- **EDA_SDA_2 (non-text):** follow its plan (sections 2–4). EDA_SDA_1's final table is saved (171,143 listings).
- **Pin the packages the next notebooks need** in `requirements.txt` (`lifelines`, `statsmodels`, `nltk`, `spacy`, `lightgbm`, `catboost`, `xgboost`, `shap`), then tell the team.
- **Delete any leftover `*_flat_*.pkl` files in `data/`** from the earlier version of the flatten cell, if they exist.
- **Tell the team** to pull, rerun `pip install -r requirements.txt`, and switch their `.env` to `S3_PATH`.

## Next

- **Role discovery:** similarity between the team's backgrounds and computer and mathematical occupations in EDA_SDA_3; hit rate against real switchers' moves in ML.
- **O*NET related occupations and skills:** check the terms of use and download them, including the **Career Changers Matrix** (latest release: up to 10 related occupations per occupation that people can move into with little extra preparation). EDA_SDA_2 needs them to pair source backgrounds with target fields; transferable fit needs them later; the Career Changers Matrix is a current, independent check on role discovery.
- **BLS occupational separations (Employment Projections, 2024–34):** download the occupational transfer and labor force exit rates for each occupation; check the terms of use. Use them for how often people leave the team's background occupations, and for projected computer and mathematical openings (pitch context). They're projections from historical patterns and show who leaves, not where they go.
- **CPS occupation transitions (IPUMS CPS ASEC, through 2025):** register, check the terms of use, and build a table of moves *into* computer and mathematical occupations by prior occupation and year (`OCC2010` vs `OCC10LY`; only people employed now and last year). Compare it with the resume dataset's background mix, and reweight the resumes if they differ a lot. Aggregates only.
- **EDA_SDA_3 (NLP):** follow its plan. Output: niche table + text features.
- **EDA_SDA_4 (statistics):** follow its plan. Output: modeling-ready table.
- **TimeSeries:** follow its plan.
- **ML:** follow its plan, including writing the hypotheses with thresholds before any modeling, and the solution report.
- **App MVP** (stage 1 in `APPLICATION.md`): list view with search, sort and filters; detail view with apply link; stars.
- **Write up the analysis** and **choose the final format** (write-up, notebook demo, or app).

## Blocked

- **Team:**
  - Each teammate writes down their background (current field and roles; no personal details), to fix the source backgrounds.
  - Assign owners for time factors (seasonal context, cadence, freshness), still open / apply-by and text models, and the prioritization model and pitch.
  - Set a meeting cadence and a place for async discussion.
  - Confirm the final due date and deliverables.
- **App:** answer the open questions in `APPLICATION.md`, section 8.
- **Project advisor:** send email 1 in `EMAILS.md`, then record the answers in `docs/decisions.md`. Get the AI-and-data answers in writing; the agent stage of the app depends on them. Question 5 (collecting career-changer profiles) blocks transferable fit.
- **Applicant profiles** (after the advisor answers): follow `docs/fit_analysis.md`. Assign owners for its task table, draft the consent form, recruit 50 career changers, convert their exports, and build the synthetic generator in `src/`.
- **PredictLeads:** send email 2 in `EMAILS.md` (12 questions), then record the answers in `docs/decisions.md`.

## Later

- **App:** transferable-fit stage (the career changer's background profile), priority model with adjustable weights, MCP servers, then agents (see `APPLICATION.md`).
- **External data:** BLS JOLTS and BLS wage statistics. Check each one's terms of use.
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Never listing text.
- **Stretch techniques:** transferable fit with embeddings; embeddings vs. TF-IDF; change-point detection (`ruptures`); topic modeling (NMF or BERTopic); several daily snapshots to observe closures directly; a one-page system card.
- **If the niche is too small or the sample biased,** stream more of the full file with filters and random sampling.
- **Move the data loaders into `src/`:** `fetch`, `stream_jsonl`, `load_cached`, `flatten_to_parts`, `map_parts`, `load_flat`.
- **Fix the README's venv name:** the README says `.venv` while your venv is `PL`.

## Always

- Clear notebook outputs before every commit.
- Never commit data, Parquet or pickle files, or `.env`.
- After a kernel restart, run the notebook from the top.
- Follow the working standards in `PROJECT_PLAN.md` and the course-template baseline in every notebook.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record every decision, with its reasoning, in `docs/decisions.md`.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.

## Finish

- **Polish all six notebooks (EDA_SDA_1–4, TimeSeries, ML) so they read as one unit**, including trimming EDA_SDA_1 (one findings section instead of three; data loading moved to `src/` or its own notebook; a shorter overview): consistent titles, numbering, terms and formatting; a clear purpose for every step; each notebook's opening linking back to the previous one's conclusion; one shared executive summary or index; imports at the top of each notebook (course standard); no repeated code; every course-template requirement checked.
- Permanently delete every copy of the data: `data/` (Parquet, pickle and `.jsonl.zst` files, including `data/flat/` and `data/clean/`), downloads, and any S3 copies.
- Confirm the deletion in writing if asked.
