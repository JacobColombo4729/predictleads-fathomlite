# Project context for Claude Code

## Project
- **Project:** role discovery and job listing prioritization for **career changers** (anyone switching from their current background to a different role or field), targeting **the roles TripleTen trains for** (computer and mathematical roles, plus sales engineers and business and market research analysts) **and adjacent tech roles** (IT support, networks and systems, database, IT project management) in the US; each listing carries its TripleTen program (`analysis/src/career_switch.py`). Suggest the roles a career changer's skills transfer to, hide listings they can't get, and rank the rest with an ML model. The **retraining compass** is the foundation (`analysis/DIFFERENTIATORS.md`): prior career → training → first tech job → that role's 2026 openings, pay, closing speed and expected growth, for changing careers and for moving across or up within tech. Source backgrounds are the team's own. The career-changer focus is part of the project description. Overview in `analysis/PROJECT_OUTLINE.md`.
- **The hackathon brief** (requirements, deliverables, out of scope; it wins over our plans unless the team decides otherwise): @analysis/BRIEF.md
- **Deliverables:** prioritization model, analysis notebooks, web app (`analysis/APPLICATION.md`, built in `app/`), final pitch. Deadline: TBD.
- **Standard:** the TripleTen course templates (`~/Desktop/TripleTen/Templates`) are the baseline for every notebook; go further with advanced methods.

## Structure
This repository joins two earlier ones with their full histories: `analysis/` (was `project_leads`, Jacob) and `app/` (was `predictleads-hackathon`, Jesse). They share one set of top-level files: `requirements.txt`, `.venv`, `.env`/`.env.example`, `.gitignore`, `.gitattributes`, `README.md` (setup, running, data rules), `data/` and `models/`. Remaining merge steps: `analysis/PRIORITY_QUEUE.md`; the six agreed merge decisions: `analysis/DECISION_LOG.md` (2026-10-11).

**`analysis/`**
- `PROJECT_OUTLINE.md`: what we're doing. `STANDARDS.md`: how we work (standards, leakage rules, scope, risks).
- `DECISION_LOG.md`: decision log. `docs/data.md`: data reference and auto-generated schemas. `DIFFERENTIATORS.md`: how the project stands out and how to pitch it. `JOB_GROWTH.md`: the expected job growth plan (BLS 2025–2035, conversion to at most 5 years, proposed extensions).
- `PRIORITY_QUEUE.md`: tasks. `APPLICATION.md`: app design, including the retraining compass (section 3).
- `notebooks/`: EDA_SDA_1 (exploration, scope) → EDA_SDA_2 (scope cut, non-text) → EDA_SDA_3 (text features, no matching) → EDA_SDA_4 (statistics, confounders) → TimeSeries → ML_Fit (fit: keyword → TF-IDF → BGE → RRF → skills → trained layer; role discovery, compass) → ML_Rank (listing model + priority queue). Each hands its final table to the next under `data/clean/`. `Profiles.ipynb` (rules in `src/career_switch.py`) finds career switches in the external resume dataset; **frozen** (refrozen 2026-10-10 after adding an IT project management bucket): its final tables are `data/profiles/modeling_table.pkl` and `retrained_candidates.csv`.
- `src/`: shared code imported into notebooks: `eda_helpers.py` (paths, charts, survival), `pl_data.py` (S3, caching, flattening, schemas), `career_switch.py` (career-switch rules, role buckets), `export_markdown.py` (notebook exports), `get_predictleads.py` (downloads the export files to `data/raw/`, optional), `stream_full.py` (streams the listings from `data/raw/` or S3), `get_resumes.py` (downloads the Kaggle resumes)

**`app/`**
- `backend/` (FastAPI), `frontend/` (React + Vite), `code/` (corpus, BGE text, ranking, role classifier, résumé fill, signals), `tests/`. How to run and test: `README.md` (steps 11–13); where it stands: `app/STATUS.md`; design docs in `app/docs/`.
- App scripts (`app/run_*.ps1`, `.sh`) run from anywhere; pipeline commands run from `app/`. App code finds the shared folders through `REPO_ROOT` (`app/`'s parent).

**Top level (shared)**
- `data/` and `models/`: local only, git-ignored, **never read by Claude** (also any leftover `analysis/data/`, `app/data/`, `app/models/`, `app/backend/data/`)
- `.env`: settings and keys for both halves, git-ignored. Environment: Python 3.11, one `.venv` from `requirements.txt`

## Data & AI rules (from project advisors)
AI tools are allowed for coding, brainstorming, and debugging. **Do not upload or paste confidential project data into them.** For Claude, that means:
- Never read any `data/` folder, any `.env`, raw data files, or `~/.aws/` (also blocked in `.claude/settings.json`).
- Never print data values (`df.head()`, sample rows) in commands you run.
- Work from code, column names, dtypes, shapes, and descriptions. Use **synthetic data** for examples and tests.
- Never put project details, data, file paths, or bucket names into web searches.
- Do not commit data, notebook outputs, credentials, or venvs.

## Getting data (both halves)
- **Every dataset is fetched by a script, never by hand:** PredictLeads files with `python -m src.get_predictleads` (to `data/raw/`, optional) or the streaming readers (`src/pl_data.py`, `src/stream_full.py`, `app/code/prepare_jobs.py` settings); the Kaggle resumes with `python -m src.get_resumes`. A new outside dataset gets its own download script in `analysis/src/` the same way (pinned version, disk-space check, prints names and sizes only).
- **Never download what's already here:** run `python -m src.get_predictleads --check` (and look for the tables, by name only) before fetching anything; the scripts reuse existing copies, and a second copy of a dataset is never made on purpose.
- **If someone's old hard-coded setup breaks,** point them to README step 9 ("If your old data setup stopped working") instead of putting the hard-coded values back.
- **Never hard-code where data lives or how to reach it:** no bucket names, `s3://` paths, AWS profile names, access keys or personal folder paths in code or docs. The S3 location and profile come only from `.env` (`S3_PATH`, `AWS_PROFILE`); local data only under the top-level `data/` (`DATA_DIR`, `REPO_ROOT / "data"`). `app/tests/test_no_hardcoded_data.py` fails on any of these, locally and on every push to GitHub (`.github/workflows/data-rules.yml`).

### Adding a new dataset (checklist; Claude reminds the user of every step)
1. **A download script** in `analysis/src/get_<name>.py`, modelled on `get_predictleads.py` and `get_resumes.py`: a pinned version or release, `src.downloads.confirm_space` before downloading (says the space needed and free, asks to proceed, stops if it doesn't fit), skips what's already there, resumes or reuses partial and hand-downloaded copies, prints names and sizes only. Credentials or locations it needs go in `.env` (and, blank, in `.env.example`), never in code.
2. **One standard folder** under the top-level `data/`, built from `DATA_DIR` (never a literal path): `data/raw/` for PredictLeads exports, `data/external/<name>/` for outside data.
3. **Update the duplicate lookup** so the new folder is known: add it (and any place people kept it by hand) to the lookup that finds existing copies (`COPY_DIRS` in `src/pl_data.py` for PredictLeads files, `local_zip()`/`present()` for the resumes, or the new script's own), so nothing is downloaded twice.
4. **Readers find it through the same code,** not their own paths: notebooks call the script's `ensure()` first (as Profiles does), and the app imports its location from shared settings.
5. **Write it down:** the README's data table (step 9), `analysis/docs/data.md` (source, license, folder, command), the end-of-project deletion list in `analysis/PRIORITY_QUEUE.md`, and the `src/` list above.
6. **Check:** `.ppun_tests.ps1` (the hard-coding check) passes, and a second run of the script skips the download.

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.

## Git
- Each person works on their own branch (`jacob`, `jesse`); commit to the branch you are on, and never commit to or rewrite the other person's branch.
- Ask before pushing, and before merging `main` into a personal branch.

## Before every commit
- Notebook outputs cleared, including every cell tagged `data-rows` (row previews).
- Nothing from `data/`, `.env`, credentials or venvs staged; no data values, file paths or bucket names in staged files (`.ppun_tests.ps1` includes the hard-coding check).
- Code changes leak nothing from the future into features (see the leakage rules in `analysis/STANDARDS.md`).
