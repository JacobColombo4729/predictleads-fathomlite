# Project context for Claude Code

## Project
- **Project:** role discovery and job listing prioritization for **career changers** (anyone switching from their current background to a different role or field), targeting **the roles TripleTen trains for** (computer and mathematical roles, plus sales engineers and business and market research analysts) **and adjacent tech roles** (IT support, networks and systems, database, IT project management) in the US; each listing carries its TripleTen program (`analysis/src/career_switch.py`). Suggest the roles a career changer's skills transfer to, hide listings they can't get, and rank the rest with an ML model. The **retraining compass** is the foundation (`analysis/DIFFERENTIATORS.md`): prior career → training → first tech job → that role's 2026 openings, pay, closing speed and expected growth, for changing careers and for moving across or up within tech. Source backgrounds are the team's own. The career-changer focus is part of the project description. Overview in `analysis/PROJECT_OUTLINE.md`.
- **Deliverables:** prioritization model, analysis notebooks, web app (`analysis/APPLICATION.md`, built in `app/`), final pitch. Deadline: TBD.
- **Standard:** the TripleTen course templates (`~/Desktop/TripleTen/Templates`) are the baseline for every notebook; go further with advanced methods.

## Structure
This repository joins two earlier ones with their full histories: `analysis/` (was `project_leads`, Jacob) and `app/` (was `predictleads-hackathon`, Jesse). They share one set of top-level files: `requirements.txt`, `.venv`, `.env`/`.env.example`, `.gitignore`, `.gitattributes`, `README.md` (setup, running, data rules), `data/` and `models/`. Remaining merge steps: `analysis/PRIORITY_QUEUE.md`; the six agreed merge decisions: `analysis/DECISION_LOG.md` (2026-10-11).

**`analysis/`**
- `PROJECT_OUTLINE.md`: what we're doing. `STANDARDS.md`: how we work (standards, leakage rules, scope, risks).
- `DECISION_LOG.md`: decision log. `docs/data.md`: data reference and auto-generated schemas. `DIFFERENTIATORS.md`: how the project stands out and how to pitch it. `JOB_GROWTH.md`: the expected job growth plan (BLS 2025–2035, conversion to at most 5 years, proposed extensions).
- `PRIORITY_QUEUE.md`: tasks. `APPLICATION.md`: app design, including the retraining compass (section 3).
- `notebooks/`: EDA_SDA_1 (exploration, scope) → EDA_SDA_2 (scope cut, non-text) → EDA_SDA_3 (text features, no matching) → EDA_SDA_4 (statistics, confounders) → TimeSeries → ML_Fit (fit: keyword → TF-IDF → BGE → RRF → skills → trained layer; role discovery, compass) → ML_Rank (listing model + priority queue). Each hands its final table to the next under `data/clean/`. `Profiles.ipynb` (rules in `src/career_switch.py`) finds career switches in the external resume dataset; **frozen** (refrozen 2026-10-10 after adding an IT project management bucket): its final tables are `data/profiles/modeling_table.pkl` and `retrained_candidates.csv`.
- `src/`: shared code imported into notebooks: `eda_helpers.py` (paths, charts, survival), `pl_data.py` (S3, caching, flattening, schemas), `career_switch.py` (career-switch rules, role buckets), `export_markdown.py` (notebook exports)

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

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.

## Git
- Each person works on their own branch (`jacob`, `jesse`); commit to the branch you are on, and never commit to or rewrite the other person's branch.
- Ask before pushing, and before merging `main` into a personal branch.

## Before every commit
- Notebook outputs cleared, including every cell tagged `data-rows` (row previews).
- Nothing from `data/`, `.env`, credentials or venvs staged; no data values, file paths or bucket names in staged files.
- Code changes leak nothing from the future into features (see the leakage rules in `analysis/STANDARDS.md`).
