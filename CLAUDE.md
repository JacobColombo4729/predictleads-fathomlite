# Project context for Claude Code

## Project
- **Project:** role discovery and job listing prioritization for **career changers** (anyone switching from their current background to a different role or field), targeting computer and mathematical roles in the US. Suggest the roles a career changer's skills transfer to, hide listings they can't get, and rank the rest with an ML model. The **retraining compass** is the foundation (`DIFFERENTIATORS.md`): prior career → training → first tech job → that role's 2026 openings, pay, closing speed and expected growth, for changing careers and for moving across or up within tech. Source backgrounds are the team's own. The career-changer focus is part of the project description. Overview in `PROJECT_OUTLINE.md`.
- **Deliverables:** prioritization model, analysis notebooks, web app (`APPLICATION.md`), final pitch. Deadline: TBD.
- **Standard:** the TripleTen course templates (`~/Desktop/TripleTen/Templates`) are the baseline for every notebook; go further with advanced methods.

## Structure
- `PROJECT_OUTLINE.md`: what we're doing. `STANDARDS.md`: how we work (standards, leakage rules, scope, risks).
- `docs/decisions.md`: decision log. `docs/data.md`: data reference and auto-generated schemas. `DIFFERENTIATORS.md`: how the project stands out and how to pitch it. `COMPASS.md`: every compass idea in one place (what it shows, path questions, the Ideas tab, honesty rules, where each piece is built). `docs/job_growth.md`: the expected job growth plan (BLS 2025–2035, conversion to at most 5 years, proposed extensions).
- `PRIORITY_QUEUE.md`: tasks. `APPLICATION.md`: app design.
- `notebooks/`: EDA_SDA_1 (exploration, scope) → EDA_SDA_2 (scope cut, non-text) → EDA_SDA_3 (text features, no matching) → EDA_SDA_4 (statistics, confounders) → TimeSeries → ML_Fit (fit: keyword → TF-IDF → BGE → RRF → skills → trained layer; role discovery, compass) → ML_Rank (listing model + priority queue). Each hands its final table to the next under `data/clean/`. `Profiles.ipynb` (rules in `src/career_switch.py`) finds career switches in the external resume dataset; **frozen** (refrozen 2026-10-10 after adding an IT project management bucket): its final tables are `data/profiles/modeling_table.pkl` and `retrained_candidates.csv`.
- `src/`: shared code imported into notebooks: `eda_helpers.py` (paths, charts, survival), `pl_data.py` (S3, caching, flattening, schemas), `career_switch.py` (career-switch rules, role buckets), `export_markdown.py` (notebook exports)
- `data/`: local data only, git-ignored, **never read by Claude**
- `.env`: non-secret settings (AWS profile, region, S3 path), git-ignored
- Environment: Python 3.11 venv, packages in `requirements.txt`

## Data & AI rules (from project advisors)
AI tools are allowed for coding, brainstorming, and debugging. **Do not upload or paste confidential project data into them.** For Claude, that means:
- Never read `data/`, `.env`, raw data files, or `~/.aws/` (also blocked in `.claude/settings.json`).
- Never print data values (`df.head()`, sample rows) in commands you run.
- Work from code, column names, dtypes, shapes, and descriptions. Use **synthetic data** for examples and tests.
- Never put project details, data, file paths, or bucket names into web searches.
- Do not commit data, notebook outputs, credentials, or venvs.

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.

## Before every commit
- Notebook outputs cleared, including every cell tagged `data-rows` (row previews).
- Nothing from `data/`, `.env`, credentials or venvs staged; no data values, file paths or bucket names in staged files.
- Code changes leak nothing from the future into features (see the leakage rules in `STANDARDS.md`).
