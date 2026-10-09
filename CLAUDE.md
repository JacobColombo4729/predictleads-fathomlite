# Project context for Claude Code

## Project
- **Project:** role discovery and job listing prioritization for **career changers** (anyone switching from their current background to a different role or field), targeting computer and mathematical roles in the US. Suggest the roles a career changer's skills transfer to, hide listings they can't get, and rank the rest with an ML model. Source backgrounds are the team's own. The career-changer focus is part of the project description. Overview in `PROJECT_OUTLINE.md`.
- **Deliverables:** prioritization model, analysis notebooks, web app (`APPLICATION.md`), final pitch. Deadline: TBD.
- **Standard:** the TripleTen course templates (`~/Desktop/TripleTen/Templates`) are the baseline for every notebook; go further with advanced methods.

## Structure
- `PROJECT_OUTLINE.md`: what we're doing. `PROJECT_PLAN.md`: how we work (standards, leakage rules, scope, risks).
- `docs/decisions.md`: decision log. `docs/data.md`: data reference and auto-generated schemas.
- `PRIORITY_QUEUE.md`: tasks. `EMAILS.md`: questions for the advisor and PredictLeads. `APPLICATION.md`: app design.
- `notebooks/`: EDA_SDA_1 (exploration, scope) → EDA_SDA_2 (non-text, niche) → EDA_SDA_3 (NLP) → EDA_SDA_4 (statistics) → TimeSeries → ML. Each hands its final table to the next under `data/clean/`. `Profiles.ipynb` (rules in `src/career_switch.py`) finds career switches in the external resume dataset, tuned in review rounds.
- `src/`: shared code (`src/eda_helpers.py`), imported into notebooks
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
- Notebook outputs cleared, including the tagged `data-rows` preview cell.
- Nothing from `data/`, `.env`, credentials or venvs staged; no data values, file paths or bucket names in staged files.
- Code changes leak nothing from the future into features (see the leakage rules in `PROJECT_PLAN.md`).
