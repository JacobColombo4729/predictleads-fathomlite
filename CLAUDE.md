# Project context for Claude Code

## Project
- **Project:** role discovery and job listing prioritization for **career changers** (anyone switching from their current background to a different role or field), targeting **the roles TripleTen trains for** (computer and mathematical roles, plus sales engineers and business and market research analysts) **and adjacent tech roles** (IT support, networks and systems, database, IT project management) in the US; each listing carries its TripleTen program (`analysis/src/career_switch.py`). Suggest the roles a career changer's skills transfer to, hide listings they can't get, and rank the rest with an ML model. The **retraining compass** is the foundation (`analysis/DIFFERENTIATORS.md`): prior career → training → first tech job → that role's 2026 openings, pay, closing speed and expected growth, for changing careers and for moving across or up within tech. Source backgrounds are the team's own. The career-changer focus is part of the project description. Overview in `analysis/PROJECT_OUTLINE.md`.
- **Deliverables:** prioritization model, analysis notebooks, web app (`analysis/APPLICATION.md`, built in `app/`), final pitch. Deadline: TBD.
- **Standard:** the TripleTen course templates (`~/Desktop/TripleTen/Templates`) are the baseline for every notebook; go further with advanced methods.

## Structure
This repository joins two earlier ones with their full histories: `analysis/` (was `project_leads`, Jacob) and `app/` (was `predictleads-hackathon`, Jesse). They share one set of top-level files: `requirements.txt`, `.venv`, `.env`/`.env.example`, `.gitignore`, `.gitattributes`, `README.md` (setup, running, data rules), `JACOB.md`/`JESSE.md` (each person's change log), `ASKS.md` (what each wants the other to make sure of; Keel keeps it), `data/` and `pretrained/`, and **`fetch/`**: where every dataset comes from and lives, for both halves (`paths.py`: `REPO_ROOT`, `DATA_DIR`, `RAW_DIR`, `external()`; `predictleads.py`: the S3 settings from `.env` and the copy in `data/raw/`; `downloads.py`: the disk-space check; one `get_<name>.py` per dataset, run as `python -m fetch.get_<name>` from the root). Remaining merge steps: `analysis/PRIORITY_QUEUE.md`.

**`analysis/`**
- `PROJECT_OUTLINE.md`: what we're doing. `STANDARDS.md`: how we work (standards, leakage rules, scope, risks).
- `DIFFERENTIATORS.md`: how the project stands out and how to pitch it.
- `PRIORITY_QUEUE.md`: tasks. `APPLICATION.md`: app design, including the retraining compass (section 3).
- `notebooks/`: EDA_SDA_1 (exploration, scope) → EDA_SDA_2 (scope cut, non-text) → EDA_SDA_3 (text features, no matching) → EDA_SDA_4 (statistics, confounders) → TimeSeries → ML_Fit (fit: keyword → TF-IDF → BGE → RRF → skills → trained layer; role discovery, compass) → ML_Rank (listing model + priority queue). Each hands its final table to the next under `data/clean/`. `Profiles.ipynb` (rules in `src/career_switch.py`) finds career switches in the external resume dataset; **frozen** (refrozen 2026-10-10 after adding an IT project management bucket): its final tables are `data/profiles/modeling_table.pkl` and `retrained_candidates.csv`.
- `src/`: shared code imported into notebooks: `eda_helpers.py` (paths, charts, survival), `pl_data.py` (S3, caching, flattening, schemas), `career_switch.py` (career-switch rules, role buckets), `export_markdown.py` (notebook exports), `stream_full.py` (streams the listings from `data/raw/` or S3). Data paths and downloads come from the top-level `fetch/`

**`app/`**
- `backend/` (FastAPI), `frontend/` (React + Vite), `code/` (corpus, BGE text, ranking, role classifier, résumé fill, signals), `tests/`. How to run and test: `README.md` (steps 11–13); where it stands: `app/STATUS.md`; design docs in `app/docs/`.
- App scripts (`app/run_*.ps1`, `.sh`) run from anywhere; pipeline commands run from `app/`. App code finds the shared folders through `REPO_ROOT` (`app/`'s parent).

**Top level (shared)**
- `data/` and `pretrained/`: local only, git-ignored, **never read by Claude** (also any leftover `models/`, `analysis/data/`, `app/data/`, `app/models/`, `app/backend/data/`)
- `.env`: settings and keys for both halves, git-ignored. Environment: Python 3.11, one `.venv` from `requirements.txt`

## Data & AI rules (from project advisors)
AI tools are allowed for coding, brainstorming, and debugging. **Do not upload or paste confidential project data into them.** For Claude, that means:
- Never read any `data/` folder, any `.env`, raw data files, or `~/.aws/` (also blocked in `.claude/settings.json`).
- Never print data values (`df.head()`, sample rows) in commands you run.
- Work from code, column names, dtypes, shapes, and descriptions. Use **synthetic data** for examples and tests.
- Never put project details, data, file paths, or bucket names into web searches.
- Do not commit data, notebook outputs, credentials, or venvs.

## Getting data (both halves)
How and why: README step 9. The rules:
- **Every dataset comes through a script in `fetch/`,** never by hand (except data that can't be downloaded, below).
- **Each dataset lives in one standard folder under `data/`,** from `fetch.paths`.
- **Never hard-code** bucket names, `s3://` paths, AWS profiles, keys or personal paths: they come from `.env`. `app/tests/test_no_hardcoded_data.py` checks this locally and on every push.
- **When someone (or their code) does it another way,** say so before writing anything, explain why (our data agreement forbids bucket names and credentials in committed files; hard-coded paths and profiles break on every other machine; one folder per dataset keeps both halves reading the same files and makes the end-of-project deletion complete), then show the right way with the steps from README step 9.

### Adding a new dataset (checklist; Claude reminds the user of every step)
1. **Its standard folder:** `external("<name>")` (`data/external/<name>/`) from `fetch.paths`, never a written-out path.
2. **A download script,** `fetch/get_<name>.py`, modelled on `get_resumes.py`: a pinned version, the space check (`confirm_space`), saves only into its folder, skips data that's already there. Any credentials go in `.env`.
3. **Can't be downloaded by script?** Put the files in its folder by hand and note the source in the README's data table (step 9).
4. **Readers get the folder from `fetch`,** in either half.
5. **Write it down:** the README's data table (step 9).

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.

## Git
- **Keel** (`.claude/agents/keel.md`, run by `.githooks/pre-push` on every push to a personal branch) tidies and reviews each push; turn it on per clone with `git config core.hooksPath .githooks` (README step 14).
- **Log every change** in the committer's file: `JACOB.md` for Jacob, `JESSE.md` for Jesse (from `git config user.name`), in the same commit. Group lines into categories that come from the changes themselves: no preset headings, add one when a change doesn't fit an existing one, and regroup as the log grows; rank the categories, and the lines within each, most important first (what the other person must act on or what breaks their work ranks highest), re-ranking as entries change; write the net result, so a later change to the same thing rewrites its line instead of adding one (moved, then deleted, reads "deleted"); end with the date and anything the other person needs to do. No other Git rules.

## Before every commit
- Notebook outputs cleared, including every cell tagged `data-rows` (row previews).
- Nothing from `data/`, `.env`, credentials or venvs staged; no data values, file paths or bucket names in staged files (`.\app\run_tests.ps1` includes the hard-coding check).
- Code changes leak nothing from the future into features (see the leakage rules in `analysis/STANDARDS.md`).
