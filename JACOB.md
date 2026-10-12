# Jacob's changes

> What Jacob changed, by category, so Jesse can catch up at a glance. Categories and the lines in each are ranked most important first. Each line is the net result (a file moved and later deleted just reads "deleted"), with the date and anything Jesse needs to do. Details are in the commits (`git log`).

## Data and downloads

- **`models/` renamed `pretrained/`** (`PRETRAINED_DIR` in `fetch/paths.py`). **Jesse:** rename your local `models/` folder to `pretrained/`, or the app won't find the BGE model. (2026-10-12)
- **BGE model download:** `python -m fetch.get_bge` gets `BAAI/bge-base-en-v1.5` (about 0.44 GB, no account) at the version the corpus's embeddings were made with. **Jesse:** run it once if you don't have the model. (2026-10-12)
- **`fetch/`** is where every dataset comes from: one download script each (`get_predictleads`, `get_resumes`, `get_bge`), one standard folder each, a disk-space check, and nothing hard-coded (bucket names and profiles come from `.env`). The PredictLeads export can be downloaded to `data/raw/`; everything reads it locally when it's there and streams from S3 otherwise. (2026-10-12)

## App

- **Deleted `app/archive/`** (48 files of old matching experiments nothing used) and the `-Archive` / `--archive` test switch; it's in the git history. **Jesse:** if you still need anything from it, restore it from git. (2026-10-12)
- **OpenAI features on by default:** `run_demo_backend.ps1` / `.sh` now start with Magic fill, the AI gap analysis and Ask the market's typed questions on; `-NoAI` / `--no-ai` turns them off. Each still asks for consent in the app and needs `OPENAI_API_KEY` in `.env`. **Jesse:** `-ResumeFill`, `-Assess` and `-Market` are gone; just run the script. (2026-10-12)
- **Job level filter on Discover:** Entry level, Experienced, Senior, Manager, Director and above, Level not stated; the ranked list starts on Entry level and each job card shows its level. Levels come from PredictLeads' `seniority` plus the title (`job_level` in `backend/api.py`: a senior title such as lead or principal counts as Senior). (2026-10-12)

## Git workflow

- **Eve reviews every push:** a pre-push agent that tidies your commits, asks about judgment calls, and keeps the asks between us in `.eve/ASKS.md` (how it works: README step 14). From a terminal, her questions come in a normal Claude conversation with the usual question boxes (her own review's session, never another Claude chat in the folder); type `/exit` when she's done. She runs on Sonnet, to stay light on Claude Pro's usage limits. **Jesse:** turn it on once with `git config core.hooksPath .githooks` (needs Claude Code; without it, pushes go up unreviewed). (2026-10-12)
- **No Git rules** beyond these change logs: work on your own branch, merge into `main` when ready, and log your changes here (README step 14). (2026-10-12)

## Docs

- **`SETUP.md` and the README Quick start:** a welcome page for newcomers, and seven setup steps at the top of the README, each linking to its details. If your clone isn't set up, Claude walks you through `SETUP.md` before anything else (`CLAUDE.md`). **Jesse:** just open Claude Code in the repo. (2026-10-12)
- **Two priority queues:** `analysis/ANALYSIS.md` (was `PRIORITY_QUEUE.md`) for the analysis and `app/APP.md` for the app, each top = next. The app design moved from `analysis/APPLICATION.md` to `app/DESIGN.md`, and `ASKS.md` to `.eve/ASKS.md`. **Jesse:** your to-dos from `app/roadmap.md` section 6 and `app/STATUS.md` "Still pending" are merged into `app/APP.md`; check nothing got lost. `app/STATUS.md` stays as you left it until you decide what to do with it (it's an ask for you in `.eve/ASKS.md`). (2026-10-12)
- **Deleted** the hackathon brief (`BRIEF.md`), the decision log (`analysis/DECISION_LOG.md`), the data reference (`analysis/docs/data.md`), the job growth plan (`analysis/JOB_GROWTH.md`) and the merge plan. (2026-10-12)
- **Decisions** are recorded in these change logs instead of a decision log. (2026-10-12)
- **Job growth** steps and extension options are now tasks in `analysis/ANALYSIS.md`. (2026-10-12)
- **Candidate-profile sourcing** (the Kaggle résumés, as the brief asks) is a short note in README step 9. (2026-10-12)

## Analysis

- **US only:** EDA_SDA_1 8b drops every listing with no country. (2026-10-12)
- **EDA_SDA_1 no longer writes schemas** (section 5 and `write_schemas` in `src/pl_data.py` are gone). (2026-10-12)
