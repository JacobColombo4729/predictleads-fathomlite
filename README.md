# predictleads-fathomlite

Role discovery and job listing prioritization for career changers, built on PredictLeads job data: which tech roles a person's background transfers to, which listings they can realistically get, and which to apply to first.

This repository joins two earlier ones, with their full commit histories:

| Folder | Was | What it holds | Start here |
|:--|:--|:--|:--|
| `analysis/` | `project_leads` (Jacob) | Analysis notebooks (EDA, time series, fit and ranking models), career-switch profiles, project docs and decision log | `analysis/README.md`, `analysis/PROJECT_OUTLINE.md` |
| `app/` | `predictleads-hackathon` (Jesse) | The web app: FastAPI backend, React frontend, corpus and ranking pipeline, tests | `app/README.md`, `app/STATUS.md` |

Each folder is still its own project root, with its own `requirements.txt`, `.env` and git-ignored `data/`; set up and run each from inside its folder. Combining them (one data layout, one requirements file, docs in one place) is planned in `analysis/docs/merge_plan.md`.

`CLAUDE.md` (instructions for Claude Code, including the data rules) and `.claude/` stay at the top so they apply to the whole repository.
