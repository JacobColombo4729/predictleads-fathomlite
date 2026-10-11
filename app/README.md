# PredictLeads Hackathon: career-changer job matching

**Bright Depth** is a demo that helps career changers (starting with TripleTen AI/ML graduates) find realistic roles in AI/ML, data and technical sales / solutions engineering. A user builds a profile (optionally with **Magic fill** from a resume), sees ranked jobs from the September 30, 2026 PredictLeads snapshot, rates them 👍/👎 to refine the list, saves the ones they like, and can send up to 20 saved jobs to the **Compass**, which finds what they have in common, how the profile lines up (with the closest line of the person's own profile as evidence) and other roles that might fit. The Compass includes an **AI skills gap analysis for each job**: one request compares every picked job's requirements with the profile and cites the exact lines used (off unless started with `-Assess`). A **Signals** tab (with **Ask the market**: ask who is hiring in plain words, by typing or talking, answered only from our data; see [docs/ask-the-market.md](docs/ask-the-market.md)) ranks companies by hiring bursts and steady hiring (see [docs/hiring-signals.md](docs/hiring-signals.md)). Dark theme by default.

**Start with [STATUS.md](STATUS.md)** for the current state and [roadmap.md](roadmap.md) for what is planned. This file is only the map.

Everything here is a hackathon prototype. Listings are **historical records, not verified open positions**, and no ranking-quality claim is made: the ranker has not been compared against alternatives on this audience.

## What is in the repo

| Path | What it is |
|---|---|
| `backend/` | FastAPI demo API: sessions, profiles, search, ranked recommendations (TF-IDF + BGE reciprocal-rank fusion), relevance feedback, saved jobs, the Compass analysis, grounded assessment (off by default) and resume extraction (opt-in). See [docs/backend-api.md](docs/backend-api.md). |
| `frontend/` | React + Vite app wired to the API (Discover, Saved, Compass, Signals, Profile). See [frontend/README.md](frontend/README.md). |
| `code/` | Python modules: matching engine (`job_matching_engine`), BGE text builders (`bge_text`), corpus and ranking pipeline (`rank_candidates`, `extract_targeted_pool`, `prepare_jobs`), role-family classifier, grounded explanations, resume extraction, review tooling, and the early data-discovery scripts. |
| `tests/`, `backend/tests/` | Live tests. Run them all with `.\run_tests.ps1`. |
| `archive/experiments/` | Earlier experiments, kept as history. Nothing live imports them. See its [README](archive/experiments/README.md). |
| `docs/` | Data discovery and preparation notes, API contracts, the grounded-explanation contract. `docs/archive/` holds historical documents. |
| `notebooks/` | Data discovery notebook (local-only view). |
| `data/`, `models/` | **Git-ignored.** Source data, corpus, embeddings and the local BGE model. |

## Set up and run

Commands below are PowerShell (Windows). On **macOS / Linux** use the `.sh` equivalents noted under each step.

1. **Environment** (one virtual environment for everything):
   ```powershell
   python -m venv .venv
   & ".\.venv\Scripts\pip" install -r requirements.txt
   cd frontend; npm install; cd ..
   ```
   macOS / Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && (cd frontend && npm install)`
2. **Data and model** (not in git): the serving corpus at `data/matching/ai_ml_serving_v1` (built by the pipeline below) and the BGE model `BAAI/bge-base-en-v1.5` (revision pinned in `code/bge_text.py`) in `models/bge-base-en-v1.5`, or set `PREDICTLEADS_BGE_MODEL_DIR`. Data lives in `data/` here unless `PREDICTLEADS_DATA_DIR` points elsewhere (for example one folder shared with `analysis/`, whose own setting is `PL_DATA_DIR`); the backend, the run scripts and the pipeline all follow it.
3. **Backend** (about 3.5 minutes to load, about 2.5 GB of memory):
   ```powershell
   .\run_demo_backend.ps1            # add -ResumeFill to enable Magic fill; -Ranker tfidf for the lighter ranker
   ```
   macOS / Linux: `./run_demo_backend.sh` (flags `--resume-fill`, `--ranker tfidf`, `--port N`). On a laptop, `--preview` runs TF-IDF on a 5,000-job sample in `data/preview/jobs.jsonl` (starts in ~20 s, no model needed, Compass unavailable); the script prints how to build the sample if it is missing.
4. **Frontend**: `cd frontend; npm run dev` (macOS / Linux: `cd frontend && npm run dev`), then open http://127.0.0.1:5173.
5. **Magic fill** sends the resume text you choose to OpenAI after you tick a consent box. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. It is off unless the backend starts with `-ResumeFill`.

## Tests

```powershell
.\run_tests.ps1                  # live Python tests
.\run_tests.ps1 -Frontend        # also type-check and build the React app
.\run_tests.ps1 -Archive         # also the archived experiment tests
```

macOS / Linux: `./run_tests.sh`, with `--frontend` and `--archive`.

## Data pipeline (rebuilding the corpus)

All commands run from the repo root with the project `.venv`; each script's `--help` lists its options.

1. `python code/extract_targeted_pool.py --scan` streams the full snapshot (about 19 GB from S3, using the `predictleads` AWS profile; about 36 minutes) and keeps AI/ML, data and technical-sales listings. `--check-sample` tests the matcher offline.
2. `python code/rank_candidates.py build-corpus --targeted-dir data/matching/targeted_pool_<timestamp>` builds the deduplicated corpus. `encode` then `finalize-encode` compute the BGE embeddings (GPU recommended).
3. `python code/rank_candidates.py export-serving` writes the serving subset the backend loads.
4. `python code/rank_candidates.py rank ...` ranks the 12 test profiles; `python code/build_review_page.py --run-dir <ranking run> --profiles <profile files>` builds an optional blinded review page.
5. `python code/role_family_classifier.py` trains and evaluates the supporting role-family classifier.

**The raw snapshot is not stored locally**, so deleting `data/matching/targeted_pool_*` or the 30,000-job random sample under `data/discovery/` means re-reading it from S3.

## Privacy and data rules

- Resumes, real profiles, credentials (`.env`) and everything under `data/` stay out of git.
- Nothing is sent to an outside AI service unless the person whose text it is agrees first. Magic fill removes emails, phone numbers and links before sending.
- Missing information is **unknown**, not a gap. Null status is not "open".

## Where to read more

1. [STATUS.md](STATUS.md) and [roadmap.md](roadmap.md)
2. [Discovery summary](docs/discovery-summary.md), [data preparation](docs/data-preparation.md), [preparation review](docs/preparation-review.md), [role exploration](docs/role-exploration.md), [artifact index](docs/artifact-index.md), [what the first/last-seen dates mean](docs/signal-semantics.md), [hiring signals](docs/hiring-signals.md), [ML inventory (what is ML, why, how validated)](docs/ml-inventory.md), [feedback data and the path to production](docs/feedback-data.md)
3. [Backend API](docs/backend-api.md), [API contracts](docs/api-contracts.md), [grounded match explanations](docs/grounded-match-explanations.md), [matching engine notes](docs/job-matching-engine.md)
4. History: [docs/archive/](docs/archive/) and [archive/experiments/](archive/experiments/README.md)
