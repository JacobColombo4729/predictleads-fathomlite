# Demo Backend API

A minimal FastAPI service in `backend/api.py` for the demo. It is deliberately **not** production authentication or a production data service. Responses keep source nulls, raw status, `availability: unverified` and snapshot provenance; nothing is synthesized.

## What it provides

- Anonymous, session-isolated profiles, saved jobs, ratings and cached assessments in SQLite, with a sliding 24-hour TTL and `DELETE /api/v1/demo-sessions/current` to reset everything.
- Profile create/read/update with version checks (`If-Match`). Each experience entry has a `kind` (`job`, `project` or `education`, default `job`) that only affects display; matching reads every entry the same way.
- Read-only search and job/company details from one historical snapshot (the 81,284 listings whose source status is not explicitly closed).
- Personalized recommendations through a replaceable ranker. The demo runs `tfidf_bge_rrf_equal` (`backend/rrf_ranker.py`, equal-weight reciprocal-rank fusion of TF-IDF and BGE cosine, depth 50, k=60); `PREDICTLEADS_API_RANKER=tfidf` serves TF-IDF alone. Scores are retrieval signals, not match percentages or hiring probabilities.
- Relevance feedback: 👍/👎 on jobs per session. Recommendations then move the query toward liked and away from rejected jobs (Rocchio-style, applied to both signals) and hide rated jobs. The model is not retrained, and ratings disappear with the session.
- A grounded assessment endpoint for one profile/job pair (`code/grounded_match_explanations.py`). Provider generation is **disabled by default** and the web app does not call it yet.
- Resume extraction ("Magic fill", `code/resume_profile_extraction.py`): **opt-in**, sends resume text to OpenAI. See below.

## Run

From the repo root, with the single project `.venv` (see the [README](../README.md)):

```powershell
.\run_demo_backend.ps1                # RRF ranker on data/matching/ai_ml_serving_v1
.\run_demo_backend.ps1 -ResumeFill    # also enable Magic fill (needs OPENAI_API_KEY in .env)
.\run_demo_backend.ps1 -Ranker tfidf  # lighter, keyword-only ranker
```

It serves http://127.0.0.1:8000 (OpenAPI at `/docs`). Startup reads the corpus and fits TF-IDF once (about 3.5 minutes, about 2.5 GB). It makes no external call unless `-ResumeFill` is set and a user triggers it.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `PREDICTLEADS_API_JOBS_PATH` | `data/matching/ai_ml_serving_v1/jobs_prepared_corpus.jsonl` | Prepared listings the API serves. Must be the same file the embeddings were built from when using `rrf`. |
| `PREDICTLEADS_API_RANKER` | `tfidf` (the start script sets `rrf`) | `rrf` or `tfidf`. |
| `PREDICTLEADS_API_CORPUS_DIR` | none | Directory with `bge_embeddings.npy` and its manifest (required for `rrf`). |
| `PREDICTLEADS_BGE_MODEL_DIR` | `models/bge-base-en-v1.5` | Local BGE model files used to encode profile queries. |
| `PREDICTLEADS_API_DB_PATH` | `backend/data/predictleads-demo.sqlite3` | Sessions, profiles, saved jobs, ratings, assessment cache (git-ignored). |
| `PREDICTLEADS_API_SESSION_TTL_SECONDS` | `86400` | Sliding anonymous session expiry (minimum 60). |
| `PREDICTLEADS_API_COOKIE_SECURE` | `0` | Set to `1` over HTTPS. |
| `PREDICTLEADS_API_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Allowed credentialed browser origins. |
| `PREDICTLEADS_RESUME_EXTRACTION` | off | `openai` enables Magic fill. The start script sets it with `-ResumeFill`. |
| `PREDICTLEADS_RESUME_MODEL` | `gpt-4.1-mini` | Model used for resume extraction. |
| `OPENAI_API_KEY` | none | Read from the environment or the project-root `.env`. Never commit it. |

## Routes

All data routes except session creation, reset, `healthz` and the snapshot description require the `pl_demo_session` HttpOnly cookie. It is anonymous session isolation, not login.

| Method and route | Purpose |
|---|---|
| `GET /healthz` | Liveness. |
| `GET /api/v1/snapshots/current` | Snapshot id, date, job count, ranker, availability note. |
| `POST /api/v1/demo-sessions` | Create a random demo session and cookie. |
| `DELETE /api/v1/demo-sessions/current` | Delete the session and everything in it; expire the cookie. |
| `POST /api/v1/profiles` | Create the session's single profile. |
| `GET /api/v1/profiles/current` | The session's profile (404 if none). |
| `GET /api/v1/profiles/{profile_id}` | Read the profile and version. |
| `PATCH /api/v1/profiles/{profile_id}` | Partial update; requires `If-Match: "{version}"`. |
| `GET /api/v1/jobs?snapshot_id=...` | Cursor-paginated search with `q`, `location`, `description=any\|with\|without`, `category`, `seniority`. `work_arrangement` filtering returns `400 unsupported_filter`. |
| `GET /api/v1/jobs/{job_id}?snapshot_id=...` | Full prepared job details and provenance. |
| `GET /api/v1/companies/{company_id}?snapshot_id=...` | Same-wrapper company values only; no companies-file join. |
| `POST /api/v1/profiles/{profile_id}/recommendations` | Ranked jobs for a profile version and snapshot, with optional filters. |
| `GET/PUT /api/v1/feedback`, `DELETE /api/v1/feedback/{snapshot_id}/{job_id}` | List, set (`good` or `bad`) and clear ratings. |
| `POST /api/v1/profiles/{profile_id}/jobs/{job_id}/assessment` | One grounded assessment; returns `assessment_state: unavailable` while the provider is off. |
| `GET/POST /api/v1/saved-jobs`, `PATCH`/`DELETE /api/v1/saved-jobs/{snapshot_id}/{job_id}` | Saved jobs with workflow `stage` and `note`. |
| `POST /api/v1/profile-extraction` | Magic fill: body has exactly one of `resume_text` or `resume_pdf_base64`. Stateless: nothing is stored or saved. Returns suggested profile fields, or `state: unavailable` when disabled. |

### Resume extraction rules

- Text-based PDFs are read locally with `pypdf` (2 MB and 10 pages at most; scanned PDFs are rejected). Emails, phone numbers and links are removed before any text is sent.
- Every returned experience, project, education or skill item must include a quote found verbatim in the resume; items that fail are dropped and counted. Desired roles, arrangements and constraints are never extracted.
- Skill "evidence" lines are the model's attribution and are not covered by the quote check.

## Tests

Source-shaped fixtures and the SQLite database are created in temporary directories; no saved project data is touched. No test makes a network call (providers are injected fakes).

```powershell
.\run_tests.ps1
```

Coverage includes session isolation, reset and expiry; profile versioning; search and pagination; recommendations and feedback; the RRF ranker with a fake encoder; saved-job idempotency; assessment and resume-extraction behavior, including PDF handling and contact-detail redaction.

## Not provided

- No claim that the ranking is good: relevance has not been validated for this audience.
- No live validation of the grounded assessment, and no web UI for it yet.
- No companies-file join, current-availability check, duplicate/repost policy, production authentication, account recovery, deployment hardening or data-retention policy.
