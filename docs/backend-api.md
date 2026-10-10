# Demo Backend API

## Status

Implemented as a minimal FastAPI prototype in `backend/api.py`. It is deliberately not production authentication or a production data service. It provides:

- Anonymous, session-isolated demo profiles and saved jobs in SQLite, with a sliding 24-hour default TTL and `DELETE /api/v1/demo-sessions/current` reset.
- Profile create/read/update with version checks (`If-Match`).
- Read-only search and job/company details from one prepared historical snapshot.
- Personalized recommendations through an injectable ranker adapter. The configured implementation is `tfidf_cosine`, delegated to `job_matching_engine.fit_tfidf` and `job_matching_engine.rank_tfidf`, version `job_matching_engine.rank_tfidf-v1`. The corpus and TF-IDF matrix are fit once per app lifespan, not per request.
- A wrapper around `grounded_match_explanations.explain_match`. Provider generation is disabled by default. Sufficient evidence with no provider yields `assessment_state: unavailable`, never fabricated content. Insufficient evidence returns the service's validated insufficient result without calling a provider. Tests inject only an offline provider.
- Validated assessment caching keyed by session, profile/version, job/snapshot, candidate/job content hash, requested analyses, model, prompt hash and schema version. Provider and response-validation failures are not cached.

The React app has not been connected and continues to use fixtures and browser storage. Ranking quality is not established. Scores are cosine-similarity retrieval signals, not match percentages. Current job availability is unverified.

## Windows Setup

Run these commands from the repository root in PowerShell. The backend environment is separate from the active ML `.venv`.

```powershell
& ".\.venv\Scripts\python.exe" -m venv .\backend\.venv
& ".\backend\.venv\Scripts\python.exe" -m pip install -r .\backend\requirements.txt

$env:PREDICTLEADS_API_JOBS_PATH = (Resolve-Path ".\data\discovery\role_exploration\roles_corrected_20261006T000000Z\jobs_prepared_random_sample.jsonl").Path
$env:PREDICTLEADS_API_DB_PATH = (Join-Path $PWD "backend\data\predictleads-demo.sqlite3")
$env:PREDICTLEADS_API_SESSION_TTL_SECONDS = "86400"
$env:PREDICTLEADS_API_COOKIE_SECURE = "0"

& ".\backend\.venv\Scripts\python.exe" -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the local OpenAPI view. The initial app startup reads the configured JSONL once and fits the TF-IDF matrix; startup can take time and memory. It does not rebuild embeddings or contact an external provider. The sample path defaults to the same saved 30,000-row file if the environment variable is omitted.

Configuration:

| Environment variable | Default | Purpose |
|---|---|---|
| `PREDICTLEADS_API_JOBS_PATH` | `data/discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl` | Read-only prepared JSONL source. |
| `PREDICTLEADS_API_DB_PATH` | `backend/data/predictleads-demo.sqlite3` | Session/profile/saved-job/assessment-cache database. Database files are ignored by Git rules. |
| `PREDICTLEADS_API_SESSION_TTL_SECONDS` | `86400` | Sliding anonymous session expiry; minimum 60 seconds. |
| `PREDICTLEADS_API_COOKIE_SECURE` | `0` | Set to `1` when served over HTTPS. Keep `0` only for local HTTP development. |
| `PREDICTLEADS_API_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Allowed credentialed browser origins. |

## Implemented Routes

All data routes except session creation/reset require the `pl_demo_session` HttpOnly cookie. It is anonymous session isolation, not login. Snapshot identity is required for job-specific operations.

| Method and route | Purpose |
|---|---|
| `GET /healthz` | Liveness only; does not require a session. |
| `POST /api/v1/demo-sessions` | Create a random demo session and cookie. |
| `DELETE /api/v1/demo-sessions/current` | Reset/delete the current session, its profile, saved jobs and assessment cache; expire the cookie. |
| `POST /api/v1/profiles` | Create the session's single profile. |
| `GET /api/v1/profiles/{profile_id}` | Read the profile and version. |
| `PATCH /api/v1/profiles/{profile_id}` | Partial update; requires `If-Match: "{version}"`. |
| `GET /api/v1/jobs?snapshot_id=...` | Cursor-paginated source-backed search. Filters: `q`, `location`, `description=any|with|without`, exact `category`, and exact `seniority`. Work-arrangement filtering returns `400 unsupported_filter` because it is not present in the prepared sample. |
| `GET /api/v1/jobs/{job_id}?snapshot_id=...` | Full prepared job details and provenance. |
| `GET /api/v1/companies/{company_id}?snapshot_id=...` | Only same-wrapper company-lite values observed in the loaded sample. It does not join the separate companies file. |
| `POST /api/v1/profiles/{profile_id}/recommendations` | TF-IDF recommendations for a profile version and snapshot, with optional supported search filters. |
| `POST /api/v1/profiles/{profile_id}/jobs/{job_id}/assessment` | One selected candidate/job grounded assessment; never ranks. Disabled mode returns unavailable for sufficient evidence. |
| `GET /api/v1/saved-jobs?snapshot_id=...` | List current-session saved jobs, optionally filtered by snapshot. |
| `POST /api/v1/saved-jobs` | Idempotently save `(snapshot_id, job_id)`. |
| `PATCH /api/v1/saved-jobs/{snapshot_id}/{job_id}` | Update candidate workflow `stage` and/or `note`. |
| `DELETE /api/v1/saved-jobs/{snapshot_id}/{job_id}` | Delete only that session's saved reference. |

Responses retain `field_states`, raw source status, `availability: unverified`, snapshot date/ID and provenance. Company descriptions, locations, size and industry are not synthesized. Missing/null/blank/absent source values remain distinct. Free-text candidate preferences are not hard filters.

## Tests

The source-shaped fixtures and SQLite database are created in temporary directories; no saved project data is modified. Tests use the actual TF-IDF adapter on a tiny corpus and injected test providers only.

```powershell
& ".\backend\.venv\Scripts\python.exe" -m unittest discover -s .\backend\tests -v
```

Coverage includes session isolation/reset/expiry, profile reload/version conflicts, search filtering and cursor pagination, snapshot-specific job/company reads, actual `tfidf_cosine` metadata, saved-job idempotency and persistence across app reload, plus assessment disabled/insufficient/mock-success/cache/error behavior.

## Unverified / Next

- No live GenAI call was made for this API. Provider output quality and paid-provider operation remain unverified; live assessment stays disabled unless an application factory receives an explicitly injected provider.
- No ranking-quality claim follows from API operation or offline tests. Human relevance validation and final model selection remain separate work.
- No separate companies-file join, current availability check, duplicate/repost policy, production auth, account recovery, deployment hardening or user-data retention policy is provided.
- Frontend API client integration is still pending. Next connect demo-session creation, profile load/save, search/recommendations, details/assessment and saved-job mutations; map nullable company names and unknown work arrangement without fabricated display values.
- The local TF-IDF fitting pass uses the existing sample and scikit-learn only. It does not read experiment ledgers, invoke embedding models, make paid calls or modify the prepared source file.

## Update 2026-10-09: ranking choice, relevance feedback, frontend connection

- **Ranker option.** `PREDICTLEADS_API_RANKER=rrf` (with `PREDICTLEADS_API_CORPUS_DIR`) serves the equal-weight TF-IDF + BGE reciprocal-rank-fusion ranker from `backend/rrf_ranker.py`. The default stays TF-IDF. `run_demo_backend.ps1` starts the RRF ranker on the serving corpus (`data/matching/ai_ml_serving_v1`, built by `python code/rank_candidates.py export-serving`: the 81,284 listings whose source status is not explicitly closed). Startup takes about 3.5 minutes and about 2.5 GB. The RRF ranker is the working reference, **not a validated winner**: no human review of its results was completed.
- **Relevance feedback.** New session-scoped table `job_feedback` and routes `PUT /api/v1/feedback` (`{job_id, snapshot_id, signal: "good"|"bad"}`, idempotent overwrite), `GET /api/v1/feedback?snapshot_id=`, `DELETE /api/v1/feedback/{snapshot_id}/{job_id}`. When a session has feedback, `POST .../recommendations` moves the query toward jobs marked good and away from jobs marked bad (Rocchio-style, constants in `backend/feedback_ranking.py`: liked weight 1.0, disliked weight 0.5, applied to both the TF-IDF and the BGE query before fusion). Rated jobs are left out of new results. Results include `similar_to_liked` (the closest liked job and its cosine similarity) and the response includes a `feedback` block (`good`, `bad`, `applied`, `method`, `note`). This adapts one session's ranking only. It does not retrain any model and makes no accuracy claim.
- **Other new routes.** `GET /api/v1/snapshots/current` (no session needed; snapshot id, date, job count, ranker, availability "unverified"), `GET /api/v1/profiles/current` (the session's profile, 404 if none). `GET /api/v1/saved-jobs` and the recommendation items now include job summaries.
- **Profile mapping.** The API profile's headline and arrangements now reach the rankers (as keywords and work-arrangement text) like the experiment profiles' soft preferences.
- **Tests.** `backend/tests/test_feedback.py` and `test_rrf_ranker.py` run offline: `backend\.venv\Scripts\python.exe -m unittest discover -s backend\tests -t .`
- **Frontend.** The React app now talks to this API through the Vite `/api` proxy: profile (create/edit), ranked recommendations with "Good match" / "Not for me" buttons, job details with a rule-based skills check ("not evidenced in your profile" wording), and saved jobs. Fixture data was removed. See `STATUS.md`.
