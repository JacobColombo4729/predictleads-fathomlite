# Demo Backend API

A minimal FastAPI service in `backend/api.py` for the demo. It is deliberately **not** production authentication or a production data service. Responses keep source nulls, raw status, `availability: unverified` and snapshot provenance; nothing is synthesized.

## What it provides

- Anonymous, session-isolated profiles, saved jobs, ratings and cached assessments in SQLite, with a sliding 24-hour TTL and `DELETE /api/v1/demo-sessions/current` to reset everything.
- Profile create/read/update with version checks (`If-Match`). Each experience entry has a `kind` (`job`, `project` or `education`, default `job`) that only affects display; matching reads every entry the same way.
- Read-only search and job/company details from one historical snapshot (the 81,284 listings whose source status is not explicitly closed).
- Search words (`filters.query`, also on `GET /api/v1/jobs`): every word must appear, in any order. Recommendations search **job titles first**; only when fewer than 5 titles match do they fall back to the whole listing text (title, occupation, description, company, tags, categories). The response's `search` object reports the query and the `scope` used (`all`, `title` or `anywhere`).
- Personalized recommendations through a replaceable ranker. The demo runs `tfidf_bge_rrf_equal` (`backend/rrf_ranker.py`, equal-weight reciprocal-rank fusion of TF-IDF and BGE cosine, depth 50, k=60); `PREDICTLEADS_API_RANKER=tfidf` serves TF-IDF alone. Scores are retrieval signals, not match percentages or hiring probabilities.
- Relevance feedback: 👍/👎 on jobs per session. Recommendations then move the query toward liked and away from rejected jobs (Rocchio-style, applied to both signals) and hide rated jobs. The model is not retrained, and ratings disappear with the session.
- **Compass** (`backend/compass.py`): analysis of 2-20 picked jobs against the saved profile, entirely local (stored BGE vectors and the local BGE model; no outside AI call). See below.
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
| `POST /api/v1/profiles/{profile_id}/compass` | Compass analysis. Body: `snapshot_id`, `profile_version`, `job_ids` (2-20). Needs the embedding (RRF) ranker, else `409 compass_unavailable`. About 4 seconds on CPU. |
| `POST /api/v1/profile-extraction` | Magic fill: body has exactly one of `resume_text` or `resume_pdf_base64`. Stateless: nothing is stored or saved. Returns suggested profile fields, or `state: unavailable` when disabled. |

### Resume extraction rules

- Text-based PDFs are read locally with `pypdf` (2 MB and 10 pages at most; scanned PDFs are rejected). Emails, phone numbers and links are removed before any text is sent.
- Every returned experience, project, education or skill item must include a quote found verbatim in the resume; items that fail are dropped and counted. Desired roles, arrangements and constraints are never extracted.
- Skill "evidence" lines are the model's attribution and are not covered by the quote check.

## Compass analysis

`POST /api/v1/profiles/{id}/compass` takes the jobs a person picked on the Saved page and returns:

- **themes** (at most 3): clusters of the picks, named from the listings' own O*NET occupation label (first clause) or common title words, each with a compass heading for the map;
- **jobs**: each pick and up to five suggestions with a `fit` (0-100), an angular offset for the map, up to two `why` lines (a verbatim profile line next to the listing line it matches) and `stretch`/`builds` requirement ids;
- **requirements**: requirement and duty sentences from the picks, clustered by meaning, with how many picks mention each (`n`), the closest profile line as `evidence`, a `coverage` (0-100) and a `status` of `supported`, `partial` or `unknown` ("not mentioned"). Lines that mention years, degrees or certifications get `verify: true` and are never marked supported;
- **next_steps**: the unsupported groups that appear most often (`n` times how little the profile shows it);
- stats, the fit scale, and the method limits.

How it works: the picks' stored embeddings give themes and fit; the profile is encoded the same way as the ranker's query. `fit` scales the cosine similarity between the profile and a listing from the 5th percentile of all listings (0) to the single best match (100). Requirement sentences come from `bge_text._job_sections` (benefits, EEO and company-pitch sections are skipped), are embedded in one batch, merged when their cosine distance is under 0.22, and compared with the person's own lines (experience, projects, education, skills; the headline is not evidence). Suggestions are the listings nearest a blend of the profile and the picks, excluding the picks, anything the person rated, near-copies of a pick's title and duplicates, diversified with maximal marginal relevance and a small push away from near-copies of the picks.

Everything shown as evidence is a verbatim line from the profile or a listing. All thresholds (coverage 0.50-0.80 cosine, supported at 70, partial at 40, explanation similarity 0.66) are heuristics tuned by eye on a few real examples; **they have not been validated**. Scores are similarities, not hiring probabilities, and listings are historical.

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
