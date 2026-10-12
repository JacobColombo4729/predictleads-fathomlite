# Proposed API Contracts

> **Note (2026-10-09):** these are the planned contracts; the implemented routes are in [backend-api.md](backend-api.md), which is the source of truth where they differ.

**Status: implemented demo API.** The routes below are served by the FastAPI prototype documented in [backend API](backend-api.md). This is not production authentication; the React app remains fixture/localStorage-backed until connected. Ranking uses the existing `job_matching_engine.rank_tfidf` implementation. Live assessment generation is disabled by default. See [frontend/backend map](archive/frontend-backend-map.md), [status](../../analysis/STATUS.md), [frontend types](../frontend/src/types.ts), [matching engine](../code/job_matching_engine.py), and [grounded explanation service](../code/grounded_match_explanations.py).

Use JSON over HTTPS at `/api/v1`. Keep the ranker behind a server interface so keyword, TF-IDF, embedding, or hybrid implementations can be replaced without changing client request shapes. Examples are illustrative, not claims about current data availability.

## Shared Data Rules

- Preserve source job IDs exactly as stable `job_id` values. Preserve company references as `{ "type": "company", "id": "..." }`; do not substitute display names for IDs.
- Preserve `snapshot_id`, snapshot date and available source-line provenance in search, recommendations, job/company details, assessments and saved-job identity. A job record from the September 30, 2026 snapshot is historical, not a current-opening feed.
- For source-derived fields, return normalized values plus `field_states` where relevant. States are `present`, `null`, `blank`, `absent`, or `invalid`; do not collapse these into one empty string. Empty candidate arrays mean “no evidence supplied,” not “candidate lacks these skills.”
- Keep raw status separate: `closed` is an explicit source value; null/absent status is `unknown`. Return availability as `unverified` unless a separate current check actually establishes it. Do not infer closure date, hiring likelihood, competition or employer intent.
- All timestamps are UTC RFC 3339. Pagination cursors are opaque. Unknown response fields should be ignored by clients; requests use a versioned route.
- Recommendation scores are uncalibrated retrieval signals. Do not label them match percentages, probabilities, eligibility decisions or hiring predictions.

A representative job payload can be shared by search, recommendation and details responses:

```json
{
  "job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa",
  "title": "Purchasing and Logistics Coordinator",
  "normalized_title": "Purchasing and Logistics Coordinator",
  "occupation": {"code": "13-1023.00", "name": "Purchasing Agents", "validated": false},
  "company_ref": {"type": "company", "id": "89df4bc2-62ca-5508-a99b-9f17f0f841ae"},
  "company_summary": {"company_id": "89df4bc2-62ca-5508-a99b-9f17f0f841ae", "name": "Bourns, Inc.", "domain": "bourns.com"},
  "location": null,
  "work_arrangement": {"value": null, "source": "not_in_sample"},
  "salary_raw": null,
  "description_available": true,
  "description_excerpt": "Purchasing & Logistics Coordinator acts as the primary contact...",
  "source_url": "https://example.invalid/listing",
  "source_status": {"raw": null, "state": "unknown"},
  "availability": {"state": "unverified", "verified_at": null},
  "observed": {"first_seen_at": "2026-06-03T00:30:35Z", "last_seen_at": "2026-06-03T00:57:37Z", "last_processed_at": "2026-06-05T01:04:36Z", "posted_at": null},
  "snapshot": {"snapshot_id": "predictleads-2026-09-30", "date": "2026-09-30", "source_line": 565144},
  "field_states": {"location": "null", "salary_raw": "null", "description": "present", "posted_at": "null"}
}
```

`company_summary` is nullable and may be absent when the same-wrapper reference cannot be resolved. Company names may also be null in the source; the current frontend's non-null `Company.name` therefore needs an adapter/display fallback that preserves unknown rather than inventing a name. The prepared sample has no verified work-arrangement field; return it as unknown/not present, and reject a work-arrangement filter as unsupported rather than infer one. Occupation is source-assigned and not accuracy-validated. The example timestamps are illustrative and need not match the example snapshot date.

## Candidate Profile

`POST /api/v1/profiles` creates the one profile for the current demo session and returns a server-assigned stable `profile_id` and version. `GET /api/v1/profiles/{profile_id}` retrieves it. `PATCH /api/v1/profiles/{profile_id}` updates supplied fields only; omitted fields remain unchanged. `null` clears nullable `location`, `constraints`, or an experience `organization`; non-null collections use `[]` to clear, and `headline` uses `""` to clear. Require `If-Match: "{version}"` on updates to prevent lost edits. A `skills: []` value records no supplied evidence; it is not a negative assessment.

```http
POST /api/v1/profiles
Content-Type: application/json
```

```json
{
  "name": "Demo Candidate",
  "headline": "Customer support specialist exploring technical roles",
  "experience": [{"id": "exp-1", "title": "Tier 2 Support Specialist", "organization": null, "description": "Escalated tickets and documented reproducible bugs."}],
  "skills": [{"id": "skill-1", "name": "SQL", "evidence": "Writes account-reporting queries."}],
  "desired_roles": ["Technical Support Engineer"],
  "location": "Austin, TX",
  "arrangements": ["Remote", "Hybrid"],
  "constraints": null
}
```

```json
{
  "profile_id": "prf_01J...",
  "version": 1,
  "updated_at": "2026-10-08T18:30:00Z",
  "profile": {"name": "Demo Candidate", "headline": "Customer support specialist exploring technical roles", "experience": [{"id": "exp-1", "title": "Tier 2 Support Specialist", "organization": null, "description": "Escalated tickets and documented reproducible bugs."}], "skills": [{"id": "skill-1", "name": "SQL", "evidence": "Writes account-reporting queries."}], "desired_roles": ["Technical Support Engineer"], "location": "Austin, TX", "arrangements": ["Remote", "Hybrid"], "constraints": null}
}
```

Map the richer frontend profile to the Python `CandidateProfile` at the ranking boundary: retain experience statements, skills and their evidence, target roles, location and preferences. Do not turn free-text constraints into hard filters; only a separately typed, explicit constraint may be enforced.

## Job Search and Pagination

`GET /api/v1/jobs?snapshot_id=predictleads-2026-09-30&q=support&location=Austin%2C%20TX&description=with&category=operations&seniority=junior&limit=20&cursor=...` returns source-backed job summaries. Supported filters are query, location, description state, category and seniority. Work arrangement is unavailable in this prepared sample; requesting it returns `400 unsupported_filter`. Unknown filter values also return `400`, not silently broaden results. A no-results search returns `200` with an empty `items` array.

```json
{
  "items": [{"job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa", "title": "Purchasing and Logistics Coordinator", "company_summary": {"company_id": "89df4bc2-62ca-5508-a99b-9f17f0f841ae", "name": "Bourns, Inc.", "domain": "bourns.com"}, "location": null, "work_arrangement": {"value": null, "state": "unknown"}, "salary_raw": null, "description_available": true, "description_excerpt": "Purchasing & Logistics Coordinator acts as the primary contact...", "source_url": "https://example.invalid/listing", "observed_on": "2026-09-30", "source_status": {"raw": null, "state": "unknown"}, "availability": {"state": "unverified", "verified_at": null}, "snapshot": {"snapshot_id": "predictleads-2026-09-30", "date": "2026-09-30"}, "field_states": {"location": "null", "salary_raw": "null"}}],
  "page": {"limit": 20, "next_cursor": null},
  "snapshot": {"snapshot_id": "predictleads-2026-09-30", "date": "2026-09-30"}
}
```

## Personalized Recommendations

`POST /api/v1/profiles/{profile_id}/recommendations` accepts required `snapshot_id`, profile version, top-k and the same documented search filters. The API, not the browser, selects a ranker. Return the actual ranker identifier/version and score type for reproducibility; keep results tied to a snapshot.

```json
{"snapshot_id": "predictleads-2026-09-30", "profile_version": 1, "limit": 10, "filters": {"query": "support", "location": "Austin, TX", "description": "with"}}
```

```json
{
  "profile_id": "prf_01J...",
  "profile_version": 1,
  "ranking": {"ranker_id": "tfidf_cosine", "version": "job_matching_engine.rank_tfidf-v1", "score_type": "cosine_similarity", "calibration": "uncalibrated"},
  "snapshot": {"snapshot_id": "predictleads-2026-09-30", "date": "2026-09-30"},
  "items": [{"rank": 1, "retrieval_score": 0.31, "job": {"job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa", "title": "Purchasing and Logistics Coordinator", "company_summary": {"company_id": "89df4bc2-62ca-5508-a99b-9f17f0f841ae", "name": "Bourns, Inc.", "domain": "bourns.com"}, "location": "Austin, TX", "work_arrangement": {"value": null, "state": "unknown"}, "salary_raw": null, "description_available": true, "description_excerpt": "Purchasing & Logistics Coordinator acts as the primary contact...", "source_url": "https://example.invalid/listing", "observed_on": "2026-09-30", "source_status": {"raw": null, "state": "unknown"}, "availability": {"state": "unverified", "verified_at": null}, "snapshot": {"snapshot_id": "predictleads-2026-09-30", "date": "2026-09-30"}}}]
}
```

`retrieval_score` orders results only within the declared ranking method/version. Do not return fabricated `Strong fit` labels. Grounded evidence is requested separately.

## Job and Company Details

- `GET /api/v1/jobs/{job_id}?snapshot_id=predictleads-2026-09-30` returns the full job representation above, including raw/structured source fields, field states and provenance. `snapshot_id` is required; it is part of the record identity.
- `GET /api/v1/companies/{company_id}?snapshot_id=predictleads-2026-09-30` returns same-wrapper company-lite values only: `{ "company_id": "...", "name": null, "domain": "...", "description": null, "location": null, "snapshot": {...}, "source_scope": "same_wrapper_company_lite_only", "full_company_record_loaded": false, "field_states": {"name": "null", "description": "absent"} }`.

Company fields such as industry, size and headquarters are nullable and must not be fabricated. This prototype only returns same-wrapper company-lite data from jobs in the loaded sample; no separate companies-file join is performed. If no same-wrapper reference exists in that sample, it returns `404 company_not_in_loaded_snapshot`. Source URLs are untrusted input: allow only `http`/`https` at render time.

## Grounded Explanation and Basic Skills-Gap Analysis

`POST /api/v1/profiles/{profile_id}/jobs/{job_id}/assessment` assesses one selected pair; it never ranks or chooses a job. Include `profile_version` and `snapshot_id`, and require that the job exists in that snapshot. The basic skills-gap view is a projection of explicitly evidenced qualifications, not a corpus-level skill extractor or the fuller Skill Opportunity Optimizer.

```json
{"profile_version": 1, "snapshot_id": "predictleads-2026-09-30", "include": ["transferable_experience", "qualification_assessment", "basic_skills_gap"]}
```

```json
{
  "assessment_state": "completed",
  "cache": "miss",
  "profile_id": "prf_01J...",
  "profile_version": 1,
  "job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa",
  "snapshot_id": "predictleads-2026-09-30",
  "requested_analyses": ["transferable_experience", "qualification_assessment", "basic_skills_gap"],
  "schema_version": "grounded_match_explanation_v1",
  "assessment": {
    "schema_version": "grounded_match_explanation_v1",
    "profile_id": "prf_01J...",
    "job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa",
    "evidence_sufficiency": "sufficient",
    "insufficiency_reason": null,
    "transferable_experience": [{"job_duty": "Investigate reporting issues", "candidate_evidence_ids": ["candidate.experience[0].description"], "job_evidence_ids": ["job.description_text[0]"], "rationale": "Both excerpts describe troubleshooting reporting issues."}],
    "qualifications": [{"qualification_id": "q-sql", "qualification": "SQL reporting", "classification": "required", "status": "supported", "candidate_evidence_ids": ["candidate.skills[0].name"], "job_evidence_ids": ["job.description_text[1]"], "rationale": "The profile explicitly lists SQL as a skill."}],
    "next_steps": [],
    "evidence_catalog": [{"evidence_id": "candidate.experience[0].description", "source": "experience[0].description", "text": "Troubleshoots reporting issues."}, {"evidence_id": "candidate.skills[0].name", "source": "skills[0].name", "text": "SQL"}, {"evidence_id": "candidate.skills[0].evidence", "source": "skills[0].evidence", "text": "Writes account-reporting queries."}, {"evidence_id": "job.description_text[0]", "source": "description_text[0]", "text": "Investigate customer reporting issues."}, {"evidence_id": "job.description_text[1]", "source": "description_text[1]", "text": "SQL reporting experience required."}]
  }
}
```

The actual Python schema requires exact evidence IDs, preserves `required`/`preferred`/`unclear` and `supported`/`explicit_gap`/`unknown`, and returns `insufficient` with empty assessment arrays when source or candidate evidence is too sparse. A basic UI may label unknown skill evidence as “not evidenced in your profile”; it must not imply the skill is absent, unnecessary, or a disqualifier. Provider failures are unavailable/error states, never negative assessments. Keep the service's `evidence_catalog` intact for rendering citations.

**Provider and cache behavior:** provider use is disabled by default. If evidence is sufficient while disabled, return `200` with `assessment_state: "unavailable"`, `reason: "provider_disabled"`, and `assessment: null`; do not invent explanation text. If the service determines evidence is insufficient, return `assessment_state: "completed"` with the real `insufficient` result without calling a provider. A configured-provider failure returns `502`/`503`, is not cached, and is never converted into an assessment. Cache only validated service results (including deterministic insufficient results) by session ID, profile ID/version, job ID, snapshot ID, canonical source/candidate content hash, model ID, prompt hash and `SCHEMA_VERSION`. A provider/schema/content version change must produce a cache miss.

Disabled, sufficient-evidence response:

```json
{"assessment_state":"unavailable","reason":"provider_disabled","profile_id":"prf_01J...","profile_version":1,"job_id":"0acf8fe5-57e4-409e-91c8-951775be8caa","snapshot_id":"predictleads-2026-09-30","requested_analyses":["transferable_experience","qualification_assessment","basic_skills_gap"],"schema_version":"grounded_match_explanation_v1","assessment":null,"cache":"miss"}
```

## Saved Jobs

`GET /api/v1/saved-jobs?snapshot_id=predictleads-2026-09-30` lists records for the current demo session, optionally filtered to a snapshot. `POST /api/v1/saved-jobs` saves a job ID plus `snapshot_id`; repeated saves are idempotent for that pair. `PATCH /api/v1/saved-jobs/{snapshot_id}/{job_id}` updates only `stage` and/or `note`. `DELETE /api/v1/saved-jobs/{snapshot_id}/{job_id}` removes the user's saved record, not the source job.

```json
{"job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa", "snapshot_id": "predictleads-2026-09-30"}
```

```json
{"job_id": "0acf8fe5-57e4-409e-91c8-951775be8caa", "snapshot_id": "predictleads-2026-09-30", "stage": "Saved", "note": "", "saved_at": "2026-10-08T18:35:00Z"}
```

Stages are `Saved`, `Considering`, `Applied`, `Interviewing`, and `Archived`. They describe the candidate's workflow, never the listing's status. Notes are private to the session and capped at 500 characters. If a listing disappears from current search, retain its saved record and historical snapshot reference.

## Errors and State Handling

Use a consistent error body and HTTP status:

```json
{"error": {"code": "profile_version_conflict", "message": "Reload the profile before saving changes.", "request_id": "req_...", "retryable": false}}
```

| Status | Example codes | Client behavior |
|---|---|---|
| `400` | `unsupported_filter`, `invalid_cursor`, `invalid_if_match`, `empty_update` | Show validation feedback; do not silently change filters. |
| `401` | `demo_session_required`, `demo_session_expired` | Start a new demo session; do not describe it as account login. |
| `404` | `profile_not_found`, `job_not_found`, `snapshot_not_found`, `company_not_in_loaded_snapshot`, `saved_job_not_found` | Preserve other user state and explain which record is unavailable. |
| `428` | `if_match_required` | Reload the profile version, then retry with `If-Match`. |
| `403` | `origin_not_allowed` | Do not retry from an unconfigured origin. |
| `409` | `profile_version_conflict` | Reload/merge before retrying. |
| `422` | `validation_error` | Show field errors or mark assessment unavailable; never invent a result. |
| `502` | `provider_invalid_response`, `assessment_validation_failed` | Offer retry or source-only fallback; do not turn failure into a weak match. Provider failures are not cached. |

Sparse source data is not necessarily an HTTP error: return `200` with an empty search page or an assessment whose `evidence_sufficiency` is `insufficient`. Keep source field states intact in both cases.

## Minimal Hackathon Persistence and Session

For a controlled demo, `POST /api/v1/demo-sessions` creates a random opaque session and sets an `HttpOnly` cookie; its JSON response is `{ "session": { "expires_at": "...", "authentication": "anonymous_demo_session" } }`. Scope exactly one profile and its saved jobs to that session. `DELETE /api/v1/demo-sessions/current` resets it by cascading deletion of its profile, saves and assessment cache, then expires the cookie. Use a configurable sliding TTL (default 24 hours); each authenticated request validates expiry and refreshes activity, while an expired session is rejected with `401 demo_session_expired` and its data is eligible for cleanup. Expired rows are deleted opportunistically during session creation and may also be removed by a maintenance task. Use `SameSite=Lax`, `Secure` under HTTPS, and origin/CSRF checks for writes. This is anonymous demo session isolation, **not production authentication or an account system**. Do not put the session secret in browser storage or expose it to JavaScript.

Use SQLite for demo sessions, profile JSON/version, saved-job rows and validated assessment cache. Keep the prepared historical job corpus read-only and keyed by `(snapshot_id, job_id)`; keep ranking resources/results separate from user state. Run expiry cleanup opportunistically at session creation and through an explicit maintenance hook. Use synthetic profiles only unless consent and data handling are explicitly approved. Production would require real authentication, authorization, privacy/retention review, operational backups and a deployment-specific persistence decision.

## Implementation Boundary

Implemented: demo session cookies and SQLite session/profile/saved-job/cache storage; versioned profile CRUD; read-only search/details for the prepared snapshot; TF-IDF ranking through an injectable adapter; and an adapter to the standalone grounded-explanation service. Provider generation is disabled by default and emits an explicit unavailable state for sufficient evidence. This remains a prototype: there is no account authentication, production persistence, full companies-file join, current-opening verification, validated ranking quality, live explanation quality, or frontend API connection. The React UI still uses mock data until its separate integration work is done.
