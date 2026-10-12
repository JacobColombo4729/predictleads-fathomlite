# Frontend / Backend Integration Map

> **Update 2026-10-09:** the screen table below describes the earlier mock-only state. The React app now calls the demo API (profile, ranked recommendations, relevance feedback, details, saved jobs); fixtures were removed. See [STATUS.md](../../../analysis/STATUS.md) and [backend-api.md](../backend-api.md). Current availability is still unverified, and company details beyond name and domain are still unknown.

## Current State

The React app is a client-side prototype. It has no API client, backend server, authentication, or shared database. `App.tsx` loads fixed `demoRecommendations`; `useProfile` and `useTracker` persist versioned state in this browser's `localStorage`. The Python matching and review tools are command-line/offline experiments, not endpoints. They run on saved historical sample data and synthetic profiles. The provider-backed requirement assessor validates source excerpts and preserves unknowns, but is also an experiment, not a production service.

Status labels below: **Implemented** means in the prototype UI/local browser state; **Experimental** means an offline Python capability or saved evaluation; **Planned** means no connected product capability exists.

## Screen Map

| Screen | Information needed | Existing support | Missing endpoints or capabilities | Fictional/demo data to replace |
|---|---|---|---|---|
| **My Profile** | Name/headline; experience and evidence; skills with evidence; target roles; location, work arrangement and constraints. | **Implemented:** editable, validated profile and preview. Versioned local persistence only. **Experimental:** Python `CandidateProfile` carries experience, evidenced skills, desired roles, location and preferences. | **Planned:** authenticated profile read/create/update/delete; server validation and schema migration; consent, access and retention rules before storing real resume/profile text. No account identity exists. | `demoProfile` (including Maya Okafor, employer, skills and history); reset-to-demo behavior. Do not seed fictional profiles into real accounts.
| **Discover Jobs** | Searchable listings and filters; profile-aware ranked results; title, company, location, arrangement, salary, description, observation date, source status and evidence. | **Implemented:** client-side text/location/arrangement/description filters over a fixed array. **Experimental:** offline keyword and TF-IDF ranking and other saved ranking experiments. The rank score is a retrieval signal, not a validated match percentage, probability, eligibility finding or hiring prediction. | **Planned:** paginated/filterable listing search; recommendation request keyed to profile and filters; stable job IDs, ranking method/version and score type; explicit missing/unknown states; current-availability verification (not provided by the historical snapshot). | All 10 `demoRecommendations`, demo order/fit labels, salaries, arrangements, observed dates, descriptions and `example.com` links. The sample corpus is historical (2026-09-30), not a live openings feed.
| **Job / Company Details** | Job source fields and provenance; source link; historical status and snapshot date; company name/domain/description/location; match evidence and requirements context. | **Implemented:** a details panel with Overview, Match evidence and Company tabs. **Experimental:** prepared records retain raw/prepared job text, tags, O*NET, status/date quality flags and same-wrapper company fields. The experimental requirement assessor returns exact job/candidate excerpts, mandatory/preferred/unclear requirements, supported/contradicted/unknown states and unknowns. | **Planned:** job detail endpoint; company lookup/enrichment endpoint and verified separate-file join; source-span evidence endpoint or validated assessment payload; separate current-availability check. Existing company data does not establish every displayed field: industry, size and headquarters are not guaranteed by the job wrapper. | Mock company narratives, industries, size/headquarters, dates/status, links and every fit explanation. Avoid turning absent fields into claims. Keep `closed`, null/unknown, missing and not verified distinct; disappearance is not a filled job.
| **Saved Jobs** | Saved job IDs and stable job summary; user's stage, note and save time; search/stage filters; detail access after ranking changes. | **Implemented:** save/remove/undo, stage, notes, search and stage filtering, stored locally. First visit seeds three fictional saves. Stages are candidate workflow state, not listing status. | **Planned:** authenticated saved-job list/create/update/delete endpoints; persist stage/note per user; handle unavailable or removed listings while retaining the user's record; sync across devices and resolve concurrent updates. | `demoTracker()` seeds `job-01`, `job-04`, `job-07`, with fictional notes and dates. Clear demo seeds before production; do not assume an ID remains in the current recommendation response.

## Grounded Explanations and Skills Gap

The UI already has an evidence tab, but its `fit`, `fitEvidence` and `requirementsToCheck` fields are static fixtures and do not update when the profile changes. The experimental assessor is a useful starting contract, not a production endpoint. For each explanation or requirement, return the exact job excerpt and candidate excerpt where available, requirement classification (`mandatory`, `preferred`, `unclear`), evidence state (`supported`, `explicitly_contradicted`, `unknown`), rationale, unknowns and evidence sufficiency. Validate excerpts against the stored source text. Missing candidate evidence must remain **unknown**, never silently count as a gap or disqualifier. Do not display an experimental duty score as a percentage.

A basic **skills-gap** view can compare candidate-evidenced skills with skills explicitly mentioned in a selected listing and show the listing excerpt and context. It must say “not evidenced in your profile” rather than “you lack this skill.” Do not label extracted tags as verified skills or requirements. The fuller **Skill Opportunity Optimizer** additionally needs validated skill extraction and corpus-level co-occurrence/role analysis; it is planned research, not a current API capability. Preserve extraction uncertainty and source spans.

## Backend Integration Checklist

1. Define API ownership, authentication, profile consent and retention, stable IDs, snapshot provenance, and a shared distinction among absent, null, blank and unknown.
2. Freeze versioned request/response schemas aligned with `Profile`, `Job`, `Company`, `Recommendation` and `TrackedJob`; add explicit `observed_at`, `snapshot_date`, raw source status and availability verification state. Do not treat the historical snapshot as live.
3. Add authenticated profile CRUD, then saved-job CRUD keyed by user and stable job ID. Remove fictional seed data from production initialization.
4. Add paginated job search and job/company detail reads from an approved prepared-data store. Test separate companies-file joins; do not infer company attributes missing from the joined record.
5. Connect the recommendation endpoint to the selected ranking implementation. Return rank and method/version metadata; label scores as uncalibrated retrieval signals until evaluation supports stronger claims.
6. Integrate match explanations as a separate, evidence-validated assessment contract. Preserve unknowns, enforce source excerpts, and keep failed/insufficient assessments visible as such.
7. Validate skill extraction before exposing a basic skills-gap view; add corpus-level optimizer outputs only after extraction and duplicate/repost handling are evaluated.
8. Add loading, empty, partial-data, stale-snapshot, provider-error and retry states; test persistence across accounts/devices and verify no demo data leaks into production.

## Recommended UI Adjustments

- Keep an unmistakable demo mode until real services are connected; remove demo badges only when the source and personalization are real.
- Replace “Strong/Possible/Weak fit” with a neutral ranked/retrieved label unless those categories have validated definitions. Never show raw ranking scores as match percentages.
- Keep “observed in snapshot” separate from “currently available”; show the snapshot date and an explicit unverified state when no live check exists.
- Render absent, null, blank and unknown distinctly where they affect a decision. Preserve “not evidenced” wording for profile gaps.
- Label match evidence as source-backed; show job and profile excerpts separately. Distinguish source-provided fields from inferred/extracted fields and show uncertainty.
- Treat company details as optional enrichment. Hide unavailable attributes or show “Not provided”; do not fill them with generated descriptions.
- Retain Saved Jobs when a listing is no longer in the current results, with the saved workflow state visibly separate from listing status.

## References

- Frontend contract and screens: [App.tsx](../../frontend/src/App.tsx), [types.ts](../../frontend/src/types.ts), [ProfilePage.tsx](../../frontend/src/pages/ProfilePage.tsx), [DiscoverPage.tsx](../../frontend/src/pages/DiscoverPage.tsx), [SavedJobsPage.tsx](../../frontend/src/pages/SavedJobsPage.tsx), [JobDetails.tsx](../../frontend/src/components/JobDetails.tsx).
- Demo data and local persistence: mockRecommendations.ts, mockProfile.ts, useProfile.ts, useTracker.ts.
- Offline ranking and evidence contracts: [job_matching_engine.py](../../code/job_matching_engine.py), [requirement_aware_reranker.py](../../archive/experiments/code/requirement_aware_reranker.py), [run_requirement_reranker.py](../../archive/experiments/code/run_requirement_reranker.py), [data-preparation.md](../data-preparation.md), [job-matching-engine.md](../job-matching-engine.md).