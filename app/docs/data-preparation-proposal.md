# PredictLeads Data Preparation Proposal

**Purpose:** Agree on a conservative preparation contract for the three-week job-seeker platform. This is a proposal only: it does not clean, rewrite, or remove source data, and does not define a model or application implementation.

**Workspace context (2026-10-06):** This remains a proposal for team review. The completed full audit and role exploration are summarized in `docs/discovery-summary.md`; local outputs are indexed in `docs/artifact-index.md`. The bounded 100,000-resource profiler observations below are dated historical evidence from an earlier prefix run, not the current full-file totals. Keep raw and generated datasets, credentials, and candidate resumes out of Git.

## 1. What We Know

### Observed facts

The saved profiler summary reports 100,000 primary job resources from 100,000 wrappers and 100,000 primary company resources from 100,000 wrappers. The jobs scan read 142,085,300 compressed bytes from a 19,031,789,393-byte object in 47.2 seconds. The companies scan read 74,057,375 bytes from a 252,118,829-byte object in 31.8 seconds and counted 505,975 included resources. Both stopped at the 100,000-primary-resource limit and both record `complete_file_scanned: false`. The jobs scan counted 100,000 included resources. Thus these are first-prefix scans, not full-file analyses. The job reservoir is random only within those scanned records, not random across the file.

Among 100,000 scanned jobs, null counts were: salary 54,270 (54.27%), `posted_at` 56,131 (56.131%), status 35,325 (35.325%), description 17,807 (17.807%), and location 8,386 (8.386%). No absent keys, blank strings, or unexpected types were reported for those audited job fields. In companies, 9,606 descriptions were null and 23 blank; location was null in 5,692 records and `company_name` in 642. No duplicate primary `(type, id)` pairs were reported. The company scan flagged one over-limit line; jobs had no malformed records or skipped resources.

Every scanned job had `relationships.company.data` as an object, and 100,000/100,000 references matched a company included in that same wrapper using type and ID. That is not a cross-file join test. Job `posted_at` parsed for 43,869 values; among ordering checks, 3,667 had `posted_at` later than `first_seen_at` and 9,296 had `last_seen_at` later than `last_processed_at`. The profiler observed 119 salary signatures among 45,730 non-null salary values. Status values were `closed` 64,675 and null 35,325. Section comparisons show variation within the scanned prefix; largest adjacent null-rate changes were 1.53 percentage points for jobs and 0.66 for companies.

### Interpretations (not established facts)

Titles, description text, location, categories, seniority, language, company context, and parsed salary are plausible search and ranking inputs. Null-heavy salary and status fields argue against making them mandatory eligibility filters. Relationship completeness in job wrappers is promising, but does not demonstrate that every reference resolves to the separate companies file.

### Proposed rules

Retain every source record and raw value. Add normalized/parsed fields alongside raw fields, with explicit quality flags and provenance. Treat missing, null, blank, invalid, and unknown as distinct states. Do not infer an open listing from null status, infer original posting time from `first_seen_at`, infer a closing time from `last_seen_at`, annualize pay automatically, fill missing content, or delete a record because an attribute is absent.

### Unresolved questions

Confirm how the daily snapshot is ordered, whether and how job IDs persist across snapshots, whether the separate companies file uses the same IDs as job `CompanyLite` references, what source evidence produces null status, how provided USD salary bounds are calculated, and how long historical/closed records should remain searchable.

## 2. Field Guide

Official definitions below are from the [Job Openings Dataset guide](https://docs.predictleads.com/guide/job_openings_dataset/data_model) and [Companies Dataset guide](https://docs.predictleads.com/guide/companies_dataset/data_model). Where schema documentation is silent, semantics remain unresolved.

| Field | Documented meaning and preparation use |
|---|---|
| Job `id`, `type` | UUID identifier and constant `job_opening`. Preserve both; use `(type, id)` as the candidate resource identity. |
| `title` | Job title string. Preserve verbatim; optionally use documented `translated_title` and `normalized_title` as separate search keys where present. |
| `description` | Source job description, nullable. Preserve source text and HTML; keep a separately derived searchable representation with provenance. |
| `relationships.company` | A reference to a `CompanyLite` in the same wrapper’s `included` array. The reference carries UUID `id` and type `company`; included company-lite attributes include domain, company name, and ticker. Match included resources by both type and ID. |
| `location`, `location_data` | `location` is built from structured `location_data` when present, otherwise falls back to the original unnormalized detected location. `location_data` is an array documented as currently containing at most one object; fields include city, state, ZIP, country, region, continent, and `fuzzy_match`. Prefer structured parts for filters and retain the display string. |
| `salary`, `salary_data` | `salary` is raw salary text. `salary_data` is parsed data with low/high bounds, currency, USD low/high bounds, and time unit (`hour`, `day`, `week`, `month`, `year`, or null). The guide says a non-range salary has equal low/high values. Preserve raw values and distinguish source currency from provided USD fields; confirm conversion basis before using USD fields. |
| `seniority` | Nullable seniority inferred from the job title; documented values include `junior`, `mid_senior`, `manager`, `director`, `head`, `vice_president`, `c_level`, `president`, `partner`, `founder`, and `not_set`. Treat as a derived signal, not ground truth. |
| `categories`, `tags` | Categories are zero or more organizational job categories. Tags are keywords found under the description. Preserve original arrays; do not treat tags as verified skills. |
| `language` | Language in which the job description is written, nullable. Suitable as a filter only when present; unknown must remain unknown. |
| `status` | The guide documents only `closed` or null; `closed` is assigned when closure-related phrases (such as “position filled”) are found on the job site. Null therefore means no documented `closed` value, not proof of open status. |
| `first_seen_at`, `last_seen_at` | ISO 8601 times the job was first and last seen. Neither means original posting date or closing date. |
| `posted_at` | Nullable ISO 8601 date-time when the job was posted. Keep distinct from observation dates. |
| `last_processed_at` | ISO 8601 time the job was last sent to processing. Do not substitute it for a source-posting or closure date. |
| Companies dataset | Company `id` is a UUID and type is `company`; attributes include domain, nullable company name, ticker, friendly name, translated/summarized descriptions, language, and location. Full company records provide richer company context than the job wrapper’s `CompanyLite`. |

For a **same-wrapper** match, use `job.relationships.company.data.(type, id)` against `wrapper.included` resources’ `(type, id)`; this achieved 100,000/100,000 matches in the bounded job prefix. For a **separate-file** join, use `(type, id)` as the proposed key, retaining unmatched rows and a join-quality flag. UUID/type definitions align in the official schemas, but cross-file coverage and snapshot consistency still need a direct test; do not assume every job has a matching row in the companies file.

## 3. Proposed Cleaning Decisions

The observations in this proposal section and its decision table refer to the profiler’s first 100,000 primary resources per file (historical prefix evidence, not full-file estimates). The separate completed audit is the source for current full-snapshot counts. “Decision needed” means the team should approve before production preparation.

| Issue and observed evidence | Recommended handling (proposal) | Platform benefit | Tradeoff / uncertainty | Team decision? |
|---|---|---|---|---|
| Missing descriptions: 17,807/100,000 jobs null; company description null 9,606/100,000 and blank 23/100,000. | Keep job searchable by other fields; retain raw null/blank state. Do not synthesize a replacement description. | Avoids fabricated evidence while preserving usable listings. | Text relevance weaker when absent. | Yes: result-card fallback wording. |
| Missing salary: 54,270/100,000 jobs null; non-null values had 119 observed format signatures. | Keep raw `salary`; parse only when unambiguous into amount bounds, currency, and pay period; keep parse confidence/reason. No automatic annualization. | Enables explicit salary filtering and transparent matching. | Free text, ambiguous symbols, range conventions, and conversion basis. | Yes: whether to expose provided USD fields and salary filters. |
| Missing location: 8,386/100,000 jobs null. | Keep raw string and structured parts separately; normalize known components without inventing a place. Make unknown-location behavior explicit in filters. | Supports exact/region filters without false geographic matches. | Remote/hybrid and multi-location semantics need checking. | Yes: whether unknown location is included by default. |
| Unknown status: 35,325/100,000 null; all observed non-null values were `closed`. | Preserve source status; represent null as `unknown`, not open. Only mark closed on explicit source value. Use observation dates as separate freshness evidence. | Prevents confidently presenting unknown listings as active. | “Last seen” freshness thresholds and source status can be stale. | Yes: user-facing labels and stale-listing policy. |
| Null vs blank vs absent | Store `field_state` or quality flags; normalize blank text for search only while preserving raw value. | Keeps audits honest and distinguishes source omissions from empty text. | Adds schema/logic complexity. | Yes: which states UI should surface. |
| Duplicates: zero duplicate primary type/ID pairs in each scanned 100,000. | Deduplicate only exact identity/version rules after broader validation; preserve snapshot history and duplicate audit counts. | Avoids duplicate results without losing history. | Identity persistence and version semantics are not confirmed. | Yes: define listing version and retention key. |
| Malformed/oversized: one company line exceeded 262,144-byte cap; no job malformed rows. | Quarantine or flag by line number/reason; continue scanning subsequent lines where safe. Keep raw bytes only in access-controlled temporary storage if approved. Never silently truncate a parsed record. | Makes pipeline failures diagnosable without corrupting content. | A hard line limit can omit valid unusually large source records. | Yes: increase limit or separately review oversized records. |
| Suspicious dates: 3,667 `posted_at > first_seen_at`; 9,296 `last_seen_at > last_processed_at`. | Preserve strings, parse separately, flag ordering anomalies; do not reorder or “correct” values automatically. | Enables freshness and chronology QA without rewriting evidence. | May reflect timezone, source correction, or undocumented process semantics. | Yes: thresholds and treatment in UI/ranking. |
| HTML in descriptions | Keep original markup and derive a safe text field with a deterministic HTML parser; preserve headings, lists, and links as useful structure. Escape output at rendering time. | Readable search and safe display. | Tables/entities and malformed markup can lose structure. | Yes: fields/format to retain for search versus display. |
| Skill-name variants | Preserve exact source spans; propose alias-to-canonical skill mapping in a separate versioned table with confidence and evidence location. | Improves matching across spelling variants. | C++, C#, .NET and negation/qualification qualifiers are easy to damage. | Yes: approve initial taxonomy and review workflow. |
| Salary currencies/units | Keep raw amount text and source currency/period. Use `salary_data` as supplied with provenance; never convert units automatically. | Supports consistent filters when values are trustworthy. | Currency symbols can be ambiguous; supplied USD conversion methodology is not stated here. | Yes: conversion display and pay-period comparison policy. |
| Company joins | Preserve relationship `(type,id)`, same-wrapper match flag, separate-file match flag, and unmatched IDs. Proposed cross-file key is `(type,id)`; test it before relying on it. | Company name/domain enrich results and employer filters. | Bounded profiler established included matches only; separate-company join coverage was not measured. | Yes: fallback when unmatched and domain use. |

Text preparation must preserve meaningful distinctions such as “not required,” “no experience,” required versus preferred qualifications, and literal technical names such as C++, C#, and .NET. Avoid destructive stop-word removal, blanket punctuation stripping, or rewriting negation.

## 4. Current Evidence and Remaining Gaps

### Historical Prefix Review (2026-10-05)

The 100,000-resource profiler and its first-prefix comparisons remain useful as dated development history, but are superseded for whole-snapshot counts by the completed full audit in `data/discovery/full_audit_run_20261005_134741/full_audit/`. The full audit reached EOF for both files. The completed jobs role exploration also reached EOF and produced a seeded uniform 30,000-record reservoir. Neither analysis ran a separate companies-file join, exact duplicate detection across versions, skill extraction, or hiring/model evaluation; those remain unmeasured or unimplemented.

Random sampling across the full compressed stream requires processing preceding compressed bytes to reach later records. The saved full jobs audit read the entire 19,031,789,393-byte object in about 2,016.6 seconds; the companies audit read 252,118,829 bytes in about 62.1 seconds. The role exploration made a separate full jobs pass and took about 9,629.6 seconds while calculating exact high-cardinality counts and retaining its 30,000-record reservoir. These are observations for the saved runs, not runtime guarantees. No additional scan is required to review the saved results.

## 5. Proposed Prepared Job Record

Keep an immutable source layer plus a versioned prepared layer. Minimum per-job record:

| Group | Suggested fields |
|---|---|
| Identity / provenance | `source_dataset`, `snapshot_key`, `wrapper_line_number`, `resource_type`, `job_id`, `source_url`, `prepared_at`, `schema_version` |
| Text | `title_raw`, `title_normalized`, `title_translated`, `description_raw`, `description_text`, `description_html_present`, `description_state`; extracted skill mentions with raw text, canonical candidate, source span, and confidence |
| Company | `company_ref_type`, `company_ref_id`, `included_company_match`, `company_file_match`, `company_domain`, `company_name`, `company_join_state` |
| Location | `location_raw`, `location_data_raw`, normalized city/state/region/country/ZIP, coordinates if present, `fuzzy_match`, `location_state` |
| Compensation | `salary_raw`, supplied salary low/high, source currency, supplied USD low/high, time unit, parse state/reason; no derived annual salary unless later approved |
| Search / rank features | seniority raw, categories raw, tags raw, contract types raw, language raw, separately normalized category/skill tokens |
| Status / time | `status_raw`, status state, `posted_at_raw`/parsed, `first_seen_at_raw`/parsed, `last_seen_at_raw`/parsed, `last_processed_at_raw`/parsed |
| QA / history | field-state flags, type/parse flags, suspicious date flags, duplicate/version flag, scan/snapshot provenance; retain historical snapshots instead of deleting old jobs |

Prepared fields are derived views, not replacements for source values. GenAI match explanations should cite retained source excerpts/fields and avoid claiming a skill, salary, location, or active status not evidenced by the record.

## 6. Team Decisions and Next Steps

| Decision | Options | Recommendation |
|---|---|---|
| Unknown status | Hide; label unknown; show with freshness warning | Keep distinct as unknown and show with an “availability unconfirmed” cue until freshness rules are agreed. |
| Missing location/salary | Exclude from filters; treat as wildcard; include with “not provided” | Include by default, but do not count unknown as satisfying a specific filter. |
| Salary normalization | Raw only; parse source bounds; convert all to annual USD | Retain raw plus parsed source currency/period; expose supplied USD fields separately; defer conversion rules. |
| Text and skills | Destructive normalize; raw plus searchable derivative; manual annotation only | Raw plus deterministic HTML/text derivative and reviewed alias layer that preserves negation and qualification context. |
| Company joins/history | Join on ID alone; join on `(type,id)`; use only included company | Propose `(type,id)` and keep both included and separate-file match flags; preserve snapshots and unmatched records pending coverage validation. |

**Next actions, in priority order:** (1) Have the team review the proposed product direction and five data policies, and obtain authoritative answers on status, date, salary-conversion, and snapshot semantics. (2) If separate-file enrichment is needed, perform a scoped `(type, id)` join-coverage and snapshot-consistency validation; the completed full audit did not perform that join. (3) Agree the immutable raw/prepared contract, then implement only approved deterministic preparation steps with before/after counts, provenance, and tests; keep raw data out of Git. Full-file audit and role-frequency exploration are already complete and documented; they do not constitute approval of these decisions.

**Candidate-resume PII is separate from job/company preparation.** Begin matching experiments with synthetic candidate profiles. If real resumes are considered later, agree on consent, access, retention/deletion, minimization, and approved model/provider handling first; do not send identifiable resumes to an external service by default. No PII detector is proposed or implemented here.

## Sources

- PredictLeads, [Job Openings Dataset – Data Model](https://docs.predictleads.com/guide/job_openings_dataset/data_model)
- PredictLeads, [Companies Dataset – Data Model](https://docs.predictleads.com/guide/companies_dataset/data_model)
- Local evidence: `data/discovery/full_audit_run_20261005_134741/full_audit/full_quality_report.md`, `full_quality_summary.json`, `data/discovery/role_exploration/roles_corrected_20261006T000000Z/run_summary.json`, `code/inspect_data.py`, and `code/profile_data_quality.py` (saved outputs preferred; no scan rerun for this proposal).
