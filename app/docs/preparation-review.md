# Preparation Review: 10,000-Job Development Slice

**Scope:** Reviewed `code/prepare_jobs.py`, `docs/data-preparation.md`, and the existing 10,000-row output in `data/prepared/jobs_sample_20261005T205936Z/`. No S3 access, audit rerun, larger preparation run, or output rewrite was performed. Example text below is brief and contact-scrubbed. This slice is a transformation check, not a representative market sample.

## Checks and Observations

- The saved JSONL contains exactly 10,000 parseable rows; summary counts are 10,000 processed/written resources from 10,000 wrappers. It stopped at `primary_resource_limit`; `complete_file_scanned` is false.
- 10,000 rows have explicit historical/not-current-verification markers. Status is kept as raw plus derived: `closed` only for explicit closed values, otherwise `unknown`.
- Company matching used both relationship type and ID against each wrapper's `included` list: 10,000 matched. No duplicate non-null `(type, id)` pairs were found in this slice.
- Description availability: 8,264 usable and 1,736 unusable (the audited records were null descriptions). Unknown status occurred 3,561 times.
- Search-title sources: normalized title 9,021, translated-title fallback 0, raw-title fallback 979. The translated fallback case was absent in this slice.
- Formatting observed: 6,041 descriptions begin with Markdown headings, 7,163 contain Markdown list items, 8,070 cleaned descriptions contain line breaks, and 7 contain recognizable HTML block/formatting tags. All 10,000 have `onet_data` as an object in the saved sample.
- Across the slice, literal occurrences of `C++` (47), `C#` (33), `.NET` (65), “not required” (510), and “no experience” (118) were all present in the corresponding cleaned text as well as raw text. These are occurrence counts, not unique-skill counts.

### Selected local examples

| Case | Saved row | Short observation |
|---|---:|---|
| Ordinary normalized-title record | 1 | `Research Associate, Discovery Biology`; usable description; explicit `closed`. |
| Missing description and unknown status | 4 | `Dedicated CDL A Truck Driver`; raw status null, derived status unknown; description unusable. |
| Raw-title fallback | 6 | `Program Management Intern (6-Week Summer Internship)` remained the searchable title because no nonblank normalized/translated choice was available. |
| HTML formatting | 371 | Raw contains `<u><strong>Job Requirements</strong></u>`; current cleaner yields readable `Job Requirements` text and keeps following list boundaries. |
| Date flag on null source date | 1 | `posted_at` is null; saved output incorrectly labels it `posted_at_not_parseable`. |

## Defects and Fixes

1. **Markdown formatting residue:** The original cleaner removed HTML tags but left Markdown heading/emphasis/link markers such as `###` and `**`. Updated `clean_description` to remove common presentation markers, retain heading/list line boundaries, and keep link labels and wording. In a read-only replay of current helpers against retained raw fields, `description_text` differs from the saved output on 8,080/10,000 rows. Example: `### Research Associate, Discovery Biology` becomes `Research Associate, Discovery Biology`.
2. **Null dates reported as parse failures:** The saved output flags 5,582 null `posted_at` values as `posted_at_not_parseable`; all 5,582 are null, and there were zero invalid nonblank date strings. Updated parsing so null/blank/missing are represented by their field-state flags, invalid nonblank strings are parse failures, and unexpected non-string dates get an `invalid_type` flag. Current replay yields 375 `posted_at_after_first_seen_at` and 956 `last_seen_at_after_last_processed_at` flags; raw values are unchanged.
3. **O*NET raw-value preservation:** The script previously replaced unexpected non-object `onet_data` with null. It now passes the source value through. This did not alter this saved slice because all 10,000 `onet_data` values are objects.

**Version check:** The script edits after this sample was generated do affect derived output: Markdown cleanup changes 8,080 `description_text` values, and null-date flag corrections change `quality_flags` on 5,582 rows. Replaying the current `prepare_resource` helper using retained raw fields reproduced the other compared derived values for all 10,000 rows. The saved slice was not regenerated and must not be treated as output from the final helper version.

Focused offline tests pass for missing/null/blank fields, title fallback precedence, HTML and Markdown text, negation and technical tokens, invalid/null/suspicious dates, duplicate identities, same-wrapper company matches, ambiguous/missing matches, O*NET raw passthrough, and exact resource-limit handling. The script compiles cleanly and has no reported Pylance errors. No S3 request was made for this review.

## Limitations and Readiness

The saved artifacts do not include the complete original wrapper, so replay verifies current transformations from retained raw fields, not byte-for-byte fidelity to the original S3 records. The 10,000 rows are the first sequential primary resources, not random or representative. Translated-title fallback was not exercised; ambiguous/missing company matches, unexpected O*NET types, and invalid nonblank dates also did not appear in this slice. HTML testing covers a few observed markup cases, not every malformed/nested pattern. Salary and location source values remain pass-through fields, not validated normalization.

**Readiness at initial review:** A fresh, separately named 10,000-row validation slice was still needed before planning a broader working dataset.

## Fresh-Slice Validation (2026-10-05)

Ran the corrected `code/prepare_jobs.py` once with `--limit 10000` into `data/prepared/jobs_sample_validation_20261005T205936Z/`. The prior `jobs_sample_20261005T205936Z/` files remain unchanged. The new summary records 10,000 wrappers/resources processed and written, stopped at `primary_resource_limit`, `complete_file_scanned: false`, 10,000 included resources inspected, no malformed/skipped records, no failures, and no duplicate non-null `(type, id)` pairs. All 10,000 company references matched included companies in their own wrappers by both type and ID.

Compared the two JSONL files row-by-row. All 10,000 `(job_type, job_id, source_line)` tuples match in order. Every persisted raw field compared (titles, description, URL, language, seniority, categories, tags, O*NET, contract types, location, salary, status, raw dates, company reference, snapshot and source provenance) has zero differences. No rows were added, lost, or reordered.

The current helpers reproduce the intended derived changes: `description_text` differs on 8,080/10,000 rows as Markdown headings/emphasis separators are normalized; `quality_flags.date_anomalies` differs on 5,582 rows because null `posted_at` values are no longer mislabeled as parse failures. The other compared derived values (searchable title and source, usability, derived status, parsed UTC dates, company match state/count/name/domain, and O*NET) have zero differences. The corrected slice has zero invalid nonblank date strings, 375 `posted_at_after_first_seen_at` flags, and 956 `last_seen_at_after_last_processed_at` flags. Null `posted_at` remains null and its field-state flag remains `null`.

Title source counts are normalized 9,021, translated 0, and raw-title fallback 979; translated fallback remains unexercised. Status remains 6,439 explicit `closed` and 3,561 `unknown`. Short examples from the existing local rows (not representative): row 1 heading `### Research Associate, Discovery Biology` becomes plain heading text; row 371’s `<strong>Job Requirements</strong>` becomes `Job Requirements` with the following list readable. A phrase detected as “not required” after cleaning was split across markup in raw source text; no wording was invented. Literal `C++`, `C#`, `.NET`, “not required,” and “no experience” occurrences remain in cleaned text.

Focused synthetic tests, the row-by-row comparison, JSONL validation, compilation, and Pylance diagnostics passed. No full audit, larger preparation run, or S3 call was made as part of the comparison itself. Existing prepared outputs remain unchanged.

**Updated readiness (2026-10-05):** The preparation helpers passed the documented 10,000-record validation, but the saved outputs remain two sequential development slices, not a broader working dataset or representative evidence. The separate full audit and role exploration do not change that preparation scope. Before scaling preparation, the team must decide sample strategy, output size/retention, resource/runtime safeguards, and how to assess translated-title fallback, rare company-match states, and HTML/Markdown edge cases. No skill extraction or separate companies-file enrichment is implemented.
