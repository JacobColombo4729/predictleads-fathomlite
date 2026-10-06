# Discovery Summary

Current evidence is from the September 30, 2026 PredictLeads snapshot. All counts below are records observed, not verified unique jobs, unique companies, or currently open listings. Full-run completion refers to the exact saved run metadata, not inference from output file sizes.

## Completed Full Audit

The saved full audit reports `stop_reason=eof`, `complete_file_scanned=true`, zero uninspected records, and no run failures for both datasets:

- Jobs: 13,415,388 primary `job_opening` records from 13,415,388 wrappers; all 19,031,789,393 compressed bytes read. The audit summary is `data/discovery/full_audit_run_20261005_134741/full_audit/full_quality_summary.json`; the human-readable report is alongside it.
- Companies: 340,669 primary `company` records from 340,669 wrappers; all 252,118,829 compressed bytes read. The same audit summary and report record EOF. The scan also observed 1,739,964 included company resources, which is a separate count from primary companies.

Selected audit observations:

- Jobs: descriptions null in 2,386,488 records and blank in 1; salary null in 7,311,883 and blank in 1; location null in 1,126,469; status null in 4,754,676 and explicitly `closed` in 8,660,712.
- Same-wrapper company references matched by `(type, id)` for all 13,415,388 jobs. This does not measure a separate-file company join.
- Companies: `company_name` null in 2,170; descriptions null in 32,820 and blank in 76; location null in 19,547.
- These are historical snapshot observations. Null status is not proof a job is open, and `first_seen_at`, `last_seen_at`, and `last_processed_at` do not establish posting or closing dates.

## Preparation Development Slices

Two local 10,000-record outputs are preserved under `data/prepared/`:

- Original: `jobs_sample_20261005T205936Z/`. Its saved summary says 10,000 processed/written and `stop_reason=primary_resource_limit`; `complete_file_scanned=false`.
- Corrected validation: `jobs_sample_validation_20261005T205936Z/`. Its summary also says 10,000 processed/written and stopped at the primary-resource limit, not EOF. The row-by-row review found the same source identities/order and raw fields as the original; current helpers intentionally change description formatting and null-date quality flags. See `docs/preparation-review.md`.

Both are sequential development slices, not random or representative samples. Company matches are against `included` resources within each wrapper only.

## Completed Full-File Role Exploration

`data/discovery/role_exploration/roles_corrected_20261006T000000Z/run_summary.json` records `stop_reason=eof`, `complete_file_scanned=true`, 13,415,388 eligible jobs, all 19,031,789,393 compressed bytes read, no errors/skips, and a 30,000/30,000 reservoir with seed `20261005`. The local `role_exploration_report.md` and CSV exports contain detailed distributions; the database is the exact-count store.

Top observed raw-title record frequencies include Customer Service Representative (25,231), Sales Associate (23,584), Maintenance Technician (23,300), Cook (19,528), and Retail Sales Associate (19,087). Top O*NET occupation frequencies include Registered Nurses (346,100), Retail Salespersons (229,808), Software Developers (222,674), General and Operations Managers (207,204), and First-Line Supervisors of Retail Sales Workers (202,662). These are record frequencies. The sample is uniform across successfully parsed eligible job records from this completed stream; low-frequency roles may be absent by chance.

The scan reached EOF and wrote its exports and summary, then exited with code 1 during report rendering because compressed label bytes were decoded incorrectly. Report decoding and CSV filename row counts were fixed locally; the report was rebuilt from the saved SQLite database and summary without restarting the scan. The saved summary and current database metadata both record completion at EOF.

## Product and Data-Preparation Direction

This workspace implements data inspection, a reversible prepared-record view, full audit, and role-frequency exploration. It does not implement a job-seeker application, career-transition recommendations, skill extraction, separate-file company enrichment, embeddings, or ML/LLM matching. Career-transition features and the preparation schema in `docs/data-preparation-proposal.md` are proposals for team review, not approved decisions.
