# Local Artifact Index

Artifact paths below are workspace-relative and remain under ignored `data/`. These are local machine outputs, not files included in a team contribution. Do not use sample-slot counts alone as completion evidence; check saved `run_summary.json` or `preparation_summary.json` fields such as `stop_reason` and `complete_file_scanned`.

## Completed Outputs

| Purpose | Local path | Verified state |
|---|---|---|
| Full jobs + companies audit | `data/discovery/full_audit_run_20261005_134741/full_audit/` | `full_quality_summary.json`: jobs 13,415,388 primary records; companies 340,669 primary records; both `stop_reason=eof`, `complete_file_scanned=true`; no failures. Report and section metrics are alongside the summary. |
| Original preparation slice | `data/prepared/jobs_sample_20261005T205936Z/` | 10,000 rows written; stopped at `primary_resource_limit`; not EOF. |
| Corrected preparation validation slice | `data/prepared/jobs_sample_validation_20261005T205936Z/` | 10,000 rows written; stopped at `primary_resource_limit`; not EOF. Raw fields/order match the earlier slice; derived text/date flags reflect the corrected helper version. |
| Full-file role exploration | `data/discovery/role_exploration/roles_corrected_20261006T000000Z/` | `run_summary.json`: 13,415,388 eligible jobs, 30,000 seeded sample, `stop_reason=eof`, `complete_file_scanned=true`, no errors/skips. Includes SQLite counts, report, title/family/occupation/pair CSVs, and prepared sample JSONL. Report was recovered offline after a report-rendering error; details are in `docs/discovery-summary.md`. |
| Job Matching Engine baseline | `data/matching/job_matching_engine_baseline/` | Local prototype run on the saved sample: 30,000 loaded, 24,514 usable descriptions, 100 recommendations across five synthetic profiles and two methods, 99 pooled review rows. No human labels or ranking metrics. Outputs include `recommendations.csv`, `relevance_review_template.csv`, and `matching_run_summary.json`; see `docs/job-matching-engine.md`. |

## Partial, Interrupted, and Superseded Outputs

These are retained for provenance and must not be presented as complete analyses:

- `data/discovery/full_audit/attempt_1_partial/`: partial audit logs, checkpoint, and section metrics.
- `data/discovery/full_audit/attempt_20261005_134741_partial/`: partial attempt and launch-failure log.
- `data/discovery/full_audit/benchmark/` and `data/discovery/benchmark_final/full_audit/benchmark/`: benchmark/checkpoint artifacts, not the completed full audit.
- `data/discovery/role_exploration/roles_20261005T223225Z_partial_storage_guard/`: interrupted storage-guard attempt.
- `data/discovery/role_exploration/roles_20261005T224114Z_partial_availability_payloads/`: interrupted attempt with oversized availability payload aggregation.
- `data/discovery/role_exploration/roles_full_20261005T174500Z_partial_oversized_onet/`: interrupted attempt with oversized O*NET aggregation.
- `data/discovery/role_exploration/roles_interrupted_array_state_20261006T000000Z/`: interrupted run archived before corrected empty-array counting; its database/WAL and partial sample are retained but not finalized.
- `data/discovery/quality_summary.json`: earlier bounded profiling/trial summary; it is not the completed full-audit result.

## Preserved Unclassified Folder

`code/prototype_v1/results/active_editor_state.json` is ignored by the existing `code/prototype_v1/` rule. The folder contains only this small editor-state file; its purpose is unclear. It is intentionally preserved and not classified as project code or removed.

## Access and Sharing

No approved private artifact-sharing destination is configured or documented in this workspace. Keep these outputs local until the team identifies an approved access-controlled location, then transfer only the artifacts needed for review under the applicable data-handling rules. Do not commit data, samples, descriptions, contact details, SQLite databases, exports, or logs to Git.
