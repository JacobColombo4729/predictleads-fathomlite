# Full-File Job Role Exploration

`code/explore_roles_and_sample.py` streams the September 30, 2026 jobs JSONL.zst from a local copy when found at the workspace key path, `data/`, or `data/raw/`; otherwise it uses the `predictleads` AWS profile in `us-east-2`. It does not download the complete compressed file or call AI/model APIs.

The completed saved run is `data/discovery/role_exploration/roles_corrected_20261006T000000Z/`. Its summary verifies EOF, 13,415,388 eligible job records, and a 30,000-record seeded reservoir. The stream and exports completed, but the first report-render attempt exited with a compressed-label decoding error. The decoder and report CSV row-count labels were corrected, and the report/complete metadata were rebuilt offline from the saved SQLite database and summary; the source scan was not repeated. See `docs/discovery-summary.md` and `docs/artifact-index.md` for evidence and artifact paths.

## Run

Run the offline checks first:

```powershell
& ".\.venv\Scripts\python.exe" .\code\explore_roles_and_sample.py --self-test
```

The full pass uses seed `20261005`, samples uniformly across eligible primary `job_opening` records, and requires a fresh timestamped output directory. By default the script prints its sample/storage estimate, checks free disk, reruns the synthetic checks before source access, then streams to EOF:

```powershell
& ".\.venv\Scripts\python.exe" .\code\explore_roles_and_sample.py
```

Optional settings are `--sample-size`, `--seed`, and `--output-dir`. An existing output directory is never overwritten. Outputs are under ignored `data/discovery/role_exploration/`.

## Outputs

Each run writes `role_counts_and_sample.sqlite3` (high-cardinality frequency tables, normalized-title/O*NET pairs, run metadata, and reservoir slots), CSV tables for raw titles, normalized titles, occupations, families and title/occupation pairs, `jobs_prepared_random_sample.jsonl`, `run_summary.json`, and `role_exploration_report.md`. SQLite uses SHA-256 keys for count/index rows and stores each exact label once in a value dictionary; labels longer than 128 UTF-8 bytes are losslessly zlib-compressed in SQLite and decoded for CSV export. The full `onet_data` object and full description/salary/location strings are not stored as aggregate labels: O*NET is counted by state/type, availability dimensions by state/type, and occupation/family/title labels are counted separately. Full descriptions are retained only for records currently in the fixed-size sample slots. Expected string dimensions report `unexpected_type` separately from null, blank, missing, and valid strings.

Progress prints every 30 seconds: wrapper/job/sample counts, elapsed time, available memory and disk, and SQLite size. Before source access, the script estimates sample and SQLite/CSV space from the earlier prepared 10k JSONL and reports available disk; during scanning it stops with `disk_safety_stop` if free space reaches the reserved finalization threshold. Population counters are flushed in bounded 5k-resource batches. Sample comparisons are computed from the final retained slots only, after scanning, so replaced reservoir entrants do not bias sample distributions. SQLite commits counts periodically and stores diagnostic checkpoint counters, but **resume from a checkpoint is not implemented**. Handled errors and interruption produce partial outputs and an explicit partial summary when the process remains alive long enough to finalize; forced termination can prevent final export/summary writing.

## Interpretation limits

Counts are record frequencies, not verified unique jobs. The reservoir includes valid primary job dictionaries regardless of description or status, but excludes malformed records and skipped non-job/non-object items. A full EOF pass makes the reservoir uniform over eligible records in that file; rare roles can still be absent by chance. Same-wrapper included companies are used only for records entering the sample; the separate companies file is not joined. No skill extraction, embeddings, LLM calls, deduplication, role-family selection, or hiring-probability inference is performed. The source is a historical snapshot, not proof of current openings.
