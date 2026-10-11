# Job Sample Preparation

`code/prepare_jobs.py` creates a small, reversible prepared view of the September 30, 2026 PredictLeads jobs snapshot. It preserves raw source values next to searchable/parsed fields; it does not clean the source dataset, select role families, extract skills, or run ML/LLM services.

## Source and outputs

Run from the project root with `.venv`. The script checks these local locations first: the S3 key path under the workspace, `data/<filename>`, and `data/raw/<filename>`. If none exists, it streams `s3://<bucket-from-env>/<prefix> using the AWS profile `predictleads` in `us-east-2`.

By default, it processes the first 10,000 primary resource objects in wrapper `data` arrays and stops exactly at that count, even if the last wrapper contains more resources. It iterates every resource in each wrapper and uses only that same wrapper's `included` companies for company enrichment. The output folder is unique per run under `data/prepared/` and will not overwrite an existing folder:

```powershell
& ".\.venv\Scripts\python.exe" .\code\prepare_jobs.py
```

Optional offline verification, which runs automatically before any input is opened:

```powershell
& ".\.venv\Scripts\python.exe" .\code\prepare_jobs.py --self-test
```

Outputs are `jobs_prepared_sample.jsonl` and `preparation_summary.json` inside the generated run folder. Existing `.gitignore` rules exclude `data/`; keep these generated records out of Git.

## Mappings and safeguards

- Identity/provenance: source `id`, `type`, key, snapshot date, and one-based source line.
- Titles: `title_raw`, `title_normalized`, and `title_translated` remain separate. `title_searchable` uses normalized title when nonblank, then translated title, then raw title, and records its source.
- Description: raw source text is retained; `description_text` is a deterministic HTML/Markdown-to-text projection that creates line boundaries around headings and list/block elements, removes common Markdown heading/emphasis/link presentation markers, ignores script/style contents, and preserves wording, negation and technical tokens. No missing text is invented.
- Search/filter fields: URL, language, seniority, categories, tags, O*NET object, contract types, raw plus structured location, and raw plus structured salary are carried separately. No skill extraction, salary conversion, or annualization is performed.
- Status: `status_raw` is preserved. `status_derived` is `closed` only for an explicit case-insensitive `closed` value; null, blank, missing, and other values remain `unknown`. All rows are flagged as a historical snapshot and not verified current openings.
- Dates: raw values and parsed UTC values are both stored. Invalid parsing, years before 1900 or after 2027, and selected ordering anomalies are flagged without modifying raw dates.
- Company: the relationship's type and ID are matched against `included` resources using both fields. Missing, incomplete, absent, and ambiguous matches are reported; the separate companies file is not joined.
- Duplicates: repeated non-null `(type, id)` pairs within this 10,000-primary slice are counted, but every occurrence is retained. This is not a full-dataset deduplication check.
- Missingness: `quality_flags` distinguishes missing, null, blank, and present for mapped fields, and records whether cleaned description text is usable.

The first 10,000 primary resources are a development slice for checking the transformation, not a random or representative sample of the file. A resource-limit stop does not imply EOF. The summary records counts, stop reason, EOF status, field flag counts, duplicates, company-match counts, failures, and source metadata when available. On handled stream failures or interruption, the script attempts to retain the partial JSONL and write an explicitly partial summary; forced process termination may prevent that final summary write.

The preparation review in `docs/preparation-review.md` checks the saved sample against the current transformation helpers. If the cleaner changes later, keep the prior run intact and use a new timestamped output folder for a newly prepared slice.
