# Job Matching Engine: First Baseline

> **Note (2026-10-09):** this describes the first keyword/TF-IDF baseline engine (`code/job_matching_engine.py`) and the earlier experiments built on it, deleted on 2026-10-12 (they're in the git history). The demo ranker is TF-IDF + BGE reciprocal-rank fusion; see [backend-api.md](backend-api.md). The commands below that use those scripts no longer run.

This is a first retrieval prototype for team review. It reuses the completed, seeded 30,000-job JSONL sample; it does not access S3, rerun an audit, regenerate the sample, call an AI service, or create relevance labels. All generated recommendations, run summaries, and review files stay under ignored `data/matching/`.

## Run

Install the project requirements in `.venv`, then run the offline checks and prototype from the repository root in PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& ".\.venv\Scripts\python.exe" .\code\job_matching_engine.py --self-test
& ".\.venv\Scripts\python.exe" .\code\job_matching_engine.py `
  --sample .\data\discovery\role_exploration\roles_corrected_20261006T000000Z\jobs_prepared_random_sample.jsonl `
  --profiles .\code\synthetic_candidate_profiles.json `
  --output-dir .\data\matching\job_matching_baseline
```

The sample path and output directory are configurable. If the prepared JSONL is missing, the command prints its path and instructions and exits without attempting a remote read. The sample and the aggregate CSVs are separately shared local artifacts; obtain them through the team's approved private location. The same sample path is already the default.

For a later, human-reviewed metrics run, complete the generated `relevance_review_template.csv` and pass it explicitly:

```powershell
& ".\.venv\Scripts\python.exe" .\code\job_matching_engine.py `
  --sample .\data\discovery\role_exploration\roles_corrected_20261006T000000Z\jobs_prepared_random_sample.jsonl `
  --output-dir .\data\matching\job_matching_reviewed `
  --labels .\data\matching\job_matching_baseline\relevance_review_template.csv
```

Metrics are omitted when the file has no genuine labels. For each profile/method, Precision@5 and nDCG@5 are emitted only if all five top positions are judged. Otherwise the metrics remain null and unjudged ranks are listed; an absent label is never treated as irrelevant.

## Data Contract and Coverage

`load_prepared_jobs` reads already-prepared JSONL objects and retains each complete source/prepared mapping on its `JobRecord`. The stable identifier is the prepared `job_id`. Duplicate IDs after the first occurrence are excluded and counted rather than silently merged. Missing IDs, malformed JSON, non-object lines, blank lines, and records with no searchable tokens have explicit exclusion reasons. Rows with missing descriptions remain eligible when title or occupation text is usable.

The load report includes lines seen, records loaded, usable descriptions, total excluded records, and counts by exclusion reason. `description_usable` requires both the preparation flag and nonblank `description_text`. Source title fields, source/derived status, same-wrapper company context, URL, and quality flags are preserved in recommendation evidence. Separate companies-file enrichment is not performed.

The prepared sample is a uniform 30,000-record reservoir from eligible resources of a historical snapshot, not the complete job population or a current-opening feed. It may omit rare roles. A null/unknown status is not proof of openness. The baseline leaves unknown status included unless a profile explicitly requests exclusion of `status_derived == "closed"`; unknown values remain unknown. A location filter is applied only when `require_location_match` is explicitly true and a profile location is supplied. A job with missing location is then excluded as `unknown_location`; no implicit remote/location inference is made. Seniority is not a hard eligibility filter because it is a source inference.

## Candidate Schema

`CandidateProfile` is reusable and contains:

- `profile_id`: synthetic or evaluation key, with no personal identifier.
- `experience`: tuple of evidence statements.
- `evidenced_skills`: tuple of skills supported by the profile.
- `desired_roles`: tuple of role/title queries.
- `location`: optional location string.
- `preferences`: extensible mapping. The initial conservative controls are `exclude_explicitly_closed`, `require_location_match`, plus soft textual preference keys `keywords`, `work_arrangement`, and `industry_preferences`.

Five clearly synthetic examples are in `code/synthetic_candidate_profiles.json`. Pass another local JSON file with `--profiles` to extend or replace the profile set. They are ranking smoke tests, not labeled users or fabricated relevance assessments. The schema is extensible to around 20 review profiles. Do not add real resume content or personal data without a separately approved handling process.

## Ranking Baselines

The output contains separate rankings for `keyword` and `tfidf_cosine`; each method returns up to ten results per profile after the explicit filters.

- Keyword retrieval uses literal token overlap from profile experience, evidenced skills, desired roles, location, and selected preferences. Job title tokens weigh 3, O*NET occupation tokens 2, and readable description tokens 1. Its weighted coverage score is a retrieval heuristic.
- TF-IDF fits one `TfidfVectorizer` on all loaded job documents, then transforms each candidate profile with that fitted vocabulary. Job text weights title 3x, O*NET occupation 2x, description 1x. Candidate text weights desired roles 3x, experience/skills 2x, preferences/location 1x. The vectorizer uses unigrams/bigrams, sublinear term frequency, a 250,000-feature cap, and a tokenizer that preserves `C++`, `C#`, and `.NET`.
- Missing descriptions are not imputed. The job remains in the corpus if other text is usable; its corpus text then consists of title and occupation fields only. Recommendation output marks `description_available` and uses the title as evidence fallback.

Scores are weighted coverage or cosine similarity, **not probabilities of hiring, application success, or candidate eligibility**. The evidence excerpt is a bounded source/prepared description excerpt for human inspection, not a generated explanation.

## Human Review and Evaluation

The generated pooled template has one row per `(profile_id, job_id)`, so reviewers label a job once even if multiple methods retrieved it. `retrieval_methods` contains a JSON mapping from each method to its rank. The rubric is:

- `0`: unsuitable for the profile and stated preferences.
- `1`: plausible; enough evidence to investigate further, with material gaps or uncertainty.
- `2`: strong fit based on the available evidence; not a hiring prediction.

Reviewers should judge against the synthetic or approved profile evidence and job evidence, note uncertainty, and leave `relevance_score` blank when unable to judge. Do not infer that an unmentioned skill is absent or unnecessary. Pool outputs from each retrieval method before review to reduce duplicate effort. Freeze held-out profiles before tuning; use the remaining profiles for iteration and reserve distinct profiles for final comparison. The initial five synthetic profiles are not sufficient evidence of general performance.

If all five top-ranked results for a profile/method have labels, Precision@5 is relevant judged results (labels 1 or 2) divided by 5. nDCG@5 uses gains `2^label - 1`. When any top-five result is unjudged, the affected metric is withheld and missing judgments are reported. A limited set of independent model-based retail labels is available for the saved controlled experiment; broad human evaluation remains pending.

A blinded, one-job-at-a-time review tool for the retail/inventory profile was deleted with the experiments on 2026-10-12; it's in the git history.

### Shared-Pool nDCG and Incomplete Judgments

In controlled comparisons, the per-profile reference pool is the union of stable job IDs returned by **all methods in all conditions**. Every method/condition nDCG uses the same ideal ordering: the five highest relevance labels in that fully judged shared pool, sorted descending. This makes the ideal DCG comparable across rankings instead of allowing a weak method to get nDCG 1 merely because its own retrieved jobs are ordered internally.

nDCG is withheld until every job in the profile's shared pool has a genuine 0/1/2 label; `shared_pool_metrics.json` lists missing job IDs. Precision@5 remains method/condition-specific and is reported only when five jobs were returned and all five have genuine labels. If fewer than five results are returned, Precision@5 is withheld; nDCG uses zero gain for the missing result slots. If the shared pool itself has fewer than five jobs, its judged jobs define the ideal and the remaining ideal ranks contribute zero gain. If a fully judged pool is all-zero, nDCG is 0. These are pool-relative metrics: they cannot account for relevant jobs that neither condition retrieved, and a shared pool formed from system outputs is not a complete relevance universe. The current retail experiment has two unjudged jobs in its 20-job pool, so nDCG remains withheld. Human reviewers should inspect pool coverage and optionally add independently discovered jobs before treating comparisons as representative.

## Observed Local Prototype Run

On October 7, 2026, the baseline loaded the existing sample at `data/discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl`: 30,000 lines, 30,000 stable-ID records loaded, 24,514 usable descriptions, and zero exclusions. The corrected TF-IDF fit/ranking duration recorded in `matching_run_summary.json` was 67.156 seconds; this excludes initial JSONL loading and output writes. It produced 100 recommendations (ten for each of five profiles under each of two methods) and 99 pooled profile/job review rows. No relevance labels were present, so no metrics were reported.

Example top results from this run, shown only to make the output concrete: the synthetic customer-facing profile with technical training retrieved “Customer Support Specialist” first by keyword score (0.3575) and “Help Desk Analyst Remote Job| 3 days ago” first by TF-IDF cosine (0.3265). The synthetic project-coordination profile retrieved “Commercial Project Coordinator” by keyword (0.4136) and “Project Coordinator” by TF-IDF (0.2893). The synthetic retail/inventory profile's TF-IDF top result was “Senior Investments Operations Associate | Clive, IA” (0.1569), a candidate for human relevance review, not evidence that the model is effective. Scores are not probabilities.

## Outputs and Limitations

The CLI writes `recommendations.csv`, `relevance_review_template.csv`, and `matching_run_summary.json`; it writes `review_metrics.json` only when labels are supplied and complete top-five judgments exist. These generated files and vectorizer objects are local-only and ignored. The baseline reports loader counts and fit/ranking runtime; no model artifact is serialized. On the current Windows `.venv` and saved sample, the corrected fit/ranking pass took about 69 seconds; total wall time also includes JSONL loading and output writing. Runtime depends on hardware and available memory because the prepared records and TF-IDF matrix are held in memory.

## Substantive-Description Experiment

The controlled experiment compares (A) the existing description-available pool (`description_usable` is true and `description_text` is nonblank) with (B) records accepted by the opt-in `substantive_description_v1` evidence rule. The rule accepts a non-boilerplate line with a concrete duty action and object, or a non-heading line with an explicit qualification signal. Minimum signal-line length is three words; overall description length and standard section headings are not eligibility requirements. Recognized metadata and generic benefit/EEO/company boilerplate are excluded from evidence. This is a transparent, unvalidated heuristic, not a measure of description quality; accepted and rejected diagnostics are saved, and rejected examples explicitly warn of possible false exclusions.

Both keyword and TF-IDF/cosine run in each condition. TF-IDF is fitted once on all loader-eligible jobs; both conditions select row indices from that same matrix, so vocabulary and IDF are held fixed. The same five profiles, profile filters, ranking weights, tokenizer, and top-k are used. The only intended difference is retrieval-pool eligibility. Every loader-eligible source record remains in the corpus; the rule does not alter source data or candidate profiles.

Run once after tests, using a fresh output path:

```powershell
$runId = Get-Date -Format "yyyyMMddTHHmmssZ"
& ".\.venv\Scripts\python.exe" .\code\job_matching_engine.py `
  --sample .\data\discovery\role_exploration\roles_corrected_20261006T000000Z\jobs_prepared_random_sample.jsonl `
  --profiles .\code\synthetic_candidate_profiles.json `
  --description-coverage-experiment `
  --labels .\data\matching\description_coverage_20261007T141953Z\retail_evaluation_20261007T200242Z\retail_independent_ratings.csv `
  --experiment-output-dir ".\data\matching\description_substantive_$runId"
```

Outputs are `description_coverage_recommendations.csv` (both conditions/methods and evidence-rule class/reason), `pooled_review_template.csv` (one row per profile/job with all condition/method ranks), `blinded_review_view.csv` (only previously unlabeled retail-profile jobs newly entering the substantive-condition top-10 union; no method/rank/score), `shared_pool_metrics.json`, `description_evidence_diagnostics.csv` (deterministic accepted/rejected examples and class/reason counts in the summary), `top5_changes.csv` (IDs entering/leaving each top five and common-job rank shifts), `review_instructions.md`, and `description_coverage_run_summary.json`. Existing independent retail labels are reused only for identical profile/job pairs. Newly retrieved items and all other profile/job pairs remain unlabeled; conversational judgments are not loaded. nDCG uses the union of retrieved jobs from both methods and both conditions as the shared per-profile ideal and is withheld until that pool is fully judged. Precision@5 requires five returned, labeled items. Generated files remain under ignored `data/`.

The saved run is `data/matching/description_substantive_20261007T201326Z/`. It loaded 30,000 source records; condition A contained 24,514 jobs and condition B 23,437. The rule classified 5,486 descriptions as missing/blank and a further 1,077 available descriptions as insufficient; 23,437 were accepted. These are rule counts, not human validation results. Top-five changes appear only in TF-IDF, for four profiles: healthcare operations (two entries), project coordination (two entries), retail inventory (one entry), and software development (two entries); keyword top fives are unchanged. The rule produced one newly retrieved, previously unlabeled retail item for blinded review. No new labels were created.

The run reused one TF-IDF fit and the same five profiles, filters, ranking weights, tokenizer, and top-k. Existing independent retail labels were reused only for identical profile/job pairs: 18 numeric labels matched the 20-job shared pool; two shared-pool jobs remain unjudged, so nDCG is withheld. Retail keyword Precision@5 was 1.0 for both conditions. Retail TF-IDF Precision@5 was withheld for description-available (4/5 judged) and was 0.6 for substantive-description (5/5 judged). This small, single-profile result is descriptive, not proof of improved recommendation usefulness. The run summary also reconciles the earlier customer-support and retail/inventory baseline top fives: 20 rows, 6 missing/unusable descriptions, and 2 usable descriptions no longer than 250 characters. That length threshold is descriptive only and did not affect condition B. Timings are hardware-specific; this local run took about 450 seconds and did not rescan or access S3.

Known limits: keyword coverage is lexical, TF-IDF does not understand semantic equivalence, title/occupation repetition is a simple field-weighting approximation, company context is only the same-wrapper enrichment already present, and location string matching is literal token containment. The evidence heuristic can miss duties or qualifications outside its vocabulary and can accept generic action phrasing; inspect the diagnostic examples before relying on it. The existing independent ratings are model-based review, not broad inter-rater validation, and do not fully cover the shared pool. A skill absent from a description does not prove it is unnecessary; source seniority does not establish eligibility. No target audience has been selected, and no general ranking-quality claim is made.
