# Retail / Inventory Recommendation Review

This local Streamlit app presents only the saved `synthetic_retail_inventory` profile's blinded review rows. It does not run matching, access S3, alter recommendation outputs, or assign relevance labels. Job text is rendered as plain text.

## Open the Review Screen

From the repository root in PowerShell, with project requirements installed:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run .\code\retail_review_app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`). To stop the app, press `Ctrl+C` in that terminal. Re-run the same command later to resume. The app opens the saved input at `data/matching/description_coverage_20261007T141953Z/blinded_review_view.csv` and writes decisions separately to `data/matching/description_coverage_20261007T141953Z/retail_review_judgments.csv`; the last visited job ID is saved in `retail_review_progress.json`. All are ignored local files. The original blinded CSV is read-only.

If the experiment directory is not present locally, obtain its approved private copy before launching. The app reports a missing input rather than looking online or rebuilding it.

## Review One Job

For each displayed job, compare the synthetic candidate's experience, evidenced skills, desired roles, preferences, and location with the source title/company/location/status and full prepared description. The description is shown in a disabled text area and is not interpreted as HTML. Missing descriptions and descriptions of 250 characters or fewer receive distinct warnings. The screen intentionally hides method, condition, rank, and score.

Choose one of:

- `0: Unsuitable` when the available evidence does not support the role fit.
- `1: Plausible` when there is some relevant evidence but substantial uncertainty or gaps.
- `2: Strong fit` when the evidence strongly aligns with the profile, without implying hiring likelihood.
- `Insufficient evidence` when the saved information is too sparse or ambiguous; provide a reason and leave `relevance_score` blank.

Leave the rating unselected and save only notes to mark a job unreviewed. Use Previous/Next to move through the fixed retail profile pool. Progress counts distinguish rated, insufficient-evidence, and unreviewed jobs. Save each judgment explicitly; saved decisions resume by stable `(profile_id, job_id)` even if the app closes.

The review CSV preserves earlier judgments and other profiles' rows if present. Ratings use the existing evaluator schema: `profile_id`, `job_id`, and `relevance_score` containing `0`, `1`, `2`, or blank. The app adds `review_status`, `insufficient_evidence_reason`, reviewer, and notes. Blank scores are ignored by `read_review_labels`; no label is inferred from status or notes.

## Evaluate Later

Do not run evaluation until reviewers have supplied genuine labels and the full shared pool for each profile is judged. From the repository root:

```powershell
& ".\.venv\Scripts\python.exe" .\code\evaluate_description_coverage.py `
  --recommendations .\data\matching\description_coverage_20261007T141953Z\description_coverage_recommendations.csv `
  --labels .\data\matching\description_coverage_20261007T141953Z\retail_review_judgments.csv
```

This reads only saved rankings and labels, writes `shared_pool_metrics.json` beside the recommendations by default, and does not refit or rerank. Shared-pool nDCG requires all retrieved jobs in each profile's union pool to have labels; Precision@5 requires five returned and judged results. Missing judgments remain missing, and no metrics are currently claimed.
