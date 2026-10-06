# PredictLeads Hackathon

This workspace contains data-quality discovery and reversible preparation tools for the September 30, 2026 PredictLeads jobs and companies snapshots. It is research and reproducibility support for a proposed job-seeker experience, not a finished application or matching model.

## Product Scope

The hackathon problem space is helping job seekers explore opportunities and possible career transitions. Evidence-backed skill search, location-aware discovery, compensation-transparent filtering, and the preparation schema are proposals for team review, not finalized requirements or team decisions. See [the proposal](docs/data-preparation-proposal.md). No app, skill extraction, cross-file company enrichment, embeddings, or ML/LLM matching is implemented here.

What is implemented:

- A completed full-file quality audit of jobs and companies, with saved report and summary.
- A bounded preparation script and two validated 10,000-job development slices.
- A completed full-file jobs role exploration, exact SQLite-backed record frequencies, and a seeded uniform 30,000-job reservoir.
- Offline synthetic self-tests and reports describing data limitations.

The data files themselves are local and ignored by Git. Counts are source-record frequencies, not verified unique jobs or currently open positions.

## Read In This Order

1. [Discovery summary](docs/discovery-summary.md) for current verified findings.
2. [Artifact index](docs/artifact-index.md) to locate local reports, summaries, samples, and partial attempts.
3. [Full role exploration report](data/discovery/role_exploration/roles_corrected_20261006T000000Z/role_exploration_report.md) when local artifacts are available.
4. [Preparation methodology](docs/data-preparation.md), then [preparation review](docs/preparation-review.md).
5. [Preparation proposal](docs/data-preparation-proposal.md) for undecided product/data-contract questions.
6. [Notebook entry point](notebooks/data_discovery.ipynb) for a local-only view of the saved documentation.

## Directory Guide

- `code/`: runnable scripts for inspection, profiling, preparation, and role exploration.
- `notebooks/`: local documentation entry notebook. It does not access AWS or run scans.
- `docs/`: methodology, findings, artifact index, review notes, and proposals.
- `data/`: ignored local source copies and generated artifacts; never add these to a contribution.
- `code/prototype_v1/`: ignored `results/active_editor_state.json`; its purpose is uncertain, so it is preserved and not treated as runnable project code.
- `.venv/`: local Python environment, ignored by Git.

## Windows PowerShell Setup

From the project root:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment executable directly instead of changing machine-wide policy:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code interpreter and notebook kernel. Python 3.12 is the environment used for the saved runs.

## AWS Profile

Only optional source scans need AWS access when a matching local source file is unavailable. Configure the named profile using your organization’s approved credential process; never paste credentials into source, notebooks, chat, or this repository. The scripts expect profile `predictleads` in region `us-east-2`.

```powershell
aws configure --profile predictleads
aws configure set region us-east-2 --profile predictleads
aws sts get-caller-identity --profile predictleads
```

Do not run the identity check unless you have AWS CLI installed and are authorized. Use approved SSO/role-based configuration instead of long-lived keys where available. The configured principal must have authorized read access to the specified snapshot objects.

## Review Saved Findings

Review existing results without network access or expensive rescans:

- Open `docs/discovery-summary.md` and `docs/artifact-index.md`.
- Open the saved audit and role-exploration Markdown reports listed in the index.
- Inspect saved JSON summaries for stop reason, EOF, record totals, errors, and source metadata.
- Open `notebooks/data_discovery.ipynb` and use **Run All**; it reads the reviewed Markdown docs, saved full-population aggregate CSVs, and the prepared 30,000-job JSONL from local paths only. It does not access AWS, rerun scans, or call AI services. To run every exploration section, obtain the sample and aggregate exports through an approved private location and place them under the documented `data/discovery/role_exploration/roles_corrected_20261006T000000Z/` path. Without these local files, the notebook explains what is missing and the sample-analysis cells skip safely.

The workspace Explorer hides `data/` to reduce clutter. To reveal it, disable the `**/data` entry in `.vscode/settings.json` or use Explorer's excluded-files visibility control. Git ignore rules remain in effect regardless of Explorer visibility.

## Optional Runs

These commands can read the full remote objects and create local artifacts. They are not needed to review saved findings. Use a fresh output location for any command that accepts one, and confirm storage/access approval first.

### Optional full audit

```powershell
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
& ".\.venv\Scripts\python.exe" .\code\profile_data_quality.py --mode audit --output-dir ".\data\discovery\audit_$runId"
```

The audit can take a substantial amount of time and disk. Use its saved run summary to verify EOF; a full-looking report is not sufficient.

### Optional preparation slice

```powershell
& ".\.venv\Scripts\python.exe" .\code\prepare_jobs.py --self-test
& ".\.venv\Scripts\python.exe" .\code\prepare_jobs.py --limit 10000
```

This creates a bounded sequential development slice, not a representative sample.

### Optional full-file role exploration and sample

```powershell
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
& ".\.venv\Scripts\python.exe" .\code\explore_roles_and_sample.py --self-test
& ".\.venv\Scripts\python.exe" .\code\explore_roles_and_sample.py --sample-size 30000 --seed 20261005 --output-dir ".\data\discovery\role_exploration\roles_$runId"
```

This reads the full jobs stream to EOF to calculate exact record frequencies and a uniform reservoir over successfully parsed eligible job resources. Check available disk and use a new output directory each time. Resume from a checkpoint is not supported.

## Data Interpretation and Safety

Missing, null, blank, and invalid field states are not interchangeable. A null historical status does not establish that a role is open; observation timestamps are not posting or closing dates. The 10,000-row slices are sequential development slices, while the 30,000-row role sample is uniform only across eligible records in the completed full-file pass. Rare roles may be absent by chance. Company enrichment in preparation uses only the same wrapper's `included` resources; no separate companies-file join is implemented. No real candidate resumes are included or processed.

See [the artifact index](docs/artifact-index.md) for local-only artifacts and transfer guidance. Do not commit, email, or upload raw job records, resumes, database files, samples, or frequency exports to unapproved locations.
