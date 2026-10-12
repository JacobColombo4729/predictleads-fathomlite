# predictleads-fathomlite

PredictLeads × TripleTen externship project: role discovery and job listing prioritization for career changers, built on PredictLeads job data. Which tech roles a person's background transfers to, which listings they can realistically get, and which to apply to first. What's built: [`STATUS.md`](analysis/STATUS.md). What's planned: [`VISION.md`](analysis/VISION.md). Every doc: [Where to read more](#where-to-read-more).

> ⚠️ **This repo is private and confidential.** Everything in it is covered by our Training Services Agreements. Read the [Data & Security Rules](#data--security-rules) before you start working.

This repository joins two earlier ones with their full commit histories: `analysis/` (was `project_leads`, Jacob) and `app/` (was `predictleads-hackathon`, Jesse). They share one environment, one `.env`, one `data/` and one `models/` at the top.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Prerequisites](#2-prerequisites)
3. [Clone the repo](#3-clone-the-repo)
4. [Create your virtual environment](#4-create-your-virtual-environment)
5. [Install packages](#5-install-packages)
6. [Register the Jupyter kernel](#6-register-the-jupyter-kernel)
7. [Set up AWS credentials](#7-set-up-aws-credentials)
8. [Create your `.env` file](#8-create-your-env-file)
9. [Data and models](#9-data-and-models)
10. [Open the project in VS Code](#10-open-the-project-in-vs-code)
11. [Test your setup](#11-test-your-setup)
12. [Run the app](#12-run-the-app)
13. [Rebuild the app's corpus](#13-rebuild-the-apps-corpus)
14. [Daily Git workflow](#14-daily-git-workflow)
15. [Data & Security Rules](#data--security-rules)
16. [Troubleshooting](#troubleshooting)
17. [Where to read more](#where-to-read-more)

---

## 1. Repository layout

```
predictleads-fathomlite/
├── .claude/  CLAUDE.md       instructions and data rules for Claude Code
├── .vscode/settings.json     shared VS Code settings
├── .env.example              template for your .env (settings and keys)
├── .gitattributes  .gitignore
├── requirements.txt          one pinned list for everything (Python 3.11)
├── .venv/  .env              your own, git-ignored
├── data/  models/            local only, git-ignored: source data, corpus, embeddings, the BGE model
├── analysis/
│   ├── notebooks/            EDA_SDA_1–4, TimeSeries, Profiles, ML_Fit, ML_Rank
│   ├── src/                  shared code imported by the notebooks
│   ├── docs/                 data reference, notebook text exports
│   └── STATUS, VISION, PRIORITY_QUEUE, DECISION_LOG, DIFFERENTIATORS, STANDARDS
│                             the project docs (see Where to read more)
└── app/
    ├── backend/              FastAPI demo API
    ├── frontend/             React + Vite app
    ├── code/                 corpus and ranking pipeline, BGE text, role classifier, résumé fill, signals
    ├── tests/                live tests (plus backend/tests/)
    ├── docs/  notebooks/  archive/
    └── run_demo_backend / run_tests (.ps1 for Windows, .sh for macOS / Linux)
```

What each doc is for: [Where to read more](#where-to-read-more).

---

## 2. Prerequisites

Install these once:

| Tool | Mac | Windows |
|---|---|---|
| **Git** | `xcode-select --install` | [git-scm.com](https://git-scm.com/download/win) |
| **Python 3.11** | `brew install python@3.11` | [python.org](https://www.python.org/downloads/), check **"Add Python to PATH"** |
| **Node.js** (for the app's frontend) | `brew install node` | [nodejs.org](https://nodejs.org) |
| **VS Code** | [code.visualstudio.com](https://code.visualstudio.com) | same |
| **AWS CLI** | `brew install awscli` | [AWS CLI installer](https://aws.amazon.com/cli/) |

In VS Code, install the **Python** and **Jupyter** extensions (both by Microsoft).

Check that everything installed:

```bash
git --version
python3.11 --version      # Windows: py -3.11 --version
node --version
aws --version
```

> **Where to put the project:** Use a folder that is **not** synced to iCloud, Dropbox, OneDrive, or Google Drive (for example `~/dev` on Mac or `C:\Users\<you>\dev` on Windows). Synced folders copy data to the cloud, which the agreement does not allow.
> **Windows:** OneDrive often syncs `Desktop` and `Documents` by default, so don't put the project there.

---

## 3. Clone the repo

Accept the GitHub collaborator invite (check your email), then:

```bash
cd ~/dev                  # or wherever you keep projects
git clone https://github.com/JacobColombo4729/predictleads-fathomlite.git
cd predictleads-fathomlite
```

---

## 4. Create your virtual environment

One venv for the whole repository, at the top. Everyone uses the **same name (`.venv`)** and **same Python version (3.11)**.

**Mac / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash)**
```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

Your prompt should now start with `(.venv)`. **Activate it every time you open a new terminal** for this project. The VS Code terminal activates it for you once you've selected the interpreter (step 10).

Once it's activated, `python` and `pip` refer to the venv on every OS, so the rest of this README uses them. The app's scripts (`app/run_*.ps1`, `app/run_*.sh`) find `.venv` on their own.

> `.venv/` is already in `.gitignore`. Never commit it. A venv only works on the machine that made it, so if you work on both a Mac and a Windows PC, create one on each. An existing venv with another name works too (one teammate's is `PL/`, also ignored): use its name in place of `.venv`, but the app's scripts expect `.venv`.

---

## 5. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
cd app/frontend && npm install && cd ../..
```

> On Windows, upgrade pip with `python -m pip`, not `pip install --upgrade pip`, which fails because pip can't replace itself while it's running.
> A few packages in the list (e.g. `appnope`) are Mac-only helpers. They install harmlessly on Windows, so use the same file everywhere.

**Mac only:** LightGBM and XGBoost need the OpenMP runtime. Install it once with Homebrew:

```bash
brew install libomp
```

**For the text notebooks (EDA_SDA_3, ML_Fit):** download spaCy's English model once:

```bash
python -m spacy download en_core_web_sm
```

**Adding a new package:**
1. `pip install <package>`
2. Add it to `requirements.txt` with a pinned version (e.g. `seaborn==0.13.2`)
3. Commit `requirements.txt` and tell the team
4. Teammates re-run `pip install -r requirements.txt` after pulling

---

## 6. Register the Jupyter kernel

With `.venv` activated:

```bash
python -m ipykernel install --user --name fathomlite --display-name "fathomlite"
```

This makes **fathomlite** selectable in both VS Code and JupyterLab, for the analysis notebooks and `app/notebooks/`.

---

## 7. Set up AWS credentials

You'll receive AWS credentials **privately** from a team lead or advisor, never over Slack or in a commit.

```bash
aws configure --profile team_project
```

Enter:
- **AWS Access Key ID:** *(your key)*
- **AWS Secret Access Key:** *(your secret)*
- **Default region:** `us-east-2` *(same as `AWS_REGION` in step 8)*
- **Output format:** `json`

Your keys are stored in `~/.aws/credentials` (Windows: `C:\Users\<you>\.aws\credentials`), **outside the project folder**, so they can't be committed or read by editor tools.

> Windows: if `aws` isn't found after installing, close and reopen your terminal (and VS Code) so it picks up the new PATH.

> 🚫 **Never** put access keys in code, notebooks, `.env`, Slack, or AI chats.

---

## 8. Create your `.env` file

One `.env` at the top of the repository holds the settings for both the notebooks and the app. Copy the template:

**Mac / Linux / Git Bash**
```bash
cp .env.example .env
```

**Windows (PowerShell)**
```powershell
Copy-Item .env.example .env
```

Open `.env` in VS Code and fill in:

```
AWS_PROFILE=team_project
AWS_REGION=us-east-2
S3_PATH=s3://<bucket>/<YYYY>/<MM>/<DD>/
```

Ask the team for the bucket and export date. Each variable goes on its own line. `scratch.py` (step 11), the notebooks and the app all read it.

`OPENAI_API_KEY` is only needed for the app's optional OpenAI features (step 12), which are off by default.

> The file must be named exactly `.env`, not `.env.txt`. On Windows, File Explorer hides extensions by default, so create and rename it from VS Code or the terminal, not Explorer.

> `.env` is in `.gitignore`. Only `.env.example` (blank values) is committed.

---

## 9. Data and models

Everything lives in two git-ignored folders at the top, shared by the notebooks and the app:

- **`data/`:** the notebooks' tables (`clean/`, `flat/`, `profiles/`) and the app's corpus and signals (`matching/`, `discovery/`, `prepared/`, `signals/`, `preview/`). The notebooks download what they need from S3; the app's corpus is built by the pipeline in step 13 or shared within the team. The raw snapshot is not stored locally, so deleting `data/matching/targeted_pool_*` or the random sample under `data/discovery/` means re-reading it from S3.
- **`models/`:** the BGE model `BAAI/bge-base-en-v1.5` (revision pinned in `app/code/bge_text.py`) in `models/bge-base-en-v1.5`, or set `PREDICTLEADS_BGE_MODEL_DIR`.

`PL_DATA_DIR` points the notebooks at another data folder (used for tests on synthetic data).

Teammates share `data/` and `models/` as zip files over a private channel; the agreement forbids public posting.

---

## 10. Open the project in VS Code

Open the **repository root** (not `analysis/` or `app/`), so the shared settings and `.env` apply:

```bash
code .
```

> If `code` isn't found on Mac: open VS Code → `Cmd + Shift + P` → **Shell Command: Install 'code' command in PATH** → restart Terminal.
> On Windows, `code` is added to PATH by the installer ("Add to PATH" option). Restart your terminal after installing.
> Or use **File → Open Folder** and pick `predictleads-fathomlite`.

Then:
1. `Cmd/Ctrl + Shift + P` → **Python: Select Interpreter** → choose the one in `.venv`
   (Mac: `.venv/bin/python`, Windows: `.venv\Scripts\python.exe`)
2. Open any notebook → click **Select Kernel** (top right) → **fathomlite**

The shared `.vscode/settings.json` works on Mac and Windows. It:
- loads `.env` into Python runs,
- activates the selected venv in new VS Code terminals,
- **turns Copilot off for this project** until our advisors confirm the AI policy (see below).

It doesn't hard-code the interpreter path, because that path differs between Mac and Windows. The Python extension detects `.venv` automatically.

---

## 11. Test your setup

**Analysis:** create a test script in `analysis/notebooks/` called `scratch.py` (it's git-ignored: each person keeps their own) with:

```python
import os, boto3, pandas as pd, sklearn
from dotenv import load_dotenv

load_dotenv()
s3_path = os.getenv("S3_PATH")
if not s3_path:
    raise RuntimeError("S3_PATH is not set — add `S3_PATH=s3://<bucket>/<prefix>/` on its own line in .env")
BUCKET, _, PREFIX = s3_path.strip().removeprefix("s3://").partition("/")

s3 = boto3.Session(profile_name=os.getenv("AWS_PROFILE")).client("s3")
resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=5)

print("pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("S3 objects visible:", resp.get("KeyCount"))
```

With `.venv` activated, run it from the repository root (same command on every OS):

```bash
python analysis/notebooks/scratch.py
```

✅ Versions plus a number means you're ready. It prints only versions and a count, no data.

**App:** run its tests (they use synthetic data, so they pass without `data/`):

```powershell
.\app\run_tests.ps1                  # live Python tests
.\app\run_tests.ps1 -Frontend        # also type-check and build the React app
.\app\run_tests.ps1 -Archive         # also the archived experiment tests
```

macOS / Linux: `./app/run_tests.sh`, with `--frontend` and `--archive`.

---

## 12. Run the app

**FathomLite** is the app: profile (with optional **Magic fill** from a résumé), ranked jobs with 👍/👎 feedback, saved jobs, the **Compass** (analysis of picked jobs, with an optional AI skills gap analysis) and **Signals** (who is hiring, with **Ask the market**). What each piece does and how well it's verified: [`STATUS.md`](analysis/STATUS.md#app-fathomlite).

Everything in the app is a hackathon prototype. Listings are **historical records, not verified open positions**, and no ranking-quality claim is made: the ranker has not been compared against alternatives on this audience.

It needs the serving corpus at `data/matching/ai_ml_serving_v1` (step 13) and the BGE model (step 9).

1. **Backend** (about 3.5 minutes to load, about 2.5 GB of memory):
   ```powershell
   .\app\run_demo_backend.ps1            # add -ResumeFill to enable Magic fill; -Ranker tfidf for the lighter ranker
   ```
   macOS / Linux: `./app/run_demo_backend.sh` (flags `--resume-fill`, `--ranker tfidf`, `--port N`). On a laptop, `--preview` runs TF-IDF on a 5,000-job sample in `data/preview/jobs.jsonl` (starts in ~20 s, no model needed, Compass unavailable); the script prints how to build the sample if it is missing.
2. **Frontend**: `cd app/frontend; npm run dev` (macOS / Linux: `cd app/frontend && npm run dev`), then open http://127.0.0.1:5173.
3. **OpenAI features** (all off by default): **Magic fill** (`-ResumeFill`) sends the résumé text you choose to OpenAI after you tick a consent box; `-Assess` turns on the skills gap analysis; `-Market` turns on typed questions in Ask the market. Each needs `OPENAI_API_KEY` in `.env`.

What is in `app/`:

| Path | What it is |
|---|---|
| `app/backend/` | FastAPI demo API: sessions, profiles, search, ranked recommendations (TF-IDF + BGE reciprocal-rank fusion), relevance feedback, saved jobs, the Compass analysis, grounded assessment (off by default) and résumé extraction (opt-in). See [app/docs/backend-api.md](app/docs/backend-api.md). |
| `app/frontend/` | React + Vite app wired to the API (Discover, Saved, Compass, Signals, Profile). See [app/frontend/README.md](app/frontend/README.md). |
| `app/code/` | Python modules: matching engine (`job_matching_engine`), BGE text builders (`bge_text`), corpus and ranking pipeline (`rank_candidates`, `extract_targeted_pool`, `prepare_jobs`), role-family classifier, grounded explanations, résumé extraction, review tooling, and the early data-discovery scripts. |
| `app/tests/`, `app/backend/tests/` | Live tests (step 11). |
| `app/archive/experiments/` | Earlier experiments, kept as history. Nothing live imports them. See its [README](app/archive/experiments/README.md). |
| `app/docs/` | Data discovery and preparation notes, API contracts, the grounded-explanation contract. `app/docs/archive/` holds historical documents. |
| `app/notebooks/` | Data discovery notebook (local-only view). |

---

## 13. Rebuild the app's corpus

Run these from `app/` with `.venv` activated; each script's `--help` lists its options. Outputs go to the top-level `data/`.

1. `python code/extract_targeted_pool.py --scan` streams the full snapshot (about 19 GB from S3, using the `predictleads` AWS profile; about 36 minutes) and keeps AI/ML, data and technical-sales listings. `--check-sample` tests the matcher offline.
2. `python code/rank_candidates.py build-corpus --targeted-dir ../data/matching/targeted_pool_<timestamp>` builds the deduplicated corpus. `encode` then `finalize-encode` compute the BGE embeddings (GPU recommended).
3. `python code/rank_candidates.py export-serving` writes the serving subset the backend loads.
4. `python code/rank_candidates.py rank ...` ranks the 12 test profiles; `python code/build_review_page.py --run-dir <ranking run> --profiles <profile files>` builds an optional blinded review page.
5. `python code/role_family_classifier.py` trains and evaluates the supporting role-family classifier.

---

## 14. Daily Git workflow

**Each person works on their own branch** (`jacob`, `jesse`; decided 2026-10-11) and merges it into `main` only when they decide a piece of work is ready (step 5). Merge `main` into your branch **at least once a day**, so the two branches never drift far apart: a long-running branch once left the team 155 commits behind.

```bash
# 1. Start on your branch, with everything already merged into main
git switch jacob                               # or jesse
git pull                                       # your branch's latest from GitHub
git fetch origin
git merge origin/main

# 2. Work, then check what changed
git status                                     # make sure NO data files, .env, or .venv appear

# 3. Clear notebook outputs before committing (nbstripout does it on commit if you ran `nbstripout --install`)
#    VS Code: "Clear All Outputs" button at the top of the notebook, then save
#    Or from the terminal (any OS, venv activated):
#    jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb

# 4. Commit and push your branch
git add <files>                                # name the files; avoid `git add .`
git commit -m "Short description of what you did"
git push

# 5. Only when YOU decide the work is ready for everyone (app tests pass, notebooks run): merge it into main.
#    This is a deliberate step, never routine.
git switch main
git pull
git merge jacob                                # or jesse
git push
git switch jacob                               # back to your branch
```

If a merge reports a conflict, fix the files it names, `git add` them, run `git commit`, then push. Ask the team if you're unsure.

> **Stuck in a strange editor after `git commit`?** That's Vim. Press `Esc`, type `:q!`, and press Enter to cancel, then commit with `-m "message"`.
> To use VS Code for commit messages instead: `git config --global core.editor "code --wait"`

**Where code goes:**
- `analysis/notebooks/`: exploration, EDA, experiments
- `analysis/src/`: reusable functions for the notebooks (import with `from src.module import func`)
- `app/`: the web app and its pipeline

---

## Data & Security Rules

These come from the Training Services Agreements we each signed. Breaking them can cost the **whole team** the professional reference.

### Data
- ✅ Keep data **only** in the local `data/` folder (git-ignored) or in S3.
- 🚫 **Never commit data**, including CSV, Parquet, Excel, JSONL, pickles, databases, and trained model files.
- 🚫 Never upload data to personal cloud drives, public Colab/Kaggle, or anywhere outside the project.
- 🚫 Never share data, screenshots of data, or notebook outputs outside the team.
- 🚫 Don't try to re-identify the anonymized data.
- Résumés and real profiles stay out of git. Missing information is **unknown**, not a gap. Null status is not "open".

### AI tools
> **Pending written guidance from our advisors.** Until then:
- 🚫 Don't paste real data, `df.head()` output, or error messages containing data values into any AI tool.
- 🚫 Don't use AI coding extensions on files in this project (Copilot is disabled in `.vscode/settings.json`).
- ✅ It's fine to ask AI general questions ("how do I groupby in pandas?") or use **synthetic** example data.
- The app sends nothing to an outside AI service unless the person whose text it is agrees first. Magic fill removes emails, phone numbers and links before sending.
- OpenAI use is allowed for the app's features (confirmed 2026-10-11), off by default. The shared spend cap is used up: any paid call needs a new decision.

### Notebooks
- ✅ **Clear all outputs before every commit.** Outputs can contain data.

### GitHub
- ✅ Repo stays **private**, with **no license**. The work belongs to PredictLeads/TripleTen.
- 🚫 Don't fork it or make it public.

### If something goes wrong
If data or credentials are exposed (committed, shared, uploaded), **tell the team immediately**. We are required to notify PredictLeads and TripleTen right away (TripleTen requires notice within **1 hour**). If AWS keys are exposed, deactivate them in the AWS console first.

### When the project ends
Everyone must **permanently delete all copies of the data** (local `data/` folder, downloads, S3 copies) and may need to confirm this in writing.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `code: command not found` | See step 10 |
| `python3.11: command not found` (Mac) | Install Python 3.11 (step 2): `brew install python@3.11` |
| `py -3.11` says no such version (Windows) | Install Python 3.11 from python.org. `py -0` lists the versions you have |
| `python` opens the Microsoft Store (Windows) | Settings → Apps → Advanced app settings → App execution aliases → turn off the `python.exe` / `python3.exe` aliases. Or just activate `.venv` first |
| Kernel not showing in VS Code | Activate `.venv`, re-run step 6, then reload VS Code (`Cmd/Ctrl + Shift + P` → **Reload Window**) |
| `ModuleNotFoundError` | Wrong kernel or venv not activated. Check the kernel says **fathomlite** |
| App script can't find Python | The scripts expect `.venv` at the repository root (step 4) |
| `LF will be replaced by CRLF` warning (Windows) | Harmless. `.gitattributes` keeps the repo's line endings consistent |
| `NoCredentialsError` / `ProfileNotFound` | Re-run step 7 and check `AWS_PROFILE` in `.env` |
| `AccessDenied` from S3 | Your credentials don't have access. Ask the team lead |
| Accidentally staged data or `.venv` | `git restore --staged <file-or-folder>` before committing |
| Accidentally **committed** data | **Don't push.** Tell the team immediately so we can remove it safely |
| PowerShell won't run `Activate.ps1` | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Where to read more

Each fact has one home; other docs link to it. Only this README and `CLAUDE.md` sit at the top; the project docs are in `analysis/`.

| Doc | Its one job |
|:--|:--|
| `README.md` (this file) | Setup, running, Git workflow and the data rules |
| [`STATUS.md`](analysis/STATUS.md) | **What's built and verified,** for the analysis (data, notebooks, findings) and the app; known limits. Start here |
| [`VISION.md`](analysis/VISION.md) | **Everything not built yet:** the agreed merge changes, screens, the retraining compass (section 3, with the job growth plan in 3.3), the priority model, agents, stages, open questions and later ideas |
| [`PRIORITY_QUEUE.md`](analysis/PRIORITY_QUEUE.md) | **The one task list** for both halves |
| [`DECISION_LOG.md`](analysis/DECISION_LOG.md) | Every decision, with its reasoning |
| [`DIFFERENTIATORS.md`](analysis/DIFFERENTIATORS.md) | **How we stand out:** the angles, the pitch, how we talk about the data, and what we never claim |
| [`analysis/STANDARDS.md`](analysis/STANDARDS.md) | How the analysis works: success criteria, working standards, validation and **leakage rules**, scope, roles, risks |
| [`analysis/docs/data.md`](analysis/docs/data.md) | Data sources, fields, quirks and auto-generated schemas |
| `analysis/docs/*.md` (EDA_SDA_*, ML_*, Profiles, TimeSeries) | Text exports of the notebooks, generated by `src/export_markdown.py`; don't edit by hand |
| `CLAUDE.md` | Instructions for Claude Code |

**App reference** (`app/docs/`):
1. Components: [backend API](app/docs/backend-api.md), [API contracts](app/docs/api-contracts.md), [hiring signals](app/docs/hiring-signals.md), [Ask the market](app/docs/ask-the-market.md), [feedback data and the path to production](app/docs/feedback-data.md), [grounded match explanations](app/docs/grounded-match-explanations.md), [matching engine notes](app/docs/job-matching-engine.md), [ML inventory (what is ML, why, how validated)](app/docs/ml-inventory.md), [what the first/last-seen dates mean](app/docs/signal-semantics.md), [frontend](app/frontend/README.md)
2. Early data discovery (the scripts still in `app/code/`): [discovery summary](app/docs/discovery-summary.md), [data preparation](app/docs/data-preparation.md), [preparation review](app/docs/preparation-review.md), [preparation proposal](app/docs/data-preparation-proposal.md), [role exploration](app/docs/role-exploration.md), [artifact index](app/docs/artifact-index.md)
3. History: [app/docs/archive/](app/docs/archive/) (including the review page and run log) and [app/archive/experiments/](app/archive/experiments/README.md)
