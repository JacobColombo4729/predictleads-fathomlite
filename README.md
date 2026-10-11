# project_leads

PredictLeads × TripleTen externship project.

**What we're building:** role discovery and job listing prioritization for career changers moving into computer and mathematical roles. See [`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md).

> ⚠️ **This repo is private and confidential.** Everything in it is covered by our Training Services Agreements. Read the [Data & Security Rules](#data--security-rules) before you start working.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repo](#2-clone-the-repo)
3. [Create your virtual environment](#3-create-your-virtual-environment)
4. [Install packages](#4-install-packages)
5. [Register the Jupyter kernel](#5-register-the-jupyter-kernel)
6. [Set up AWS credentials](#6-set-up-aws-credentials)
7. [Create your `.env` file](#7-create-your-env-file)
8. [Open the project in VS Code](#8-open-the-project-in-vs-code)
9. [Test your setup](#9-test-your-setup)
10. [Daily Git workflow](#10-daily-git-workflow)
11. [Data & Security Rules](#data--security-rules)
12. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Install these once:

| Tool | Mac | Windows |
|---|---|---|
| **Git** | `xcode-select --install` | [git-scm.com](https://git-scm.com/download/win) |
| **Python 3.11** | `brew install python@3.11` | [python.org](https://www.python.org/downloads/), check **"Add Python to PATH"** |
| **VS Code** | [code.visualstudio.com](https://code.visualstudio.com) | same |
| **AWS CLI** | `brew install awscli` | [AWS CLI installer](https://aws.amazon.com/cli/) |

In VS Code, install the **Python** and **Jupyter** extensions (both by Microsoft).

Check that everything installed:

```bash
git --version
python3.11 --version      # Windows: py -3.11 --version
aws --version
```

> **Where to put the project:** Use a folder that is **not** synced to iCloud, Dropbox, OneDrive, or Google Drive (for example `~/dev` on Mac or `C:\Users\<you>\dev` on Windows). Synced folders copy data to the cloud, which the agreement does not allow.
> **Windows:** OneDrive often syncs `Desktop` and `Documents` by default, so don't put the project there.

---

## 2. Clone the repo

Accept the GitHub collaborator invite (check your email), then:

```bash
cd ~/dev                  # or wherever you keep projects
git clone https://github.com/JacobColombo4729/project_leads.git
cd project_leads
```

You should now see (plus a few planning files):

```
project_leads/
├── .claude/
├── .gitattributes
├── .gitignore
├── .env.example
├── .vscode/settings.json
├── CLAUDE.md
├── README.md and the project docs (what each one is for: the table at the top of STANDARDS.md)
├── requirements.txt
├── docs/          (decision log, data reference, job growth plan, notebook text exports)
├── notebooks/
└── src/
```

---

## 3. Create your virtual environment

Everyone uses the **same name (`.venv`)** and **same Python version (3.11)**.

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

Your prompt should now start with `(.venv)`. **Activate it every time you open a new terminal** for this project. The VS Code terminal activates it for you once you've selected the interpreter (step 8).

Once it's activated, `python` and `pip` refer to the venv on every OS, so the rest of this README uses them.

> `.venv/` is already in `.gitignore`. Never commit it. A venv only works on the machine that made it, so if you work on both a Mac and a Windows PC, create one on each. An existing venv with another name works too (one teammate's is `PL/`, also ignored): use its name in place of `.venv`.

---

## 4. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
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

## 5. Register the Jupyter kernel

With `.venv` activated:

```bash
python -m ipykernel install --user --name project_leads --display-name "Python (project_leads)"
```

This makes **Python (project_leads)** selectable in both VS Code and JupyterLab.

---

## 6. Set up AWS credentials

You'll receive AWS credentials **privately** from a team lead or advisor, never over Slack or in a commit.

```bash
aws configure --profile team_project
```

Enter:
- **AWS Access Key ID:** *(your key)*
- **AWS Secret Access Key:** *(your secret)*
- **Default region:** `us-east-2` *(same as `AWS_REGION` in step 7)*
- **Output format:** `json`

Your keys are stored in `~/.aws/credentials` (Windows: `C:\Users\<you>\.aws\credentials`), **outside the project folder**, so they can't be committed or read by editor tools.

> Windows: if `aws` isn't found after installing, close and reopen your terminal (and VS Code) so it picks up the new PATH.

> 🚫 **Never** put access keys in code, notebooks, `.env`, Slack, or AI chats.

---

## 7. Create your `.env` file

`.env` holds **non-secret** settings. The repo already includes a blank template, `.env.example`. Copy it to `.env`:

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

Ask the team for the bucket and export date. Each variable goes on its own line. Both `scratch.py` (step 9) and the EDA notebook read it.

> The file must be named exactly `.env`, not `.env.txt`. On Windows, File Explorer hides extensions by default, so create and rename it from VS Code or the terminal, not Explorer.

> `.env` is in `.gitignore`. Only `.env.example` (blank values) is committed.

---

## 8. Open the project in VS Code

```bash
code .
```

> If `code` isn't found on Mac: open VS Code → `Cmd + Shift + P` → **Shell Command: Install 'code' command in PATH** → restart Terminal.
> On Windows, `code` is added to PATH by the installer ("Add to PATH" option). Restart your terminal after installing.
> Or use **File → Open Folder** and pick `project_leads`.

Then:
1. `Cmd/Ctrl + Shift + P` → **Python: Select Interpreter** → choose the one in `.venv`
   (Mac: `.venv/bin/python`, Windows: `.venv\Scripts\python.exe`)
2. Open any notebook → click **Select Kernel** (top right) → **Python (project_leads)**

The shared `.vscode/settings.json` works on Mac and Windows. It:
- loads `.env` into Python runs,
- activates the selected venv in new VS Code terminals,
- **turns Copilot off for this project** until our advisors confirm the AI policy (see below).

It doesn't hard-code the interpreter path, because that path differs between Mac and Windows. The Python extension detects `.venv` automatically.

---

## 9. Test your setup

Create a test script in `notebooks/` called `scratch.py` (it's git-ignored: each person keeps their own) with:

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

With `.venv` activated, run it from the project root (same command on every OS):

```bash
python notebooks/scratch.py
```

✅ Versions plus a number means you're ready. It prints only versions and a count, no data.

---

## 10. Daily Git workflow

**Never commit directly to `main`.** Work on a branch and open a pull request.

```bash
# 1. Start from the latest main
git checkout main
git pull

# 2. Create a branch for your task
git checkout -b yourname/short-description     # e.g. jacob/eda-job-openings

# 3. Work, then check what changed
git status                                     # make sure NO data files, .env, or .venv appear

# 4. Clear notebook outputs before committing
#    VS Code: "Clear All Outputs" button at the top of the notebook, then save
#    Or from the terminal (any OS, venv activated):
#    jupyter nbconvert --clear-output --inplace path/to/notebook.ipynb

# 5. Commit and push
git add <files>
git commit -m "Short description of what you did"
git push -u origin yourname/short-description
```

6. On GitHub, open a **Pull Request** into `main` and ask a teammate to review.

> **Stuck in a strange editor after `git commit`?** That's Vim. Press `Esc`, type `:q!`, and press Enter to cancel, then commit with `-m "message"`.
> To use VS Code for commit messages instead: `git config --global core.editor "code --wait"`

**Where code goes:**
- `notebooks/`: exploration, EDA, experiments
- `src/`: reusable functions (import with `from src.module import func`)

---

## Data & Security Rules

These come from the Training Services Agreements we each signed. Breaking them can cost the **whole team** the professional reference.

### Data
- ✅ Keep data **only** in a local `data/` folder (git-ignored) or in S3.
- 🚫 **Never commit data**, including CSV, Parquet, Excel, pickles, and trained model files.
- 🚫 Never upload data to personal cloud drives, public Colab/Kaggle, or anywhere outside the project.
- 🚫 Never share data, screenshots of data, or notebook outputs outside the team.
- 🚫 Don't try to re-identify the anonymized data.

### AI tools
> **Pending written guidance from our advisors.** Until then:
- 🚫 Don't paste real data, `df.head()` output, or error messages containing data values into any AI tool.
- 🚫 Don't use AI coding extensions on files in this project (Copilot is disabled in `.vscode/settings.json`).
- ✅ It's fine to ask AI general questions ("how do I groupby in pandas?") or use **synthetic** example data.

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
| `code: command not found` | See step 8 |
| `python3.11: command not found` (Mac) | Install Python 3.11 (step 1): `brew install python@3.11` |
| `py -3.11` says no such version (Windows) | Install Python 3.11 from python.org. `py -0` lists the versions you have |
| `python` opens the Microsoft Store (Windows) | Settings → Apps → Advanced app settings → App execution aliases → turn off the `python.exe` / `python3.exe` aliases. Or just activate `.venv` first |
| Kernel not showing in VS Code | Activate `.venv`, re-run step 5, then reload VS Code (`Cmd/Ctrl + Shift + P` → **Reload Window**) |
| `ModuleNotFoundError` | Wrong kernel or venv not activated. Check the kernel says **Python (project_leads)** |
| `LF will be replaced by CRLF` warning (Windows) | Harmless. `.gitattributes` keeps the repo's line endings consistent |
| `NoCredentialsError` / `ProfileNotFound` | Re-run step 6 and check `AWS_PROFILE` in `.env` |
| `AccessDenied` from S3 | Your credentials don't have access. Ask the team lead |
| Accidentally staged data or `.venv` | `git restore --staged <file-or-folder>` before committing |
| Accidentally **committed** data | **Don't push.** Tell the team immediately so we can remove it safely |
| PowerShell won't run `Activate.ps1` | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
