# team_project

PredictLeads × TripleTen externship project.

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

> **Where to put the project:** Use a folder that is **not** synced to iCloud, Dropbox, OneDrive, or Google Drive (for example `~/dev`). Synced folders copy data to the cloud, which the agreement does not allow.

---

## 2. Clone the repo

Accept the GitHub collaborator invite (check your email), then:

```bash
cd ~/dev                  # or wherever you keep projects
git clone https://github.com/JacobColombo4729/team_project.git
cd team_project
```

You should now see:

```
team_project/
├── .gitignore
├── .env.example
├── .vscode/settings.json
├── README.md
├── requirements.txt
├── notebooks/
└── src/
```

---

## 3. Create your virtual environment

Everyone uses the **same name (`.venv`)** and **same Python version (3.11)**.

**Mac / Linux**
```bash
python3.11 -m venv .venv # I replaced .venv to 'PL' for my venv name
source .venv/bin/activate # PL/bin/activate for me
```

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. **Activate it every time you open a new terminal** for this project.

> `.venv/` is already in `.gitignore`. Never commit it.

---

## 4. Install packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
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

This makes **Python (team_project)** selectable in both VS Code and JupyterLab.

---

## 6. Set up AWS credentials

You'll receive AWS credentials **privately** from a team lead or advisor, never over Slack or in a commit.

```bash
aws configure --profile team_project
```

Enter:
- **AWS Access Key ID:** *(your key)*
- **AWS Secret Access Key:** *(your secret)*
- **Default region:** `us-east-1` *(confirm with the team)*
- **Output format:** `json`

Your keys are stored in `~/.aws/credentials`, **outside the project folder**, so they can't be committed or read by editor tools.

> 🚫 **Never** put access keys in code, notebooks, `.env`, Slack, or AI chats.

---

## 7. Create your `.env` file

`.env` holds **non-secret** settings. Copy the template:

Open the team_project folder in VS Code (File → Open Folder).
In the Explorer sidebar, hover over the project name and click the New File icon.
Name it exactly '.env.example', with the leading dot and no .txt.
Paste in the following and save with Cmd + S:

```
AWS_PROFILE=
AWS_REGION=
S3_BUCKET=
```

Repeat with a new file called `.env` and fill in:

```
AWS_PROFILE=team_project
AWS_REGION=us-east-2
S3_BUCKET=<ask the team>
```

> `.env` is in `.gitignore`. Only `.env.example` (blank values) is committed.

---

## 8. Open the project in VS Code

```bash
code .
```

> If `code` isn't found on Mac: open VS Code → `Cmd + Shift + P` → **Shell Command: Install 'code' command in PATH** → restart Terminal.
> Or use **File → Open Folder** and pick `team_project`.

Then:
1. `Cmd/Ctrl + Shift + P` → **Python: Select Interpreter** → choose `.venv`
2. Open any notebook → click **Select Kernel** (top right) → **Python (team_project)**

The shared `.vscode/settings.json` points VS Code to `.venv` and loads `.env` automatically. It also **turns Copilot off for this project** until our advisors confirm the AI policy (see below).

---

## 9. Test your setup

Create a scratch notebook in `notebooks/` called `scratch.py` and run:

```python
import os, boto3, pandas as pd, sklearn
from dotenv import load_dotenv

load_dotenv()
s3 = boto3.Session(profile_name=os.getenv("AWS_PROFILE")).client("s3")
resp = s3.list_objects_v2(Bucket=os.getenv("S3_BUCKET"), MaxKeys=5)

print("pandas", pd.__version__, "| sklearn", sklearn.__version__)
print("S3 objects visible:", resp.get("KeyCount"))
```

python notebooks/scratch.py

✅ Versions plus a number means you're ready. Then **clear the output and delete the scratch notebook.**

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
#    VS Code: "Clear All Outputs" button at the top of the notebook

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
| `python3.11: command not found` | Install Python 3.11 (step 1). On Mac: `brew install python@3.11` |
| Kernel not showing in VS Code | Activate `.venv`, re-run step 5, then reload VS Code (`Cmd/Ctrl + Shift + P` → **Reload Window**) |
| `ModuleNotFoundError` | Wrong kernel or venv not activated. Check the kernel says **Python (team_project)** |
| `NoCredentialsError` / `ProfileNotFound` | Re-run step 6 and check `AWS_PROFILE` in `.env` |
| `AccessDenied` from S3 | Your credentials don't have access. Ask the team lead |
| Accidentally staged data or `.venv` | `git restore --staged <file-or-folder>` before committing |
| Accidentally **committed** data | **Don't push.** Tell the team immediately so we can remove it safely |
| PowerShell won't run `Activate.ps1` | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
