# Project context for Claude Code

## Project
- **Project question:** _TODO: one sentence the whole team agrees on._
- **Deliverables:** _TODO (e.g., analysis notebook, model, final presentation)._
- **Deadline:** _TODO_
- **Methods in scope:** text analysis with TF-IDF and related techniques; supervised and unsupervised ML. _(Edit as the team decides.)_

## Structure
- `notebooks/`: exploration and experiments (one subfolder per person is fine)
- `src/`: reusable Python code, imported into notebooks
- `data/`: local data only, git-ignored, **never read by Claude**
- `.env`: non-secret settings (AWS profile, region, S3 path), git-ignored
- Environment: Python 3.11 venv, packages in `requirements.txt`

## Data & AI rules (from project advisors)
AI tools are allowed for coding, brainstorming, and debugging. **Do not upload or paste confidential project data into them.** For Claude, that means:
- Never read `data/`, `.env`, raw data files, or `~/.aws/` (also blocked in `.claude/settings.json`).
- Never print data values (`df.head()`, sample rows) in commands you run.
- Work from code, column names, dtypes, shapes, and descriptions. Use **synthetic data** for examples and tests.
- Never put project details, data, file paths, or bucket names into web searches.
- Do not commit data, notebook outputs, credentials, or venvs.

## Agents
| Agent | Role | When to use |
|---|---|---|
| `focus` | Guards scope | New ideas, drift checks |
| `depth-maxer` | Pushes rigor and understanding | Choosing or reviewing methods |
| `discourse` | One-round devil's advocate | Major decisions |
| `researcher` | Finds methods, docs, citations | Unknowns, best practice |
| `quality-control` | Methodology + data-safety checks | Before every commit |

**Protocol:**
- Use agents at **decision points and before commits**, not on every small edit.
- Debates are **one round**, moderated by the main session (`/debate <question>`).
- **Final decisions belong to the user.** Agents flag trade-offs; they don't override.
- Keep agent outputs short and in their defined formats.

## Commands
- `/debate <question>`: structured focus → research → depth → discourse → recommendation
- `/precommit`: quality-control review of staged changes

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.
