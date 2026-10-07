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

**Interaction protocol (agents interact through the main session):**
- **Automatic cross-checks:** after any agent responds, route its output to its counterpart without being asked:
  - depth-maxer proposal → **focus** (scope) → **discourse** (challenge)
  - researcher recommendation → **discourse** (challenge) → **focus** (scope)
  - focus marks an idea "Out of scope" but it affects result validity → **depth-maxer** (rebut)
  - any modeling or pipeline change → **quality-control**
- **Handoffs:** when an agent ends with `HANDOFF → <agent>: <request>`, call that agent next with the request.
- **Debates** use `/debate <question>`: two rounds (positions, then rebuttal), then stop.
- **Shared memory:** all agents read `docs/decisions.md`. The main session appends an entry after every debate or major review. Agents never edit it.
- **Loop guard:** max 6 agent calls per task unless the user approves more. Never repeat the same agent on the same question without new information.
- **Final decisions belong to the user.** Agents flag trade-offs; they don't override.
- Keep agent outputs short and in their defined formats.

## Commands
- `/debate <question>`: two-round debate (positions → rebuttal), recommendation, logged to `docs/decisions.md`
- `/precommit`: quality-control review of staged changes

## Conventions
- Fit all preprocessing inside sklearn `Pipeline`s; set `random_state`.
- Report metrics as CV mean ± std, compared against a baseline.
- Clear notebook outputs before committing.
