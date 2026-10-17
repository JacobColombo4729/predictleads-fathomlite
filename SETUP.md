# Setup

> Start here if you're new to the repo (hi, Jesse 👋). It gets you running and shows you around; the README has the details for every step.

## 1. Get set up (about 20 minutes)

1. **Accept the GitHub invitation** (the email from GitHub, or the repo's invitations page), then clone `predictleads-fathomlite`.
2. **Switch to your branch:** `git switch jesse`, then `git merge origin/main` for everything new.
3. **Environment:** one Python 3.11 `.venv` at the top, from `requirements.txt` (README steps 4–6).
4. **`.env`:** copy `.env.example` to `.env` and fill in the AWS settings (`S3_PATH`, `AWS_PROFILE`) and your `OPENAI_API_KEY` (README steps 7–8).
5. **Data:**
   - **BGE model:** `python -m fetch.get_bge`.
   - **The app's serving corpus:** move your existing copy into `data/matching/ai_ml_serving_v1` (one `data/` folder at the top now), or rebuild it (README step 13).
   - **The raw PredictLeads export** (optional): `python -m fetch.get_predictleads`.
   - **spaCy's English model** (needed by EDA_SDA_3 and ML_Fit): `python -m spacy download en_core_web_sm`.
   - **Running the notebooks** (to rebuild the tables and the hand-offs): `EDA_SDA_1`, `Profiles`, `EDA_SDA_2`, `EDA_SDA_3`, `EDA_SDA_4`, `TimeSeries`, `ML_Fit`, `ML_Rank`, in that order (README step 9).
6. **Eve (optional):** `git config core.hooksPath .githooks` turns her on. She runs Claude at every push and commit, so it uses your Claude allowance; leave it off if that matters.
7. **Run the app:** `.\app\run_demo_backend.ps1` (Mac: `./app/run_demo_backend.sh`), then `cd app/frontend; npm run dev`, and open http://127.0.0.1:5173 (README step 12).

## 2. Meet Eve, the reviewer, and the helper agents

Every time you push your branch, Eve (Claude, following [`.claude/agents/eve.md`](.claude/agents/eve.md)) tidies what's clearly messy in a commit of her own, asks about judgment calls in a normal Claude conversation with the usual question boxes (type `/exit` when she's done), and ends with an open question: what you want the other person to make sure they do (kept in [`.eve/ASKS.md`](.eve/ASKS.md)). How she works, and how to skip her: README step 14.

Three more helpers run only when you ask, in any Claude Code session in this repo (definitions in `.claude/agents/`):
- **Thread** keeps a notebook's text true to its results and the notebooks after it consistent: `thread <notebook>` after a code change, `thread <notebook> after a rerun` after a run.
- **Pace** makes a notebook faster and lighter on memory without changing its results, measured on synthetic data: `pace <notebook>` before a long run.
- **Foreman** drafts tasks you mark [agent] in your queue (`analysis/ANALYSIS.md` for Jacob, `app/APP.md` for Jesse), one branch each for you to review, in a `/loop` you start yourself (README step 14). It never merges or pushes.

None of them reads data or keys, and none runs unless you turn it on or ask for it by name. Each run uses your Claude allowance.

## 3. How we work together

Work on your own branch, merge into `main` when something is ready, and log every change in your own change log ([`JESSE.md`](JESSE.md), [`JACOB.md`](JACOB.md)). That's all; details in README step 14.

## 4. Where things live

| Where | What |
|---|---|
| [`app/APP.md`](app/APP.md) | The app's priority queue. Your to-dos from `roadmap.md` and the old `STATUS.md` are merged into it; check nothing's missing |
| [`app/DESIGN.md`](app/DESIGN.md) | The app's design (moved from the analysis folder) |
| [`analysis/ANALYSIS.md`](analysis/ANALYSIS.md) | The analysis priority queue, for when you cross over |
| `fetch/` | Where every dataset comes from: one download script per dataset, nothing hard-coded (bucket names and keys only in `.env`) |
| `pretrained/` | Downloaded models (it used to be called `models/`) |
| [`CLAUDE.md`](CLAUDE.md) | The project rules Claude follows |

## 5. What's new in the app

- **Job level filter on Discover:** Entry level, Experienced, Senior, Manager, Director and above, or Level not stated. The ranked list starts on Entry level, and each job card shows its level.
- **AI features on by default:** Magic fill, the gap analysis and Ask the market. Just run the script; `-NoAI` turns them off. The old `-ResumeFill`, `-Assess` and `-Market` flags are gone.

## 6. The one rule that never bends

Never paste or upload project data into any AI tool. The data rules are in README step 9 and `CLAUDE.md`.
