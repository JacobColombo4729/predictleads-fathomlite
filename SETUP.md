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
6. **Turn on Eve:** `git config core.hooksPath .githooks`. She needs Claude Code installed.
7. **Run the app:** `.\app\run_demo_backend.ps1` (Mac: `./app/run_demo_backend.sh`), then `cd app/frontend; npm run dev`, and open http://127.0.0.1:5173 (README step 12).

## 2. Meet Eve, the reviewer

Every time you push your branch, Eve (Claude, following [`.claude/agents/eve.md`](.claude/agents/eve.md)) tidies what's clearly messy in a commit of her own, asks about judgment calls in a normal Claude conversation with the usual question boxes (type `/exit` when she's done), and ends with an open question: what you want the other person to make sure they do (kept in [`.eve/ASKS.md`](.eve/ASKS.md)). How she works, and how to skip her: README step 14.

**Your first ask is already waiting:** decide what happens to `app/STATUS.md`. Either delete it, or merge what's still useful into `app/APP.md` and drop the rest.

## 3. How we work together

Work on your own branch, merge into `main` when something is ready, and log every change in your own change log ([`JESSE.md`](JESSE.md), [`JACOB.md`](JACOB.md)). That's all; details in README step 14.

## 4. Where things live

| Where | What |
|---|---|
| [`app/APP.md`](app/APP.md) | The app's priority queue. Your to-dos from `roadmap.md` and `STATUS.md` are merged into it; check nothing's missing |
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
