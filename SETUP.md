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

Every time you push your branch, Eve (Claude, following [`.claude/agents/eve.md`](.claude/agents/eve.md)) reads what's going up and:
- **tidies** what's clearly messy: stray files, dead code, debug prints, broken links, a missing change log line. She does this in her own `Eve: tidy` commit, which goes up with your next push;
- **asks** about anything that needs a judgment call, with lettered options. Answer in the terminal (`1a 2c`) and she applies your choices. If your answers open something up, she follows up, three rounds at most;
- **ends with an open question:** what you want the other person to make sure they do. Those asks live in [`.claude/ASKS.md`](.claude/ASKS.md), and Eve brings them up on the other person's pushes until they're done. That's how the two of you stay on the same page.

She never stops a push, touches data, or changes what your code does without your answer. She runs on your own Claude Code subscription. Pushing from VS Code's buttons skips her questions, and the report is saved in `.git/eve-last-review.md`. To skip her for one push: `EVE_SKIP=1 git push`. More in README step 14.

**Your first ask is already waiting:** decide what happens to `app/STATUS.md`. Either delete it, or merge what's still useful into `app/APP.md` and drop the rest.

## 3. How we work together

- **Work on your own branch** (`jesse`, `jacob`), and merge into `main` when something is ready.
- **Log every change** in your own change log ([`JESSE.md`](JESSE.md), [`JACOB.md`](JACOB.md)): grouped by topic, most important first, one line per change showing the end result. Read the other one to catch up. If you use Claude, it keeps your log for you.
- **No other rules** (README step 14).

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
