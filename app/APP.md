# App

> The app's priority queue. Top = next. Delete a task when it's done. The design is in [`DESIGN.md`](DESIGN.md); the analysis side has its own queue in [`../analysis/ANALYSIS.md`](../analysis/ANALYSIS.md).
> **[agent] = Jesse's Foreman may draft it in the background** (code and tests on made-up data, on its own branch for you to review; never merges, pushes or calls a paid API; README step 14). Start a task's line with [agent] to hand it off, or `[agent: Jacob]` to hand it to Jacob's Foreman; start the loop in a second Claude Code window: `/loop Follow "The loop" in .claude/agents/foreman.md, with a cap of 3 tasks`.

---

## Now

- **Jesse:** clone the repo, `git switch jesse`, `git merge origin/main`, then `python -m fetch.get_bge` for the model; Eve and the helper agents are optional and use your Claude allowance, so they stay off unless you ask (`SETUP.md` section 2); log your changes in `JESSE.md`, and read `JACOB.md` to catch up.
- **Test Ask the market** on the Signals tab as a person: typed questions, the example chips, **Talk** with a real microphone, and **Research this company**. So far Talk was only tested with a synthetic voice.
- **End-to-end rehearsal** on a real profile with desired roles; fix what breaks. First edit the draft desired roles on the two real profiles if they're wrong.
- **AI skills gap analysis:** live-tested and working (Jesse, 2026-10-10); still to record the cost and the time for 10 and 20 jobs. Anything that sends résumé text to a provider needs consent and a budget decision.

## Next

- **Discover: company size filter** (the job level filter is built): **Startup**, **Mid-size**, **Large company**. The companies table has **no employee count**, so size needs a proxy: the company's listing volume in any field (EDA_SDA_1's bands: 1–4, 5–49, 50–499, 500+ listings), plus publicly traded (`ticker`) and having a parent company. Name the options for what they measure (for example "hires rarely / sometimes / at volume") unless a proxy holds up as size; decide the cutoffs from EDA_SDA_1 and 2.
- **Collect ratings for the first measured number:** the team uses Discover normally and rates jobs (with a reason on each 👎), then runs `python code/feedback_report.py`: the share of 👍 in the top 10, and per source for the exploration listings. (Labelling review pairs to measure the ranker was dropped for time on 2026-10-09; the demo ranker, RRF, is unvalidated until this runs.)
- **Retraining compass view** ([`DESIGN.md`](DESIGN.md) sections 2 and 3); its 2026 side is ready (EDA_SDA_2 3.12).
- **Ideas tab** ([`DESIGN.md`](DESIGN.md), a separate tab): skills to build (EDA_SDA_3 skills vs. a résumé, in `ML_Fit`), career options and time to switch (Profiles), pay increase (EDA_SDA_1 7.2 current-field pay vs. EDA_SDA_2 3.12 role pay, hourly), expected growth, and stepping-stone roles; built from `ML_Fit`'s compass tables. Time frames stay vague until the team decides (`../analysis/ANALYSIS.md`).
- **MVP check against the design** (stage 1 in [`DESIGN.md`](DESIGN.md)): list view with search, sort and filters; detail view with apply link; stars.
- **Demo plan:** live or recorded, a recorded fallback, local or deployed.
- **Clean notebook and write-up** for the app, built on [docs/ml-inventory.md](docs/ml-inventory.md), with measured versus unvalidated claims stated plainly.
- **Refresh stale docs** (the frontend README) as items land.
- **Crisp logos:** the FathomLite logo images were cut from a screenshot of the design sheet, so they're low resolution. Get the original files from the designer and replace `frontend/src/assets/fathomlite-mark.png`, `fathomlite-icon.png` and `frontend/public/favicon.png`; for the exact landing page look, also the original background image (it replaces the two `landing__wall` images and the gradient in `frontend/src/styles/landing.css`).

## Blocked

- **Open design questions:** answer [`DESIGN.md`](DESIGN.md) section 9.

## Later

- **Connect the app to the analysis only after every notebook is complete** (EDA_SDA_3, EDA_SDA_4, TimeSeries, ML_Fit, ML_Rank): the shared listing set → the app's corpus and embeddings → backend features (timing, the flags ranked lower, the retraining compass).
  - **Recruiter names and contact details:** assumed allowed in the app (2026-10-12); filter them out if the advisors or PredictLeads say no. Confirm before the final presentation. The analysis keeps only an anonymous recruiter key (EDA_SDA_1), so names and contacts would come from the raw listings when the app is connected. (OpenAI use: confirmed allowed 2026-10-11.)
- **Later stages** ([`DESIGN.md`](DESIGN.md)): the transferable-fit stage (the career changer's background profile), the priority model with adjustable weights, MCP servers, then agents.
- **"Refine my matches"** (decision pending since 2026-10-09): a per-user button trained on the user's own 👍/👎 ratings; plan, thresholds and risks in [`roadmap.md`](roadmap.md). Revisit after the core demo and write-up.
- More ideas, ranked by value: [`roadmap.md`](roadmap.md) section 7.

---

## Keep in mind (honesty rules and limits)

- **Nothing is validated yet:** the demo ranker (equal-weight TF-IDF + BGE RRF), the Compass thresholds and relevance feedback have no measured result. Say so plainly in the demo until the ratings above give a first number.
- **Listings are historical, not verified openings.** Listings marked `closed` in the source are left out; the 81,284 served have unknown status, and openness is never inferred from dates or a missing status.
- **Missing information on a profile is unknown, not a gap.** Years, degree and certification requirements are never marked "supported" automatically.
- **A Signals burst is a timing trigger, not a volume forecast** (about 64% of the volume carries into the next month, vs. 99% for sustained hiring).
- **Personal data:** résumés and real profiles stay under the git-ignored `data/`; nothing goes to an outside AI service without the person's consent each time. The six held-out weighted-RRF profiles stay unopened (they're the clean test set).
- **Known limits:** title and O*NET matching can miss unusual titles and include false hits; duplicates collapse only when company, title and description are identical; the role-family classifier (94.7% test accuracy) matches the provider's labels, not fit, so it isn't used as a fit predictor.
