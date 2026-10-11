# Merge plan: `project_leads` (Jacob) + `predictleads-hackathon` (Jesse)

**A proposal for Jesse and Jacob to confirm.** Updated 2026-10-11. **The histories are joined** (2026-10-11): both repositories now live, unchanged, in a third one, `predictleads-fathomlite`, with `project_leads` in `analysis/` and `predictleads-hackathon` in `app/`. The two originals are kept intact. Nothing has been *combined* yet (data, rules, requirements, docs): that waits for both of us to agree on the six decisions in section 3.

## 1. The short version

- **One repository,** `predictleads-fathomlite`, with both commit histories kept: the app in `app/`, the analysis in `analysis/`, one set of docs, one data layout. **Done so far:** the join (each half still runs on its own, with its own `requirements.txt`, `.env` and `data/`). **Still to do:** everything after the join (section 6).
- **The originals stay intact:** `project_leads` and `predictleads-hackathon` are not moved, renamed or changed; new work happens in `predictleads-fathomlite`. They can be archived on GitHub later (read-only, still intact).
- **When:** the join was done 2026-10-11, ahead of gates 1 and 4, because it changes no code or rules (section 5). Combining the two halves follows as soon as the six decisions are agreed. The project's target date is about 2026-10-23.
- **Why now:** the two projects have converged on the same product (career changers, the 2026-09-30 data export, a ranked list, a "Compass"), and each has what the other lacks. Merging early stops us building the same thing twice.

## 2. The two repositories, and the merged one

### Today, side by side

```
project_leads/  (Jacob: analysis)            predictleads-hackathon/  (Jesse: app)
├── notebooks/                               ├── backend/        FastAPI: api, rrf_ranker, feedback_ranking,
│   ├── EDA_SDA_1–4, TimeSeries              │                   compass, signals (+ tests)
│   ├── Profiles  (real career switches)     ├── frontend/       React + Vite: Discover, Saved, Compass,
│   └── ML_Fit, ML_Rank  (planned)           │                   Signals, Profile
├── src/                                     ├── code/           data scan, corpus, BGE text, ranking,
│   ├── stream_full.py  (full-file stream)   │                   role classifier, résumé fill, signals
│   ├── listing_rows.py (cleaning rules)     ├── tests/
│   ├── career_switch.py (role buckets)      ├── notebooks/      data_discovery
│   ├── eda_helpers.py, pl_data.py           ├── docs/           API, data prep, matching engine, signals
│   └── export_markdown.py                   ├── archive/        earlier experiments
├── docs/        decisions, data, plans      ├── README.md, STATUS.md, roadmap.md
├── PROJECT_OUTLINE, STANDARDS,              ├── requirements.txt (19 packages)
│   APPLICATION, DIFFERENTIATORS,            ├── run_demo_backend / run_tests (.ps1, .sh)
│   PRIORITY_QUEUE, CLAUDE.md                ├── .env.example (OPENAI_API_KEY)
├── requirements.txt (187 packages)          ├── data/    (git-ignored)
├── .env.example                             └── models/  (git-ignored: local BGE model)
└── data/        (git-ignored)
```

### Now: joined, not yet combined (2026-10-11)

```
predictleads-fathomlite/             (private, on Jacob's GitHub account; Jesse a collaborator)
├── analysis/          all of project_leads, unchanged (its own requirements.txt, .env, data/)
├── app/               all of predictleads-hackathon, unchanged (its own requirements.txt, .env, data/, models/)
├── CLAUDE.md          paths updated for analysis/ and app/; data rules cover both halves
├── .claude/           Claude's read-deny rules for data/ and .env in either half
├── README.md          a map of the two halves
└── .gitignore         repo-wide safety net (each half keeps its own)
```

The repository is on Jacob's personal account for now; it can be transferred to a shared GitHub organization later (Settings → Transfer ownership: history, issues and collaborators move with it, and old URLs redirect).

### After combining (the target)

```
predictleads-fathomlite/
├── app/
│   ├── backend/        Jesse's API, plus timing, hiding rules and the retraining compass
│   ├── frontend/       Jesse's React app, plus the retraining compass view
│   ├── code/           Jesse's corpus, BGE and ranking code, built on the shared listing set
│   └── tests/
├── analysis/
│   ├── notebooks/      EDA_SDA_1–4, TimeSeries, Profiles, ML_Fit, ML_Rank
│   └── src/            stream, cleaning rules, role buckets, helpers
├── docs/
│   ├── project/        Jacob's docs, unchanged: PROJECT_OUTLINE, STANDARDS,
│   │                   APPLICATION, DIFFERENTIATORS
│   ├── analysis/       decisions, data, job growth, fit analysis, merge plan, notebook exports
│   └── app/            Jesse's docs: API, data prep, matching engine, signals, roadmap
├── README.md           one map of the repository, for both sides
├── PRIORITY_QUEUE.md   the one task list, kept intact (Jesse's remaining work added as items)
├── STATUS.md           Jesse's: how to run the app and where it stands
├── CLAUDE.md           stays at the top (Claude Code reads it there)
├── requirements.txt    one pinned list for both
├── .env.example        S3 settings and OPENAI_API_KEY as placeholders
├── data/               git-ignored: one layout, rebuilt from the same data export
└── models/             git-ignored
```

### How the pieces connect after the merge

```mermaid
flowchart LR
    S3[(PredictLeads export<br/>2026-09-30, 13.4M listings)] --> STREAM[analysis/src: stream and cleaning<br/>one listing set and its flags]
    STREAM --> NB[analysis notebooks<br/>EDA, TimeSeries, feature screening]
    RES[(Public résumé dataset)] --> PROF[Profiles: 2,813 real career changers]
    NB --> FEAT[Features and tables<br/>timing, ghosts, hiding rules, market per role]
    PROF --> RC[Retraining compass tables]
    NB --> RC
    STREAM --> CORPUS[app/code: corpus and BGE embeddings]
    CORPUS --> RANK[Ranker: TF-IDF + BGE with RRF<br/>Jesse's, the baseline]
    FEAT --> RANK
    RANK --> API[app/backend API]
    RC --> API
    API --> UI[app/frontend: Discover, Saved,<br/>Saved-jobs insights, Retraining compass, Signals]
```

## 3. Six decisions to confirm first

### Decision 1: which jobs are in the project, and how they're labelled

- **Jacob has:** every listing whose O*NET occupation is in the **Computer and Mathematical** group (codes starting 15-), minus credentialed math (actuaries, mathematicians), sorted into **8 role buckets**: IT support, networks and systems, security, database, software development, quality assurance, data science, IT project management. The same buckets label the career changers in Profiles, which is how the retraining compass joins "where people moved" to "today's openings". About 1.0 million listings in the full file.
- **Jesse has:** listings chosen by job title and occupation for **three target areas: AI/ML, data, and technical sales / solutions engineering** (about 320,000; the app serves about 81,000). A 17-class role classifier exists but isn't used by the app.
- **Where they differ:** AI/ML and data roles are inside Jacob's scope (mostly "data science" and "software development"). **Technical sales is not:** sales engineers are in O*NET's *Sales* group. Jacob's IT support, networks, security, QA, database and IT project management roles aren't in Jesse's.
- **Update (2026-10-10), Jacob's side:** the scope is now **the roles TripleTen trains for, plus adjacent tech roles**, a background-neutral definition: the eight computer and mathematical buckets, plus **technical sales** (sales engineers) and **business and data analysis** (management and market research analysts); IT managers and financial analysts are out. Every listing carries its TripleTen program. Jesse's three areas map onto programs: AI/ML → AI & Machine Learning, data → Data Analytics, technical sales → Forward Deployed Engineering / AI Automation.
- **Proposed:** one combined set: Jacob's computer and mathematical scope **plus technical sales as a ninth bucket**. Jesse's three areas become app filters that map onto the buckets. Technical sales gets listings but no retraining-compass paths (Profiles has no technical-sales career changers).

### Decision 2: which listings count as "still open"

- **Jacob has:** a listing is open if it was **last seen within 2 days** of September 30, the data's end date. The `status` field is only ever "closed" or empty, never "open", so it isn't used. (Checked: the crawler isn't weekly, so the 2-day rule doesn't drop open listings.)
- **Jesse has:** the app drops `status = closed` and **shows every listing with an empty status**, labelled "unknown, not verified as open". That includes listings last seen weeks or months before September 30 (the data's end date), which have most likely disappeared.
- **Proposed:** the app uses the last-seen rule, so users don't see listings that are already gone, and keeps Jesse's honest "historical listing" wording. With the planned date picker, "open" means live on the chosen date.

### Decision 3: sending text to outside AI (OpenAI)

- **Jacob has:** the project advisors' rule: **no PredictLeads data or listing text goes to outside AI tools.** In the app design, anything that reads listing text runs locally. Whether an outside model may ever see listing text is an open question to the advisors.
- **Jesse has:** two features that call OpenAI (`gpt-4.1-mini`), both off by default:
  - **Magic fill:** sends a résumé's text (contact details removed) to fill in the profile; the user ticks a consent box each time.
  - **Grounded explanations:** in OpenAI mode, **the listing's description and the candidate's evidence are sent to the provider** (his `docs/grounded-match-explanations.md`).
- **Proposed:** both stay **off by default**. **Grounded explanations in OpenAI mode are not used** until the advisors allow listing text to go to an outside model (the local mock stays for development). Magic fill only with the person's consent, as now, and pending the advisors' answer on résumés.

### Decision 4: who ranks the jobs

- **Jacob has:** a *plan*, not built yet. `ML_Fit` suggests roles from a person's background using the same steps Jesse built (keywords, TF-IDF, BGE embeddings, combined with RRF), then a model trained on the 2,813 real career moves. `ML_Rank` adds a timing model (chance a listing is still open, an apply-by date) and combines fit, accessibility and pay into one priority score.
- **Jesse has:** a **working ranker**: a profile goes in, listings are scored with TF-IDF and BGE, combined with RRF, and updated as the user rates jobs (👍/👎). Its quality hasn't been tested (the planned human review was dropped for time).
- **Proposed:** don't rebuild it. **Jesse's ranker is the starting point** of `ML_Fit`. Jacob adds what's missing (role suggestions learned from real career moves, the timing model, the hiding rules) and measures whether each addition beats Jesse's ranker.

### Decision 5: how the data is prepared

- **Jacob has:** `src/stream_full.py` keeps **all computer and mathematical listings with every field**, plus a slim table of every English US listing (for company hiring volume). Cleaning rules in `src/listing_rows.py` (hourly pay, reposts, text flags); EDA_SDA_2's flags (ghost listings, staffing agencies, accessibility). Listings without a description are kept and flagged.
- **Jesse has:** his own scan of the same file (`code/extract_targeted_pool.py`) keeps his ~320,000 target-area listings; his corpus keeps those **with a usable description** (~245,000) and collapses exact duplicates; BGE embeddings computed on his GPU.
- **Why it matters:** two pipelines give two listing counts, two duplicate rules and two ideas of what's included, so the app and the analysis would disagree.
- **Proposed:** Jacob's stream and cleaning define the listings and their flags; Jesse's corpus and embedding steps run on that set (listings with a usable description). **Cost:** his embeddings are recomputed for the larger set (roughly three times today's, a few hours on his GPU).

### Decision 6: what "Compass" means

- **Jacob has:** the **retraining compass**, the project's foundation from the team meeting: prior career → training people took → first tech job → that role's 2026 openings, pay and closing speed (Profiles and EDA_SDA_2).
- **Jesse has:** a **Compass tab**: pick 3–20 saved jobs and see their common themes, fit tiers, how the profile lines up with each requirement (with evidence lines), next steps and "hidden fit" suggestions. Runs locally.
- **Proposed:** keep both (they answer different questions) under different names, for example **"Saved-jobs insights"** for Jesse's tab and **"Retraining compass"** for Jacob's paths. Which one keeps the word "Compass" is our call.

## 4. Smaller conflicts (mechanical)

| Conflict | Resolution |
|:--|:--|
| `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `notebooks/`, `docs/` exist in both | Merge each into one |
| **`sentence-transformers` 5.1 (Jesse) vs. 6.1 (Jacob)**, a major version; 9 other pins differ; FastAPI and uvicorn only on Jesse's side | One pinned `requirements.txt`. Check Jesse's ranker and embeddings on 6.1, or pin one version for both |
| Docs on both sides; some overlap | **Nothing is deleted.** Top level keeps four files: `README.md` (one map), `PRIORITY_QUEUE.md` (**intact, the one task list**), `STATUS.md` (Jesse's run-and-status page) and `CLAUDE.md`. Jacob's other docs move unchanged to `docs/project/`; analysis docs to `docs/analysis/`; Jesse's to `docs/app/`. Only true overlap is removed: `roadmap.md`'s goal and user journey fold into `PROJECT_OUTLINE.md`, its remaining work into `PRIORITY_QUEUE.md`; the rest stays as `docs/app/roadmap.md`. Every link between docs (and the paths in `CLAUDE.md`) is updated in the same commit |
| The S3 location is written in Jesse's committed docs; Jacob's rule keeps it only in `.env` | Move it to `.env.example` as a placeholder |
| Windows scripts (`.ps1`) and Mac scripts (`.sh`); `.venv` vs. `PL` | Keep both script sets; one venv name in the README |
| Notebook outputs | `nbstripout` for the whole repo |
| Data rules and assistant instructions (`CLAUDE.md`) | Apply repo-wide |
| `code/` and `src/` | Keep both; consolidate after the deadline |
| Signals (Jesse) vs. planned company-history features and watch list (Jacob) | Signals stays; combine later |

## 5. Readiness gates

**Before combining the two halves (all five):**

| Gate | Ready when | Who |
|:--|:--|:--|
| 1. Decisions agreed | Both confirm (or change) decisions 1–6 and log them | Jesse and Jacob |
| 2. Streams finished | Jacob's full stream and its second pass (`--rows`) have completed (**met 2026-10-10**) | Jacob |
| 3. Nothing in flight | Both repos committed and pushed; every branch merged into `main`; no run in progress (**met 2026-10-11**: both repos down to `main` only, clean and pushed) | Jesse and Jacob |
| 4. Paths ready | Each side finds the project root and `data/` through a setting, not "the folder with `requirements.txt`" (**met 2026-10-11** on `jacob`: notebooks find `analysis/` by `src/eda_helpers.py`; app code reaches the top-level `data/`, `models/` and `.env` through `REPO_ROOT`) | Jacob's side (Claude); Jesse's side |
| 5. EDA_SDA_1 and 2 at full scale | Both notebooks reworked for the full file and rerun: the cleaned listing set, scope, role buckets, the last-seen rule and the hiding rules the app will use are final (**met 2026-10-11**) | Claude (code), Jacob (runs) |

**Why the join went ahead of gates 1 and 4:** joining only moves each repository into its folder; no code, rule or data changes. Each half is still its own project root (the analysis code finds `analysis/`, the app runs from `app/`), so gate 4 is needed only for the single `data/` layout, and gate 1 only for the combining work. Gate 3 is what the join itself needed.

Why gate 5: the app rebuilds its corpus and embeddings from the shared listing set and copies these rules, so they should be final before it does; reworking the notebooks before the move also avoids rewriting code while it's being reorganized. EDA_SDA_3 and 4, TimeSeries, ML_Fit and ML_Rank don't feed the app yet, so they don't hold the merge.

**After combining:** the full-file rework of the remaining analysis notebooks (EDA_SDA_3 and 4, TimeSeries) inside `analysis/`; rebuilding the app's corpus and embeddings from the shared listing set; wiring timing, hiding rules and the retraining compass into the backend; consolidating code.

## 6. Steps

**Joining (done 2026-10-11), leaving both originals intact:**

- [x] Fresh clones of both repositories in a temporary folder; in each clone, every file moved into `analysis/` or `app/` (`git mv`; `CLAUDE.md` and `.claude/` stay at the top). The move commits exist only in the third repository, never in the originals
- [x] New repository `predictleads-fathomlite`; both histories joined with `git merge --allow-unrelated-histories` (no conflicts: the two halves share no paths); `git log --follow` traces files to their original commits
- [x] Top-level `CLAUDE.md` (paths), `.claude/settings.json` (data rules for both halves), `README.md` (map) and `.gitignore`
- [x] App tests pass from `app/` (256); the analysis code finds `analysis/` as its root
- [x] Pushed to GitHub (private, Jacob's account); temporary clones deleted
- [ ] Jesse added as a collaborator; each of us moves our local `data/`, `models/` and `.env` to the top of the repository and creates the one `.venv`
- [ ] Notebooks run on the new layout

**Combining (after gate 1; each person on their own branch, merged into `main` often, per the git workflow decision of 2026-10-11):**

- [ ] Gates 1 and 4 met
- [ ] Docs into `docs/project/`, `docs/analysis/` and `docs/app/`, links between docs and the paths in `CLAUDE.md` updated in the same commit
- [x] One `requirements.txt`, one `.env.example`, one `.gitignore`, one `.gitattributes`, one `README.md`, one `data/` and `models/` at the top (2026-10-11, on `jacob`; docs stay in each half)
- [ ] `nbstripout` for the whole repository
- [ ] Then: shared listing set → corpus and embeddings → backend features
- [ ] Optional: transfer the repository to a shared GitHub organization; archive the two originals
