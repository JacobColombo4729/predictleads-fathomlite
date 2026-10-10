# Merge plan: `project_leads` + `predictleads-hackathon`

**A proposal for Jesse and Jacob to confirm.** Drafted 2026-10-10. Nothing is merged until we both agree on the decisions in section 4.

## 1. The short version

- **One repository,** with both commit histories kept: the app in `app/`, the analysis in `analysis/`, one set of docs and one data layout.
- **When:** about **2026-10-13 to 10-14**, right after `project_leads` reruns EDA_SDA_2 on the full file, and before either side builds more ranking code. The target date is about 2026-10-23, so later merges cost more than they save.
- **First, six decisions** (section 4): scope and role taxonomy, which listings count as live, outside AI services, one ranker, one data pipeline, and what "Compass" means.

## 2. Why merge

The two projects have converged on the same product: career changers, the 2026-09-30 snapshot, a ranked list with a "Compass". Each has what the other lacks.

| `predictleads-hackathon` (Jesse) | `project_leads` (Jacob) |
|:--|:--|
| A working app: FastAPI backend, React frontend, profiles, ranked jobs, feedback, saved jobs | Analysis notebooks (EDA_SDA_1–4, TimeSeries) with every decision logged |
| TF-IDF + BGE ranker (RRF), embeddings for about 245,000 listings | **Timing:** how long listings stay open, an apply-by estimate, ghost-listing flags |
| Saved-jobs analysis (themes, fit tiers, evidence lines) | **Retraining compass:** where 2,813 real career changers (public resume dataset) moved, joined to each role's 2026 market |
| Signals: hiring bursts by company | Career-change accessibility: hiding rules, entry-level and "open to newcomers" signals |
| Resume "Magic fill" (opt-in) | A full-file stream (about 13.4 million listings; about 1.0 million in scope) |

Both sides already scanned the full file (13.4 million listings), so the data agrees at the source.

## 3. Proposed layout and how

```
app/          backend/, frontend/, code/ (today's predictleads-hackathon)
analysis/     notebooks/, src/ (today's project_leads)
docs/         both sets of docs; one README as the map
data/         one git-ignored layout, rebuilt from the same snapshot
```

1. Each side finishes and pushes open work (including `feature/mac-support` into `main`).
2. On a new branch, each repo moves its files into its folder (`git mv`, so file history follows).
3. Join the two with `git merge --allow-unrelated-histories`: no commit is lost.
4. Fix the shared files (section 5), check that the app's tests pass and the notebooks run.
5. Open one pull request; both of us review before it goes to `main`.

## 4. Decisions to confirm first

| # | Decision | The difference today | Proposal |
|:-:|:--|:--|:--|
| 1 | **Scope and role taxonomy** | App: AI/ML, data and technical sales or solutions engineering (its own 17-class role families). Analysis: computer and mathematical occupations in 8 role buckets (O*NET codes, shared with the career-switch data) | One taxonomy: role buckets, with the app's families mapped onto them. Decide whether technical sales (an O*NET sales occupation) stays in scope |
| 2 | **Which listings count as live** | App: excludes `status = closed` and treats the rest as unknown. Analysis: `status` is never "open", so a listing is live if last seen within 2 days of the snapshot (checked: no weekly crawl pattern) | Use the last-seen rule, so the app doesn't show listings that disappeared weeks ago; keep the app's honest "historical listing" wording |
| 3 | **Outside AI services** | App: Magic fill and grounded explanations call an outside model, with consent. Analysis rule (from the project advisors): no PredictLeads data or listing text goes to outside AI tools | Keep both features off by default until the advisors answer; never send listing text. Resume text only with the person's consent, as today |
| 4 | **One ranker** | App: TF-IDF + BGE with RRF (working). Analysis plan (`ML_Fit`): the same steps, then a model trained on real career moves, plus a timing model (`ML_Rank`) | The app's ranker becomes the baseline; the analysis adds the trained layer and timing on top, measured against it |
| 5 | **One data pipeline** | App: targeted pool (about 320,000), corpus with descriptions (about 245,000). Analysis: computer and mathematical scope (about 1.0 million) plus a lean table of every listing | One pipeline produces the scope and its features; the app's corpus and embeddings are built from it |
| 6 | **"Compass" means two things** | App: insights about a person's saved jobs. Analysis: the retraining compass (real career paths) | Rename one: for example "Saved-jobs insights" for the app's tab, and keep "retraining compass" for the paths |

## 5. Smaller conflicts (mechanical)

| Conflict | Resolution |
|:--|:--|
| `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `notebooks/`, `docs/` exist in both | Merge each into one |
| **`sentence-transformers` 5.1 vs. 6.1** (a major version), plus 9 other pins that differ | One `requirements.txt`. Check that the app's embeddings and ranker work on 6.1, or pin one version for both; embeddings must be re-encoded if the model code changes |
| Two task trackers: `STATUS.md` and `roadmap.md` vs. `PRIORITY_QUEUE.md` and `PROJECT_OUTLINE.md` | Keep `STATUS.md` (start here) and one task list; fold the rest in |
| S3 location written in committed docs (app) vs. kept only in `.env` (analysis rule) | Move it to `.env.example` as a placeholder |
| Windows scripts (`.ps1`) and Mac scripts (`.sh`); `.venv` vs. `PL` | Keep both script sets; one venv name in the README |
| Notebook outputs | `nbstripout` for the whole repo (outputs never reach GitHub) |
| Data rules and assistant instructions (`CLAUDE.md`) | Apply them repo-wide |
| Code packages `code/` and `src/` | Keep both for now; consolidate after the deadline |
| Hiring signals (app) vs. company-history features and a company watch list (analysis plan) | Combine later; the app's Signals tab stays |

## 6. Checklist

- [ ] Both agree on decisions 1–6 (and log them in the decision log)
- [ ] `feature/mac-support` merged into `main`; both sides pushed
- [ ] Merge branch: files moved, histories joined, shared files merged
- [ ] App tests pass; notebooks run on the merged layout
- [ ] Pull request reviewed by both
- [ ] After merging: rebuild the app's corpus from the shared pipeline, wire timing, hiding rules and the retraining compass into the backend

## 7. What happens to each side until then

- **Analysis:** finishes the full-file stream and reruns EDA_SDA_2; pauses `ML_Fit` (it would duplicate the app's ranker).
- **App:** keeps polishing the demo; no new ranking or data-pipeline code until the merge.
