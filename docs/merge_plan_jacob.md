# Jacob's Plan: answers to the merge questions

My answers to every question about merging `project_leads` and `predictleads-hackathon`. Background for each question is in [`merge_plan.md`](merge_plan.md) (section numbers in brackets).

**How to use it:** tick one option per question (`[x]`), or write your own under "Other", then add a line on why. **Jesse:** copy this file as `merge_plan_jesse.md`, clear the ticks, and fill in yours; then we compare the two and log what we agree on in the decision log.

★ = the proposal in the merge plan.

---

## A. The six decisions

### A1. Which jobs are in the project? [3, decision 1]
- [ ] ★ Computer and mathematical (8 role buckets) **plus technical sales as a ninth bucket**; the app's three areas become filters
- [ ] Computer and mathematical only (technical sales out)
- [ ] Jesse's three areas only (AI/ML, data, technical sales)
- [ ] Other:

Why:

### A2. Which labels do the jobs use? [3, decision 1]
- [ ] ★ The 8 O*NET role buckets (they match the career-switch data, so the retraining compass works), with the app's areas mapped onto them
- [ ] Jesse's 17-class role families, with the role buckets mapped onto them
- [ ] Both, kept separate
- [ ] Other:

Why:

### A3. Which listings count as still open? [3, decision 2]
- [ ] ★ Last seen within 2 days of the snapshot; `status` ignored; "historical listing" wording kept
- [ ] Not marked `status = closed` (the app's rule today)
- [ ] Show both, labelled differently
- [ ] Other:

Why:

### A4. Grounded explanations (listing text sent to OpenAI) [3, decision 3]
- [ ] ★ Not used in OpenAI mode until the advisors allow it; the local mock stays for development
- [ ] Allowed with the user's consent
- [ ] Removed from the merged repo
- [ ] Other:

Why:

### A5. Magic fill (résumé text sent to OpenAI) [3, decision 3]
- [ ] ★ Off by default; only with the person's consent each time, pending the advisors' answer on résumés
- [ ] Off until the advisors answer
- [ ] Removed from the merged repo
- [ ] Other:

Why:

### A6. Who ranks the jobs? [3, decision 4]
- [ ] ★ Jesse's ranker (TF-IDF + BGE with RRF) is the baseline; Jacob adds role suggestions from real career moves, the timing model and the hiding rules, each measured against it
- [ ] Jacob's planned ranker replaces it
- [ ] Two rankers side by side in the app
- [ ] Other:

Why:

### A7. Which pipeline prepares the data? [3, decision 5]
- [ ] ★ Jacob's stream and cleaning define the listings and flags; Jesse's corpus and embeddings are rebuilt on that set (a few hours on his GPU)
- [ ] Jesse's pipeline; Jacob's flags are added to it
- [ ] Keep both; reconcile counts later
- [ ] Other:

Why:

### A8. Listings without a description in the app [3, decision 5]
- [ ] Shown, labelled "no description" (they can't be text-ranked, so they sort after ranked ones)
- [ ] Hidden in the app, kept in the analysis
- [ ] Other:

Why:

### A9. What "Compass" means [3, decision 6]
- [ ] ★ Both kept: Jesse's tab becomes "Saved-jobs insights"; "Retraining compass" stays for the career paths
- [ ] Jesse's tab keeps "Compass"; the career paths get another name
- [ ] Other names:

Why:

---

## B. Timing and process

### B1. When to merge [5]
- [ ] ★ As soon as the four readiness gates are met (decisions agreed, both streams finished, nothing in flight, paths set by a setting)
- [ ] On a fixed date:
- [ ] After the deadline (about 2026-10-23)

Why:

### B2. The deadline
- [ ] About 2026-10-23 (from Jesse's roadmap)
- [ ] Other:

### B3. How changes reach `main` after the merge
- [ ] Pull requests reviewed by the other person
- [ ] Each person merges their own area (`app/` or `analysis/`); pull requests for shared files only
- [ ] Other:

### B4. Who owns what
- `app/` (backend, frontend):
- `analysis/` (notebooks, pipeline):
- Shared files (README, requirements, data layout):
- Deck and demo:

---

## C. The repository

### C1. Where the merged repo lives
- [ ] ★ A new shared GitHub organization (both repos move in; it shows on both our profiles)
- [ ] Jacob's account
- [ ] Jesse's account

Organization name:

### C2. Repository name
- [ ] `predictleads-career-changers`
- [ ] `project_leads`
- [ ] `predictleads-hackathon`
- [ ] Other:

### C3. Visibility
- [ ] ★ Private (the docs describe PredictLeads data under an agreement)
- [ ] Public, after removing anything about the data source

### C4. Docs layout [4]
- [ ] ★ Top level: `README.md`, `PRIORITY_QUEUE.md` (intact, the one task list), `STATUS.md`, `CLAUDE.md`; Jacob's docs unchanged in `docs/project/`; `docs/analysis/` and `docs/app/`; only `roadmap.md`'s overlap folded in
- [ ] Other:

---

## D. Mechanical choices

### D1. `sentence-transformers` version [4]
- [ ] 6.1 (Jacob's), after checking Jesse's ranker and embeddings still work
- [ ] 5.1 (Jesse's)
- [ ] Whichever passes Jesse's tests; embeddings re-encoded if it changes

### D2. Virtual environment name in the README
- [ ] `.venv` (any existing name still works)
- [ ] Other:

### D3. Notebook outputs
- [ ] ★ `nbstripout` for the whole repo (already on in `project_leads`)
- [ ] Clear by hand before each commit

### D4. Data and assistant rules (`CLAUDE.md`)
- [ ] ★ Repo-wide
- [ ] Analysis folder only

### D5. Code packages `code/` and `src/`
- [ ] ★ Keep both until after the deadline
- [ ] Combine during the merge

---

## E. Anything else to raise with Jesse

-
