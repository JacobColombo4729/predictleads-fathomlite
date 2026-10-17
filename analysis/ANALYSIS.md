# Analysis

> What's left, top first. Delete a task when it's done. The app has its own queue: [`../app/APP.md`](../app/APP.md). **[agent]** = Jacob's Foreman may draft it (README step 14).

**Scope frozen (2026-10-16, under 2 weeks left): nothing new is built before the pitch.** Cut for now: combined model steps 6 (the trained layer driving the Compass) and 8 (fine-tuning BGE); growth beyond the slider at weight 0 (BLS); ML_Rank sections 9, 10 and 12, H7 and 7f's extra baselines; differentiation ideas 1–7 (1, 3 and 5 stay as tested functions in `src/differentiators.py`); new ideas and features. Their plans are in Git history (commit `893ece8`).

## Now

1. **Cut and condense the notebooks,** one at a time in pipeline order: EDA_SDA_1–4 (may merge), TimeSeries, Profiles (frozen: text only), ML_Fit, ML_Rank, Evaluate. For each: a keep / cut / condense table → agree → edit → File: Revert File → rerun → regenerate its export. Keep every hand-off a later notebook or the app reads, and the source of every quoted number; a cut step leaves one "tested, not used" line. Small fixes go in with each notebook's review: ML_Fit's `split_people` cutoff (counted on all people, not on development people); EDA_SDA_4 9d's text rows and whether to test `career_open`.
2. **Rerun Evaluate on combined model v2,** then update the app's number in `SUMMARY.md` (76.4% is v1's).
3. **"Why this listing" in the app:** pass `breakdown` through in `app/backend/api.py` and add a compact block to the job detail view: the four factors with their share of the score, and the role with its top reason. Touches Jesse's app: agree it with him first.
4. **Check the docs after the notebooks:** every figure `SUMMARY.md`, `PROJECT.md` and `../DIFFERENTIATORS.md` quote still exists, and condense `DIFFERENTIATORS.md`. During each notebook's review, also repoint its mentions of `STANDARDS.md` (deleted; its rules are in the outline, How we keep it honest) and `analysis/DIFFERENTIATORS.md` (now at the root).
5. **Any time:** delete `data/flat/all_fields_lean_full/` (1.1 GB; nothing reads it) and the sample-era tables in `data/clean/` (`analysis_table.pkl`, `scope_table.pkl`, `job_openings_lean_200000_v2.pkl`).

## Pitch

- One slide with Evaluate's comparison chart: baseline → text match (`rrf`) → plus the queue (`mlrank`) → plus role fit (combined), on the same 563 career changers.
- Demo the combined model only (`combined_demo/`), rehearsed on the laptop that presents.

## Blocked

- **Team:** confirm the final due date and deliverables.
- **Project advisor:** AI tools and data, hosting, presenting results, data deletion; get the AI-and-data answers in writing and record them in your change log.
- **PredictLeads:** the data questions in `PROJECT.md` (Questions for PredictLeads).

## Later

- **Profiles is frozen** (refrozen 2026-10-10); reopen only if needed, then rerun it and reload `modeling_table.pkl` downstream.
- **Volunteer applicant profiles** (`docs/volunteer_profiles.md`), once the advisor answers.
- **Known gaps in `.claude/settings.json`:** `Read(./.env.*)` also blocks `.env.example`; `Read` rules don't stop shell commands (the guard in `safety/` covers those).

## Always

- Clear notebook outputs, including every `data-rows` cell, before committing.
- After a notebook is edited on disk: **File: Revert File**, then run.
- After any change to `src/`, restart the kernel before rerunning.

## Finish

- **Permanently delete every copy of the data:** `data/` (including `data/handoffs/`), `reports/`, the GPU PC's data folder, `~/dev/predictleads-hackathon/data/`, downloads (including `~/Downloads/1_corpus_and_signals.zip`), wherever the hand-off zip was shared, and any S3 copies. Confirm the deletion in writing if asked.
