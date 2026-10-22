# Analysis

> What's left, top first. Delete a task when it's done. The app has its own queue: [`../app/APP.md`](../app/APP.md). **[agent]** = Jacob's Foreman may draft it (README step 14).

**Scope frozen (2026-10-16, under 2 weeks left): nothing new is built before the pitch.** Cut for now: combined model steps 6 (the trained layer driving the Compass) and 8 (fine-tuning BGE); growth beyond the slider at weight 0 (BLS); ML_Rank sections 9, 10 and 12, H7 and 7f's extra baselines; differentiation ideas 1–7 (1, 3 and 5 stay as tested functions in `src/differentiators.py`); new ideas and features. Their plans are in Git history (commit `893ece8`).

## Now

1. **Cut and condense ML_Fit, ML_Rank and Evaluate** (a keep / cut / condense table → agree → edit → File: Revert File → rerun → regenerate its export). Keep every hand-off a later notebook or the app reads, and the source of every quoted number; a cut step leaves one "tested, not used" line. Small fixes go in with each notebook's review: ML_Fit's `split_people` cutoff (counted on all people, not on development people). **EDA_SDA_1 done (2026-10-17):** redesigned (80 cells; sections 1 Load, 2 Cleaning, 3 Every field, 4 Scope, 5 Factors, 6 Time, 7 Hand-off, 8 Findings; its section numbers changed, other files still cite the old ones, fix those when each notebook is reviewed), reran clean (49/49 cells, 0 errors; fixed a `KeyError` the redesign left in the Q3 ranking cell, `fields.drop(columns="scope")` with no `scope` column — it had been silently killing the cell behind the Q3 findings), opener trimmed and its three 40+-line code cells split, report regenerated. **The real source of "notebooks feel massive" turned out to be the 5-minute reports (`reports/<name>.md`), not notebook prose:** `src/export_markdown.py` now gives every report a Contents block; `ML_Rank` and `Evaluate` had no report at all and now do; `EDA_SDA_3`, `EDA_SDA_4` and `ML_Fit`'s reports no longer duplicate the ~1,900-word notebook-chain table already in `PROJECT.md` (2,500 / 4,075 / 2,047 words, down from 3,368 / 5,396 / 3,899); `EDA_SDA_4`'s report also dropped a 72-row Cox stats dump in favor of its 2-bullet verdict (the full table stays in `reports/EDA_SDA_4_full.md` and the notebook). So EDA_SDA_3, EDA_SDA_4, TimeSeries and Profiles need no further notebook edits for this — only ML_Fit, ML_Rank and Evaluate still get a real keep/cut/condense pass. **EDA_SDA_2 (2026-10-17):** a separate readability pass, same as EDA_SDA_1's — shorter title, "(you are here)" dropped (bold row instead), its notebook-chain table swapped for `PROJECT.md`'s wording (it was bespoke and missing Profiles and Evaluate), and 7 of its ~25 dead `EDA_SDA_1 <old section>` citations fixed (the ones confidently remappable to EDA_SDA_1's current 1–8 structure: recruiters → 5f, company volume bands → 5c twice, full weeks → 6a, and two unmappable "Q3b"/"Q3e" citations dropped rather than guessed). **The other ~18 EDA_SDA_1 citations in EDA_SDA_2** (old numbers like 6.3, 6.5, 8.0, 8b, 8c, 10a, "section 11", some load-bearing in code, e.g. c3's `FileNotFoundError` message) are still stale and need real remapping to the current structure — due at EDA_SDA_2's own review, not fixed ad hoc. A real bug also turned up and is fixed: `4e`'s feature dictionary tried to regex/AST-parse a cell comment in EDA_SDA_1.ipynb the redesign deleted (`StopIteration` on any fresh run); now imports `FEATURES` from `src/feature_dictionary.py` directly. EDA_SDA_2 reran clean (82/82, 0 errors) and its exports are regenerated. Next: ML_Fit's keep/cut/condense table.
2. **Keep the code lightweight,** after the reviews: one pass over `src/` and the notebooks' code. No helper defined in two places (notebooks import `src/eda_helpers.py` instead of copying it), no dead code or unused imports, short cells.
3. **Rerun Evaluate on combined model v2,** then update the app's number in `SUMMARY.md` (76.4% is v1's).
4. **"Why this listing" in the app:** pass `breakdown` through in `app/backend/api.py` and add a compact block to the job detail view: the four factors with their share of the score, and the role with its top reason. Touches Jesse's app: agree it with him first.
5. **Check the docs after the notebooks:** every figure `SUMMARY.md`, `PROJECT.md` and `../DIFFERENTIATORS.md` quote still exists, and condense `DIFFERENTIATORS.md`. During each notebook's review, also repoint its mentions of `STANDARDS.md` (deleted; its rules are in the outline, How we keep it honest) and `analysis/DIFFERENTIATORS.md` (now at the root).
6. **Any time:** delete `data/flat/all_fields_lean_full/` (1.1 GB; nothing reads it) and the sample-era tables in `data/clean/` (`analysis_table.pkl`, `scope_table.pkl`, `job_openings_lean_200000_v2.pkl`).

## Pitch

- One slide with Evaluate's comparison chart: baseline → text match (`rrf`) → plus the queue (`mlrank`) → plus role fit (combined), on the same 563 career changers.
- Demo the combined model only (`combined_demo/`), rehearsed on the laptop that presents.

## Blocked

- **Team:** confirm the final due date and deliverables.
- **Project advisor:** AI tools and data, hosting, presenting results, data deletion; get the AI-and-data answers in writing and record them in your change log.
- **PredictLeads:** the data questions in `PROJECT.md` (Questions for PredictLeads).

## Later

- **Clean up `src/`, once every notebook has its Lean rewrite** (`docs/lean_conventions.md`): delete what's now dead rather than relocate it - the four categorical helpers in `rows_table.py` (`cat_contains`/`cat_map`/`relabel`/`with_missing`, orphaned once `EDA_SDA_1.ipynb` is retired), `src/eda_helpers.py`'s formatting/chart/survival helpers once nothing imports them, and anything else a notebook's own Lean rewrite stops using. Related to "Keep the code lightweight" above, but gated on the Lean rewrite actually finishing, not the keep/cut/condense reviews - do this pass after, not instead of, that one.
- **Profiles is frozen** (refrozen 2026-10-10); reopen only if needed, then rerun it and reload `modeling_table.pkl` downstream.
- **Volunteer applicant profiles** (`docs/volunteer_profiles.md`), once the advisor answers.
- **Known gaps in `.claude/settings.json`:** `Read(./.env.*)` also blocks `.env.example`; `Read` rules don't stop shell commands (the guard in `safety/` covers those).

## Always

- Clear notebook outputs, including every `data-rows` cell, before committing.
- After a notebook is edited on disk: **File: Revert File**, then run.
- After any change to `src/`, restart the kernel before rerunning.

## Finish

- **Permanently delete every copy of the data:** `data/` (including `data/handoffs/`), `reports/`, the GPU PC's data folder, `~/dev/predictleads-hackathon/data/`, downloads (including `~/Downloads/1_corpus_and_signals.zip`), wherever the hand-off zip was shared, and any S3 copies. Confirm the deletion in writing if asked.
