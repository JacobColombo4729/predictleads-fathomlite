# Analysis

> What's left, top first. Delete a task when it's done. The app has its own queue: [`../app/APP.md`](../app/APP.md). **[agent]** = Jacob's Foreman may draft it (README step 14).

**Scope frozen for the pitch (deadline 2026-10-29): nothing new is built before the pitch.** Cut for now: combined model steps 6 (the trained layer driving the Compass) and 8 (fine-tuning BGE); growth beyond the slider at weight 0 (BLS); ML_Rank sections 9, 10 and 12, H7 and 7f's extra baselines; differentiation ideas 1–7 (1, 3 and 5 stay as tested functions in `src/differentiators.py`); new ideas and features. Their plans are in Git history (commit `893ece8`).

## Now

1. **Confirm the seven figures `EDA_SDA_1_Lean`'s text quotes but no output contains** — 16.8% (crawl artifacts), 12.3% (remote contract label), 0.69% (the two added TripleTen roles), 16,590 / 8,758 / 4,006 (the three dropped occupations), 28.7% (closes within a week) and 74.5% (closed by September 30). These are the notebook's own results and the pitch quotes several, so Jacob has to be able to point at the cell behind each one. They cannot be checked by text search: the cells print shares as fractions (4c-1 prints 0.071749 where the prose says 7.2%), so `python -m src.notebook_tools stale notebooks/EDA_SDA_1_Lean.ipynb` flags them whether they are right or wrong. Look at 4c-1, 5b and section 6 against the current outputs; it needs no rerun. The notebook itself is **done**: it ran all 71 cells in a fresh kernel in order on 2026-10-25 (14.2 min, no errors) and reproduced the earlier piecemeal run exactly - all 42 output-producing cells byte-identical - so it is the one Jacob presents, with the swap for `EDA_SDA_1.ipynb` still held until after the pitch (see Later). The same `stale` check is worth one pass over `EDA_SDA_2_Lean` before the pitch for the same reason.
2. **Decide which combined-model version the app ships, and what the pitch claims about the rankers.** Evaluate reran on v2 (2026-10-23): it scores **68.2%** [64.5%, 72.1%] at top 3 against v1's 76.4%, and **no hypothesis holds** — `mlrank` 69.4%, `rrf` 67.9%, combined 68.2% all sit within 1.5 points of each other, and at top 5 none beats the 87.0% baseline. `ML_Fit`'s trained layer (81.7%) and `ML_Rank`'s backtest (0.500) are unchanged. **Why:** v2 blends the trained layer's role choice with the role mix of the person's text matches (to stop a real résumé leading with the wrong role) and puts a stated desired role first — and this answer key can see neither fix, because Profiles' backgrounds *are* the trained layer's own inputs and carry no desired role. So the number argues for v1 by construction, and v2's advantage currently rests on four résumé checks (`combined/tests/test_roles_from_profile.py`), not on held-out people. **Needs a call with Jesse:** which version the demo runs, and whether the pitch keeps a ranker-comparison slide at all. **Jesse's docs still quote v1** in five places (`app/APP.md` line 45; `app/docs/ml-inventory.md` lines 9, 24, 25, 83, including the "if challenged" script) — his files, so flagged rather than edited.
3. **Check the docs after the notebooks** (in progress, 2026-10-23): every figure `SUMMARY.md`, `PROJECT.md` and `../DIFFERENTIATORS.md` quote still matches the notebook that now produces it — all three of ML_Fit, ML_Rank and Evaluate reran on 2026-10-22/23, and Evaluate's ranker numbers changed with combined v2, so this is the pass that catches anything left over. Then condense `DIFFERENTIATORS.md`. The notebooks' own `STANDARDS.md` citations are already repointed to `PROJECT.md` (all three passes); what's left is the docs.
4. **"Why this listing" in the app:** pass `breakdown` through in `app/backend/api.py` and add a compact block to the job detail view: the four factors with their share of the score, and the role with its top reason. Touches Jesse's app: agree it with him first.
5. **Remap EDA_SDA_2's ~18 dead `EDA_SDA_1` citations** to EDA_SDA_1's current 1–8 structure (old numbers like 6.3, 6.5, 8.0, 8b, 8c, 10a and "section 11"; some are load-bearing in code, e.g. c3's `FileNotFoundError` message). 7 of ~25 were remapped on 2026-10-17 and the rest were left for "EDA_SDA_2's own review" — which the keep/cut/condense pass turned out not to include, so they need their own slot rather than being lost with that finished task. A grader reading EDA_SDA_2 hits them. Rerun EDA_SDA_2 if any load-bearing one changes.
6. **Review ML_Fit's fast pass line by line, before final submission** (not before 2026-10-29's pitch rehearsal, but before the work is actually submitted): the keep/cut/condense edit, the `split_people` fix, and the rerun were done in one pass without Jacob's line-by-line sign-off, on his request, to save time now. Confirm nothing was cut that a later notebook or the app actually reads, and that every quoted number still has its source.
7. **Keep the code lightweight,** after the reviews: one pass over `src/` and the notebooks' code. No helper defined in two places (notebooks import `src/eda_helpers.py` instead of copying it), no dead code or unused imports, short cells.
8. **Any time:** delete `data/flat/all_fields_lean_full/` (1.1 GB; nothing reads it) and the sample-era tables in `data/clean/` (`analysis_table.pkl`, `scope_table.pkl`, `job_openings_lean_200000_v2.pkl`).

## Pitch

- **The comparison slide needs rethinking** ("Decide which combined-model version the app ships", in the queue above): on v2 the ladder is flat — `rrf` 67.9%, `mlrank` 69.4%, combined 68.2%, baseline 63.2%, none of them separated — so "baseline → text match → plus the queue → plus role fit" can no longer be shown as a rising staircase. What the evidence carries on held-out people is the role result (81.7%, ML_Fit 6b) and the listing backtest (precision@10 0.500, ML_Rank 8e).
- Demo the combined model only (`combined_demo/`), rehearsed on the laptop that presents.

## Blocked

- **Team:** confirm the final due date and deliverables.
- **Project advisor:** AI tools and data, hosting, presenting results, data deletion; get the AI-and-data answers in writing and record them in your change log.
- **PredictLeads:** the data questions in `PROJECT.md` (Questions for PredictLeads).

## Later

- **Batch/dropout detection: replace the fixed 5-listings/60%-share rule with a company-size-adjusted (shuffled-null) test, and treat `crawl_dropout` as censored, not a real close, in Kaplan-Meier.** Checked on synthetic data, not yet implemented: the current rule's false-positive rate is already ~0% at every company size (5-400), so that's not the problem - but it misses real one-day clustering entirely below 60% concentration (0% detection at 15-50%, where a company-size-adjusted null test catches it with full power and holds false positives near 1%). Separately, treating `crawl_dropout` listings as a real close (current behavior) understates median days-open by about 4% and overstates "closed within 7 days" by about 2 points at the project's actual ~1-3% dropout rate, against a known-true synthetic curve; treating them as censored (`finished = ~jobs.live & ~jobs.crawl_dropout` instead of `~jobs.live`) recovers the true numbers exactly. Both are small, well-evidenced fixes that would change real `EDA_SDA_1`/`EDA_SDA_2` numbers, so held until after the 2026-10-29 pitch rather than built now.
- **Retire `EDA_SDA_1.ipynb` after the pitch, once `EDA_SDA_2` runs clean on the Lean table.** `EDA_SDA_1_Lean` is finished and verified - it reran clean top to bottom on 2026-10-25, reproducing itself exactly (2026-10-24; 7c: same 908,291 rows, only the packed `category_bits` and `contract_bits` differ), but everything downstream was built on the original's `data/clean/analysis_table.parquet`, so until then the original stays the data source and the Lean notebook is the one presented. To retire it: rerun `EDA_SDA_2` on `data/clean/lean/analysis_table.parquet` and check it matches; move the hand-off to `data/clean/` (`LEAN_DIR` in 2a-1); delete `EDA_SDA_1.ipynb`, `docs/EDA_SDA_1.md` and `reports/EDA_SDA_1*`; drop `EDA_SDA_1`'s layout from `src/export_markdown.py`. The Lean exports (`docs/EDA_SDA_1_Lean.md`, `reports/EDA_SDA_1_Lean*.md`) already exist. No date is set.
- **`EDA_SDA_2_Lean` is narrowed to "which in-scope listings do we keep, tag or hide?" (decided 2026-10-24), so `EDA_SDA_2` cannot be retired until the rest has a Lean home.** Its product is `scope_flags.parquet` in `data/clean/lean/` (a row per listing keyed by `id`: role bucket, level, and one column per hide or tag rule), which joins to `EDA_SDA_1_Lean`'s in-scope table. Left out of it: how listings behave (source, freshness, pay disclosure, volume and reposts, missing values, outliers; answered in `EDA_SDA_1_Lean` 5 and 6 and tested with the confounders held fixed in `EDA_SDA_4`) and the feature screening. Not built there, so `EDA_SDA_4_Lean` has to: `copy_type` and the copy flags, `on_employer_domain`, the company facts (`is_public`, `has_parent`, `n_subsidiaries`, `company_country`), `has_posted`, the 21 company-history, market and pay features of 3.13a, and `feature_screening.pkl` (EDA_SDA_4 forwards the features whose verdict starts with "keep"). `PROJECT.md`'s notebooks table still gives the old question for `EDA_SDA_2`; change it when the swap happens. The faithful port that reproduced the original is commit `df93cc7`.
- **`EDA_SDA_3_Lean` is `EDA_SDA_3` wired to the Lean chain (2026-10-25), and has not been run yet.** It is the
  original cell for cell; what changed is where it reads and writes, and four columns. **Reads**
  `data/clean/lean/analysis_table.parquet` (`EDA_SDA_1_Lean`) joined to `scope_flags.parquet` (`EDA_SDA_2_Lean`) by
  `id`, instead of `EDA_SDA_2`'s single scope table, with 1b taking each column from whichever table has it and
  printing anything the chain no longer carries. **Writes** `data/clean/lean/text_table.parquet`, and its caches
  (`eda3_text_features`, `eda3_desc_clean`, `eda3_lemmas_sample`, `eda3_skills`, `eda3_validation/`) go to
  `data/clean/lean/` too, so no file from a run of the original can be picked up. **The four columns:**
  `volume_band` is rebuilt in 1b from `company_earlier` with `EDA_SDA_2` 3.3's bands (`EDA_SDA_2_Lean` does not
  carry it, and `company_activity.parquet` must not be used for it - that count is a total and would leak the
  future into 4b); `stale` is `EDA_SDA_2_Lean`'s `open_90_days`; `hidden_occupation` is gone because
  `EDA_SDA_1_Lean` drops those occupations as rows; `onet_code` comes from `EDA_SDA_1_Lean` (6d-6f skip themselves
  if it is missing). The scope is 878,937 listings against the original's 908,291, so **every number moves**: the
  findings cells and the conclusion were emptied to the questions they answer, and the original's wording is in
  `notebooks/EDA_SDA_3.ipynb` and git if any is wanted back. **Left:** run it, write the findings and the
  conclusion from that run, then export (`src/export_markdown.py` has no `EDA_SDA_3_Lean` layout yet). Retiring
  `EDA_SDA_3.ipynb` needs the same steps as `EDA_SDA_1`'s above, plus pointing `EDA_SDA_4` at the Lean
  `text_table.parquet`. Note `EDA_SDA_4` reads that table whole and uses its column list to decide which
  `EDA_SDA_2` features it supersedes, so the column names have to stay as they are.

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
