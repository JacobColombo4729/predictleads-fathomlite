# Lean notebook conventions

Standing rules for every `_Lean` rewrite (`EDA_SDA_1_Lean` first; the plan is one for every notebook in the
chain: `EDA_SDA_2`-`4`, `TimeSeries`, `Profiles`, `ML_Fit`, `ML_Rank`, `Evaluate`). This is a side project, not
on the pitch's critical path - it runs alongside the real notebooks, which keep their normal cut/condense
treatment, and each `_Lean` notebook only swaps in for its original once it reruns clean with the same results.
Written down once, here, so each new `_Lean` notebook applies them from the start instead of re-deriving them,
and so a rule doesn't quietly drift between notebooks. Update this file when a new situation forces a new call -
don't leave the reasoning sitting only in a commit or a conversation.

## The goal

Every cell should be explainable from what's in front of you - the way a TripleTen project's code is - even
where that costs more lines than the original. Not: smaller cells. Not: fewer helper calls for their own sake.
The test is "can I say what this line does," not "is this file short."

## Rules

1. **No shared formatting/summary-table helpers.** `src/eda_helpers.py`-style functions (`show`, `show_shaded`,
   `compact`, `pct`, ...) guess formatting from a column's name and hide the actual `print()`/`display()` call.
   Use plain `print()` + `display()` and inline `f"{x:.1%}"`/`f"{x:,}"` formatting instead. This is a pure
   style call - no verification needed, since it changes nothing about the numbers, only how they're shown.

2. **No "clever" per-category helpers when plain pandas already does the same thing - but check first, don't
   assume.** `cat_contains`/`cat_map` existed because they apply once per pandas category instead of once per
   row. Plain `.str.contains(pattern, na=False)` and `.map()` already do this automatically on a categorical
   column. Confirmed on synthetic data at full scale (13.4M rows) before dropping them: same results, same
   speed. Don't drop a helper like this without timing the replacement against it first - the justification for
   the original helper might turn out to still be true.

3. **A bit-packed or otherwise encoded column gets a plain, named alternative - after checking the memory
   cost, not instead of checking it.** `contract_bits`/`category_bits` pack several labels into one integer for
   memory reasons at 13.4M rows; `label_onehot()` (`src/rows_table.py`) gives one boolean column per label
   instead, after confirming the memory cost (roughly 15 labels x 13.4M rows of booleans, well under a
   gigabyte) is actually affordable. The original encoding stays in the shared file, untouched, for whatever
   else still reads it - add the plain alternative alongside it, in the same file, with its own section header,
   rather than silently replacing it or forking a new file for one function.

4. **Genuinely advanced math (not TripleTen's curriculum) gets its own teaching doc and its own small,
   explicit source file - not inlined, not left as an opaque import.** Kaplan-Meier survival estimation
   (`src/kaplan_meier.py`, `docs/kaplan_meier.md`) is the model: the doc teaches the method in general first (no
   project details, the way a textbook would), then applies it to this project second; the code is decomposed
   into small functions that each map to one term of the formula, so the doc can point at exactly one function
   per step. The existing shared-file version (`src/eda_helpers.py`'s `km_curve`/`surv_at`/`km_median`) stays
   untouched for whatever else still calls it. Checked against the original on a hand-checkable small case (with
   ties) and at a realistic scale before trusting it.

5. **Data-engineering infrastructure stays as a "given" call into a shared file - like TripleTen's
   `pd.read_csv()` - but which columns it returns, and why, needs its own visible reasoning.** Loading 13.4M
   rows from the PredictLeads export is real engineering, not something to rewrite inline. What's not
   acceptable is a bare column list with no pointer to where that list's reasoning lives. Point at the shared
   file's own documented decision (e.g. `src/listing_columns.py`'s `COLS` dict, now commented by group with a
   reason for each), or write the reasoning in the notebook directly if it doesn't exist anywhere yet.

6. **When something is deferred to a later section or a later notebook - not dropped - say so by name, in
   both the code and the surrounding markdown.** `EDA_SDA_1_Lean`'s section 2 doesn't carry `recruiter_title`,
   `salary_text`, `location_text`, `url`, `record_type` or `company_record_type` - not because they're
   unimportant (the app needs several of them), but because they're scope-only and get joined in section 4.
   Name the exact fields, name where they get picked up, and say why the deferral exists, in a markdown cell a
   reader will actually see - not only in a code comment three cells away.

7. **Every "can we simplify/drop this" claim gets checked against the original, on synthetic data, before
   being trusted - not just argued from reasoning.** This has caught real things: the categorical helpers really
   did match and run at the same speed; `relabel`'s plain replacement was ~10x slower in absolute terms that
   turned out to be 0.5 seconds, irrelevant next to everything else the notebook does. Arguing from first
   principles is how you form the hypothesis; synthetic data is how you check it before it goes in the
   notebook.

8. **The notebook's actual Purpose, questions and place in the chain carry over unchanged.** Lean rewrites how
   the code reads, not what the notebook is for. Copy the original Purpose section verbatim (just add the
   `(Lean)` marker to the title); don't restate the project's goals in your own words as part of a style
   rewrite.

## Open / still being decided

- `crawl_dropout` listings should be treated as censored instead of as a real close, and the one-day batch
  rule's fixed 5-listings/60%-share cutoff should become a company-size-adjusted significance test - both
  confirmed on synthetic data (the current rule misses real clustering entirely below 60%; treating dropout as
  a close biases median days-open short by ~4% at the project's real dropout rate). Decided, queued in
  `ANALYSIS.md` for after the 2026-10-29 pitch, not built now - a real methodology change, not a style one.
- Whether other notebooks' shared infrastructure (e.g. `src/ts_helpers.py` for `TimeSeries`) has its own
  version of the bit-mapping/auto-formatting issues found in `rows_table.py`/`eda_helpers.py` - not yet checked,
  since no other `_Lean` notebook has started.
