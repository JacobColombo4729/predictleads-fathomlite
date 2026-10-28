# Lean notebook conventions

Standing rules for every `_Lean` rewrite (`EDA_SDA_1_Lean` first; the plan is one for every notebook in the
chain: `EDA_SDA_2`-`4`, `TimeSeries`, `Profiles`, `ML_Fit`, `ML_Rank`, `Evaluate` - though where a notebook already reads Lean its `_Lean` is a replica rather than a rewrite, see the end). This is on the pitch's
critical path: Jacob presents these findings, so he has to understand every cell. The originals stay as the
reference, and each `_Lean` notebook only swaps in for its original once it reruns clean with the same results.
Written down once, here, so each new `_Lean` notebook applies them from the start instead of re-deriving them,
and so a rule doesn't quietly drift between notebooks. Update this file when a new situation forces a new call -
don't leave the reasoning sitting only in a commit or a conversation.

## The goal

**Every cell must be 100% defensible by Jacob, who presents it.** This is the whole point of `_Lean`, and it comes
before everything else in this file. The test is "can Jacob say what this line does, and why it is there?", not
"is this file short" and not "does it match the original's numbers".

What that means in practice:

- **A `_Lean` notebook is written from its question, not ported from the original.** The original is a reference for
  what to compute, never a script to translate. If a block of the original can't be written in the plain pandas a
  TripleTen project uses (`groupby`, `value_counts`, `merge`, `str.contains`, simple masks, `np.where`), it is cut,
  simplified or moved to its own explained `src/` file with a teaching doc (rule 4). It is never copied in.
- **Matching the original is a check, not the goal.** Where a simpler rule gives a slightly different number, the
  simpler rule wins and the difference is written down. Features the original made for the models are rebuilt in the
  notebook that needs them, in the same plain style.
- **One section at a time.** Each section is read by Jacob before it is run, and the next one isn't started until he
  can say what every cell in it does.
- **Size follows the question.** Cut analysis that doesn't answer the notebook's question, and don't repeat what an
  earlier notebook already answers. A notebook that needs 120 cells is answering too many questions.

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

   **Check the replacement's dtype, not only its values and speed, and shape the synthetic column like the real
   one.** This rule's first version cost a run: `cat_map` returns plain `float64`, but `.map()` on a categorical
   *keeps the categorical dtype* whenever every label maps to a different value - and a categorical can't be
   multiplied, so `EDA_SDA_1_Lean`'s 2d-3 died on `mid * factor` ("cannot perform the numpy op multiply") after
   2a-2d had already run. The first synthetic check missed it because its made-up labels included one the
   mapping didn't recognize, which makes the result numeric; the real `salary_unit`'s five labels map one-to-one,
   which doesn't. The fix is an explicit `.astype("float64")` with a comment saying why, pinned by
   `tests/test_lean_cat_map.py` (both directions: the cast matches `cat_map`, and the uncast version really does
   come back categorical). So: a replacement is only checked once the synthetic column has the same *shape* of
   mapping as the real one, and the comparison covers dtype as well as values.

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

9. **Advanced math earns its teaching doc by deciding something. Shrink a course method to a size Jacob can
   defend; never cut the course method itself.** Rule 4 says what to do with genuinely advanced math once it is
   staying. This rule is the step before it: decide whether it stays at all. The test is whether the method
   changes a decision the notebook actually makes. If the keep/drop calls come from a screening rule fixed
   before the run, a method that only describes the features more elaborately is evidence for a decision
   already made, and it does not earn a teaching doc, an opaque import or a reader's attention. Two things
   this rule is **not**: an excuse to drop a TripleTen course method (the templates are the baseline - a
   course method gets shrunk to two or three named, explained instances, not deleted), and a judgement about
   whether the math is correct (these are all correct; they are simply not load-bearing).

   **Worked example, decided 2026-10-25: `EDA_SDA_4`.** Its 27 cells call 34 functions from
   `src/stats_tests.py` (670 lines, 50 functions, no teaching doc) - which rule 4 calls an opaque import, and
   which makes this the most expensive Lean rewrite in the chain, not the cheapest. Its own 10a says the
   decisions come from `EDA_SDA_2`'s screening rule (3.13, fixed before that notebook ran) and that "sections
   3-9 describe the features and decide nothing". So:

   - **Kept, heavy (its teaching doc is `docs/cox_regression.md`, written 2026-10-26):** the adjusted **Cox** regression with
     its **proportional-hazards check**. It is the only test that holds every confounder at once - the
     within-group comparisons hold one at a time - and it is the rigor claim the notebook's Purpose makes out
     loud. The **joint Wald** test goes: it refines p-values that carry no information at this sample size.
   - **Kept, shrunk to course size:** hypothesis tests (section 5) keep H0/H1, alpha and effect sizes on two or
     three named tests; the 113-test battery and the Benjamini-Hochberg correction that existed only to
     survive it are cut. Correlation keeps Pearson, Spearman and Cramer's V; **VIF** goes, since
     multicollinearity matters for linear coefficients and nothing downstream fits one.
   - **Demoted to plain pandas, question kept:** competing risks (8c) becomes a share table of how listings
     ended (closed vs. likely ghost) instead of **Aalen-Johansen**; left truncation (8b) becomes a sensitivity
     check on listings whose posting date is near first sighting instead of an entry-time estimator; drift
     (9e) keeps the standardized mean difference and drops **PSI** and its thresholds.
   - **Cut outright:** 9f (**mutual information**, conditional MI within confounder cells, and the shuffled
     noise floor). It is the heaviest block in the notebook and gates nothing; the real gates are 9a
     stability over months, 9b within-company gaps and 9d the leakage audit, all plain `groupby`.
   - **Inlined:** the eight plain helpers the decisions actually rest on (`describe_by_group`, `rate_by`,
     `stratified_gap`, `effect_by_group`, `stability_over_months`, `within_company_effect`,
     `censoring_summary`, `leakage_audit`). Plumbing (`downcast`, `tests_table`, `decision_log`,
     `modeling_ready`) stays a shared call under rule 5.

   Net: 12 sections to about 9, 34 shared functions to roughly four plus plumbing, one new teaching doc
   instead of seven. **What this costs, written down so it can be answered out loud:** no cause-specific
   incidence curves, no formal left-truncation correction, no multiple-comparison correction. If a reviewer
   asks about any of the three, the answer is that the simpler check pointed the same way and the decision
   rule was fixed before the run. Not started: queued for after the 2026-10-29 pitch, which `EDA_SDA_4` is not
   cited in (`SUMMARY.md` and `../DIFFERENTIATORS.md` cite it zero times).

## Replicas: notebooks that already read Lean

A `_Lean` notebook is not always a rewrite. Where a notebook already meets the goal above - plain
pandas, nothing hidden behind a helper, no math outside the curriculum - there is nothing to write from its
question, so its `_Lean` is an exact replica of it and the work is threading, not rewriting: the `(Lean)` title,
`## Purpose`, the shared series table with its own row bolded, a rule 1 sweep, and output paths that cannot
collide with the original's.

- **`Profiles` (checked 2026-10-25).** It already met the goal: 20 code cells, 361 code lines, written in
  `value_counts`, `np.where` and `groupby`, with one `LogisticRegression` (curriculum) and one `scipy.stats`
  call, so rule 9 asks nothing of it. Its 18 `show()` calls were swept in place the same day, and `show`
  dropped from its `src/eda_helpers` import (chart setup and colours stay, as in `EDA_SDA_1_Lean`).
  `Profiles_Lean` is a replica of that swept version, threaded into the family. **The one change that is not
  cosmetic:** its three saved tables (`retrained_candidates.csv`, `modeling_table.pkl`, `pre_switch_text.pkl`)
  were moved to `data/profiles/lean/`. `modeling_table.pkl` is what `ML_Fit` trains on and `Evaluate` scores
  against, and `Profiles` is frozen (refrozen 2026-10-10), so a run of the draft must not be able to replace
  them - the same reason `EDA_SDA_1_Lean` keeps its tables in `data/clean/lean/`. The hand-off moves back only
  when the replica replaces the original. **`Profiles_Lean` was rerun 2026-10-27** (its tables go to `data/profiles/lean/`, so the frozen ones are
  untouched); its quoted figures match the frozen `Profiles` run. `Profiles` itself has not been rerun, so its
  stored outputs and `docs/Profiles.md` still show the old `show` formatting until after 2026-10-29. **Still open:**
  `src/career_switch.py` holds the actual career-switch rules (`RULES`, `find_switches`, `background_area`,
  `fingerprints`), shared with the app and so unable to move into the notebook. They are rules rather than
  advanced math, so rule 4 does not strictly apply - but they are the substance behind the retraining compass
  and the 2,813 career changers, which makes them the strongest candidate for a teaching doc of their own.
  Not decided; not before the pitch.
- **`TimeSeries` (threaded 2026-10-26, rerun 2026-10-27).** It already met the goal: 26 code cells, plain pandas,
  its time-series work in `src/ts_helpers.py` (96 lines, 11 small functions, no bit-packing or name-guessing
  formatting). Its 8 `show()`, 6 `pct()` and 3 `compact()` calls were swept to `print()` + `display()` and inline
  f-strings, and its Kaplan-Meier now comes from `src/kaplan_meier.py` (as in `EDA_SDA_1_Lean`). **The one change
  that is not cosmetic:** its three saved tables (`timeseries_daily`, `timeseries_weekly_bucket`,
  `timeseries_cadence`) go to `data/clean/lean/`, so a run cannot replace the ones `ML_Rank` reads. **Rerun
  2026-10-27 (26/26 cells, no errors) on the original chain**: it reads `EDA_SDA_2`'s scope table (908,291
  listings), and every figure reproduces `TimeSeries`' run, so the findings are the original's, checked against
  this run. **Still open:** STL, SARIMAX and the ADF/KPSS tests are outside the curriculum and have no teaching
  doc. Saved outputs show tables as raw floats (`0.111854`, `138035.0`) where the original's helpers rounded
  them; a display choice for the notebook, not decided.
- **`ML_Rank` (threaded 2026-10-26, rerun 2026-10-27).** It already met the goal and needed no rule 1 sweep (it
  used plain `print()` and `display()`); the models are in `src/rank_model.py`, and its one heavy method, the Cox
  regression, has the teaching doc `docs/cox_regression.md` (linked from 7c). **The one change that is not
  cosmetic:** the hand-off (`listing_model.joblib`, `listing_scores.parquet`, `queue_config.json`) goes to
  `data/clean/lean/ml_rank/`, because the original's folder is what `app/backend/ml_rank_ranker.py`,
  `app/backend/api.py` and `combined/` load. **Rerun 2026-10-27 (25/25 cells, no errors) on the original chain**
  (`modeling_ready`, `scope_table` and `fit_scores` from `data/clean/`, 908,291 listings); every result reproduces
  `ML_Rank`'s run, including the test-month precision@10 of 0.500, and only the timings differ. Saved outputs show
  the app's folder untouched. **Still open:** the survival forest and LambdaRank are outside the curriculum and
  have no teaching doc.

- **`ML_Fit` (threaded 2026-10-26, rerun 2026-10-27).** It already met the goal: plain pandas and scikit-learn, with the methods in `src/fit_eval.py`, `fit_model.py`, `fit_handoff.py` and `onet.py`. Its 19 `show()` calls were swept to `print()` + `display()` (rule 1), and the unused survival helpers left the import. **The one change that is not cosmetic:** all twelve writes (the four compass tables, `fit_scores`, `fit_breakdown`, `fit_inputs`, `fit_choice.json` and the `ml_fit/` hand-off) go to `data/clean/lean/`, because `ML_Rank` reads `fit_scores.parquet` from `data/clean/` and `combined/role_fit.py` loads `ml_fit/`. **Rerun 2026-10-27 (18/18 cells, no errors) on the original chain** (`data/clean/` and the frozen `Profiles` tables); every number reproduces `ML_Fit`'s run, including the trained layer's 81.7% at top 3, and only the timings differ. Saved path lines show every artifact in the Lean folder. **Still open:** the dropped formatting leaves some tables as raw numbers (8c's typical-pay table prints `1269173.0` listings and no `$`; 2a, 3a, 5a and 8a lack thousands separators), a display choice not decided; the compass thresholds are the open question with Jesse (`.eve/ASKS.md`) and are unchanged.

- **`Evaluate` (threaded 2026-10-26, rerun 2026-10-28).** It already met the goal: 9 code cells, 77 lines, plain pandas, with the work in `combined/evaluate.py`, `src/fit_eval.py` and `src/rank_model.py`. Its 6 `show()` calls were swept to `print()` + `display()`, `show` dropped from its import, and its byline separator fixed from `·` to `×`. **Nothing to redirect:** it saves nothing, it only loads the rankers the app serves and scores them. **Rerun 2026-10-28 (9/9 cells, no errors) on the original hand-offs**: through `combined/evaluate.py` it reads `data/clean/combined_v1/`, `ml_fit/`, `fit_scores.parquet`, `ml_rank/` and `data/profiles/`, not the Lean copies in `data/clean/lean/`, so every hit rate and interval reproduces `Evaluate`'s 2026-10-23 run (combined v2 68.2% at top 3, trained layer 81.7%) and only the timings differ. **Still open:** the company-clustered bootstrap (`rm.clustered_bootstrap_diff`) behind 4c's decision rule has no teaching doc, and it does decide something; 4a's timings table prints raw floats (`0.411741`), a display choice not decided.

## Open / still being decided

- `crawl_dropout` listings should be treated as censored instead of as a real close, and the one-day batch
  rule's fixed 5-listings/60%-share cutoff should become a company-size-adjusted significance test - both
  confirmed on synthetic data (the current rule misses real clustering entirely below 60%; treating dropout as
  a close biases median days-open short by ~4% at the project's real dropout rate). Decided, queued in
  `ANALYSIS.md` for after the 2026-10-29 pitch, not built now - a real methodology change, not a style one.
- Whether other notebooks' shared infrastructure (e.g. `src/ts_helpers.py` for `TimeSeries`) has its own
  version of the bit-mapping/auto-formatting issues found in `rows_table.py`/`eda_helpers.py` - not yet checked,
  since no other `_Lean` notebook has started.
