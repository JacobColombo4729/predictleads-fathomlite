# EDA_SDA_1_Lean — presentation notes

## Purpose

---

## Section 1 — Load & build

- PredictLeads' full job-openings export: 13.4M listings, March 30–September 30
- Three stages: peek at structure on a sample → flatten → build the full tables
- `BUILD_MODE`: bulk (one parallel pass, needs RAM) vs. stream (three light passes, slower) — a hardware tradeoff, not a correctness one

---

## Section 2 — Cleaning, the shape of it

- Full-file fixes only: nothing dropped except genuine data errors (bad dates, duplicate ids)
- Five pieces: load + per-row extras → standardize labels → dates/duplicates/reposts → salary → derived flags + crawl artifacts
- Every action lands in one `report` list — so the notebook can show *exactly* what changed, in order

---

## The one decision worth defending: one-hot vs. bit flags

- A listing can carry several contract types / categories at once — needs a multi-label encoding
- Original: packs them into one 64-bit integer, one bit per label — memory-cheap, but reading it back means bit-shift math
- Lean: one boolean column per label instead (`contract_full_time`, `category_engineering`, ...)
- Checked the cost first: ~15 labels × 13.4M rows of booleans is well under a gigabyte — same ballpark, plain columns
- Added as a new function (`label_onehot`) next to the original in `rows_table.py`; didn't touch the bit version — other notebooks still read it

---

## Dropping the "clever" categorical helpers

- `relabel` / `cat_contains` / `cat_map`: applied a function once per *category*, not once per row — the stated reason they existed
- Tested the claim before trusting it: ran plain `.str.contains()` / `.map()` on a 13.4M-row synthetic categorical column
- Result: pandas already does the same per-category trick automatically — same speed, no custom function needed
- So: deleted from the notebook's usage, not touched in the shared file (other notebooks still use them)

---

## Cleaning, the parts worth highlighting

- **Dates:** drop impossible timelines (ends before it starts, etc.); blank implausible employer-posted dates, keep the listing
- **Duplicates & reposts:** a fingerprint per listing (company + title + description) — about a 1-in-200,000 chance of a false match across 13M rows
- **Salary:** fix reversed ranges, catch mislabeled units ("$25 a year" is really hourly pay), convert everything to one hourly rate, blank the outliers
- **Crawl artifacts:** `batch_first_seen` / `crawl_dropout` — flagged and kept, never dropped, because they distort individual companies more than totals

---

## Section 3 intro — why Kaplan-Meier

- The question: how fast do listings close, compared across occupation fields?
- The problem: 22.3% of listings were still open when data collection stopped — not a tail case, a quarter of the data
- And it's uneven: a listing posted in late September has barely had time to close
- Any "typical time to close" built only from already-closed listings reads short — worst for the most recent postings

---

## Kaplan-Meier, the 5-minute version

- Estimates the probability something *hasn't* happened yet, as a function of time
- It's a decreasing curve — "decay" is fair — but there's no single rate; nonparametric, shape comes straight from the data
- Mechanics in one line: at each day something closed, take the share of at-risk listings that *didn't* close that day, multiply those shares together in order
- "Censored" here = still open on September 30 — not a data problem, just where the clock stopped

---

## The two wrong shortcuts (and the sneaky third one)

- Drop censored listings, average the rest → biased short: the slow closers haven't closed yet, so they're excluded
- "Fix" that by assuming censored listings close at that average → still wrong, ignores how long each one's *already* been open
- Treat them as closed today, or as never closing → both are guesses in the other direction
- Kaplan-Meier's actual fix: use each listing for exactly as long as it was actually observed, no more, no less

---

## Why this isn't a one-off technique

- `TimeSeries` runs Kaplan-Meier by month (each month's cohort has a different amount of follow-up)
- `EDA_SDA_4` fits a Cox model — a related survival method, same underlying problem
- `ML_Rank`'s priority queue trains an actual survival forest for its "chance still open" factor
- None of these work without treating closed and still-open listings together

---

## Where this stands right now

- Done: sections 1–2 fully rewritten, section 3's intro and the Kaplan-Meier rationale drafted
- Verified, not just argued: every "can we drop the clever version" claim checked against synthetic data first (one-hot, plain pandas, and the Kaplan-Meier math itself all matched the original exactly)
- Not done: section 3's actual field-comparison code, the chart-style constants, a real run on the full file
- One open methodology question, separate from the rewrite: should `crawl_dropout` listings be treated as censored instead of as a real close — not yet decided

---

## If they only remember one thing

- [your words here] — e.g. "every shortcut I took, I checked against the original first, on fake data, before trusting it"
