<!-- Generated from notebooks/EDA_SDA_1_Lean.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# EDA/SDA 1 (Lean): Field Comparison and Scope

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

Clean PredictLeads' full job-openings file (every listing first seen March 30 – September 30, 2026), compare fields, cut to the scope (**the roles TripleTen trains for, plus adjacent tech roles**) and explore it.

- **Q1 (section 3):** how do computer and mathematical roles compare with every other field?
- **Q2 (section 5):** which factors should drive prioritization for a career changer?
- **Q3 (section 6):** what time information can prioritization rely on?

Fix or flag, never drop, except data errors; aggregates only, never rows. **"Still open"** means last seen within 2 days of September 30.

**Hands off (section 7):** the in-scope table (every field) and its descriptions, the occupation summary, and company activity.

| Notebook | Question | Product |
|:--|:--|:--|
| **EDA_SDA_1** | **What's in the data, and what's the scope?** | **The scope table** |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? | Cleaned and flagged scope table (ghosts, hiding rules) |
| EDA_SDA_3 | What do the descriptions say? | Text features and O*NET skills |
| EDA_SDA_4 | Which features hold up? | `modeling_ready.parquet` (the features that survived) |
| TimeSeries | When do listings appear and close? | Timing patterns by month |
| Profiles | Who really changed into or within tech or math, and how? (frozen) | The 2,813 career changers |
| ML_Fit | Which roles does a career changer fit? | The trained layer (role fit), saved for the app |
| ML_Rank | Which listings should they apply to first? | Listing scores and the queue's weights, saved for the app |
| Evaluate | How well do the app's rankers do on real career moves? | The comparison: `rrf`, `mlrank`, combined |

Same table as [`../PROJECT.md`](../PROJECT.md) ("The notebooks"), kept in step with it.

## Contents

- [Purpose](#purpose)
  - [About this Lean draft](#about-this-lean-draft)
- [1. Load and build](#1-load-and-build)
- [2. Data cleaning](#2-data-cleaning)
  - [PredictLeads crawl artifacts: when the date is the PredictLeads crawler's, not the listing's](#predictleads-crawl-artifacts-when-the-date-is-the-predictleads-crawlers-not-the-listings)
- [3. Q1: How do computer and mathematical roles compare with every other field?](#3-q1-how-do-computer-and-mathematical-roles-compare-with-every-other-field)
  - [Why this needs Kaplan-Meier](#why-this-needs-kaplan-meier)
- [4. The scope: where the data narrows to the roles TripleTen trains for](#4-the-scope-where-the-data-narrows-to-the-roles-tripleten-trains-for)
  - [4a Why this scope](#4a-why-this-scope)
  - [4b Language and country](#4b-language-and-country)
  - [4c The scope cut](#4c-the-scope-cut)
  - [4d Fields only the scope carries](#4d-fields-only-the-scope-carries)
- [5. Q2: Which factors should drive prioritization?](#5-q2-which-factors-should-drive-prioritization)
  - [5b Application window](#5b-application-window)
  - [5c Competition: company hiring volume](#5c-competition-company-hiring-volume)
  - [Why we need the Gehan-Wilcoxon test](#why-we-need-the-gehan-wilcoxon-test)
  - [5d Source](#5d-source)
  - [5e Competition: reposts](#5e-competition-reposts)
  - [5f Recruiters](#5f-recruiters)
  - [5g Who is open to newcomers](#5g-who-is-open-to-newcomers)
  - [5h Do the flags go with how long listings stay open?](#5h-do-the-flags-go-with-how-long-listings-stay-open)
- [6. Q3: What time information can prioritization rely on?](#6-q3-what-time-information-can-prioritization-rely-on)
- [7. The in-scope table (hand-off to EDA_SDA_2)](#7-the-in-scope-table-hand-off-to-eda_sda_2)
- [8. Findings](#8-findings)
  - [Limits](#limits)

### About this Lean draft

A plain-pandas rewrite of this notebook: every cell uses only pandas/numpy/matplotlib and the project's real
data-loading code (`src/pl_data.py`, `src/rows_table.py`, `src/stream_full.py`) - no shared formatting or
summary-table helpers (`src/eda_helpers.py`). The goal is code that can be explained line by line, the way a
TripleTen project can, even where that costs more lines than the original. This will replace `EDA_SDA_1`
once it reruns clean with the same results. It is on the pitch's critical path, not a side project: Jacob
presents these findings, so he has to be able to explain every cell that produces them. The Lean rewrites of
the other notebooks in the chain wait until after the pitch (`../ANALYSIS.md`). The standing rules
behind every call made here (what gets inlined, what stays a shared-file call, how a dropped helper gets
checked before it's trusted) are written down once in `docs/lean_conventions.md`, not re-argued per notebook.

Covers all eight sections (1-8); see the closing note at the end for where it differs from `EDA_SDA_1` and what's left
before it replaces it.

## 1. Load and build

1b–1d show the raw structure on a 200,000-document sample, then flatten it (`data/flat/`). 1e builds the full file (13.4M listings) from `data/raw/` (`python -m fetch.get_predictleads`) or S3 (`src/stream_full.py`). `BUILD_MODE`: **bulk** (about 35 minutes, 16 GB of RAM), **stream** (a few hours, for an 8 GB machine) or **auto**; both build the same tables, and finished steps are skipped.

## 2. Data cleaning

Full-file fixes (13.4M listings), data quality only: nothing is removed but data errors. The cleaning report (2f) lists every action.

- **Text** becomes counts and flags in the stream (`src/listing_columns.py`). Without a description a text flag is unknown, not False.
- **Dates:** impossible timelines are removed; implausible `posted` dates are blanked, and the listing stays.
- **Reposts** are kept and flagged, with features built from earlier copies only. Duplicate ids are dropped.
- **Pay** becomes one hourly rate; reversed ranges are swapped, wrong units re-read, and values outside $10k–$1M a year blanked.
- **Derived columns:** `live` (last seen within 2 days of September 30, so still open then: its lifespan is censored), `days_open`, `company_earlier` and the career-change flags.
- **PredictLeads crawl artifacts** (batch first sightings, dropouts) are flagged and kept.

**Why listings without a description stay in scope:** about 17-18% of listings have none, and that share isn't
steady - it rises to 23-25% from June on, against 6-8% in March-April, which points at a PredictLeads collection change, not a
steady rate of employers skipping the field. These listings also close in a median of 3 days against 30 for
everything else - the same shape as a PL crawl artifact (section 2e), not obviously real behavior, and still
unverified (asked of PredictLeads directly, `PROJECT.md`). Missing pay looked predictive only because of this
group, a confound `EDA_SDA_2` already caught and corrected for. So `has_desc` is flagged, not dropped, the same
as every other text blocker in this project (years asked, degree required, clearance, ...): a label and an
opt-in filter, with the count of what it would hide always shown - not a silent cut at the data layer that would
also erase most of June onward before PredictLeads has explained why this happened.

**Not here yet, on purpose:** this section works on all 13.4M listings, before the scope cut, so a few fields the
app needs later - `recruiter_title`, `salary_text` (pay as written), `location_text`, `url`, `record_type`,
`company_record_type` - are not loaded or computed in this notebook at all. They are fetched only for the ~900K
in-scope listings, in section 4 (`rows_table.scope_fields()`, joined into the in-scope table there) - loading them for
every listing here would cost memory for the 93% that never reach the scope.

### PredictLeads crawl artifacts: when the date is the PredictLeads crawler's, not the listing's

PredictLeads doesn't watch job postings directly - it periodically re-crawls each company's job pages and
records what it currently sees there. `first_seen` is the first time the PL crawler happened to notice a listing,
not necessarily when it was actually posted; `last_seen` is the last time the PL crawler saw it, not necessarily
when it closed. Almost always these track reality closely. Two patterns don't, and both are caused by the
PL crawler, not by hiring:

- **`batch_first_seen`:** PredictLeads starts crawling a company for the first time, or the company rebuilds its
  careers page and gets re-indexed - and dozens of jobs that already existed all get a `first_seen` of that one
  day. It looks like a hiring surge; it's really the PL crawler catching up.
- **`crawl_dropout`:** PredictLeads loses track of a company's page (a PL crawler bug, a site change it can't
  parse), and a batch of that company's listings all get a `last_seen` of that one day. It looks like they all
  closed together; `days_open` for those listings may be cut artificially short.

**The one-day batch rule** detects both the same way: a company with at least 5 listings, where 60% or more of
them share one exact day for the date in question (`first_seen` for batches, `last_seen` for dropouts), gets
flagged. Those two numbers (5, 60%) are a judgment call, not derived from anything - meant to catch an obvious
pile-up, not flag every small coincidence. The data's first two days are excluded from batch detection entirely:
at the very start of the PL crawl, *every* company's existing backlog gets dated to day one, which would make every
company look batched - that's a dataset-wide startup effect, not the per-company signal this rule is for.

Both get **flagged, kept, never dropped** - they're still real listings, just ones where a timing-based measure
(how many jobs a company posted on one day, or how long a listing stayed open) might read oddly for that
company.

**Return to this later (queued in `ANALYSIS.md`, after the 2026-10-29 pitch):** checked on synthetic data, the
fixed 5-listings/60%-share rule has a real blind spot - it misses genuine one-day clustering entirely below 60%
concentration, where a company-size-adjusted (shuffled-null) significance test would catch it reliably. Separately,
treating `crawl_dropout` listings as a real close (instead of censored) measurably biases the Kaplan-Meier curve
short - modest at the project's actual dropout rate (~4% on median days-open), but real. Both are evidenced, not
just suspected; neither is fixed here because it would change real `EDA_SDA_1`/`EDA_SDA_2` numbers during a
scope freeze.

**What the data shows**

- **Cleaning removed no listings:** no duplicate ids, no impossible timelines, no reversed pay ranges and no
mislabelled pay units - every one of the 5,625,697 listings that state pay states a unit the converter
recognizes. It flagged 1,366,992 reposts (10.5%; a further 2,223,437 listings, 17.1%, have no description and so
can't be checked at all), blanked 349,499 employer posting dates (2.7%: placeholders before 2000, or dates after
the PredictLeads crawler first saw the listing) and blanked 984 pay values outside $10K-$1M a year.
- **11,650,472 original listings from 323,665 companies** feed the every-field comparison in section 3; 22.3% of
them were still open on September 30, which is why section 3 needs Kaplan-Meier rather than a plain average.
- **PredictLeads crawl artifacts are small but concentrated:** 2.4% of listings were first seen in a company's one-day batch
(16.8% of the companies with 5+ listings have one) and 1.2% disappeared in a one-day dropout (9.9% of those
companies). They distort single companies far more than the totals, so they are flagged, never dropped.
- **The same numbers as `EDA_SDA_1`,** action by action: the original prints them rounded (13.0M, 1.4M, 349K,
11.7M, 324K) and reports the same 984 blanked pay values. Section 2 takes 5.2 minutes here.

## 3. Q1: How do computer and mathematical roles compare with every other field?

Every occupation group before the scope cut, on original English listings; weekly rates use full weeks only.
**Context, not cause:** fields differ in company size, source and seniority.

### Why this needs Kaplan-Meier

**22.3% of all original listings were still open on September 30** - not a tail case, roughly a quarter of the
data - and it isn't spread evenly: a listing posted in late September has barely had time to close, so it's far
more likely to still be open than one posted in April. Any "typical time to close" or "closed within 7 days"
computed only from listings that have already closed would read short, worst for the most recently posted
listings in every field. `median days open` and `closed within N days` below use the Kaplan-Meier curve
(`src/kaplan_meier.py`, `docs/kaplan_meier.md`) for exactly this reason, computed within each occupation group
from its closed *and* still-open listings together - not just the ones that happen to still be open.

This isn't a one-off, either: `TimeSeries` runs Kaplan-Meier by month for the same reason (each month's cohort
has a different amount of follow-up); `EDA_SDA_4` fits a Cox model; `ML_Rank`'s priority queue trains an actual
survival forest (C-index 0.725) for its "chance still open" factor. None of those are possible without a method
that uses the closed and still-open listings together.

**Saved for later notebooks:** the occupation summary (pay, timing and openness per occupation; the compass's "your pay increase"), `occupation_summary.pkl`, and company activity (each company's listings in any field and newest listing; the "company still posting" ghost signal), `company_activity.parquet`. Aggregates only; nothing is shown. Like the cleaned rows, both are written to `data/clean/lean/` until this notebook replaces `EDA_SDA_1`.

## 4. The scope: where the data narrows to the roles TripleTen trains for

Before the cut, every field; after it, the scope only. Nothing is deleted: the cleaned rows stay on disk.

| Decision | Choice | Why |
|:--|:--|:--|
| **Language** | English; listings with no language label dropped | The text features and the app are English; unlabeled listings have no description (4b) |
| **Country** | Listings that name the US; no-country listings dropped | Pay is in dollars and the job-growth projections (BLS) are for the US; only these are known US openings (4b) |
| **Scope** | The roles TripleTen trains for, plus adjacent tech roles; credentialed math out | TripleTen's program catalog, a neutral definition (4a) |
| **Not a tech path** | Three occupations dropped: document management specialists (15-1299.03), GIS technologists (15-1299.02) and search marketing strategists (13-1161.01) | Records work, mapping and search marketing aren't a tech path a career changer would take. Each is found by an exact O*NET code, so the rule can't misfire, and leaving them in would only add noise to every later table (4c, 4c-4) |
| **Description** | Listings with no description are kept and flagged `has_desc`, not dropped | 17-18% have none, rising from 6-8% in March-April to 23-25% from June (which points at a PredictLeads collection change, not employers changing, and is still unverified), and they close in a median 3 days vs. 30. Dropping them would erase most of June onward and hide a confounder: missing pay looks predictive only through this group. For model fitting, every text feature is blank (not False) without a description, so `has_desc` has to sit next to them for a model to tell "no description" from "a description that doesn't mention it" (section 2; `EDA_SDA_2` 3.9, `EDA_SDA_4`, `ML_Rank`) |
| **Time** | April–September; late March kept, flagged `partial_month` | Left out of per-month counts only |

Only whole occupations that aren't a tech path are dropped here. Seniority, long-open listings and staffing agencies are tagged or hidden in `EDA_SDA_2`, never dropped.

### 4a Why this scope

- **TripleTen's program catalog** (tripleten.com, checked 2026-10-10) defines it: a neutral, outside definition. IT support, networks and systems, database and IT project management have no program but stay as **adjacent tech roles,** since IT support is the most open entry point.
- **The data supports it:** a switch pays off (section 3), 2,813 career changers in the résumé data moved into these roles (`Profiles.ipynb`), and they're hard to enter, which is where hiding and ranking help.
- **What is left out:** credentialed math (actuaries and mathematicians need a credential) and three occupations that aren't a tech path, document management specialists (15-1299.03), GIS technologists (15-1299.02) and search marketing strategists (13-1161.01). The test for an occupation: would a TripleTen graduate, or someone moving up from IT support or networking, plausibly apply? The full list of occupations in the scope is 4c-4.

### 4b Language and country

What the two rules cost: no-country listings may be US remote jobs, and unlabeled listings may be the ones with no description.

**What the data shows**

- **Listings with no country (8.0% of cleaned listings) are largely US remote jobs:** 30% mention remote work vs. 13% with a country, and 12.3% use a remote contract label vs. 7.8%. 93% come from a US-based company, 2.6% from a company elsewhere and 4.3% from an unknown one; all are dropped.
- **No language label means no description:** 623K of the 624K unlabeled listings have none, so dropping them also removes about 28% of all listings without a description (623K of 2.2M). 95% of cleaned listings are English, and no listing outside the US remains after the stream.
- **A correction to `EDA_SDA_1`:** it reports the remote contract label at 0.83% vs. 0.51%. That came from the packed `contract_bits` column, which holds only the first 63 labels *alphabetically* of 161, and "remote" falls after them. The plain one-hot columns here keep every label, so the figures above replace those.

### 4c The scope cut

In the scope: occupation group *Computer and Mathematical* or an O*NET code starting 15-, or one of two TripleTen roles outside the group (sales engineers, market research analysts); minus credentialed math (actuaries, mathematicians) and three occupations that aren't a tech path (document management specialists (15-1299.03), GIS technologists (15-1299.02) and search marketing strategists (13-1161.01)). Each listing gets a **role bucket** and a **TripleTen program** (`src/career_switch.py`), or "adjacent tech role" where no program trains for it.

**Which occupations are in the scope?** The scope is the TripleTen roles plus adjacent tech roles, found from O*NET codes, so it can include an occupation that is not a tech path a career changer would take. The table below lists every occupation in the scope with its code, its name and its size. The test for each: would a TripleTen graduate, or someone moving from IT support or networking, plausibly apply? An occupation that fails it is dropped in 4c-1, next to credentialed math, because the scope's definition lives here.

**What the data shows**

- **The scope is 878,937 listings, 6.5% of the file:** computer and mathematical is 7.2% of English US listings, the two added TripleTen roles add 0.69%, and credentialed math (0.05%) is set aside. The O*NET-code fallback adds nothing (the group is always present), and every scope code maps to a role bucket.
- **Three occupations that aren't a tech path are dropped** (29,354 listings, 3.2% of the TripleTen and adjacent roles): search marketing strategists (16,590), GIS technologists (8,758) and document management specialists (4,006). 4c-4 lists the 37 occupations that remain.
- **Software development is the largest bucket (29.8%),** then networks and systems (19.1%), IT project management (12.9%) and data science (11.9%); technical sales is the smallest named bucket (1.1%), and the catch-all "other computer roles" is 0.4%.
- **By TripleTen program:** AI Software Engineering 28.6%, Data Analytics 13.1%, Cybersecurity 5.1%, AI & Machine Learning 4.9%, Quality Assurance 3.8%, UX/UI Design 1.2%, Forward Deployed Engineering / AI Automation 1.1%; **42.2% are adjacent tech roles.**

### 4d Fields only the scope carries

From the stream's full records: the **full description** (`scope_descriptions.parquet`, for EDA_SDA_3), an **anonymous recruiter key** and title (names and contacts aren't kept), **pay as written** (parsed into hourly pay where missing), **location as written** and the **url**. These exist only for the scope's listings (see "Not here yet, on purpose" in section 2).

**What the data shows**

- **Every in-scope listing was found in the stream's full records;** the descriptions are saved for EDA_SDA_3 (2.2 GB of text).
- **Pay as written exists for 48.8%,** and parsing it added hourly pay for 1.5% of listings (46.7% to 48.2%). Every listing carries a location as written.

**What the data shows**

- **The in-scope table is 878,937 listings from 68,863 companies.** Timing, role bucket, source and seniority are complete; description 82%, pay 48%, years stated 46%, employer posting date 42%.
- **Contract types are messy** (free-text labels; full time 55.0%, remote 21.9%, hybrid 20.6%, onsite 19.8%, contract 8.7%, none 17.6%). Every scope company is in the companies table.
- **A correction to `EDA_SDA_1`:** its text said "none 27%" and left out remote, for the same reason as in 4b (the packed column kept only 63 of 161 labels). Its description, pay, years-stated and posting-date shares (78%, 45%, 43%, 39%) were also out of step with its own printed tables (82%, 48%, 46%, 42%); the figures above match this notebook's run.

## 5. Q2: Which factors should drive prioritization?

**In-scope listings only** (originals, `so`; repost rates use every listing). A factor only helps ranking if it **varies**: 5a shows the main factors by role bucket and program, with each one's spread (max / min); 5b–5h look closer at one family each.

**What the data shows**

- **Volume and manager-level seniority vary most between buckets** (27 and 25 times; the range leaves out the 3.8K-listing catch-all "other computer roles"): technical sales gets 340 new listings a week and software development 9,200; manager-level is 1.7% in IT support and 43.7% in IT project management. Of the accessibility measures, entry-level seniority varies 10.8 times and open to career changers 4.8.
- **Pay varies 2.4 times** ($31.10 an hour in IT support to $73.54 in software development), **while timing varies only 1.7** (medians of 22 to 38 days).
- **By program:** Data Analytics roles are the most open (28.9% show any sign) at $49.28 an hour; AI Software Engineering and AI & Machine Learning pay the most (about $74) but are among the least open (18 to 19%).

### 5b Application window

Share still open N days after first seen (Kaplan–Meier, so still-open listings count as unfinished): the basis for an **apply-by estimate**.

**What the data shows**

- **The window is short:** 28.7% of in-scope listings close within a week, the median is 24.1 days, and 44.8% are still open after 30 days.
- **Most buckets move alike** (67 to 73% still open after a week). **Technical sales is the slow outlier** (84% after a week, median 38 days); IT support, database, quality assurance and other computer roles are the fastest (median 22 days).
- **PredictLeads crawl artifacts barely move timing:** without them the median falls from 24.1 to 23.0 days, and the share open after a week stays at 71%.

### 5c Competition: company hiring volume

Volume is the company's listings **in any field first seen before each listing** (`company_earlier`), as the model will see it.

### Why we need the Gehan-Wilcoxon test

The table above shows medians for four groups of companies, from 45 days at the lowest-volume companies down to 12 at the highest. A median alone can't say whether gaps that size could just be chance, and a plain test on days open would throw away the still-open listings (the same mistake section 3 explains). So the next two cells use a **Gehan-Wilcoxon test**, the survival version of a Wilcoxon test: it compares Kaplan-Meier curves, giving the most weight to early days, when most listings are still being watched and most of them close.

It is run two ways. **All four bands at once** asks whether any of the four curves differ from the others, which is what the chart shows. **Every pair of bands** (six pairs) asks whether each step is real, and not just the two ends. First the three neighboring steps: 1-4 vs. 5-49, 5-49 vs. 50-499 and 50-499 vs. 500+. Then the pairs two steps apart: 1-4 vs. 50-499 and 5-49 vs. 500+. Last, the two ends: 1-4 vs. 500+.

How to read the results: a **p-value** is the chance of seeing a gap this large if the groups really closed at the same rate. A tiny p-value means the gap is real, **not that it is big**; the size is in the medians (45, 34, 28 and 12 days). It also doesn't say company volume *causes* faster closing, since big companies differ in other ways; `EDA_SDA_4`'s Cox model holds those fixed. The method and a worked example are in `docs/gehan_wilcoxon.md`.

**How to read the chi-square column.** Each chi-square measures how far two bands' closing patterns are from what we'd expect if they closed at the same rate: **0 means identical curves, and bigger means a bigger gap.** It is a score of evidence, not a size: it has no units, and it grows with the number of listings, so with hundreds of thousands per band it comes out in the thousands.

To turn a chi-square into a verdict, compare it with a cutoff. For a single pair (1 degree of freedom), above **3.84** means the gap is unlikely to be chance at the usual 5% level (p < 0.05), and above **10.83** means p < 0.001. For the four-band test in the chart title (3 degrees of freedom) the cutoffs are 7.81 and 16.27. Our values are far above all of these, so every p-value reads "< 0.001": the steps between bands are real, not luck. Use the chi-square to compare rows (a pair at 20,000 differs more strongly than one at 2,000), and use the medians (45, 34, 28, 12 days) for how big the differences are.

**What the data shows**

- **Company volume is the strongest timing signal:** counting only earlier listings in any field, in-scope listings at companies with 500+ listings close in a median **12 days, vs. 45 at companies with 1 to 4**. The four bands differ (Gehan-Wilcoxon test across all four; the chi-square and p are in the chart title), and the pairwise table shows each step between bands separately.
- **The gradient is steady across bands** (45, 34, 28 and 12 days) and holds for listings first seen from May (56 vs. 12 days).
- **High-volume employers dominate:** the top 10% of companies post 47 to 74% of each bucket's listings, and reposting rises with volume (2.4% at 1 to 4 listings to 8.7% at 500+).

### 5d Source

Hiring platform, job board or company site: share, arrivals and window. Job boards are under-represented in the PredictLeads crawl.

**What the data shows**

- **62.7% of in-scope listings are on company sites or other domains,** 29.6% on hiring platforms and 7.8% on job boards.
- **Windows differ widely by source:** LinkedIn, Oracle and Taleo listings close in a median 14.5 to 15.7 days, company sites 21, Workday 32, Greenhouse 49, Lever 67 and Ashby 72.
- **The mix is similar in every bucket** (company sites 53 to 66%), so source separates listings, not roles. Platform choice tracks employer size, so source is tested alongside company volume.
- **A correction to `EDA_SDA_1`:** its written shares (65%, 28%, 7.1%, and 56 to 70% across buckets) don't match its own printed tables (62%, 30%, 7.9%); the figures above match both this notebook's run and the original's tables.

### 5e Competition: reposts

A repost is the same job listed again under a new id: how common, by bucket and source, and how soon after the original.

**What the data shows**

- **6.5% of in-scope listings are reposts:** most in technical sales (9.4%) and business and data analysis (9.4%), fewest in quality assurance (4.6%) and IT support (4.9%).
- **Job boards repost far more than other sources** (24.3% of their listings, vs. 4.1% on hiring platforms and 4.9% on company sites).
- **Reposts come back fast:** a median 7.6 days after the original, and a quarter within a day, so many are re-listings or PredictLeads crawler refreshes rather than a failed hire.

### 5f Recruiters

Anonymous keys only; counts and shares.

**What the data shows**

- **Only 1.3% of in-scope listings name a recruiter:** 5,139 anonymous recruiters; the 432 with 5+ listings post 38% of recruiter listings.
- **Recruiter listings close much faster:** a median 11.2 days vs. 24.4 without one; for recruiters with 5+ listings, **6.5 days** (48% still open after a week vs. 71%).
- **Many recruiter listings may come from staffing agencies,** so EDA_SDA_2 tests recruiters against its staffing-agency flag before they're used.

### 5g Who is open to newcomers

The text rules behind the career-change flags, among listings with a description.

**What the data shows**

- **IT support is the entry door:** 48% of its listings that state years ask for 2 or fewer, and 22% are entry level (5a). Then business and data analysis (33% show any sign). The small catch-all "other computer roles" (3.8K listings) shows the most openness (65%) but is too small and mixed to read as a door. IT project management is the hardest (2.1% entry level, 9.1% open).
- **Open listings pay less, but still well:** a median $38.46 an hour, vs. $64.90 across the scope ($26.00 for open IT support listings). Degree requirements are rare (3.9%), but 61.3% of listings that state years ask for 5 or more.

### 5h Do the flags go with how long listings stay open?

Each row compares listings with and without a flag, among those where it's known. Keyword flags misfire, so EDA_SDA_3 checks them; urgency is left to its NLP.

**What the data shows**

- **Disclosed pay and long descriptions go with listings staying open longer** (79% and 81% still open after a week, vs. 64% and 68% without). **Remote wording shows no gap** (78% vs. 79%).
- **Listings open to career changers close a bit faster** (75% still open after a week vs. 79%).
- **One feature at a time:** these overlap with company volume and source, so EDA_SDA_4's Cox model separates them.

## 6. Q3: What time information can prioritization rely on?

Six months of data rules out seasons. Checks: history and weekly arrivals (full weeks only), whether `first_seen` follows the employer or the PredictLeads crawler, which listings count as still open, and whether the date picker is fair.

**What the data shows**

- **Six months of steady history:** no seasonal advice (7 distinct months, no complete year), but 74.5% of in-scope listings closed by September 30 (25.5% were still open), so timing is well observed. A median **30,303 new in-scope listings arrive per full week**; the quietest week is 0.76x the median and the busiest 1.27x.
- **`first_seen` is a good freshness clock:** it lags the employer's posting date by a median 1.4 days (75% within 3.4). Its weekdays follow the PredictLeads crawler (Friday peak, 19.8%), while posting dates (41% of listings) peak Tuesday to Thursday.
- **Still-open rules (merge decision 2):** our rule (last seen within 2 days) counts 25.5% of in-scope listings as open, and the app's signal rule (7 days, not closed) 25.4%, mostly the same listings (94% overlap). **The app's current listing filter (status not closed) counts 39.6%, and 26.2% of those were last seen 30+ days before September 30.**
- **The date picker is fair from July:** live listings reach 80% of September's level, and the share first seen in the past 30 days sits near 50% in July and August (50.1% and 48.3%) and 43.5% in September.

## 7. The in-scope table (hand-off to EDA_SDA_2)

`data/clean/lean/analysis_table.parquet` (it moves to `data/clean/` when this notebook replaces `EDA_SDA_1`), with the full descriptions in `scope_descriptions.parquet` (joined by `id`); the dictionary lists every column.

## 8. Findings

| | |
|:--|:--|
| **Data** | 13,415,388 listings in the file; 13,017,464 English (or unlabeled) and US (or no country); first seen March 30 to September 30, 2026 (the data's end date) |
| **After cleaning** | 13.0M listings, none removed; 1.4M reposts flagged (10.5%); 11.7M originals from 324K companies; 22.3% still open on September 30 |
| **In-scope table** | 879K listings (822K originals) from 69K companies in 11 role buckets (3 occupations that aren't a tech path dropped, 29K listings); 25.5% still open on September 30; 58% under a TripleTen program, 42% adjacent tech roles |

**Q1, how tech compares:** computer and mathematical is the best-paid (1st of 23) and most remote (1st) field, 4th in volume, but 22nd on entry-level roles and openness, and last on asking 2 years or fewer. The scope's design follows from it: hide what a newcomer can't get, rank the rest, and point to the open doors (IT support, business and data analysis, Data Analytics roles).

**Q2, which factors drive prioritization** (inside the scope)

- **Company volume is the strongest timing signal:** listings close in a median 12 days at the highest-volume companies (500+ earlier listings) vs. 45 at the lowest-volume (1 to 4), with the bands in between stepping down steadily (45, 34, 28, 12); every step is a real difference (Gehan-Wilcoxon test, 5c).
- **The window is short,** and it's what the model predicts: 29% of listings are gone within a week, median 24 days (5b).
- **Source and recruiters matter:** 15 days on LinkedIn or Taleo vs. 72 on Ashby; 11 days with a named recruiter, though only 1.3% of listings name one (5d, 5f).
- **Pay and openness vary most between roles:** $31 to $74 an hour, and openness from 9% (IT project management) to 44% (IT support) (5a, 5g).
- **Weak, or flags only:** remote wording shows no gap, reposts are mostly re-listings, and PredictLeads crawl artifacts barely move timing (5b, 5e, 5h).

**Q3, time information:** posting age works for every listing (`first_seen` lags the posting date by a median 1.4 days); day of week only through `posted` (41%); no seasonality from six months. The still-open rule should be last-seen based: the app's status filter keeps 26% of listings last seen 30+ days before September 30 (6c).

### Limits
- **Six months of history,** and listings first seen before March 30 are missing, so early dates undercount what was open (the date picker starts July 1).
- **We see listings, not hires:** openness is what listings ask for.
- **The text flags are unvalidated keyword rules,** so they stay labels and opt-in filters.
- **Section 3 is descriptive;** EDA_SDA_4 tests the gaps and confounders (staffing agencies are flagged in EDA_SDA_2).

**Handoff:** the in-scope table, its descriptions, the occupation summary and company activity go to **EDA_SDA_2**.
