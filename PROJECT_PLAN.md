# Project Plan

PredictLeads × TripleTen externship project.

**Status:** Draft for team discussion · **Last updated:** 2026-10-06

> This is a living document. Propose changes in a pull request, and record agreed outcomes in the [Decision log](#8-decision-log).
> Items marked **TBD** still need a team decision or input from PredictLeads or our advisors.
> Follow the [Data & Security Rules](README.md#data--security-rules): no data values, `df.head()` output or screenshots in this file.

---

## Contents

1. [Goals](#1-goals)
2. [Data](#2-data)
3. [Current status](#3-current-status)
4. [Three-week plan](#4-three-week-plan)
5. [Analysis plan](#5-analysis-plan)
6. [Roles](#6-roles)
7. [Scope](#7-scope)
8. [Decision log](#8-decision-log)
9. [Risks and compliance](#9-risks-and-compliance)
10. [Open questions](#10-open-questions)
11. [Discussion notes](#11-discussion-notes)

---

## 1. Goals

**Proposed. Needs team agreement.**

**This is a job listing prioritization project.** A job seeker sees far more listings than they can apply to. We rank job listings **for a given applicant**: which ones to apply to first, and why.

| | |
|---|---|
| **Overall goal** | Find the best job search strategy, with a narrow focus and maximum depth. Prioritization is that focus. |
| **Focused question** | For a given applicant, **which job listings should they apply to first** to give themselves the best chance of landing a role? |
| **Approach** | A prioritization model built from measurable factors. **Time series analysis** covers when listings appear and how long they stay open. **NLP** covers what listing text says about employer intent and competition. |
| **Role of the EDA** | Decide where to narrow down and go in depth: which factors drive the ranking, which time information is reliable, and which segment has enough data |
| **Segment** | One occupation group in one country, chosen by the EDA |
| **Side** | The applicant only. "Best time to post" for employers would need hiring outcomes we don't have. |
| **Final deliverables** | A validated prioritization model, an analysis write-up, a web app with a priority queue of listings (see [`APPLICATION.md`](APPLICATION.md)), and a final pitch |
| **Timebox** | Three weeks |
| **Final due date** | TBD |

### Prioritization factors

A listing's priority combines several factors, each measured from fields in our data. We can't see applications or hires, so each one is a proxy. We keep them all in play for now, and the EDA narrows them to the ones with real, usable signal:

| Factor | Question it answers for the applicant | Measured from |
|---|---|---|
| **Still open / apply-by** | Is it still worth applying, and by when? | `first_seen_at` and `last_seen_at` (both 100% filled): how long similar listings stayed open |
| **Freshness** | Is it new enough that I'm early? | `first_seen_at`: days since the listing was first seen |
| **Competition** | How many others am I likely up against? | Reposts (the same company, title and description listed again); listings per company (`relationships.company.data.id`); open listings in the same occupation (`onet_data` with the two dates) |
| **Pay** | Is it worth an application? | `salary_data` (about 43% filled), converted to annual with `salary_time_unit` |
| **Employer intent** | Is the employer in a hurry or flexible? | `description` (82% filled) and `title`: urgency wording, remote mentions, description length |
| **Source** | Where is it posted, and how often should I check there? | `url` (100% filled): the hiring platform or company site |
| **Fit** *(not yet applicable)* | How well does it match this applicant? | Listing side: `title`; `description` (required skills, qualifications, and **years of experience required**, e.g. "3+ years"); `seniority`; `onet_data.occupation_name`; `categories`; `contract_types`; location and remote. Compared with the applicant's résumé using TF-IDF or local embeddings. |

> **Note: fit can't be applied yet.** Its listing-side fields are in our data, but it also needs the applicant's résumé, which isn't in our data. It's listed so the model has a place for it once a résumé is provided.

### Time context that comes with the ranking

Alongside the ranked list, the applicant gets timing advice where the data supports it: **when** to search (busy vs. quiet months), **how often** to check for new high-priority listings and where, and **how fast** to apply once one appears. The EDA (Q2) decides which of these are supported.

### Success criteria

- The prioritization model ranks listings better than a simple baseline (newest first) on a later time period, using outcomes we can measure, such as whether a listing is still open.
- Every factor in the model is backed by a measured pattern with a confidence interval, or reported plainly as having no usable signal.
- Findings are nuanced: factors matter differently by role, source or listing type, rather than generic advice like "apply early".
- Every ranking can be explained: each listing's score shows the factors behind it.
- Claims stay within what the data measures: listings appearing and disappearing, not hires.

---

## 2. Data

Daily snapshots in S3, laid out as `YYYY/MM/DD/`. We are currently working from the **2026-09-30** snapshot.

| Dataset | S3 file | Compressed size | Notes |
|---|---|---|---|
| Companies | `companies_YYYY_MM_DD.jsonl.zst` | ~252 MB | Small enough to load fully |
| Job openings | `job_openings_YYYY_MM_DD.jsonl.zst` | ~19 GB | Too big for memory; filter while streaming |

**Format.** Each line is a JSON:API document with two keys:

- `data`: a list of the actual records, usually with `id`, `type`, `attributes` and `relationships`.
- `included`: a list of related objects that the records reference, such as the company behind a job opening. The same object repeats across documents, so remove duplicates by `(type, id)`.

**Local cache.** The notebook `notebooks/jacob/EDA.ipynb` saves each raw dataset as Parquet in `data/` (git-ignored) after the first read. It flattens the tables in saved parts under `data/flat/`, so a crash can resume, and caches the cleaned table in `data/clean/`.

### Fields the project depends on

| Need | Fields |
|---|---|
| Timing | `attributes.first_seen_at` (100%), `attributes.last_seen_at` (100%), `attributes.posted_at` (44%) |
| Open or closed | `attributes.last_seen_at` relative to the snapshot; `attributes.status` (65%) |
| Segment | `attributes.onet_data.family` and `.occupation_name` (99.9%), `attributes.seniority` (100%), `attributes.location_data` (100%) for country, `attributes.contract_types` (100%) |
| Text (NLP) | `attributes.description` (82%), `attributes.title` (100%); `attributes.tags` and `attributes.categories` (100%, contents still to check) |
| Source | `attributes.url` (100%): its domain shows the hiring platform or company site. Only the platform is kept, never the full URL. |
| Pay | `attributes.salary_data.*` (43%), with `salary_time_unit` |
| Competition proxies | Reposts (derived in cleaning); company posting volume through `relationships.company.data.id` |
| Company information | Name and domain from `job_openings_included` (99.6% / 100%). Description (90%) and location (94%) from the companies table, joined on `relationships.company.data.id` = company `id`. |

### Data quirks

- **The samples are the *first* N documents of the file, not random samples.** If the file is sorted, for example by company or date, the sample is biased. The EDA checks this (Q2b).
- **`first_seen_at` is when PredictLeads' crawler found a posting,** not when the employer posted it. How precise it is depends on how often each company is crawled. The EDA compares it with `posted_at` where both exist (Q2c).
- **A posting disappearing isn't the same as being filled.** It could also be cancelled, expired or moved.
- **Dates are stored as strings.** The cleaning step parses them.
- **Salaries mix time units** (`salary_time_unit`). The cleaning step converts them to one annual figure.
- **`status` is 35% empty.** Use `last_seen_at` relative to the snapshot date to judge whether a posting is still live.
- **Not every job's company may be in the companies file.** The EDA measures the match rate.

### Schemas

Like `df.info()` for each flattened table: column names, non-null counts and dtypes only, never data values. The schema cell in `notebooks/jacob/EDA.ipynb` generates this section. Rerun that cell to refresh it; don't edit it by hand.

<!-- schema:start -->
_Generated 2026-10-06 10:32 from the 2026-09-30 snapshot._

#### `companies`: 340,669 rows × 23 columns

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 340,669 | 0.0% | object |
| 1 | `type` | 340,669 | 0.0% | object |
| 2 | `attributes.domain` | 340,669 | 0.0% | object |
| 3 | `attributes.company_name` | 338,499 | 0.6% | object |
| 4 | `attributes.friendly_company_name` | 338,521 | 0.6% | object |
| 5 | `attributes.ticker` | 5,770 | 98.3% | object |
| 6 | `attributes.meta_title` | 339,031 | 0.5% | object |
| 7 | `attributes.meta_description` | 302,619 | 11.2% | object |
| 8 | `attributes.description` | 307,849 | 9.6% | object |
| 9 | `attributes.description_short` | 307,849 | 9.6% | object |
| 10 | `attributes.language` | 337,961 | 0.8% | object |
| 11 | `attributes.location` | 321,122 | 5.7% | object |
| 12 | `attributes.location_data` | 340,669 | 0.0% | object |
| 13 | `attributes.last_extraction_states` | 340,669 | 0.0% | object |
| 14 | `relationships.redirects_to.data` | 0 | 100.0% | float64 |
| 15 | `relationships.redirects_to.meta.reason` | 18 | 100.0% | object |
| 16 | `relationships.parent_company.data` | 0 | 100.0% | float64 |
| 17 | `relationships.subsidiary_companies.data` | 340,669 | 0.0% | object |
| 18 | `relationships.similar_companies.data` | 340,669 | 0.0% | object |
| 19 | `relationships.parent_company.data.id` | 25,287 | 92.6% | object |
| 20 | `relationships.parent_company.data.type` | 25,287 | 92.6% | object |
| 21 | `relationships.redirects_to.data.id` | 2,013 | 99.4% | object |
| 22 | `relationships.redirects_to.data.type` | 2,013 | 99.4% | object |

#### `companies_included`: 706,570 rows × 5 columns (duplicates removed by type and id)

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 706,570 | 0.0% | object |
| 1 | `type` | 706,570 | 0.0% | object |
| 2 | `attributes.domain` | 706,570 | 0.0% | object |
| 3 | `attributes.company_name` | 682,429 | 3.4% | object |
| 4 | `attributes.ticker` | 8,300 | 98.8% | object |

#### `job_openings`: 200,000 rows × 34 columns (from the first 200,000 job-opening documents)

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 200,000 | 0.0% | object |
| 1 | `type` | 200,000 | 0.0% | object |
| 2 | `attributes.title` | 200,000 | 0.0% | object |
| 3 | `attributes.translated_title` | 3,568 | 98.2% | object |
| 4 | `attributes.normalized_title` | 181,356 | 9.3% | object |
| 5 | `attributes.description` | 164,510 | 17.7% | object |
| 6 | `attributes.url` | 200,000 | 0.0% | object |
| 7 | `attributes.first_seen_at` | 200,000 | 0.0% | object |
| 8 | `attributes.last_seen_at` | 200,000 | 0.0% | object |
| 9 | `attributes.last_processed_at` | 200,000 | 0.0% | object |
| 10 | `attributes.contract_types` | 200,000 | 0.0% | object |
| 11 | `attributes.categories` | 200,000 | 0.0% | object |
| 12 | `attributes.onet_data.code` | 199,906 | 0.0% | object |
| 13 | `attributes.onet_data.family` | 199,906 | 0.0% | object |
| 14 | `attributes.onet_data.occupation_name` | 199,906 | 0.0% | object |
| 15 | `attributes.posted_at` | 87,583 | 56.2% | object |
| 16 | `attributes.recruiter_data.name` | 1,231 | 99.4% | object |
| 17 | `attributes.recruiter_data.title` | 1,173 | 99.4% | object |
| 18 | `attributes.recruiter_data.contact` | 1,231 | 99.4% | object |
| 19 | `attributes.salary` | 91,327 | 54.3% | object |
| 20 | `attributes.salary_data.salary_low` | 86,154 | 56.9% | float64 |
| 21 | `attributes.salary_data.salary_high` | 86,154 | 56.9% | float64 |
| 22 | `attributes.salary_data.salary_currency` | 86,154 | 56.9% | object |
| 23 | `attributes.salary_data.salary_low_usd` | 86,154 | 56.9% | float64 |
| 24 | `attributes.salary_data.salary_high_usd` | 86,154 | 56.9% | float64 |
| 25 | `attributes.salary_data.salary_time_unit` | 86,154 | 56.9% | object |
| 26 | `attributes.seniority` | 200,000 | 0.0% | object |
| 27 | `attributes.status` | 129,097 | 35.5% | object |
| 28 | `attributes.language` | 190,772 | 4.6% | object |
| 29 | `attributes.location` | 183,338 | 8.3% | object |
| 30 | `attributes.location_data` | 200,000 | 0.0% | object |
| 31 | `attributes.tags` | 200,000 | 0.0% | object |
| 32 | `relationships.company.data.id` | 200,000 | 0.0% | object |
| 33 | `relationships.company.data.type` | 200,000 | 0.0% | object |

#### `job_openings_included`: 51,006 rows × 5 columns (from the first 200,000 job-opening documents), duplicates removed by type and id

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 51,006 | 0.0% | object |
| 1 | `type` | 51,006 | 0.0% | object |
| 2 | `attributes.domain` | 51,006 | 0.0% | object |
| 3 | `attributes.company_name` | 50,796 | 0.4% | object |
| 4 | `attributes.ticker` | 3,054 | 94.0% | object |
<!-- schema:end -->

---

## 3. Current status

**Done**
- [x] Environment set up, with S3 access working
- [x] Companies file loads fully and is cached as Parquet
- [x] Job openings sample loads (first 200,000 documents) and is cached as Parquet
- [x] File structure identified as JSON:API (`data` / `included`)
- [x] First flatten of `data` and `included` (companies: all documents; job openings: first 20,000), and the [Schemas](#schemas) section generated from it
- [x] EDA notebook restructured into numbered steps with explanations (`notebooks/jacob/EDA.ipynb`)
- [x] Flattening rebuilt to work in saved parts, so a crash resumes instead of starting over. It now covers all 200,000 sampled job openings. Not yet run on the real data.
- [x] Cleaning step written (types, invalid dates, duplicates and reposts, salary units, missing values, with a cleaning report). Tested on synthetic data; not yet run on the real data.
- [x] Prioritization EDA written for Q1–Q3. Tested on synthetic data.

**What the schemas already tell us** (from the first 20,000 job openings)
- `first_seen_at` and `last_seen_at` are 100% filled, so every posting has a timeline.
- `posted_at` is only 44% filled, which limits the crawler-versus-employer check to that subset.
- `url` is 100% filled, so every posting has a source.
- `description` is 82% filled, so roughly 4 in 5 postings have text for NLP features.
- `onet_data` is 99.9% filled, so every posting can be placed in an occupation group.
- `salary_data` is about 43% filled.

**Next** (the full task list is in [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md))
- [ ] Run the notebook on the real data, and fill in its findings section
- [ ] Team agrees on the focus and the segment

---

## 4. Three-week plan

### Priority queue

The ordered list of every task, including repo setup and housekeeping, is in [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md). It is the single source of truth for what to do next.

### Technique tiers

Each tier builds on the one below it. Finish **Good** before starting **Better**, and **Better** before **Outstanding**. When a new technique comes up, place it in a tier or leave it out.

**Good: the minimum we ship**
- Cleaning pipeline with a cleaning report
- EDA answering Q1–Q3; segment and targets chosen
- Hypotheses written down before any modeling
- **Application window:** Kaplan–Meier survival curves for the segment, with confidence intervals
- **Demand over time:** monthly and weekly new-posting volume, with seasonal decomposition (STL) if there are two or more years of history
- **Arrival cadence:** new postings per week by source and by company, turned into "check every N days"
- Every pattern compared with a simple baseline (the overall average, or last year's value)
- Reproducibility: pinned packages, fixed seeds, snapshot date, saved posting-ID list

**Better: makes the findings specific and hard to dispute**
- **Text predicts timing:** Cox regression with text features (salary disclosed, remote, urgency, TF-IDF terms) to show which posting traits close faster
- **Closes-within-N-days classifier:** TF-IDF + logistic regression vs. a non-text baseline, trained on earlier months and tested on later ones, with folds grouped by company
- **Repost analysis:** how often jobs are reposted, after how long, and what reposted postings have in common
- **Day-of-week patterns,** only if the crawler check shows the dates reflect employers
- **Prioritization model,** combining chance still open, competition proxies, pay and employer intent, compared with a newest-first baseline
- Bootstrap confidence intervals throughout, with Benjamini–Hochberg correction across many comparisons
- **Run log:** settings, seeds and metrics for every run, in a CSV or a local MLflow store. Metrics and settings only, never posting text.

**Outstanding: depth beyond a standard project**
- **Fit:** TF-IDF and local embeddings to match a profile to postings, added to the prioritization score
- **Embeddings vs. TF-IDF** as text features for predicting timing
- **Change-point detection** (`ruptures`) on demand, to find hiring surges and freezes
- **Topic modeling** (NMF or BERTopic) to find sub-roles within the segment that have different timing
- **Several daily snapshots,** to observe postings closing directly instead of inferring it from `last_seen_at`
- **Calibration** of the closes-within-N predictions (reliability diagram, Brier score)
- **Robustness:** rerun the main findings on a second occupation group or a different time window
- **System card:** one page on intended use, data, metrics, limitations and known failure modes

### Week 1: data and EDA

- [ ] Run flattening, cleaning and the prioritization EDA on the real data
- [ ] Choose the segment (Q3), the time scales (Q2) and the targets (Q1)
- [ ] Write the hypotheses for each prioritization factor
- [ ] If the sample is too small or biased for the segment, stream more of the full file with filters
- [ ] Move shared loaders into `src/`

**Exit criteria:** segment, targets and hypotheses agreed.

### Week 2: timing analysis

- [ ] Still open / apply-by: application window for the segment
- [ ] Seasonal context: demand over time
- [ ] Source and freshness: arrival cadence by source and company
- [ ] Text features vs. timing: Cox regression and the closes-within-N classifier
- [ ] Repost analysis

**Exit criteria:** each layer has a measured result, or a documented null result.

### Week 3: prioritization model, write-up, pitch

- [ ] Prioritization model for live listings
- [ ] Validate the prioritization model against a newest-first baseline, and add the timing advice that comes with the ranking
- [ ] Write-up: methods, results with confidence intervals, limitations
- [ ] Final pitch

**Exit criteria:** prioritization model, write-up and pitch ready.

---

## 5. Analysis plan

### Questions the EDA answers

The EDA is built to answer these three before any modeling:

| # | Question | What it decides |
|---|---|---|
| Q1 | Which factors should drive prioritization? | The factors that vary enough between listings, roles and sources to change the ranking |
| Q2 | What time information can prioritization rely on? | Apply-by estimates, check cadence, and seasonal context: which are supported |
| Q3 | Which segment do we go in depth on? | Where the data is rich enough for nuanced, trustworthy findings |

### Candidate targets

Kept broad on purpose; the EDA narrows them.

| Measure | Prioritization factor | What it tells an applicant |
|---|---|---|
| Days a posting stays open | 3. How fast | How long the window to apply is |
| Closed within N days | 3. How fast | How urgent applying is |
| New postings per week or month | 1, 2. When, how often | When to search and how often to check |
| Postings open at once | 4. Which | How many options exist right now |
| Repost rate | 4. Which | Jobs that didn't fill may mean less competition |
| Pay, and whether it's disclosed | 4. Which | Whether a posting is worth an application |
| Urgency and remote wording | 3, 4. How fast, which | Employer intent |
| Source (hiring platform or company site) | 2. Where | Where to check |
| Company concentration | 2, 4. Where, which | Whether a short watch list covers most openings |

### Hypotheses

Drafts, to be made specific (segment, thresholds) after the EDA and **before** any modeling:

- **H1, application window:** in the chosen segment, half of postings close within N days, and the window differs between sources or posting types.
- **H2, seasonality:** monthly new-posting volume varies by season more than chance would explain.
- **H3, cadence:** new postings arrive unevenly across sources and companies, so a targeted check schedule covers more openings than checking everything equally.
- **H4, text and timing:** text features predict whether a posting closes within N days better than a model without text, on a later time period.

If a hypothesis fails, that's still a finding. Report it.

### Validation design

- **Temporal holdout:** train on earlier months, test on later ones. A ranking has to work going forward.
- **Folds grouped by company,** so models learn about postings, not one company's habits.
- **Censoring:** still-open postings are kept and handled by survival methods, never dropped.
- **Baselines:** the segment's overall average, last year's value for seasonality, newest-first for prioritization.
- **Statistics:** bootstrap 95% confidence intervals; Benjamini–Hochberg correction when testing many features or segments.

### Leakage rules

- **Features may only use what's known when a posting first appears:** its text, source, company, pay and `first_seen_at`. Never use `last_seen_at`, `status`, days open or the repost flag as a feature for predicting timing; they come from the future.
- Never tune on the test period.

### Text preprocessing

Done in cleaning step 6.1 of the notebook: strip HTML, decode entities, collapse whitespace. Still to add for modeling: remove recurring boilerplate (equal-opportunity statements, benefits sections) where section headers make it detectable.

### TF-IDF configuration

For text features and, later, fit. Start from these defaults (`TfidfVectorizer`) and tune on the training period only:

| Setting | Start | Why |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures phrases like "customer success" and "project management" |
| `sublinear_tf` | `True` | Damps repeated keywords |
| `max_df` | about `0.5` | Drops boilerplate that appears in many postings |
| `min_df` | `2`–`5` | Drops typos and one-off terms |
| `stop_words` | `"english"` | Standard |

### Embedding configuration

For the Outstanding tier. A small local `sentence-transformers` model such as `all-MiniLM-L6-v2`. Chunk long descriptions, normalize vectors, and cache embeddings in `data/`, keyed by posting ID. No data goes to outside services.

### Reproducibility

- Package versions are pinned in `requirements.txt`.
- Save the list of posting IDs in the analysis table, and the snapshot date.
- Fix random seeds, and record each run's settings with its results.

---

## 6. Roles

| Person | Focus | Owns |
|---|---|---|
| Jacob | EDA, data pipeline, cleaning | `notebooks/jacob/`, loaders in `src/` |
| TBD | Time factors: seasonal context, arrival cadence, freshness | |
| TBD | Still open / apply-by and text models (employer intent) | |
| TBD | Prioritization model and pitch | |

---

## 7. Scope

- One segment: one occupation group in one country
- The applicant's side only
- Job listing prioritization for a given applicant, with timing advice (when to search, how often to check, how fast to apply) where the data supports it
- "Best chance" built from proxies the data supports

**How to describe the machine learning:** time series and survival analysis of posting activity, plus supervised learning on posting text to predict timing (classification and Cox regression), validated on a later time period. Text representations (TF-IDF, embeddings) serve as features and, in the Outstanding tier, for fit.

---

## 8. Decision log

| Date | Decision | Why | Status |
|---|---|---|---|
| 2026-10-05 | Cache parsed datasets as Parquet in `data/` | Parsing the JSON takes about 10 minutes per run | Adopted |
| 2026-10-05 | Use a 200,000-document sample of job openings for EDA | The full file is about 19 GB compressed | Adopted |
| 2026-10-06 | Remove duplicate `included` objects before flattening | Flattening every copy first crashed the kernel | Adopted |
| 2026-10-06 | Narrow focus with maximum depth | Three weeks isn't enough for breadth | Adopted |
| 2026-10-06 | Flatten in saved parts that resume after a crash; skip the raw load once flattened | The laptop has 8 GB of RAM, and the full flatten crashed | Adopted |
| 2026-10-06 | Add a cleaning step with a cleaning report before any analysis | Timing results depend on valid dates, de-duplicated postings and comparable salaries | Adopted |
| 2026-10-06 | Focus: job listing prioritization for a given applicant, combining NLP and time series; the EDA decides where to go in depth | Uses the strongest fields in the data (complete timelines, text, source); gives a specific, explainable answer to "what should I apply to first?" | Proposed |
| 2026-10-06 | Applicant side only | Applicant-side patterns are observable; "best time to post" needs hiring outcomes we don't have | Proposed |
| 2026-10-06 | Keep "best chance" broad until the EDA shows which proxies carry signal | Avoids locking in a target before seeing the data | Proposed |
| 2026-10-06 | Flag reposts instead of deleting them; leave them out of timing estimates | Each job counts once, and reposts stay available as a competition signal | Proposed |
| 2026-10-06 | Use `first_seen_at` for posting age, not `posted_at` | `posted_at` is only 44% filled; `first_seen_at` is 100% | Proposed |
| 2026-10-06 | Treat a posting as live if seen within 2 days of the snapshot | A closed posting is a useless recommendation; live postings are censored in survival analysis | Proposed |
| | Segment: occupation group × country | | TBD after EDA |
| | Prioritization factors and time information the model uses | | TBD after EDA |
| | Tool for processing the full job openings file | | TBD |
| | Virtual environment name: README says `.venv`, Jacob uses `PL` | Consistent setup across the team | TBD |

---

## 9. Risks and compliance

| Risk | Mitigation |
|---|---|
| **Sending real job data to outside AI services** | The advisors' rules (in `CLAUDE.md`) allow AI tools for coding, but no confidential data may be uploaded to them. Run any embeddings locally. |
| **AI coding tools on project files** | Allowed for coding, brainstorming and debugging, per `CLAUDE.md`. Claude never reads `data/` or `.env`, and works from code, column names, dtypes and shapes only. |
| **Crawler timing mistaken for employer behavior** | Compare `first_seen_at` with `posted_at` (Q2c). Only claim day-of-week patterns if they match. |
| **Disappearing mistaken for filled** | Word every claim as "postings close" or "disappear", never "get filled". State it in the limitations. |
| **Too little history for seasonality** | Q2a measures it. If there's under two years, use several older snapshots, or drop the seasonal layer and say so. |
| **Look-ahead leakage in timing models** | Follow the [leakage rules](#leakage-rules): features only from what's known when a posting appears. |
| Biased sample: the first N documents, not random | Q2b checks the sort order. If the file is sorted, sample randomly while streaming the full file. |
| Too few postings in the chosen segment | Q3 sizes it. Fall back to streaming the full file with filters, or a bigger segment. |
| Data leaves approved locations | Keep data only in `data/` or S3. Never commit data. Clear notebook outputs before every commit. |
| Notebook kernel crashes lose work | Flattening resumes from saved parts; cleaning is cached. Run cells from the top after a restart. |
| End-of-project data deletion | Track every local copy (`data/`, downloads) so all of them can be deleted and confirmed in writing |

---

## 10. Open questions

**For the team**
- Do we agree on this focus?
- Which "best chance" factors matter most to us: still open, less competition, fit, pay, employer intent?
- Who owns which workstream?
- What meeting cadence should we use, and where do we discuss async?
- How should the final results be presented alongside the web app: a write-up, a notebook demo, or both?

**For our project advisor.** The ready-to-send email is in [`EMAILS.md`](EMAILS.md); record her answers in the [Decision log](#8-decision-log).

1. **Scope check:** is job listing prioritization for a given applicant (NLP + time series, one occupation group × one country) appropriate for the project?
2. **Deliverables and due date:** the final due date, and the expected format (write-up, presentation, demo).
3. **Web app and multi-agent requirements:** is there a rubric or a list of must-haves?
4. **AI agents and data:** may an agent backed by an outside LLM process posting text, or only aggregates, model outputs and public data? Are local models fine?
5. **Résumés:** may team members use their own de-identified résumés for the fit feature, and are there storage rules?
6. **External data:** may we combine public datasets (BLS JOLTS, BLS wage statistics, O*NET) with the PredictLeads data?
7. **Hosting:** may the app be hosted, and where, or must it stay a local demo?
8. **Presenting results:** may the final presentation show real postings and company names, or only aggregates and synthetic examples?
9. **Data deletion:** what's the procedure at the end, and does she need written confirmation?

**For PredictLeads.** The ready-to-send email is in [`EMAILS.md`](EMAILS.md).

1. **Project fit.** We're proposing a job listing prioritization project: ranking job listings for a given applicant by which to apply to first. The ranking would combine how long listings like it stay open, competition signals such as reposts, pay, what the listing text says, and fit with the applicant's résumé, for one occupation group in one country. Does that fit what you hoped to get from this project? Is there a use case, customer type or deliverable format you'd like us to aim for instead?

2. **Schema documentation.** Do you have documentation for the job openings and companies datasets, such as field definitions, allowed values and how each field is populated? We especially want it for the date fields (`first_seen_at`, `last_seen_at`, `posted_at`) and for `onet_data` (`code`, `family`, `occupation_name`): how are O*NET codes assigned, from the title, the description or both?

3. **Crawl timing.** How often do you crawl each company's careers page or hiring platform? Is `first_seen_at` the first crawl that found a posting, and `last_seen_at` the last crawl that still saw it? Does crawl frequency vary by company, platform or over time? We want to know how precise these dates are, in days.

4. **Daily snapshots and history.** Is each daily file a full copy of all postings, including ones that have closed, or only what changed since the previous day? How far back does a snapshot's history go? Are older snapshots available if we need them to measure seasonality?

5. **Posting status.** What values can `status` take, what does each one mean, and how is it determined? In our sample it's only filled for about two thirds of postings. What's the most reliable way to tell whether a posting is still open, `status` or `last_seen_at`? And if it's `last_seen_at`, how recent should it be?

6. **Sources.** Do postings come only from company careers pages and hiring platforms, or also from job boards? Is `url` always the original source of the posting?

7. **Field origins.** Are `normalized_title`, `categories`, `tags` or `seniority` derived from O*NET, or produced by the same classifier? And what do `tags` contain: skills, keywords or something else?

8. **File order and sampling.** Is the job openings file sorted in any way, for example by company, date or crawl order? We've been working from the first 200,000 documents and want to know whether that's representative. If it isn't, is there a smaller or pre-sampled extract you'd recommend instead of streaming the full 19 GB file?

9. **Duplicates and reposts.** Does the same job ever appear as more than one posting, for example when it's reposted or listed in several locations? If so, is there a field that links the copies?

10. **Description text.** Is `description` the original posting text, or has it been cleaned, for example by stripping HTML or removing boilerplate? How is `language` detected?

11. **Final presentation.** Can we show real posting titles, company names and short description excerpts in the final presentation, or should we use synthetic examples only?

---

## 11. Discussion notes

Add the newest meeting at the top. Copy this template:

```markdown
### YYYY-MM-DD: <meeting topic>
**Attendees:**
**Discussed:**
-
**Decisions:** (also add these to the Decision log)
-
**Action items:**
- [ ] <task> (owner, due date)
```
