# Data Reference

PredictLeads data used by the project. Aggregates only: no data values, `df.head()` output or screenshots in this file.

## Sources

PredictLeads exports its whole database to S3 every day, one folder per day (`YYYY/MM/DD/`); it calls each day's export a *snapshot*. We use the export of **2026-09-30**, so **September 30, 2026 is the data's end date:** the files hold every listing the crawler found up to that day, with the first and last day it saw each one, and nothing after. "Still open" means last seen within 2 days before September 30.

| Dataset | S3 file | Compressed size | Notes |
|---|---|---|---|
| Companies | `companies_YYYY_MM_DD.jsonl.zst` | ~252 MB | Small enough to load fully |
| Job openings | `job_openings_YYYY_MM_DD.jsonl.zst` | ~19 GB | Too big for memory; we use the first 200,000 documents |

**Format.** Each line is a JSON:API document: `data` (the records, with `id`, `type`, `attributes`, `relationships`) and `included` (related objects such as the company behind a job opening; duplicates removed by `(type, id)`).

**Local cache.** `notebooks/EDA_SDA_1.ipynb` caches raw data as Parquet in `data/` (git-ignored), flattens tables in resumable parts under `data/flat/`, and saves cleaned tables under `data/clean/`.

**Full file (streamed 2026-10-10, `src/stream_full.py`):** 13,415,388 job-opening documents, one listing each (the same count as the hackathon app's full scan). Kept: **1,019,346 scope listings** (English or unlabeled, US or no country, computer and mathematical by occupation group or a 15- O*NET code; all columns, `data/flat/job_openings_full/`), their related objects, and a lean all-fields table of 13,017,464 English US listings (typed Parquet, no text, `data/flat/all_fields_lean_full/`). 5.8 GB on disk; 51 minutes. **Second pass (`--rows`, 2026-10-10):** EDA_SDA_1's cleaned row for the same 13,017,464 listings (count matches), plus `job_title`, a has-recruiter flag, `last_processed`, pay as posted with its currency, and a translated flag (typed Parquet, no text, `data/flat/all_fields_rows_full/`). 2.23 GB on disk; 223 minutes. **Third pass (`--extra`, 2026-10-10):** every field for the two TripleTen roles outside computer and mathematical (sales engineers; management and market research analysts), English or unlabeled and US or no country: 94,467 listings (`data/flat/job_openings_extra/`), 0.47 GB; 23 minutes.

## External data: structured resumes

| Dataset | Local folder | License | Notes |
|---|---|---|---|
| Kaggle `suriyaganesh/resume-dataset-structured`, version 2 (2024-11-14) | `data/external/resume_structured/` | MIT | About 54,000 resumes, OCR'd and LLM-structured from public resume collections. Files are numbered (`01_people.csv` … `06_skills.csv`). Tables: `people` (never loaded: names and contact details), `experience` (title, firm, start/end MM/YYYY, location; only title and dates are loaded), `education`, `person_skills`, `skills`, `abilities`. Skills are undated. Used by `notebooks/Profiles.ipynb` |

**What it gives us:** 54,933 resume records → 17,952 unique people (each resume appeared about 3 times) → 16,237 who held a computer and mathematical job → **2,813 career changers**: 2,049 moved between tech roles (sideways or up; Profiles 4a.3 splits them), 407 retrained after working in another field, 357 retrained as new grads. For each: background, training field, first tech role, the switch date and the time it took. Most switches are from 2010–2019, so the dataset supplies career paths; the 2026 listings supply openings, pay and timing.

**Saved by `notebooks/Profiles.ipynb`** (git-ignored, under `data/profiles/`; rerun after any change to the rules):
- `modeling_table.pkl` (section 5): the final Profiles table for `ML_Fit`. All 16,237 people who reached a computer and mathematical job, one row each: pre-switch features, the role at the switch as the label, outcome columns (never features), and the `career_changer` flag marking the 2,813 career changers that results are reported on.
- `retrained_candidates.csv` (4a.2a): the 764 retrained career changers, for browsing locally.

## External data: job growth expectations

Official occupational employment projections (for example BLS's 10-year projections): expected employment change by occupation code. They join to listings and role buckets through the occupation part of `onet_code`. **Purpose:** recommendations based on future estimates, never more than **5 years** out (the exact horizon within that is the team's call). The 10-year rate converts to any shorter one at the same yearly rate: growth over *n* years = (1 + 10-year growth)^(*n*/10) − 1. To download into `data/external/` once the terms of use are checked and PredictLeads consents in writing to combining their data with it (agreement 4.3). **Current release: BLS Employment Projections 2025–2035** ([data tables](https://www.bls.gov/emp/data.htm)). Full plan: [`VISION.md` section 3.3](../VISION.md#33-expected-job-growth). Other public sources there (OEWS, JOLTS, Indeed postings and the fed funds rate on FRED, FOMC projections, Projections Central) are used only if the team adopts the matching extension. IT project management (15-1299.09) has no projection of its own and uses 15-1299 as a proxy.

## Fields the project depends on

| Need | Fields |
|---|---|
| Timing | `attributes.first_seen_at` (100%), `attributes.last_seen_at` (100%), `attributes.posted_at` (44%) |
| Open or closed | `attributes.last_seen_at` relative to September 30, the data's end date; `attributes.status` (65%) |
| Segment | `attributes.onet_data.family`, `.occupation_name` and `.code` (99.9%; the code maps to a role bucket), `attributes.seniority` (100%), `attributes.location_data` (100%) for country, `attributes.contract_types` (100%) |
| Text (NLP) | `attributes.description` (82%), `attributes.title` (100%); `attributes.tags` and `attributes.categories` (100%, contents still to check) |
| Source | `attributes.url` (100%): its domain shows the hiring platform or company site. Only the platform is kept, never the full URL. |
| Pay | `attributes.salary_data.*` (43%), with `salary_time_unit` |
| Career-change accessibility | `attributes.seniority` (100%) for entry level; `attributes.description` (82%) for years of experience required, degree required, "or equivalent experience", entry-level and training wording, and career-changer wording (keyword rules in `notebooks/EDA_SDA_1.ipynb` 6.1) |
| Competition proxies | Reposts (derived in cleaning); company posting volume through `relationships.company.data.id` |
| Company information | Name and domain from `job_openings_included` (99.6% / 100%). Description (90%) and location (94%) from the companies table, joined on `relationships.company.data.id` = company `id`. |

## Data quirks

- **What `first_seen_at` and `last_seen_at` mean:** the first and the last day PredictLeads' crawler found a listing on the web. A listing is **live on a date** if it was first seen on or before that date and last seen on or after it; **days open** = last seen − first seen. Listings last seen within 2 days before September 30 (the data's end date) count as still open; when they close is unknown (censored). The date picker in the app uses the same rule.

- **The samples are the *first* N documents of the file, not random samples.** If the file is sorted, for example by company or date, the sample is biased. The EDA checks this (Q2b).
- **`first_seen_at` is when PredictLeads' crawler found a posting,** not when the employer posted it. How precise it is depends on how often each company is crawled. The EDA compares it with `posted_at` where both exist (Q2c).
- **A posting disappearing isn't the same as being filled.** It could also be cancelled, expired or moved.
- **Dates are stored as strings.** The cleaning step parses them.
- **Salaries mix time units** (`salary_time_unit`). The cleaning step (EDA_SDA_1 6.5) converts them to one hourly rate, `salary_hourly_usd` (salaried pay ÷ 2,080 hours, full time).
- **`status` is never "open":** it's "closed" or empty. Use `last_seen_at` relative to September 30 (the data's end date) to judge whether a listing is still live.
- **Every job opening's company is in the companies file** (100% match). The companies file has no company size or employee count.
- **The export holds a rolling six-month window:** listings first seen 2026-03-30 to 2026-09-30, so long-running listings that started earlier are missing (left truncation).
- **`posted_at` timestamps sit at 22:00 or 23:00 UTC,** which is midnight Central European time, so they're probably local dates.
- **Listings without a language label almost all lack a description** (99.8%).
- **`seniority`'s labels aren't documented.** The entry-level flag counts only `junior` for now; check every label before relying on it (EDA_SDA_1 Q3b).
- **"Company site / other" (60.7% of listings) means "not on our lists of hiring platforms and job boards",** so it's an upper bound on listings hosted on employers' own sites. In the computer and mathematical scope, 58.5% of these are on the employer's own domain, 38% of all scope listings (EDA_SDA_2 3.2d).
- **Job-board listings stop after May 2026:** 494 and 526 first seen in April and May, then 4, 2, 2 and 0 from June to September (EDA_SDA_2 3.2c). A change in collection, not in employers; job-board timing results cover April–May only. Asked of PredictLeads.
- **More listings have no description from June 2026:** 2–10% of scope listings first seen in March–April lack one, about 30% from June on, mostly on company sites (EDA_SDA_2 3.9). They close in a median 3 days vs. 30, so "has a description" is a standard confounder; asked of PredictLeads.

## Schemas

Like `df.info()` for each flattened table: column names, non-null counts and dtypes only. Generated by the schema cell in `notebooks/EDA_SDA_1.ipynb`; rerun that cell to refresh it, don't edit by hand.

<!-- schema:start -->
_Generated 2026-10-06 10:32 from the 2026-09-30 data export._

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
