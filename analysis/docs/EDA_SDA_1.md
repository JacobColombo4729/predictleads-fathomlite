<!-- Generated from notebooks/EDA_SDA_1.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 1 (Full File, Field Comparison and Scope)

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

For a career changer, the project suggests the computer and mathematical roles their skills transfer to, hides the listings they can't get, and ranks the rest; the **retraining compass** is the foundation. **This notebook** cleans PredictLeads' full job-openings file, compares computer and mathematical roles with every other field, applies the filters we already know we want (the scope: **the roles TripleTen trains for, plus adjacent tech roles**), and explores the scope by role bucket:
- **Q3 (section 7):** how do computer and mathematical roles compare with every other field?
- **Q1 (9.1):** which factors should drive prioritization for a career changer?
- **Q2 (9.2):** what time information can prioritization rely on?

**The data:** PredictLeads' export of **September 30, 2026, the data's end date** (PredictLeads exports its whole database daily and calls each export a *snapshot*): every listing its crawler found from March 30 to September 30, with the first and last day it saw each one. **"Still open"** means last seen within 2 days before September 30. 13,415,388 listings; 13,017,464 English (or unlabeled) and US (or no country). Sections 2–5 show the raw structure on the first 200,000 documents; the analysis uses the full file.

**Hands off to EDA_SDA_2** (section 11): the scope table (TripleTen-role and adjacent tech listings, every field), their full descriptions, and two summaries of every field (occupation pay and timing, company activity).

**Conventions:**
- Every decision is recorded with its reasoning; the filters and why (section 8).
- Fix or flag, never drop, except data errors; reposts become a feature.
- Aggregates only: shares, medians and compact counts (13.0M, 412K); never rows. Clear outputs before committing.
- Each table or chart group ends with **What the data shows**: up to three bullets.

**The series** (each notebook hands its final table to the next):

| # | Notebook | Question | Hands off |
|:-:|:--|:--|:--|
| **1** | **EDA_SDA_1 (you are here)** | **What's in the full file, how does tech compare, and what's in the scope?** | **Scope table (every field) and its descriptions; occupation summary and company activity (every field)** |
| 2 | EDA_SDA_2 | Which scope listings do we keep and how do they behave? Additional cleaning, ghost listings, non-text checks by role bucket, blockers hidden, the compass's 2026 side | Scope table, non-text features |
| 3 | EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Scope table + text features |
| 4 | EDA_SDA_4 | Which features hold up? Statistics | Modeling-ready table |
| 5 | TimeSeries | When do listings appear and close? | Timing features |
| 6 | ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| 7 | ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |
| side | Profiles (frozen) | Who changed careers into or within tech or math, and how? (public resume dataset; three career-changer pools) | The compass's past side (path tables) and the modeling table for ML |

## 1. Setup

Libraries and the data loader, `src/pl_data.py`: the S3 connection, cached raw documents, resumable flattening and the schema writer. Sections 2–5 each call one of its functions.

## 2. Load raw data

Both datasets are files in S3, part of a full export PredictLeads makes every day (it calls each day's export a *snapshot*); we use the one from September 30, 2026, stored as compressed JSON lines (`.jsonl.zst`): **companies** (~252 MB compressed) are loaded in full; **job openings** (~19 GB) are too big for 8 GB of RAM, so we read the first 200,000 documents. Each is cached as Parquet in `data/`, so later runs take seconds instead of about 10 minutes. Once every table is flattened (section 4), this step skips the raw documents and S3 entirely.

**Sections 2–5 use these 200,000 documents** to show the raw structure, flatten it and write the schemas. **The analysis from section 6 on uses the full file:** `src/stream_full.py` reads all 13.4 million listings from S3 line by line and saves 6.1e's cleaned row for every English (or unlabeled) US (or no-country) listing, 13,017,464 rows (`python -m src.stream_full --rows`).

## 3. Inspect structure

Each line of a file is a JSON:API document with two keys: `data` (the records) and `included` (related objects the records point to, such as the company behind a job opening). Column names and Python types only.

## 4. Flatten

Turns the nested documents into ordinary tables, one row per record, with nested fields as dotted columns (for example `attributes.first_seen_at`): the companies and job openings, and each file's related objects with duplicates removed. Each table is built in parts of 20,000 records saved to `data/flat/` as they finish, so after a crash a rerun skips finished parts, and a finished table is never rebuilt.

## 5. Schema

Writes a `df.info()`-style summary of every flattened table (column names, non-null counts, dtypes) into `docs/data.md`, only when something changed.

## 6. Data cleaning

Cleaning comes before any analysis, because every timing factor depends on valid dates. **Data-quality fixes happen here** (impossible dates, duplicates, placeholders, mixed units, inconsistent labels). **Scope decisions wait for section 10** and are applied together in section 11, since they're judgment calls that need the full data as evidence. Reposts are kept as a feature, not removed. Each step shows summary tables only and logs the rows it touched; 6.7 has the full cleaning report.

### 6.0 Chart style

One style for every chart in the notebook: a light surface, recessive grid, thin marks, and a fixed color order (blue, orange, aqua, yellow) so a color always means the same thing within a chart. Charts show aggregates only.

### 6.1 Cleaned rows and text cleaning

The stream (`src/stream_full.py --rows`) ran 6.1e on every listing in the file and saved the cleaned rows as typed Parquet, without text. This section loads them.

**Decisions, and why:**
- **Descriptions are cleaned** (HTML removed, entities decoded, whitespace collapsed), so word counts measure words, not markup.
- **Descriptions become features:** `has_desc`, `has_html`, `desc_words`, `mentions_remote`, `urgency` (saved, but analyzed only in EDA_SDA_3's NLP), and `desc_hash` (the repost fingerprint for 6.4). *Why:* memory; the full text stays in the stream's scope parts for NLP.
- **Career-change flags:** `years_required` (most years asked, lower end of each range; blank if none; above 20 ignored), `degree_required`, `equivalent_experience`, `entry_wording`, `changer_wording`. *Why:* what a listing asks of a newcomer is known the moment it appears. They're keyword rules, so EDA_SDA_3 checks them for false positives.
- **Without a description, text flags are unknown (blank), not False:** "we can't tell" isn't "it doesn't say so". `mentions_remote` is still True when the title says remote. Comparisons (Q1f, Q1g) set listings with a flag against listings *known* not to have it.
- **The `url` becomes its source:** a hiring platform, a job board, or the company's own site. Full URLs aren't needed.
- **13 million rows in 8 GB of RAM:** only light columns are held in memory (dates, numbers, flags, repeated text as categories). Titles, descriptions and list fields are reduced part by part to fingerprints, counts and flags (`src/full_rows.py`), and the text and list columns are added back from disk when the final table is saved (11c).

**Every field in the original job-openings table, and where it's kept** (34 flattened columns)

The stream saved two sets: **every listing** (13.0M cleaned rows, this section, no long text) and **the scope listings** (about 1.0M computer and mathematical listings, every original field, including the full description). The scope table (section 11) carries both; the full descriptions sit next to it in `scope_descriptions.parquet`.

| Column | Description | Where it's kept |
|:--|:--|:--|
| `id` | Listing id; links every table and the descriptions file | Every listing |
| `type` | Record type, the same for every listing (kept for completeness) | Scope (`record_type`) |
| `attributes.title` | Title as posted, plus `title_norm` (lowercased, spaces collapsed) to match reposts | Every listing |
| `attributes.translated_title` | Title translated by PredictLeads; almost always empty. Only a flag is kept, since the scope is English only | Every listing, as `was_translated` |
| `attributes.normalized_title` | PredictLeads' standardized title (`job_title`), falling back to the posted title where it's missing (`job_title_fallback`) | Every listing |
| `attributes.description` | Full listing text. Too large for 13M listings (about 50 GB), so every listing keeps counts and flags (`has_desc`, `has_html`, `desc_words`, the text flags, the repost fingerprint `desc_hash`) | **Full text: scope only** (`scope_descriptions.parquet`); counts and flags: every listing |
| `attributes.url` | Link to the listing. Every listing keeps its source (`source_type`, `platform`); the scope keeps the link, which the app needs | Every listing (source); scope (`url`) |
| `attributes.first_seen_at` | When the crawler first found the listing (`first_seen`): freshness, `days_open`, `month` | Every listing |
| `attributes.last_seen_at` | When the crawler last saw it (`last_seen`): `live` and `days_open` | Every listing |
| `attributes.last_processed_at` | When PredictLeads last processed the listing (`last_processed`); a leakage audit only, since it likely moves with `last_seen` | Every listing |
| `attributes.contract_types` | Full time, part time, remote and so on (a list; labels unified) | Every listing |
| `attributes.categories` | PredictLeads category labels (a list) | Every listing |
| `attributes.onet_data.code` | O*NET-SOC occupation code (`onet_code`): role buckets and job growth join on it | Every listing |
| `attributes.onet_data.family` | O*NET occupation group (`occupation_group`): the field comparison (section 7) and the scope cut | Every listing |
| `attributes.onet_data.occupation_name` | O*NET occupation name (`occupation`) | Every listing |
| `attributes.posted_at` | The employer's posting date, where given (`posted`); implausible dates blanked (6.3) | Every listing |
| `attributes.recruiter_data.name` | Recruiter's name: personal data, never stored. An anonymous key per recruiter (`recruiter_key`, salted fingerprint of name and company) lets us study recruiters without knowing who they are | Scope, as `recruiter_key`; every listing, as the flag `has_recruiter` |
| `attributes.recruiter_data.title` | The recruiter's job title, e.g. technical recruiter (`recruiter_title`) | Scope |
| `attributes.recruiter_data.contact` | Recruiter's contact details: personal data | **Not kept** (only the `has_recruiter` flag) |
| `attributes.salary` | Pay as written (`salary_text`); parsed into hourly pay where `salary_data` has no numbers (8d) | Scope |
| `attributes.salary_data.salary_low` | Low end of the pay range, in the posted currency (`salary_low`) | Every listing |
| `attributes.salary_data.salary_high` | High end of the pay range, in the posted currency (`salary_high`) | Every listing |
| `attributes.salary_data.salary_currency` | Currency of the posted pay (`salary_currency`) | Every listing |
| `attributes.salary_data.salary_low_usd` | Low end converted to USD (`salary_low_usd`) | Every listing |
| `attributes.salary_data.salary_high_usd` | High end converted to USD (`salary_high_usd`) | Every listing |
| `attributes.salary_data.salary_time_unit` | Hour, day, week, month or year (`salary_unit`): converts pay to `salary_hourly_usd` (6.5) | Every listing |
| `attributes.seniority` | Seniority level (`seniority`) | Every listing |
| `attributes.status` | PredictLeads status, only "closed" or empty (`status`); not used as a live flag: `live` comes from `last_seen` | Every listing |
| `attributes.language` | Listing language (`language`): the English-only filter | Every listing |
| `attributes.location` | Location as written (`location_text`), e.g. a city or "Remote - US"; for the app's location filter | Scope |
| `attributes.location_data` | Structured locations (a list of places). Every listing keeps `country` (first location) and `n_locations`; EDA_SDA_2 reads the state from the scope's records | Every listing (country, count); scope (full records, in the stream's scope parts) |
| `attributes.tags` | PredictLeads tags (a list; about half empty) | Every listing |
| `relationships.company.data.id` | The hiring company's id (`company_id`): company volume, links to the companies table | Every listing |
| `relationships.company.data.type` | Type of the linked company record, the same for every listing (kept for completeness) | Scope (`company_record_type`) |

The file's related objects (`job_openings_included`: each company's domain, name and stock ticker) come from the companies table through `company_id` (EDA_SDA_2).

**What the data shows**
- **13.0M English US listings** from the full file of 13.4M (3.0% removed for another language or country). 83% have a description, a median **608 words**, so later embedding work needs chunking.
- **Among listings with a description:** 13% mention remote work and 27% state years of experience (median 3; 90% ask for 8 or fewer). Only 10% use entry-level or training wording, 3.7% require a degree and **0.4% invite career changers**.
- **60% are on company sites or other domains,** 29% on hiring platforms and 11% on job boards.

### 6.2 Types and standard categories

**Decisions, and why:**
- **Dates become UTC timestamps,** so they can be compared. Text that won't parse becomes missing, and is counted: guessing would invent data.
- **Category labels are trimmed and lowercased,** and `contract_types` spellings are unified ("full_time" = "full time"), so one value counts once.
- **`title_norm`** (lowercased, whitespace collapsed) is used to match reposts: built part by part into the repost fingerprint (6.1f) and saved with the final table (11c).
- **Dates were parsed in the stream** (text that wouldn't parse became missing there).

**What the data shows**
- **The dates are complete:** every listing has a first and last sighting between March 30 and September 30. The employer's posting date exists for 44%, with placeholders (1969) and future dates handled in 6.3.
- **Labels were already clean** at full scale: trimming and lowercasing merged nothing. Contract types are the exception: 166 labels, only 5 merged by spelling, and many overlapping ("intern" and "internship", "m/f"), which EDA_SDA_2 sorts out.
- **`status` is "closed" (65%) or empty, never "open".** Seniority: mid/senior 53%, junior 30%, manager 12%.

### 6.3 Invalid dates

**Decisions, and why:**
- **Missing `first_seen` or `last_seen`, `last_seen` before `first_seen`, or dates after September 30 (the data's end date): removed.** These are data errors, and such a listing can't be placed in time.
- **`posted` more than a day after `first_seen`, or before 2000: only `posted` is blanked.** The crawler can't find a listing before it exists, and 1969-12-30 is a placeholder; the listing itself stays.

**What the data shows**
- **No impossible timelines:** no listing lacks a date, ends before it starts, or runs past September 30, so nothing was removed (the date-loss warning didn't trigger).
- **349K employer posting dates (2.7%) were blanked:** placeholders before 2000, or dates after the crawler first saw the listing.
- **Listings are seen for a median 16.7 days;** a quarter for 40+ days, and 10% for more than 78.

### 6.4 Duplicates and reposts

**Decisions, and why:**
- **The same record `id` twice:** one copy kept (a data error).
- **Reposts** (same company, normalized title and description fingerprint, new `id`; matched on `repost_fp`, one 64-bit fingerprint of the three) **are kept and flagged** with `is_repost`; the earliest copy is the original. Features use **earlier copies only**, so they're known when a listing appears: `times_listed`, `days_since_first_listed`, `days_since_repost_last_listed`. Whether a listing *will* be reposted is future information and isn't used. Listings without a description can't be matched.
- **The EDA tables count each job once** (originals, `jo` in 6.7): a counting view, not a filter.

**What the data shows**
- **No duplicate ids.** 1.4M listings (10%) repeat an earlier listing's company, title and description; they're flagged as reposts and kept. 17% can't be checked, having no description.
- **Reposts come from job boards:** 32% of job-board listings are reposts, vs. 9% on company sites and 6% on hiring platforms, so job boards mostly re-list jobs posted elsewhere.
- **Most jobs are listed once** (8.9M), with a long tail: 612K listings are at least the fifth listing of their job.

### 6.5 Salary to one hourly rate

**Decisions, and why:**
- **The midpoint of the low and high values, converted to an hourly rate** (a year is 2,080 hours, 260 workdays, 52 weeks or 12 months), so hourly and salaried pay compare. Many career changers come from hourly jobs, so hourly reads naturally. Hourly listings keep their posted rate; converting salaried pay assumes full-time work.
- **Reversed ranges** (low above high) are swapped, and **clearly wrong units** are re-read: "$25 a year" is hourly pay, "$85,000 an hour" is yearly pay. Each is flagged `pay_suspect`.
- **A missing or unknown unit, or a result outside $4.81–$480.77 an hour** ($10k–$1M a year): only the salary is blanked, and counted.

**What the data shows**
- **43% disclose pay** (5.6M listings), all with a recognized time unit: hourly for 24% of all listings, yearly for 18%.
- **At full scale nothing needed fixing:** no reversed ranges or mislabelled units; 984 values outside $10K–$1M a year were blanked.
- **Median $29.08 an hour across every field,** with the middle half between $20 and $50.

### 6.6 Missing values

The share of postings missing each field we plan to use. It shows what each analysis can rely on. For example, the crawler-vs-employer check in 7.1 only uses postings that have `posted`.

**What the data shows**
- **The timeline, title, seniority and occupation are complete;** the description is missing for 17%, the country for 8.0% and the language for 4.8%.
- **The sparse fields are pay (57% missing), the employer's posting date (58%) and the named platform (60%,** since company sites aren't named). Analyses that use them say so.

### 6.7 Cleaning report and derived timing columns

**Decisions, and why:**
- **`live`: last seen within 2 days of September 30 (the data's end date), so still open then,** since the crawler doesn't revisit every listing daily. A live listing's lifespan is *censored*, which Q1b handles.
- **`days_open`** (first seen to last seen) and **`month`** (month first seen, the time horizon).
- **`jo`: originals only, for the EDA tables;** every cleaned listing, reposts flagged, is saved to disk and starts the scope table in section 8 (memory: one copy in RAM at a time).
- **Crawl artifacts, flagged and kept (6.8):** `batch_first_seen` (a company's listings first seen in one batch, when the company was added to the crawl) and `crawl_dropout` (a company's listings that vanished on one day, when the crawler likely lost the site). Same definitions as the hackathon app's signals; timing results in section 9 are shown with and without them.
- **`company_earlier`:** the company's listings in any field first seen before this one, counted before the cut, so company volume reflects all of a company's hiring.
- **Career-change flags:** `entry_seniority`, `low_experience_ask` (2 years or fewer) and `career_open` (either of those, or entry-level/training or career-changer wording): a deliberately broad first pass for Q1g and Q3. "Or equivalent experience" is left out, since it usually means experience in the same field.

**What the data shows**
- **Cleaning removed no listings.** It flagged 1.4M reposts, blanked 349K posting dates and 984 pay values.
- **11.7M original listings from 324K companies** feed the every-field comparison (section 7); 22.3% were still open on September 30.

### 6.8 Crawl artifacts (flagged, kept)

Two ways the crawler, not the employer, shapes the dates (measured on the hackathon app's corpus too):
- **Batch first sightings:** when PredictLeads adds a company to its crawl, every listing already on its site is "first seen" the same day. That's not a hiring surge, but it inflates company volume, weekly arrivals and freshness.
- **Crawl dropouts:** when the crawler loses a site, its listings "disappear" the same day. That's not jobs closing, and it cuts days open short.

Both are flags; nothing is removed. Section 9 shows the timing results with and without them.

**What the data shows**
- **Crawl artifacts are small but concentrated:** 2.4% of listings were first seen in a company's one-day batch (17% of companies with 5+ listings have one), and 1.2% disappeared in a one-day dropout (9.9% of those companies).
- **They distort single companies more than the totals,** so they're flagged, never dropped; 9.1 shows timing with and without them.

## 7. Q3: How do computer and mathematical roles compare with every other field?

**Every field, before the computer and mathematical filter (8c).** Before narrowing to the scope, one look at the whole market: every occupation group on the measures a career changer cares about, with computer and mathematical highlighted. Original listings, English only (the language rule of section 8; the stream already kept only US or no-country listings). Two things from every field are saved for later notebooks (7.2).

**Read it as context, not cause:** fields differ in company size, source mix and seniority, so "tech closes faster" can partly reflect who posts tech jobs. Like-for-like comparisons happen inside the scope (EDA_SDA_2 and 4).

### 7.1 Every field side by side

| Measure | What it tells a career changer |
|:--|:--|
| Share of all listings, new per week (full weeks) | How big the field is |
| Median hourly pay; pay disclosed | Whether a move pays, and how often pay is shown |
| Median days open (Kaplan–Meier); closed within 7 days | How fast to act |
| Entry-level seniority; asks 2 years or fewer; open to career changers (any sign) | How reachable the field is for a newcomer |
| Degree required; mentions remote (listings with a description) | Two common blockers or draws |

**What the data shows**
- **Computer and mathematical is the best-paid and most remote field:** a median **$66.73 an hour** vs. $26.44 for the median field, and 33% of listings mention remote work vs. 9.5%. It's the 4th largest (8.1% of English listings, about 34K new a week).
- **It's also among the hardest to enter:** 9.4% entry level (22nd of 23 fields), **17% of listings that state years ask for 2 or fewer (last of 23; median field 52%)**, and 21% show any sign of openness (22nd). The most open fields (food preparation, building and grounds, transportation) pay $18–22 an hour.
- **Its listings close at an average pace** (median 23.4 days, 14th), though 30% are gone within a week (4th). Worth the move, hard to enter: exactly where hiding what a newcomer can't get, and ranking the rest, helps.

### 7.2 Saved from every field, before the cut

Two summaries need every field, so they're saved here; everything after section 8 is the scope only.
- **Occupation summary:** pay, timing and openness per occupation (original English listings), for the compass's "your pay increase" (pay in the user's current field) and for comparisons. `data/clean/occupation_summary.pkl`.
- **Company activity:** each company's listings in any field and its newest listing date (all cleaned listings), for company volume and the "company still posting" ghost signal (EDA_SDA_2 3.1). `data/clean/company_activity.parquet`. Each listing's count of its company's earlier listings (`company_earlier`) is already a column (6.7a).

## 8. The filters: where the data narrows to the roles TripleTen trains for

**8b** applies language and country; **8c is the scope filter:** the step where 13.0M listings in every field become the roles TripleTen trains for, plus adjacent tech roles. Everything before 8c covers every field; everything after it is the scope only.

These decisions were made on the first 200,000 documents (and the scope on 2026-10-09/10), so they're applied before any further exploration: every later table describes the people we're building for. Nothing is deleted: the 13.0M cleaned rows stay on disk, and section 7 has already summarized every field.

| Decision | Choice | Why | Applied |
|:--|:--|:--|:--|
| **Language** | English only | The text features and the app are English | 8b |
| **No language label** | Drop | Can't be confirmed as English; almost all have no description (8a) | 8b |
| **Country** | United States only: listings that name the US, plus listings with **no country whose company is based in the US** (flagged `country_missing`); other no-country listings are dropped | Focus solely on US jobs without losing US remote roles, which often have no location (8a): every kept listing has evidence of being a US job. Decided 2026-10-10 (option B) | The stream (US or no country) and 8b |
| **Field: the scope** | **Roles TripleTen trains for, plus adjacent tech roles:** computer and mathematical (by occupation group or a 15- O*NET code), plus sales engineers (technical sales: TripleTen's Forward Deployed Engineering and AI Automation) and management and market research analysts (business and data analysis: TripleTen's Data Analytics) | Decided 2026-10-09 (computer and mathematical); on 2026-10-10 anchored to TripleTen's programs, a neutral outside definition rather than any team member's background (IT managers and financial analysts out: borderline); section 7 shows how the group compares | **8c** |
| **Credentialed math** | Out (actuaries, mathematicians) | Credential-gated, and no career switches into it in the profiles | **8c** |
| **Time horizon** | April–September by month; the data's first two days (late March) kept, flagged `partial_month` | Hide, never remove: they're left out of per-month counts only | 8b |
| **Seniority** | Keep every level, labelled | Manager-level listings are hidden per applicant (EDA_SDA_2), not here | – |
| **Follow-up window** | Keep each listing's full record | A long-open listing could be a ghost or a hard-to-fill role; EDA_SDA_2 separates them | – |
| **Additional cleaning** | In EDA_SDA_2 (2b) and EDA_SDA_3 | Staffing agencies, salary sanity, internships, multi-location listings, description cleanup: fix or flag, never drop | – |

### Prioritization factors (Q1)

| Factor | Choice | Reason | Status |
|:--|:--|:--|:--|
| Still open / apply-by | The model's **target**, not an input | It's what the ranking predicts from the other factors | Decided |
| Company hiring volume | Keep, counted from the company's listings in **any field** posted **before** each listing (`company_earlier`) | The strongest signal on the sample. High volume could mean **more competition or more openings to win**, so its direction is still open | Kept; interpretation open |
| Source / platform | Keep for now: `source_type` and `platform`; keep job-board rows | Whether source matters beyond company volume, whether platform gaps reflect crawler detection, and whether job-board listings duplicate company-site ones are checked in EDA_SDA_2 | Kept for now |
| Freshness (days since first seen) | Keep for now: `first_seen`, with `posted` alongside | How large the late-found tail is, and whether late-found listings close sooner, are checked in EDA_SDA_2 | Kept for now |
| Pay disclosed | Keep for now, alongside pay level | How much pay-transparency laws and hourly vs. salaried jobs drive it is checked in EDA_SDA_2; also lets pay level be used without dropping listings | Kept for now |
| Pay level | Keep | A direct measure of whether a listing is worth an application | Decided |
| Description length | Keep until EDA_SDA_3 | Length may not be intent | Kept for now |
| Urgency wording | **Left to EDA_SDA_3's NLP** (the column is saved, not analyzed here) | The keyword flag misfires ("urgent care") and showed no gap in days open on the sample; reading the wording in context can tell real urgency apart | Moved to EDA_SDA_3 |
| Remote mentions | Keep | Differs widely between fields and matters to applicants | Decided |
| Role bucket | Keep as its own feature (inside the scope the occupation group is one value) | The eight buckets a career changer chooses between; supports showing related roles in the app | Decided |
| Reposts | **Keep as a feature:** `is_repost`, `times_listed`, `days_since_first_listed`, `days_since_repost_last_listed` | A job posted again and again may be hard to fill. Built from earlier copies only, so known when a listing appears | Decided |
| Hiring difficulty | Keep: age so far, `times_listed`, `days_since_repost_last_listed` | A role open for long or relisted may be hard to fill; EDA_SDA_2 first separates real difficulty from ghost listings (done in 3.1: likely ghosts are hidden, other listings open 90+ days are labelled, and "possibly hard to fill" was dropped) | Kept for now |
| Career-change accessibility | Keep: `entry_seniority`, `years_required`, `low_experience_ask`, `entry_wording`, `changer_wording`, `degree_required`, `equivalent_experience`, `career_open` | Core to the project: hard blockers **hide** a listing; softer signals raise it. From seniority and listing text, so known when a listing appears (no leakage). Keyword rules, checked in EDA_SDA_3 | Decided; rules validated in EDA_SDA_3 |
| Recruiters | Keep: `recruiter_key` (anonymous), `recruiter_title` | A recruiter who fills roles fast or posts on a pattern is timing information; names and contacts are never kept | Kept for now (Q1h) |
| Transferable fit | Keep for later | Needs career-changer profiles (`Profiles.ipynb`) and O*NET skills | Kept, not yet applicable |

### 8.0 Why this scope: the roles TripleTen trains for, English, United States

| Choice | Why | Evidence |
|:--|:--|:--|
| **The roles TripleTen trains for, plus adjacent tech roles** (computer and mathematical, plus sales engineers (technical sales: TripleTen's Forward Deployed Engineering and AI Automation) and management and market research analysts (business and data analysis: TripleTen's Data Analytics)) | 0. **The scope's heading is TripleTen's program catalog** (tripleten.com, checked 2026-10-10: AI Software Engineering, AI & Machine Learning, Data Analytics, Quality Assurance, Cybersecurity, UX/UI Design, AI Automation, AI Systems Engineering; Forward Deployed Engineering and AI Cybersecurity coming soon). It's a neutral, outside definition: no role is in because of a team member's background. Each listing carries its program (`tripleten_program`); IT support, networks and systems, database and IT project management have no program but stay as **adjacent tech roles**, since IT support is the most open entry point and the profiles show real paths through them. Reasons 1–4 are for the core computer and mathematical group. 1. **It's the field we're breaking into ourselves:** the team are TripleTen students retraining into tech, so this was the field that interested us most and that we understand from the inside, and it fits the project brief (career changers moving into computer and mathematical roles). **The data then had to support it as a viable scope, and it does:** 2. **A switch pays off there:** higher pay, more remote work, and a large, steady flow of openings (section 7 ranks the field on each). 3. **Real paths in exist:** 2,813 career changers in the resume data moved into these roles (`Profiles.ipynb`), so the compass can pair the people who switched with the 2026 openings. 4. **It's hard to enter** (few entry-level listings), which is exactly where hiding what a newcomer can't get and ranking the rest helps | 8.0 table below; section 7 |
| **English only** | The text features (EDA_SDA_3), the matching models (ML_Fit) and the app read English. Listings with no language label almost never have a description, so there's nothing to read | 8.0 table below; 8a |
| **United States** | The project focuses solely on US jobs: pay is in dollars, job-growth projections (BLS) are for the US, and mixing in foreign jobs would confound pay and timing. The data is effectively US-only already (on the 200,000-row sample, every listing with a country was in the US). Listings with **no country** are kept only when their company is US-based, since they're more often remote (8a); the rest are dropped | 8.0 table below; 8a |

**One limit:** the stream (6.1) removed other languages and countries in one step, so at full scale the share it removed can't be split between "not English" and "not US"; on the sample, language accounted for nearly all of it.

**What the data shows**
- **The evidence supports the scope:** computer and mathematical ranks 1st of 23 fields on pay and on remote work, 4th on volume and 22nd on entry-level roles.
- **Language and country remove little:** 95% of cleaned listings are English; the 4.8% with no language label have essentially no description (623K of 624K); no listing outside the US remains after the stream.
- **8.0% have no country, and 93% of those come from a US-based company** (kept).

### 8a Checks before the filters

Two quick checks that can change which rows the scope keeps (every cleaned listing).
- **Missing country vs. remote:** remote listings often have no location. If the listings with no country are mostly remote, dropping them all would remove many US remote jobs, which is why 8b keeps those whose company is US-based.
- **Where the no-country listings' companies are based:** the share with a US-based company (kept), a company elsewhere, or a company with no location (both dropped).
- **Missing language vs. missing description:** PredictLeads probably detects language from the description. If the unlabeled listings are mostly the ones without a description, dropping them is really a "has description" filter.

**What the checks show**
- **Listings with no country are largely US remote jobs:** 30% mention remote work vs. 13% with a country, and they use a remote contract label more often (0.83% vs. 0.51%). 93% come from a US-based company and are kept; 2.6% from a company elsewhere and 4.3% from an unknown one are dropped.
- **No language label means no description:** 623K of the 624K unlabeled listings have none, so dropping them also removes about 28% of all listings without a description (623K of 2.2M).

### 8b Language and country

English only (listings with no language label dropped); United States only: listings that name the US, plus listings with no country whose company is based in the US (flagged `country_missing`); every other no-country listing is dropped. Counted on every cleaned listing, in every field.

### 8c The scope filter: the roles TripleTen trains for, plus adjacent tech roles

**This is the cut.** A listing is in the scope when:
1. its **occupation group** is *Computer and Mathematical*, **or** its **O*NET code starts with 15-** (the same family; this catches listings whose group is missing), **or** it's one of the **two TripleTen roles outside that group:** sales engineers (technical sales: TripleTen's Forward Deployed Engineering and AI Automation) and management and market research analysts (business and data analysis: TripleTen's Data Analytics); and
2. it isn't **credentialed math**: actuaries (15-2011) and mathematicians (15-2021), which need credentials a career changer doesn't have, and which no one in the career-switch profiles moved into.

Every other role is set aside here (it stays on disk, and section 7 already compared every field). The scope is split into **role buckets** by O*NET code (`src/career_switch.py`): the eight computer and mathematical buckets plus the two added ones. **Each listing also gets its TripleTen program** (`tripleten_program`): AI Software Engineering, UX/UI Design, AI & Machine Learning, Data Analytics, Quality Assurance, Cybersecurity, or Forward Deployed Engineering / AI Automation; listings no program trains for (IT support, networks and systems, database, IT project management, other computer roles) are **adjacent tech roles**. IT managers and financial analysts are out (decided 2026-10-10: borderline, a move-up role and a finance-credentialed one). The career-switch profiles (frozen) have paths into the eight only, so for the added buckets the compass shows 2026 openings, pay and timing without past switches. From here on, every table is the scope only.

**What the data shows**
- **The scope is 1.0M listings, 7.5% of the file:** computer and mathematical is 7.5% of English US listings, the two added TripleTen roles add 0.71%, and credentialed math is 0.05%. The O*NET-code fallback adds nothing (the group is always present), and every scope code maps to a role bucket.
- **Software development is the largest bucket (30%),** then networks and systems (18%), IT project management (12%) and data science (11%); technical sales is the smallest (1.0%).
- **By TripleTen program:** AI Software Engineering 29%, Data Analytics 14%, Cybersecurity 4.9%, AI & Machine Learning 4.8%, Quality Assurance 3.6%, UX/UI Design 1.3%, Forward Deployed Engineering / AI Automation 1.0%; **41% are adjacent tech roles.**

### 8d Fields only the scope carries

The stream kept **every original field for the scope listings** (`data/flat/job_openings_full/`), so the scope gets fields the 13.0M cleaned rows don't carry:
- **The full description** of every scope listing, saved next to the scope table in `data/clean/scope_descriptions.parquet` (id and description; EDA_SDA_3 reads it). Listings outside the scope keep only description counts and flags.
- **Recruiters:** an anonymous key per recruiter (`recruiter_key`: a fingerprint of name and company with a random salt made fresh each run and never saved, so names can't be recovered) and the recruiter's job title. Names and contact details are personal data and aren't kept.
- **Pay as written** (`salary_text`), parsed into hourly pay where `salary_data` has none (`pay_from_text`), with the same bounds as 6.5.
- **Location as written** (`location_text`), for EDA_SDA_2's no-country listings, the **url** (the app links to the listing) and the two record types (kept for completeness).

**What the data shows**
- **Every scope listing was found in the stream's full records;** 78% have a description (2.3 GB of text saved for EDA_SDA_3).
- **Pay as written exists for 45%,** and parsing it added hourly pay for 1.4% of listings (43% to 45%).
- **Recruiters are rare:** 1.2% of listings name one (stored as an anonymous key), 1.1% give their title; 90% carry a location as written.

## 9. Prioritization EDA inside the scope, by role bucket

**From here on: scope listings only** (the scope filter in 8c: the roles TripleTen trains for, plus adjacent tech roles).

Q1 measures every candidate factor across the **role buckets** (the eight computer and mathematical buckets, other computer roles, and the two added TripleTen buckets) and, in Q1a, by **TripleTen program**, Q2 checks which time information the data supports, and 9.3 shows the scope's mix. All use the scope's original listings (`so`); repost rates use every scope listing (`final`).

### 9.1 Q1: Which factors should drive prioritization?

A factor only helps ranking if it **varies**, so Q1a shows each factor across the scope, across the role buckets, and its spread (max ÷ min). Q1b–Q1h look closer.

| Measure | Prioritization factor | What it tells an applicant | Caveat |
|---|---|---|---|
| Median days open; closed within 7 days | Still open / apply-by | How long, and how urgent, the window is | Disappearing ≠ filled |
| New listings per week (full weeks) | Freshness, source | How often to check | Depends on crawl frequency |
| Still open on September 30 | Still open | How much is rankable now | One day only |
| Repost rate | Competition | Jobs that didn't fill | Could also mean a bad job |
| Pay disclosed, median pay | Pay | Whether it's worth applying | Under half disclose pay |
| Mentions remote | Employer intent | Flexible or not | Keyword flag (urgency wording is left to EDA_SDA_3's NLP) |
| From a hiring platform | Source | Where listings are hosted | From the url domain |
| Top 10 companies' share | Competition, source | Whether a short watch list covers most listings | |
| Entry-level seniority; asks 2 years or fewer; open to career changers | Career-change accessibility | Whether a newcomer can get in | Keyword rules; validated in EDA_SDA_3 |

**What the data shows**
- **Accessibility varies most between buckets:** entry-level seniority 10.8× (2.1% in IT project management to 23% in IT support), open to career changers 4.8× (9.2% to 44%), asks 2 years or fewer 4.7×.
- **Pay varies 2.4×** ($31.00 an hour in IT support to $73.23 in software development) **and volume 29×** (364 new a week in technical sales to 11K in software development), while timing varies only 1.8× (medians of 21–38 days).
- **By program:** Data Analytics roles are the most open (30% show any sign, 13% ask for 2 years or fewer) at $47.64 an hour; AI Software Engineering and AI & Machine Learning pay the most (about $73.50) but are among the least open (18–19%).

#### Q1b. Still open / apply-by

The share of listings still open N days after first being seen, using Kaplan–Meier survival curves. These treat still-open listings as unfinished rather than ignoring them, which would make windows look shorter than they are. This is the basis for an **apply-by estimate** on each listing.

**What the data shows**
- **The window is short:** 29% of scope listings close within a week, the median is 24.1 days, and 45% are still open after 30 days.
- **Most buckets move alike** (66–74% still open after a week). **Technical sales is the slow outlier** (83% after a week, median 38 days); database is the fastest (median 21 days).
- **Crawl artifacts barely move timing:** without them the median falls from 24.1 to 22.4 days, and the share open after a week stays at 70–71%.

#### Q1c. Source

Which hiring platforms, job boards or company sites host the scope's listings, and whether arrival rate and window differ by source. The crawl covers company sites and platforms, so job boards are under-represented.

**What the data shows**
- **65% of scope listings are on company sites or other domains,** 28% on hiring platforms and 7.1% on job boards.
- **Windows differ widely by source:** LinkedIn, Oracle and Taleo listings close in a median 15 days, company sites 21, Workday 32, Greenhouse 49, Lever 66 and Ashby 71.
- **The mix is similar in every bucket** (company sites 56–70%), so source separates listings, not roles. Platform choice tracks employer size, so source is tested alongside company volume.

#### Q1d. Competition: company hiring volume

Whether a few companies post most listings (a short watch list for an applicant), and whether listings from high-volume companies close faster or slower. **Volume counts a company's listings in any field** (section 7.2), since a bank posting 2,000 jobs, 50 of them in tech, is a high-volume employer. Bands: 1–4, 5–49, 50–499 and 500+ listings (rescaled for the full file: the sample's 20+ band held 89% of listings).

**What the data shows**
- **Company volume is the strongest timing signal:** counting only earlier listings in any field, scope listings at companies with 500+ listings close in a median **10 days, vs. 50 at companies with 1–4** (Gehan–Wilcoxon test, p < 0.001); over the whole six months, 14 vs. 56.
- **The gradient is steady across bands** (50, 35, 28 and 10 days) and holds for listings first seen from May (60 vs. 10 days).
- **High-volume employers dominate:** half of scope listings come from companies with 500+ listings in any field, and the top 10% of companies post 47–73% of each bucket's listings. Reposting rises with volume (1.2% to 7.6%).

#### Q1e. Competition: reposts

A repost is the same job (company, title and description) listed again under a new id. It suggests the first listing didn't fill, which could mean less competition, or a hard-to-fill or unattractive role. Here: how common reposts are by role bucket, and the gap between original and repost.

**What the data shows**
- **6.1% of scope listings are reposts:** most in technical sales (8.9%) and business and data analysis (8.1%), fewest in quality assurance (4.3%) and IT support (4.6%).
- **Reposts come back fast:** a median 7.6 days after the original, and a quarter within a day, so many are re-listings or crawler refreshes rather than a failed hire.

#### Q1f. Employer intent: text signals (the NLP link)

A first look at whether simple text features go with listings staying open longer or shorter. Each row compares listings with and without the feature. These are keyword flags, not models, and they misfire ("remote patient monitoring" counts as remote), so EDA_SDA_3 checks a sample of matches. Urgency wording is left out here: its keyword flag misfired ("urgent care") and showed no gap, so EDA_SDA_3 reads it with NLP instead. If any of them show a gap, richer text features (TF-IDF, topics, embeddings) are worth modeling as prioritization inputs.

**What the data shows**
- **Disclosed pay, long descriptions and hiring platforms go with listings staying open longer:** 79–82% still open after a week, vs. 64–67% without.
- **Remote wording shows no gap** (77% vs. 79%).
- **One feature at a time:** these overlap with company volume and source, so EDA_SDA_4's Cox model separates them.

#### Q1g. Career-change accessibility

How open each role bucket is to someone new to the field, and whether open listings behave differently. Text measures use listings with a description.

| Measure | What it tells a career changer | Caveat |
|---|---|---|
| Entry-level seniority | Whether there are roles a newcomer can start in | PredictLeads' label; its rules aren't documented |
| States years required; median years asked; asks 2 years or fewer | How much field experience listings expect | Lower end of each range |
| Entry-level or training wording; career-changer wording | Whether the employer trains or invites newcomers | Keyword rules |
| Degree required; "or equivalent experience" | Whether a missing degree rules the applicant out | One listing can say both |
| Open to career changers (any sign) | The first-pass summary flag | Broad on purpose |
| Median pay | Whether the role is worth the move | Under half disclose pay |

**What the data shows**
- **IT support is the entry door:** 23% entry level, 48% of listings that state years ask for 2 or fewer, and 44% show any sign of openness. Then other computer roles (37%) and business and data analysis (34%). IT project management is the hardest (2.1% entry level, 9.2% open).
- **Open listings pay less, but still well:** a median $37.98 an hour, vs. $64.14 across the scope ($26.00 for open IT support listings).
- **Listings open to career changers close a bit faster** (75% still open after a week vs. 80%). Degree requirements are rare (3.7%), but 60% of listings that state years ask for 5 or more.

#### Q1h. Recruiters

Whether a named recruiter changes how a listing behaves, and whether some recruiters post a lot. Recruiters are an anonymous key only (8d): no names or contacts are kept or shown, and only counts and shares appear here.

**What the data shows**
- **Only 1.2% of scope listings name a recruiter:** 5,527 anonymous recruiters; the 437 with 5+ listings post 37% of recruiter listings.
- **Recruiter listings close much faster:** a median 11.3 days vs. 24.3 without one; for recruiters with 5+ listings, **6.5 days** (48% still open after a week vs. 71%).
- **The titles point to staffing and talent-acquisition roles** (talent acquisition specialist, technical recruiter, account and delivery managers), so EDA_SDA_2 tests this against its staffing-agency flag before it's used.

### 9.2 Q2: What time information can prioritization rely on?

| Time scale | Use in prioritization | Needs |
|---|---|---|
| Days since a listing appeared | Apply-by estimates and freshness | Closed listings with both dates |
| Week / day of week | How often to check for new listings | Dates that reflect employers, not the crawler |
| Season / month | When to search | About two years of history |

Checks: how far back `first_seen` goes; whether `posted` and `first_seen` weekdays differ (crawler vs. employer); weekly arrivals over **full weeks only** (the first and last weeks are shown greyed and labelled partial). The old check that the file isn't sorted (which would have biased a first-N sample) isn't needed: every listing is used now.

**What the data shows**
- **Six months of steady history:** no seasonal advice, but 75% of scope listings closed by September 30, so timing is well observed. A median **36K new scope listings arrive per full week**; the quietest week is 0.76× the median and the busiest 1.21×.
- **`first_seen` is a good freshness clock:** it lags the employer's posting date by a median 1.5 days (75% within 3.5). Its weekdays follow the crawler (Friday peak), while posting dates (38% of listings) peak Tuesday to Thursday.
- **Still-open rules (merge decision 2):** our rule (last seen within 2 days) counts 25% of scope listings as open, and the app's signal rule (7 days, not closed) 26%, mostly the same listings (92% overlap). **The app's current listing filter (status not closed) counts 42%, and 27% of those were last seen 30+ days before September 30.**

### 9.3 The scope's mix: seniority, description length and other fields

Seniority and description length by role bucket, then the fields the project may use later: language, status values, contract types, what `tags` and `categories` hold, how many listings link to a company in the companies table, and whether the date picker is fair.

**What the data shows**
- **Mid/senior dominates every bucket** (75–91%). Junior is highest in IT support (23%) and other computer roles (21%); IT project management is 44% manager level.
- **Descriptions are long everywhere** (median 664–819 words; 79–88% over 400), so EDA_SDA_3's text work needs chunking for small embedding models.
- **Other fields:** contract types are messy (161 labels; full time 53%, hybrid 19%, contract 8.7%, none 30%), and categories are a clean 27-label taxonomy. Every scope company is in the companies table. **The date picker is fair from July:** live listings reach 79% of September's level, and the share first seen in the past 30 days settles near 50%.

## 10. Findings

**Q1, which factors drive prioritization** (inside the scope)

| Factor | Evidence | Verdict |
|:--|:--|:--|
| Company hiring volume (any field, earlier listings only) | Median 10 days at companies with 500+ listings vs. 50 at 1–4; steady across bands and from May on; Gehan–Wilcoxon p < 0.001 (Q1d) | **Strongest:** model it first, with staffing agencies separated (EDA_SDA_2) |
| Still open / apply-by | 29% gone within a week, median 24.1 days; technical sales slowest (38 days), database fastest (21) (Q1b) | **The target:** apply-by estimates by role bucket |
| Source | Medians from 15 days (LinkedIn, Oracle, Taleo) to 71 (Ashby); company sites 21 (Q1c) | **Strong:** model with company volume, which it tracks |
| Recruiters | Median 11.3 days with a named recruiter vs. 24.3 without; 6.5 days for recruiters with 5+ listings; only 1.2% of listings (Q1h) | **Promising but rare:** test against the staffing flag before using it |
| Pay | Disclosed for 45%; disclosure goes with longer windows (79% vs. 64% open at 7 days); $31–$73 an hour by bucket (Q1a, Q1f) | **Useful:** as a value factor, and disclosure as a timing signal |
| Career-change accessibility | Open to career changers from 9.2% (IT project management) to 44% (IT support); open listings pay a median $37.98 (Q1g) | **Core:** hides and raises listings; rules validated in EDA_SDA_3 |
| Employer intent (text) | Long descriptions 81% vs. 67% open at 7 days; remote wording no gap (Q1f); urgency left to EDA_SDA_3 | **Mixed:** test richer NLP in EDA_SDA_3 |
| Reposts | 6.1% of the scope; back a median 7.6 days later; 32% of job-board listings (6.4, Q1e) | **A feature,** read with the source: many are re-listings |
| Crawl artifacts | 2.4% batch first sightings, 1.2% dropouts; timing barely moves without them (6.8, Q1b check) | **Flags, not filters** |
| Transferable fit | Needs a career changer's profile | Profiles and O*NET skills, in ML_Fit |

**Q2, time information:** posting age is supported for every listing (`first_seen` lags the posting date by a median 1.5 days); day of week only through `posted` (38% of scope listings); no seasonality from six months; weekly rates use 25 full weeks. **Still-open rule:** ours (last seen within 2 days) and the app's signal rule (7 days, not closed) agree on most listings; the app's current status filter counts 27% of its "open" listings that were last seen 30+ days before September 30, which supports a last-seen rule for merge decision 2.

**Q3, how tech compares:** computer and mathematical is the best-paid (1st of 23) and most remote (1st) field, 4th in volume, but 22nd on entry-level roles and openness, and last on asking 2 years or fewer. The scope's design follows from it: hide what a newcomer can't get, rank the rest, and point to the open doors (IT support, business and data analysis, Data Analytics roles).

## 11. The scope table

Saved to `data/clean/analysis_table.parquet` for EDA_SDA_2: the scope's listings (reposts flagged), every column above plus the text and list columns added back from the stream's rows. **Descriptions:** the full text of every scope listing is in `data/clean/scope_descriptions.parquet` (8c), joined by `id`. 11d shows how complete the key fields are; 11e lists every column with its role, meaning and coverage.

**What the data shows**
- **Timing, role bucket, source and seniority are complete;** description 78%, pay 45%, years stated 43%, employer posting date 39%.
- **43% of listings have 8+ of the 10 key fields and 94% have 5+,** so ranking can count on the core fields for nearly every listing; pay, years and posting date add detail where present.

## 12. Simple SDA: summary statistics of the final table

A short statistical summary of the final table: its size, how its key measures are distributed, and how it splits by month, role bucket and source. Aggregates only.

**What the data shows**
- **The scope table: 1.0M listings from 84K companies,** 24% still open on September 30, 6.1% reposts, and an even spread across months (15–18% each).
- **Median 17 days open** (10% over 84), **median pay $63.90 an hour** (middle half $43.70–$84.40), and a median 747 words per description.
- **Career-change signals are thin:** 10% entry level, 7.7% ask for 2 years or fewer, 4.2% use entry-level or training wording and 0.5% invite career changers; 22% show any sign.

## 13. Final conclusion

### Summary

| | |
|:--|:--|
| **Data** | 13,415,388 listings in the file; 13,017,464 English (or unlabeled) and US (or no country); first seen March 30 to September 30, 2026 (the data's end date) |
| **After cleaning** | 13.0M listings, none removed; 1.4M reposts flagged (10%); 11.7M originals from 324K companies; 22.3% still open on September 30 |
| **Scope table** | 1.0M listings (947K originals) from 84K companies in 11 role buckets; 24% still open on September 30; 59% under a TripleTen program, 41% adjacent tech roles |

**Key findings:**
1. **Tech is worth it and hard to enter:** computer and mathematical is the best-paid field ($66.73 an hour vs. $26.44 for the median field) and the most remote (33% vs. 9.5%), but 22nd of 23 on entry-level roles and openness.
2. **Company volume is the strongest timing signal:** listings at companies with 500+ listings in any field close in a median 10 days, vs. 50 at companies with 1–4.
3. **The window is short:** 29% of scope listings are gone within a week (median 24 days); technical sales is the slowest bucket (38 days).
4. **The open doors:** IT support (44% show openness, 23% entry level) and Data Analytics roles (30%); AI Software Engineering and AI & Machine Learning roles pay the most (about $73.50 an hour) but are among the least open (18–19%).
5. **Still-open rules differ in practice:** a status-based filter counts 42% of scope listings as open, 27% of them last seen a month or more before September 30; a last-seen rule (ours: 25%) avoids that.

### Decisions
- **Order:** clean every listing; compare computer and mathematical roles with every other field (section 7); apply the filters we already know we want, first (section 8); explore the scope by role bucket (section 9).
- **Filters (section 8; the computer and mathematical cut is 8c):** English only (listings with no language label dropped); United States only: listings that name the US, plus no-country listings whose company is based in the US (flagged `country_missing`; other no-country listings dropped); the roles TripleTen trains for, plus adjacent tech roles: computer and mathematical by occupation group or a 15- O*NET code, plus sales engineers (technical sales: TripleTen's Forward Deployed Engineering and AI Automation) and management and market research analysts (business and data analysis: TripleTen's Data Analytics) (2026-10-10; IT managers and financial analysts out); credentialed math (actuaries, mathematicians) out; the data's first two days (late March) kept, flagged `partial_month`.
- **Kept from every field before the cut:** the occupation summary (pay and timing per occupation, for the compass's "your pay increase"), company activity (listings in any field, newest listing) and each listing's count of its company's earlier listings in any field.
- **Nothing capped, nothing removed in cleaning but data errors:** reposts become a feature; weekly rates leave out the partial first and last weeks, but the listings stay.

### Limits
- **Six months of history,** so no seasonal advice. `first_seen` follows the crawler, so day-of-week patterns need `posted`.
- **We see listings appear and disappear, not hires.** Openness to career changers is what listings *ask for*, not who gets hired.
- **The text flags are keyword rules** with likely false positives ("urgent care") until EDA_SDA_3 validates them.
- **The six-month window misses older listings:** listings first seen before 2026-03-30 aren't in the data, so the earliest dates undercount what was open; the app's date picker starts at July 1.
- **The field comparison (section 7) is descriptive:** fields differ in company size, source mix and seniority, so like-for-like comparisons happen inside the scope (EDA_SDA_2 and 4). The gaps here aren't significance tests; EDA_SDA_4 section 5 tests them.
- **High-volume posters include staffing agencies,** which can inflate company volume; EDA_SDA_2 (2b) flags them.

**Handoff:** the scope table (`analysis_table.parquet`: English US TripleTen-role and adjacent tech listings, April–September 2026 plus late March flagged `partial_month`, reposts flagged, every candidate factor, role buckets and company history in any field), its descriptions (`scope_descriptions.parquet`), the occupation summary and company activity (every field) go to **EDA_SDA_2**, which cleans further (2b), separates ghost listings, checks non-text factors by role bucket and builds the **compass's 2026 side**.
