<!-- Generated from notebooks/EDA_SDA_1.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# EDA/SDA 1: Field Comparison and Scope

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

Clean PredictLeads' full job-openings file (every listing first seen March 30 – September 30, 2026), compare fields, cut to the scope (**the roles TripleTen trains for, plus adjacent tech roles**) and explore it.

- **Q1 (section 3):** how do computer and mathematical roles compare with every other field?
- **Q2 (section 5):** which factors should drive prioritization for a career changer?
- **Q3 (section 6):** what time information can prioritization rely on?

Fix or flag, never drop, except data errors; aggregates only, never rows. **"Still open"** means last seen within 2 days of September 30.

**Hands off (section 7):** the scope table (every field) and its descriptions, the occupation summary, and company activity.

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
- [1. Load and build](#1-load-and-build)
- [2. Data cleaning](#2-data-cleaning)
- [3. Q1: How do computer and mathematical roles compare with every other field?](#3-q1-how-do-computer-and-mathematical-roles-compare-with-every-other-field)
- [4. The scope: where the data narrows to the roles TripleTen trains for](#4-the-scope-where-the-data-narrows-to-the-roles-tripleten-trains-for)
  - [4a Why this scope](#4a-why-this-scope)
  - [4b Language and country](#4b-language-and-country)
  - [4c The scope cut](#4c-the-scope-cut)
  - [4d Fields only the scope carries](#4d-fields-only-the-scope-carries)
- [5. Q2: Which factors should drive prioritization?](#5-q2-which-factors-should-drive-prioritization)
  - [5b Application window](#5b-application-window)
  - [5c Competition: company hiring volume](#5c-competition-company-hiring-volume)
  - [5d Source](#5d-source)
  - [5e Competition: reposts](#5e-competition-reposts)
  - [5f Recruiters](#5f-recruiters)
  - [5g Who is open to newcomers](#5g-who-is-open-to-newcomers)
  - [5h Do the flags go with how long listings stay open?](#5h-do-the-flags-go-with-how-long-listings-stay-open)
- [6. Q3: What time information can prioritization rely on?](#6-q3-what-time-information-can-prioritization-rely-on)
- [7. The scope table (hand-off to EDA_SDA_2)](#7-the-scope-table-hand-off-to-eda_sda_2)
- [8. Findings](#8-findings)
  - [Limits](#limits)

## 1. Load and build

1b–1d show the raw structure on a 200,000-document sample, then flatten it (`data/flat/`). 1e builds the full file (13.4M listings) from `data/raw/` (`python -m fetch.get_predictleads`) or S3 (`src/stream_full.py`). `BUILD_MODE`: **bulk** (about 35 minutes, 16 GB of RAM), **stream** (a few hours, for an 8 GB machine) or **auto**; both build the same tables, and finished steps are skipped.

## 2. Data cleaning

Full-file fixes (13.4M listings), data quality only: nothing is removed but data errors. The cleaning report (2f) lists every action.

- **Text** becomes counts and flags in the stream (`src/listing_rows.py`). Without a description a text flag is unknown, not False.
- **Dates:** impossible timelines are removed; implausible `posted` dates are blanked, and the listing stays.
- **Reposts** are kept and flagged, with features built from earlier copies only. Duplicate ids are dropped.
- **Pay** becomes one hourly rate; reversed ranges are swapped, wrong units re-read, and values outside $10k–$1M a year blanked.
- **Derived columns:** `live` (last seen within 2 days of September 30, so still open then: its lifespan is censored), `days_open`, `company_earlier` and the career-change flags.
- **Crawl artifacts** (batch first sightings, dropouts) are flagged and kept.

**What the data shows**
- **Cleaning removed no listings:** no duplicate ids, no impossible timelines, no reversed pay ranges or mislabelled units. It flagged 1.4M reposts (10%; 17% of listings can't be checked, having no description), blanked 349K employer posting dates (2.7%: placeholders before 2000, or dates after the crawler first saw the listing) and 984 pay values outside $10K–$1M a year.
- **11.7M original listings from 324K companies** feed the every-field comparison (section 3); 22.3% were still open on September 30.
- **Crawl artifacts are small but concentrated:** 2.4% of listings were first seen in a company's one-day batch (17% of companies with 5+ listings have one), and 1.2% disappeared in a one-day dropout (9.9% of those companies). They distort single companies more than the totals, so they're flagged, never dropped.

## 3. Q1: How do computer and mathematical roles compare with every other field?

Every occupation group before the scope cut, on original English listings; weekly rates use full weeks only. **Context, not cause:** fields differ in company size, source and seniority.

**What the data shows**
- **Computer and mathematical is the best-paid and most remote field:** a median **$66.73 an hour** vs. $26.44 for the median field, and 33% of listings mention remote work vs. 9.5%. It's the 4th largest (8.1% of English listings, about 34K new a week).
- **It's also among the hardest to enter:** 9.4% entry level (22nd of 23 fields), **17% of listings that state years ask for 2 or fewer (last of 23; median field 52%)**, and 21% show any sign of openness (22nd). The most open fields (food preparation, building and grounds, transportation) pay $18–22 an hour.
- **Its listings close at an average pace** (median 23.4 days, 14th), though 30% are gone within a week (4th). Worth the move, hard to enter: exactly where hiding what a newcomer can't get, and ranking the rest, helps.

**Saved for later notebooks:** the occupation summary (pay, timing and openness per occupation; the compass's "your pay increase"), `occupation_summary.pkl`, and company activity (each company's listings in any field and newest listing; the "company still posting" ghost signal), `company_activity.parquet`.

## 4. The scope: where the data narrows to the roles TripleTen trains for

Before the cut, every field; after it, the scope only. Nothing is deleted: the cleaned rows stay on disk.

| Decision | Choice | Why |
|:--|:--|:--|
| **Language** | English; listings with no language label dropped | The text features and the app are English; unlabeled listings have no description (4b) |
| **Country** | Listings that name the US; no-country listings dropped | Pay is in dollars and the job-growth projections (BLS) are for the US; only these are known US openings (4b) |
| **Scope** | The roles TripleTen trains for, plus adjacent tech roles; credentialed math out | TripleTen's program catalog, a neutral definition (4a) |
| **Time** | April–September; late March kept, flagged `partial_month` | Left out of per-month counts only |

Seniority, long-open listings and staffing agencies are labelled or hidden, never dropped.

### 4a Why this scope

- **TripleTen's program catalog** (tripleten.com, checked 2026-10-10) defines it: a neutral, outside definition. IT support, networks and systems, database and IT project management have no program but stay as **adjacent tech roles,** since IT support is the most open entry point.
- **The data supports it:** a switch pays off (section 3), 2,813 career changers in the résumé data moved into these roles (`Profiles.ipynb`), and they're hard to enter, which is where hiding and ranking help.

### 4b Language and country

What the two rules cost: no-country listings may be US remote jobs, and unlabeled listings may be the ones with no description.

**What the data shows**
- **Listings with no country (8.0% of cleaned listings) are largely US remote jobs:** 30% mention remote work vs. 13% with a country, and they use a remote contract label more often (0.83% vs. 0.51%). 93% come from a US-based company, 2.6% from a company elsewhere and 4.3% from an unknown one; all are dropped.
- **No language label means no description:** 623K of the 624K unlabeled listings have none, so dropping them also removes about 28% of all listings without a description (623K of 2.2M). 95% of cleaned listings are English, and no listing outside the US remains after the stream.

### 4c The scope cut

In the scope: occupation group *Computer and Mathematical* or an O*NET code starting 15-, or one of two TripleTen roles outside the group (sales engineers, market research analysts); minus credentialed math (actuaries, mathematicians). Each listing gets a **role bucket** and a **TripleTen program** (`src/career_switch.py`), or "adjacent tech role" where no program trains for it.

**What the data shows**
- **The scope is 908K listings, 6.8% of the file:** computer and mathematical is 7.2% of English US listings, the two added TripleTen roles add 0.69%, and credentialed math is 0.05%. The O*NET-code fallback adds nothing (the group is always present), and every scope code maps to a role bucket.
- **Software development is the largest bucket (29%),** then networks and systems (19%), IT project management (12%) and data science (12%); technical sales is the smallest (1.1%).
- **By TripleTen program:** AI Software Engineering 28%, Data Analytics 14%, Cybersecurity 5.0%, AI & Machine Learning 4.8%, Quality Assurance 3.7%, UX/UI Design 1.1%, Forward Deployed Engineering / AI Automation 1.1%; **42% are adjacent tech roles.**

### 4d Fields only the scope carries

From the stream's full records: the **full description** (`scope_descriptions.parquet`, for EDA_SDA_3), an **anonymous recruiter key** and title (names and contacts aren't kept), **pay as written** (parsed into hourly pay where missing), **location as written** and the **url**.

**What the data shows**
- **Every scope listing was found in the stream's full records;** the descriptions are saved for EDA_SDA_3 (2.3 GB of text).
- **Pay as written exists for 45%,** and parsing it added hourly pay for 1.4% of listings (43% to 45%). 90% carry a location as written.

**What the data shows**
- **The scope table is 908K listings from 72K companies.** Timing, role bucket, source and seniority are complete; description 78%, pay 45%, years stated 43%, employer posting date 39%.
- **Contract types are messy** (161 labels; full time 55%, hybrid 20%, contract 8.8%, none 27%). Every scope company is in the companies table.

## 5. Q2: Which factors should drive prioritization?

**Scope listings only** (originals, `so`; repost rates use every listing). A factor only helps ranking if it **varies**: 5a shows the main factors by role bucket and program, with each one's spread (max ÷ min); 5b–5h look closer at one family each.

**What the data shows**
- **Accessibility varies most between buckets:** entry-level seniority 10.8×, open to career changers 4.8×.
- **Pay varies 2.4×** ($31.10 an hour in IT support to $73.54 in software development) **and volume 27×** (340 new a week in technical sales to 9,200 in software development), while timing varies only 1.7× (medians of 22–38 days).
- **By program:** Data Analytics roles are the most open (30% show any sign) at $47.84 an hour; AI Software Engineering and AI & Machine Learning pay the most (about $74) but are among the least open (18–19%).

### 5b Application window

Share still open N days after first seen (Kaplan–Meier, so still-open listings count as unfinished): the basis for an **apply-by estimate**.

**What the data shows**
- **The window is short:** 29% of scope listings close within a week, the median is 24.2 days, and 45% are still open after 30 days.
- **Most buckets move alike** (67–74% still open after a week). **Technical sales is the slow outlier** (84% after a week, median 38 days); IT support, database, quality assurance and other computer roles are the fastest (median 22 days).
- **Crawl artifacts barely move timing:** without them the median falls from 24.2 to 23.1 days, and the share open after a week stays at 71%.

### 5c Competition: company hiring volume

Volume is the company's listings **in any field first seen before each listing** (`company_earlier`), as the model will see it.

**What the data shows**
- **Company volume is the strongest timing signal:** counting only earlier listings in any field, scope listings at companies with 500+ listings close in a median **12 days, vs. 44 at companies with 1–4** (Gehan–Wilcoxon test, p < 0.001).
- **The gradient is steady across bands** (44, 34, 28 and 12 days) and holds for listings first seen from May (56 vs. 12 days).
- **High-volume employers dominate:** the top 10% of companies post 47–74% of each bucket's listings, and reposting rises with volume.

### 5d Source

Hiring platform, job board or company site: share, arrivals and window. Job boards are under-represented in the crawl.

**What the data shows**
- **65% of scope listings are on company sites or other domains,** 28% on hiring platforms and 7.1% on job boards.
- **Windows differ widely by source:** LinkedIn, Oracle and Taleo listings close in a median 15 days, company sites 21, Workday 32, Greenhouse 49, Lever 66 and Ashby 71.
- **The mix is similar in every bucket** (company sites 56–70%), so source separates listings, not roles. Platform choice tracks employer size, so source is tested alongside company volume.

### 5e Competition: reposts

A repost is the same job listed again under a new id: how common, by bucket and source, and how soon after the original.

**What the data shows**
- **6.5% of scope listings are reposts:** most in technical sales (9.4%) and business and data analysis (8.9%), fewest in quality assurance (4.6%) and IT support (4.9%).
- **Reposts come back fast:** a median 7.6 days after the original, and a quarter within a day, so many are re-listings or crawler refreshes rather than a failed hire.

### 5f Recruiters

Anonymous keys only; counts and shares.

**What the data shows**
- **Only 1.3% of scope listings name a recruiter:** 5,527 anonymous recruiters; the 437 with 5+ listings post 37% of recruiter listings.
- **Recruiter listings close much faster:** a median 11.3 days vs. 24.5 without one; for recruiters with 5+ listings, **6.5 days** (48% still open after a week vs. 72%).
- **Many recruiter listings may come from staffing agencies,** so EDA_SDA_2 tests recruiters against its staffing-agency flag before they're used.

### 5g Who is open to newcomers

The text rules behind the career-change flags, among listings with a description.

**What the data shows**
- **IT support is the entry door:** 48% of its listings that state years ask for 2 or fewer, and 22% are entry level (5a). Then other computer roles (36% show any sign) and business and data analysis (33%). IT project management is the hardest (2.1% entry level, 9.1% open).
- **Open listings pay less, but still well:** a median $38.00 an hour, vs. $64.62 across the scope ($26.00 for open IT support listings). Degree requirements are rare (3.9%), but 61% of listings that state years ask for 5 or more.

### 5h Do the flags go with how long listings stay open?

Each row compares listings with and without a flag, among those where it's known. Keyword flags misfire, so EDA_SDA_3 checks them; urgency is left to its NLP.

**What the data shows**
- **Disclosed pay and long descriptions go with listings staying open longer** (about 80% still open after a week, vs. about 65% without). **Remote wording shows no gap** (77% vs. 79%).
- **Listings open to career changers close a bit faster** (75% still open after a week vs. 79%).
- **One feature at a time:** these overlap with company volume and source, so EDA_SDA_4's Cox model separates them.

## 6. Q3: What time information can prioritization rely on?

Six months of data rules out seasons. Checks: history and weekly arrivals (full weeks only), whether `first_seen` follows the employer or the crawler, which listings count as still open, and whether the date picker is fair.

**What the data shows**
- **Six months of steady history:** no seasonal advice, but 75% of scope listings closed by September 30 (25% were still open), so timing is well observed. A median **31K new scope listings arrive per full week**; the quietest week is 0.76× the median and the busiest 1.27×.
- **`first_seen` is a good freshness clock:** it lags the employer's posting date by a median 1.5 days (75% within 3.4). Its weekdays follow the crawler (Friday peak), while posting dates (41% of listings) peak Tuesday to Thursday.
- **Still-open rules (merge decision 2):** our rule (last seen within 2 days) counts 25% of scope listings as open, and the app's signal rule (7 days, not closed) 25%, mostly the same listings (94% overlap). **The app's current listing filter (status not closed) counts 39%, and 26% of those were last seen 30+ days before September 30.**
- **The date picker is fair from July:** live listings reach 80% of September's level, and the share first seen in the past 30 days settles near 50%.

## 7. The scope table (hand-off to EDA_SDA_2)

`data/clean/analysis_table.parquet`, with the full descriptions in `scope_descriptions.parquet` (joined by `id`); the dictionary lists every column.

## 8. Findings

| | |
|:--|:--|
| **Data** | 13,415,388 listings in the file; 13,017,464 English (or unlabeled) and US (or no country); first seen March 30 to September 30, 2026 (the data's end date) |
| **After cleaning** | 13.0M listings, none removed; 1.4M reposts flagged (10%); 11.7M originals from 324K companies; 22.3% still open on September 30 |
| **Scope table** | 908K listings (849K originals) from 72K companies in 11 role buckets; 25% still open on September 30; 58% under a TripleTen program, 42% adjacent tech roles |

**Q1, how tech compares:** computer and mathematical is the best-paid (1st of 23) and most remote (1st) field, 4th in volume, but 22nd on entry-level roles and openness, and last on asking 2 years or fewer. The scope's design follows from it: hide what a newcomer can't get, rank the rest, and point to the open doors (IT support, business and data analysis, Data Analytics roles).

**Q2, which factors drive prioritization** (inside the scope)

- **Company volume is the strongest timing signal:** a median 12 days at companies with 500+ listings, vs. 44 at 1–4 (5c).
- **The window is short,** and it's what the model predicts: 29% of listings are gone within a week, median 24 days (5b).
- **Source and recruiters matter:** 15 days on LinkedIn or Taleo vs. 71 on Ashby; 11 days with a named recruiter, though only 1.3% of listings name one (5d, 5f).
- **Pay and openness vary most between roles:** $31–$74 an hour, and openness from 9% (IT project management) to 44% (IT support) (5a, 5g).
- **Weak, or flags only:** remote wording shows no gap, reposts are mostly re-listings, and crawl artifacts barely move timing (5b, 5e, 5h).

**Q3, time information:** posting age works for every listing (`first_seen` lags the posting date by a median 1.5 days); day of week only through `posted` (41%); no seasonality from six months. The still-open rule should be last-seen based: the app's status filter keeps 26% of listings last seen 30+ days before September 30 (6c).

### Limits
- **Six months of history,** and listings first seen before March 30 are missing, so early dates undercount what was open (the date picker starts July 1).
- **We see listings, not hires:** openness is what listings ask for.
- **The text flags are unvalidated keyword rules,** so they stay labels and opt-in filters.
- **Section 3 is descriptive;** EDA_SDA_4 tests the gaps and confounders (staffing agencies are flagged in EDA_SDA_2).

**Handoff:** the scope table, its descriptions, the occupation summary and company activity go to **EDA_SDA_2**.
