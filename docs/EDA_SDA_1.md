<!-- Generated from notebooks/EDA_SDA_1.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 1 (Exploratory and Statistical Data Analysis)

PredictLeads × TripleTen externship · Jacob Colombo

**The project:** for a given career changer (anyone switching from their current background to a different role or field), suggest the computer and mathematical roles their skills transfer to (**role discovery**), hide the listings they can't get, and rank the rest by which to apply to first, and why. The **retraining compass** is the foundation: prior career → training → first tech job → that role's 2026 openings, pay, closing speed and expected growth. **This notebook** explores the data and decides where to go in depth: which factors, which time information, and which segment.

### Where this notebook fits

| # | Notebook | Question | Hands off |
|:-:|:--|:--|:--|
| **1** | **EDA_SDA_1 (you are here)** | **What's in the data, and what's the scope?** | **Final table: English US listings, April–September 2026, all groups and candidate factors** |
| 2 | EDA_SDA_2 | Which listings do we study? Additional cleaning, ghost listings, non-text checks, the niche (computer and mathematical, by role bucket), blockers hidden, the compass's 2026 side | Niche table, non-text features |
| 3 | EDA_SDA_3 | What does the text add? NLP features; validating the career-change flags | Niche table + text features |
| 4 | EDA_SDA_4 | Which features hold up? Statistics | Modeling-ready table |
| 5 | TimeSeries | When do listings appear and close? | Timing features |
| 6 | ML | Which roles fit a career changer, and how should listings be ranked? | The prioritization model |
| side | Profiles (frozen) | Who changed careers into or within tech or math, and how? (public resume dataset; three career-changer pools) | The compass's past side (path tables) and the modeling table for ML |

Each notebook hands its final table to the next. The TripleTen course templates are the baseline for every notebook, taken further with advanced methods.

---

## Executive summary

| | |
|:--|:--|
| **Data** | 200,000 listings from 51,006 companies, first seen 2026-03-30 to 2026-09-30 (PredictLeads snapshot of 2026-09-30) |
| **After cleaning** | 195,801 original listings (no rows removed; 4,199 reposts flagged), 42,399 still open |
| **Final table** | 182,367 English-language listings from 46,810 companies, April–September 2026 (US, plus flagged listings with no country); 41,825 still open |

### Key findings

| Finding | Evidence | What it means for a career changer |
|:--|:--|:--|
| **Company hiring volume is the strongest signal** | Listings at companies with 20+ openings close in a median **13 days** vs. **30** for one-off listings, counting only each company's earlier listings (the way the model sees it; 14 vs. 32 over the whole six months) | High-volume employers' listings need action first |
| **The window is short** | **28%** of listings are gone within a week; the median within about **22 days** | Apply before feeling fully qualified |
| **Source matters** | **15–19 days** (SmartRecruiters, LinkedIn, company sites) vs. **28–37** on several platforms | Source adjusts urgency |
| **Some listing traits carry signal; keyword flags don't** | Disclosed pay, long descriptions and hiring platforms: **11–16 points** more still open after a week; urgency and remote wording: **no gap** | Model the traits; test richer NLP than keyword flags |
| **The best-paid field is the hardest to enter** | Computer and mathematical: **$137.5k** median pay, **9%** junior, **77%** mid/senior | Target it, and hide what a career changer can't get |
| **How open listings are to career changers** | Entry level ranges from **3%** (management) to **70%** (food preparation); **22%** of computer and mathematical listings show any sign of openness | Its own ranking factor |

What separates a listing worth applying to today is mostly **the listing itself** (employer volume, source, pay disclosure), not the job title: those gaps (2.3–2.5×) are bigger than the gap between fields (1.7×), and all are known the moment a listing appears.

### Decisions
- **Scope:** English listings in the United States, first seen April–September 2026. Listings with no language label are dropped, since they can't be confirmed; listings with no country are kept, flagged, since they mention remote work twice as often (10a); EDA_SDA_2 keeps those with a US-based company or remote wording.
- **Nothing capped:** every listing keeps its full follow-up (ghost listing vs. struggling employer is Part 2's question), and reposts become a feature.
- **Target: computer and mathematical**, split into role buckets in EDA_SDA_2. Manager-level listings are hidden only for applicants without management experience; text blockers (years asked, licenses) only after EDA_SDA_3 validates them.
- **This table keeps every group, seniority level and candidate factor,** so Part 2 can compare before narrowing.

### Limits
- **Six months of history,** so no seasonal advice. `first_seen` follows the crawler, so day-of-week patterns need `posted` (41% of listings).
- **We see listings appear and disappear, not hires.** Openness to career changers is what listings *ask for*, not who gets hired.
- **The text flags are keyword rules** with likely false positives ("urgent care") until EDA_SDA_3 validates them.
- **The six-month window misses older listings:** listings first seen before 2026-03-30 aren't in the data, so the earliest dates undercount what was open; the app's date picker starts at July 1 (8b).
- **High-volume posters include staffing agencies,** which can inflate company volume; EDA_SDA_2 (2b) flags them.

Detailed findings: section 9. Every decision with its evidence: section 10.

---

### Work plan

**Prioritization factors (candidates).** Each is a proxy measured from the listing data; the EDA shows which carry usable signal. For a career changer, the first question is *can I get it*; the rest answer *is it worth applying now*.

| Factor | Question for the career changer | Measured from |
|---|---|---|
| **Career-change accessibility** | Is it realistically open to a newcomer? Hard blockers hide it; softer signals raise it | `seniority`; description: years required, degree, "or equivalent experience", entry-level and career-changer wording |
| **Still open / apply-by** | Is it still worth applying, and by when? | `first_seen_at`, `last_seen_at` (100% filled) |
| **Freshness** | Am I early? | Days since `first_seen_at` |
| **Competition** | How many others am I up against? | Reposts; listings per company |
| **Pay** | Is it worth the move? | `salary_data` (about 43%), annualized |
| **Employer intent** | In a hurry or flexible? | Description and title: urgency, remote, length |
| **Source** | Where is it posted, and how often should I check? | `url`: platform, job board or company site |
| **Transferable fit** *(later)* | Do my old field's skills match? | Listing text vs. the career changer's profile, with O*NET skills; needs profiles (`Profiles.ipynb`) |

**Clarifying questions**
- How complete and clean is the data, and how far back does it go?
- How long do listings stay open, and what separates fast from slow ones (company volume, source, pay, reposts, text)?
- How open are listings to career changers: entry level, years asked, degrees?
- What time information is reliable: posting dates, crawler dates, months, weekdays?
- Which scope (language, country, months, segment) has enough data for depth?

| # | Question | What it decides |
|---|---|---|
| Q1 | Which factors should drive prioritization for a career changer? | The factors that vary enough to change the ranking, including career-change accessibility (Q1g) |
| Q2 | What time information can prioritization rely on? | Apply-by estimates, check cadence and seasonal context: which are supported |
| Q3 | Which segment do we go in depth on? | The target is computer and mathematical (decided 2026-10-09); Q3 shows how much of it a career changer can reach |

**Data rules:** aggregates only (counts, shares, quantiles). The one exception is the final-table preview in 11d, tagged `data-rows`; clear outputs before committing.

### Notebook roadmap

1. **Setup** · 2. **Load raw data** · 3. **Inspect structure** · 4. **Flatten** · 5. **Schema**: each one call into `src/pl_data.py`
6. **Data cleaning:** types, invalid dates, duplicates and reposts, salary units, missing values, career-change flags
7. **Prioritization EDA:** Q1 factors (Q1g: career-change accessibility), Q2 time information, Q3 segment
8. **Other feasibility checks** · 9. **Findings** · 10. **Decisions, defended** · 11. **Final filters** · 12. **Summary statistics** · 13. **Conclusion**

## 1. Setup

Libraries and the data loader, `src/pl_data.py`: the S3 connection, cached raw documents, resumable flattening and the schema writer. Sections 2–5 each call one of its functions.

## 2. Load raw data

Both datasets are daily snapshots in S3, stored as compressed JSON lines (`.jsonl.zst`): **companies** (~252 MB compressed) are loaded in full; **job openings** (~19 GB) are too big for 8 GB of RAM, so we read the first 200,000 documents. Each is cached as Parquet in `data/`, so later runs take seconds instead of about 10 minutes. Once every table is flattened (section 4), this step skips the raw documents and S3 entirely.

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

### 6.1 Lean table and text cleaning

Reads the flattened job openings one part at a time and keeps only the columns the analysis needs, since the full table with descriptions doesn't fit in memory.

**Decisions, and why:**
- **Descriptions are cleaned** (HTML removed, entities decoded, whitespace collapsed), so word counts measure words, not markup.
- **Descriptions become features:** `has_desc`, `has_html`, `desc_words`, `mentions_remote`, `urgency`, and `desc_hash` (the repost fingerprint for 6.4). *Why:* memory; the full text stays in the flattened parts for NLP.
- **Career-change flags:** `years_required` (most years asked, lower end of each range; blank if none; above 20 ignored), `degree_required`, `equivalent_experience`, `entry_wording`, `changer_wording`. *Why:* what a listing asks of a newcomer is known the moment it appears. They're keyword rules, so EDA_SDA_3 checks them for false positives.
- **Without a description, text flags are unknown (blank), not False:** "we can't tell" isn't "it doesn't say so". `mentions_remote` is still True when the title says remote. Comparisons (Q1f, Q1g) set listings with a flag against listings *known* not to have it.
- **The `url` becomes its source:** a hiring platform, a job board, or the company's own site. Full URLs aren't needed.
- **Cached** in `data/clean/` and rebuilt only when an expected column is missing, since this step is slow.

**What the data shows**
- **200,000 listings from 51,006 companies.** 82.3% have a description, only 0.3% had HTML, and the median is 607 words, so later embedding work needs chunking.
- **Up to 60.7% of listings live on company sites** (strictly "company site or other": any domain not on our platform and job-board lists), then hiring platforms (29.0%) and job boards (10.3%, mostly LinkedIn). Most listings can link straight to the employer's own page.
- **Among listings with a description, 13.1% mention remote work and 11.4% use urgency wording.** `status` and `country` have a single value (6.2, Q3).
- **Career-change flags:** 26.4% of listings state years of experience required (median 3; middle half 1–5; 10% ask 8 or more). Among listings with a description, 10.0% use entry-level or training wording, 6.6% accept equivalent experience, 3.7% require a degree, and only 0.4% use career-changer wording.

### 6.2 Types and standard categories

**Decisions, and why:**
- **Dates become UTC timestamps,** so they can be compared. Text that won't parse becomes missing, and is counted: guessing would invent data.
- **Category labels are trimmed and lowercased,** and `contract_types` spellings are unified ("full_time" = "full time"), so one value counts once.
- **`title_norm`** (lowercased, whitespace collapsed) is created to match reposts.

**What the data shows**
- **Every date parsed.** `first_seen` spans exactly 2026-03-30 to 2026-09-30: a rolling six-month window.
- **`posted` has placeholders** (the earliest is 1969-12-30, blanked in 6.3), and its timestamps sit at midnight Central European time, so it's probably a local date (Q2c).
- **`status` is never "open"** (64.5% closed, 35.5% missing), so `last_seen` decides what's live (6.7). Seniority is 53% mid/senior and 30% junior; English is 92.4%.

### 6.3 Invalid dates

**Decisions, and why:**
- **Missing `first_seen` or `last_seen`, `last_seen` before `first_seen`, or dates after the snapshot: removed.** These are data errors, and such a listing can't be placed in time.
- **`posted` more than a day after `first_seen`, or before 2000: only `posted` is blanked.** The crawler can't find a listing before it exists, and 1969-12-30 is a placeholder; the listing itself stays.

**What the data shows**
- **No impossible timelines:** PredictLeads' timeline fields are clean.
- **5,454 `posted` values (2.7%) were blanked.**
- **Raw durations:** median 16.4 days; the maximum, 183.7 days, equals the six-month window. These treat still-open listings as ended; Q1b corrects that with survival curves.

### 6.4 Duplicates and reposts

**Decisions, and why:**
- **The same record `id` twice:** one copy kept (a data error).
- **Reposts** (same company, normalized title and description fingerprint, new `id`) **are kept and flagged** with `is_repost`; the earliest copy is the original. Features use **earlier copies only**, so they're known when a listing appears: `times_listed`, `days_since_first_listed`, `days_since_repost_last_listed`. Whether a listing *will* be reposted is future information and isn't used. Listings without a description can't be matched.
- **The EDA tables count each job once** (originals, `jo` in 6.7): a counting view, not a filter.

**What the data shows**
- **No duplicate ids, and 4,199 reposts (2.1%):** a lower bound, since 35,490 listings without a description can't be checked.
- **Most jobs are listed once,** but a long tail is relisted 10 or more times (1,268 listings are a job's 5th listing or later).
- **Job boards repost far more** (8.4%) than company sites (1.8%) or hiring platforms (0.5%).

### 6.5 Salary to one annual unit

**Decisions, and why:**
- **The midpoint of the low and high values, converted to a year** (2,080 hours, 260 workdays, 52 weeks or 12 months), so hourly and yearly pay compare. This assumes full-time work, which overstates part-time pay.
- **A missing or unknown unit, or a result outside $10k–$1M a year:** only the salary is blanked, and counted.

**What the data shows**
- **86,154 listings (43.1%) disclose pay,** all with a recognized unit: hourly is most common (48,215), then yearly (35,120).
- **Only 14 outliers** fell outside $10k–$1M a year.
- **The median midpoint is about $60.5k a year** (middle half $41.6k–$104k), with two humps: hourly and salaried jobs.

### 6.6 Missing values

The share of postings missing each field we plan to use. It shows what each analysis can rely on. For example, the crawler-vs-employer check in 7.1 only uses postings that have `posted`.

**What the data shows**
- **The timeline fields, title, seniority and occupation group are essentially complete,** so factors built on them cover every listing.
- **`posted` (58.9% missing) and salary (56.9%) are the weak fields;** `platform` is missing for 60.7% by design (company sites have none).
- **`country` is missing for 8.4%;** every listing that has one is in the United States.

### 6.7 Cleaning report and derived timing columns

**Decisions, and why:**
- **`live`: last seen within 2 days of the snapshot,** since the crawler doesn't revisit every listing daily. A live listing's lifespan is *censored*, which Q1b handles.
- **`days_open`** (first seen to last seen) and **`month`** (month first seen, the time horizon).
- **`jo`: originals only, for the EDA tables;** `jobs` keeps reposts, flagged, and starts the final table.
- **Career-change flags:** `entry_seniority`, `low_experience_ask` (2 years or fewer) and `career_open` (either of those, or entry-level/training or career-changer wording): a deliberately broad first pass for Q1g and Q3. "Or equivalent experience" is left out, since it usually means experience in the same field.

**What the data shows**
- **Cleaning removed almost nothing:** no rows were dropped, and 4,199 reposts were set aside, leaving **195,801 original listings** for the EDA.
- **42,399 listings (21.7%) are still open at the snapshot.** That's the pool a prioritized list would rank today.
- **The main fixes were to fields, not rows:** placeholder `posted` dates, salary units, and `contract_types` labels.

## 7. Prioritization EDA

Q1 measures every candidate factor (ending with career-change accessibility in Q1g), Q2 checks which time information the data supports, and Q3 compares segments. All use the cleaned originals (`jo`); repost rates use `jobs`.

### 7.1 Q1: Which factors should drive prioritization?

A factor only helps ranking if it **varies**, so Q1a shows each factor overall, across the largest occupation groups, and its spread (max ÷ min). Q1b–Q1g look closer.

| Measure | Prioritization factor | What it tells an applicant | Caveat |
|---|---|---|---|
| Median days open; closed within 7 days | Still open / apply-by | How long, and how urgent, the window is | Disappearing ≠ filled |
| New listings per week | Freshness, source | How often to check | Depends on crawl frequency |
| Still open at snapshot | Still open | How many listings are rankable now | One day only |
| Repost rate | Competition | Jobs that didn't fill | Could also mean a bad job |
| Salary disclosed, median salary | Pay | Whether it's worth applying | Under half disclose pay |
| Urgency wording, mentions remote | Employer intent | In a hurry or flexible | Keyword flags |
| From a hiring platform | Source | Where listings are hosted | From the url domain |
| Top 10 companies' share | Competition, source | Whether a short watch list covers most listings | Sample may over-represent some companies |
| Entry-level seniority; asks 2 years or fewer; open to career changers | Career-change accessibility | Whether a newcomer can get in | Keyword rules; validated in EDA_SDA_3 |

**What the data shows**
- **Biggest gaps between groups:** remote 15.5× (2% food preparation to 35% computer and mathematical, among listings with a description), reposts 10.3× (0.9% to 9.0%), pay 3.6× ($38k to $137.5k).
- **Moderate:** new listings per week 3.1×, urgency 2.2×, days open 1.7× (15.7 to 26.1). Salary disclosure and hiring-platform share barely differ between groups, but still separate listings within one (Q1c, Q1f).
- **So remote, reposts and pay change a lot by field,** and the model must be tuned per segment.
- **Career-change accessibility varies most of all:** entry-level seniority 20.7× (3% management to 70% food preparation), open to career changers 4.6× (17% to 79%), asks 2 years or fewer 2.3× (7% computer and mathematical to 15%).

#### Q1b. Still open / apply-by

The share of listings still open N days after first being seen, using Kaplan–Meier survival curves. These treat still-open listings as unfinished rather than ignoring them, which would make windows look shorter than they are. This is the basis for an **apply-by estimate** on each listing.

**What the data shows**
- **The window is short:** 28% of listings are gone within 7 days, 59% within 30 and 78% within 60; the median within about 22 days.
- **It differs by field:** transportation and computer and mathematical lose 34% in the first week; food preparation 22%.
- **For prioritization:** a week's delay costs a fifth to a third of the options, so apply-by estimates must be specific to the occupation group.

#### Q1c. Source

Which hiring platforms, job boards or company sites host listings, and whether arrival rate and window differ by source. The crawl covers company sites and platforms, so job boards are under-represented.

**What the data shows**
- **Median days open runs from 15 (SmartRecruiters), 17 (LinkedIn) and 19 (company sites) to 33–37** (Paylocity, Greenhouse, BambooHR).
- **Company sites close fastest:** 66% still open after a week vs. 82–89% on most platforms. The mix is similar in every group (company sites host 55–69%), so source separates listings, not fields.
- **Caveat:** platforms may show closures to the crawler differently, and platform choice tracks employer size (Q1d), so source is tested alongside company volume.

#### Q1d. Competition: company hiring volume

Whether a few companies post most listings (a short watch list for an applicant), and whether listings from high-volume companies close faster or slower.

**What the data shows**
- **The strongest single signal, and it holds:** listings from companies with 20+ openings stay open a median 14 days vs. 32 for one-off listings when volume counts all six months. Counting only each company's **earlier** listings (the way the model will see it), it's **13 vs. 30**, and 13 vs. 33 among listings first seen from May, with a month or more of history. Fewer listings land in the top band that way (26% vs. 42%), since a company's early listings count as low volume.
- **High-volume employers post 42% of all listings;** in every group the top 10% of companies post 39–60% (healthcare practitioners is the most concentrated).
- **Watching about 300–1,100 companies per group** (those with 5+ listings) covers a large share of openings. Reposts are more common there too (3% vs. 1%).

#### Q1e. Competition: reposts

A repost is the same job (company, title and description) listed again under a new id. It suggests the first listing didn't fill, which could mean less competition, or a hard-to-fill or unattractive role. Here we check how common reposts are and how long the gap is between original and repost.

**What the data shows**
- **Reposting is concentrated in hourly, high-turnover fields:** food preparation (9.0%), transportation (4.9%) and sales (3.1%). Most other occupation groups sit near 1%.
- **Reposts come back quickly:** a median of 16 days after the original, and a quarter within 6 days.
- **For prioritization:** a repost is a useful competition signal in those three fields. The employer didn't fill the role the first time. Elsewhere it's too rare to matter.

#### Q1f. Employer intent: text signals (the NLP link)

A first look at whether simple text features go with listings staying open longer or shorter. Each row compares listings with and without the feature. These are keyword flags, not models, and they misfire ("urgent care" counts as urgency, "remote patient monitoring" as remote), so EDA_SDA_3 checks a sample of matches. If any of them show a gap, richer text features (TF-IDF, topics, embeddings) are worth modeling as prioritization inputs.

**What the data shows**
- **Disclosed pay, long descriptions and hiring platforms go with listings staying open longer** (share still open after a week, with vs. without): hiring platform 83% vs. 67%, long description 81% vs. 69%, salary disclosed 78% vs. 67%.
- **Urgency and remote wording show no gap** once listings are compared only with listings *known* not to use them: urgency 79% vs. 79%, remote 76% vs. 79%. Earlier versions showed gaps (79% vs. 71%, 76% vs. 71%) because listings without a description, which close faster, were counted as "no".
- **So keyword flags alone carry little timing signal;** description length and richer NLP features (EDA_SDA_3) are still worth testing. One feature at a time, and they overlap with company size and source, so a multivariate model (Cox regression) must separate them.

#### Q1g. Career-change accessibility

How open each occupation group is to someone new to the field, and whether open listings behave differently. Text measures use listings with a description.

| Measure | What it tells a career changer | Caveat |
|---|---|---|
| Entry-level seniority | Whether there are roles a newcomer can start in | PredictLeads' label; its rules aren't documented |
| States years required; median years asked; asks 2 years or fewer | How much field experience listings expect | Lower end of each range |
| Entry-level or training wording; career-changer wording | Whether the employer trains or invites newcomers | Keyword rules |
| Degree required; "or equivalent experience" | Whether a missing degree rules the applicant out | One listing can say both |
| Open to career changers (any sign) | The first-pass summary flag | Broad on purpose |
| Median pay | Whether the field is worth the move | Under half disclose pay |

**What the data shows**
- **Fields differ widely.** Entry-level seniority runs from 3% (management) and 9% (computer and mathematical) to 70% (food preparation). Among listings that state years, 17% of computer and mathematical listings ask for 2 or fewer (median 5 years), vs. 74–80% in food preparation, healthcare support and healthcare practitioners.
- **Degrees are rarely required (4%),** and computer and mathematical listings accept equivalent experience most often (14%).
- **Listings open to career changers close a little faster:** 73% still open after a week vs. 80%. Computer and mathematical is low on openness (22% show any sign) but its open listings still pay a median $83k, vs. $44k for open listings overall.
- **Caveat:** keyword flags, one at a time. EDA_SDA_3 validates the rules and EDA_SDA_4 separates the effects.

### 7.2 Q2: What time information can prioritization rely on?

| Time scale | Use in prioritization | Needs |
|---|---|---|
| Days since a listing appeared | Apply-by estimates and freshness | Closed listings with both dates |
| Week / day of week | How often to check for new listings | Dates that reflect employers, not the crawler |
| Season / month | When to search | About two years of history |

Checks: how far back `first_seen` goes; whether the file is sorted (which would bias the first-200,000 sample); and whether `posted` and `first_seen` weekdays differ (crawler vs. employer).

**What the data shows**
- **Only six months of history,** so no seasonal advice. Weekly volume is steady at about 7,200 new listings (a dip in June), and the file isn't sorted by date or company, so the sample shows no ordering bias.
- **`first_seen` is a good freshness clock** (found a median 1.5 days after posting; 90% within 8 days), **but its weekdays and hours follow the crawler** (Friday peak). Day-of-week patterns must use `posted` (Central European dates, 41% of listings); hour of day isn't supported.
- **For prioritization:** posting age is fully supported; a day-of-week factor uses `posted`; seasonal context is out of reach.

### 7.3 Q3: Which segment do we go in depth on?

**Decided 2026-10-09: computer and mathematical (US),** split in EDA_SDA_2 into role buckets defined the same way as the career-switch profiles. Q3 still compares every large group, to show how much of the target a career changer can reach and what the alternatives were. Criteria: openness to newcomers (`entry_share`, `career_open_share`), pay, listings and companies (`top_company_share`), measurable timing (`closed_share`, `posted` coverage), variation in the Q1 factors, text length (Q3e), and monthly variation (`monthly_cv`, a lead only, since sampling affects it). The source backgrounds (the team's own) aren't in the listing data; EDA_SDA_2 pairs them with targets.

**What the data shows**
- **Country isn't a choice:** every listing with a country is in the US. **Every large group has enough data:** the top ten have 8,000–25,500 listings from 3,000–12,000 companies, and no company holds more than 2%.
- **Computer and mathematical** (15,688 listings, 7,152 companies) has the highest pay ($137.5k), the most remote listings (35% of those with a description), among the fastest closing (34% gone in a week) and the longest descriptions (median 749 words), **but only 9% junior and 77% mid/senior.** Alternatives: healthcare practitioners (largest and most concentrated, but many roles need a license; general knowledge, not measured) and sales or food preparation (junior-heavy, lower pay).
- **Decided:** computer and mathematical, hiding the listings a career changer can't get rather than switching fields. EDA_SDA_2 checks that enough remains.
- **Most open to career changers:** building and grounds cleaning (85% any sign, 77% entry level), food preparation (79%), transportation (78%) and personal care (72%), all with median pay under $47k. Computer and mathematical: 22% open, 9% entry level, $137.5k.

## 8. Other feasibility checks

Fields the project may use later: language (for an English-only corpus), status values, contract types, what `tags` and `categories` hold, and how many job openings link to a company in the companies table.

**What the data shows**
- **English covers 92.3%** of listings, and `status` holds only "closed" or nothing.
- **`contract_types` needed unifying** (6.2); its onsite, remote and hybrid labels give a second remote signal.
- **`tags` are too sparse to use as skills** (53% empty, 7,196 values), while **`categories`** is a clean 27-value taxonomy (99% filled). Every job opening's company is in the companies table.
- **The date picker is fair from July (8b):** live listings jump from 2,459 on April 1 and 23,884 on May 1 to 32,000–40,000 from June on. Their age mix settles from July, when about half were first seen in the past 30 days (June: 66%; April: all of them). Earlier dates are skewed toward young listings because the data starts on 2026-03-30.

## 9. Findings: where to narrow down

**Q1, which factors drive prioritization**

| Factor | Evidence | Verdict |
|---|---|---|
| Competition: company hiring volume | 14 vs. 32 median days open over the whole six months, 13 vs. 30 counting earlier listings only (20+ openings vs. one-off); these companies post 42% of listings | **Strongest factor:** holds with earlier listings only (13 vs. 30 days, Q1d check); model it first |
| Source | 15 (SmartRecruiters), 17 (LinkedIn), 19 (company sites) to 37 (BambooHR) median days; 66% vs. 82–89% open after a week | **Strong:** model it with company volume, since the two overlap |
| Still open / apply-by | 28% gone in 7 days, median about 22 days; 22–34% in the first week by group | **Strong:** the basis for apply-by estimates; group-specific |
| Employer intent (text) | Long descriptions go with longer windows (81% vs. 69% open at 7 days); urgency and remote wording show no gap | **Mixed:** test description length and richer NLP; keyword flags alone carry little |
| Pay | Disclosed for 43%; disclosure goes with longer windows (78% vs. 67%); medians $38k–$137.5k by group | **Useful:** as a value factor, and disclosure as a timing signal |
| Competition: reposts | 2.1% overall; 3–9% in food preparation, transportation and sales | **Segment-dependent** |
| Freshness | `first_seen` lags `posted` by a median 1.5 days | **Reliable** for every listing |
| Career-change accessibility | Entry level, years required, degree, training and career-changer wording (entry level 3–70% by field; 22% of computer and mathematical listings show any sign of openness). Only 9% of computer and mathematical listings are junior | **Core:** hard blockers hide a listing; softer signals raise it. Validated in EDA_SDA_3 before anything is hidden |
| Transferable fit | Needs a career changer's profile | Not testable here; uses `Profiles.ipynb` and O*NET skills later |

**Source and the app:** up to 60.7% of listings are on a company site (the share includes any domain not on our platform lists), so the app can link most ranked listings straight to the employer's page; EDA_SDA_2 (3.2) checks the domains.

**Q2, time information:** posting age is supported for every listing; day of week only through `posted` (41%, Central European dates); seasonal context isn't supported with six months of history.

**Q3, segment:** computer and mathematical in the US (decided 2026-10-09). Only 9% of its listings are junior, so EDA_SDA_2 hides the listings a career changer can't get, splits the rest into role buckets, and checks that enough remains.

## 10. Final analysis: defending our decisions

Every decision about the final table, defended with the evidence from sections 6–9. Nothing is filtered until section 11 applies them all at once. *Decided* is applied in section 11; *Applied in EDA_SDA_2* is decided here and applied there; *Kept for now* and *Open* await later checks.

| Decision | Options | Recommendation | Choice | Status |
|:--|:--|:--|:--|:--|
| **Language** | All languages; English only | English only (92.4% of listings) | English only | Decided |
| **Listings with no language label** (4.6%) | Keep; drop | Drop: we can't confirm they're English | Drop | Decided. Check 10a: 99.8% of these have no description, so this drop also removes about a quarter of the listings without descriptions |
| **Country** | United States is the only country present | United States | United States | Decided; listings without a country are handled in the next row |
| **Listings with no country** (8.4%) | Keep; drop; keep and place them with the company's location | Keep, flagged `country_missing`, and resolve in EDA_SDA_2 | Keep, flagged | **Applied in EDA_SDA_2** (decided 2026-10-09): kept if the company is US-based (companies table) or the listing mentions remote work or uses the remote contract type, flagged `country_inferred`; the rest are dropped. *Why:* check 10a shows they mention remote twice as often (28.3% vs. 12.6%), and remote roles matter to career changers (35% of computer and mathematical listings) |
| **Occupation groups** | One **main** group plus **related** groups (Q3) | Main: computer and mathematical (decided 2026-10-09). Related: the team's source-background groups, kept for role discovery and comparison | Computer and mathematical, plus the source-background groups | **Applied in EDA_SDA_2,** so Part 2 can still compare groups first. Inside it, role bucket (from `onet_code`) is a dimension, not a filter; only buckets too thin on both the listings and the profiles side are dropped |
| **Seniority** | All levels; hide manager level for everyone; hide it per applicant; entry level only | Keep all levels; hide manager-level listings per applicant | Keep all, labelled | **Decided (2026-10-09):** manager-level listings are hidden only for applicants without management or senior experience, since prior experience can qualify a career changer. Entry level only would drop mid-level listings that ask for little experience. Text blockers (years asked, licenses) hide nothing until EDA_SDA_3 validates them |
| **Time horizon** | Weekly; monthly | Monthly, April–September (drop the 2-day partial March) | Monthly | Decided |
| **Time scales** | Month; week; day of week; hour of day | Month, week and day of week | Month, week, day of week | Decided. Day of week uses `posted` (41%, Central European dates), since `first_seen` days follow the crawler. Hour of day isn't supported |
| **Follow-up window** | Drop listings open more than 50 days; cap at 50 days; keep the full record | Keep the full record | Keep the full record | Decided: a long-open listing could be a ghost listing (rank lower) **or** an employer still struggling to hire (rank higher). Age alone can't tell them apart; EDA_SDA_2 separates the two |
| **Prioritization factors (Q1)** | See below | | Keep all candidates for now | **Kept for now:** non-text questions are checked in EDA_SDA_2, text questions in EDA_SDA_3 |
| **Additional data cleaning** | | | Applied in EDA_SDA_2 | **Decided:** six fixes in EDA_SDA_2 section 2b (staffing agencies, salary sanity, date consistency, internships and temporary roles, title normalization, multi-location listings) and description cleanup in EDA_SDA_3; fix or flag, never drop |

### Prioritization factors (Q1)

| Factor | Choice | Reason | Status |
|:--|:--|:--|:--|
| Still open / apply-by | The model's **target**, not an input | It's what the ranking predicts from the other factors | Decided |
| Company hiring volume | Keep, counted from listings posted **before** each listing | Strongest signal (14 vs. 32 median days open). High volume could mean **more competition or more openings to win**, so its direction is still open | Kept; interpretation open |
| Source / platform | Keep for now: `source_type` and `platform`; keep job-board rows | Whether source matters beyond company volume, whether platform gaps reflect crawler detection, and whether job-board listings duplicate company-site ones are checked in EDA_SDA_2 | Kept for now |
| Freshness (days since first seen) | Keep for now: `first_seen`, with `posted` alongside | How large the late-found tail is, and whether late-found listings close sooner, are checked in EDA_SDA_2 | Kept for now |
| Pay disclosed | Keep for now, alongside pay level | How much pay-transparency laws and hourly vs. salaried jobs drive it is checked in EDA_SDA_2; also lets pay level be used without dropping listings | Kept for now |
| Pay level | Keep | A direct measure of whether a listing is worth an application | Decided |
| Description length; urgency wording | Keep until EDA_SDA_3 | Length may not be intent; urgent listings stay open longer, which needs explaining | Kept for now |
| Remote mentions | Keep | Differs widely between groups (2–35%) and matters to applicants | Decided |
| Occupation group | Keep as its own feature | Large differences between groups; supports showing related jobs in the app | Decided |
| Reposts | **Keep as a feature:** `is_repost`, `times_listed`, `days_since_first_listed`, `days_since_repost_last_listed` | A job posted again and again may be hard to fill. Built from earlier copies only, so known when a listing appears | Decided |
| Hiring difficulty | Keep: age so far, `times_listed`, `days_since_repost_last_listed` | A role open for long or relisted may be hard to fill; EDA_SDA_2 first separates real difficulty from ghost listings | Kept for now |
| Career-change accessibility | Keep: `entry_seniority`, `years_required`, `low_experience_ask`, `entry_wording`, `changer_wording`, `degree_required`, `equivalent_experience`, `career_open` | Core to the project: hard blockers **hide** a listing; softer signals raise it. From seniority and listing text, so known when a listing appears (no leakage). Keyword rules, checked in EDA_SDA_3 | Decided; rules validated in EDA_SDA_3 |
| Transferable fit | Keep for later | Needs career-changer profiles (`Profiles.ipynb`) and O*NET skills | Kept, not yet applicable |

**Data-quality fixes are not scope decisions:** no rows were removed in cleaning; 5,454 placeholder `posted` dates and 14 salary outliers were blanked, and labels were unified (cleaning report in 6.7).

### 10a Checks before the final filters

Two quick checks that can change which rows the final table keeps. Run them before section 11.
- **Missing country vs. remote:** remote listings often have no location. If the listings with no country are mostly remote, dropping them removes much of a feature we're keeping.
- **Missing language vs. missing description:** PredictLeads probably detects language from the description. If the unlabeled listings are mostly the ones without a description, dropping them is really a "has description" filter.

**What the checks show**
- **Listings without a country mention remote work twice as often.** Among listings with a description, 28.3% of the 16,778 without a country mention remote vs. 12.6% of those with one, and 11.5% vs. 7.7% use the "remote" contract type. (An earlier version showed 8.6% vs. 11.0%, because listings without a description were counted as "no".) **Decision:** keep them in this table, flagged `country_missing`. EDA_SDA_2 keeps those whose company is US-based or that mention remote work (`country_inferred`), and drops the rest.
- **Missing language almost always means no description:** 9,214 of the 9,228 (99.8%). So the language filter also removes about a quarter of the listings without descriptions (9,214 of 35,490), which is why descriptions rise from 82% to 87% in the final table. **Decision:** keep the drop, and record that it doubles as a partial "has a description" filter.

## 11. Final filters: building the EDA_SDA_1 final table

Applies every decision from section 10, all at once. Nothing earlier filters the data. The table is saved to `data/clean/` for EDA_SDA_2. 11d previews the first rows (clear before committing); 11e lists every column with its role and meaning.

## 12. Simple SDA: summary statistics of the final table

A short statistical summary of the final table: its size, how its key measures are distributed, and how it splits by month, occupation group and source. Aggregates only.

**What the data shows**
- **182,367 listings** (91.2% of the 200,000) from 46,810 companies. The filters removed 15,111 for language and 2,522 from March; listings with no country are kept, flagged. **41,825 (22.9%) are still open.**
- **A median 18.2 days open;** 10% stay open longer than 79 days: the long-open listings Part 2 examines.
- **Coverage:** 87.0% have a description and 45.0% a salary (median $60.7k; middle half $41.6k–$104k).
- **Mix:** about 35,000 listings in each of April and May, 25,500 in June, 27,400–30,300 in July–September; computer and mathematical is 7.7%; 58.7% company site or other; reposts 2.3%.
- **Career-change flags:** 30.1% of the final table is entry level, 12.7% asks for 2 years or fewer, and 47.3% shows any sign of being open to career changers. Among listings with a description, 10.0% use entry-level or training wording, 3.7% require a degree and 0.4% use career-changer wording.

## 13. Final conclusion

Part 1 loaded and cleaned 200,000 listings (no rows removed), measured every candidate prioritization factor, checked which time information holds up, and compared segments for career changers. The main result: what makes a listing urgent is mostly the listing itself (employer hiring volume, source, pay disclosure), not the job title, and the best-paid field, computer and mathematical, is the hardest to enter. Every decision is defended in section 10 and applied in section 11.

**Handoff:** the final table (182,367 English listings, April–September 2026, including flagged listings with no country, with every group, seniority level and candidate factor) is saved for **EDA_SDA_2**, which cleans it further (2b), narrows it to the computer and mathematical niche by role bucket, and builds the **compass's 2026 side** from it: each role's openings, pay and closing speed, mapped from `onet_code`. Findings at a glance are in the executive summary; details in section 9.
