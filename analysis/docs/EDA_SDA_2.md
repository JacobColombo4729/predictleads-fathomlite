<!-- Generated from notebooks/EDA_SDA_2.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# EDA/SDA 2: Scope Cleaning and Behavior

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

`EDA_SDA_1.ipynb` cleaned the full file, compared every field and cut the scope: **the roles TripleTen trains for, plus adjacent tech roles** (about 0.9M listings). This notebook does the **non-text** in-depth analysis of that scope: additional cleaning (staffing agencies, contract types, titles, the app's filter fields), ghost listings vs. employers still hiring, source, freshness, pay, reposts and copies, company facts, missing data, outliers, what a career changer can reach, and the **retraining compass's 2026 side** by role bucket and TripleTen program. It saves **its own final table** for `EDA_SDA_3.ipynb` (text), then `EDA_SDA_4.ipynb` (statistics), `TimeSeries`, `ML_Fit` and `ML_Rank`.

**Inputs from EDA_SDA_1:** the scope table (`analysis_table.parquet`: English, listings that name the US, the scope filter applied; every candidate factor; role buckets; `tripleten_program`; `company_earlier` in any field; crawl-artifact flags), and company activity in any field (`company_activity.parquet`).

**Conventions:**
- Every decision is recorded with its reasoning; fix or flag, never drop.
- Every factor is checked against the confounders (section 2): within groups, not by filtering.
- Aggregates only (shares, medians, compact counts); never rows, except the cell tagged `data-rows` (the 2b.1 review of flagged company names): clear it before committing.
- Each table or chart group ends with **What the data shows**: up to three bullets.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Product |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, and what's the scope? | The scope table |
| **EDA_SDA_2** | **Which scope listings do we keep, and how do they behave?** | **Cleaned and flagged scope table (ghosts, hiding rules)** |
| EDA_SDA_3 | What do the descriptions say? | Text features and O*NET skills |
| EDA_SDA_4 | Which features hold up? | `modeling_ready.parquet` (the features that survived) |
| TimeSeries | When do listings appear and close? | Timing patterns by month |
| Profiles | Who really changed into or within tech or math, and how? (frozen) | The 2,813 career changers |
| ML_Fit | Which roles does a career changer fit? | The trained layer (role fit), saved for the app |
| ML_Rank | Which listings should they apply to first? | Listing scores and the queue's weights, saved for the app |
| Evaluate | How well do the app's rankers do on real career moves? | The comparison: `rrf`, `mlrank`, combined |

### Clarifying questions
- Which long-open listings are ghost listings, and which are employers still trying to hire?
- Does source matter beyond company hiring volume, or is it company volume in disguise?
- How precise is freshness, given the crawler finds listings a median of 1.5 days late?
- Is pay disclosure driven by state pay-transparency laws or job type, rather than employer intent?
- Does high company volume mean more competition, or more openings to win? And are fast-closing recruiters just staffing agencies?
- How are values missing: at random, or tied to groups (source, role bucket, state)?
- How much of the scope is left once the reliable hiding rules (seniority, likely ghosts, non-tech occupations) are applied, and how much more would an opt-in text filter hide?
- Which role buckets have enough listings and enough career changers for their own analysis?

### Roadmap
1. **Setup:** EDA_SDA_1's scope table and company activity
2. **Analysis views and confounders;** **additional cleaning** (2b): staffing agencies and recruiters, dates, contract types and job type, titles (with an AI/ML flag), multi-location, the app's filter fields
3. **Checks and analyses** (non-text): the compass's 2026 side (3.12) and the **feature screening** (3.13)
4. **EDA_SDA_2 final table:** decisions defended, then applied, with a reason for each step
5. **Conclusion and next steps** (into EDA_SDA_3)

## Contents

- [Purpose](#purpose)
  - [Clarifying questions](#clarifying-questions)
  - [Roadmap](#roadmap)
- [2. Analysis views](#2-analysis-views)
  - [Confounders: compare within groups, don't filter](#confounders-compare-within-groups-dont-filter)
- [2b. Additional cleaning](#2b-additional-cleaning)
- [3. Checks and analyses](#3-checks-and-analyses)
  - [3.1 Ghost listings vs. desperate employers](#31-ghost-listings-vs-desperate-employers)
  - [3.2 Source / platform](#32-source-platform)
  - [3.3 Freshness and crawler lag](#33-freshness-and-crawler-lag)
  - [3.4 Pay disclosed](#34-pay-disclosed)
  - [3.5 Reposts and company hiring volume](#35-reposts-and-company-hiring-volume)
  - [3.6 The scope: the roles TripleTen trains for, by role bucket](#36-the-scope-the-roles-tripleten-trains-for-by-role-bucket)
  - [3.7 Blind spots from Part 1](#37-blind-spots-from-part-1)
  - [3.8 Joining the companies table](#38-joining-the-companies-table)
  - [3.9 Missing-data patterns](#39-missing-data-patterns)
  - [3.10 Outliers](#310-outliers)
  - [3.11 Career-change accessibility, and hiding what a career changer can't get](#311-career-change-accessibility-and-hiding-what-a-career-changer-cant-get)
  - [3.12 The retraining compass: the 2026 side](#312-the-retraining-compass-the-2026-side)
  - [3.13 Feature screening: which candidates go forward](#313-feature-screening-which-candidates-go-forward)
- [4. EDA_SDA_2 final table](#4-eda_sda_2-final-table)
  - [4.0 Decisions](#40-decisions)
  - [4.0 Prioritization factors, after this notebook](#40-prioritization-factors-after-this-notebook)
- [5. Conclusion and next steps](#5-conclusion-and-next-steps)
  - [Answers to the clarifying questions](#answers-to-the-clarifying-questions)
  - [For the career changer](#for-the-career-changer)
  - [Next steps](#next-steps)

**What the data shows**
- **The scope from EDA_SDA_1: 908K listings (849K originals) from 72K companies;** 24.7% still open on September 30.
- **57.8% are under a TripleTen program;** the rest are adjacent tech roles.

## 2. Analysis views

Some analyses are skewed by part of the table. Each analysis states which **view** it uses:

- **All listings:** everything, reposts included — ranking as an applicant sees it; the repost features.
- **Originals only:** reposts excluded — counts, volumes, windows, company concentration (a repost would double-count).
- **With `posted`:** about 42% of listings — day-of-week patterns, crawler lag.
- **With a salary:** about 48% of listings — pay level and comparisons.
- **Closed listings:** not open on September 30 — simple descriptive durations (survival methods use every listing instead).
- **Long-open listings:** open a long time (3.1's threshold) — ghost vs. desperate-employer checks.
- **Without likely ghosts:** excludes `likely_ghost` (3.1d) — every timing result shown with and without them, from 3.1 on.
- **Open to career changers:** `career_open`, or 3.11's stricter rule — accessibility comparisons, in and outside the scope.
- **The scope:** EDA_SDA_1's scope table, by role bucket (3.6) and program — everything in this notebook.
- **Other fields:** EDA_SDA_1's section 7 and occupation summary — comparing the scope with other fields.

### Confounders: compare within groups, don't filter

Some factors travel together, so a raw gap can belong to something else. **Every factor test in section 3 is checked against these,** by comparing within groups (stratifying), not by filtering them out.

- **Company hiring volume** (bands 1–4 / 5–49 / 50–499 / 500+): the strongest signal — every factor compared within bands.
- **Source and platform:** the crawler sees closures differently by platform — checked within source and platform.
- **Staffing agencies** (`is_staffing`): post at volume, repost often — shown with and without them.
- **Role bucket and seniority:** different roles close at different speeds for unrelated reasons — checked within bucket (3.6) and by seniority.
- **Month first seen:** later months had less time to close — survival methods censor still-open listings; month as a split.
- **State** (3.7d): pay-transparency laws vary pay disclosure — checked within state.
- **Company type** (`is_public`, `has_parent`): public companies stay open longer at high volume — shown separately.
- **No description** (`has_desc`, 3.9): closes much faster — shown with and without them.
- **Ghost listings** (`likely_ghost`, 3.1d): inflate days open — timing shown without them too.

**The rule:** each factor shows the raw gap, then the gap within volume band × source type, and within role bucket where there are enough listings. EDA_SDA_4 then confirms the survivors with one model that adjusts for all of these at once.

## 2b. Additional cleaning

**Decided (2026-10-09):** five fixes and the app's filter fields here (salary sanity moved into EDA_SDA_1 section 2, description cleanup to EDA_SDA_3), run before the checks in section 3 so they use clean pay, dates and titles. Same rules as EDA_SDA_1: **fix or flag, never drop**; each step's own cell reports how many listings it changed.

- **Staffing agencies:** flagged by company name or domain words, an "about" text, or (added 2026-10-11) agency-style recruiter titles on 3+ listings — a label and opt-in filter, never hidden by default (an unvalidated text rule, decided 2026-10-13).
- **Salary sanity, standardized to hourly** — moved to EDA_SDA_1 section 2: one hourly pay column (`salary_hourly_usd`) for the whole project, bad units and reversed ranges fixed and flagged (`pay_suspect`) there.
- **Date consistency:** EDA_SDA_1 (section 2) already fixed impossible timelines; here, `posted` more than a year before `first_seen` is flagged (`posted_old`), feeding the ghost-listing check (3.1).
- **Contract types, internships and temporary roles:** contract labels (unified by EDA_SDA_1) plus title words split into employment-type flags and one `job_type` label for the app's filter.
- **Title normalization:** lowercased; brackets, locations, requisition IDs and pay stripped, remote/hybrid tags kept as a flag (`title_clean`); `ai_ml_title` flags AI, machine learning or data-science titles.
- **Multi-location listings:** `n_locations` above 1 → one row per listing for timing; the state list feeds the app's location filter.
- **Filter fields for the app:** role level, work arrangement and schedule, built from `seniority`, contract labels and title words.
- **Description cleanup** — moved to EDA_SDA_3, which loads the text directly; this table keeps only description features (`desc_words`, `desc_hash`, `has_html`) to fit in memory.

**Leakage:** every fix uses only what the listing showed when it was posted, so none of them leaks the outcome.

**What the data shows**
- **Staffing agencies are a small, fast group:** 2.0% of listings (753 companies), closing in a median 12.6 days vs. 24.0 for employers. **The new recruiter-title rule adds 33 companies:** agency-style titles (account, delivery or client manager) close in 6.2 days and come from name-flagged agencies about twice as often as corporate recruiter titles (13% vs. 6.8%). It doubles the share of busy recruiters' listings flagged (10% → 21%) but barely moves the total (1.9% → 2.0%); a first-pass keyword rule, not hand-checked (decided 2026-10-13): a label ("staffing agency, from the title") and an opt-in filter, not something that hides or demotes on its own. **Recruiters aren't just staffing agencies:** among employers, recruiters with 5+ listings still close in 8.1 days vs. 24.1 with none named.
- **Contract labels:** 55% full time, 9.7% contract, 3.8% part time, 3.3% internship, 18% none. Full and part time together is rare (1.2%). Where the work happens is stated for 54% (remote 17%, onsite 15%, hybrid 13%, several 8.3%), and remote labels agree with the description 94% of the time.
- **Titles and levels:** cleaning changed 25% of titles; **9.0% name AI, machine learning or data science** (25% of data science, 14% of networks and systems). Role levels: mid_senior 77% (kept whole; 40% of it has a senior title, flagged `senior_title`), entry 10%, manager 9.4%, leadership 3.7%. No listing names more than one location: PredictLeads lists each location separately.

## 3. Checks and analyses

Record each result and the decision it leads to here, then apply it when building the final table (section 4). **From 3.1 on, every timing result is also shown without likely ghosts** (section 2's views). **Every factor test follows the confounders rule in section 2:** raw gap, then the gap within volume band × source type (and role bucket where possible); only gaps that hold count.

**3.13 closes the section with the feature screening:** every candidate feature through the same checks, one keep/drop table.

### 3.1 Ghost listings vs. desperate employers

A long-open listing may be a **ghost listing** (forgotten or no longer monitored: rank lower or flag it) or a **desperate employer** (still actively hiring: rank higher). Age alone can't tell them apart.

| Signal | Points to "still hiring" | Points to "ghost listing" |
|:--|:--|:--|
| Relisted (`times_listed`, `days_since_repost_last_listed`) | Relisted recently | Never relisted in months |
| Company still posting new listings | The company is actively recruiting | The company stopped posting long ago |
| Source | Hiring platforms often auto-expire listings, so a long-open one was likely kept on purpose | Company career pages can host forgotten listings indefinitely |
| `posted` vs. `first_seen` | Recently refreshed | Very old posting date |
| `last_processed_at` (meaning to confirm with PredictLeads) | May record content changes | |

**Decided (2026-10-10): hide, never remove.** **Likely ghosts** (older than **90 days** on the date being viewed and from a company gone quiet, 3.1d) are hidden in the app (a note says how many, and why); other listings older than 90 days stay visible, labelled "open 90+ days" and ranked lower; every listing stays in the analysis and the model, with still-open listings censored as usual. Age on a date is known at that moment, so hiding by it uses no future information. Removing listings by how long they *ended up* open would bias the timing model (it would never see a long window) and throw away struggling employers, who may be the most open to a career changer.

Steps: describe the long-open listings (how many, which occupation groups and sources); **check the 90-day threshold** (do closing rates level off past 90 days? are those listings relisted, or did their company stop posting?) and keep it or adjust it with that evidence; measure each signal; combine them into a **"likely ghost listing" flag**, used to rank lower or label, never to remove. Only listings first seen before July can reach 90 days in this six-month window. Evergreen wording ("always accepting applications") is added in EDA_SDA_3.

#### 3.1e The career-changer view

The app ranks only what a career changer can see, so the key results are also shown on that view: **listings visible to a newcomer without management experience** (no manager level or above, no likely ghost, no hidden occupation). Senior-titled listings stay in it, flagged (`senior_title`): **decided 2026-10-11, the app shows them ranked lower with a "senior" label rather than hiding them** (81% of them ask for 5+ years and 4.9% show openness, vs. 43% and 17% for other mid_senior listings, 3.7c; but a title is a rough signal, and some career changers have senior experience). A third column shows the view without them. The feature screening (3.13) uses this view first.

**Result:** the view keeps 86% of the scope's original listings and looks like the whole scope (25% still open, a median 24.1 days, 29% closed within a week), paying a little less ($61.92 vs. $64.62 an hour). Without senior titles it keeps 55%: lower pay ($50.67), more entry level (17%) and more openness to career changers (34% vs. 22%), with similar timing.

**What the data shows**
- **Closing slows with age:** of listings first seen in April–May, 53% close in their first 30 days; past 60 days, 30–36% of those still open close each month. Old listings still close, so age alone isn't a ghost.
- **19% of open listings are older than 90 days** (40K of 216K); open job-board listings are almost all old (95%), since job boards stop after May (3.2c).
- **7.5K likely ghosts** (0.9% of the scope, 3.5% of open listings): stale and from a company with no new listing in 60 days, or with a posting date over a year old. The other stale listings are no more open to newcomers (16% vs. 20% for younger open listings), so "possibly hard to fill" is dropped as a label.

### 3.2 Source / platform

| Check | If the result is… | Then… |
|:--|:--|:--|
| Median days open by source, *within* each company-volume band | Source gaps hold within every band | Keep source as its own factor |
| | Gaps mostly disappear | Drop it, or keep it as a minor factor |
| Share of job-board listings whose company + title also appears on a company site or platform | High (copies) | Drop job-board rows to avoid double-counting |
| | Low | Keep them |
| Platform sizes in the chosen groups | Only a few platforms have 1,000+ listings | Name those platforms; group the rest as "other platform" |

**Also (added 2026-10-10):** why every open job-board listing is over 90 days old (3.1b); the domain check compares each listing's web domain with its company's domain (loaded for the scope only, never printed).

**What the data shows**
- **Source matters beyond company volume, and the two interact:** at companies with 500+ listings, company-site listings close in a median 8 days vs. 22 on hiring platforms; at companies with 1–4, 64 vs. 47. Job boards sit at 12–18 days in every band.
- **Platforms differ widely:** Built In, Indeed, SmartRecruiters, Taleo, Oracle and LinkedIn close in 10–16 days; Greenhouse, Breezy, Lever and Ashby in 49–71, partly in how they show closures to the crawler.
- **Job boards stop after May** (about 30K listings a month in April–May, about 200 a month after): a collection change to ask PredictLeads about. **36.0% of all scope listings are on the employer's own domain,** so the app can link many straight to the employer.

### 3.3 Freshness and crawler lag

| Check | If the result is… | Then… |
|:--|:--|:--|
| Share of listings found more than 7, 14 and 30 days after `posted` | Tiny | Use `first_seen` only |
| | Meaningful | Add a "found late" flag |
| Survival curves: found late vs. found promptly | Found-late listings close much sooner | The flag matters for apply-by estimates |
| Survival curves: listings with vs. without a `posted` date | Similar | `posted` can be used where present |
| | Different | Stick with `first_seen` only |

**Also (added 2026-10-10): are high-volume companies crawled more often?** The crawler's visit times aren't in the data, so crawler lag (posting date → first seen) stands in for crawl frequency, and the company-volume effect from 3.5a (12 vs. 44 days) is re-measured from the employer's posting date. If it shrinks a lot, part of it was crawl timing.

**What the data shows**
- **The crawler is prompt:** a median 1.5 days from the employer's posting date to first seen (1.1 on hiring platforms, 3.2 on job boards); 34% found within a day.
- **Busy companies are crawled more often** (company sites: a median lag of 9.8 days at 1–4 listings vs. 1.2 at 500+), which inflates their speed. **Like-for-like, measured from the posting date, the volume effect is about 1.5×** (31 vs. 22 days), not the nearly 4× seen from first seen.
- **Late-found listings** (7+ days after posting) have long windows from the posting date (47 days), so freshness uses `first_seen` with `posted` alongside.

### 3.4 Pay disclosed

| Check | If the result is… | Then… |
|:--|:--|:--|
| Disclosure rate by US state (needs the state from `location_data`) | Much higher in pay-transparency states (CA, CO, NY, WA) | Add state, or a "pay-law state" flag |
| Disclosure by hourly vs. salaried, and by seniority | Strongly tied to hourly | Its effect may really be job type |
| Days open by disclosure, within company-volume bands | The gap holds | Keep as a factor |
| | The gap disappears | Keep only as the companion flag for pay level |

**As built (2026-10-10):** state comes later (3.7, from `location_data`), so 3.4 uses role level, source, company volume and role bucket. Disclosure is compared within company-volume bands and without likely ghosts (the confounders rule).

**What the data shows**
- **Disclosure tracks source more than role level:** hiring platforms disclose pay for 53–64% of listings, company sites 36–52%, job boards 12–16%; by level, 47–56%.
- **Disclosed listings stay open longer at busy companies** (22 vs. 6 days at 500+ listings); much of the "not disclosed" group is listings with no description, which close in days (3.9).
- **Pay by bucket:** software development and IT project management lead (median $72–74 an hour); IT support is lowest ($31), business and data analysis $47.

### 3.5 Reposts and company hiring volume

- **Cross-posted vs. reposted (added 2026-10-10).** EDA_SDA_1 counts every later copy of the same job (same company, title and description) as a repost, so a copy on a second board while the first is still up looks the same as a job relisted after it closed. Split them, using each listing's earlier copies only, so nothing comes from the future:
  - **`is_crosspost`:** an earlier copy is still live when this one appears, from a different source (the platform, or "company site" or "job board" when there's no platform).
  - **`is_relisted`:** earlier copies exist, but none is still live: the job came back after closing.
  - **`n_sources_so_far`:** how many different sources carry the job at the moment this listing appears (1 = only here).
  - **Hypothesis: posting on several boards signals urgency.** Test it, don't assume it: days open (Kaplan–Meier) by `n_sources_so_far` and cross-posted vs. relisted vs. single, within company-volume bands and with and without staffing agencies (2b.1). Applicant-tracking systems often send one posting to several boards automatically, so check within platform too: if the gap disappears there, it's the software, not the employer's urgency.
  - **Copies in different cities aren't cross-posts:** each listing names one location (2b.5), so a job in several cities appears as several listings. The table has no per-listing city yet, so **source stands in for it:** a live earlier copy on a different source is a cross-post; on the same source, most likely another city (`is_same_source_copy`).
  - **Matching (decided after the first run, 2026-10-10):** copies are matched on company and clean title, not the description: boards reformat descriptions, and the strict match found 0 cross-posts (a looser check found 460 same-title copies within 7 days).
- **Reposts** (a decided feature: how strongly should it count?): days open by `times_listed` and by relisted vs. cross-posted; repost rate in the chosen groups; repost rate within company-volume bands, to see whether it adds anything beyond company volume.
- **Company hiring volume:** does high volume mean **more competition, or more openings to win?** Its direction in the ranking depends on the answer. Count each company's listings from before each listing only, so the future isn't used. The companies table has no company size or employee count, so listing volume is our only measure of how much a company hires.
- **For career changers:** do high-volume employers post a larger share of entry-level listings or training wording? If so, volume also signals that a newcomer can get in.

**What the data shows**
- **Company volume is steady and real:** median days open of 44, 34, 28 and 12 across the four bands (earlier listings in any field), the same without staffing agencies and nearly the same without likely ghosts. Entry level is about 10% in every band, so volume is about speed, not access.
- **Copies of the same job:** 70% of listings are single; 21% are same-source copies (most likely the same job in another city); 7.5% are relisted after earlier copies closed; 1.2% are cross-posted.
- **Relisted copies close fast** (median 10.6 days): most likely crawler refreshes, not new hiring. Cross-posts close somewhat faster than single listings (22 vs. 28 days), but not consistently within source type (3.5c), so the urgency idea finds little support.

### 3.6 The scope: the roles TripleTen trains for, by role bucket

**Decided (2026-10-09; applied at the start, in 1d, since 2026-10-10):** the scope is **computer and mathematical listings in the US**, with listings a career changer can't get hidden (3.11). It isn't narrowed to one role: role discovery needs several roles to choose from, and a single role would leave too few listings and too few career-switch examples. Instead, **role bucket** is a dimension inside the scope, defined the same way on both sides, so the listings (what's open) and the career changers in `Profiles.ipynb` (who got there) line up role by role:

| Role bucket | Listings: O*NET codes (2019 SOC) | Profiles: resume titles |
|:--|:--|:--|
| data science | 15-2051 data scientists, 15-2041 statisticians, 15-2031 operations research, 15-1221 research scientists, 15-2099 | data analyst / scientist / engineer, BI, statistician, machine learning |
| database | 15-1242, 15-1243 | database administrator or developer, SQL developer |
| quality assurance | 15-1253 | QA analyst or tester |
| security | 15-1212 | information, cyber or IT security analyst |
| networks and systems | 15-1211, 15-1231, 15-1241, 15-1244 | network or systems administrator, engineer or analyst; DevOps; cloud |
| it support | 15-1232 | help desk, desktop support, IT support |
| software development | 15-1251, 15-1252, 15-1254, 15-1255 | software or web developer, programmer |
| other computer roles | 15-1299 | (none) |
| technical sales (added 2026-10-10) | 41-9031 sales engineers | (none) |
| business and data analysis (added 2026-10-10) | 13-1111 management analysts, 13-1161 market research analysts (titles naming analysis, research or insights; split 2026-10-11) | (none) |
| marketing specialists (split 2026-10-11; since 2026-10-13 a label and an opt-in filter, not hidden by default) | 13-1161 listings whose title names no analysis or research | (none) |
| credentialed math | 15-2011 actuaries, 15-2021 mathematicians | (none) |

The listing side is `role_bucket(onet_code)` (`ONET_DETAILED_BUCKETS` for detailed 15-1299 codes first, then `ONET_ROLE_BUCKETS`, which also covers the older 2010 codes) and the resume side is `RULES["cm_buckets"]`, both in `src/career_switch.py`.

**Steps:**
1. **Map** every computer and mathematical listing to a role bucket from `onet_code`, and report unmapped codes (aggregate counts) so the mapping can be fixed.
2. **Size each bucket** on originals only: listings, listings still open, companies and the top company's share, with and without manager-level listings (3.11; they're hidden per applicant, so the scope keeps them).
3. **Count career changers per bucket** from `Profiles.ipynb`'s modeling table (`career_changer`, by the role at the switch; unique people, after duplicate removal), split by pool: retrained after working, retrained as new grads, and changed roles within tech.
4. **Depth check per bucket:** measure the 95% confidence-interval width of the Kaplan–Meier share still open after 7 days in each sub-group. Enough depth means **±5 points or narrower** (a width of 10 or less), tight enough to show the 5–16 point gaps the EDA found, in at least 80% of sub-groups. Splits are tried from finest to coarsest, combining sub-groups that are too wide: month × source type, then source type, then month, then the whole bucket. The finest split that passes is how finely that bucket is analyzed in section 3. (About 300+ listings per sub-group are needed for ±5, so most buckets won't support all 18 month × source sub-groups.)
5. **Keep a bucket** only if it passes the depth check on the listings side **and** has enough career changers to learn from (at least 30 unique people, counting **all three pools**). The 30 was first set for retrained people only, before the within-tech pool existed; counting retrained only would drop data science (20 retrained, 240 in all), which the compass needs to show is reached through other tech roles. Thin buckets are combined with a close one or analyzed only as part of the scope; their listings always stay in the table. **Credentialed math** (actuaries, mathematicians) is out of the scope from 1d: credential-gated, with no career switches into it. **IT project managers** (O*NET 15-1299.09, about 12% of the scope) have their own bucket since 2026-10-10, on both sides (a Profiles rules change). Record each keep, combine or drop with its reason.
6. **Save the scope** (kept buckets, blockers hidden) as its own table, the base for section 4 and EDA_SDA_3.

**Also described per bucket** (not used to choose buckets): variation in the factors (window, source, company volume, reposts, pay), coverage of key fields (description, salary, `posted`), words per description, and openness to career changers (`entry_share`, `career_open_share`, years asked, degree required).

**Related groups:** the team's source-background fields were set aside in EDA_SDA_1's cut; they live on in its occupation summary (pay, timing, openness) and through O*NET skills and the Career Changers Matrix for role discovery. **The two added buckets** (technical sales, business and data analysis) have no career-switch paths in the frozen profiles, so 3.6e marks them for combined analysis; their listings stay, and the compass shows their 2026 market.

**What the data shows**
- **Every bucket is deep enough on the listings side** (7-day intervals of 10 points or less in at least 80% of sub-groups, by source type or finer).
- **Eight buckets get their own analysis and compass paths** (30+ career changers each; quality assurance is the thinnest at 38). Technical sales, business and data analysis and other computer roles have no career-switch paths in the frozen profiles, so they're analyzed combined; their listings stay.
- **O*NET 13-1161 is mostly marketing:** 76% of its 39K original listings have no analyst, research or insights title. They're labelled marketing specialists (29K; 21% entry level, $28.90 an hour where open to newcomers), leaving business and data analysis at 34K. **Decided 2026-10-13: the title rule will not be hand-checked, so marketing specialists are a label and an opt-in filter, no longer hidden by default;** **IT project management is large** (106K) but mostly reached from another tech role (189 of its 262 career changers).

### 3.7 Blind spots from Part 1

| Blind spot | Check or fix |
|:--|:--|
| The "closed" outcome is wrong for reposted jobs: the earlier listing disappears without being filled | In the analysis (not as a model input), treat a listing that was later reposted as "closed without filling" or censored |
| The 2-day "still open" tolerance is a guess | Distribution of days between `last_seen` and September 30 (the data's end date); a bump near 7 days means it's too tight |
| Converting salaried pay to hourly assumes full time (2,080 hours) | Hourly listings keep their posted rate, so only salaried part-time listings are off; flag them with `contract_types` (part time) |
| Company volume counted within the sample | The companies table has no size or employee count. It does have a description (industry, via NLP in EDA_SDA_3), a ticker (publicly traded), parent and subsidiary companies, and similar companies: check whether these add anything beyond volume |
| State isn't in the table | Add it from `location_data`, if PredictLeads includes it |
| Missing country vs. remote, missing language vs. description | Confirm the EDA_SDA_1 section 10a results |
| `entry_seniority` counts only the `junior` label | Check every seniority label and add any that are entry level, such as intern, to `ENTRY_LEVELS` |

**As built (2026-10-10):** 3.7a–3.7e cover the live tolerance, salaried part-time pay (`pay_part_time_salaried`), the seniority labels and state (`state`, from the location records), with state-level pay. The reposted-closing question is answered by 3.5 (relisted copies close in 10.6 days, read as crawler refreshes); the companies table's extra fields and the no-country check move to 3.8.

**What the data shows**
- **The 2-day "still open" tolerance holds:** 225K listings were last seen within 2 days of September 30, then the counts drop (7K at 2–3 days) with no bump near 7 days.
- **Seniority labels mostly agree with titles,** but `mid_senior` mixes levels: 36% of its titles say senior, lead, principal or staff. **Kept whole, with a `senior_title` flag:** senior-titled listings ask more (81% want 5+ years where stated vs. 43%), are rarely open to newcomers (4.9% vs. 17%) and pay more ($77.40 vs. $55.29 an hour). Internships are mostly entry level (25K of 31K).
- **State is known for 90% of listings.** California (13%), Texas (8.8%) and New York (7.5%) lead. **Pay-transparency states disclose pay far more often** (Colorado 68%, Washington 67%, Massachusetts 66%, California 63% vs. Texas 30%, Florida 32%, Georgia 29%).

### 3.8 Joining the companies table

Merge the companies table into the listings (a **left join** on `company_id` = company `id`, so every listing stays). Course standard: one master table from all sources, with keys checked first. Join checks: company ids are unique in the companies table (no row multiplication), and match rate (100% in EDA_SDA_1). Columns to bring in: `ticker` (publicly traded), parent company, subsidiary and similar companies, location, and description (its text goes to EDA_SDA_3). The table has **no company size or employee count**.

**Listings with no country** are dropped in EDA_SDA_1 8b (decided 2026-10-12), so the scope has none.

**What the data shows**
- **The companies table matches every scope company** (71,911 of 71,919); 5.7% of companies are publicly traded and 12.0% have a parent company.
- **The volume effect is a private-company pattern:** at 500+ listings, public companies' listings stay open a median 22 days vs. 8 for others.
- **No-country listings** (10% of the scope at the time, then kept by EDA_SDA_1 because their company is in the US) were more often remote (38% vs. 17%) and rarely had a description (37% vs. 81%); EDA_SDA_1 8b now drops them (decided 2026-10-12).

### 3.9 Missing-data patterns

**First classify each field** (structural, censored, or missing for a reason), then treat it. Most of ours aren't truly missing, so very little is imputed.

| Field | Missing (final table) | Why | Handling |
|:--|:--|:--|:--|
| `first_seen`, `last_seen`, title, seniority, occupation group, `onet_code` | ~0% | n/a | Nothing. A listing with no `onet_code` can't get a role bucket, so it's outside the scope (report the count) |
| `last_seen` of **live** listings | 25% still open | **Censored,** not missing | Never imputed or dropped; survival methods handle it |
| `platform` | 61% | **Structural:** company sites have no platform | Not missing; `source_type` says "company site / other" |
| `status` | 38% | Never says "open" | Not used; `last_seen` decides what's live |
| `posted` | 58% | Many sources publish no date | **Not imputed.** Freshness uses `first_seen`; day of week uses the "with `posted`" view; add `has_posted`, and check whether listings with and without it differ |
| Salary | 53% | **Missing for a reason** (pay-transparency laws, hourly vs. salaried) | **Pay level isn't filled in.** `salary_disclosed` stays its own factor. Trees use blanks natively; linear models get a median by role bucket × seniority **plus** a missing flag, fit inside the `Pipeline` on training months only. The app shows "not disclosed" |
| Description and its text flags | 18% | Some sources expose no text | Listing kept; text flags are **unknown** (blank), not False, since EDA_SDA_1's fix; `has_desc` is the flag |
| `years_required` | Blank when no years are stated | **"Not stated" isn't zero** | Kept blank, with a "states years" flag; not stated never counts as a blocker |
| Repost features | Blank without a description | Can't be matched | Kept blank, not 0 (0 would claim "first listing") |
| `tags` | 53% empty | Sparse | Not used; `categories` (99%) instead |
| Company description, location (joined in 3.8) | 10%, 6% | Not in the companies file | "Unknown" category |

**Course standard:** percent missing, missingness **by group** (source, occupation group, role bucket, month, state), and `df.isnull().corr()` to see which fields go missing together. Every imputation or "Unknown" category is justified with plots split by the outcome (closed vs. still open). **Rules:** impute only inside the model `Pipeline`, fit on the training months (leakage rules in `STANDARDS.md`), and add a missing flag whenever a value is filled in, since missingness itself can carry signal (salary disclosure does).

**What the data shows**
- **Most blanks are structural or "not stated":** platform (61%: company sites have none), posting date (58%), years asked (54%), pay (52%), recruiter (99%).
- **No description is the key missing field:** 18% of the scope, rising from 8% in April to 23–25% from June, highest on company sites (28%). These listings close in a median 3 days vs. 30, so missing pay or years looks predictive only through them; `has_desc` stays a standard confounder.
- **Missingness tracks source and month, not role bucket** (10–25% no description across buckets). Nothing is imputed here.

### 3.10 Outliers

Boxplots and IQR rules for days open, words per description, hourly pay, times listed. Course standard: inspect, explain, and keep or remove with a reason. Advanced: robust (median / IQR) summaries, and log scales for the right-skewed measures.

**What the data shows**
- **The long tails are real, not errors:** 6.2% of closed listings run past the IQR fence (82 days), none past the log-scale fence; descriptions over about 1,650 words are 1.2%.
- **Pay outliers are mostly senior roles:** 1.4% are above $146 an hour, 66% of them senior or above. 237 listings under the federal minimum wage and 253 over $250 an hour are flagged and their hourly pay blanked (0.06%).
- **Times listed:** 92% are first listings, 3.1% have been listed 5+ times.

### 3.11 Career-change accessibility, and hiding what a career changer can't get

**Decided (2026-10-09; text rules revised 2026-10-13): listings a career changer can't get are hidden where the rule is reliable.** Seniority, likely ghosts and non-tech occupations hide by default. The text rules are keyword rules that EDA_SDA_3 will not hand-validate (decided 2026-10-13), so they are **labels and opt-in filters** the career changer turns on, never hidden by default:

| Blocker | Example | Hidden in |
|:--|:--|:--|
| Seniority, per applicant | Manager level and above (`manager`, `head`, `director`, `vice_president`, `president`, `c_level`, `partner`, `founder`) for applicants without management or senior experience | The app, per profile; this notebook's table keeps them, labelled |
| Text, opt-in filter | Too many years of experience asked; a required license, clearance or sponsorship | EDA_SDA_3's final table, as labels; the app filters only when the user turns it on |
| Text, opt-in filter, per applicant | A degree or certification this applicant doesn't have ("I have no degree") | The app, on request, per profile |
| Occupation, for everyone | Document management specialists (O*NET 15-1299.03): records work, not a tech path (decided 2026-10-10) | The app; flagged `hidden_occupation` in 3.0 and kept in the table |
| Title rules, opt-in filter | Marketing specialists (13-1161 without an analyst title, 3.0); staffing agency from recruiter titles (2b.1). First-pass keyword rules, not hand-checked (decided 2026-10-13) | Labels in the table; the app filters only when the user turns it on |

Report how many listings each rule hides. Then check:

| Check | If the result is… | Then… |
|:--|:--|:--|
| Accessibility flags by company-volume band and by source | Entry-level listings cluster at high-volume employers or on certain platforms | Account for it, so accessibility isn't company size or source in disguise |
| Pay of listings open to career changers vs. the rest, within the scope | A large pay gap | Show it in the app, and let the pay weight trade off against accessibility |
| Days open by accessibility, within the scope | Reachable listings close faster | Raise their urgency in the ranking |
| Degree required by role bucket | Near-universal in a bucket, with little "or equivalent experience" | That bucket is a poor target |
| Each part of `career_open` separately | One part (for example, entry-level wording) adds little | Use a stricter rule for the "open to career changers" view |

**What the data shows**
- **What a career changer without management experience doesn't see: 15% of open listings** (likely ghosts 3.7%, manager level or above 11%, hidden occupations 0.4%: document management; marketing specialists are only labelled, with an opt-in filter). An "asks 5+ years" filter would hide another 32% if a user turned it on (EDA_SDA_3 labels it and does not hide it by default); "degree required, no equivalent" fires on only 2.6%.
- **Openness isn't company size or source in disguise:** 18–28% show any sign in every volume band × source cell. Seniority and years asked do most of the work (each is the only sign for 60–76% of its listings).
- **Reachable listings pay $10–35 an hour less in every bucket** (smallest in IT support, largest in other computer roles), and close a little faster at small companies (30 vs. 39 days at 1–4 listings).

### 3.12 The retraining compass: the 2026 side

For each kept role bucket, after blockers are hidden: **open listings on September 30, new listings per week, median hourly pay (where disclosed), median days open (Kaplan–Meier), entry-level share, openness to career changers, remote share, and expected job growth** (a placeholder column until the projections are downloaded) (official projections by occupation code; time frame to be decided). Rows in the same role order as Profiles (`ROLE_ORDER` in `src/career_switch.py`: infrastructure → code → data).

Join it by role bucket to Profiles' three path tables, so every path shows its 2026 openings, pay, closing speed and growth:
- **Prior career → first tech job** and **field trained in → first tech job** (4a.2b): the compass for changing careers.
- **Tech job before → after the move** (4a.3a), with the sideways-or-up split: the compass for moving across or up within tech.

Every path carries its people count, and paths under about 10 people are marked thin. Growth sits next to the entry-level share, since projections cover the whole occupation. Its source and method are in job growth tasks in `ANALYSIS.md` (BLS 2025–2035, employment-weighted by bucket, at most 5 years). Aggregates only; this table feeds the app's Retraining compass and the deck (`DIFFERENTIATORS.md`).

**What the data shows**
- **The 2026 market by bucket:** software development (66K open, 9.2K new a week, $73.54 an hour) and networks and systems (41K open) are the largest; IT support is the door (44% open to career changers, 22% entry level, $31.10); technical sales is the slowest (median 38 days) but the most remote (29%).
- **By TripleTen program:** AI Software Engineering (63K open) and AI & Machine Learning pay the most (about $74); **Quality Assurance and Data Analytics are the most open to career changers** (27% and 26%), Data Analytics at $51.20 now that marketing specialists are out of its market; adjacent tech roles hold 88K open listings.
- **Most paths are thin:** 78% of prior-career paths and 67% of training paths rest on fewer than 10 people (within-tech moves: 43%), so the compass always shows how many people a path rests on. The largest within-tech move is IT support → networks and systems (279 people).

### 3.13 Feature screening: which candidates go forward

Every candidate feature goes through **the same checks**, so the choice of what EDA_SDA_4, TimeSeries and ML_Rank use rests on evidence, not habit. Ten families of candidates:

| Family | Examples | Built in |
|:--|:--|:--|
| 1. Raw listing fields | seniority, source, job type, work arrangement, AI/ML title, recruiter, pay currency | EDA_SDA_1, 2b |
| 2. The listing's own history | times listed, copies, posting lag, batch first sightings | EDA_SDA_1 6.4, 6.8; 3.5 |
| 3. Company history (earlier listings only) | past closing speed of its closed listings, entry-level share, repost and pay-disclosure habits, roles open, hiring trend | 3.13a |
| 4. The market at first sighting | open listings in the same role bucket (and state) that day, new listings that week, weekday and hour | 3.13a |
| 5. Pay in context | hourly pay, its percentile within role bucket and state, range width, hourly vs. salaried | 3.13a |
| 6. Text (keyword flags for now) | years asked, degree, entry-level and career-changer wording, remote, length | EDA_SDA_1; EDA_SDA_3 replaces them |
| 7. Company facts | publicly traded, parent company, subsidiaries | 3.8 |
| 8. Location | state, a pay-transparency-law state | 3.7, 3.13a |
| 9. O*NET Job Zone | how much preparation a role needs | Waiting for the O*NET download |
| 10. Expected job growth | BLS 2025–2035 by occupation | Waiting for PredictLeads' consent to the join |

**The checks, fixed before looking at results:**
- **Outcome:** closed within 7 days of first sighting, on original listings first seen at least 7 days before September 30 (so every one has a full week of follow-up). **Population: the career-changer view (3.1e) decides; the whole scope and the view without senior titles are shown alongside. Repost features are screened on every listing, reposts included (on originals they can't vary).**
- **Coverage:** share with a value, and its lowest share by source type and by month.
- **Leakage:** is the value known when the listing is first seen? Features built from later information are dropped; `last_processed_at` is audited (3.13b).
- **Signal within the confounders:** each feature is cut into levels (true / false, quintiles, or the 10 most common values, with "missing" as its own level). Within each cell of company volume × source type × has a description, the 7-day close rate's spread across levels (levels with 400+ listings), averaged over cells. Some spread appears by chance, more with more levels, so the same spread is measured with the outcome **shuffled within each cell (the noise floor)**, and the signal is the spread above it. **Under 2 points above the noise floor = no signal beyond the confounders.**
- **Stability:** the same spread in April–June and in July–September, and whether the levels keep their order. **Unstable** if one half's spread is under a third of the other's, or the order flips.
- **A second measure (added 2026-10-11): information within the confounders** (conditional mutual information, as a share of the outcome's uncertainty, above the same shuffled noise floor). The spread looks at the gap between levels; information also weighs how many listings sit in each level, so a big gap in a rare level counts for little. **The bar, fixed before looking:** what a yes/no feature splitting listings in half with a 2-point gap would carry. A feature that passes the spread but not the information bar is kept and watched ("the two measures disagree").
- **Redundancy:** Cramér's V between features' levels, on listings where both have a value (a 200,000-listing sample), so sharing blanks (every text flag is blank without a description) doesn't count. **V of 0.7 or more** with a stronger feature = keep one.

A gradient-boosting importance check comes later, at the start of ML_Rank, as a cross-check, not the decision.

**What the data shows**
- **68 candidates screened, 48 go forward** (1 recomputed on training months, 8 watched), 9 dropped as redundant with a stronger partner, 9 dropped and 2 held for EDA_SDA_3's text checks; 2 more wait for data. Screened on 691K career-changer listings with a full week of follow-up (28.9% closed within 7 days); the whole scope and the view without senior titles agree within 1–3 points, except two: `n_categories` is stronger on the whole scope (14.5 vs. 10.6 points) and `desc_words` weaker without senior titles (7.7 vs. 13.0), so both are read as depending on the population, not as fixed effects.
- **The strongest signals beyond the confounders:** has a description (37 points), the company's past closing speed (29), company volume (24), the platform (21), its past closings (21) and its open listings now (16). **Company history is the strongest new family:** all nine kept. **The second measure (information) agrees:** rank correlation 0.87, disagreeing on 3 of 68. Two rare text flags (career-changer and entry wording) pass the spread but carry little information, so they're watched.
- **Dropped:** AI/ML title (0.8 points), the recruiter's earlier listings (0.3), the agency-style recruiter title (too rare to measure within the confounders), the repost counts once copy type is known, `crawl_dropout` (not known at first sighting) and `last_processed` (it leaks the outcome, 3.13b). `senior_title` sits right at the bar (2.1), so it labels and ranks rather than hides.

## 4. EDA_SDA_2 final table

Every decision in this notebook, defended with its result from sections 2b and 3, then applied all at once in 4b. The table is saved as its own file (`scope_table.parquet`), so EDA_SDA_1's table stays unchanged. **Hide, never remove:** no listing is dropped here (the scope cut and the US rule are EDA_SDA_1's); everything is a flag the app hides or labels by.

### 4.0 Decisions

| Decision | Evidence (full scale) | Choice | Status |
|:--|:--|:--|:--|
| **The scope** | The roles TripleTen trains for plus adjacent tech roles (EDA_SDA_1 8.0, 8c): 908K listings | As cut by EDA_SDA_1 | Applied in EDA_SDA_1 |
| **Role buckets** | Every bucket is deep enough on the listings side; eight have 30+ career changers (3.6) | Eight buckets with their own analysis and compass paths; technical sales, business and data analysis and other computer roles analyzed combined (no career-switch paths); document management flagged `hidden_occupation` | Decided; listings always kept |
| **O*NET 13-1161 split by title** | 76% of its 39K original listings are marketing specialists (no analyst, research or insights title) (3.6) | `role_bucket` = marketing specialists, a label and an opt-in filter since 2026-10-13 (a first-pass title rule, not hand-checked; `hidden_occupation` covers document management only) | **Applied in 3.0; hiding revised 2026-10-13** |
| **IT project management** | 9.1% open to career changers, 2.1% entry level, 44% manager level; 189 of its 262 career changers came from another tech role (3.6, 3.11) | Kept as a later move: not suggested as a first tech job unless the prior career has a real path into it | Decided (app and ML_Fit) |
| **Listings with no country** | Before the drop: 10% of the scope, all from US-based companies; more often remote (38% vs. 17%), rarely with a description (37%) | Dropped in EDA_SDA_1 8b (decided 2026-10-12; were kept and flagged `country_inferred`) | Applied in EDA_SDA_1 8b |
| **Likely ghosts** | Closing slows with age but never stops; 19% of open listings are older than 90 days; 7.5K likely ghosts (3.5% of open) (3.1) | Flag `likely_ghost`, hidden in the app; other stale listings labelled "open 90+ days" and ranked lower | Decided; timing also shown without them |
| **Manager level or above** | 11% of open listings (3.11) | Labelled, hidden per applicant | Decided (app) |
| **Text blockers** | "Asks 5+ years" would hide 32% of open listings; "degree required, no equivalent" only 2.6% (3.11) | Previews only | Decided 2026-10-13: labels and opt-in filters, not validated (EDA_SDA_3) |
| **Staffing agencies and recruiters** | 2.0% of listings (753 companies, 33 of them from recruiter titles), closing in 12.6 vs. 24.0 days; recruiters with 5+ listings close fast even at employers (8.1 vs. 24.1 days) (2b.1) | Flag `is_staffing` (the app labels "posted by a recruiter"); `recruiter_key` stays a candidate feature; the title rule (`staffing_by_title`) is a label and opt-in filter, not hand-checked (2026-10-13) | Decided |
| **Contract types and where the work happens** | 55% full time, 9.7% contract, 3.3% internship; work arrangement stated for 54% (2b.3, 2b.6) | Employment flags, `job_type`, `work_arrangement`, `schedule` for the app's filters | Decided |
| **AI/ML titles** | 9.0% of listings, across buckets (25% of data science) (2b.4); no closing-speed signal (0.8 points, 3.13) | `ai_ml_title` as a view for the hackathon app's AI/ML audience, not a ranking feature | Decided |
| **Senior titles in `mid_senior`** | 36% of `mid_senior` titles; 81% ask 5+ years vs. 43%, 4.9% vs. 17% open to newcomers, $77 vs. $55 an hour (3.7c); 2.1 points of signal, right at the bar (3.13) | `mid_senior` kept whole; `senior_title` flag; the app shows them ranked lower with a "senior" label | Decided (app, ML_Rank) |
| **The career-changer view** | 86% of original listings, same timing, slightly lower pay; 55% without senior titles (3.1e) | `cc_visible` decides the feature screening; key tables show it beside the whole scope | Decided |
| **`last_processed`** | 88% within 7 days of the last sighting (3.13b) | Never a feature: it leaks the outcome | Decided |
| **Feature screening** | 68 candidates: 48 go forward, 9 redundant, 9 dropped, 2 held, plus 2 waiting for data; a second measure (information) agrees on 65 of 68 (3.13) | The verdicts are saved (`feature_screening.pkl`) for EDA_SDA_4, TimeSeries and ML_Rank | Decided; a gradient-boosting check in ML_Rank cross-checks it |
| **Pay** | 237 listings under $7.25 and 253 over $250 an hour (0.06%); 1,389 part-time listings with salaried pay (3.7, 3.10) | Flag `pay_implausible` and blank its hourly pay; flag `pay_part_time_salaried` | **Applied in 4b** |
| **Missing values** | Mostly structural or "not stated"; no description for 18% (23–25% from June), closing in 3 days (3.9) | Nothing imputed here; `has_desc` a standard confounder; `has_posted` added | Decided |
| **Outliers** | Real tails (3.10) | Keep; times listed grouped 1 / 2–4 / 5 or more (`times_listed_band`) | **Applied in 4b** |
| **"Possibly hard to fill"** | No more open to newcomers than younger listings (16% vs. 20%) (3.1) | Dropped as a separate label | **Applied in 4b** |

### 4.0 Prioritization factors, after this notebook

| Factor | Result (full scale) | Choice | Status |
|:--|:--|:--|:--|
| Company hiring volume | 12 vs. 44 days (500+ vs. 1–4, from first seen); about 1.5× like-for-like from the posting date; a private-company pattern (public 22 vs. others 8 days at 500+) (3.3, 3.5, 3.8) | Keep (`company_earlier`, `volume_band`) | Kept; EDA_SDA_4 tests volume × company type |
| Source and platform | Interacts with volume (company sites 8 vs. 64 days); platforms 10–71 days; 36.0% on the employer's own domain; job boards stop after May (3.2) | Keep `source_type`, `platform`, `on_employer_domain` | Kept; a standard confounder |
| Freshness | Median crawler lag 1.5 days; late-found listings have long windows from posting (3.3) | Keep `first_seen`, with `posted` and `has_posted` | Kept |
| Pay disclosed | Driven by source and state (pay-transparency states 63–68% vs. about 30%) (3.4, 3.7) | Keep, compared within state and source | Kept; EDA_SDA_4 tests it within state |
| Pay level | Reachable listings pay $10–35 an hour less in every bucket (3.11) | Keep, hourly | Decided; the app shows the trade-off |
| Recruiters | Fast closing, beyond staffing agencies, but only 1.3% of listings (2b.1b); naming a recruiter carries 12.5 points of signal, the recruiter's earlier listings none (3.13) | Keep `has_recruiter`; drop `recruiter_earlier` | Screened |
| Company history (new) | The company's past closing speed is the strongest new signal (29 points); entry-level share, pay-disclosure habit, open roles and hiring trend all clear the bar (3.13) | Keep all nine, from earlier listings only | Screened |
| Copies and reposts | 70% single, 21% same-source copies, 7.5% relisted (close in 10.6 days), 1.2% cross-posted (3.5) | Keep `copy_type` and the repost features | Kept; little urgency signal |
| Company type | Public companies' listings stay open longer at high volume (3.8) | Keep `is_public`, `has_parent` | Kept; a standard confounder |
| State | Known for 90%; pay transparency shows (3.7) | Keep `state` | Kept; the app's location filter and a confounder |
| Career-change accessibility | 20% show any sign; similar in every volume band × source cell; seniority and years asked do most of the work (3.11) | Keep `career_open` and its parts | Kept; the text parts are unvalidated keyword rules in EDA_SDA_3 (labels) |

**What the final table shows**
- **The final scope table: 908K listings (849K originals) from 72K companies in 12 role buckets (marketing specialists split out, labelled), 128 columns** (`scope_table.parquet`, 0.26 GB). Nothing was removed; every decision is a flag.
- **23.7% are open on September 30 and visible to everyone** (no likely ghost or hidden occupation). Flags kept: likely ghosts 0.9%, open 90+ days 4.7%, staffing agencies 2.0%, hidden occupations 0.4% (document management specialists; marketing specialists are only labelled), implausible pay 0.06%, senior title 36%.
- **The added fields are complete** except state (90%) and company country (92%).

## 5. Conclusion and next steps

**The scope:** 908K listings (849K originals) from 72K companies in 12 role buckets (marketing specialists split out and labelled), cleaned, flagged and saved for EDA_SDA_3 (`scope_table.parquet`). Nothing was removed; every decision is a flag. The full data confirms the sample's main lesson: **how the data was collected explains many of the biggest gaps.** Crawl timing, job boards stopping after May and listings with no description (18%, closing in about 3 days) all look like signals until they're checked, so "has a description", company type and state stay standard confounders.

**What goes forward to modeling:** 48 of 68 candidate features, screened on the career-changer view, with a second measure (information) agreeing on 65 of 68 (3.13). The company's own history is the strongest new signal; AI/ML titles, the recruiter's history and the repost counts add nothing beyond the confounders, and `last_processed` leaks the outcome. `mid_senior` stays whole, with senior titles labelled and ranked lower.

### Answers to the clarifying questions

| Question | Answer |
|:--|:--|
| Which long-open listings are ghosts, and which are still hiring? | Company activity separates them. **7.5K likely ghosts are hidden** (3.5% of open listings); other listings open 90+ days stay visible, labelled and ranked lower, since 30–36% of old open listings still close each month (3.1) |
| Does source matter beyond company volume? | Yes, and the two interact: at 500+ listings, company-site listings close in 8 days vs. 22 on hiring platforms; at 1–4, 64 vs. 47. Platforms range from 10 to 71 days (3.2) |
| How precise is freshness? | A median 1.5 days late; busy companies are crawled more often, so like-for-like the volume effect is about 1.5×, not nearly 4× (3.3) |
| Is pay disclosure about state laws or source rather than intent? | Largely: pay-transparency states disclose 63–68% vs. about 30% elsewhere, hiring platforms 53–64% vs. job boards 12–16% (3.4, 3.7) |
| Competition or more openings? Are fast recruiters just staffing agencies? | Busy employers fill faster, about a third like-for-like, mostly private companies; the data has no applicant counts, so volume reads as "apply sooner". Recruiters close fast even at employers (8.1 vs. 24.1 days), not only staffing agencies (2b.1b, 3.3, 3.5, 3.8) |
| How are values missing? | Tied to source and month, not role bucket; no description rises to 23–25% from June. Nothing is imputed here (3.9) |
| How much is left once blockers are hidden? | 85% of open listings stay visible to a career changer without management experience (marketing specialists, 13-1161 without an analyst title, are only labelled since 2026-10-13); an opt-in "5+ years" filter would hide another 32% if turned on; EDA_SDA_3 labels it and does not hide by default (3.11) |
| Which role buckets are deep enough? | All 12 on the listings side (marketing specialists labelled, not hidden by default since 2026-10-13); eight also have enough career changers for their own compass paths. Technical sales, business and data analysis and other computer roles are analyzed combined (3.6) |
| Which features carry signal beyond the confounders? | 48 of 68 screened go forward. Strongest: has a description, the company's past closing speed, volume, platform, past closings and open roles. Dropped: AI/ML title, the recruiter's history, repost counts, `crawl_dropout`, `last_processed` (3.13) |

### For the career changer
- **IT support is the clearest door** (44% open, 22% entry level) at the lowest pay ($31 an hour); **Quality Assurance and Data Analytics** are the most open TripleTen programs (27% and 26%; Data Analytics at $51.20).
- **Reachable listings pay $10–35 an hour less in every bucket;** the app shows the trade-off, and the compass shows where people moved next (IT support → networks and systems is the most common move, 279 people).
- **Senior-titled listings are mostly out of reach** (81% ask 5+ years, 4.9% open to newcomers): shown with a "senior" label and ranked lower, not hidden.
- **Most paths are thin,** so the compass always shows how many people a path rests on.

### Next steps
- **EDA_SDA_3 (text):** EDA_SDA_2's two title rules (agency-style recruiter titles, the 13-1161 split) are not hand-checked (dropped 2026-10-13): labels and opt-in filters; clean descriptions from `scope_descriptions.parquet`; report the "5+ years" and degree rules as unvalidated labels and opt-in filters (the hand validation was dropped, decided 2026-10-13), so no text blocker hides listings by default; urgency read in context; map skills to O*NET; add evergreen wording to the ghost check.
- **EDA_SDA_4 (statistics):** volume × company type, pay disclosure within state and source, recruiters, and whether reachable listings close faster, in the adjusted Cox model.
- **Ask PredictLeads:** why job-board listings stop after May, and why descriptions go missing more often from June.
