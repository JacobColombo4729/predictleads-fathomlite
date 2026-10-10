<!-- Generated from notebooks/EDA_SDA_2.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 2 (Non-Text Analysis)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

`EDA_SDA_1.ipynb` explored the data, defended the scope decisions, and built its final table (sections 10 and 11). This notebook does the **non-text** in-depth analysis: it checks the factors kept for now, separates ghost listings from desperate employers, analyzes each time scale, narrows the target, **computer and mathematical**, to the listings a career changer can get, splits it into **role buckets** aligned with the career-switch profiles (our niche), pairs it with the team's backgrounds, and builds **its own final table**. The NLP work comes next, in `EDA_SDA_3.ipynb`; statistics in `EDA_SDA_4.ipynb`; then the `TimeSeries` and `ML` notebooks.

**Inputs from EDA_SDA_1** (decision tables in its section 10):
- English-language listings in the United States (no language label: dropped), plus listings with **no country**, kept and flagged `country_missing` for 3.8 to resolve
- Monthly horizon, April–September 2026
- Every listing's full follow-up: long-open listings kept, nothing capped
- All occupation groups and seniority levels, and all candidate factor columns, including reposts as a feature
- Career-change accessibility flags: `entry_seniority`, `years_required`, `low_experience_ask`, `entry_wording`, `changer_wording`, `degree_required`, `equivalent_experience`, and the first-pass `career_open` (EDA_SDA_1 Q1g)

**Conventions carried over:**
- Every decision is recorded with its reasoning.
- Data-quality fixes happen in cleaning; scope decisions are defended first, then applied at the end (section 4).
- Aggregates only (counts, shares, quantiles, model estimates); never `df.head()` or sample rows. Clear outputs before committing.


**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, and what's the scope? | Final table 1 |
| EDA_SDA_2 | Which listings do we study? Non-text checks, ghost listings, the career-change target | Niche table, non-text features |
| EDA_SDA_3 | What features does the text add? (NLP) | Niche table + text features |
| EDA_SDA_4 | Which features hold up? (statistics) | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing patterns and features |
| ML | How should listings be ranked for a career changer? | The prioritization model |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

**Data rules:** aggregates only, except cells tagged `data-rows` (the 2b.1 review of flagged company names): clear them before committing. Never paste rows into AI tools.

### Clarifying questions
- Which long-open listings are ghost listings, and which are employers still trying to hire?
- Does source matter beyond company hiring volume, or is it company volume in disguise?
- How precise is freshness, given the crawler finds listings a median of 1.5 days late?
- Is pay disclosure driven by state pay-transparency laws or job type, rather than employer intent?
- Does high company volume mean more competition, or more openings to win?
- How are values missing: at random, or tied to groups (source, occupation group, state)?
- How much of computer and mathematical is left once listings a career changer can't get are hidden, and is it enough for depth?
- Do high-volume employers post more entry-level listings, so that volume also signals reachability?
- Which role buckets inside computer and mathematical have enough listings (after hiding blockers) **and** enough career changers (any pool) to keep, and which are too thin?

### Roadmap

1. **Setup:** libraries and EDA_SDA_1's final table
2. **Analysis views:** which slice of the table each analysis uses, and why; **additional cleaning** (2b): staffing agencies, date consistency, internships and temporary roles, title normalization, multi-location listings
3. **Checks and analyses** (non-text)
4. **EDA_SDA_2 final table:** decisions defended, then applied at the end, with a reason for each step
5. **Conclusion and next steps** (into EDA_SDA_3)

## 2. Analysis views

Some analyses are skewed by part of the table. Each analysis states which **view** it uses:

| View | Rows | Use it for | Why |
|:--|:--|:--|:--|
| **All listings** | Everything in the final table, reposts included | Ranking listings as an applicant sees them; the repost features | Each listing is a row the app could show |
| **Originals only** | Reposts (`is_repost`) excluded | Counts, volumes, application windows, company concentration | A reposted job counted twice inflates volumes and splits one long-open job into two short ones |
| **With `posted`** | About 41% of listings | Day-of-week patterns; crawler lag | `first_seen` days follow the crawler's schedule |
| **With a salary** | About 43% of listings | Pay level and pay comparisons | Pay is missing for the rest |
| **Closed listings** | Not still open at the snapshot | Simple descriptive durations | Survival methods use every listing; this view is only for plain summaries |
| **Long-open listings** | Open for a long time (threshold set in 3.1) | Ghost listings vs. desperate employers | The question only applies to them |
| **Open to career changers** | `career_open`, or the stricter rule chosen in 3.11 | Accessibility comparisons, inside and outside the niche | Shows whether reachable listings behave differently |
| **The niche** | Computer and mathematical listings, by role bucket (3.6) | The segment analysis and the final table | Focus |

## 2b. Additional cleaning

**Decided (2026-10-09):** five fixes here (salary sanity moved into EDA_SDA_1 6.5, description cleanup to EDA_SDA_3), run before the checks in section 3 so they use clean pay, dates and titles. Same rules as EDA_SDA_1: **fix or flag, never drop** (a flagged listing leaves only the analyses it would distort), keep the original column next to every cleaned one, and report how many listings each step changes (counts only). Spot-checks of examples are done locally by a teammate.

| # | Fix | How | Result |
|:-:|:--|:--|:--|
| 1 | **Staffing agencies** | Flag listings posted by recruiters or staffing firms: company name and domain words (staffing, recruiting, talent, well-known agency names), or an "about" text that says staffing or recruiting firm (companies table). A review cell (2b.1 review, row-level) lists the most-flagged companies so a teammate can check the keywords | `is_staffing`, kept. Report their share by field and their days open; compute company volume with and without them (3.5), since agencies post at high volume and repost often. The app labels them "posted by a recruiter" |
| – | **Salary sanity, standardized to hourly** | **Moved to EDA_SDA_1 6.5:** all pay cleaning in one place. `salary_hourly_usd` (salaried pay ÷ 2,080 hours) is the one pay column for the whole project; reversed ranges and clearly wrong units are fixed there and flagged `pay_suspect` | Arrives clean in EDA_SDA_1's table |
| 2 | **Date consistency** | EDA_SDA_1 (6.3) already removed impossible timelines and blanked `posted` dates after `first_seen`. Here: `posted` more than a year before `first_seen` | `posted_old` flag, kept; it feeds the ghost-listing check (3.1) |
| 3 | **Internships and temporary roles** | `contract_types` plus title words (intern, internship, temporary, temp, seasonal, contract-to-hire) | `job_type` (internship, temporary, contract, regular), kept: they can be entry points for career changers. Report their share in the niche and their days open; the app can filter them |
| 4 | **Title normalization** | Lowercase; strip bracketed text, locations after " - " or " \| ", requisition IDs, pay, and "remote" or "hybrid" tags (kept as a flag) | `title_clean`; the original title stays. Recheck the repost count and the role-bucket mapping with it |
| 5 | **Multi-location listings** | `n_locations` (from EDA_SDA_1) above 1 | `multi_location` flag; one row per listing for timing. The state list for the app's location filter comes with `location_data` in 3.7 |
| – | **Description cleanup** | **Moved to EDA_SDA_3:** this table keeps only description features (`desc_words`, `desc_hash`, `has_html`), not the text, to fit in memory; EDA_SDA_3 loads the text anyway | `desc_clean` there, with HTML and boilerplate removed |

**Leakage:** every fix uses only what the listing showed when it was posted, so none of them leaks the outcome.

## 3. Checks and analyses

Record each result and the decision it leads to here, then apply it when building the final table (section 4).

### 3.1 Ghost listings vs. desperate employers

A long-open listing may be a **ghost listing** (forgotten or no longer monitored: rank lower or flag it) or a **desperate employer** (still actively hiring: rank higher). Age alone can't tell them apart.

| Signal | Points to "still hiring" | Points to "ghost listing" |
|:--|:--|:--|
| Relisted (`times_listed`, `days_since_repost_last_listed`) | Relisted recently | Never relisted in months |
| Company still posting new listings | The company is actively recruiting | The company stopped posting long ago |
| Source | Hiring platforms often auto-expire listings, so a long-open one was likely kept on purpose | Company career pages can host forgotten listings indefinitely |
| `posted` vs. `first_seen` | Recently refreshed | Very old posting date |
| `last_processed_at` (meaning to confirm with PredictLeads) | May record content changes | |

Steps: describe the long-open listings (how many, which occupation groups and sources); measure each signal; combine them into a **"likely ghost listing" flag**; decide whether to filter, rank lower, or show the flag in the app. Evergreen wording ("always accepting applications") is added in EDA_SDA_3.

### 3.2 Source / platform

| Check | If the result is… | Then… |
|:--|:--|:--|
| Median days open by source, *within* each company-volume band | Source gaps hold within every band | Keep source as its own factor |
| | Gaps mostly disappear | Drop it, or keep it as a minor factor |
| Share of job-board listings whose company + title also appears on a company site or platform | High (copies) | Drop job-board rows to avoid double-counting |
| | Low | Keep them |
| Platform sizes in the chosen groups | Only a few platforms have 1,000+ listings | Name those platforms; group the rest as "other platform" |

### 3.3 Freshness and crawler lag

| Check | If the result is… | Then… |
|:--|:--|:--|
| Share of listings found more than 7, 14 and 30 days after `posted` | Tiny | Use `first_seen` only |
| | Meaningful | Add a "found late" flag |
| Survival curves: found late vs. found promptly | Found-late listings close much sooner | The flag matters for apply-by estimates |
| Survival curves: listings with vs. without a `posted` date | Similar | `posted` can be used where present |
| | Different | Stick with `first_seen` only |

### 3.4 Pay disclosed

| Check | If the result is… | Then… |
|:--|:--|:--|
| Disclosure rate by US state (needs the state from `location_data`) | Much higher in pay-transparency states (CA, CO, NY, WA) | Add state, or a "pay-law state" flag |
| Disclosure by hourly vs. salaried, and by seniority | Strongly tied to hourly | Its effect may really be job type |
| Days open by disclosure, within company-volume bands | The gap holds | Keep as a factor |
| | The gap disappears | Keep only as the companion flag for pay level |

### 3.5 Reposts and company hiring volume

- **Reposts** (a decided feature: how strongly should it count?): days open by `times_listed`; repost rate in the chosen groups; repost rate within company-volume bands, to see whether it adds anything beyond company volume.
- **Company hiring volume:** does high volume mean **more competition, or more openings to win?** Its direction in the ranking depends on the answer. Count each company's listings from before each listing only, so the future isn't used. The companies table has no company size or employee count, so listing volume is our only measure of how much a company hires.
- **For career changers:** do high-volume employers post a larger share of entry-level listings or training wording? If so, volume also signals that a newcomer can get in.

### 3.6 The niche: computer and mathematical, by role bucket

**Decided (2026-10-09):** the niche is **computer and mathematical listings in the US**, with listings a career changer can't get hidden (3.11). It isn't narrowed to one role: role discovery needs several roles to choose from, and a single role would leave too few listings and too few career-switch examples. Instead, **role bucket** is a dimension inside the niche, defined the same way on both sides, so the listings (what's open) and the career changers in `Profiles.ipynb` (who got there) line up role by role:

| Role bucket | Listings: O*NET codes (2019 SOC) | Profiles: resume titles |
|:--|:--|:--|
| data and analytics | 15-2051 data scientists, 15-2041 statisticians, 15-2031 operations research, 15-1221 research scientists, 15-2099 | data analyst / scientist / engineer, BI, statistician, machine learning |
| database | 15-1242, 15-1243 | database administrator or developer, SQL developer |
| quality assurance | 15-1253 | QA analyst or tester |
| security | 15-1212 | information, cyber or IT security analyst |
| networks and systems | 15-1211, 15-1231, 15-1241, 15-1244 | network or systems administrator, engineer or analyst; DevOps; cloud |
| it support | 15-1232 | help desk, desktop support, IT support |
| software development | 15-1251, 15-1252, 15-1254, 15-1255 | software or web developer, programmer |
| other computer roles | 15-1299 | (none) |
| credentialed math | 15-2011 actuaries, 15-2021 mathematicians | (none) |

The listing side is `role_bucket(onet_code)` (`ONET_ROLE_BUCKETS`, which also covers the older 2010 codes) and the resume side is `RULES["cm_buckets"]`, both in `src/career_switch.py`.

**Steps:**
1. **Map** every computer and mathematical listing to a role bucket from `onet_code`, and report unmapped codes (aggregate counts) so the mapping can be fixed.
2. **Size each bucket** on originals only: listings, listings still open, companies and the top company's share, with and without manager-level listings (3.11; they're hidden per applicant, so the niche keeps them).
3. **Count career changers per bucket** from `Profiles.ipynb`'s modeling table (`career_changer`, by the role at the switch; unique people, after duplicate removal), split by pool: retrained after working, retrained as new grads, and changed roles within tech.
4. **Depth check per bucket:** split it as the in-depth analysis will (for example 6 months × 3 source types), and measure the 95% confidence-interval width of the share still open after 7 days and the median days open in each sub-group. Enough depth means about **±5 percentage points or narrower**, tight enough to show the 5–16 point differences the EDA found. Combine sub-groups that are wider.
5. **Keep a bucket** only if it passes the depth check on the listings side **and** has enough career changers to learn from (at least 30 unique people, counting **all three pools**). The 30 was first set for retrained people only, before the within-tech pool existed; counting retrained only would drop data and analytics (22 retrained, 238 in all), which the compass needs to show is reached through other tech roles. Thin buckets are combined with a close one or dropped (likely credentialed math, which has no resume titles). Record each keep, combine or drop with its reason.
6. **Save the niche** (kept buckets, blockers hidden) as its own table, the base for section 4 and EDA_SDA_3.

**Also described per bucket** (not used to choose buckets): variation in the factors (window, source, company volume, reposts, pay), coverage of key fields (description, salary, `posted`), words per description (EDA_SDA_1 Q3e), and openness to career changers (`entry_share`, `career_open_share`, years asked, degree required).

**Related groups:** the team's source-background groups stay in the table for comparison and role discovery: skill overlap through O*NET skills and the Career Changers Matrix, and TF-IDF similarity once EDA_SDA_3 exists.

### 3.7 Blind spots from Part 1

| Blind spot | Check or fix |
|:--|:--|
| The "closed" outcome is wrong for reposted jobs: the earlier listing disappears without being filled | In the analysis (not as a model input), treat a listing that was later reposted as "closed without filling" or censored |
| The 2-day "still open" tolerance is a guess | Distribution of days between `last_seen` and the snapshot; a bump near 7 days means it's too tight |
| Converting salaried pay to hourly assumes full time (2,080 hours) | Hourly listings keep their posted rate, so only salaried part-time listings are off; flag them with `contract_types` (part time) |
| Company volume counted within the sample | The companies table has no size or employee count. It does have a description (industry, via NLP in EDA_SDA_3), a ticker (publicly traded), parent and subsidiary companies, and similar companies: check whether these add anything beyond volume |
| State isn't in the table | Add it from `location_data`, if PredictLeads includes it |
| Missing country vs. remote, missing language vs. description | Confirm the EDA_SDA_1 section 10a results |
| `entry_seniority` counts only the `junior` label | Check every seniority label (EDA_SDA_1 Q3b) and add any that are entry level, such as intern, to `ENTRY_LEVELS` |

### 3.8 Joining the companies table

Merge the companies table into the listings (a **left join** on `company_id` = company `id`, so every listing stays). Course standard: one master table from all sources, with keys checked first. Join checks: company ids are unique in the companies table (no row multiplication), and match rate (100% in EDA_SDA_1). Columns to bring in: `ticker` (publicly traded), parent company, subsidiary and similar companies, location, and description (its text goes to EDA_SDA_3). The table has **no company size or employee count**.

**Then resolve listings with no country** (`country_missing`, kept by EDA_SDA_1): take the country from the company's `location_data`. Keep the listing if the company is US-based **or** the listing mentions remote work or uses the "remote" contract type, and flag it `country_inferred`; drop the rest. Report how many are kept by each route and how many are dropped, and check whether `country_inferred` listings behave differently (days open, source, words per description).

### 3.9 Missing-data patterns

**First classify each field** (structural, censored, or missing for a reason), then treat it. Most of ours aren't truly missing, so very little is imputed.

| Field | Missing (final table) | Why | Handling |
|:--|:--|:--|:--|
| `first_seen`, `last_seen`, title, seniority, occupation group, `onet_code` | ~0% | n/a | Nothing. A listing with no `onet_code` can't get a role bucket, so it's outside the niche (report the count) |
| `last_seen` of **live** listings | 23% still open | **Censored,** not missing | Never imputed or dropped; survival methods handle it |
| `platform` | 60.7% | **Structural:** company sites have no platform | Not missing; `source_type` says "company site / other" |
| `status` | 35% | Never says "open" | Not used; `last_seen` decides what's live |
| `posted` | 55% | Many sources publish no date | **Not imputed.** Freshness uses `first_seen`; day of week uses the "with `posted`" view; add `has_posted`, and check whether listings with and without it differ |
| Salary | 53% | **Missing for a reason** (pay-transparency laws, hourly vs. salaried) | **Pay level isn't filled in.** `salary_disclosed` stays its own factor. Trees use blanks natively; linear models get a median by role bucket × seniority **plus** a missing flag, fit inside the `Pipeline` on training months only. The app shows "not disclosed" |
| Description and its text flags | 9.8% | Some sources expose no text | Listing kept; text flags are **unknown** (blank), not False, since EDA_SDA_1's fix; `has_desc` is the flag |
| `years_required` | Blank when no years are stated | **"Not stated" isn't zero** | Kept blank, with a "states years" flag; not stated never counts as a blocker |
| Repost features | Blank without a description | Can't be matched | Kept blank, not 0 (0 would claim "first listing") |
| `tags` | 53% empty | Sparse | Not used; `categories` (99%) instead |
| Company description, location (joined in 3.8) | 10%, 6% | Not in the companies file | "Unknown" category |

**Course standard:** percent missing, missingness **by group** (source, occupation group, role bucket, month, state), and `df.isnull().corr()` to see which fields go missing together. Every imputation or "Unknown" category is justified with plots split by the outcome (closed vs. still open). **Rules:** impute only inside the model `Pipeline`, fit on the training months (leakage rules in `HOW_WE_WORK.md`), and add a missing flag whenever a value is filled in, since missingness itself can carry signal (salary disclosure does).

### 3.10 Outliers

Boxplots and IQR rules for days open, words per description, hourly pay, times listed. Course standard: inspect, explain, and keep or remove with a reason. Advanced: robust (median / IQR) summaries, and log scales for the right-skewed measures.

### 3.11 Career-change accessibility, and hiding what a career changer can't get

**Decided (2026-10-09): listings a career changer can't get are hidden.** Three kinds of blocker, each hidden where it's reliable:

| Blocker | Example | Hidden in |
|:--|:--|:--|
| Seniority, per applicant | Manager level and above (`manager`, `head`, `director`, `vice_president`, `president`, `c_level`, `partner`, `founder`; from EDA_SDA_1 Q3b) for applicants without management or senior experience | The app, per profile; this notebook's table keeps them, labelled |
| Text, for everyone | Too many years of experience asked; a required license | EDA_SDA_3's final table, after the rules are validated |
| Text, per applicant | A degree or certification this applicant doesn't have | The app, per profile |

Report how many listings each rule hides. Then check:

| Check | If the result is… | Then… |
|:--|:--|:--|
| Accessibility flags by company-volume band and by source | Entry-level listings cluster at high-volume employers or on certain platforms | Account for it, so accessibility isn't company size or source in disguise |
| Pay of listings open to career changers vs. the rest, within the niche | A large pay gap | Show it in the app, and let the pay weight trade off against accessibility |
| Days open by accessibility, within the niche | Reachable listings close faster | Raise their urgency in the ranking |
| Degree required by occupation group | Near-universal in a group, with little "or equivalent experience" | That group is a poor target |
| Each part of `career_open` separately | One part (for example, entry-level wording) adds little | Use a stricter rule for the "open to career changers" view |

### 3.12 The retraining compass: the 2026 side

For each kept role bucket, after blockers are hidden: **open listings at the snapshot, new listings per week, median pay (where disclosed), median days open (Kaplan–Meier), entry-level share, and expected job growth** (official projections by occupation code; time frame to be decided). Rows in the same role order as Profiles (`ROLE_ORDER` in `src/career_switch.py`: infrastructure → code → data).

Join it by role bucket to Profiles' three path tables, so every path shows its 2026 openings, pay, closing speed and growth:
- **Prior career → first tech job** and **field trained in → first tech job** (4a.2b): the compass for changing careers.
- **Tech job before → after the move** (4a.3a), with the sideways-or-up split: the compass for moving across or up within tech.

Every path carries its people count, and paths under about 10 people are marked thin. Growth sits next to the entry-level share, since projections cover the whole occupation. Aggregates only; this table feeds the app's Retraining compass and the deck (`DIFFERENTIATORS.md`).

## 4. EDA_SDA_2 final table

_To do after section 3:_ defend each decision with its result from section 3, then apply them all at once, at the end: the niche (computer and mathematical, kept role buckets), labelling manager-level listings (hidden per applicant in the app), the additional cleaning, each factor kept or dropped, and the ghost-listing decision. Record every step with its reason, in the same format as EDA_SDA_1 section 10 and 11. Save it as a separate file, so EDA_SDA_1's table stays unchanged.

## 5. Conclusion and next steps

_To do._ Next: `EDA_SDA_3.ipynb`, the NLP features, built on this notebook's niche table.
