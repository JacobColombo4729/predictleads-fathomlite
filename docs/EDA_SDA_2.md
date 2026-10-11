<!-- Generated from notebooks/EDA_SDA_2.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 2 (Non-Text Analysis)

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

`EDA_SDA_1.ipynb` cleaned the full file, compared every field and cut the scope: **the roles TripleTen trains for, plus adjacent tech roles** (about 1.0M listings). This notebook does the **non-text** in-depth analysis of that scope: additional cleaning (staffing agencies, contract types, titles, the app's filter fields), ghost listings vs. employers still hiring, source, freshness, pay, reposts and copies, company facts, missing data, outliers, what a career changer can reach, and the **retraining compass's 2026 side** by role bucket and TripleTen program. It saves **its own final table** for `EDA_SDA_3.ipynb` (text), then `EDA_SDA_4.ipynb` (statistics), `TimeSeries`, `ML_Fit` and `ML_Rank`.

**Inputs from EDA_SDA_1:** the scope table (`analysis_table.parquet`: English, US or a US-based company, the scope filter applied; every candidate factor; role buckets; `tripleten_program`; `company_earlier` in any field; crawl-artifact flags), and company activity in any field (`company_activity.parquet`).

**Conventions:**
- Every decision is recorded with its reasoning; fix or flag, never drop.
- Every factor is checked against the confounders (section 2): within groups, not by filtering.
- Aggregates only (shares, medians, compact counts); never rows, except the cell tagged `data-rows` (the 2b.1 review of flagged company names): clear it before committing.
- Each table or chart group ends with **What the data shows**: up to three bullets.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| **EDA_SDA_2 (you are here)** | **Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings** | **Scope table, non-text features** |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

### Clarifying questions
- Which long-open listings are ghost listings, and which are employers still trying to hire?
- Does source matter beyond company hiring volume, or is it company volume in disguise?
- How precise is freshness, given the crawler finds listings a median of 1.5 days late?
- Is pay disclosure driven by state pay-transparency laws or job type, rather than employer intent?
- Does high company volume mean more competition, or more openings to win? And are fast-closing recruiters just staffing agencies?
- How are values missing: at random, or tied to groups (source, role bucket, state)?
- How much of the scope is left once listings a career changer can't get are hidden?
- Which role buckets have enough listings and enough career changers for their own analysis?

### Roadmap
1. **Setup:** EDA_SDA_1's scope table and company activity
2. **Analysis views and confounders;** **additional cleaning** (2b): staffing agencies and recruiters, dates, contract types and job type, titles (with an AI/ML flag), multi-location, the app's filter fields
3. **Checks and analyses** (non-text), ending with the compass's 2026 side
4. **EDA_SDA_2 final table:** decisions defended, then applied, with a reason for each step
5. **Conclusion and next steps** (into EDA_SDA_3)

**What the data shows**
- **The scope from EDA_SDA_1: 1.0M listings (947K originals) from 84K companies;** 24.4% still open on September 30.
- **58.6% are under a TripleTen program;** the rest are adjacent tech roles. 9.9% have no country on the listing and come from a US-based company.

## 2. Analysis views

Some analyses are skewed by part of the table. Each analysis states which **view** it uses:

| View | Rows | Use it for | Why |
|:--|:--|:--|:--|
| **All listings** | Everything in the final table, reposts included | Ranking listings as an applicant sees them; the repost features | Each listing is a row the app could show |
| **Originals only** | Reposts (`is_repost`) excluded | Counts, volumes, application windows, company concentration | A reposted job counted twice inflates volumes and splits one long-open job into two short ones |
| **With `posted`** | About 39% of scope listings | Day-of-week patterns; crawler lag | `first_seen` days follow the crawler's schedule |
| **With a salary** | About 45% of scope listings | Pay level and pay comparisons | Pay is missing for the rest |
| **Closed listings** | Not still open on September 30 | Simple descriptive durations | Survival methods use every listing; this view is only for plain summaries |
| **Long-open listings** | Open for a long time (threshold set in 3.1) | Ghost listings vs. desperate employers | The question only applies to them |
| **Without likely ghosts** | Every listing except `likely_ghost` (3.1d) | Every timing result (days open, share still open, open on September 30) is shown both with and without them, from 3.1 on | A forgotten listing's "days open" isn't a real hiring window, so ghosts can stretch timing results; they stay in the data either way |
| **Open to career changers** | `career_open`, or the stricter rule chosen in 3.11 | Accessibility comparisons within the scope; against other fields through EDA_SDA_1's section 7 | Shows whether reachable listings behave differently |
| **The scope** | EDA_SDA_1's scope table: the roles TripleTen trains for plus adjacent tech roles, by role bucket (3.6) and TripleTen program | Everything in this notebook | Focus |
| **Other fields** | EDA_SDA_1's section 7 and its occupation summary (aggregates; no listings) | Comparing the scope with other fields; pay in a user's current field for the compass | The listings themselves stay in EDA_SDA_1's cleaned rows |

### Confounders: compare within groups, don't filter

Some factors travel together, so a raw gap can belong to something else. **Every factor test in section 3 is checked against these,** by comparing within groups (stratifying), not by filtering them out: filtering throws data away and only answers the question for what's left. Filtering is used only when a slice can't answer the question at all (for example day of week without `posted`).

| Confounder | Why it matters here | How each check handles it |
|:--|:--|:--|
| **Company hiring volume** (earlier listings only, bands 1 / 2–4 / 5–19 / 20+) | The strongest signal; it overlaps with source, pay disclosure and staffing agencies | Every factor is compared within volume bands |
| **Source and platform** | Platforms show closures to the crawler differently | Within source type, and within platform where there are enough listings |
| **Staffing agencies** (`is_staffing`, 2b.1) | They post at volume and repost often | Gaps shown with and without them |
| **Role bucket and seniority** | A senior data scientist role and an IT support role close at different speeds for unrelated reasons | Within role bucket (3.6) where buckets are big enough; seniority as a split |
| **Month first seen** | Later months have had less time to close | Survival methods (still-open listings censored); month as a split for timing gaps |
| **State** (3.7d) | Pay-transparency states disclose pay far more often and have longer windows | Pay-disclosure gaps checked within state |
| **Company type** (`is_public`, `has_parent`, 3.8) | At 20+ listings, public companies' listings stay open 20 days vs. 6 for others | Volume effects shown separately for public and other companies |
| **No description** (`has_desc`, 3.9) | 23% of listings, mostly from June on; they close in 3 days vs. 30 | Timing results shown with and without them |
| **Ghost listings** (`likely_ghost`, 3.1d) | Forgotten listings inflate days open and the count still open | Timing results also shown without likely ghosts |

**The rule:** each factor shows the **raw gap**, then the **gap within volume band × source type**, and within role bucket where there are enough listings. A factor counts as real only if the gap holds within the groups. EDA_SDA_4 then confirms the survivors with one model that adjusts for all of these at once.

## 2b. Additional cleaning

**Decided (2026-10-09):** five fixes and the app's filter fields here (salary sanity moved into EDA_SDA_1 6.5, description cleanup to EDA_SDA_3), run before the checks in section 3 so they use clean pay, dates and titles. Same rules as EDA_SDA_1: **fix or flag, never drop** (a flagged listing leaves only the analyses it would distort), keep the original column next to every cleaned one, and report how many listings each step changes (counts only). Spot-checks of examples are done locally by a teammate.

| # | Fix | How | Result |
|:-:|:--|:--|:--|
| 1 | **Staffing agencies** | Flag listings posted by recruiters or staffing firms: company name and domain words (staffing, recruiting, talent, well-known agency names), or an "about" text that says staffing or recruiting firm (companies table). A review cell (2b.1 review, row-level) lists the most-flagged companies so a teammate can check the keywords | `is_staffing`, kept. Report their share by field and their days open; compute company volume with and without them (3.5), since agencies post at high volume and repost often. The app labels them "posted by a recruiter" |
| – | **Salary sanity, standardized to hourly** | **Moved to EDA_SDA_1 6.5:** all pay cleaning in one place. `salary_hourly_usd` (salaried pay ÷ 2,080 hours) is the one pay column for the whole project; reversed ranges and clearly wrong units are fixed there and flagged `pay_suspect` | Arrives clean in EDA_SDA_1's table |
| 2 | **Date consistency** | EDA_SDA_1 (6.3) already removed impossible timelines and blanked `posted` dates after `first_seen`. Here: `posted` more than a year before `first_seen` | `posted_old` flag, kept; it feeds the ghost-listing check (3.1) |
| 3 | **Contract types, internships and temporary roles** | Contract labels (unified by EDA_SDA_1) split into employment-type flags, several of which can be true, plus title words (intern, temporary, seasonal, contract-to-hire) | `is_full_time`, `is_part_time`, `is_contract`, `is_temporary`, `is_internship`; and `job_type` (internship, temporary, contract, regular), one label for the app's filter. A table of the common label combinations shows the conflicts |
| 4 | **Title normalization** | Lowercase; strip bracketed text, locations after " - " or " \| ", requisition IDs, pay, and "remote" or "hybrid" tags (kept as a flag) | `title_clean`; the original title stays. Recheck the repost count with it. **`ai_ml_title`:** the title names AI, machine learning or data science (the hackathon app's AI/ML audience, across buckets) |
| 5 | **Multi-location listings** | `n_locations` (from EDA_SDA_1) above 1 | `multi_location` flag; one row per listing for timing. The state list for the app's location filter comes with `location_data` in 3.7 |
| 6 | **Filter fields for the app** | Role level from `seniority` (entry = junior; `mid_senior` split into mid and senior by title words, the Profiles rule; manager; leadership = director and above); where the work happens from contract labels ("fully remote", "hybrid", "on site"...) and title words; schedule from the employment flags | `level`, `work_arrangement` (remote, hybrid, onsite, several, not stated), `schedule` (full time, part time, not stated); with `job_type` (2b.3) and the state list (3.7) they back every filter in the app |
| – | **Description cleanup** | **Moved to EDA_SDA_3:** this table keeps only description features (`desc_words`, `desc_hash`, `has_html`), not the text, to fit in memory; EDA_SDA_3 loads the text anyway | `desc_clean` there, with HTML and boilerplate removed |

**Leakage:** every fix uses only what the listing showed when it was posted, so none of them leaks the outcome.

**What the data shows**
- **Staffing agencies are a small, fast group:** 2.0% of listings (800 companies flagged), closing in a median 13.7 days vs. 23.5 for employers. **Recruiters aren't just staffing agencies:** among employers, listings from recruiters with 5+ listings close in 7.6 days vs. 24.1 with no recruiter named (2b.1b), though only 1.2% of listings name one.
- **Contract labels:** 53% full time, 9.7% contract, 3.7% part time, 3.3% internship, 19% none. Full and part time together is rare (1.1%). Where the work happens is stated for 53% (remote 19%, onsite 14%, hybrid 12%, several 7.8%), and remote labels agree with the description 95% of the time.
- **Titles and levels:** cleaning changed 26% of titles; **9.1% name AI, machine learning or data science** (26% of data science, 15% of networks and systems). Role levels: mid 47%, senior 30%, entry 10%, manager 9.3%, leadership 3.5%. No listing names more than one location: PredictLeads lists each location separately.

## 3. Checks and analyses

Record each result and the decision it leads to here, then apply it when building the final table (section 4). **From 3.1 on, every timing result is also shown without likely ghosts** (section 2's views). **Every factor test follows the confounders rule in section 2:** raw gap, then the gap within volume band × source type (and role bucket where possible); only gaps that hold count.

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

**What the data shows**
- **Closing slows with age:** of listings first seen in April–May, 52% close in their first 30 days; past 60 days, 28–34% of those still open close each month. Old listings still close, so age alone isn't a ghost.
- **19% of open listings are older than 90 days** (45K of 237K); open job-board listings are almost all old (95%), since job boards stop after May (3.2c).
- **10K likely ghosts** (1.1% of the scope, 4.4% of open listings): stale and from a company with no new listing in 60 days, or with a posting date over a year old. The other stale listings are no more open to newcomers (17% vs. 21% for younger open listings), so "possibly hard to fill" is dropped as a label.

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
- **Source matters beyond company volume, and the two interact:** at companies with 500+ listings, company-site listings close in a median 7 days vs. 21 on hiring platforms; at companies with 1–4, 70 vs. 46. Job boards sit at 12–18 days in every band.
- **Platforms differ widely:** Built In, Indeed, SmartRecruiters, Taleo, Oracle and LinkedIn close in 10–16 days; Greenhouse, Breezy, Lever and Ashby in 49–71, partly in how they show closures to the crawler.
- **Job boards stop after May** (about 30K listings a month in April–May, about 200 a month after): a collection change to ask PredictLeads about. **38.6% of all scope listings are on the employer's own domain,** so the app can link many straight to the employer.

### 3.3 Freshness and crawler lag

| Check | If the result is… | Then… |
|:--|:--|:--|
| Share of listings found more than 7, 14 and 30 days after `posted` | Tiny | Use `first_seen` only |
| | Meaningful | Add a "found late" flag |
| Survival curves: found late vs. found promptly | Found-late listings close much sooner | The flag matters for apply-by estimates |
| Survival curves: listings with vs. without a `posted` date | Similar | `posted` can be used where present |
| | Different | Stick with `first_seen` only |

**Also (added 2026-10-10): are high-volume companies crawled more often?** The crawler's visit times aren't in the data, so crawler lag (posting date → first seen) stands in for crawl frequency, and the company-volume effect from 3.5a (7 vs. 39 days) is re-measured from the employer's posting date. If it shrinks a lot, part of it was crawl timing.

**What the data shows**
- **The crawler is prompt:** a median 1.5 days from the employer's posting date to first seen (1.1 on hiring platforms, 3.2 on job boards); 34% found within a day.
- **Busy companies are crawled more often** (company sites: a median lag of 9.9 days at 1–4 listings vs. 1.2 at 500+), which inflates their speed. **Like-for-like, measured from the posting date, the volume effect is about 1.5×** (32 vs. 22 days), not the 5× seen from first seen.
- **Late-found listings** (7+ days after posting) have long windows from the posting date (48 days), so freshness uses `first_seen` with `posted` alongside.

### 3.4 Pay disclosed

| Check | If the result is… | Then… |
|:--|:--|:--|
| Disclosure rate by US state (needs the state from `location_data`) | Much higher in pay-transparency states (CA, CO, NY, WA) | Add state, or a "pay-law state" flag |
| Disclosure by hourly vs. salaried, and by seniority | Strongly tied to hourly | Its effect may really be job type |
| Days open by disclosure, within company-volume bands | The gap holds | Keep as a factor |
| | The gap disappears | Keep only as the companion flag for pay level |

**As built (2026-10-10):** state comes later (3.7, from `location_data`), so 3.4 uses role level, source, company volume and role bucket. Disclosure is compared within company-volume bands and without likely ghosts (the confounders rule).

**What the data shows**
- **Disclosure tracks source more than role level:** hiring platforms disclose pay for 52–63% of listings, company sites 27–49%, job boards 12–16%; by level, 42–53%.
- **Disclosed listings stay open longer at busy companies** (22 vs. 6 days at 500+ listings); much of the "not disclosed" group is listings with no description, which close in days (3.9).
- **Pay by bucket:** software development and IT project management lead (median $72–73 an hour); IT support is lowest ($31), business and data analysis $42.

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
- **Company volume is steady and real:** median days open of 50, 35, 28 and 10 across the four bands (earlier listings in any field), the same without staffing agencies and nearly the same without likely ghosts. Entry level is about 10% in every band, so volume is about speed, not access.
- **Copies of the same job:** 70% of listings are single; 21% are same-source copies (most likely the same job in another city); 7.9% are relisted after earlier copies closed; 1.1% are cross-posted.
- **Relisted copies close fast** (median 7.9 days): most likely crawler refreshes, not new hiring. Cross-posts barely differ from single listings, so the urgency idea finds little support.

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
| business and data analysis (added 2026-10-10) | 13-1111 management analysts, 13-1161 market research analysts | (none) |
| credentialed math | 15-2011 actuaries, 15-2021 mathematicians | (none) |

The listing side is `role_bucket(onet_code)` (`ONET_DETAILED_BUCKETS` for detailed 15-1299 codes first, then `ONET_ROLE_BUCKETS`, which also covers the older 2010 codes) and the resume side is `RULES["cm_buckets"]`, both in `src/career_switch.py`.

**Steps:**
1. **Map** every computer and mathematical listing to a role bucket from `onet_code`, and report unmapped codes (aggregate counts) so the mapping can be fixed.
2. **Size each bucket** on originals only: listings, listings still open, companies and the top company's share, with and without manager-level listings (3.11; they're hidden per applicant, so the scope keeps them).
3. **Count career changers per bucket** from `Profiles.ipynb`'s modeling table (`career_changer`, by the role at the switch; unique people, after duplicate removal), split by pool: retrained after working, retrained as new grads, and changed roles within tech.
4. **Depth check per bucket:** measure the 95% confidence-interval width of the Kaplan–Meier share still open after 7 days in each sub-group. Enough depth means **±5 points or narrower** (a width of 10 or less), tight enough to show the 5–16 point gaps the EDA found, in at least 80% of sub-groups. Splits are tried from finest to coarsest, combining sub-groups that are too wide: month × source type, then source type, then month, then the whole bucket. The finest split that passes is how finely that bucket is analyzed in section 3. (About 300+ listings per sub-group are needed for ±5, so most buckets won't support all 18 month × source sub-groups.)
5. **Keep a bucket** only if it passes the depth check on the listings side **and** has enough career changers to learn from (at least 30 unique people, counting **all three pools**). The 30 was first set for retrained people only, before the within-tech pool existed; counting retrained only would drop data science (22 retrained, 238 in all), which the compass needs to show is reached through other tech roles. Thin buckets are combined with a close one or analyzed only as part of the scope; their listings always stay in the table. **Credentialed math** (actuaries, mathematicians) is out of the scope from 1d: credential-gated, with no career switches into it. **IT project managers** (O*NET 15-1299.09, about 13% of the scope) have their own bucket since 2026-10-10, on both sides (a Profiles rules change). Record each keep, combine or drop with its reason.
6. **Save the scope** (kept buckets, blockers hidden) as its own table, the base for section 4 and EDA_SDA_3.

**Also described per bucket** (not used to choose buckets): variation in the factors (window, source, company volume, reposts, pay), coverage of key fields (description, salary, `posted`), words per description (EDA_SDA_1 Q3e), and openness to career changers (`entry_share`, `career_open_share`, years asked, degree required).

**Related groups:** the team's source-background fields were set aside in EDA_SDA_1's cut; they live on in its occupation summary (pay, timing, openness) and through O*NET skills and the Career Changers Matrix for role discovery. **The two added buckets** (technical sales, business and data analysis) have no career-switch paths in the frozen profiles, so 3.6e marks them for combined analysis; their listings stay, and the compass shows their 2026 market.

**What the data shows**
- **Every bucket is deep enough on the listings side** (7-day intervals of 10 points or less in at least 80% of sub-groups, by source type or finer).
- **Eight buckets get their own analysis and compass paths** (30+ career changers each; quality assurance is the thinnest at 39). Technical sales, business and data analysis and other computer roles have no career-switch paths in the frozen profiles, so they're analyzed combined; their listings stay.
- **IT project management is large** (116K originals; 15-1299.09) but mostly reached from another tech role (190 of its 264 career changers).

### 3.7 Blind spots from Part 1

| Blind spot | Check or fix |
|:--|:--|
| The "closed" outcome is wrong for reposted jobs: the earlier listing disappears without being filled | In the analysis (not as a model input), treat a listing that was later reposted as "closed without filling" or censored |
| The 2-day "still open" tolerance is a guess | Distribution of days between `last_seen` and September 30 (the data's end date); a bump near 7 days means it's too tight |
| Converting salaried pay to hourly assumes full time (2,080 hours) | Hourly listings keep their posted rate, so only salaried part-time listings are off; flag them with `contract_types` (part time) |
| Company volume counted within the sample | The companies table has no size or employee count. It does have a description (industry, via NLP in EDA_SDA_3), a ticker (publicly traded), parent and subsidiary companies, and similar companies: check whether these add anything beyond volume |
| State isn't in the table | Add it from `location_data`, if PredictLeads includes it |
| Missing country vs. remote, missing language vs. description | Confirm the EDA_SDA_1 section 10a results |
| `entry_seniority` counts only the `junior` label | Check every seniority label (EDA_SDA_1 Q3b) and add any that are entry level, such as intern, to `ENTRY_LEVELS` |

**As built (2026-10-10):** 3.7a–3.7e cover the live tolerance, salaried part-time pay (`pay_part_time_salaried`), the seniority labels and state (`state`, from the location records), with state-level pay. The reposted-closing question is answered by 3.5 (relisted copies close in 3.5 days, read as crawler refreshes); the companies table's extra fields and the no-country check move to 3.8.

**What the data shows**
- **The 2-day "still open" tolerance holds:** 237K listings were last seen within 2 days of September 30, then the counts drop (9K at 2–3 days) with no bump near 7 days.
- **Seniority labels mostly agree with titles,** but `mid_senior` hides seniors: 35% of its titles say senior, lead, principal or staff, so 2b.6's split is needed. Internships are mostly entry level (28K of 35K).
- **State is known for 81% of listings.** California (12%), Texas (7.9%) and New York (6.7%) lead. **Pay-transparency states disclose pay far more often** (Colorado 68%, Washington 67%, Massachusetts 66%, California 63% vs. Texas 30%, Florida 32%, Georgia 29%).

### 3.8 Joining the companies table

Merge the companies table into the listings (a **left join** on `company_id` = company `id`, so every listing stays). Course standard: one master table from all sources, with keys checked first. Join checks: company ids are unique in the companies table (no row multiplication), and match rate (100% in EDA_SDA_1). Columns to bring in: `ticker` (publicly traded), parent company, subsidiary and similar companies, location, and description (its text goes to EDA_SDA_3). The table has **no company size or employee count**.

**Listings with no country** were resolved in EDA_SDA_1 8b (kept only when the company is based in the US); 3.8c checks whether they behave differently.

**What the data shows**
- **The companies table matches every scope company** (84,181 of 84,189); 5.0% of companies are publicly traded and 10.8% have a parent company.
- **The volume effect is a private-company pattern:** at 500+ listings, public companies' listings stay open a median 20 days vs. 7 for others.
- **No-country listings** (10% of the scope, kept by EDA_SDA_1 because their company is in the US) are more often remote (38% vs. 17%) and rarely have a description (37% vs. 81%).

### 3.9 Missing-data patterns

**First classify each field** (structural, censored, or missing for a reason), then treat it. Most of ours aren't truly missing, so very little is imputed.

| Field | Missing (final table) | Why | Handling |
|:--|:--|:--|:--|
| `first_seen`, `last_seen`, title, seniority, occupation group, `onet_code` | ~0% | n/a | Nothing. A listing with no `onet_code` can't get a role bucket, so it's outside the scope (report the count) |
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

**Course standard:** percent missing, missingness **by group** (source, occupation group, role bucket, month, state), and `df.isnull().corr()` to see which fields go missing together. Every imputation or "Unknown" category is justified with plots split by the outcome (closed vs. still open). **Rules:** impute only inside the model `Pipeline`, fit on the training months (leakage rules in `STANDARDS.md`), and add a missing flag whenever a value is filled in, since missingness itself can carry signal (salary disclosure does).

**What the data shows**
- **Most blanks are structural or "not stated":** platform (64%: company sites have none), posting date (61%), years asked (57%), pay (55%), recruiter (99%).
- **No description is the key missing field:** 22% of the scope, rising from 11% in April to 28–30% from June, highest on company sites (33%). These listings close in a median 4.4 days vs. 30, so missing pay or years looks predictive only through them; `has_desc` stays a standard confounder.
- **Missingness tracks source and month, not role bucket** (17–28% no description across buckets). Nothing is imputed here.

### 3.10 Outliers

Boxplots and IQR rules for days open, words per description, hourly pay, times listed. Course standard: inspect, explain, and keep or remove with a reason. Advanced: robust (median / IQR) summaries, and log scales for the right-skewed measures.

**What the data shows**
- **The long tails are real, not errors:** 6.1% of closed listings run past the IQR fence (85 days), none past the log-scale fence; descriptions over about 1,660 words are 1.1%.
- **Pay outliers are mostly senior roles:** 1.4% are above $145 an hour, 65% of them senior or above. 326 listings under the federal minimum wage and 264 over $250 an hour are flagged and their hourly pay blanked (0.06%).
- **Times listed:** 92% are first listings, 3.1% have been listed 5+ times.

### 3.11 Career-change accessibility, and hiding what a career changer can't get

**Decided (2026-10-09): listings a career changer can't get are hidden.** Three kinds of blocker, each hidden where it's reliable:

| Blocker | Example | Hidden in |
|:--|:--|:--|
| Seniority, per applicant | Manager level and above (`manager`, `head`, `director`, `vice_president`, `president`, `c_level`, `partner`, `founder`; from EDA_SDA_1 Q3b) for applicants without management or senior experience | The app, per profile; this notebook's table keeps them, labelled |
| Text, for everyone | Too many years of experience asked; a required license | EDA_SDA_3's final table, after the rules are validated |
| Text, per applicant | A degree or certification this applicant doesn't have | The app, per profile |
| Occupation, for everyone | Document management specialists (O*NET 15-1299.03): records work, not a tech path (decided 2026-10-10) | The app; flagged `hidden_occupation` in 3.0 and kept in the table |

Report how many listings each rule hides. Then check:

| Check | If the result is… | Then… |
|:--|:--|:--|
| Accessibility flags by company-volume band and by source | Entry-level listings cluster at high-volume employers or on certain platforms | Account for it, so accessibility isn't company size or source in disguise |
| Pay of listings open to career changers vs. the rest, within the scope | A large pay gap | Show it in the app, and let the pay weight trade off against accessibility |
| Days open by accessibility, within the scope | Reachable listings close faster | Raise their urgency in the ranking |
| Degree required by role bucket | Near-universal in a bucket, with little "or equivalent experience" | That bucket is a poor target |
| Each part of `career_open` separately | One part (for example, entry-level wording) adds little | Use a stricter rule for the "open to career changers" view |

**What the data shows**
- **What a career changer without management experience doesn't see: 15% of open listings** (likely ghosts 4.4%, manager level or above 11%, document management 0.4%). The "asks 5+ years" rule would hide another 30%, so it waits for EDA_SDA_3's validation; "degree required, no equivalent" fires on only 2.4%.
- **Openness isn't company size or source in disguise:** 18–28% show any sign in every volume band × source cell. Seniority and years asked do most of the work (each is the only sign for 61–76% of its listings).
- **Reachable listings pay $9–35 an hour less in every bucket** (smallest in IT support, largest in other computer roles), and close a little faster at small companies (32 vs. 42 days at 1–4 listings).

### 3.12 The retraining compass: the 2026 side

For each kept role bucket, after blockers are hidden: **open listings on September 30, new listings per week, median hourly pay (where disclosed), median days open (Kaplan–Meier), entry-level share, openness to career changers, remote share, and expected job growth** (a placeholder column until the projections are downloaded) (official projections by occupation code; time frame to be decided). Rows in the same role order as Profiles (`ROLE_ORDER` in `src/career_switch.py`: infrastructure → code → data).

Join it by role bucket to Profiles' three path tables, so every path shows its 2026 openings, pay, closing speed and growth:
- **Prior career → first tech job** and **field trained in → first tech job** (4a.2b): the compass for changing careers.
- **Tech job before → after the move** (4a.3a), with the sideways-or-up split: the compass for moving across or up within tech.

Every path carries its people count, and paths under about 10 people are marked thin. Growth sits next to the entry-level share, since projections cover the whole occupation. Its source and method are in `docs/job_growth.md` (BLS 2025–2035, employment-weighted by bucket, at most 5 years). Aggregates only; this table feeds the app's Retraining compass and the deck (`DIFFERENTIATORS.md`).

**What the data shows**
- **The 2026 market by bucket:** software development (74K open, 11K new a week, $73.20 an hour) and networks and systems (44K open) are the largest; IT support is the door (44% open to career changers, 23% entry level, $31.00); technical sales is the slowest (median 38 days) but the most remote (29%).
- **By TripleTen program:** AI Software Engineering (71K open) and AI & Machine Learning pay the most ($73.50); **Data Analytics roles are the most open** (30%) at $47.60; adjacent tech roles hold 96K open listings.
- **Most paths are thin:** 78% of prior-career paths and 67% of training paths rest on fewer than 10 people (within-tech moves: 43%), so the compass always shows how many people a path rests on. The largest within-tech move is IT support → networks and systems (279 people).

## 4. EDA_SDA_2 final table

Every decision in this notebook, defended with its result from sections 2b and 3, then applied all at once in 4b. The table is saved as its own file (`scope_table.parquet`), so EDA_SDA_1's table stays unchanged. **Hide, never remove:** no listing is dropped here (the scope cut and the US rule are EDA_SDA_1's); everything is a flag the app hides or labels by.

### 4.0 Decisions

| Decision | Evidence (full scale) | Choice | Status |
|:--|:--|:--|:--|
| **The scope** | The roles TripleTen trains for plus adjacent tech roles (EDA_SDA_1 8.0, 8c): 1.0M listings | As cut by EDA_SDA_1 | Applied in EDA_SDA_1 |
| **Role buckets** | Every bucket is deep enough on the listings side; eight have 30+ career changers (3.6) | Eight buckets with their own analysis and compass paths; technical sales, business and data analysis and other computer roles analyzed combined (no career-switch paths); document management flagged `hidden_occupation` | Decided; listings always kept |
| **IT project management** | 9.2% open to career changers, 2.1% entry level, 44% manager level; 190 of its 264 career changers came from another tech role (3.6, 3.11) | Kept as a later move: not suggested as a first tech job unless the prior career has a real path into it | Decided (app and ML_Fit) |
| **Listings with no country** | 10% of the scope, all from US-based companies; more often remote (38% vs. 17%), rarely with a description (37%) (3.8) | Kept, flagged (`country_inferred`) | Applied in EDA_SDA_1 8b |
| **Likely ghosts** | Closing slows with age but never stops; 19% of open listings are older than 90 days; 10K likely ghosts (4.4% of open) (3.1) | Flag `likely_ghost`, hidden in the app; other stale listings labelled "open 90+ days" and ranked lower | Decided; timing also shown without them |
| **Manager level or above** | 11% of open listings (3.11) | Labelled, hidden per applicant | Decided (app) |
| **Text blockers** | "Asks 5+ years" would hide 30% of open listings; "degree required, no equivalent" only 2.4% (3.11) | Previews only | Open: EDA_SDA_3 validates them first |
| **Staffing agencies and recruiters** | 2.0% of listings, closing in 13.7 vs. 23.5 days; recruiters with 5+ listings close fast even at employers (7.6 vs. 24.1 days) (2b.1) | Flag `is_staffing` (the app labels "posted by a recruiter"); `recruiter_key` stays a candidate feature | Decided |
| **Contract types and where the work happens** | 53% full time, 9.7% contract, 3.3% internship; work arrangement stated for 53% (2b.3, 2b.6) | Employment flags, `job_type`, `work_arrangement`, `schedule` for the app's filters | Decided |
| **AI/ML titles** | 9.1% of listings, across buckets (26% of data science) (2b.4) | `ai_ml_title` as a view for the hackathon app's AI/ML audience | Decided |
| **Pay** | 326 listings under $7.25 and 264 over $250 an hour (0.06%); 1,455 part-time listings with salaried pay (3.7, 3.10) | Flag `pay_implausible` and blank its hourly pay; flag `pay_part_time_salaried` | **Applied in 4b** |
| **Missing values** | Mostly structural or "not stated"; no description for 22% (28–30% from June), closing in 4.4 days (3.9) | Nothing imputed here; `has_desc` a standard confounder; `has_posted` added | Decided |
| **Outliers** | Real tails (3.10) | Keep; times listed grouped 1 / 2–4 / 5 or more (`times_listed_band`) | **Applied in 4b** |
| **"Possibly hard to fill"** | No more open to newcomers than younger listings (17% vs. 21%) (3.1) | Dropped as a separate label | **Applied in 4b** |

### 4.0 Prioritization factors, after this notebook

| Factor | Result (full scale) | Choice | Status |
|:--|:--|:--|:--|
| Company hiring volume | 10 vs. 50 days (500+ vs. 1–4, from first seen); about 1.5× like-for-like from the posting date; a private-company pattern (public 20 vs. others 7 days at 500+) (3.3, 3.5, 3.8) | Keep (`company_earlier`, `volume_band`) | Kept; EDA_SDA_4 tests volume × company type |
| Source and platform | Interacts with volume (company sites 7 vs. 70 days); platforms 10–71 days; 38.6% on the employer's own domain; job boards stop after May (3.2) | Keep `source_type`, `platform`, `on_employer_domain` | Kept; a standard confounder |
| Freshness | Median crawler lag 1.5 days; late-found listings have long windows from posting (3.3) | Keep `first_seen`, with `posted` and `has_posted` | Kept |
| Pay disclosed | Driven by source and state (pay-transparency states 63–68% vs. about 30%) (3.4, 3.7) | Keep, compared within state and source | Kept; EDA_SDA_4 tests it within state |
| Pay level | Reachable listings pay $9–35 an hour less in every bucket (3.11) | Keep, hourly | Decided; the app shows the trade-off |
| Recruiters | Fast closing, beyond staffing agencies, but only 1.2% of listings (2b.1b) | Keep `recruiter_key` as a candidate | For the feature screening |
| Copies and reposts | 70% single, 21% same-source copies, 7.9% relisted (close in 7.9 days), 1.1% cross-posted (3.5) | Keep `copy_type` and the repost features | Kept; little urgency signal |
| Company type | Public companies' listings stay open longer at high volume (3.8) | Keep `is_public`, `has_parent` | Kept; a standard confounder |
| State | Known for 81%; pay transparency shows (3.7) | Keep `state` | Kept; the app's location filter and a confounder |
| Career-change accessibility | 20% show any sign; similar in every volume band × source cell; seniority and years asked do most of the work (3.11) | Keep `career_open` and its parts | Kept; EDA_SDA_3 validates the text parts |

**What the final table shows**
- **The final scope table: 1.0M listings (947K originals) from 84K companies in 11 role buckets, 105 columns** (`scope_table.parquet`, 0.25 GB). Nothing was removed; every decision is a flag.
- **23.2% are open on September 30 and visible to everyone** (no likely ghost or hidden occupation). Flags kept: likely ghosts 1.1%, open 90+ days 4.7%, staffing agencies 2.0%, no country (company in the US) 9.9%, implausible pay 0.06%.
- **The added fields are complete** except state (81%) and company country (93%).

## 5. Conclusion and next steps

**The scope:** 1.0M listings (947K originals) from 84K companies in 11 role buckets, cleaned, flagged and saved for EDA_SDA_3 (`scope_table.parquet`). Nothing was removed; every decision is a flag. The full data confirms the sample's main lesson: **how the data was collected explains many of the biggest gaps.** Crawl timing, job boards stopping after May and listings with no description (22%, closing in about 4 days) all look like signals until they're checked, so "has a description", company type and state stay standard confounders.

### Answers to the clarifying questions

| Question | Answer |
|:--|:--|
| Which long-open listings are ghosts, and which are still hiring? | Company activity separates them. **10K likely ghosts are hidden** (4.4% of open listings); other listings open 90+ days stay visible, labelled and ranked lower, since 28–34% of old open listings still close each month (3.1) |
| Does source matter beyond company volume? | Yes, and the two interact: at 500+ listings, company-site listings close in 7 days vs. 21 on hiring platforms; at 1–4, 70 vs. 46. Platforms range from 10 to 71 days (3.2) |
| How precise is freshness? | A median 1.5 days late; busy companies are crawled more often, so like-for-like the volume effect is about 1.5×, not 5× (3.3) |
| Is pay disclosure about state laws or source rather than intent? | Largely: pay-transparency states disclose 63–68% vs. about 30% elsewhere, hiring platforms 52–63% vs. job boards 12–16% (3.4, 3.7) |
| Competition or more openings? Are fast recruiters just staffing agencies? | Busy employers fill faster, about a third like-for-like, mostly private companies; the data has no applicant counts, so volume reads as "apply sooner". Recruiters close fast even at employers (7.6 vs. 24.1 days), not only staffing agencies (2b.1b, 3.3, 3.5, 3.8) |
| How are values missing? | Tied to source and month, not role bucket; no description rises to 28–30% from June. Nothing is imputed here (3.9) |
| How much is left once blockers are hidden? | 85% of open listings stay visible to a career changer without management experience; the "5+ years" rule would hide another 30%, so it waits for EDA_SDA_3 (3.11) |
| Which role buckets are deep enough? | All 11 on the listings side; eight also have enough career changers for their own compass paths. Technical sales, business and data analysis and other computer roles are analyzed combined (3.6) |

### For the career changer
- **IT support is the clearest door** (44% open, 23% entry level) at the lowest pay ($31 an hour); **Data Analytics roles** are the most open TripleTen program (30%) at $47.60.
- **Reachable listings pay $9–35 an hour less in every bucket;** the app shows the trade-off, and the compass shows where people moved next (IT support → networks and systems is the most common move, 279 people).
- **Most paths are thin,** so the compass always shows how many people a path rests on.

### Next steps
- **EDA_SDA_2, phase B:** the feature screening (every candidate feature through the same checks, one keep/drop table).
- **EDA_SDA_3 (text):** clean descriptions from `scope_descriptions.parquet`; validate the "5+ years" and degree rules before any text blocker hides listings; urgency read in context; map skills to O*NET; add evergreen wording to the ghost check.
- **EDA_SDA_4 (statistics):** volume × company type, pay disclosure within state and source, recruiters, and whether reachable listings close faster, in the adjusted Cox model.
- **Ask PredictLeads:** why job-board listings stop after May, and why descriptions go missing more often from June.
