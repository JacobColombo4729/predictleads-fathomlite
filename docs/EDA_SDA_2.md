<!-- Generated from notebooks/EDA_SDA_2.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 2 (Non-Text Analysis)

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

`EDA_SDA_1.ipynb` explored the data, defended the scope decisions, and built its final table (sections 10 and 11). This notebook does the **non-text** in-depth analysis: it checks the factors kept for now, separates ghost listings from desperate employers, analyzes each time scale, narrows the target, **computer and mathematical**, to the listings a career changer can get, splits it into **role buckets** aligned with the career-switch profiles (our niche), pairs it with the team's backgrounds, and builds **its own final table**. The NLP work comes next, in `EDA_SDA_3.ipynb`; statistics in `EDA_SDA_4.ipynb`; then `TimeSeries`, `ML_Fit` and `ML_Rank`.

**Inputs from EDA_SDA_1** (decision tables in its section 10):
- English-language listings in the United States (no language label: dropped), plus listings with **no country**, kept and flagged `country_missing` for 3.8 to resolve
- Monthly horizon, April–September 2026
- Every listing's full follow-up: long-open listings kept, nothing capped
- All seniority levels and candidate factor columns, including reposts as a feature. **Only computer and mathematical listings are kept (1d, decided 2026-10-10);** every other group is cut, leaving just a per-occupation summary of pay and timing (1c)
- Career-change accessibility flags: `entry_seniority`, `years_required`, `low_experience_ask`, `entry_wording`, `changer_wording`, `degree_required`, `equivalent_experience`, and the first-pass `career_open` (EDA_SDA_1 Q1g)

**Conventions carried over:**
- Every decision is recorded with its reasoning.
- Data-quality fixes happen in cleaning; scope decisions are defended first, then applied at the end (section 4).
- Aggregates only (counts, shares, quantiles, model estimates); never `df.head()` or sample rows. Clear outputs before committing.


**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, and what's the scope? | Final table 1 |
| EDA_SDA_2 | Which listings do we study? The niche cut, additional cleaning, non-text checks, ghost listings | Niche table, non-text features, occupation summary |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Niche table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

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

**What the data shows**
- **The niche: 14,019 computer and mathematical listings** from 6,685 companies (7.6% of 184,889), 3,458 still open. None needed the O*NET-code fallback. Credentialed math (120 listings: actuaries, mathematicians) is out of scope, and the other 170,750 listings are cut; both are kept only in the summary of 1,008 occupations in 24 groups (1c).
- **In that summary, computer and mathematical pays the most** (median $66.63 an hour, original listings) but has among the fewest entry-level listings (9%) and signs of openness to career changers (21%). Its median window, 22.5 days, is close to most groups'.

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
| **Without likely ghosts** | Every listing except `likely_ghost` (3.1d) | Every timing result (days open, share still open, open at the snapshot) is shown both with and without them, from 3.1 on | A forgotten listing's "days open" isn't a real hiring window, so ghosts can stretch timing results; they stay in the data either way |
| **Open to career changers** | `career_open`, or the stricter rule chosen in 3.11 | Accessibility comparisons within the niche; against other groups only through the 1c summary | Shows whether reachable listings behave differently |
| **The niche** | The whole table after the cut in 1d: computer and mathematical listings, by role bucket (3.6) | Everything in this notebook | Focus |
| **Other groups** | The 1c occupation summary only (aggregates; no listings) | Comparing the niche with other fields; pay in a user's current field for the compass | The listings themselves are cut |

### Confounders: compare within groups, don't filter

Some factors travel together, so a raw gap can belong to something else. **Every factor test in section 3 is checked against these,** by comparing within groups (stratifying), not by filtering them out: filtering throws data away and only answers the question for what's left. Filtering is used only when a slice can't answer the question at all (for example day of week without `posted`).

| Confounder | Why it matters here | How each check handles it |
|:--|:--|:--|
| **Company hiring volume** (earlier listings only, bands 1 / 2–4 / 5–19 / 20+) | The strongest signal; it overlaps with source, pay disclosure and staffing agencies | Every factor is compared within volume bands |
| **Source and platform** | Platforms show closures to the crawler differently | Within source type, and within platform where there are enough listings |
| **Staffing agencies** (`is_staffing`, 2b.1) | They post at volume and repost often | Gaps shown with and without them |
| **Role bucket and seniority** | A senior data scientist role and an IT support role close at different speeds for unrelated reasons | Within role bucket (3.6) where buckets are big enough; seniority as a split |
| **Month first seen** | Later months have had less time to close | Survival methods (still-open listings censored); month as a split for timing gaps |
| **Ghost listings** (`likely_ghost`, 3.1d) | Forgotten listings inflate days open and the count still open | Timing results also shown without likely ghosts |

**The rule:** each factor shows the **raw gap**, then the **gap within volume band × source type**, and within role bucket where there are enough listings. A factor counts as real only if the gap holds within the groups. EDA_SDA_4 then confirms the survivors with one model that adjusts for all of these at once.

## 2b. Additional cleaning

**Decided (2026-10-09):** five fixes and the app's filter fields here (salary sanity moved into EDA_SDA_1 6.5, description cleanup to EDA_SDA_3), run before the checks in section 3 so they use clean pay, dates and titles. Same rules as EDA_SDA_1: **fix or flag, never drop** (a flagged listing leaves only the analyses it would distort), keep the original column next to every cleaned one, and report how many listings each step changes (counts only). Spot-checks of examples are done locally by a teammate.

| # | Fix | How | Result |
|:-:|:--|:--|:--|
| 1 | **Staffing agencies** | Flag listings posted by recruiters or staffing firms: company name and domain words (staffing, recruiting, talent, well-known agency names), or an "about" text that says staffing or recruiting firm (companies table). A review cell (2b.1 review, row-level) lists the most-flagged companies so a teammate can check the keywords | `is_staffing`, kept. Report their share by field and their days open; compute company volume with and without them (3.5), since agencies post at high volume and repost often. The app labels them "posted by a recruiter" |
| – | **Salary sanity, standardized to hourly** | **Moved to EDA_SDA_1 6.5:** all pay cleaning in one place. `salary_hourly_usd` (salaried pay ÷ 2,080 hours) is the one pay column for the whole project; reversed ranges and clearly wrong units are fixed there and flagged `pay_suspect` | Arrives clean in EDA_SDA_1's table |
| 2 | **Date consistency** | EDA_SDA_1 (6.3) already removed impossible timelines and blanked `posted` dates after `first_seen`. Here: `posted` more than a year before `first_seen` | `posted_old` flag, kept; it feeds the ghost-listing check (3.1) |
| 3 | **Internships and temporary roles** | `contract_types` plus title words (intern, internship, temporary, temp, seasonal, contract-to-hire) | `job_type` (internship, temporary, contract, regular), kept: they can be entry points for career changers. Report their share in the niche and their days open; the app can filter them |
| 4 | **Title normalization** | Lowercase; strip bracketed text, locations after " - " or " \| ", requisition IDs, pay, and "remote" or "hybrid" tags (kept as a flag) | `title_clean`; the original title stays. Recheck the repost count and the role-bucket mapping with it |
| 5 | **Multi-location listings** | `n_locations` (from EDA_SDA_1) above 1 | `multi_location` flag; one row per listing for timing. The state list for the app's location filter comes with `location_data` in 3.7 |
| 6 | **Filter fields for the app** | Role level from `seniority` (entry = junior; `mid_senior` split into mid and senior by title words, the Profiles rule; manager; leadership = director and above); work mode from contract types and title; schedule from contract types | `level`, `work_mode` (remote, hybrid, onsite, not stated), `schedule` (full time, part time, not stated); with `job_type` (2b.3) and the state list (3.7) they back every filter in the app |
| – | **Description cleanup** | **Moved to EDA_SDA_3:** this table keeps only description features (`desc_words`, `desc_hash`, `has_html`), not the text, to fit in memory; EDA_SDA_3 loads the text anyway | `desc_clean` there, with HTML and boilerplate removed |

**Leakage:** every fix uses only what the listing showed when it was posted, so none of them leaks the outcome.

**What the data shows**
- **Staffing agencies are rare in the niche:** 248 listings (1.8%) from 96 companies, and their listings close faster (median 14 vs. 22.6 days). Most flags (79 of 96 companies) come from the "about" text alone, so check the 2b.1 review cell before relying on them.
- **Role level:** entry 9.7%, mid 46.3%, senior 30.4% (split from `mid_senior` by title words), manager 10.0%, leadership 3.6%. Senior and leadership listings stay open longest (median 28 days); manager listings close fastest (17.5).
- **Work mode and schedule are missing for about half** (47.7% and 45.3%), and those listings close much faster (18 and 14 days vs. 26–31). That's most likely the source: platforms that fill in contract types may also keep listings up longer. It's a confounder to check within source type in section 3 before "not stated" is read as a signal. Where stated: remote 20.0%, hybrid 18.8%, onsite 13.6%; full time 51.3%, part time 2.5%. The description agrees: 94.6% of "remote" listings mention remote, vs. 6.4% of "not stated".
- **Job types:** 87.3% regular, 8.7% contract, 2.4% internships (337), 1.6% temporary; internships and contract roles close about as fast as regular ones (22–24 days), temporary ones faster (18).
- **Also:** 3,665 titles cleaned (11,489 distinct titles → 10,978), and matching reposts on clean titles finds the same 156; 50 postings were dated more than a year before first seen. **No listing names more than one location:** a job in several cities appears as separate listings, which the repost and cross-post split in 3.5 has to account for.

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

**Decided (2026-10-10): hide, never remove.** Listings older than **90 days** on the date being viewed are hidden in the app (a note says how many, and why); every listing stays in the analysis and the model, with still-open listings censored as usual. Age on a date is known at that moment, so hiding by it uses no future information. Removing listings by how long they *ended up* open would bias the timing model (it would never see a long window) and throw away struggling employers, who may be the most open to a career changer.

Steps: describe the long-open listings (how many, which occupation groups and sources); **check the 90-day threshold** (do closing rates level off past 90 days? are those listings relisted, or did their company stop posting?) and keep it or adjust it with that evidence; measure each signal; combine them into a **"likely ghost listing" flag**, used to rank lower or label, never to remove. Only listings first seen before July can reach 90 days in this six-month window. Evergreen wording ("always accepting applications") is added in EDA_SDA_3.

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

- **Cross-posted vs. reposted (added 2026-10-10).** EDA_SDA_1 counts every later copy of the same job (same company, title and description) as a repost, so a copy on a second board while the first is still up looks the same as a job relisted after it closed. Split them, using each listing's earlier copies only, so nothing comes from the future:
  - **`is_crosspost`:** an earlier copy is still live when this one appears, from a different source (the platform, or "company site" or "job board" when there's no platform).
  - **`is_relisted`:** earlier copies exist, but none is still live: the job came back after closing.
  - **`n_sources_so_far`:** how many different sources carry the job at the moment this listing appears (1 = only here).
  - **Hypothesis: posting on several boards signals urgency.** Test it, don't assume it: days open (Kaplan–Meier) by `n_sources_so_far` and cross-posted vs. relisted vs. single, within company-volume bands and with and without staffing agencies (2b.1). Applicant-tracking systems often send one posting to several boards automatically, so check within platform too: if the gap disappears there, it's the software, not the employer's urgency.
  - **Copies in different cities aren't cross-posts:** each listing names one location (2b.5), so a job in several cities appears as several listings. Count copies with the same location as the job on another board, and copies in other locations as a multi-city posting (its own count).
  - **Check the matching:** boards sometimes reformat descriptions, so copies may not share a description fingerprint. Compare with a looser match (same company and `title_clean`, first seen within 7 days) and report how many more cross-posts it finds.
- **Reposts** (a decided feature: how strongly should it count?): days open by `times_listed` and by relisted vs. cross-posted; repost rate in the chosen groups; repost rate within company-volume bands, to see whether it adds anything beyond company volume.
- **Company hiring volume:** does high volume mean **more competition, or more openings to win?** Its direction in the ranking depends on the answer. Count each company's listings from before each listing only, so the future isn't used. The companies table has no company size or employee count, so listing volume is our only measure of how much a company hires.
- **For career changers:** do high-volume employers post a larger share of entry-level listings or training wording? If so, volume also signals that a newcomer can get in.

### 3.6 The niche: computer and mathematical, by role bucket

**Decided (2026-10-09; applied at the start, in 1d, since 2026-10-10):** the niche is **computer and mathematical listings in the US**, with listings a career changer can't get hidden (3.11). It isn't narrowed to one role: role discovery needs several roles to choose from, and a single role would leave too few listings and too few career-switch examples. Instead, **role bucket** is a dimension inside the niche, defined the same way on both sides, so the listings (what's open) and the career changers in `Profiles.ipynb` (who got there) line up role by role:

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

The listing side is `role_bucket(onet_code)` (`ONET_DETAILED_BUCKETS` for detailed 15-1299 codes first, then `ONET_ROLE_BUCKETS`, which also covers the older 2010 codes) and the resume side is `RULES["cm_buckets"]`, both in `src/career_switch.py`.

**Steps:**
1. **Map** every computer and mathematical listing to a role bucket from `onet_code`, and report unmapped codes (aggregate counts) so the mapping can be fixed.
2. **Size each bucket** on originals only: listings, listings still open, companies and the top company's share, with and without manager-level listings (3.11; they're hidden per applicant, so the niche keeps them).
3. **Count career changers per bucket** from `Profiles.ipynb`'s modeling table (`career_changer`, by the role at the switch; unique people, after duplicate removal), split by pool: retrained after working, retrained as new grads, and changed roles within tech.
4. **Depth check per bucket:** measure the 95% confidence-interval width of the Kaplan–Meier share still open after 7 days in each sub-group. Enough depth means **±5 points or narrower** (a width of 10 or less), tight enough to show the 5–16 point gaps the EDA found, in at least 80% of sub-groups. Splits are tried from finest to coarsest, combining sub-groups that are too wide: month × source type, then source type, then month, then the whole bucket. The finest split that passes is how finely that bucket is analyzed in section 3. (About 300+ listings per sub-group are needed for ±5, so most buckets won't support all 18 month × source sub-groups.)
5. **Keep a bucket** only if it passes the depth check on the listings side **and** has enough career changers to learn from (at least 30 unique people, counting **all three pools**). The 30 was first set for retrained people only, before the within-tech pool existed; counting retrained only would drop data and analytics (22 retrained, 238 in all), which the compass needs to show is reached through other tech roles. Thin buckets are combined with a close one or analyzed only as part of the niche; their listings always stay in the table. **Credentialed math** (actuaries, mathematicians) is out of the niche from 1d: credential-gated, with no career switches into it. **IT project managers** (O*NET 15-1299.09, about 13% of the niche) have their own bucket since 2026-10-10, on both sides (a Profiles rules change). Record each keep, combine or drop with its reason.
6. **Save the niche** (kept buckets, blockers hidden) as its own table, the base for section 4 and EDA_SDA_3.

**Also described per bucket** (not used to choose buckets): variation in the factors (window, source, company volume, reposts, pay), coverage of key fields (description, salary, `posted`), words per description (EDA_SDA_1 Q3e), and openness to career changers (`entry_share`, `career_open_share`, years asked, degree required).

**Related groups:** the team's source-background groups are cut with the rest (1d); they live on as rows of the 1c occupation summary (pay, timing, openness) and through O*NET skills and the Career Changers Matrix for role discovery.

**What the data shows**
- **Eight buckets get their own analysis and compass paths:** software development (4,572 listings, 475 career changers), networks and systems (2,724; 670), IT project management (1,832; 264), data and analytics (1,739; 237), security (744; 350), IT support (711; 383), database (698; 395) and quality assurance (546; 39, just over the 30 needed).
- **No bucket supports month × source type:** software development and networks and systems can be split by month; the rest are analyzed as a whole bucket. Section 3's checks use each bucket's finest deep-enough split.
- **IT support is the clearest entry door:** the most entry-level listings (22%) and the fastest closing (median 18.3 days), but by far the lowest pay ($32.23 an hour vs. $61–72 for the others). Quality assurance is next most open (14% entry level).
- **IT project management is the hardest to enter:** 2% entry level and 55% manager or leadership listings, with pay tied for highest ($72.32 an hour). Of its 264 career changers, 190 moved in from another tech role, so it's mostly a later move, not a first step; the compass shows it that way.
- **"Other computer roles" is down to 298 listings** (GIS technologists 163, document management 70, others 65): no matching career switches, so no separate analysis; its listings stay in the table.

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
| Degree required by role bucket | Near-universal in a bucket, with little "or equivalent experience" | That bucket is a poor target |
| Each part of `career_open` separately | One part (for example, entry-level wording) adds little | Use a stricter rule for the "open to career changers" view |

### 3.12 The retraining compass: the 2026 side

For each kept role bucket, after blockers are hidden: **open listings at the snapshot, new listings per week, median hourly pay (where disclosed), median days open (Kaplan–Meier), entry-level share, openness to career changers, remote share, and expected job growth** (a placeholder column until the projections are downloaded) (official projections by occupation code; time frame to be decided). Rows in the same role order as Profiles (`ROLE_ORDER` in `src/career_switch.py`: infrastructure → code → data).

Join it by role bucket to Profiles' three path tables, so every path shows its 2026 openings, pay, closing speed and growth:
- **Prior career → first tech job** and **field trained in → first tech job** (4a.2b): the compass for changing careers.
- **Tech job before → after the move** (4a.3a), with the sideways-or-up split: the compass for moving across or up within tech.

Every path carries its people count, and paths under about 10 people are marked thin. Growth sits next to the entry-level share, since projections cover the whole occupation. Aggregates only; this table feeds the app's Retraining compass and the deck (`DIFFERENTIATORS.md`).

**What the data shows**
- **The 2026 market by bucket:** software development is the largest (1,175 open at the snapshot, about 174 new a week), then networks and systems (676; 103), IT project management (432; 70) and data and analytics (431; 66); quality assurance (148) and IT support (129) are the smallest. Pay is highest in software and IT project management ($72 an hour) and lowest in IT support ($32).
- **Openness varies most by bucket:** 44% of IT support listings show a sign of openness to career changers vs. 9% of IT project management; remote work is similar everywhere (17–23%, data and analytics highest).
- **The strongest paths are within tech:** IT support → networks and systems (279 people), networks and systems → IT support (199), IT support → security (163), developers → databases (142) and → data and analytics (137). From outside tech, the largest are business → networks and systems (31), arts and media → software (30), and computer science → software (75) by training.
- **Most paths are thin:** 78% of prior-career paths, 67% of training paths and 43% of within-tech paths rest on fewer than 10 people. The compass shows every count and marks thin paths, and `ML_Fit` should fall back to the bucket-level pattern when a user's exact path is thin.
- **Expected growth is still empty:** it fills once the official occupational projections are downloaded (queued). Saved for `ML_Fit`: `role_market.pkl` (8 buckets) and `compass_paths_2026.pkl` (223 paths).

## 4. EDA_SDA_2 final table

_To do after section 3:_ defend each decision with its result from section 3, then apply them all at once, at the end: the niche (computer and mathematical, kept role buckets), labelling manager-level listings (hidden per applicant in the app), the additional cleaning, each factor kept or dropped, and the ghost-listing decision. Record every step with its reason, in the same format as EDA_SDA_1 section 10 and 11. Save it as a separate file, so EDA_SDA_1's table stays unchanged.

## 5. Conclusion and next steps

_To do._ Next: `EDA_SDA_3.ipynb`, the text features, built on this notebook's niche table.
