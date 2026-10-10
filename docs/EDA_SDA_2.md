<!-- Generated from notebooks/EDA_SDA_2.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 2 (Non-Text Analysis)

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

`EDA_SDA_1.ipynb` explored the data, defended the scope decisions, and built its final table (sections 10 and 11). This notebook does the **non-text** in-depth analysis: it checks the factors kept for now, separates ghost listings from desperate employers, analyzes each time scale, narrows the target, **computer and mathematical**, to the listings a career changer can get, splits it into **role buckets** aligned with the career-switch profiles (our scope), pairs it with the team's backgrounds, and builds **its own final table**. The NLP work comes next, in `EDA_SDA_3.ipynb`; statistics in `EDA_SDA_4.ipynb`; then `TimeSeries`, `ML_Fit` and `ML_Rank`.

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
| EDA_SDA_2 | Which listings do we study? The scope cut, additional cleaning, non-text checks, ghost listings | Scope table, non-text features, occupation summary |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Scope table + text features |
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
- **The scope: 14,019 computer and mathematical listings** from 6,685 companies (7.6% of 184,889), 3,458 still open. None needed the O*NET-code fallback. Credentialed math (120 listings: actuaries, mathematicians) is out of scope, and the other 170,750 listings are cut; both are kept only in the summary of 1,008 occupations in 23 groups (1c).

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
| **Open to career changers** | `career_open`, or the stricter rule chosen in 3.11 | Accessibility comparisons within the scope; against other groups only through the 1c summary | Shows whether reachable listings behave differently |
| **The scope** | The whole table after the cut in 1d: computer and mathematical listings, by role bucket (3.6) | Everything in this notebook | Focus |
| **Other groups** | The 1c occupation summary only (aggregates; no listings) | Comparing the scope with other fields; pay in a user's current field for the compass | The listings themselves are cut |

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
| 3 | **Internships and temporary roles** | `contract_types` plus title words (intern, internship, temporary, temp, seasonal, contract-to-hire) | `job_type` (internship, temporary, contract, regular), kept: they can be entry points for career changers. Report their share in the scope and their days open; the app can filter them |
| 4 | **Title normalization** | Lowercase; strip bracketed text, locations after " - " or " \| ", requisition IDs, pay, and "remote" or "hybrid" tags (kept as a flag) | `title_clean`; the original title stays. Recheck the repost count and the role-bucket mapping with it |
| 5 | **Multi-location listings** | `n_locations` (from EDA_SDA_1) above 1 | `multi_location` flag; one row per listing for timing. The state list for the app's location filter comes with `location_data` in 3.7 |
| 6 | **Filter fields for the app** | Role level from `seniority` (entry = junior; `mid_senior` split into mid and senior by title words, the Profiles rule; manager; leadership = director and above); work mode from contract types and title; schedule from contract types | `level`, `work_mode` (remote, hybrid, onsite, not stated), `schedule` (full time, part time, not stated); with `job_type` (2b.3) and the state list (3.7) they back every filter in the app |
| – | **Description cleanup** | **Moved to EDA_SDA_3:** this table keeps only description features (`desc_words`, `desc_hash`, `has_html`), not the text, to fit in memory; EDA_SDA_3 loads the text anyway | `desc_clean` there, with HTML and boilerplate removed |

**Leakage:** every fix uses only what the listing showed when it was posted, so none of them leaks the outcome.

**What the data shows**
- **Staffing agencies are rare in the scope:** 246 listings (1.8%) from 96 companies, and their listings close faster (median 14 vs. 22.5 days). Most flags (79 of 96 companies) come from the "about" text alone, so check the 2b.1 review cell before relying on them.
- **Role level:** entry 9.6%, mid 46.3%, senior 30.5% (split from `mid_senior` by title words), manager 10.0%, leadership 3.5%. Senior and leadership listings stay open longest (median 28 days); manager listings close fastest (17.2).
- **Work mode and schedule are missing for about half** (47.8% and 45.4%), and those listings close much faster (18 and 14 days vs. 26–31). That's most likely the source: platforms that fill in contract types may also keep listings up longer. It's a confounder to check within source type in section 3 before "not stated" is read as a signal. Where stated: remote 20.0%, hybrid 18.6%, onsite 13.7%; full time 51.2%, part time 2.5%. The description agrees: 94.5% of "remote" listings mention remote, vs. 6.4% of "not stated".
- **Job types:** 87.3% regular, 8.8% contract, 2.3% internships (325), 1.6% temporary; internships and contract roles close about as fast as regular ones (22–24 days), temporary ones faster (18).
- **Also:** 3,628 titles cleaned (11,389 distinct titles → 10,880), and matching reposts on clean titles finds the same 155; 50 postings were dated more than a year before first seen. **No listing names more than one location:** a job in several cities appears as separate listings, which the repost and cross-post split in 3.5 has to account for.

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
- **90 days is a reasonable stale line, but not a sharp break.** Among listings first seen in April–May, the share of open listings that close within the next 30 days falls steadily: 54% in the first month, 42%, 32%, then 26% from day 90 to 120. A 90-day-old listing is about half as likely to close next month as a new one, yet a quarter still close each month, so many stale listings aren't dead; the ghost flag tells them apart.
- **666 of 3,447 open listings (19%) are more than 90 days old,** from 14% (quality assurance) to 22% (software development). Every open job-board listing (47) is more than 90 days old, likely how job boards keep listings up; 3.2 looks at it.
- **One signal separates ghosts from active listings:** only 44% of stale listings come from a company that posted anything new in the last 60 days, vs. 92% of younger ones. Relisting, very old posting dates and source barely differ, so the flag rests almost entirely on company activity.
- **374 likely ghosts (56% of stale);** without them, median days open drops 1–3 days per bucket and open counts about 10% (3.12a).
- **"Possibly hard to fill" is dropped:** those 292 stale listings from still-active companies are no more open to career changers than younger listings (19.6% vs. 20.0% show any sign; 7.9% vs. 10.2% entry level), so the idea that a struggling employer welcomes newcomers doesn't hold here. Stale listings that aren't likely ghosts are just "stale". **In the app, only likely ghosts are hidden;** other stale listings stay visible, labelled "open 90+ days" and ranked lower, since a quarter of 90-day-old listings still close each month (3.1a). Likely ghosts are clearly less open (14.5%).

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
- **Sources:** 65.4% of scope listings are on a company site or other domain, 26.7% on a hiring platform and 7.8% on a job board. Workday (1,345) and LinkedIn (1,071) are the biggest named platforms.
- **Source and company volume interact:** for a company's first listing, company sites and hiring platforms close at a similar pace (48.1 and 42.1 days), but for companies with 20+ earlier listings, company-site listings close in 6.1 days vs. 18.2 on platforms. The volume effect is strongest on company sites, so the model gets both, not source alone. Without likely ghosts, the pattern is the same.
- **Platforms differ a lot:** SmartRecruiters, LinkedIn and Oracle around 14 days; Greenhouse 54, Ashby 71 and Lever 75. Some of that is likely how each platform shows closures to the crawler, so platform is a timing feature to treat with care.
- **Job-board listings stop after May:** 494 and 526 first seen in April and May, then 4, 2, 2 and 0 from June to September, while the other sources keep growing. That's a change in collection, not in employers, and it's why every open job-board listing is over 90 days old (3.1b). Job-board timing results cover April–May only; asked of PredictLeads.
- **Employer's own site:** 58.5% of "company site / other" listings are on the employer's own domain, which is **38% of all scope listings** (a lower bound: careers sites on a different domain don't match). The app's "apply on the employer's site" covers at least 38%, not the 60.7% upper bound from EDA_SDA_1.

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
- **The crawler is quick:** a median 1.5 days from the employer's posting date to first seen; 85% of listings are found within a week, 3% more than 30 days late. Hiring platforms are found fastest (1.0 days), job boards slowest (3.1).
- **Busy companies are found faster:** on company sites the median lag is 4.4 days for a company's first listing vs. 1.3 days for companies with 20+ earlier listings, so the crawler likely visits busy companies more often.
- **The company-volume effect is real but much smaller like-for-like.** Among listings with a posting date, a company's first listing stays open 26.0 days from first seen (31.5 from the posting date) vs. 17.3 (21.8) for companies with 20+: about **1.4–1.5×**, not the 5.5× in 3.5a. The large gap comes mostly from listings without a posting date (largely company career sites), where crawl timing likely plays a bigger part. **Busy employers do fill faster, by roughly a third, not five times.**
- **Late-found listings had been up a while:** listings found 7+ days after posting stay open 46 days counted from the posting date (26 from first seen), vs. 22 for listings found within a day.
- The chart (3.3e) shows the curves for both measurements; each median is where a curve crosses the dashed 50% line.

### 3.4 Pay disclosed

| Check | If the result is… | Then… |
|:--|:--|:--|
| Disclosure rate by US state (needs the state from `location_data`) | Much higher in pay-transparency states (CA, CO, NY, WA) | Add state, or a "pay-law state" flag |
| Disclosure by hourly vs. salaried, and by seniority | Strongly tied to hourly | Its effect may really be job type |
| Days open by disclosure, within company-volume bands | The gap holds | Keep as a factor |
| | The gap disappears | Keep only as the companion flag for pay level |

**As built (2026-10-10):** state comes later (3.7, from `location_data`), so 3.4 uses role level, source, company volume and role bucket. Disclosure is compared within company-volume bands and without likely ghosts (the confounders rule).

**What the data shows**
- **About 41% of entry, mid and manager listings disclose pay,** rising to 48% for senior and 51% for leadership. Hiring platforms disclose most (55–67%, rising with company volume), company sites 34–46%, and job boards almost never (under 1%: pay doesn't seem to carry over from job boards).
- **Listings that disclose pay stay open much longer, in every volume band:** 20.1 vs. 2.5 days for companies with 20+ listings, 43.8 vs. 35.0 for a company's first listing, and the same without likely ghosts. The 2.5 days for undisclosed listings at busy companies is suspiciously short and looks like the crawl-timing effect from 3.3, so disclosure partly marks source and visibility. EDA_SDA_4 tests it within source before it counts as a factor.
- **Pay is mostly posted yearly** (85%; 14.5% hourly), so most hourly figures are full-time equivalents.
- **Hourly pay by bucket (medians):** software development and IT project management $72 (middle half $54–92), database $68, security $66, data science $64, quality assurance $62, networks and systems $61, and IT support $32 (middle half $24.50–42), the clearest pay step for a career changer moving on from an entry door.

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
- **Company hiring volume is the strongest signal in the scope:** listings from companies with 20+ earlier listings (any field) close in a median **7.1 days** vs. **38.9** for a company's first listing, about 5.5× (EDA_SDA_1 found 2.3× across all fields). It holds without staffing agencies (7.1 vs. 39.0) and without likely ghosts (7.1 vs. 33.8), and within every source type: company sites 6.1 vs. 48.1 days, hiring platforms 18.2 vs. 42.1; job boards show a weaker gap (12.0 vs. 16.5).
- **Big employers aren't more open to newcomers:** 8–11% of listings are entry level in every volume band, so volume is about speed, not access.
- **Checked in 3.3:** busy companies are crawled more often, and like-for-like (listings with a posting date) the gap shrinks to about 1.4–1.5×: the volume effect is real, but much of the 5.5× is crawl timing.
- **Copies of the same job are rare, and cross-posts almost absent:** of 14,019 listings, 13 are cross-posted (another source had the job live), 407 are same-source copies (most likely the same job in another city) and 345 are relisted; the rest are single. PredictLeads probably records one source per job, so **the multi-board urgency idea can't be tested with this data**: 13 listings are too few, even though they close faster (14 vs. 24 days).
- **"Relisted" copies close in a median 3.5 days,** far faster than anything else. That looks like the crawler seeing the same posting refreshed (a new id), not a new hiring round; repost features (`times_listed` and the rest) should be read with that in mind, and EDA_SDA_4 tests whether they carry real signal.

### 3.6 The scope: computer and mathematical, by role bucket

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

**Related groups:** the team's source-background groups are cut with the rest (1d); they live on as rows of the 1c occupation summary (pay, timing, openness) and through O*NET skills and the Career Changers Matrix for role discovery.

**What the data shows**
- **Eight buckets get their own analysis and compass paths:** software development (4,572 listings, 475 career changers), networks and systems (2,724; 670), IT project management (1,832; 264), data science (1,739; 237), security (744; 350), IT support (711; 383), database (698; 395) and quality assurance (546; 39, just over the 30 needed).
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

**As built (2026-10-10):** 3.7a–3.7e cover the live tolerance, salaried part-time pay (`pay_part_time_salaried`), the seniority labels and state (`state`, from the location records), with state-level pay. The reposted-closing question is answered by 3.5 (relisted copies close in 3.5 days, read as crawler refreshes); the companies table's extra fields and the no-country check move to 3.8.

**What the data shows**
- **The 2-day "still open" rule holds:** 2,583 listings were last seen within a day of the snapshot and 747 within two, then the count drops to 128 at 2–3 days. There's no bump near 7 days, so the crawler isn't weekly, and the rule doesn't count many open listings as closed.
- **Salaried part-time pay is negligible:** 9 of the 6,030 listings with pay (flagged `pay_part_time_salaried`, kept).
- **The seniority labels agree with titles:** 76% of `manager` and 93% of `director` titles say so; `junior` titles say junior, entry or associate (35%) or intern (23%); and **36% of `mid_senior` titles say senior, lead, principal or staff**, which confirms splitting it into mid and senior (2b.6). Internships are entry level (315 of 325).
- **State is known for 80% of the scope** (location records also hold city, zip code and region). California (1,598), Texas (1,060), New York (910), Virginia (700) and Washington (525) lead.
- **Pay transparency laws show in the data:** California, Washington, Massachusetts and Colorado disclose pay for 61–70% of listings, Texas, Florida and Georgia for 26–30%. Pay is highest in California ($82.74 an hour) and Washington ($81.27). The disclosing states also have longer windows (California 31 days vs. Texas 17), so **state is a confounder for 3.4's "disclosed listings stay open longer"**; EDA_SDA_4 tests disclosure within state.

### 3.8 Joining the companies table

Merge the companies table into the listings (a **left join** on `company_id` = company `id`, so every listing stays). Course standard: one master table from all sources, with keys checked first. Join checks: company ids are unique in the companies table (no row multiplication), and match rate (100% in EDA_SDA_1). Columns to bring in: `ticker` (publicly traded), parent company, subsidiary and similar companies, location, and description (its text goes to EDA_SDA_3). The table has **no company size or employee count**.

**Then resolve listings with no country** (`country_missing`, kept by EDA_SDA_1): take the country from the company's `location_data`. Keep the listing if the company is US-based **or** the listing mentions remote work or uses the "remote" contract type, and flag it `country_inferred`; drop the rest. Report how many are kept by each route and how many are dropped, and check whether `country_inferred` listings behave differently (days open, source, words per description).

**What the data shows**
- **The join is clean:** 14,019 listings before and after, 6,684 of the scope's 6,685 companies matched, and no duplicate company ids. 14.7% of companies are publicly traded, 17.7% have a parent company and 81.7% are US-based.
- **Company type changes the volume effect:** for a company's first few listings, public and other companies close at a similar pace (1 posting: 33.5 vs. 39.3 days; 2–4: 30.5 vs. 27.5). At high volume they split: at 20+ listings, public companies' listings stay open a median of 20.3 days and others' only 6.0 (with a parent 20.1, without 6.1). Most of the "high-volume companies close fast" effect (3.5) comes from private, independent high-volume posters. **Staffing agencies don't explain it:** without them the 20+ median is still 6.0 days (6.1 without a parent). **Missing descriptions explain part of it (3.9d):** 41% of these companies' listings have no description, and those close in about 3 days. With a description only, the 20+ medians are 26.2 days (public) vs. 14.7 (others): the gap shrinks from 14 days to 11 but stays. Relisted copies (3.5) are the next suspect. Public companies are also a bigger share at high volume (9% of first listings, 27% at 20+). `is_public` and `has_parent` are kept as features, and EDA_SDA_4 tests volume × company type, with and without relisted copies.
- **Most no-country listings resolve to the US:** of 1,549, 1,337 belong to US-based companies and 80 more are remote; 132 (0.9% of the scope) are out of scope and are dropped in section 4.
- **Inferred-country listings close at the same pace** (22.1 vs. 22.6 days) but **61% have no description** (vs. 19% of other listings), 42% are remote (vs. 18%) and 12% are on hiring platforms (vs. 29%). Their text features in EDA_SDA_3 will be thin, so fit falls back on the title for them.
- **The out-of-scope listings look like crawl stubs:** median 0 days open, 77% with no description, none remote. Dropping them costs little.

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
- **25 of 86 columns have blanks, and few are truly missing.** Most are structural (platform: company sites have none), censored, or "not stated" (years, posting date). Each is labelled in 3.9a with its handling, and nothing is imputed here. Fields built from the same source go missing together by design: 10 text fields go blank exactly when the description does, and the pay columns go blank with pay.
- **23% of the scope has no description, and it's a collection change:** 2–10% of listings first seen in March–April lack one, about 30% from June on. It's mostly company-site listings (34%, vs. 6% on hiring platforms) and private companies (21–41%, vs. 5–15% for public ones).
- **Listings with no description close in 3 days, vs. 30.** They look like the crawl stubs from 3.8: seen briefly, little content.
- **That explains the "missing pay" and "missing years" gaps.** Among listings with a description, missing pay (28 vs. 30 days) and no years stated (28 vs. 30) barely differ. Without that check, both would look like strong signals. A missing posting date goes the other way: those listings stay open longer (39 vs. 22 days).
- **Missingness doesn't depend much on role bucket** (no description 16–28%, no pay 52–59%). It does depend on source: 99% of job-board listings have no pay, but nearly all have a posting date. 81% of entry-level listings state no years.
- **Decision:** "has a description" joins the standard confounders. Timing results are shown with and without listings that have no description, and ML_Rank checks that `has_desc` is known when a listing is first seen before using it as a feature (otherwise it would leak how the listing ends).

### 3.10 Outliers

Boxplots and IQR rules for days open, words per description, hourly pay, times listed. Course standard: inspect, explain, and keep or remove with a reason. Advanced: robust (median / IQR) summaries, and log scales for the right-skewed measures.

**What the data shows**
- **Days open: a real long tail, kept.** 6.6% of closed listings are above the 1.5 × IQR fence (81 days), but none are above the log-scale fence. Listings above the fence look like the rest (old posting date 0.3% vs. 0.1%, staffing agency 1.0% vs. 1.9%, public 19% vs. 20%, has a description 74% vs. 72%). They're slow hires, not errors. Long-open listings that are still open are already handled by the 90-day line (3.1).
- **Words per description: kept.** 1.2% are above 1,680 words and 0.1% above the log-scale fence. The short end matters more: **5% of company-site descriptions are under 50 words** (vs. 0.3% on hiring platforms). These are likely cut-off pages. EDA_SDA_3 treats text flags from them as weak.
- **Hourly pay: kept, with 8 implausible values flagged.** 1.5% are above $142.67 an hour (about $297,000 a year), and 0.2% above the log-scale fence ($207). They're mostly senior roles (61% vs. 48%) with salaried pay. That's real top-end pay, not unit errors: none were fixed in EDA_SDA_1. **3 listings below the federal minimum wage ($7.25) and 5 above $250 an hour** are flagged `pay_implausible` in section 4. Those listings stay; their pay is left out of pay summaries and shown as "not disclosed".
- **Times listed: the IQR rule doesn't fit counts.** 98.6% of the scope are first listings, so the IQR is 0 and every repost counts as an "outlier". Reposts are kept and grouped as 1 / 2–4 / 5 or more in features (0.5% are 5 or more, none from staffing agencies). With so few reposts in the scope, repost features will carry little signal.
- **Robust summaries are used throughout:** medians and IQRs, with log scales for days open and words (skew 1.8 and 0.5).

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
- **The rules in force hide 22% of open listings from a career changer without management experience:** likely ghosts 10.9% (all 376 are open by definition), manager level or above 11.7%, document management 0.3%. Across the whole scope it's 16%.
- **The "asks 5+ years" preview would hide another big slice: 31% of open listings.** That's too much for an unvalidated rule. EDA_SDA_3 checks it before it's applied: required vs. preferred, and the year counts the regex picks up.
- **The degree flag looks too narrow to trust:** only 3–6% of listings per bucket are flagged "degree required", fewer than say "or equivalent experience" (10–18%). The degree pattern probably misses common wording ("Bachelor's in computer science"). EDA_SDA_3 validates it before any degree rule is used. On today's flag, no bucket is a poor target for people without a degree (degree with no equivalent: 2–5%).
- **Openness isn't company size or source in disguise:** 16–21% in almost every volume × source cell. The one exception is hiring-platform listings from companies with 20+ listings (36%), likely high-volume entry-level hiring programs. Scope-wide it's 19.6%, or 21.4% without IT project management.
- **Each part of `career_open`:** entry-level seniority (10.3% of listings) and "2 years or fewer" (9.6%) do most of the work, and are usually the only sign (60% and 77%). Entry-level or training wording (3.9%) is the only sign just 32% of the time, so it mostly confirms the others. Career-changer wording is rare (0.6%). A stricter "open to career changers" view can use seniority and years alone.
- **Reachable listings pay much less, in every bucket:** $11–37 an hour less (software development $43 vs. $78, IT support $26 vs. $37). That's the trade-off the app shows: the pay weight trades off against accessibility, and the compass shows both the entry pay and the pay later on.
- **Reachable listings close a bit faster at small companies:** 28 vs. 35 days for a company's first listing, 24 vs. 33 at 2–4. There's no difference at 20+. It's a modest urgency signal; EDA_SDA_4 tests it before the ranking uses it.

### 3.12 The retraining compass: the 2026 side

For each kept role bucket, after blockers are hidden: **open listings at the snapshot, new listings per week, median hourly pay (where disclosed), median days open (Kaplan–Meier), entry-level share, openness to career changers, remote share, and expected job growth** (a placeholder column until the projections are downloaded) (official projections by occupation code; time frame to be decided). Rows in the same role order as Profiles (`ROLE_ORDER` in `src/career_switch.py`: infrastructure → code → data).

Join it by role bucket to Profiles' three path tables, so every path shows its 2026 openings, pay, closing speed and growth:
- **Prior career → first tech job** and **field trained in → first tech job** (4a.2b): the compass for changing careers.
- **Tech job before → after the move** (4a.3a), with the sideways-or-up split: the compass for moving across or up within tech.

Every path carries its people count, and paths under about 10 people are marked thin. Growth sits next to the entry-level share, since projections cover the whole occupation. Its source and method are in `docs/job_growth.md` (BLS 2025–2035, employment-weighted by bucket, at most 5 years). Aggregates only; this table feeds the app's Retraining compass and the deck (`DIFFERENTIATORS.md`).

**What the data shows**
- **The 2026 market by bucket:** software development is the largest (1,175 open at the snapshot, about 174 new a week), then networks and systems (676; 103), IT project management (432; 70) and data science (431; 66); quality assurance (148) and IT support (129) are the smallest. Pay is highest in software and IT project management ($72 an hour) and lowest in IT support ($32).
- **Openness varies most by bucket:** 44% of IT support listings show a sign of openness to career changers vs. 9% of IT project management; remote work is similar everywhere (17–23%, data science highest).
- **The strongest paths are within tech:** IT support → networks and systems (279 people), networks and systems → IT support (199), IT support → security (163), developers → databases (142) and → data science (137). From outside tech, the largest are business → networks and systems (31), arts and media → software (30), and computer science → software (75) by training.
- **Most paths are thin:** 78% of prior-career paths, 67% of training paths and 43% of within-tech paths rest on fewer than 10 people. The compass shows every count and marks thin paths, and `ML_Fit` should fall back to the bucket-level pattern when a user's exact path is thin.
- **Without likely ghosts** (3.1d), each bucket's median days open drops 1–3 days (security 26.8 → 24.2, software 24.1 → 22.2) and its open count about 10%: ghosts stretch timing a little, so both versions are kept.
- **Expected growth is still empty:** it fills from `role_growth.pkl` (BLS 2025–2035, rolled up to buckets by 2025 employment; plan in `docs/job_growth.md`) once PredictLeads consents in writing to combining their data with BLS data. Saved for `ML_Fit`: `role_market.pkl` (8 buckets) and `compass_paths_2026.pkl` (223 paths).

## 4. EDA_SDA_2 final table

Every decision in this notebook, defended with its result from sections 2b and 3, then applied all at once in 4b. Nothing is removed earlier except the scope cut (1d). Same format as EDA_SDA_1 sections 10 and 11. The table is saved as its own file (`scope_table.pkl`), so EDA_SDA_1's table stays unchanged. **Hide, never remove:** the only rows dropped are listings outside the project's scope; everything else is a flag the app hides or labels by.

### 4.0 Decisions

| Decision | Evidence | Choice | Status |
|:--|:--|:--|:--|
| **The scope** | Computer and mathematical is the main group (EDA_SDA_1 Q3); credentialed math (actuaries, statisticians) needs a license or degree a career changer doesn't have | Computer and mathematical, without credentialed math: 14,019 listings | Applied at the start (1d); a per-occupation summary of the full table is kept (1c) |
| **Role buckets** | Eight buckets with enough listings and career changers on both sides (3.6); document management specialists are records work, not a tech path | Eight buckets plus "other computer roles"; document management flagged `hidden_occupation` | Decided; hidden in the app |
| **IT project management** | Lowest openness (9%), 2% entry level, 55% manager level or above; 190 of its 264 career changers came from another tech role (3.6, 3.11) | Kept, as a later move: not suggested as a first tech job unless the prior career has a real path into it | Decided (app and ML_Fit) |
| **Listings with no country** | 1,337 belong to US-based companies and 80 are remote; the other 132 close on first sight, 77% with no description (3.8) | Keep the first two groups, flagged `country_inferred`; **drop the 132** | **Applied in 4b** |
| **Likely ghosts** | Closing slows sharply after 90 days; a listing over 90 days old from a company gone quiet, or with a posting date over a year old, is likely forgotten (3.1) | Flag `likely_ghost`, hidden in the app; other listings over 90 days old stay visible, labelled and ranked lower (`stale`) | Decided; timing results also shown without them |
| **Manager level or above** | 11.7% of open listings; prior experience can qualify some career changers (3.11) | Label, hidden per applicant | Decided (app) |
| **Text blockers** | "Asks 5+ years" would hide 31% of open listings; the degree flag fires on only 3–6% (3.11) | Previews only | Open: EDA_SDA_3 validates them first |
| **Staffing agencies** | They post at volume and repost often (2b.1); they don't explain the fast closings at high-volume private companies (3.8) | Flag `is_staffing`; the app labels them "posted by a recruiter" | Decided |
| **Pay** | Converted to hourly (EDA_SDA_1 6.5). 3 listings under the federal minimum wage and 5 over $250 an hour (3.10); 9 part-time listings with salaried pay (3.7) | Flag `pay_implausible` and blank its hourly pay (the posted range stays); flag `pay_part_time_salaried` | **Applied in 4b** (`pay_implausible`) |
| **Missing values** | Most blanks are structural, censored or "not stated" (3.9) | Nothing imputed here; imputation happens inside the model `Pipeline`, fit on training months, with a missing flag. `has_posted` added | Decided |
| **No description** | 23% of the scope (about 30% from June: a collection change); these listings close in 3 days vs. 30 (3.9) | Kept, `has_desc` is a standard confounder; timing results shown with and without them | Decided. ML_Rank checks `has_desc` is known at first sighting |
| **Outliers** | Long days open, long descriptions and high pay are real tails (none above the log-scale fence for days open) (3.10) | Keep; group times listed as 1 / 2–4 / 5 or more (`times_listed_band`) | **Applied in 4b** (`times_listed_band`) |
| **"Possibly hard to fill"** | Only a small share of stale listings; can't be told apart from a forgotten listing (3.1) | Dropped as a separate label | **Applied in 4b** (column removed) |

### 4.0 Prioritization factors, after this notebook

| Factor | Result | Choice | Status |
|:--|:--|:--|:--|
| Company hiring volume | About 1.4–1.5× faster closing like-for-like, not 5.5× (crawl timing explains part, 3.3, 3.5); much larger at private, independent companies (3.8) | Keep, counted from earlier listings only (`company_earlier`, `volume_band`) | Kept; EDA_SDA_4 tests volume × company type |
| Source and platform | Applying on the employer's site is possible for at least 38% of listings (3.2); job-board listings stop after May (a collection change) | Keep `source_type`, `platform`, `on_employer_domain` | Kept; a standard confounder |
| Freshness | Crawl timing shapes `first_seen`; the employer's posting date gives a second view where present (3.3) | Keep `first_seen`, with `posted` and `has_posted` alongside | Kept |
| Pay disclosed | Disclosed listings stay open longer, partly crawl timing; pay-transparency states disclose far more often (3.4, 3.7) | Keep, compared within state | Kept; EDA_SDA_4 tests it within state |
| Pay level | Reachable listings pay $11–37 an hour less in every bucket (3.11) | Keep, hourly | Decided; the app's pay weight trades off against accessibility |
| Copies and reposts | Only 13 cross-posts, so the urgency idea can't be tested; relisted copies close in 3.5 days, read as crawler refreshes; 98.6% of the scope are first listings (3.5, 3.10) | Keep `copy_type` and the repost features; expect little signal | Kept |
| Company type | At 20+ listings, public companies' listings stay open 26 days vs. 15 for others, among listings with a description (3.8, 3.9) | Keep `is_public`, `has_parent` | Kept; a standard confounder |
| State | Known for 80% of listings; pay transparency laws show (3.7) | Keep `state` | Kept; the app's location filter and a confounder |
| Career-change accessibility | 19.6% of listings show any sign; seniority and years asked do most of the work; reachable listings close a bit faster at small companies (3.11) | Keep `career_open` and its parts | Kept; EDA_SDA_3 validates the text parts, EDA_SDA_4 tests the timing gap |

**What the final table shows**
- **13,887 listings from 6,641 companies** (13,732 originals), 3,436 still open at the snapshot. The 132 out-of-scope no-country listings are the only rows removed. Five of them were likely ghosts, which is why 371 remain (376 in 3.11).
- **3,055 open listings are visible to everyone** once likely ghosts (371) and document management (70) are hidden. For a career changer without management experience, manager-level listings are also hidden, leaving about 78% of open listings (3.11).
- **Everything else is kept and flagged:** 661 open 90+ days (labelled), 245 from staffing agencies (labelled), 1,417 with an inferred country, 8 with implausible pay (blanked).
- **86 columns, every one described** in 4e (38 added in this notebook). `times_listed_band` is filled for 77%: the rest have no description to match copies on.
- **Saved as `scope_table.pkl`** for EDA_SDA_3 (`PART2_TABLE` in `src/eda_helpers.py`). EDA_SDA_1's table is unchanged.

## 5. Conclusion and next steps

**The scope:** 13,887 computer and mathematical listings in eight role buckets, cleaned, flagged and saved for EDA_SDA_3. The main lesson: **how the data was collected explains many of the biggest gaps.** Crawl timing, job boards stopping after May, and listings with no description from June on (which close in 3 days) all look like signals until they're checked. So "has a description", company type and state join the standard confounders, and timing results are shown with and without likely ghosts and no-description listings.

### Answers to the clarifying questions

| Question | Answer |
|:--|:--|
| Which long-open listings are ghosts, and which are still hiring? | Company activity separates them: 44% of stale listings come from a company that posted in the last 60 days, vs. 92% of younger ones. **371 likely ghosts are hidden;** other listings open 90+ days stay visible, labelled and ranked lower, since a quarter of 90-day-old listings still close each month. "Possibly hard to fill" was dropped: those listings are no more open to newcomers (3.1) |
| Does source matter beyond company volume? | Yes, and the two interact: the volume effect is strongest on company sites (6 vs. 48 days) and weaker on hiring platforms (18 vs. 42). Platforms differ widely (14 to 75 days), partly in how they show closures. Both stay in the model; platform with care (3.2) |
| How precise is freshness? | Precise enough: a median 1.5 days late, 85% found within a week. Busy companies are crawled more often, which inflates their speed: like-for-like, the volume effect is about 1.4–1.5×, not 5.5× (3.3) |
| Is pay disclosure about state laws or source rather than intent? | Largely: pay-transparency states disclose 61–70% vs. 26–30% elsewhere, and hiring platforms 55–67% vs. job boards under 1%. Disclosed listings stay open longer, partly crawl timing. EDA_SDA_4 tests disclosure within state and source (3.4, 3.7) |
| Does high company volume mean competition or more openings? | Busy employers fill faster, by about a third like-for-like, mostly private, independent ones (public companies' listings stay open 26 vs. 15 days at 20+). The data can't show competition (no applicant counts), so volume is read as "apply sooner" (3.3, 3.5, 3.8) |
| How are values missing? | Tied to source and time, not role bucket. 23% of listings have no description (about 30% from June, mostly company sites), and they close in 3 days, which made missing pay and missing years look like signals until checked. Nothing is imputed here (3.9) |
| How much is left once blockers are hidden? | About 78% of open listings stay visible to a career changer without management experience (22% hidden: likely ghosts, manager level, document management). The "5+ years" rule would hide another 31%, so it waits for EDA_SDA_3's validation (3.11) |
| Do high-volume employers post more entry-level listings? | No: 8–11% entry level in every volume band, and openness is 16–21% almost everywhere. Volume is about speed, not access (3.5, 3.11) |
| Which role buckets are deep enough? | Eight: software development, networks and systems, IT project management, data science, security, IT support, database and quality assurance (the thinnest, 39 career changers). IT project management is kept as a later move, not a first job. Other computer roles stay in the table without their own analysis (3.6) |

### For the career changer
- **IT support is the clearest entry door** (22% entry level, 44% open to career changers) but pays least ($32 an hour). The well-paid buckets ($61–72) are much less open.
- **Reachable listings pay $11–37 an hour less in every bucket.** The app shows this trade-off instead of hiding it, and the compass shows where people moved next.
- **Most paths are thin** (under 10 people), so the compass always shows how many people a path rests on.

### Next steps
- **EDA_SDA_3 (text):** clean descriptions; validate the "5+ years" and degree rules (the degree pattern fires on only 3–6% of listings) before any text blocker hides listings; treat text flags from very short or missing descriptions as weak; map skills to O*NET; add evergreen wording to the ghost check.
- **EDA_SDA_4 (statistics):** test volume × company type (with and without relisted copies), pay disclosure within state and source, and whether reachable listings close faster, all in the adjusted Cox model with the full confounder list.
- **ML_Rank:** check that `has_desc` and the text flags are known when a listing is first seen before using them as features.
- **Ask PredictLeads:** why job-board listings stop after May, and why descriptions go missing more often from June on.
