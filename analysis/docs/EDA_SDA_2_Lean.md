<!-- Generated from notebooks/EDA_SDA_2_Lean.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# EDA/SDA 2 (Lean): Which Listings We Keep, Tag or Hide

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

`EDA_SDA_1_Lean` cleaned the full file, compared every field and cut the scope: **the roles TripleTen trains for, plus adjacent tech roles** (about 0.9M listings). This notebook decides, listing by listing, what the app does with each one: **keep, tag or hide.** A fourth treatment, **drop**, happens before it, in `EDA_SDA_1_Lean`'s scope cut.

- **Keep:** the listing is in the table and shown as it is.
- **Tag:** it stays visible with a tag (for example "posted by a recruiter" or "senior"), and the app can rank it lower or offer an opt-in filter. We use a tag when the rule is a keyword rule nobody has hand-checked, or when the person should decide (like a job level).
- **Hide:** the app doesn't show it by default. Only a rule we trust hides.
- **Drop:** the listing is out of scope (not a tech path a career changer can take) and is removed from the table, so it can't confound anything. Only whole occupations, identified by their O*NET code, are dropped; a keyword rule never drops.

Everything else stays in the data. The notebook also shows which role buckets are big enough for their own analysis, and the **2026 market by role bucket and TripleTen program** (the retraining compass's 2026 side). It saves its own final table for `EDA_SDA_3.ipynb` (text), then `EDA_SDA_4.ipynb` (statistics), `TimeSeries`, `ML_Fit` and `ML_Rank`.

**Input from EDA_SDA_1_Lean:** the in-scope table (`analysis_table.parquet`) and company activity in any field (`company_activity.parquet`).

**Conventions:**
- Every rule is written in plain pandas and recorded with its reasoning and the evidence for it.
- Aggregates only (shares, medians, counts); never rows.
- Each table or chart group ends with **What the data shows**: up to three bullets.
- How listings behave once they are kept (company volume, source, freshness, pay disclosure, reposts, missing values) is answered in `EDA_SDA_1_Lean` (sections 5 and 6) and tested with the confounders held fixed in `EDA_SDA_4`. It is not repeated here.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, how does tech compare to other fields, and what's in scope? | Scope table, descriptions, occupation summary, company activity |
| **EDA_SDA_2** | **Which in-scope listings do we keep, tag or hide?** | **Scope table, non-text features** |
| EDA_SDA_3 | What do the descriptions say? Skills mapped to O*NET | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| Profiles | Who really changed into or within tech or math, and how? Career-switch rules on the public résumé data (frozen) | The 2,813 career changers, and the table ML_Fit trains on |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |
| Evaluate | How well do the app's rankers do on real career moves? `rrf`, `mlrank` and the combined model | The ranker comparison |

### Clarifying questions
- Which long-open listings are ghost listings, and which are employers still trying to hire?
- Which listings do we tag as posted by a staffing agency or recruiter, and are fast-closing recruiters just staffing agencies?
- How much of the scope is left once the hiding rules are applied, and how much more would an opt-in text tag hide?
- Which role buckets have enough listings and enough career changers for their own analysis?
- What does the 2026 market look like by role bucket and TripleTen program?

### Roadmap
1. **Setup:** load the in-scope table and company activity
2. **The rules at a glance:** one table of every drop, hide, tag and keep rule
3. **Hide:** likely ghosts
4. **Tag:** manager level and above, open 90+ days, staffing agencies, senior titles, the text rules (years asked, degree)
5. **Keep:** the role buckets, and how much of the scope the rules leave
6. **The 2026 market** by role bucket and program (the compass's 2026 side)
7. **The final table**, saved for EDA_SDA_3
8. **Conclusion and next steps**

## Contents

- [Purpose](#purpose)
  - [Clarifying questions](#clarifying-questions)
  - [Roadmap](#roadmap)
  - [About this Lean draft](#about-this-lean-draft)
- [2. The rules at a glance](#2-the-rules-at-a-glance)
- [3. Hide: likely ghosts](#3-hide-likely-ghosts)
- [4. Tag](#4-tag)
  - [4.1 Manager level or above](#41-manager-level-or-above)
  - [4.2 Open 90+ days](#42-open-90-days)
  - [4.3 Staffing agency or recruiter](#43-staffing-agency-or-recruiter)
  - [4.4 Senior title](#44-senior-title)
  - [4.5 Years asked and degree required](#45-years-asked-and-degree-required)
- [5. Keep: the role buckets](#5-keep-the-role-buckets)
  - [5.1 Marketing specialists](#51-marketing-specialists)
  - [5.2 How big is each role bucket?](#52-how-big-is-each-role-bucket)
  - [5.3 Which buckets get their own analysis?](#53-which-buckets-get-their-own-analysis)
- [6. The 2026 market by role bucket and program](#6-the-2026-market-by-role-bucket-and-program)
- [7. The rules together, and the final table](#7-the-rules-together-and-the-final-table)
  - [7.1 How many listings does each rule touch?](#71-how-many-listings-does-each-rule-touch)
  - [7.2 The final table](#72-the-final-table)
- [8. Conclusion and next steps](#8-conclusion-and-next-steps)
  - [Answers to the clarifying questions](#answers-to-the-clarifying-questions)
  - [For the career changer](#for-the-career-changer)
  - [What this notebook leaves to others](#what-this-notebook-leaves-to-others)
  - [Next steps](#next-steps)

### About this Lean draft

Written from the question above, in plain pandas (`groupby`, `value_counts`, `merge`, `str.contains`, simple masks), the way `EDA_SDA_1_Lean` was. It is **not** a port of `EDA_SDA_2`: the original also studied how listings behave, built the app's filter fields and modeling features, and screened 68 candidate features. Those are left to the notebooks that need them (`docs/lean_conventions.md`, "The goal"). The faithful port that reproduced the original is commit `df93cc7`.

It reads `EDA_SDA_1_Lean`'s tables from `data/clean/lean/` and writes its own there, not beside `EDA_SDA_2`'s, until it replaces the original.

The in-scope table has about 70 columns; the rules below use these. Loading only them keeps the table small enough for an 8 GB laptop.

| Used for | Columns |
|:--|:--|
| Which listing it is | `id`, `company_id`, `title` |
| When | `first_seen`, `last_seen`, `posted`, `month`, `full_week` (first seen in a full week, not the partial first or last week), `partial_month` |
| Whether it is still open | `live` (last seen within 2 days of September 30), `days_open` |
| What it is | `onet_code`, `role_bucket`, `tripleten_program`, `seniority`, `source_type` |
| What it says | `years_required`, `degree_required`, `equivalent_experience`, `entry_wording`, `changer_wording`, `career_open`, `mentions_remote` |
| Pay | `salary_hourly_usd` |
| Whether it is a repost, and who posted it | `is_repost`, `recruiter_key` (an anonymous key per recruiter, never a name) |

## 2. The rules at a glance

Each rule below answers one question about a listing. Sections 3 to 5 define each rule in code, show what it catches, and give the evidence for the treatment. The two **drop** rules are applied upstream, in `EDA_SDA_1_Lean`'s scope cut next to credentialed math (actuaries and mathematicians), so they are listed here and not built here.

**Why four treatments.** *Drop* is for whole occupations that aren't a tech path, found by an exact O*NET code, so the rule can't misfire. *Hide* is for the one rule we trust but can't prove from a code alone (a ghost listing can't be observed; the 90-day and 60-day cutoffs are choices). *Tag* is for the keyword rules nobody hand-checked, and for anything the person should decide (their own level). A wrong tag costs a label; a wrong drop or hide costs a listing a career changer might have wanted.

## 3. Hide: likely ghosts

A long-open listing is one of two things. It may be a **ghost listing**: forgotten, or no longer watched by the employer. Or it may belong to an employer that is **still trying to hire**. Age alone can't tell them apart, so we look at what else we know about the listing.

**The rule** (decided 2026-10-10). A listing is a **likely ghost** when it is still open on September 30, was first seen more than 90 days earlier, **and** either its company has posted nothing new in 60 days or its own posting date is more than a year before it was first seen. The app hides likely ghosts and says how many it hid and why. A listing that is open 90+ days but doesn't meet the second part isn't hidden; it is tagged "open 90+ days" (section 4).

**The cutoffs are choices, not measurements.** Nobody can observe a ghost, so 90 days and 60 days can't be proven right. 3a shows how listings actually age, so you can see why 90 days is a sensible line and that the curve has no natural break to point at instead.

**What the data shows**

- **Closing slows with age but never stops.** Of listings first seen in April-May, 53% close in their first 30 days. After day 60, about a third of the listings still open close in each following 30 days (35%, 29% and 30%). There is no break in the curve to point at, so 90 days is a choice.
- **90 days is a sensible line.** Only 17% of listings are still open at day 90, so a listing open past it has outlived 83% of its peers. 19% of listings open on September 30 are older than 90 days (39,045 of 209,565).
- **Old listings are far more often from quiet companies:** 18.3% of the listings open more than 90 days come from a company with nothing new in 60 days, against 2.4% of the younger open ones. That is the signal that separates "forgotten" from "still hiring".
- **7,229 likely ghosts:** 3.4% of the listings open on September 30, 0.9% of all original listings, and 2.4% to 4.1% of the open listings in every role bucket. The app hides them and says how many.

## 4. Tag

A tag is a column the app can show on a listing, rank by or turn into an opt-in filter. None of these listings is hidden. Each tag below is defined in a few lines and comes with the evidence for it.

| Tag | Section |
|:--|:--|
| Manager level or above | 4.1 |
| Open 90+ days | 4.2 |
| Staffing agency or recruiter | 4.3 |
| Senior title | 4.4 |
| Asks 5+ years, degree required with no equivalent | 4.5 |

### 4.1 Manager level or above

PredictLeads gives every listing a `seniority` label. `level` groups the labels into four levels, and a listing is **manager level or above** when its level is manager or leadership. This is a tag and not a hide because the person decides: a career changer with management experience wants these listings, and the app's job-level filter already lets them choose. Today the app hides the tag by default for an applicant who says they have no management experience.

### 4.2 Open 90+ days

A listing that is still open, was first seen more than 90 days ago, and isn't a likely ghost. It stays visible. The tag lets the app rank it below newer listings, because 3a shows that old listings keep closing but at a slower rate.

### 4.3 Staffing agency or recruiter

A listing posted by a staffing agency or recruiting firm, found from its company: the company's name or website, or its about text, names staffing or recruiting. It's a first-pass keyword rule that nobody has hand-checked (decided 2026-10-13), so it is a tag, and the app can offer it as an opt-in filter. The original also tagged companies from agency-style recruiter titles (33 companies, 0.1% of listings); that rule is left out.

### 4.4 Senior title

The title says senior, lead, principal, staff or architect. PredictLeads' `mid_senior` label covers every experienced individual contributor, so it can't separate a senior engineer from a mid-level one; the title can. It's a keyword rule, so it is a tag: the app shows these listings ranked lower with a "senior" tag and doesn't hide them (decided 2026-10-11), because some career changers have senior experience.

### 4.5 Years asked and degree required

Two text rules from the description: **asks 5+ years** (the most years of experience the description states is 5 or more) and **degree required, no equivalent** (it says a bachelor's, master's or associate degree is required and doesn't accept equivalent experience). They are keyword rules, so they are tags the person can turn into filters ("I have no degree"), and they hide nothing. A listing with no description, or that states no years, isn't tagged: unknown is counted as not tagged, and the table shows how many are unknown.

**What the data shows**

- **Manager level and above is 13.0% of original listings and 11.2% of those still open.** The levels ask a median 1 year (entry), 5 (mid_senior), 6 (manager) and 10 (leadership), so a career changer without management experience mostly can't get these listings. It is a tag because the person decides, and the app's job-level filter already lets them.
- **Open 90+ days: 31,816 of the 209,565 listings still open (15.2%)** are tagged and ranked lower. A further 7,229 (3.4%) are likely ghosts and hidden.
- **Staffing agencies and recruiters are 1.9% of listings** (679 companies: 126 by name or website, 553 by about text only). Their listings close in a median 13.1 days against 23.6 for employers. **Recruiters aren't only staffing agencies:** among employers, recruiters with 5+ listings still close in 7.6 days, against 24.1 where no recruiter is named. The rule is a first-pass keyword rule, so it tags and never hides.
- **Senior titles are 36.4% of listings and mostly out of reach for a newcomer.** Within `mid_senior`, senior-titled listings ask a median 6 years against 4 (81% ask 5 or more, against 43%), are open to career changers 4.8% of the time against 16.5%, and pay $77.81 an hour against $56.25. They stay visible, ranked lower, with a "senior" tag.
- **Text tags:** 32.5% of the open listings ask for 5+ years (another 103,399 state no years), and only 2.6% require a degree with no equivalent.

## 5. Keep: the role buckets

Everything the rules above don't tag or hide is kept. This section looks at what is kept: the role buckets (a bucket is a group of O*NET occupations that do similar work, such as "networks and systems"), how big each one is, and which are big enough to analyze on their own. The marketing-specialist tag comes first, because it changes the buckets.

### 5.1 Marketing specialists

O*NET 13-1161 ("market research analysts and marketing specialists") mixes two kinds of work. A listing whose title names analysis or research stays in **business and data analysis**, which the Data Analytics program trains for. The rest are **marketing specialists**: marketing work, not what that program trains for. The split is a title keyword rule nobody has hand-checked (decided 2026-10-13), so marketing specialists are a tag and an opt-in filter, and the listings stay.

### 5.2 How big is each role bucket?

Original listings only (a repost would count the same job twice).

### 5.3 Which buckets get their own analysis?

A bucket gets its own analysis, and its own paths in the retraining compass, when it passes **two tests**, both fixed before looking at results:

1. **Enough listings.** We want to say something about a bucket by source and by month. The margin of error on a share is at most ±5 points when a group has about 400 listings (1.96 × √(0.25 / 400) = 0.049), so we split a bucket as finely as we can while at least 80% of its groups have 400+ listings. The splits we try, finest first: month × source type, then source type, then month, then the whole bucket.
2. **Enough career changers.** At least 30 people in the résumé data (Profiles) moved into the bucket, counting all three groups: retrained after working in another field, retrained as a new graduate, and changed roles within tech.

A bucket that passes one test but not the other is analyzed together with a close bucket. Its listings always stay in the table.

**What the data shows**

- **O*NET 13-1161 is mostly marketing:** it has 23,292 original listings, and 64% (15,013) have no analyst, research or insights word in the title. They are tagged marketing specialists (28.1% entry level, $36.06 an hour) and stay in the table; the other 36% stay in business and data analysis. The title rule has not been hand-checked, which is why it tags and doesn't drop.
- **Four buckets are large:** software development (242,105 original listings), networks and systems (159,062), IT project management (106,253) and data science (98,782). Technical sales (8,965) is the smallest named bucket, and "other computer roles" (3,494) is a catch-all.
- **Eight buckets get their own analysis and compass paths.** Each has enough listings to split by source type or finer, and 38 (quality assurance, the thinnest) to 669 career changers. Business and data analysis, marketing specialists, technical sales and other computer roles have no career changers in the frozen profiles, so they are analyzed with a close bucket; their listings stay. IT project management is large but mostly reached from another tech role (189 of its 262 career changers).

## 6. The 2026 market by role bucket and program

The retraining compass joins two things: where people moved (from the résumé data, Profiles) and what is open now (from the listings). This section builds the listings side: for each role bucket and each TripleTen program, what is open, how fast listings close, and what they pay and ask. The next cell joins it to Profiles' paths.

One column needs a definition. **Visible to a newcomer** means a listing that isn't a likely ghost and isn't manager level or above, the listings a career changer without management experience would see in the app. The compass shows pay for that view, because it is the pay a newcomer can actually reach.

**What the data shows**

- **The 2026 market by bucket:** software development (66K open, 9.2K new a week, $73.54 an hour) and networks and systems (41K open) are the largest; IT support is the door (44% open to career changers, 22% entry level, $31.10); technical sales is the slowest to close (median 38 days) and the most likely to mention remote work (41% of descriptions).
- **By TripleTen program:** AI Software Engineering (63K open) and AI & Machine Learning pay the most (about $74); **Quality Assurance and Data Analytics are the most open to career changers** (27% and 26%), Data Analytics at $51.50 an hour now that marketing specialists are out of its market; adjacent tech roles hold 86K open listings.
- **Most compass paths are thin:** 78% of prior-career paths and 67% of training paths rest on fewer than 10 people (within-tech moves: 43%), so the compass always shows how many people a path rests on. The largest within-tech move is IT support to networks and systems (279 people).

## 7. The rules together, and the final table

### 7.1 How many listings does each rule touch?

For each rule: how many listings match, and how many of the original listings still open on September 30 (the ones a person would see in the app). Then the answer to the question behind the hide and tag rules: **how much of the scope is left?**

### 7.2 The final table

One row per in-scope listing, keyed by `id`: the role bucket, the level, and one column for each rule, so any later notebook (or the app) can join it to `EDA_SDA_1_Lean`'s in-scope table by `id` without copying that table's 70 columns. Nothing is removed: a listing a rule hides or tags is still a row.

**What the data shows**

- **One hide rule, and it touches little:** likely ghosts are 0.9% of the scope and 3.4% of the original listings still open on September 30. After it, 96.6% of the open listings remain.
- **An applicant with no management experience still sees 85.7% of the open listings** (179,562 of 209,565) once manager level and above, 11.2% of the open listings, is also set aside. The tags are far larger than the hide: 37.5% of the open listings have a senior title, and 15.2% are open 90+ days.
- **The text tags would hide the most if turned into filters:** an opt-in "asks 5+ years" filter would remove another 30.6% of what an applicant sees, while "degree required, no equivalent" would remove 2.5%. Staffing agencies and recruiters are 1.5% of the open listings, and marketing specialists 1.9%.
- **The final table is `scope_flags.parquet`:** 878,937 listings and 14 columns, keyed by `id`, so it joins to `EDA_SDA_1_Lean`'s in-scope table without copying it.

## 8. Conclusion and next steps

**Keep, tag, hide, drop.** `EDA_SDA_1_Lean` drops three occupations that aren't a tech path (document management specialists, GIS technologists and search marketing strategists, 29,354 listings), leaving the scope of 878,937 listings (821,589 originals) from 68,863 companies. This notebook then decides what the app does with each listing. **One rule hides:** likely ghosts. **Seven rules tag:** manager level and above, open 90+ days, staffing agency or recruiter, senior title, asks 5+ years, degree required with no equivalent, and marketing specialist (O*NET 13-1161 titles without an analysis word). Everything else is kept. Nothing in the scope is removed here.

### Answers to the clarifying questions

| Question | Answer |
|:--|:--|
| Which long-open listings are ghosts, and which are still hiring? | A company that has gone quiet separates them. **7,229 likely ghosts are hidden** (3.4% of the open listings); the other 31,816 listings open 90+ days (15.2%) stay visible, tagged and ranked lower. The age cutoff is a choice: about 30% of old open listings still close every month, so there is no natural break (3) |
| Which recruiter and agency listings do we tag, and are fast recruiters just staffing agencies? | 1.9% of listings (679 companies) are tagged as posted by a staffing agency or recruiter, and they close in a median 13.1 days against 23.6 for employers. Recruiters close fast even at employers (7.6 days for recruiters with 5+ listings against 24.1 where none is named), so the tag and the recruiter signal are not the same thing. The tag is a first-pass keyword rule, so it tags and never hides (4.3) |
| How much is left once the hiding rules are applied? | **96.6% of the open listings** after the hide rule, and **85.7% for an applicant with no management experience** (also setting aside manager level and above). An opt-in "asks 5+ years" filter would remove another 30.6%; "degree required, no equivalent" another 2.5% (7.1) |
| Which role buckets have enough listings and career changers? | All twelve have enough listings to split by source type or finer; eight also have 30+ career changers (38 to 669) and get their own analysis and compass paths. Business and data analysis, marketing specialists, technical sales and other computer roles have none in the frozen profiles, so they are analyzed with a close bucket (5.3) |
| What does the 2026 market look like, for the compass? | Software development (66K open, $73.54 an hour) and networks and systems (41K) are the largest; IT support is the door (44% open to career changers, $31.10); technical sales is the slowest to close (38 days). Most compass paths are thin, so the compass shows how many people each rests on (6) |

### For the career changer
- **IT support is the clearest door** (44% open, 22% entry level) at the lowest pay ($31 an hour); **Quality Assurance and Data Analytics** are the most open TripleTen programs (27% and 26%; Data Analytics at $51.50).
- **Senior-titled listings are mostly out of reach** (81% ask 5+ years, 4.8% open to newcomers): shown with a "senior" tag and ranked lower, not hidden.
- **Most paths are thin,** so the compass always shows how many people a path rests on. The largest within-tech move is IT support to networks and systems (279 people).

### What this notebook leaves to others
- **How listings behave** (company volume, source, freshness, pay disclosure, reposts, missing values) is answered in `EDA_SDA_1_Lean` (5, 6) and tested with the confounders held fixed in `EDA_SDA_4`.
- **The modeling columns** the original built (copy detection, the company-history, market and pay features, the company facts) and their screening move to `EDA_SDA_4_Lean`. The original `EDA_SDA_2` stays their source until then, so it can't be retired yet (`../ANALYSIS.md`).
- **The app's filter fields** (job type, work arrangement, schedule, title cleaning) are left out; the app builds what it needs.

### Next steps
- **EDA_SDA_3 (text):** the keyword rules here (staffing agency, marketing specialist) and in the text (years asked, degree) are unvalidated, so they stay tags and opt-in filters; clean descriptions from `scope_descriptions.parquet`; map skills to O*NET; add evergreen wording to the ghost check.
- **EDA_SDA_4 (statistics):** the behavior analysis and the feature screening, with the confounders held fixed.
- **Ask PredictLeads:** why job-board listings stop after May, and why descriptions go missing more often from June.
