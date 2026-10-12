# Project Outline

**PredictLeads × TripleTen externship** · Jacob Colombo · Updated 2026-10-10
*A working proposal; the team's final focus is still being agreed.*
*How we work: [`STANDARDS.md`](STANDARDS.md) · Decisions: [`DECISION_LOG.md`](DECISION_LOG.md) · Data: [`DATA.md`](../DATA.md)*

---

## The idea

**Our applicant is a career changer:** anyone switching from their current background to a different role or field. They see far more listings than they can apply to, and most of them ask for experience in the new role that they don't have yet. > **Pitch:** Everyone tells career changers what they match. We tell them what's realistically reachable, which training actually got people like them there, and how fast they have to move, using real career switches and live 2026 listings.

For a given career changer, we're building a tool that does two things:
- **Role discovery:** suggests which roles their skills transfer to, within the scope: **the roles TripleTen trains for, plus adjacent tech roles** (computer and mathematical roles, sales engineers, business and market research analysts; each labelled with its TripleTen program).
- **Prioritization:** ranks the listings in the roles they choose: which to apply to first, and why.

The ranking weighs two questions for every listing:
- **Can I realistically get it?** Career-change accessibility: entry-level seniority, years of experience required, a degree *or equivalent experience*, training offered, career-changer wording; later, **transferable fit** (which skills carry over from the applicant's old field). Listings with hard blockers (too many years asked, a degree or license the applicant lacks, manager-level roles for applicants without management experience) are **hidden**.
- **Is it worth applying now?** How long listings like it stay open, competition, pay and source. The model learns how much each factor counts.

We can't see applications or hires, so the ranking rests on what PredictLeads' data *does* show: how long listings stay open, who posts them, where, and what they say, including what they ask of a newcomer.

**The end product:** a web app with a **priority queue** of listings for a career changer. Each listing shows how open it is to someone new to the field, its estimated apply-by date, why it ranks where it does, and a link to apply. **In the computer and mathematical scope, at least 38% of listings are on the employer's own website** (the listing's domain matches the company's; EDA_SDA_2 3.2d), and up to 65% if every "company site / other" listing were, so many ranked listings can link straight to the employer's posting page.

## What makes this different

- **Paths real people took:** which training led people like the user into which tech roles, learned from real career switches.
- **A retraining compass (the foundation):** prior career → the training people took → their first tech job → that role's 2026 openings, pay, closing speed and expected job growth. It serves two moves: **changing careers** (retraining into a new field) and **moving up** (training to reach senior or manager roles in the user's own field).
- **An Ideas tab (to try):** a separate tab built on the compass, from the user's résumé: skills to build, career options down the line, the pay increase, expected job growth and stepping-stone roles. Time frames are kept vague until the team decides.
- **Moving across tech roles:** people already in tech or math who moved into a different role (2,049 in the resumes, for example IT support → networks or security, developers → databases or data), so the tool also serves someone changing direction within tech. Moving *up* (to senior or manager roles) is counted separately in Profiles 4a.3.
- **How fast to move:** an apply-by estimate on every listing; and listings a career changer can't get are hidden.
- **Pick a date:** the priority list as it stood on any date in the data (for example July vs. September 2026), to show how the best options change over time.

Details, evidence and pitch wording: [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md). Every compass idea in one place: [`APPLICATION.md`](APPLICATION.md#3-the-retraining-compass), section 3.

## Data

- **Source:** PredictLeads job openings and companies datasets, the export of 2026-09-30, the data's end date (PredictLeads exports its whole database every day and calls each export a snapshot; we use one).
- **Analyzed:** the full file (EDA_SDA_1, 2026-10-11): 13,415,388 listings; 13,017,464 English (or unlabeled) and US (or no country), first seen 2026-03-30 to 2026-09-30. Cleaning removed none; 1.4M reposts flagged; 11.7M originals from 324K companies.
- **The scope table (EDA_SDA_1):** 1.0M listings (947K originals) from 84K companies in 11 role buckets, the roles TripleTen trains for plus adjacent tech roles (59% under a TripleTen program, 41% adjacent); English, US (only listings that name the US; no-country listings dropped 2026-10-12, so these counts change on the next rerun); 24% still open on September 30.
- **EDA_SDA_2 (full scale, 2026-10-11):** the same 1.0M scope listings, cleaned and flagged (nothing removed), saved as `scope_table.parquet`; 23.2% are open on September 30 and visible to everyone once likely ghosts and document management are hidden.
- **Career switches:** the public Kaggle structured resume dataset (MIT license): 54,933 resume records → 17,952 unique people (each resume appeared about 3 times; almost all with readable job dates) → **2,813 career changers**: 2,049 changed roles within tech, 407 retrained after working in another field, 357 retrained as new grads from another field. Career paths from real switches spanning two decades; the 2026 listings supply today's openings, pay and timing. Built in `notebooks/Profiles.ipynb`.
- **Job growth expectations:** expected employment growth by occupation over at most the next 5 years, from official occupational projections, used to recommend roles with the future in mind. They're published by occupation code, so they plug straight into the O*NET codes on listings and role buckets. Current release: BLS 2025–2035; plan in [`JOB_GROWTH.md`](JOB_GROWTH.md).
- **Data handling:** stays local; nothing is shared with outside AI services; only aggregates are reported.

## What we've found so far

| Finding | Evidence | What it means for a career changer |
|:--|:--|:--|
| **Company hiring volume is the strongest signal** | At companies with 500+ listings in any field, scope listings close in a median **10 days**, vs. **50** at companies with 1–4, counting only each company's earlier listings (the way the model sees it; 14 vs. 56 over the whole six months; EDA_SDA_1 Q1d). Part of it is crawl timing: on the sample, like-for-like (listings with an employer posting date, EDA_SDA_2 3.3) the gap was about 1.4–1.5× | Listings from high-volume employers need fast action |
| **The application window is short** | **29%** of scope listings are gone within a week; the median is **24 days**; technical sales is the slowest bucket (38 days) | "Still open / apply-by" is a core factor |
| **Source matters** | LinkedIn, Oracle and Taleo: **15 days**; company sites **21**; Greenhouse, Lever and Ashby **49–71** (median days open) | Source adjusts urgency |
| **Some listing traits carry signal; keyword flags don't** | Disclosed pay, long descriptions and hiring platforms: 12–16 points more still open after a week; remote wording: no gap; urgency left to EDA_SDA_3's NLP | Model the traits; test richer NLP |
| **Factors differ by role** | In the scope: pay **$31–$73 an hour**, openness to career changers **9–44%**, entry level **2–23%** by role bucket | Role-aware estimates and hiding rules |
| **The best-paid field is the hardest to enter** | Computer and mathematical is 1st of 23 fields on pay (**$66.73** an hour) and remote work (33%), but 22nd on entry level (**9.4%**) and openness (21%), and last on asking 2 years or fewer (17%) | We target it (the roles TripleTen trains for) and hide what a career changer can't get |
| **How open listings are to career changers** | Entry level ranges from **3%** (management) to **76%** (building and grounds) by field; in the scope, IT support is the door (**44%** open, 23% entry level) and Data Analytics roles follow (30%) | Becomes its own ranking factor |
| **Collection effects look like signals until checked** | 22% of scope listings have no description (28–30% from June on) and close in a median **4.4 days vs. 30**; job-board listings stop after May (EDA_SDA_2 3.2, 3.9) | Missing pay and missing years looked predictive only because of them; every timing result is shown with and without them |
| **The volume effect depends on company type** | At companies with 500+ listings, public companies' listings stay open **26 days** vs. **18** for others, among listings with a description (EDA_SDA_2 3.8, 3.9) | Fast closing is mostly a private, high-volume-employer pattern |
| **Ghost listings can be told apart** | 78% of stale listings come from a company still posting, vs. 97% of younger ones; **10K likely ghosts** (4.4% of open listings) are hidden (EDA_SDA_2 3.1) | Fewer dead ends in the queue |
| **What a career changer can reach** | The rules in force hide **15%** of open scope listings (likely ghosts, manager level, document management); reachable listings pay **$9–35 an hour less** in every role bucket (EDA_SDA_2 3.11) | The app shows the pay trade-off instead of hiding it |
| **Entry doors and later moves** | IT support: 23% entry level, 44% open to career changers, $31 an hour; Data Analytics roles 30% open at $47.60; IT project management: 2% entry level, 9% open, $72, mostly reached from another tech role (EDA_SDA_2 3.6, 3.12) | IT support is a first step; IT project management a later move |
| **Real career switches exist in public resumes** | **2,813 career changers** (funnel in the Data section above) | Training data for role discovery and fit. Rules spot-checked on random samples |

The listing itself matters more than the job title: two listings in the same field can differ more than two different fields do.

## Prioritization factors

| Factor | Measured from |
|:--|:--|
| Career-change accessibility | Hard blockers hide a listing; softer signals raise it. Seniority (entry level); listing text: years of experience required, degree required, "or equivalent experience", entry-level and training wording, career-changer wording |
| Still open / apply-by | First-seen and last-seen dates (survival analysis) |
| Freshness | Days since first seen (the crawler finds listings a median of 1.5 days after posting) |
| Competition | Company hiring volume; reposts (times listed, days since last listed), split into relisted after closing vs. cross-posted on several boards at once |
| Urgency (not testable here) | How many boards carry the same job at once: only 13 cross-posted listings in the scope (EDA_SDA_2 3.5), so it isn't used |
| Hiring difficulty | Listing age and reposts. Likely ghosts are hidden; other listings open 90+ days are labelled and ranked lower ("possibly hard to fill" was dropped, EDA_SDA_2 3.1). Reposts: 6.1% of the scope; 7.9% of listings are relisted copies of a closed job, closing in a median 7.9 days |
| Pay | Hourly pay rate, standardized in EDA_SDA_1 6.5 (salaried pay ÷ 2,080 hours; disclosed for about 47% of listings) |
| Employer intent | Listing text: urgency, remote, description length, evergreen wording |
| Source | Hiring platform, job board, or company site; whether the listing is on the employer's own domain |
| Company type | Publicly traded, has a parent company (companies table); changes how fast high-volume employers close (EDA_SDA_2 3.8) |
| Occupation group | O*NET occupation group; also for showing related jobs |
| Job growth expected | Expected employment growth for the listing's role over at most 5 years. The same for every listing in a role, so it shifts role recommendations (role discovery, compass) toward growing roles, as a visible, adjustable weight |
| Transferable fit | Listing text vs. the career changer's profile, with O*NET skills connecting the old and new occupations (learned from real career switches in `Profiles.ipynb`; volunteer profiles later as a recent test set) |

## How we're doing it: seven notebooks

Each notebook answers one question and hands its final table to the next. The TripleTen course templates are the baseline for structure and methods, taken further with advanced techniques.

| # | Notebook | Question | Key methods | Status |
|:-:|:--|:--|:--|:--|
| 1 | **EDA_SDA_1** | What's in the data, and what's the scope? | Cleaning with an audit trail, survival curves, factor comparisons, scope decisions defended with evidence; career-change accessibility (Q1g) | **Done** |
| 2 | **EDA_SDA_2** | Which scope listings do we keep, and how do they behave? | Additional cleaning (staffing agencies and recruiters, contract types, titles with an AI/ML flag, the app's filter fields); ghost listings vs. desperate employers; checks on source, freshness, pay, reposts, company volume; companies-table join; missing-data patterns; **narrowing computer and mathematical to what a career changer can get**; pairing it with the team's backgrounds; outliers; hiding rules counted; the compass's 2026 side | **Done at full scale** (2026-10-11): scope table of 1.0M listings |
| 3 | **EDA_SDA_3** | What do the descriptions say? | Text features only, no matching models: description cleanup; evergreen, urgency and hiring-difficulty phrases; career-change flags and hiding rules validated (experience, education, credentials, training); skills mapped to O*NET | **Next** |
| 4 | **EDA_SDA_4** | Which features hold up? | Pearson + Spearman correlation, multicollinearity (VIF), hypothesis tests with effect sizes, confounding and company-clustering checks, competing risks | Planned |
| 5 | **TimeSeries** | When do listings appear and close? | Decomposition, stationarity tests, lag features, ARIMA / SARIMA, survival by month | Planned |
| 6 | **ML_Fit** | Which roles and listings fit a career changer? | Keyword overlap → TF-IDF cosine → BGE embeddings (local) → RRF hybrid; skills-based matching on O*NET skills; a trained layer on real career moves (Option B); hit rate@3/@5 per pool, old-field bias; SHAP; role discovery and the compass | Planned |
| 7 | **ML_Rank** | How should listings be ranked? | Listing timing model (logistic regression, LightGBM, CatBoost, XGBoost; Cox and survival models; LambdaRank); combined with fit, accessibility and pay into the priority queue by an evidence-based, tunable formula; evaluated against newest-first; SHAP | Planned |
| side | **Profiles** | Who really changed into or within tech or math, and how? | Public resume dataset; duplicate removal; whole-timeline rules (`src/career_switch.py`); random spot-checks | **Frozen** (refrozen 2026-10-10 after adding an IT project management bucket): rules spot-checked; final tables in `data/profiles/` |

## Key decisions so far

- **Focus: career changers** (part of the project description): anyone switching from their current background to a different role or field. Every factor, segment and model choice is judged by what it means for someone switching fields.
- **Career-change accessibility is its own factor,** read from seniority and listing text. It's known when a listing appears, so it's safe to model.
- **Scope:** English-language listings in the United States, first seen April–September 2026 (only six months of history, so no seasonal claims). The data's first two days, in late March, are kept and flagged `partial_month` rather than removed.
- **No early filtering:** data-quality problems are fixed during cleaning; scope filters are applied only at the end, after each is defended.
- **Long-open listings are kept:** they may be ghost listings *or* employers struggling to hire. EDA_SDA_2 told them apart: likely ghosts (old, from a company gone quiet) are hidden in the app; other listings open 90+ days stay visible, labelled.
- **Reposts are a feature, not noise:** a job posted again and again may be hard to fill. In the scope they're rare (98.6% first listings), so they're expected to add little.
- **Standard confounders, compared within groups:** company volume, source and platform, staffing agencies, role bucket and seniority, month, state, company type and whether the listing has a description; timing results are also shown without likely ghosts.
- **IT project management is a later move:** kept, but not suggested as a first tech job unless the user's prior career has a real path into it.
- **Two features:** role discovery (which roles in the scope, the roles TripleTen trains for plus adjacent tech roles, a career changer's skills transfer to) and prioritization (ranking listings in the chosen roles).
- **Target: the roles TripleTen trains for, plus adjacent tech roles (US):** computer and mathematical roles (best-paid and most remote of 23 fields, but 22nd on entry level), plus sales engineers and business and market research analysts. Listings a career changer can't get are hidden. EDA_SDA_2 analyzes 11 role buckets aligned with the career-switch profiles; every listing is kept.
- **Source backgrounds: the team's own.**
- **Hide what a career changer can't get:** validated text blockers (EDA_SDA_3), and per-applicant blockers in the app: a missing degree or license, or manager-level roles for applicants without management experience. Every other factor is weighted by the model during ML training.
- **Job growth expectations replace the wider external-data list:** one number per role over at most 5 years, from official projections that join on occupation code, so recommendations account for where roles are heading.
- **Pick a date (as-of date):** the queue can be shown as it stood on any date in the data: listings live that day (first seen on or before it, last seen on or after it), scored, and its model trained, only with what was known that day.
- **Two supervised models, combined by a formula:** listings → how long they stay open (PredictLeads), and career changers → the role bucket they moved into (the 2,813 career changers). No data links a person to a listing, so the role buckets connect the two. Volunteer profiles, once the advisor approves, are a recent test set. See the 2026-10-09 entry in [`DECISION_LOG.md`](DECISION_LOG.md).

## Honest limits

- We see listings appear and disappear, **not hires**. A listing closing could mean filled, cancelled or moved.
- `first_seen_at` reflects PredictLeads' crawl timing, so day-of-week patterns rely on posting dates (about 45% of listings).
- Six months of data rules out seasonal advice.
- We can't see who gets hired, so we can't tell whether employers actually hire career changers. Accessibility is read from what listings *ask for*.
- **Collection changes in the data:** job-board listings stop after May 2026 and listings without a description rise to about 30% from June; both are handled as confounders, and asked of PredictLeads.
- The career-change flags are keyword rules until EDA_SDA_3 validates them (the degree flag fires on only 3–6% of scope listings, so its pattern likely misses common wording). Hiding listings on an unvalidated rule could hide listings a career changer could get, so rules hide nothing until they're validated.
- The career-switch rules were spot-checked on random samples, not every person, and most switches are from 2010–2019 (median 2014), so the resumes teach career paths while the 2026 listings supply the market. O*NET's current Career Changers Matrix checks that the paths still hold.

- Job growth figures are projections, not forecasts we make or check, and they're national, by occupation. They assume a full-employment economy and leave out business cycles; expected accuracy is moderate for years 1–2 and low to moderate for years 3–5 (`JOB_GROWTH.md`). IT project management has no projection of its own; it uses the 15-1299 catch-all as a proxy. A growing occupation doesn't mean more entry-level openings; our listings measure those.
- The resumes only show people who made a switch, not those who tried and didn't, so paths show where switchers went, never how likely a switch is to work.
- Early dates in the date picker miss long-open listings first seen before 2026-03-30 (the six-month window), so July onward gives fair comparisons: live listings jump from 2,459 on April 1 to about 32,000 by June 1, and their age mix settles from July (EDA_SDA_1 8b).

## Questions for PredictLeads

Crawl frequency and date precision; what `status` and `last_processed_at` mean; the time zone of posting dates; whether older daily exports exist; how duplicates and reposts are handled; how `seniority` is assigned; whether real listings can appear in our final presentation; **why job-board listings stop after May 2026** (494 and 526 in April and May, then almost none; EDA_SDA_2 3.2c); and **why more listings have no description from June on** (about 30%, vs. 2–10% in March–April; EDA_SDA_2 3.9).
