# Vision

**Status:** What's still to build · **Last updated:** 2026-10-12

> Everything planned for the app that isn't built yet. What's already built, in the analysis and the app (**FathomLite**, in `app/`), is in [`STATUS.md`](STATUS.md); when a piece here is built, move it there. Tasks: [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md).
> Items marked **TBD** need a team decision. Follow the [Data & Security Rules](../README.md#data--security-rules).

---

## Agreed changes still to make

From the six merge decisions, agreed by Jesse and Jacob on 2026-10-11 (summary in [`DECISION_LOG.md`](DECISION_LOG.md)):

| Decision | What changes in the app |
|:--|:--|
| 1. One scope | The analysis scope (the roles TripleTen trains for plus adjacent tech roles, in ten role buckets, section 3.2) replaces the app's three areas (AI/ML, data, technical sales), which become filters that map onto the buckets |
| 2. "Still open" | A listing counts as open when it was last seen within 2 days of the data's end, 2026-09-30 (or of the date picked), instead of every listing without a "closed" status; the "historical listing" wording stays |
| 4. Who ranks | The app's TF-IDF + BGE (RRF) ranker is `ML_Fit`'s starting point; each addition (role suggestions learned from real career moves, timing, hiding rules) has to beat it |
| 5. One listing set | The app's corpus and BGE embeddings are rebuilt from the analysis stream's listings (those with a usable description), replacing its own targeted pool; about three times as many embeddings to compute |
| 6. Two compasses | The current Compass tab is renamed **Saved-jobs insights**; the **Retraining compass** (section 3) is added as its own view |

**Connections to build** between the analysis and the app:

| Connection | What it brings |
|:--|:--|
| Analysis listing set → app corpus | Decision 5 |
| `ML_Rank` → backend | Timing (chance still open, apply-by date) and the hiding rules |
| `ML_Fit` → backend → retraining compass view | Suggested roles, transferable fit and the compass tables |
| BLS projections → `ML_Fit` | Expected growth per role, after PredictLeads' written consent to combine data ([section 3.3](#33-expected-job-growth)) |

---

## 1. Vision

A career changer has limited time, a limited number of applications, and a background most listings in their new field don't ask for. The app shows them **where an application is most worth spending**: live postings in the field they're moving into, in a searchable list, ranked by a priority model we tune ourselves, with the reasons shown. Every listing answers two questions: **can I realistically get it** (how open it is to someone new to the field, and which of their skills carry over) and **is it worth applying now**.

**The user** is a career changer (anyone switching from their current background to a different role or field) moving into the roles TripleTen trains for and adjacent tech roles (computer and mathematical roles, plus sales engineers and business and market research analysts), from one of the team's backgrounds, or a teammate with their own profile. The app does two things, with the **retraining compass** at the center: **role discovery** (which roles fit them, which paths lead there, and what to build next) and the **priority queue** (which listings in those roles to apply to first). Listings they can't get are hidden. Someone already in tech or math gets the same tools for **moving up the career ladder** into a different role.

The priority model isn't guesswork. It's built from the full analysis, starting with the EDA in `analysis/notebooks/EDA_SDA_1.ipynb`. Every number the app will show (openness to career changers, apply-by date, chance still open, transferable fit, competition, pay) comes from a target we studied and validated there.

The core of the app is the **priority queue**: listings ranked for this career changer. The timing advice from the analysis sits around it:

| From the analysis | In the app |
|---|---|
| Prioritization model | The priority queue: the list sorted by the priority score |
| Career-change accessibility | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer: years asked, degree or equivalent experience, training offered |
| Still open / apply-by | An estimated apply-by date on every listing, and an apply-by filter |
| Source and arrival cadence | Check schedule: which sources and companies to watch, and how often; alerts for new listings |
| Job growth expectations | Expected growth of each role over at most 5 years; it shifts role suggestions toward growing roles (a visible, adjustable weight) and is shown in role discovery and the compass |
| Listing activity over time | A date picker: the queue as it stood on any date in the data |

---

## 2. Screens and features

### List view (Discover)

*Today Discover ranks listings by fit to the profile, with a search applied on a button press and filters for location and description. Still to build:*

- **Ranking by the priority score** (section 4) instead of fit alone.
- **Date picker ("as of"):** shows the queue as it stood on a chosen date, for example July 1 vs. September 1, 2026. See [Pick a date](#pick-a-date-as-of-view).
- **Live search:** the list narrows with every letter typed. It matches titles, companies and, optionally, description text.
- **Sort:** priority (default), openness to career changers, apply-by date, transferable fit, pay, freshness.
- **Filters:** any number at once, in any combination, and each one shows how many postings match. **They apply to the priority queue itself:** the queue is always the filtered set, ranked by priority score, so "remote or hybrid, entry or mid, data science, Texas" gives a ranked queue of just those. Active filters show as removable chips above the list, and a combination can be saved as a named view (for example "Remote data jobs"):
  - **Occupation:** one or more role buckets (data science, software development, IT support, …; built in EDA_SDA_2 3.6), and within them specific O*NET occupations. Picking roles in role discovery or the compass sets this filter
  - **Priority:** minimum score, or top N
  - **Apply-by:** closing within N days, or before a date
  - **Hidden automatically:** listings a career changer can't get (too many years asked, a degree or license this user lacks, or manager-level roles if the user has no management experience), listings for occupations that aren't a tech path (document management specialists, and marketing specialists: O*NET 13-1161 titles that don't name analysis or research, decided 2026-10-11), and **likely ghost listings**: more than 90 days old on the date being viewed **and** from a company that has posted nothing new in 60 days, or with a posting date over a year old (EDA_SDA_2 3.1). A note shows how many were hidden, and why. **Other listings over 90 days old stay visible,** labelled "open 90+ days" and ranked lower by the priority model; the freshness filter hides them for anyone who wants only recent listings. Hidden listings are never deleted: a "show hidden" toggle brings them back
  - **Training offered:** only listings that offer training or welcome career changers
  - **Transferable fit:** minimum score (needs the user's background profile)
  - **Pay:** minimum hourly rate; pay disclosed only
  - **Freshness:** first seen within N days. **The age shown on a listing** (decided 2026-10-11): "posted X days ago" from the employer's posting date when there is one (39% of scope listings; implausible dates blanked), otherwise "found X days ago" from first seen, so a listing the crawler found late doesn't look fresher than it is (the crawler finds listings a median 1.5 days after posting, EDA_SDA_2 3.3)
  - **Competition:** low, medium or high
  - **Source:** hiring platform or company site; specific platforms
  - **Location:** one or more states (from `location_data`), listings naming several locations included where any matches (today: one free-text location)
  - **Work mode:** remote, hybrid, onsite (`work_mode`; "not stated" shown separately rather than hidden)
  - **Schedule and job type:** full time, part time (`schedule`); internship, temporary, contract, regular (`job_type`)
  - **Role level:** entry, mid_senior, manager, leadership (`level`; leadership = director, head, vice president, president, c_level, founder, partner). "mid_senior" stays whole, as PredictLeads labels it (LinkedIn's single level for experienced individual contributors); a "senior" label marks titles with senior, lead, principal, staff or architect (`senior_title`, decided 2026-10-11), and for users without senior experience these listings are **shown but ranked lower, not hidden** (81% ask for 5+ years and 4.8% show openness to career changers, vs. 42% and 17% for other mid_senior listings; EDA_SDA_2 3.7c). Manager and leadership listings stay hidden for users without management experience unless they turn on "show hidden"
  - Every filter allows several choices at once (for example remote **or** hybrid, entry **or** mid); different filters narrow each other (remote or hybrid **and** entry or mid **and** data science)
- **Each row shows:** title, company, priority score, an "open to career changers" badge, apply-by date, transferable fit and pay.

### Saved jobs

*Saving, stages and notes are built. Still to build:*

- Every search, sort and filter working on the saved list too, so the user can rank their own shortlist.
- **Reminders** when a saved posting's apply-by date is close.

### Detail view

*Today it shows the listing, its full text, a skills check, the company and a link to the source listing. Still to build:*

- **Apply-by date** with its estimated chance still open, and a short explanation of what it's based on
- **What it asks of a newcomer:** years of experience, degree or equivalent experience, licenses, training offered
- **Transferable fit** with the skills that carry over from the user's old field, and gaps (requirements their profile doesn't show yet)
- **Competition** signals: reposted, company hiring at volume, similar postings open
- **Rating breakdown by feature** ("why it's ranked here"), right under the title:
  - **Each factor's rating** (chance still open, career-change accessibility, transferable fit, pay, competition, freshness, and expected growth for the role), on one 0–100 scale, with a plain-language line ("pays more than 70% of listings in this role").
  - **How much each factor added to the priority score:** rating × the user's current weight, shown as bars that add up to the total, so moving a weight slider visibly changes the breakdown.
  - **What drives the model-based ratings:** for chance still open and fit, the features that pushed the rating up or down the most (SHAP values, grouped under plain names such as "company hiring at volume", "posted on the employer's site", "skills that carry over"), top three each way.
  - **Unknown is shown as unknown:** a factor the listing can't support (no pay, no description) says "not disclosed" or "not enough text" instead of a made-up rating, and the breakdown says how it was handled.
  - Ratings are estimates from postings appearing and disappearing, never a chance of being hired.

### Role discovery ("Roles for you")

- From the background profile, suggests the roles (O*NET occupations) their skills transfer to.
- Learned from real career switches: people in a public resume dataset who retrained into computer and mathematical roles (`analysis/notebooks/Profiles.ipynb`).
- Each suggested role shows the skills that carry over, the main gaps, how many reachable listings are open now, and median pay.
- Picking roles sets the queue's occupation filter, which the user can widen or narrow like any other filter.
- **IT project management is a later move** (decided 2026-10-10): only 9% of its listings show openness to career changers and 2% are entry level. It isn't suggested as a first tech job and isn't in a newcomer's default occupation filter (the user can add it), unless their prior career has a real path straight into it (business: 17% of retrained business people), in which case it's shown with its low openness. Users already in tech see it under moving across and up.

### Retraining compass, moving across and up, and the Ideas tab

*What each shows, the path questions, the Ideas tab's ideas and the honesty rules are in [section 3](#3-the-retraining-compass); this section covers only the screens.*

- **Retraining compass view:** starts from the user's prior career (or current tech role, for moving across and up). It shows the training people like them took, the roles it led to, and each role's 2026 openings, pay, closing speed and expected growth. Every path shows "based on N people". It sits next to the existing Compass tab, renamed **Saved-jobs insights** (decision 6): that one analyses saved jobs, this one career paths.
- **Moving across and up:** the same view for users already in tech, with each move marked sideways or up. IT project management appears here as a step up.
- **Ideas tab:** a separate tab built from the résumé on the compass (skills to build, career options, pay increase, expected growth, stepping-stone roles); to try, with time frames kept vague.
- Picking a role in any of these sets what the priority queue ranks. The Coach explains the paths in plain language. Everything comes from the compass tables assembled in `ML_Fit`'s last section.

### Pick a date (as-of view)

- The user picks a date; the queue shows the listings **live on that day** and ranks them with **only what was known that day**. Comparing July with September shows how the best options change.
- **What the dates mean:** `first_seen_at` is the first day PredictLeads' crawler found a listing (a median of 1.5 days after the employer posted it); `last_seen_at` is the last day the crawler still found it. A listing is live on a date when it was first seen on or before it and last seen on or after it. **Still open** (decision 2): a listing counts as open when it was last seen within 2 days of the data's end, 2026-09-30, or of the date picked.
- **No hindsight in the scores:** listing age is counted up to the chosen date; reposts and company hiring volume only from listings first seen before it; and `last_seen_at` only decides which listings were live, never a score.
- **No hindsight in the model:** the model that scores a date is trained only on what was known by that date; a listing still open on that date counts as still open in training, even though we later know when it closed. The simple version for the demo offers only dates in the test period (September), which the main model never learned from; any earlier date needs a model retrained as of that date.
- **A built-in check:** for past dates we know which listings actually closed and when, so the app can show how good its apply-by estimates were.
- **Fair from July:** listings first seen before 2026-03-30 are missing, so earlier dates undercount long-open listings. Live listings jump from 2,459 on April 1 to about 32,000 by June 1, and their age mix settles from July (EDA_SDA_1 8b); the picker starts at July 1.

### Background profile (for transferable fit)

*The Profile page, with Magic fill from a résumé, is built. Still to build:*

- Pasted text or a LinkedIn data export (the Profile, Positions, Education and Skills files) as further ways in.
- Picking the field to move into, or letting the app suggest related fields from O*NET's related occupations.
- A transferable-fit score for every posting from `ML_Fit`'s model, which starts from the app's TF-IDF + BGE ranker (RRF, run locally; decision 4) and adds skills matched through O*NET and a layer trained on real career moves, so the old field's wording doesn't dominate.
- **Privacy stays as now:** the résumé is read locally and leaves the machine only through Magic fill, with consent; personal details are never stored or logged beyond the session unless the user chooses to save them.

### Market panel and check schedule

*The Signals tab and Ask the market are built. Still to build:*

- **Market:** this month's new-posting volume for the field, and each role's expected job growth.
- **Check schedule:** "check Greenhouse and these 5 companies every 3 days," based on how fast new listings arrive and close.

---

## 3. The retraining compass

Every compass idea in one place: what it is, what it shows, the ideas still to try, and where each piece is built. The compass is the project's foundation and its lead differentiator ([`DIFFERENTIATORS.md`](DIFFERENTIATORS.md)). None of it is in the app yet. Decisions behind each piece are dated in [`DECISION_LOG.md`](DECISION_LOG.md).

### 3.1 What it is

**Prior career → the training people took → their first tech job → that role's 2026 openings, pay, closing speed and expected job growth.**

It serves two moves:
- **Changing careers:** retraining from another field into a computer and mathematical role.
- **Moving across or up within tech:** people already in tech moving to a different role, or toward senior and manager roles.

It's built from **2,813 real career switches** in a public resume dataset (`analysis/notebooks/Profiles.ipynb`), joined to the **2026 PredictLeads listings** by role bucket. Career paths come from real switches spanning two decades; the 2026 listings supply today's openings, pay and timing.

### 3.2 What it shows

**Ready in the analysis** (not yet in the app): where people from each prior career and each kind of training landed, moves within tech (Profiles 4a.2b, 4a.3), each role's 2026 market (open listings, new per week, hourly pay, closing speed, entry-level share, openness to career changers, remote share) and the people behind each path (EDA_SDA_2 3.12).

**Still to build:**

| Piece | What the user sees | Built in | Status |
|:--|:--|:--|:--|
| **Expected job growth** | Growth for each role over **at most the next 5 years**, from official occupational projections (BLS plugs in by occupation code); it drives forward-looking recommendations (section 3.3) | EDA_SDA_2 3.12; used in `ML_Fit` | Column ready; BLS 2025–2035 projections released; needs PredictLeads' consent; plan in [section 3.3](#33-expected-job-growth) |
| **Suggested roles for this user** | Roles the user's background transfers to | `ML_Fit` | Planned |
| **The compass views** | Section 2 | The app, from `ML_Fit`'s compass tables | Planned |

**Role buckets:** eight computer and mathematical buckets, the same on listings and résumés (IT support, networks and systems, security, database, software development, quality assurance, data science, IT project management), plus two on the listings side only (decided 2026-10-10): **technical sales** (sales engineers) and **business and data analysis** (management and market research analysts). The career-switch profiles have no paths into the last two, so the compass shows their market but no paths. Every listing also carries its TripleTen program (`analysis/src/career_switch.py`).

### 3.3 Expected job growth

The compass's **expected job growth** column: the whole plan. Decisions behind it: [`DECISION_LOG.md`](DECISION_LOG.md) (2026-10-09, 2026-10-10); data reference: [`analysis/docs/data.md`](docs/data.md).

#### Why: recommendations based on future estimates

The point of expected job growth is to **recommend roles with the future in mind**, not just to describe them: among the roles a person fits, a role expected to grow ranks higher than one expected to shrink.
- **One number per role bucket**, from official occupational projections.
- **At most 5 years out.** Projections further ahead are too uncertain to act on; the exact horizon within 5 years is still the team's call.
- **A visible adjustment, not a trained input.** Growth is a future estimate, so it can't be checked against the historical career moves the fit model learns from. `ML_Fit` applies it after the fit model, shows how much it moved each suggestion, and lets the user change its weight.
- **Never alone.** It's occupation-wide, so it's always shown with the role's entry-level share and today's openings: a fast-growing role with no entry-level listings is still a later move.
- **Projections, not our forecasts:** national, by occupation, published by BLS. We convert them; we don't make or check them.

#### Sources

The core needs only BLS Employment Projections. The rest serve the proposed extensions (below) or context.

| Source | What it gives | Role | Link |
|:--|:--|:--|:--|
| **BLS Employment Projections 2025–2035** (released) | 2025 employment, 2035 projected employment and 10-year change for about 800 occupations | **Core:** the growth number | [bls.gov/emp](https://www.bls.gov/emp/) · [2025–35 summary](https://www.bls.gov/news.release/ecopro.nr0.htm) · [data tables](https://www.bls.gov/emp/data.htm) |
| Occupational Outlook Handbook | Outlook, pay and education per occupation | Context for the write-up and the Coach | [bls.gov/ooh](https://www.bls.gov/ooh/) · [Computer and IT](https://www.bls.gov/ooh/computer-and-information-technology/) |
| BLS OEWS | Yearly employment and wages by occupation | Backtesting BLS (options a, d) | [bls.gov/oes](https://www.bls.gov/oes/) |
| O*NET OnLine | Occupation codes, skills, Bright Outlook flags | Checks the title → code mapping; skills | [onetonline.org](https://www.onetonline.org/) |
| Projections Central | State short-term (2-year) and long-term projections | State-level view (option e) | [projectionscentral.org](https://projectionscentral.org/) · [long-term notes](https://projectionscentral.org/aboutlt) · [state sites](https://projectionscentral.org/stateprojectionssites) |
| BLS JOLTS | Monthly openings, hires and quits by industry | Cyclical signal (options b, d) | [bls.gov/jlt](https://www.bls.gov/jlt/) |
| Indeed job postings (FRED) | Daily postings indexes by field since 2020 | Leading signal (options b, d) | [Indeed series](https://fred.stlouisfed.org/tags/series?t=indeed) · [total](https://fred.stlouisfed.org/series/ihlidxus) · [software dev](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE) · [scientific R&D](https://fred.stlouisfed.org/series/IHLIDXUSTPSCREDE) |
| Fed funds rate (FRED) | Interest-rate history | Rate sensitivity (option b) | [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) |
| FOMC Summary of Economic Projections | The Fed's projected rate paths | Rate scenarios (option b) | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) |

#### Method (core)

1. **Download** the occupation table from the 2025–2035 projections into `data/external/` once the terms of use are checked.
2. **Join** on the 6-digit occupation code: the first part of `onet_code` on listings (for example `15-2051`) matches BLS's code.
3. **Roll up to role buckets** with `ONET_ROLE_BUCKETS`, weighting each occupation by its 2025 employment.
   - **Catch-all code:** BLS projects only 6-digit codes, so the detailed codes inside 15-1299 ("computer occupations, all other") that `ONET_DETAILED_BUCKETS` maps to their own buckets have no projection of their own. **IT project management (15-1299.09) gets 15-1299's growth as a proxy, flagged as such**; for security and networks and systems, whose 15-1299 parts are small, the 6-digit rollup is used as is.
4. **Convert to the chosen horizon** at the same yearly rate:
   - Yearly rate: (1 + 10-year growth)^(1/10) − 1
   - Growth over *n* years: (1 + 10-year growth)^(*n*/10) − 1, with *n* ≤ 5
5. **Save** `role_growth.pkl` with: role bucket, occupation codes, 2025 employment, 10-year growth, yearly rate, *n*-year growth, projections vintage. EDA_SDA_2 3.12 adds it as the compass's growth column.

**Write-up wording:** "expected growth (official projections)", never "will grow". BLS projections assume a full-employment economy and leave out business cycles; say so in the limitations.

#### Proposed extensions (not decided)

The 2026-10-09 decision dropped the other external datasets. Each option below would reopen part of it, so each needs a team decision first (run `/debate`, then log the outcome).

| Option | What it adds | Cost | Recommendation |
|:--|:--|:--|:--|
| **a. Error band from BLS's own track record** | Compare an earlier BLS projection (for example 2015–2025) with actual OEWS employment by occupation group; show growth as a range, not a single number | Small: public data, one table | **Adopt.** Strengthens the honesty rules; fast-changing data and AI roles carry the most uncertainty |
| **b. Interest-rate scenarios** | A near-term adjustment for years 1–2 under baseline, higher-for-longer and faster-cut rate paths from the Fed's projections, with sensitivity estimated from rate history and job postings (Indeed, JOLTS), fading to the BLS trend by year 5 | Medium: a lagged regression and three scenarios | **Stretch.** Rates are one driver among several (the 2022–23 tech pullback also reflects post-COVID overhiring and AI); no causal claims |
| **c. Momentum from our listings** | Each bucket's new-listings trend from TimeSeries, shown **next to** growth | Small: TimeSeries already builds the series | **Show, don't blend.** Six months of listings can't be backtested against employment or support seasonal claims |
| **d. Full horizon-weighted blend** | A yearly forecast mixing a near-term signal (postings trend + JOLTS/Indeed + rate scenario) with the BLS trend; details below | Large: needs a postings history long enough to backtest | **Later / stretch.** Our six months can't support it; would rely on Indeed and JOLTS instead |
| **e. State level** | Projections Central's state projections instead of national | Small–medium | Only if the app adds location-based growth |

**Option b in detail (interest-rate scenarios):**
1. **Don't forecast rates ourselves.** Use three paths from the FOMC projections: **baseline**, **higher-for-longer**, **faster cuts**.
2. **Estimate sensitivity:** regress the year-over-year change in Indeed postings or JOLTS openings for the field on **lagged** fed funds changes; test 6, 12 and 18-month lags and pick by out-of-sample fit; report the coefficient with its confidence interval.
3. **Apply to years 1–2 only,** fading to the BLS trend.
4. **No causal claims:** rates are one driver; 2022–23 also reflects post-COVID overhiring and AI.

**Option d in detail (the horizon-weighted blend):**

```
g(t) = w(t) × g_near(t) + (1 − w(t)) × g_BLS
```

| Year | w(t), near-term weight (starting assumption) |
|:-:|:-:|
| 1 | 0.8 |
| 2 | 0.6 |
| 3 | 0.4 |
| 4 | 0.2 |
| 5 | 0.0 |

- `g_BLS` is the yearly rate from the method above; `g_near` comes from postings trends (validated against Indeed and JOLTS) adjusted by the rate scenario.
- Postings aren't hires: scale them to employment through their historical relationship, as a directional signal only.
- **Tune the weights in a backtest**, run `/debate` on them, and log the final choice.

**Validation for options a and d:**
- **BLS track record:** an earlier projection (for example 2015–2025) vs. OEWS actuals, by occupation group.
- **Rolling-origin backtest:** rerun the method as if standing in past years (for example 2019, 2021, 2023) and compare with actuals.
- **Metrics:** MAPE and direction (growth vs. decline right).
- **Baselines to beat:** BLS interpolation alone, and "last year's growth continues".
- **Bands:** 80% and 95% intervals sized from past errors, widening with the horizon.

#### Expected accuracy (state in the write-up)

| Horizon | Confidence | Main driver |
|:--|:--|:--|
| Years 1–2 | Moderate | Postings trend and rate scenarios (if options b/d are adopted), otherwise BLS |
| Years 3–5 | Low to moderate, wide bands | BLS structural trend |
| Over 5 years | Not shown as a recommendation (decided cap); BLS 10-year figure as context only | BLS |

Fast-changing roles (data, AI and ML) carry extra uncertainty.

#### Outputs and tasks

- [ ] **Get PredictLeads' written consent to combine their data with BLS data** (Training Services Agreement 4.3; already an open advisor question on external data)
- [ ] Check the BLS terms of use; download the 2025–2035 occupation table into `data/external/`
- [ ] Join on occupation code, roll up to role buckets (employment-weighted), save `role_growth.pkl`
- [ ] Add the growth column to EDA_SDA_2 3.12 and rerun
- [ ] Team picks the horizon (at most 5 years) and decides on options a–e; log in `DECISION_LOG.md`
- [ ] `ML_Fit`: apply growth as a visible, adjustable adjustment after the fit model; show how much it moved each suggestion
- [ ] If option a: growth with an 80%/95% band per role bucket
- [ ] If options b/d: a yearly table per role bucket × scenario, years 1–5, and a fan chart per bucket (baseline line, scenario band, uncertainty band, BLS 10-year marker)
- [ ] A backtest summary (method vs. baselines) and a methods-and-limitations note: projections, national, occupation-wide, no business cycles

#### Data handling

BLS, FRED and Federal Reserve data are public. Store downloads in `data/external/` (git-ignored) like other external data, and track them for the final deletion. Nothing from our listings goes into any outside service. **Joining BLS figures to PredictLeads listings combines their data with third-party material, which the agreement allows only with PredictLeads' prior written consent (4.3):** get it in writing before the join, and log it in `DECISION_LOG.md`.

### 3.4 Path questions it answers

From the resume data; the app's Coach explains them in plain language.
- **Should I retrain for a role directly, or take a first tech role and move from there?** For data science, 218 career changers moved in from another tech role and only 19 retrained straight into it.
- **Which training led people like me to this role, and how many people is that based on?**
- **How long did it take?** People who retrained after working took a median of 15 months from their last old-field job to their first tech job.
- **Which role is the easiest door in?** IT support: the most entry-level listings (23%) and the most openness (44%), but the lowest pay ($31 an hour). Among TripleTen programs, Data Analytics roles are the most open (30%, $47.60 an hour).

### 3.5 Moving across and up within tech

- **2,049 people moved between tech roles** after 12+ months in the first. The strongest paths: IT support → networks and systems (279), networks and systems → IT support (199), IT support → security (163), developers → databases (142) and → data science (137).
- **Most moves are sideways:** 79% kept the same level, 8.5% moved up, 12.5% moved down (levels read from title words, so rough). "Moving up" needs more evidence before it leads the pitch.
- **IT project management is a later move, not a first step:** 2% of its listings are entry level and 9% show openness, and 190 of its 264 career changers came from another tech role. Business backgrounds also move there (17% of retrained business people). **Decided 2026-10-10:** not suggested as a first tech job unless the user's prior career has a real path into it; shown to users already in tech as a step up; scope-wide openness figures are also shown without it.
- **To try:** a "trained, then moved up in the same role" pool (for example a master's before a senior role). Profiles' rules don't capture it yet; it would reopen Profiles.

### 3.6 The Ideas tab

A separate tab built on the compass: a personal outlook from the user's résumé. **Ideas to try, not decided. Time frames stay vague** ("down the line", "in the coming years") until the team picks them. Every number is labelled an estimate, worded as "people like you" and "expected", never a promise, and the résumé stays local.

| Idea | What it says | Where the numbers come from |
|:--|:--|:--|
| **Skills to build** | "Based on your current résumé, we think you should build these skills" | Listing skills mapped to O*NET (EDA_SDA_3) vs. the résumé (`ML_Fit`) |
| **Career options down the line** | "Here's what your career options could look like" | Roles similar people reached and how long it took (Profiles), with each role's 2026 openings |
| **Pay increase** | "Here's your pay increase" | Hourly pay in each option (EDA_SDA_2 3.12) vs. the user's current field (EDA_SDA_1 7.2, the occupation summary), or a rate they enter |
| **Job growth** | "Here's the job growth we expect in your career" (within 5 years) | Official projections, next to the role's entry-level share |
| **Stepping-stone roles** | Roles that bridge two functions on the way to a bigger move (for example engineer → pre-sales engineer → director) | Levels at each move (Profiles) |

### 3.7 How it connects to the rest of the app

- **Picking a role in the compass sets the priority queue's occupation filter;** every other filter (location, work mode, schedule, job type, role level) combines with it.
- **The date picker** shows the queue as of a chosen date (fair from July 1, for example July vs. September).
- **Hide, never remove:** listings out of reach, likely ghosts (over 90 days old and from a company gone quiet) or in an occupation that isn't a tech path (document management or marketing specialists) are hidden with a count and a reason, and a "show hidden" toggle brings them back. Other listings over 90 days old stay visible, labelled "open 90+ days" and ranked lower.

### 3.8 Honesty rules

*The general rules (no causation, hires, success rates, entry-level growth or certain futures) are in [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md), "What we never claim". Compass-specific:*
- **Job growth sits next to the role's entry-level share,** and carries its accuracy (moderate for years 1–2, low to moderate for 3–5; [section 3.3](#33-expected-job-growth)); IT project management's figure is a proxy from the 15-1299 catch-all.
- **Show the sample size on every path.** Most paths are thin (78% of prior-career paths rest on fewer than 10 people), so the compass falls back to the bucket-level pattern when a user's exact path is thin.

### 3.9 Still to build, and where

| Step | Notebook | Status |
|:--|:--|:--|
| Skills mapped to O*NET | `EDA_SDA_3.ipynb` | Planned |
| Suggested roles and skills to build per user | `ML_Fit.ipynb` | Planned |
| **Assembly** into the compass and Ideas tab tables for the app | `ML_Fit.ipynb`'s last section (a separate `Compass.ipynb` is proposed) | Planned |
| The app views | Section 2: Retraining compass, Moving across and up, Ideas tab | Designed |

### 3.10 Open decisions

- **Time frames** for career options and job growth (team), **with growth capped at 5 years**.
- **A separate `Compass.ipynb`** that assembles everything that doesn't need ML now, with `ML_Fit` adding personal suggestions later, instead of assembling at the end of `ML_Fit`.
- **The "moved up in the same role" pool** (would reopen Profiles).
- **Job growth extensions** (proposed, [section 3.3](#proposed-extensions-not-decided)): an error band from BLS's own track record, interest-rate scenarios for years 1–2, listing momentum shown next to growth, a full horizon-weighted blend, and state-level projections. Each would reopen part of the 2026-10-09 external-data decision.

---

## 4. The priority model

The priority score ranks postings by how much an application is worth. It combines:

| Factor | Meaning | Source |
|---|---|---|
| **Chance still open** | Probability the posting is still open today | Survival model |
| **Apply-by date (estimated)** | The date by which the posting is likely to have closed | `first_seen_at` + the predicted window for postings like it. Postings rarely list a real deadline, so this is always labelled an estimate. |
| **Career-change accessibility** | How open the posting is to someone new to the field | Seniority and listing text: years required, degree or equivalent experience, training, career-changer wording |
| **Transferable fit** | How well the skills from the user's old field match the posting | `ML_Fit`, starting from the app's TF-IDF + BGE ranker (RRF; decision 4): TF-IDF, BGE (local) and O*NET skills, combined by a model trained on real career moves |
| **Competition** | Lower is better | Reposts, company hiring volume, similar open postings, external market tightness |
| **Pay** | Hourly pay rate, where disclosed (salaried pay converted at 2,080 hours a year) | Cleaned salary data (`salary_hourly_usd`), compared with listings in the same role |
| **Freshness** | Days since first seen | `first_seen_at` (the crawler finds listings a median of 1.5 days after posting) |
| **Hiring difficulty** | Long-open or reposted listings | Listing age and reposts. Likely ghosts are hidden; other listings open 90+ days are labelled and ranked lower ("possibly hard to fill" was dropped, EDA_SDA_2 3.1). Reposts: 6.1% of the scope; 7.9% of listings are relisted copies of a closed job, closing in a median 7.9 days |
| **Employer intent** | What the wording signals | Listing text: urgency, remote, description length, evergreen wording |
| **Source** | Where the listing is posted | Hiring platform, job board, or company site; whether the listing is on the employer's own domain |
| **Company type** | Changes how fast high-volume employers close (EDA_SDA_2 3.8) | Publicly traded, has a parent company (companies table) |
| **Occupation group** | Also used to show related jobs | O*NET occupation group |
| **Expected job growth** | The same for every listing in a role, so it shifts role recommendations rather than listings within a role | BLS projections ([section 3.3](#33-expected-job-growth)), as a visible, adjustable weight |

*Not used:* **urgency from cross-posting** (how many boards carry the same job at once): only 13 cross-posted listings in the scope (EDA_SDA_2 3.5). Competition splits reposts into relisted after closing vs. cross-posted on several boards at once.

**How it's tuned:** the analysis decides which factors carry real signal and how strongly. The weights start from those findings and are checked against a newest-first baseline on a later time period. Users can adjust the weights with sliders, for example "reachability matters more to me than pay."

Every score shows its parts, so a user can see why a posting ranks where it does: the detail view breaks the score down by factor, and the model-based factors by their top features (section 2, detail view).

---

## 5. Multi-agent MCP design

### Why it fits

Each MCP server wraps one capability we build anyway: postings, timing, fit, market data, the user's tracker. The agents then do real jobs a job seeker would otherwise do by hand. This has four benefits:
- **Data boundaries:** each server enforces what leaves it, so each outside-AI call sends only what its feature needs, with the user's consent.
- **Natural-language control:** "show remote analyst roles closing this week with fit above 70%" becomes filters on the list view.
- **Background work:** agents watch for new postings and approaching deadlines while the user is away.
- **Swappable parts:** a better fit model or a new data source plugs in without changing the app.

### MCP servers

| Server | Tools (examples) | Data |
|---|---|---|
| `postings` | Search and filter live postings; get one posting's features | Our cleaned postings, stored locally |
| `timing` | Chance still open; apply-by date; check schedule | Our survival and arrival models |
| `fit` | Suggest roles from a background profile; score it against postings; explain which skills transfer; list gaps | `ML_Fit`'s model: local TF-IDF and BGE, O*NET skills |
| `priority` | Score and rank postings with the tuned model and the user's weights | Outputs of `timing`, `fit` and competition signals |
| `market` | Job growth expectations by role (with ranges and scenarios if adopted); listing volume by month | Job growth expectations; our listings |
| `tracker` | Save, set status, list saved postings, reminders | The user's own data |

### Agents

| Agent | Job | Servers it uses |
|---|---|---|
| **Background** | Reads the career changer's profile and extracts past roles, skills and target fields | `fit` |
| **Scout** | Follows the check schedule and finds new postings that match the user | `postings`, `timing` |
| **Ranker** | Scores and ranks postings into the priority queue | `priority`, `timing`, `fit` |
| **Coach** | Explains the queue, the retraining compass and the strategy in plain language; answers questions about which path to take; turns questions into filters ("what should I apply to this week?") | All, read-only |
| **Bridge** | Compares the career changer's profile with a posting's requirements, and suggests what to emphasize from their old field and what to learn | `fit` |
| **Reminder** | Flags saved postings whose apply-by date is close, and new high-priority matches | `tracker`, `timing` |
| **Orchestrator** | Decides which agent runs when, and combines their answers | All |

### Data rule for the agents

OpenAI use is allowed (confirmed 2026-10-11). The rules the app already follows carry over to the agents:
- Every outside-AI feature is **off by default** and asks for the user's **consent each time**; résumé text has emails, phone numbers and links removed before sending.
- Answers come **only from our data**, citing what was used (as Ask the market and the gap analysis do).
- A local model (for example through Ollama) remains an option for anything that doesn't need an outside service.

### Bonus: works with any MCP client

Because the tools are standard MCP servers, they can also be used outside our app, for example from Claude Desktop. A user could ask their own assistant "what are my top 5 postings this week?" That's a strong demo of the multi-agent design.

---

## 6. External data (candidates)

All public. Each would need its terms of use checked.

| Dataset | What it adds | Used in |
|---|---|---|
| Job growth expectations (BLS Employment Projections 2025–2035) | Expected employment growth by role bucket over at most 5 years; sources, method and extensions in [section 3.3](#33-expected-job-growth). Joined only after PredictLeads' written consent (agreement 4.3) | Role discovery, retraining compass, market panel. Joins directly on occupation code |
| O*NET skills, abilities and related occupations | The skills each occupation needs, and which occupations are close | Transferable fit; suggesting target fields from a background; skills to build (compass) |

---

## 7. Build in stages

Each stage is usable on its own, so we always have something to demo. Only what's left in each stage is listed.

| Stage | Still to build | Depends on |
|---|---|---|
| **1. MVP** | Date picker; sort and the full filter set; unreachable listings hidden, with what each listing asks of a newcomer; apply-by, chance still open, pay and freshness on every listing | Still open / apply-by and the EDA's chosen factors |
| **2. Transferable fit and role discovery** | Suggested roles; retraining compass; Ideas tab; moving across and up within tech; job growth expectations; the transferable-fit score and filter; skills that carry over, and gaps | `ML_Fit`; O*NET skills; job growth expectations |
| **3. Priority model** | Tuned priority score with adjustable weights; the rating breakdown by feature in the detail view | Full analysis; `ML_Rank`'s per-listing contributions |
| **4. Agents** | MCP servers, then the agents, starting with Ranker, Coach and Reminder | Stages 1–3 |
| **5. Market panel** | Listing volume by month, with each role's expected growth; if the team adopts the scenario extensions ([section 3.3](#33-expected-job-growth), options b and d), a fan chart per role (baseline, scenario band, uncertainty band) | Stage 1 |

---

## 8. Logistics still open

| Topic | Draft | Status |
|---|---|---|
| MCP servers | Built with the official `mcp` Python SDK, beside the FastAPI backend | Planned |
| Hosting | Today a local demo; does it need hosting? | TBD |
| Timeline | Target about 2026-10-23 | Target |
| Owner | | TBD |

---

## 9. Open questions

- **Course requirements:** what's required for the web app and the multi-agent part (rubric, must-haves), or is it open-ended?
- **Agent ambition:** are agents the centerpiece of the app, or a layer on top of a solid queue? This decides how much of week 3 goes to them.
- **Team split:** who is best suited to own the app (front end, MCP servers), so the analysis isn't starved?
- **Queue fields:** is anything missing, such as a skills-gap column or company information?
- **Time frames:** how far ahead the compass looks for career options and job growth (not decided yet; growth at most 5 years).
- **Growth as a range:** show expected growth as one number or with an error band from BLS's past accuracy ([section 3.3](#33-expected-job-growth), option a)?
- **Recruiter details:** may the app show recruiter names and contact details from the listings? (to ask the advisors)
- **External data:** which datasets are allowed, and do we need permission to combine them with PredictLeads data?
- **Hosting:** does the app need to be hosted, or is a local demo enough?
- **App ranker:** does RRF really beat TF-IDF for this audience? The demo uses RRF by default; no relevance result exists (`STATUS.md`).
- **Listing activity in the app:** decided 2026-10-09 to represent it as an observation window (`first_seen_at` to `last_seen_at`, median 24 days across the app's corpus), kept as raw and parsed values; no feature built yet. A "recently observed" feature can be added later.
- **Questions for PredictLeads:** crawl frequency and date precision; what `status` and `last_processed_at` mean; the time zone of posting dates; whether older daily exports exist; how duplicates and reposts are handled; how `seniority` is assigned; whether real listings can appear in our final presentation; **why job-board listings stop after May 2026** (494 and 526 in April and May, then almost none; EDA_SDA_2 3.2c); and **why more listings have no description from June on** (about 30%, vs. 2–10% in March–April; EDA_SDA_2 3.9).

---

## 10. Ideas from the app's roadmap (not decided)

From `app/roadmap.md` (2026-10-10), ranked on value to the demo and to PredictLeads, effort, whether we can check it against data, and fit with the rule of **no inferred claims** (anything shown should be a count, a quote, or a clearly labelled model measurement; [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md#what-we-never-claim)). Some overlap the analysis plan: idea 3 is the apply-by estimate (section 4), and idea 2 is close to "skills to build" (section 3.6).

| # | Idea | Value | Effort | Can we verify it? | Fits "no inference"? |
|---|---|---|---|---|---|
| 1 | **Near-duplicate / repost detection** (MinHash or embedding similarity on listing text) | High for PredictLeads: raises the accuracy of every count on Signals; data-quality story | About 1 day | Yes: sample matches by hand, and see how counts change | Yes: it removes inflation, adds no claims |
| 2 | **Skills across top jobs** (run the Compass requirement analysis over a user's top Discover results, show the most frequent requirements and the verbatim profile evidence) | High for users: the core "what should I learn" answer; reuses existing Compass code | About half a day | Partly: same heuristics as the Compass | Yes: shows frequencies and the person's own words |
| 3 | **Listing age and lifetime** (first descriptive: "observed for N days"; later a survival model for how long listings stay up) | Medium-high for PredictLeads; separates fresh from stale listings | Descriptive: a few hours. Model: 1-2 days | Yes: temporal holdout | Descriptive yes. A model prediction is inference: show it only as a labelled estimate, if at all |
| 4 | **Hiring-pattern clustering** (discover burst, steady, seasonal shapes from weekly curves) | Medium; improves Signals' credibility over hand-set thresholds | About 1 day | Yes: compare with the current rules | Yes: descriptive |
| 5 | **"Refine my matches"** (per-user model from 👍/👎; design below) | Medium-high for demo impact; evidence is thin without ratings | About half a day | Only with an opt-in rating export | Mostly: it is personalised ranking, labelled as such |
| 6 | **Profile what-if** (how would adding one skill change the ranking) | Low-medium | 1 day | Hard | Borderline: speculative |
| 7 | **Salary estimator** (fill missing pay with a regression) | Medium for users | 1-2 days | Yes: error ranges | **No**: it fills a blank with an estimate. Only acceptable if clearly labelled "estimate"; we recommend skipping |
| 8 | **Grounded resume bullets** (GenAI) | Medium | 1 day + consent and budget | Hard | Risky: generated text, needs strict grounding |

**Suggested order if time allows:** measuring the ranker (`PRIORITY_QUEUE.md`), then ideas 2 and 1, then 3 (descriptive part), then 5. Skip 7 and 6 for the demo.

**"Refine my matches": design notes, decision pending (2026-10-09)**
- After about 20-30 👍/👎 ratings, a button trains a small model on that user's ratings and re-ranks their results. Today's feedback is only a live Rocchio nudge.
- Design: unlock at about 10 ratings with at least 5 of each kind; heavily regularized logistic regression or ridge on BGE vectors of rated jobs; cross-validate on the user's ratings and refuse to apply the model if it does not beat chance; fuse with the RRF rank; show "refined with N 👍 and M 👎, predicted your ratings with X% accuracy, K of your top 10 changed"; keep some variety.
- Prerequisite: the rating log, built 2026-10-10 ([`app/docs/feedback-data.md`](../app/docs/feedback-data.md)). Production needs users to opt in (the switch exists). Risks: small samples, selection bias, ambiguous 👎, no accounts, consent for storing a taste model.

**Not planned:** Career Adjacency graph (needs validated duties and O*NET assignments); market momentum across exports (only one export); PDF OCR; accounts and persistent profiles; deployment hardening.
