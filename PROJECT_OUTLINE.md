# Project Outline

**PredictLeads × TripleTen externship** · Jacob Colombo · Updated 2026-10-09
*A working proposal; the team's final focus is still being agreed.*
*How we work: [`HOW_WE_WORK.md`](HOW_WE_WORK.md) · Decisions: [`docs/decisions.md`](docs/decisions.md) · Data: [`docs/data.md`](docs/data.md)*

---

## The idea

**Our applicant is a career changer:** anyone switching from their current background to a different role or field. They see far more listings than they can apply to, and most of them ask for experience in the new role that they don't have yet. > **Pitch:** Everyone tells career changers what they match. We tell them what's realistically reachable, which training actually got people like them there, and how fast they have to move, using real career switches and live 2026 listings.

For a given career changer, we're building a tool that does two things:
- **Role discovery:** suggests which computer and mathematical roles their skills transfer to.
- **Prioritization:** ranks the listings in the roles they choose: which to apply to first, and why.

The ranking weighs two questions for every listing:
- **Can I realistically get it?** Career-change accessibility: entry-level seniority, years of experience required, a degree *or equivalent experience*, training offered, career-changer wording; later, **transferable fit** (which skills carry over from the applicant's old field). Listings with hard blockers (too many years asked, a degree or license the applicant lacks, manager-level roles for applicants without management experience) are **hidden**.
- **Is it worth applying now?** How long listings like it stay open, competition, pay and source. The model learns how much each factor counts.

We can't see applications or hires, so the ranking rests on what PredictLeads' data *does* show: how long listings stay open, who posts them, where, and what they say, including what they ask of a newcomer.

**The end product:** a web app with a **priority queue** of listings for a career changer. Each listing shows how open it is to someone new to the field, its estimated apply-by date, why it ranks where it does, and a link to apply. **Up to 60.7% of listings live on employers' own sites** (the share also counts other domains that aren't hiring platforms or job boards), so most ranked listings can link straight to the employer's posting page.

## What makes this different

- **Paths real people took:** which training led people like the user into which tech roles, learned from real career switches.
- **A retraining compass:** training → the roles it led to → each role's 2026 openings, pay and closing speed.
- **How fast to move:** an apply-by estimate on every listing; and listings a career changer can't get are hidden.

Details, evidence and pitch wording: [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md).

## Data

- **Source:** PredictLeads job openings and companies datasets, snapshot of 2026-09-30.
- **Analyzed so far:** 200,000 job listings from 51,006 companies, first seen 2026-03-30 to 2026-09-30.
- **Analysis table after cleaning and scoping:** 182,367 English-language listings from 46,810 companies, April–September 2026 (US, plus flagged listings with no country); 41,825 still open at the snapshot.
- **Career switches:** the public Kaggle structured resume dataset (MIT license): 54,933 resume records → 17,952 unique people (each resume appeared about 3 times; 99.7% with readable job dates) → **2,539 career changers** under rules version 8: 1,836 changed roles within tech, 364 retrained after working in another field, 339 retrained as new grads from another field. Career paths from real switches spanning two decades; the 2026 listings supply today's openings, pay and timing. Built in `notebooks/Profiles.ipynb`.
- **Data handling:** stays local; nothing is shared with outside AI services; only aggregates are reported.

## What we've found so far

| Finding | Evidence | What it means for a career changer |
|:--|:--|:--|
| **Company hiring volume is the strongest signal** | Listings at companies with 20+ openings close in a median **14 days**, vs. **32** for one-off listings | Listings from high-volume employers need fast action |
| **The application window is short** | **28%** of listings are gone within a week; the median within about **22 days** | "Still open / apply-by" is a core factor |
| **Source matters** | Company sites and LinkedIn: **17–19 days**; several hiring platforms: **28–37 days** | Source adjusts urgency |
| **Some listing traits carry signal; keyword flags don't** | Disclosed pay, long descriptions and hiring platforms: 11–16 points more still open after a week; urgency and remote wording: no gap | Model the traits; test richer NLP |
| **Factors differ by field** | Remote **2–35%**, reposts **0.9–9%**, median pay **$38k–$137.5k** by occupation group | Build the model for one target field |
| **The best-paid field is the hardest to enter** | Computer and mathematical: **9%** junior, **77%** mid/senior; sales and food preparation are junior-heavy but pay less | We target it anyway (pay, text, the team's backgrounds) and hide what a career changer can't get |
| **How open listings are to career changers** | Entry level ranges from **3%** (management) to **70%** (food preparation); **22%** of computer and mathematical listings show any sign of openness | Becomes its own ranking factor |
| **Real career switches exist in public resumes** | 54,933 resume records → 17,952 unique people (each resume appeared about 3 times; 99.7% with readable job dates) → **2,539 career changers** under rules version 8: 1,836 changed roles within tech, 364 retrained after working in another field, 339 retrained as new grads from another field | Training data for role discovery and fit. Tighter rules (version 5) and a spot-check are in progress |

The listing itself matters more than the job title: two listings in the same field can differ more than two different fields do.

## Prioritization factors

| Factor | Measured from |
|:--|:--|
| Career-change accessibility | Hard blockers hide a listing; softer signals raise it. Seniority (entry level); listing text: years of experience required, degree required, "or equivalent experience", entry-level and training wording, career-changer wording |
| Still open / apply-by | First-seen and last-seen dates (survival analysis) |
| Freshness | Days since first seen (the crawler finds listings a median of 1.5 days after posting) |
| Competition | Company hiring volume; reposts (times listed, days since last listed) |
| Hiring difficulty | Listing age, repeated reposts; ghost listing vs. employer still hiring |
| Pay | Annualized salary (disclosed for about 47% of listings) |
| Employer intent | Listing text: urgency, remote, description length, evergreen wording |
| Source | Hiring platform, job board, or company site |
| Occupation group | O*NET occupation group; also for showing related jobs |
| Transferable fit | Listing text vs. the career changer's profile, with O*NET skills connecting the old and new occupations (learned from real career switches in `Profiles.ipynb`; volunteer profiles later as a recent test set) |

## How we're doing it: six notebooks

Each notebook answers one question and hands its final table to the next. The TripleTen course templates are the baseline for structure and methods, taken further with advanced techniques.

| # | Notebook | Question | Key methods | Status |
|:-:|:--|:--|:--|:--|
| 1 | **EDA_SDA_1** | What's in the data, and what's the scope? | Cleaning with an audit trail, survival curves, factor comparisons, scope decisions defended with evidence; career-change accessibility (Q1g) | **Done** |
| 2 | **EDA_SDA_2** | Which listings do we study? | Ghost listings vs. desperate employers; checks on source, freshness, pay, reposts, company volume; companies-table join; missing-data patterns; **narrowing computer and mathematical to what a career changer can get**; pairing it with the team's backgrounds | Next |
| 3 | **EDA_SDA_3** | What features does the text add? | Text normalization, lemmatization, TF-IDF; evergreen, urgency and hiring-difficulty phrases; career-change flags validated (experience, education, licenses, training); transferable skills | Planned |
| 4 | **EDA_SDA_4** | Which features hold up? | Pearson + Spearman correlation, multicollinearity (VIF), hypothesis tests with effect sizes, confounding and company-clustering checks, competing risks | Planned |
| 5 | **TimeSeries** | When do listings appear and close? | Decomposition, stationarity tests, lag features, ARIMA / SARIMA, survival by month | Planned |
| 6 | **ML** | Which roles fit a career changer, and how should listings be ranked? | Baselines; logistic regression, random forest, LightGBM, CatBoost, XGBoost; survival and learning-to-rank models; ROC-AUC, C-index, NDCG; SHAP explanations | Planned |
| side | **Profiles** | Who really retrained into tech or math? | Public resume dataset; duplicate removal; whole-timeline rules (`src/career_switch.py`); random spot-checks | **Built**; rules v6 (labels everyone, career changers are a view); spot-check in progress |

## Key decisions so far

- **Focus: career changers** (part of the project description): anyone switching from their current background to a different role or field. Every factor, segment and model choice is judged by what it means for someone switching fields.
- **Career-change accessibility is its own factor,** read from seniority and listing text. It's known when a listing appears, so it's safe to model.
- **Scope:** English-language listings in the United States, first seen April–September 2026 (only six months of history, so no seasonal claims).
- **No early filtering:** data-quality problems are fixed during cleaning; scope filters are applied only at the end, after each is defended.
- **Long-open listings are kept:** they may be ghost listings *or* employers struggling to hire. Part 2 tells them apart instead of assuming.
- **Reposts are a feature, not noise:** a job posted again and again may be hard to fill.
- **Two features:** role discovery (which computer and mathematical roles a career changer's skills transfer to) and prioritization (ranking listings in the chosen roles).
- **Target: computer and mathematical (US),** for its pay, remote share, rich text and the team's backgrounds. Only 9% of its listings are junior, so listings a career changer can't get are hidden. EDA_SDA_2 splits it into role buckets (data and analytics, IT support, software development, …) aligned with the career-switch profiles, and drops only buckets too thin on both sides.
- **Source backgrounds: the team's own.**
- **Hide what a career changer can't get:** validated text blockers (EDA_SDA_3), and per-applicant blockers in the app: a missing degree or license, or manager-level roles for applicants without management experience. Every other factor is weighted by the model during ML training.
- **Two supervised models, combined by a formula:** listings → how long they stay open (PredictLeads), and career changers → the role bucket they moved into (the 2,539 career changers). No data links a person to a listing, so the role buckets connect the two. Volunteer profiles, once the advisor approves, are a recent test set. See [`docs/fit_analysis.md`](docs/fit_analysis.md).

## Honest limits

- We see listings appear and disappear, **not hires**. A listing closing could mean filled, cancelled or moved.
- `first_seen_at` reflects PredictLeads' crawl timing, so day-of-week patterns rely on posting dates (about 45% of listings).
- Six months of data rules out seasonal advice.
- We can't see who gets hired, so we can't tell whether employers actually hire career changers. Accessibility is read from what listings *ask for*.
- The career-change flags are keyword rules until EDA_SDA_3 validates them. Hiding listings on an unvalidated rule could hide listings a career changer could get, so rules hide nothing until they're validated.
- The career-switch profiles aren't reviewed yet (a spot-check is in progress), and most switches are from 2008–2016, so the resumes teach career paths while the 2026 listings supply the market. O*NET's current Career Changers Matrix checks that the paths still hold.

## Questions for PredictLeads

Crawl frequency and date precision; what `status` and `last_processed_at` mean; the time zone of posting dates; whether older snapshots exist; how duplicates and reposts are handled; how `seniority` is assigned; and whether real listings can appear in our final presentation.
