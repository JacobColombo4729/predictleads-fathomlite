# Project Outline

**PredictLeads × TripleTen externship** · Jacob Colombo · Updated 2026-10-09
*A working proposal; the team's final focus is still being agreed.*
*How we work: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) · Decisions: [`docs/decisions.md`](docs/decisions.md) · Data: [`docs/data.md`](docs/data.md)*

---

## The idea

**Our applicant is a career changer:** anyone switching from their current background to a different role or field. They see far more listings than they can apply to, and most of them ask for experience in the new role that they don't have yet. For a given career changer, we're building a tool that does two things:
- **Role discovery:** suggests which computer and mathematical roles their skills transfer to.
- **Prioritization:** ranks the listings in the roles they choose: which to apply to first, and why.

The ranking weighs two questions for every listing:
- **Can I realistically get it?** Career-change accessibility: entry-level seniority, years of experience required, a degree *or equivalent experience*, training offered, career-changer wording; later, **transferable fit** (which skills carry over from the applicant's old field). Listings with hard blockers (manager level and above, too many years asked, a degree or license the applicant lacks) are **hidden**.
- **Is it worth applying now?** How long listings like it stay open, competition, pay and source. The model learns how much each factor counts.

We can't see applications or hires, so the ranking rests on what PredictLeads' data *does* show: how long listings stay open, who posts them, where, and what they say, including what they ask of a newcomer.

**The end product:** a web app with a **priority queue** of listings for a career changer. Each listing shows how open it is to someone new to the field, its estimated apply-by date, why it ranks where it does, and a link to apply. **60.7% of listings live on employers' own sites** (or other domains that aren't hiring platforms or job boards), so most ranked listings can link straight to the employer's posting page.

## Data

- **Source:** PredictLeads job openings and companies datasets, snapshot of 2026-09-30.
- **Analyzed so far:** 200,000 job listings from 51,006 companies, first seen 2026-03-30 to 2026-09-30.
- **Analysis table after cleaning and scoping:** 171,143 English-language US listings from 41,950 companies, April–September 2026; 39,596 still open at the snapshot.
- **Data handling:** stays local; nothing is shared with outside AI services; only aggregates are reported.

## What we've found so far

| Finding | Evidence | What it means for a career changer |
|:--|:--|:--|
| **Company hiring volume is the strongest signal** | Listings at companies with 20+ openings close in a median **14 days**, vs. **32** for one-off listings | Listings from high-volume employers need fast action |
| **The application window is short** | **28%** of listings are gone within a week; the median within about **22 days** | "Still open / apply-by" is a core factor |
| **Source matters** | Company sites and LinkedIn: **17–19 days**; several hiring platforms: **28–37 days** | Source adjusts urgency |
| **Text carries signal** | Urgency wording, long descriptions, remote mentions and disclosed pay go with longer windows (5–16 points at 7 days) | Worth modeling with NLP |
| **Factors differ by field** | Remote **2–25%**, reposts **0.9–9%**, median pay **$38k–$137.5k** by occupation group | Build the model for one target field |
| **The best-paid field is the hardest to enter** | Computer and mathematical: **9%** junior, **77%** mid/senior; sales and food preparation are junior-heavy but pay less | We target it anyway (pay, text, the team's backgrounds) and hide what a career changer can't get |
| **How open listings are to career changers** | *to fill in after rerunning EDA_SDA_1 Q1g* | Becomes its own ranking factor |

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
| Transferable fit | Listing text vs. the career changer's profile, with O*NET skills connecting the old and new occupations (needs applicant profiles, so it comes later) |

## How we're doing it: six notebooks

Each notebook answers one question and hands its final table to the next. The TripleTen course templates are the baseline for structure and methods, taken further with advanced techniques.

| # | Notebook | Question | Key methods | Status |
|:-:|:--|:--|:--|:--|
| 1 | **EDA_SDA_1** | What's in the data, and what's the scope? | Cleaning with an audit trail, survival curves, factor comparisons, scope decisions defended with evidence; career-change accessibility (Q1g) | **Done; rerun needed** for the new career-change flags |
| 2 | **EDA_SDA_2** | Which listings do we study? | Ghost listings vs. desperate employers; checks on source, freshness, pay, reposts, company volume; companies-table join; missing-data patterns; **narrowing computer and mathematical to what a career changer can get**; pairing it with the team's backgrounds | Next |
| 3 | **EDA_SDA_3** | What features does the text add? | Text normalization, lemmatization, TF-IDF; evergreen, urgency and hiring-difficulty phrases; career-change flags validated (experience, education, licenses, training); transferable skills | Planned |
| 4 | **EDA_SDA_4** | Which features hold up? | Pearson + Spearman correlation, multicollinearity (VIF), hypothesis tests with effect sizes, confounding and company-clustering checks, competing risks | Planned |
| 5 | **TimeSeries** | When do listings appear and close? | Decomposition, stationarity tests, lag features, ARIMA / SARIMA, survival by month | Planned |
| 6 | **ML** | Which roles fit a career changer, and how should listings be ranked? | Baselines; logistic regression, random forest, LightGBM, CatBoost, XGBoost; survival and learning-to-rank models; ROC-AUC, C-index, NDCG; SHAP explanations | Planned |

## Key decisions so far

- **Focus: career changers** (part of the project description): anyone switching from their current background to a different role or field. Every factor, segment and model choice is judged by what it means for someone switching fields.
- **Career-change accessibility is its own factor,** read from seniority and listing text. It's known when a listing appears, so it's safe to model.
- **Scope:** English-language listings in the United States, first seen April–September 2026 (only six months of history, so no seasonal claims).
- **No early filtering:** data-quality problems are fixed during cleaning; scope filters are applied only at the end, after each is defended.
- **Long-open listings are kept:** they may be ghost listings *or* employers struggling to hire. Part 2 tells them apart instead of assuming.
- **Reposts are a feature, not noise:** a job posted again and again may be hard to fill.
- **Two features:** role discovery (which computer and mathematical roles a career changer's skills transfer to) and prioritization (ranking listings in the chosen roles).
- **Target: computer and mathematical (US),** for its pay, remote share, rich text and the team's backgrounds. Only 9% of its listings are junior, so listings a career changer can't get are hidden. EDA_SDA_2 picks the niche inside it.
- **Source backgrounds: the team's own.**
- **Hide what a career changer can't get:** manager level and above (EDA_SDA_2), validated text blockers (EDA_SDA_3), and per-applicant blockers such as a missing degree or license (in the app). Every other factor is weighted by the model during ML training.
- **Transferable fit uses a trained model (Option B)** on 50 real career changers and 1,000 synthetic profiles. See [`docs/fit_analysis.md`](docs/fit_analysis.md).

## Honest limits

- We see listings appear and disappear, **not hires**. A listing closing could mean filled, cancelled or moved.
- `first_seen_at` reflects PredictLeads' crawl timing, so day-of-week patterns rely on posting dates (about 45% of listings).
- Six months of data rules out seasonal advice.
- We can't see who gets hired, so we can't tell whether employers actually hire career changers. Accessibility is read from what listings *ask for*.
- The career-change flags are keyword rules until EDA_SDA_3 validates them. Hiding listings on an unvalidated rule could hide listings a career changer could get, so rules hide nothing until they're validated.
- No applicant profiles yet: transferable fit needs career-changer profiles, collected with consent and extended with synthetic ones.

## Questions for PredictLeads

Crawl frequency and date precision; what `status` and `last_processed_at` mean; the time zone of posting dates; whether older snapshots exist; how duplicates and reposts are handled; how `seniority` is assigned; and whether real listings can appear in our final presentation.
