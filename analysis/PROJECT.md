# Project

**For career changers moving into tech:** which roles your background transfers to, which listings you can realistically get, and which to apply to first. Built on PredictLeads' 2026 job listings and 2,813 real career switches from public résumés. PredictLeads × TripleTen externship.

**Start with [`SUMMARY.md`](SUMMARY.md)**: the three numbers, the one path (résumé → role fit → candidate pool → queue) and how we know it works, on one page. This outline is the detail behind it. What's left to do: [`ANALYSIS.md`](ANALYSIS.md). How it stands out and how to pitch it: [`../DIFFERENTIATORS.md`](../DIFFERENTIATORS.md). The app: [`../app/DESIGN.md`](../app/DESIGN.md).

## Data

- **Listings:** PredictLeads' job openings and companies, the export of 2026-09-30. The full file has 13.4M listings (13.0M English and US, first seen 2026-03-30 to 2026-09-30).
- **The scope (EDA_SDA_1):** 908,291 listings from 72K companies in 11 role buckets: the roles TripleTen trains for (computer and mathematical roles, sales engineers, business and market research analysts; 58% of the scope) plus adjacent tech roles (IT support, networks and systems, database, IT project management; 42%). US, English. Each listing carries its TripleTen program.
- **The app serves** the 193,043 scope listings still open when the data ends, with a usable description and no duplicates.
- **Career switches (Profiles):** a public Kaggle résumé dataset (MIT license): 54,933 records → 17,952 people → **2,813 career changers** (2,049 moved within tech, 407 retrained after working in another field, 357 retrained as new grads). The résumés supply career paths; the 2026 listings supply openings, pay and timing.
- **O*NET 31.0** (CC BY 4.0): skills and related occupations, joined on each listing's occupation code.
- **Handling:** the data stays local and under our data agreement; only aggregates are reported.

## What we found

| Finding | Evidence | What it means |
|:--|:--|:--|
| **The window is short** | 29% of scope listings are gone within a week; median 24 days; technical sales is slowest (38) | "Still open" is a core factor |
| **Company hiring volume is the strongest timing signal** | Listings at companies with 500+ earlier listings close in a median 12 days, against 44 at companies with 1–4 (EDA_SDA_1 Q1d); about 1.5× like-for-like (EDA_SDA_2 3.3), mostly at private companies (3.8) | High-volume employers need fast action |
| **Source matters** | Median days open: LinkedIn, Oracle and Taleo 15; company sites 21; Greenhouse, Lever and Ashby 49–71 | Source adjusts urgency |
| **The best-paid field is the hardest to enter** | Computer and mathematical is 1st of 23 fields on pay ($66.73 an hour) but 22nd on entry level (9.4%) and last on asking 2 years or fewer (17%) | Hide what a career changer can't get; show the rest honestly |
| **Entry doors differ by role** | IT support: 22% entry level, 44% open to career changers, $31 an hour; Data Analytics roles 26% open at $51.20; IT project management 2% entry level, mostly reached from another tech role (EDA_SDA_2 3.6, 3.12) | IT support is a first step; IT project management a later move |
| **Ghost listings can be told apart** | 82% of stale listings come from a company still posting, against 98% of younger ones; 7.5K likely ghosts (3.5% of open listings) are hidden (EDA_SDA_2 3.1) | Fewer dead ends |
| **Reachable listings pay less** | The hiding rules remove 15% of open scope listings; reachable ones pay $10–35 an hour less in every role (EDA_SDA_2 3.11) | The app shows the trade-off instead of hiding it |
| **Collection effects look like signals until checked** | 18% of listings have no description (23–25% from June) and close in a median 3 days, against 30; job-board listings stop after May (EDA_SDA_2 3.2, 3.9) | Missing pay looked predictive only because of them; timing results are shown with and without them |
| **Text match alone isn't enough** | The best text method (RRF) reaches 70.0% at top 3 against the baseline's 63.2%; skills matching alone is below the baseline (ML_Fit 6b) | Role fit has to be learned from real career moves |

**Tested, no usable signal:** evergreen wording doesn't line up with likely ghosts (EDA_SDA_3 4); cross-post urgency isn't consistent within source type (EDA_SDA_2 3.5c); keyword flags such as remote wording show no timing gap.

## What the models use

| Factor | From | Used as |
|:--|:--|:--|
| **Fit** | Profiles' features (prior career, training, timing), text similarity to each role (keyword, TF-IDF, BGE, RRF), O*NET skills that carry over | ML_Fit's trained layer; the queue's fit factor |
| **Still open** | Listing features known when it first appears: company volume (earlier listings only), source, platform, pay, description, text flags, repost history | XGBoost's chance still open 14 days later; the queue's second factor |
| **Accessibility** | Seniority hides manager-level listings for users without management experience; years asked, degree, "or equivalent experience", entry and career-changer wording are labels and opt-in filters | The queue's third factor |
| **Pay** | Standardized hourly pay (salaried ÷ 2,080 hours; disclosed for about 48%) | The queue's fourth factor (weight 0; a slider in the app) |
| **Hidden by default** | Likely ghosts, manager level, document management | Never ranked; shown with a count and reason |

Growth (BLS projections) would be a visible weight on role suggestions; it's at 0 until the projections are added.

## How we keep it honest

**Graded on real outcomes, with no labelling.** Each of the 2,813 career changers has the role they actually moved into. A method sees only their background before the switch and is scored on whether that role is in its top 3 and top 5. No AI grader, no human labels.
- **The same baseline for every method:** the most common roles among the training people.
- **Split by person:** development and test people never overlap; folds are grouped by person.
- **Tested on people and months it never saw ("held out"):** 563 of the career changers, and the last month of listings, were set aside before anything was built, then scored once at the end, so the scores show how the models do on new people, not how well they memorised the old ones.
- **95% intervals** from resampling people; results per pool (moved within tech, retrained after working, new grads).
- **Listings are tested forward in time:** trained on April–July, tuned on August, tested on September; folds grouped by company.

**No peeking at the future (leakage rules).**
- Features use only what's known when a listing first appears: its text, seniority, source, company, pay, `first_seen_at` and repost history from *earlier* copies. Never `last_seen_at`, `status`, days open, or whether it's reposted later.
- Counts over time (company volume, reposts) count only listings first seen before it.
- Still-open listings are kept and handled by survival methods (censored), never dropped: dropping them makes windows look shorter than they are.
- Career-switch counts used as features are built from training people only, so no test person counts toward their own answer.
- The test period is never used for tuning.

**Confounders compared within groups, not filtered out:** company volume, source and platform, staffing agencies, role bucket and seniority, month, state, company type and whether a listing has a description. A factor counts only if its gap holds within those groups; EDA_SDA_4 confirms the survivors in one Cox regression with company-clustered errors.

**Evidence only:** the models are built only from the scope's listings and Profiles' public résumés. The team's own résumés and backgrounds never enter training, tuning, evaluation or the analysis. Anyone may enter a résumé in the app, where the finished model only scores it; tests and reported results use fictional profiles (`app/code/ai_ml_candidate_profiles.json`). The team's ratings and app feedback are usability feedback only.

**Hide, never remove:** hidden listings stay in the analysis and the models, and the app shows how many each rule hides and why. Text rules are unvalidated keyword rules, so they're labels and opt-in filters, never hidden by default.

## Built once, runs on a laptop

No LLM is trained. BGE (`BAAI/bge-base-en-v1.5`, 0.44 GB) is a pretrained text-embedding model, downloaded at a pinned version and only used to turn text into vectors. What we train are tabular models (ML_Fit's trained layer, ML_Rank's listing models), about 0.05 GB each.

| Step | On the desktop (GPU) | On a laptop |
|---|---|---|
| Read the export into the listing tables (`src.stream_full`) | about 35 minutes (bulk mode: 16 GB RAM, 8 threads) | 3–5 hours (stream mode, for 8 GB machines) |
| Score all 2,813 career changers against all 8 roles (ML_Fit) | not timed | 6.3 minutes, BGE on the CPU included |
| Tune the trained layer (six candidate models) | not timed | 1.4 minutes |
| Encode the 193,043 serving listings with BGE | 58 minutes | not run; estimated several hours: the one step worth a GPU |
| Load the app's rankers, then rank one person | 3.3 minutes; 0.4–1.5 s per person | to time on Jesse's laptop (no GPU needed; about 2.5 GB of memory) |

**The same results on any machine:** every package version pinned (`requirements.txt`, Python 3.11); fixed seeds (people split with seed 0, models with 12345) and a run log of each run's settings and metrics (`src/run_log.py`); one fixed export (2026-09-30) with the scope's listing ids saved; the trained models saved once and checked by rescoring the test people from the saved files (largest probability gap 1.5e-06, the same top 3 roles for 100%); and the saved models and encoded listings shared as one checked zip (`python -m fetch.get_handoffs`). The PredictLeads data can't be handed over, so outsiders can rerun the code but not the data; the tests run on made-up data (`analysis/tests/`, `combined/tests/`, `app/tests/`).

## The notebooks

Each answers one question and hands its final table to the next (`data/clean/`). The TripleTen course templates are the baseline; shared code lives in `src/`.

| Notebook | Question | Product |
|:--|:--|:--|
| **EDA_SDA_1** | What's in the data, and what's the scope? | The scope table |
| **EDA_SDA_2** | Which scope listings do we keep, and how do they behave? | Cleaned and flagged scope table (ghosts, hiding rules) |
| **EDA_SDA_3** | What do the descriptions say? | Text features and O*NET skills |
| **EDA_SDA_4** | Which features hold up? | `modeling_ready.parquet` (the features that survived) |
| **TimeSeries** | When do listings appear and close? | Timing patterns by month |
| **Profiles** | Who really changed into or within tech, and how? (frozen) | The 2,813 career changers |
| **ML_Fit** | Which roles does a career changer fit? | The trained layer (role fit), saved for the app |
| **ML_Rank** | Which listings should they apply to first? | Listing scores and the queue's weights, saved for the app |
| **Evaluate** | How well do the app's rankers do on real career moves? | The comparison: `rrf`, `mlrank`, combined |

## Limits

- **We see listings open and close, not hires.** A listing closing could mean filled, cancelled or moved, so we say "closes", never "gets filled", and never "you'll get this job".
- **Accessibility is what listings ask for,** not who gets hired.
- **The résumés show people who switched,** not those who tried and didn't, so paths show where switchers went, never the odds a switch works. Most switches are from 2010–2019 (median 2014); the 2026 listings supply today's market.
- **Six months of listings:** no seasonal claims. Crawl timing shapes `first_seen_at`, so day-of-week patterns use posting dates (about 42% of listings).
- **Collection changes:** job-board listings stop after May 2026, and listings without a description rise from June; both are handled as confounders and asked of PredictLeads.
- **Text rules are unvalidated keyword rules** (the degree flag fires on only 3.7% of listings, so it likely misses common wording): labels and filters, never hidden by default.
- **Only 8 roles have a fit model:** Profiles has no moves into sales engineering or business and market research analysis.

## Questions for PredictLeads

Crawl frequency and date precision; what `status` and `last_processed_at` mean; the time zone of posting dates; whether older daily exports exist; how duplicates and reposts are handled; how `seniority` is assigned; whether real listings can appear in our final presentation; why job-board listings stop after May 2026 (about 30K and 33K in April and May, then about 200 a month; EDA_SDA_2 3.2c); and why more listings have no description from June on (23–25%, against 6–8% in March–April; EDA_SDA_2 3.9).
