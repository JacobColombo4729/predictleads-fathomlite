# Job Growth Expectations

PredictLeads × TripleTen externship · The plan for the compass's **expected job growth** column. Status as of 2026-10-10.
*Why it exists: [`COMPASS.md`](../COMPASS.md) 2b · Decisions: [`docs/decisions.md`](decisions.md) (2026-10-09, 2026-10-10) · Data reference: [`docs/data.md`](data.md)*

## 1. What's decided

- **One number per role bucket**, from official occupational projections, over **at most 5 years** (the exact horizon is the team's call).
- **Purpose:** recommendations based on future estimates. Among roles a person fits, growing roles rank higher.
- **A visible adjustment, not a trained input:** applied after the fit model in `ML_Fit`, with a user-adjustable weight. Historical career moves can't validate a future estimate.
- **Never alone:** always shown with the role's entry-level share and today's openings from our listings.
- **Projections, not our forecasts:** national, by occupation, published by BLS. We convert them; we don't make or check them.

## 2. Source

| Source | Use | Link |
|:--|:--|:--|
| **BLS Employment Projections 2025–2035** (released) | 2025 employment, 2035 projected employment and 10-year change by occupation code | [Home](https://www.bls.gov/emp/) · [Summary](https://www.bls.gov/news.release/ecopro.nr0.htm) · [Data tables](https://www.bls.gov/emp/data.htm) |
| Occupational Outlook Handbook | Readable context per occupation, for the write-up and the Coach | [Computer and IT](https://www.bls.gov/ooh/computer-and-information-technology/) |
| BLS OEWS | Actual employment and wages by occupation over time (backtests, option 4a) | [OEWS](https://www.bls.gov/oes/) |
| O*NET OnLine | Occupation codes, skills and Bright Outlook flags; checks the title → code mapping | [O*NET OnLine](https://www.onetonline.org/) |

### All sources considered

The core needs only BLS Employment Projections. The rest serve the proposed extensions (section 4) or context.

| Source | What it gives | Role | Link |
|:--|:--|:--|:--|
| BLS Employment Projections | 10-year employment change for about 800 occupations | **Core:** the growth number | [bls.gov/emp](https://www.bls.gov/emp/) · [2025–35 summary](https://www.bls.gov/news.release/ecopro.nr0.htm) · [data](https://www.bls.gov/emp/data.htm) |
| Occupational Outlook Handbook | Outlook, pay and education per occupation | Context for the write-up and the Coach | [bls.gov/ooh](https://www.bls.gov/ooh/) |
| BLS OEWS | Yearly employment and wages by occupation | Backtesting BLS (4a, 4d) | [bls.gov/oes](https://www.bls.gov/oes/) |
| O*NET OnLine | Occupation codes, skills, Bright Outlook | Code mapping, skills | [onetonline.org](https://www.onetonline.org/) |
| Projections Central | State short-term (2-year) and long-term projections | State-level view, if the team goes beyond national | [projectionscentral.org](https://projectionscentral.org/) · [long-term notes](https://projectionscentral.org/aboutlt) · [state sites](https://projectionscentral.org/stateprojectionssites) |
| BLS JOLTS | Monthly openings, hires and quits by industry | Cyclical signal (4b, 4d) | [bls.gov/jlt](https://www.bls.gov/jlt/) |
| Indeed job postings (FRED) | Daily postings indexes by field since 2020 | Leading signal (4b, 4d) | [Indeed series](https://fred.stlouisfed.org/tags/series?t=indeed) · [total](https://fred.stlouisfed.org/series/ihlidxus) · [software dev](https://fred.stlouisfed.org/series/IHLIDXUSTPSOFTDEVE) · [scientific R&D](https://fred.stlouisfed.org/series/IHLIDXUSTPSCREDE) |
| Fed funds rate (FRED) | Interest-rate history | Rate sensitivity (4b) | [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) |
| FOMC Summary of Economic Projections | The Fed's projected rate paths | Rate scenarios (4b) | [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) |

## 3. Method (core)

1. **Download** the occupation table from the 2025–2035 projections into `data/external/` once the terms of use are checked.
2. **Join** on the 6-digit occupation code: the first part of `onet_code` on listings (for example `15-2051`) matches BLS's code.
3. **Roll up to role buckets** with `ONET_ROLE_BUCKETS`, weighting each occupation by its 2025 employment.
4. **Convert to the chosen horizon** at the same yearly rate:
   - Yearly rate: (1 + 10-year growth)^(1/10) − 1
   - Growth over *n* years: (1 + 10-year growth)^(*n*/10) − 1, with *n* ≤ 5
5. **Save** `role_growth.pkl` with: role bucket, occupation codes, 2025 employment, 10-year growth, yearly rate, *n*-year growth, projections vintage. EDA_SDA_2 3.12 adds it as the compass's growth column.

**Write-up wording:** "expected growth (official projections)", never "will grow". BLS projections assume a full-employment economy and leave out business cycles; say so in the limitations.

## 4. Proposed extensions (not decided)

The 2026-10-09 decision dropped the other external datasets. Each option below would reopen part of it, so each needs a team decision first (run `/debate`, then log the outcome).

| Option | What it adds | Cost | Recommendation |
|:--|:--|:--|:--|
| **a. Error band from BLS's own track record** | Compare an earlier BLS projection (for example 2015–2025) with actual OEWS employment by occupation group; show growth as a range, not a single number | Small: public data, one table | **Adopt.** Strengthens the honesty rules; fast-changing data and AI roles carry the most uncertainty |
| **b. Interest-rate scenarios** | A near-term adjustment for years 1–2 under baseline, higher-for-longer and faster-cut rate paths from the Fed's projections ([FOMC](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)), with sensitivity estimated from rate history ([FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS)) and job postings ([Indeed on FRED](https://fred.stlouisfed.org/tags/series?t=indeed), [JOLTS](https://www.bls.gov/jlt/)), fading to the BLS trend by year 5 | Medium: a lagged regression and three scenarios | **Stretch.** Rates are one driver among several (the 2022–23 tech pullback also reflects post-COVID overhiring and AI); no causal claims |
| **c. Momentum from our listings** | Each bucket's new-listings trend from TimeSeries, shown **next to** growth | Small: TimeSeries already builds the series | **Show, don't blend.** Six months of listings can't be backtested against employment or support seasonal claims |
| **d. Full horizon-weighted blend** | A yearly forecast mixing a near-term signal (postings trend + JOLTS/Indeed + rate scenario) with the BLS trend; details below | Large: needs a postings history long enough to backtest | **Later / stretch.** Our six months can't support it; would rely on Indeed and JOLTS instead |
| **e. State level** | Projections Central's state projections instead of national | Small–medium | Only if the app adds location-based growth |

### 4b in detail: interest-rate scenarios

1. **Don't forecast rates ourselves.** Use three paths from the FOMC projections: **baseline**, **higher-for-longer**, **faster cuts**.
2. **Estimate sensitivity:** regress the year-over-year change in Indeed postings or JOLTS openings for the field on **lagged** fed funds changes; test 6, 12 and 18-month lags and pick by out-of-sample fit; report the coefficient with its confidence interval.
3. **Apply to years 1–2 only,** fading to the BLS trend.
4. **No causal claims:** rates are one driver; 2022–23 also reflects post-COVID overhiring and AI.

### 4d in detail: the horizon-weighted blend

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

- `g_BLS` is the yearly rate from section 3; `g_near` comes from postings trends (validated against Indeed and JOLTS) adjusted by the rate scenario.
- Postings aren't hires: scale them to employment through their historical relationship, as a directional signal only.
- **Tune the weights in a backtest**, run `/debate` on them, and log the final choice.

### Validation for 4a and 4d

- **BLS track record:** an earlier projection (for example 2015–2025) vs. OEWS actuals, by occupation group.
- **Rolling-origin backtest:** rerun the method as if standing in past years (for example 2019, 2021, 2023) and compare with actuals.
- **Metrics:** MAPE and direction (growth vs. decline right).
- **Baselines to beat:** BLS interpolation alone, and "last year's growth continues".
- **Bands:** 80% and 95% intervals sized from past errors, widening with the horizon.

### Expected accuracy (state in the write-up)

| Horizon | Confidence | Main driver |
|:--|:--|:--|
| Years 1–2 | Moderate | Postings trend and rate scenarios (if 4b/4d adopted), otherwise BLS |
| Years 3–5 | Low to moderate, wide bands | BLS structural trend |
| Over 5 years | Not shown as a recommendation (decided cap); BLS 10-year figure as context only | BLS |

Fast-changing roles (data, AI and ML) carry extra uncertainty.

### Outputs

- [ ] `role_growth.pkl` (section 3), and the compass growth column
- [ ] If 4a: growth with an 80%/95% band per role bucket
- [ ] If 4b/4d: a yearly table per role bucket × scenario, years 1–5, and a fan chart per bucket (baseline line, scenario band, uncertainty band, BLS 10-year marker)
- [ ] A backtest summary (method vs. baselines) and a methods-and-limitations note

## 5. Tasks

- [ ] **Get PredictLeads' written consent to combine their data with BLS data** (Training Services Agreement 4.3; already an open advisor question on external data)
- [ ] Check the BLS terms of use; download the 2025–2035 occupation table into `data/external/`
- [ ] Join on occupation code, roll up to role buckets (employment-weighted), save `role_growth.pkl`
- [ ] Add the growth column to EDA_SDA_2 3.12 and rerun
- [ ] Team picks the horizon (at most 5 years) and decides on options 4a–4e; log in `docs/decisions.md`
- [ ] `ML_Fit`: apply growth as a visible, adjustable adjustment after the fit model; show how much it moved each suggestion
- [ ] Limitations in the write-up: projections, national, occupation-wide, no business cycles

## 6. Data handling

BLS, FRED and Federal Reserve data are public. Store downloads in `data/external/` (git-ignored) like other external data, and track them for the final deletion. Nothing from our listings goes into any outside service. **Joining BLS figures to PredictLeads listings combines their data with third-party material, which the agreement allows only with PredictLeads' prior written consent (4.3):** get it in writing before the join, and log it in `docs/decisions.md`.
