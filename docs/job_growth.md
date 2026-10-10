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
| BLS OEWS | Actual employment by occupation over time (section 4c backtest only) | [OEWS](https://www.bls.gov/oes/) |

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

## 5. Tasks

- [ ] **Get PredictLeads' written consent to combine their data with BLS data** (Training Services Agreement 4.3; already an open advisor question on external data)
- [ ] Check the BLS terms of use; download the 2025–2035 occupation table into `data/external/`
- [ ] Join on occupation code, roll up to role buckets (employment-weighted), save `role_growth.pkl`
- [ ] Add the growth column to EDA_SDA_2 3.12 and rerun
- [ ] Team picks the horizon (at most 5 years) and decides on options 4a–4c; log in `docs/decisions.md`
- [ ] `ML_Fit`: apply growth as a visible, adjustable adjustment after the fit model; show how much it moved each suggestion
- [ ] Limitations in the write-up: projections, national, occupation-wide, no business cycles

## 6. Data handling

BLS, FRED and Federal Reserve data are public. Store downloads in `data/external/` (git-ignored) like other external data, and track them for the final deletion. Nothing from our listings goes into any outside service. **Joining BLS figures to PredictLeads listings combines their data with third-party material, which the agreement allows only with PredictLeads' prior written consent (4.3):** get it in writing before the join, and log it in `docs/decisions.md`.
