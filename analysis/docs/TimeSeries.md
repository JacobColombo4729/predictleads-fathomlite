<!-- Generated from notebooks/TimeSeries.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: Time Series

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

When do listings in the career-change target appear and close? Answers "when to search" and "how often to check", and creates timing features for the `ML_Rank` notebook.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### Clarifying questions
- How many new scope listings appear per day and week, and is there a weekly rhythm?
- Is that rhythm the employers' or the crawler's? (`posted`, read as Central European dates, vs. `first_seen`)
- Is there a trend across April–September? By role bucket, too: if the team adopts the proposed listing-momentum view (job growth tasks in `ANALYSIS.md`), each bucket's new-listings trend is shown next to expected growth, never blended into it (six months can't be backtested against employment). The full blend (4d) would draw its near-term signal from Indeed and JOLTS, which go back years, not from our six months.
- Can next-day or next-week volume be forecast better than a naive baseline?
- How fast do listings close in each month (survival by month)?
- Do listings open to career changers arrive on a different rhythm, for example entry-level hiring waves?

**Data rules:** aggregates only. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard (TimeSeries template) | Taken further |
|:--|:--|:--|
| 1. Setup and load | Load the table, check it | Only the needed columns of EDA_SDA_2's scope table (memory); originals only (no reposts) |
| 2. The series | `resample()` to daily and weekly counts of new listings | Separate series from `first_seen` and from `posted` (CET), to separate crawler from employer rhythm; with and without job boards and likely ghosts; partial weeks marked, not dropped |
| 3. Analysis **before** features | Plot, rolling mean and std, decomposition (trend, seasonal, residual; weekly period), **ADF test** | STL; ACF / PACF; KPSS as a cross-check; differencing orders chosen from the tests |
| 4. Forecasting | Calendar features, **lags**, rolling mean and std **with `.shift()`**; train / validation / test in time order; **DummyRegressor**, LinearRegression, RandomForest with a tuning loop, ARIMA; **RMSE**; test used once | Three naive baselines (mean, last value, same weekday last week); SARIMA with weekly seasonality; gradient boosting; MAE; rolling-origin cross-validation; prediction intervals |
| 5. Survival and cadence | | Kaplan–Meier by month first seen (still-open listings censored); "check every N days" from the closing curve |
| 6. Role buckets | | Weekly new-listings trend per bucket (the listing-momentum view; shown, never blended into job growth) |
| 7. Conclusion and handoff | What was done, best model, limitations (six months of history) | Saved aggregates for `ML_Rank` and the app |

**Inputs and views.** Everything comes from EDA_SDA_2's scope table (`scope_table.parquet`), loaded column by column. Counts use **original listings only** (`is_repost` excluded). The two cuts that matter are shown side by side: **without likely ghosts** (the app hides them) and **without job boards** (they stop after May, so they bend any trend). Dates: `first_seen` follows the crawler (UTC days); `posted` is the employer's date, read as Central European dates. Only six months of data: **no seasonal claims**.

## 2. The series: new listings per day and per week

Two clocks. **`first_seen`** is when the crawler found the listing, so it follows the crawler's schedule. **`posted`** is the employer's own date (about 40% of listings have one), so it follows employers. If a weekly pattern shows in both, it is real hiring behavior; if only in `first_seen`, it is the crawler. The comparison is the day-of-week rule in `STANDARDS.md`.

*Comment (after the run): level, any step change, the weekly rhythm in the daily panel, and what the grey partial weeks hide.*

*Comment (after the run): does the employer's weekly count follow the crawler's? Posting dates near September 30 undercount, because a listing posted but not yet crawled is not in the table.*

*Comment (after the run): which weekdays are heavy on each clock. A weekend dip only in `posted` is employer behavior; a pattern only in `first_seen` is the crawler's schedule.*

*Comment (after the run): the job-board share by month shows where the collection changed; any step in the weekly count at that point is the collection, not the market. Likely ghosts are, by definition, over 90 days old and open on September 30, so they were all first seen before July: dropping them lowers the early months and tilts the trend upward. Trend claims therefore use the all-originals and no-job-board views; the no-ghost view is what the app shows.*

**What the data shows**
- (fill in after the run: level and range of new listings per day and per week)
- (fill in after the run: weekday shares on each clock, and whether the weekly rhythm is the crawler's or the employers')
- (fill in after the run: when the job-board share falls and what it does to the weekly count)
- (fill in after the run: days with zero listings, if any, and what they look like)

## 3. Analysis before features

**Series used from here:** daily new original listings by `first_seen`, all sources (`SERIES`). That is what a career changer meets in the app. Set it to `"first_seen_no_board"` to repeat sections 3 and 4 on the steadier collection method (decide after 2e).

*Comment (after the run): a drifting mean or a std that grows with the level means the series is not stationary and may need a log or a difference.*

*Comment (after the run): trend line vs. the 28-day mean; the size of the weekly swing in listings; whether the remainder has spikes (a company batch, a crawler outage).*

*Comment (after the run): which lags stand out; this guides the lag features in section 4 (1, 2, 3, 7, 14).*

*Comment (after the run): what ADF and KPSS agree on for the level, and which differencing makes the series stationary. Six months is short, so a weak trend can pass for a unit root; the two tests together are a guide, not a proof.*

**What the data shows**
- (fill in after the run: trend direction and size from STL, and its strength)
- (fill in after the run: weekly seasonal strength and the heaviest and lightest weekday)
- (fill in after the run: ACF / PACF lags that matter)
- (fill in after the run: ADF and KPSS verdicts and the chosen differencing)

## 4. Forecasting new listings per day

**Question:** can tomorrow's count be forecast better than a naive guess? Useful for the app's "expect about N new listings" line, and a check that the timing signal is real before `ML_Rank` uses it.

| Decision | Choice | Why |
|:--|:--|:--|
| Target | New original listings per day (`SERIES`) | What the app sees |
| Features | Lags 1, 2, 3, 7, 14; rolling mean and std over 7 and 14 days; weekday; US federal holiday | Each lag and rolling statistic is shifted one day, so day t never sees itself; weekday and holiday are known ahead |
| Split | Train April - July, validate August, test September; no shuffling | Time order; the test month is used once, at the end (4h) |
| Baselines | Training mean (`DummyRegressor`), last value (yesterday), same weekday last week | Beating "yesterday" and "last week" is the real bar |
| Models | Linear regression, random forest (tuned on validation), gradient boosting, SARIMA | Course standard plus weekly seasonality |
| Scores | RMSE (listings; large misses count more) and MAE | Reported on validation, then rolling-origin cross-validation (mean ± std) |
| Forecast horizon | One day ahead, from actual history | Every model is scored the same way; SARIMA uses its fixed parameters on the actual past |

*Comment (after the run): which model wins on validation and by how much over "same weekday last week". If nothing clearly beats the baselines, say so: the forecast adds little over a weekday rule.*

*Comment (after the run): does the validation ranking hold across folds, and is the gap to the baselines larger than the fold-to-fold spread?*

*Comment (after the run): the chosen model's test RMSE and MAE against the baselines, in listings per day and as a share of the average daily count. Compare with its validation score: a large gap means validation was lucky. September also ends on the data's last day, so the final days can be thin if the crawl lags.*

**What the data shows**
- (fill in after the run: validation winner, RMSE and MAE vs. the three baselines)
- (fill in after the run: cross-validation mean ± std, and whether the ranking holds)
- (fill in after the run: test RMSE and MAE of the chosen model vs. the baselines, interval coverage)
- (fill in after the run: the best SARIMA order and whether weekly seasonality was worth modeling)
- **Limits:** six months, one crawler, job boards stopping after May: the model describes this period, not a season.

## 5. How long listings stay open, and how often to check

**Survival by month first seen.** Kaplan–Meier on days open, with listings still open on September 30 **censored** (we know they lasted at least that long, not when they close). Later months have less follow-up, so each curve stops at the days every listing in that month could have been followed.

**Cadence.** Someone who checks every N days meets a new listing U days after it appears, with U between 0 and N (equally likely). They **catch** it if it is still open then, and with `k` days left to apply if it is still open at U + k. The catch rate is the average of the closing curve over that window.

*Comment (after the run): do the monthly curves overlap or does closing speed change by month? A later month closing faster can be real, or the crawler losing sites; check against the job-board stop (2e) before reading it as the market.*

*Comment (after the run): where each line crosses the dashed 80% line is the longest gap between checks that still catches four in five listings.*

*Comment (after the run): does the answer differ by bucket enough to give a different "check every N days" per role, or is one number fine?*

**What the data shows**
- (fill in after the run: share still open after 7, 14 and 30 days, and median days open, by month)
- (fill in after the run: whether closing speed moved between months, and what else could explain it)
- (fill in after the run: the recommended gap between checks, e.g. "check every N days to catch 80% with 3 days to apply")
- (fill in after the run: buckets whose cadence differs from the overall one)
- **Limits:** `days_open` ends at the crawler's last sighting, which can come before the real closing (and a crawl dropout ends it early), so true time left is a little longer than shown.

## 6. New listings per week by role bucket (listing momentum)

Each bucket's weekly count of new listings, for the "listing momentum" view next to expected job growth (BLS). **Shown beside growth, never blended into it:** six months cannot be backtested against employment, and the full growth blend would take its near-term signal from Indeed and JOLTS, which go back years. Two views per bucket, all sources and without job boards, because the job-board stop after May moves some buckets more than others.

*Comment (after the run): which buckets rise, fall or stay flat, and whether the dashed (no job board) line tells the same story as the solid one.*

*Comment (after the run): buckets whose views disagree are not given a momentum label.*

**What the data shows**
- (fill in after the run: the buckets with rising and falling weekly listings, in both views)
- (fill in after the run: buckets where the job-board stop changes the story)
- (fill in after the run: weekly volume per bucket, the largest and smallest)
- **Limits:** new listings measure openings posted, not net hiring; a bucket with falling listings can still grow in jobs. These are shown next to the BLS growth, never added to it.

### What was done
- (fill in after the run: one paragraph: the series, the two clocks, what the decomposition and tests showed)
- (fill in after the run: the forecast result in one sentence, with the test RMSE and MAE vs. the best baseline)
- (fill in after the run: the cadence advice: "check every N days")
- (fill in after the run: the per-bucket momentum in one sentence)

### Answers to the clarifying questions
| Question | Answer |
|:--|:--|
| How many new scope listings per day and week; is there a weekly rhythm? | (fill in after the run) |
| Is the rhythm the employers' or the crawler's? | (fill in after the run) |
| Is there a trend across April-September, and by role bucket? | (fill in after the run) |
| Can next-day volume be forecast better than a naive baseline? | (fill in after the run) |
| How fast do listings close in each month? | (fill in after the run) |
| Do listings open to career changers arrive on a different rhythm? | Not tested here: `career_open` comes from EDA_SDA_3's text features. Add it as a cut once that table exists. |

### Limitations
- Six months of history: no seasonal claims, and the trend is a trend of this period only.
- The crawler's schedule and collection changes (job boards stop after May; descriptions go missing more from June) shape `first_seen`; the no-job-board view and the `posted` clock are the guards.
- `days_open` ends at the last sighting; crawl dropouts end it early.
- A forecast of arrivals says nothing about quality or fit; it is a timing aid.

### Handoff
- **Saved** (`data/clean/`): `timeseries_daily.parquet` (counts per day, four views), `timeseries_weekly_bucket.parquet` (per bucket and week, all sources and without job boards), `timeseries_cadence.parquet` (catch rates by gap between checks).
- **To `ML_Rank`:** the closing curve by month tells how much to trust "still open" at different ages; the cadence table is the basis of the "apply by" and "check every N days" copy in the app. Features known at first sight stay as in `STANDARDS.md`: weekday of `first_seen` follows the crawler, so it is a feature for when a listing appears to the app, not for employer behavior.
- **To the app:** the listing-momentum view per bucket next to BLS growth (never blended), and the "expect about N new listings" line from the chosen forecast.
- **Open:** repeat section 4 on `first_seen_no_board` if 2e shows the job-board stop moves the series; add the `career_open` cut once EDA_SDA_3 exists.
