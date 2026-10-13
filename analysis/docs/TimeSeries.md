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

*Comment: both clocks are heavy Tuesday to Friday and light on the weekend, so the weekly rhythm is not only the crawler's. The weekend dip is deeper on `posted` (Saturday 7.4% and Sunday 3.9%) than on `first_seen` (11% and 8.4%), and Monday is lighter on `first_seen` (11%) than on `posted` (14%).*

*Comment: the job-board share by month shows where the collection changed (22% and 23% in April and May, 0.19% in June); any step in the weekly count at that point is the collection, not the market. Likely ghosts are, by definition, over 90 days old and open on September 30, so they were all first seen before July: dropping them lowers the early months and tilts the trend upward. Trend claims therefore use the all-originals and no-job-board views; the no-ghost view is what the app shows.*

**What the data shows**
- **Level and range:** the 183 days (27 weeks, 25 of them full) average 4,586 new original listings a day (median 4,686, standard deviation 1,600, from 653 to 14,314), about 32,000 a week (4,586 x 7). Without likely ghosts the mean is 4,546, without job boards 4,237, and by the employer's posting date 1,826 (only 40.9% of listings have one). By month first seen the counts are 138K, 142K, 124K, 137K, 142K and 156K (April to September).
- **A weekly rhythm on both clocks:** Tuesday to Thursday carry 17% of the week each on `first_seen` (19% on `posted`), Friday 19% (18%), Monday 11% (14%), Saturday 11% (7.4%) and Sunday 8.4% (3.9%). The employers' own dates have the deeper weekend dip (Saturday and Sunday are 11.3% of the week, against 19.4% for the crawler), so the dip is real employer behavior, and the crawler's weekend share is higher than the employers'. Posting dates are read as Central European dates, so a listing posted late in a US day can fall on the next one.
- **The job-board stop is large but the monthly count does not step down with it:** job boards were 22% of April and 23% of May listings and 0.19% of June, falling to 0.09% in September, yet monthly counts are 124K in June and 156K in September. Other sources therefore make up a larger part of the count after May; this notebook cannot say whether that is the market or the collection widening, which is why trend claims use the no-board view too. Likely ghosts are 1.6% and 2.1% of April and May listings and 0.0% from July, as expected from their definition.
- **No day is empty, and the one big peak is job boards:** no day has zero new listings and the lowest has 653. The busiest day (14,314, over three times the mean) falls to 7,326 without job boards, and barely moves (14,027) without company batch first sightings, so it is not a company batch.

## 3. Analysis before features

**Series used from here:** daily new original listings by `first_seen`, all sources (`SERIES`). That is what a career changer meets in the app. Set it to `"first_seen_no_board"` to repeat sections 3 and 4 on the steadier collection method (decide after 2e).

*Comment (after the run): a drifting mean or a std that grows with the level means the series is not stationary and may need a log or a difference.*

*Comment (after the run): trend line vs. the 28-day mean; the size of the weekly swing in listings; whether the remainder has spikes (a company batch, a crawler outage).*

*Comment (after the run): which lags stand out; this guides the lag features in section 4 (1, 2, 3, 7, 14).*

*Comment: for the level, ADF does not reject a unit root (p = 0.132) and KPSS does not reject stationarity (p = 0.1, the largest value the test reports), so the notebook calls it inconclusive; one difference makes the series stationary on ADF (p = 0.000), so d = 1 and no seasonal difference is needed. Six months is short, so a weak trend can pass for a unit root; the two tests together are a guide, not a proof.*

**What the data shows**
- **The trend is weak:** the STL trend component has a strength of 16% (0 = none, 1 = the whole series). Its direction and size are in the chart, not in a printed table, so they are not restated here; the monthly counts in section 2 (124K to 156K) and the weekly momentum in section 6 give the size by month and bucket.
- **The weekly pattern is strong:** the weekly seasonal component has a strength of 61%. On the crawler's clock Friday is the heaviest weekday (19% of the week) and Sunday the lightest (8.4%), from 2d.
- **ACF / PACF:** the lags are in the charts; section 4 uses lags 1, 2, 3, 7 and 14 (as planned), and the SARIMA search later settled on terms at lags 1 and 2 and a weekly (7-day) autoregressive and moving-average pair.
- **Stationarity:** the level is inconclusive (ADF p = 0.132, KPSS p = 0.1); the 1-day difference, the 7-day difference and both are stationary on ADF (p = 0.000), and KPSS never rejects. The chosen differencing is d = 1, D = 0 (period 7). The tests cannot rule out a weak trend passing for a unit root in six months.

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

*Comment: linear regression has the lowest RMSE of the models on August (552.0), but nothing beats "same weekday last week" (RMSE 548.1, MAE 426.0): linear regression's MAE is 460.1, SARIMA's 453.8 and the random forest's 457.8. The forecast adds little over a weekday rule on validation: the two are tied (3.9 RMSE apart). By design the choice is made among the models only (4g picks the best non-baseline RMSE), since the baselines are the bar to beat, not candidates; so linear regression goes forward, chosen before September was touched.*

*Comment: the ranking does not hold from validation to cross-validation. SARIMA is best (600.8), then the random forest (625.9), linear regression (662.3) and the weekday baseline (693.9). SARIMA's 93.1-listing edge over the baseline is smaller than the fold-to-fold spread (107.9 and 139.9), so it is not a clear win.*

*Comment: on September the chosen linear regression scores RMSE 840.5 and MAE 620.7 listings a day, about 18% and 14% of the six-month average of 4,586, and beats the weekday baseline (1018.1 and 766.7) by 177.6 and 146.0. Its validation RMSE was 552.0, so the test error is 52% higher: validation was the easier month. September also ends on the data's last day, so the final days can be thin if the crawl lags. This is the one test run; nothing is to be tuned against it. Read it as one comparison of a model with a naive baseline that tied it on validation, not as proof the model is better; switching to the baseline (or to SARIMA, which led the cross-validation) after seeing September would be choosing with hindsight, so the choice stands as made.*

**What the data shows**
- **Validation (August):** linear regression has the lowest RMSE among the models (552.0, MAE 460.1), followed by SARIMA (577.0, 453.8), the random forest (640.0, 457.8; best of 24 settings) and gradient boosting (787.8, 650.3). The weekday baseline is as good or better (548.1, MAE 426.0); the other two baselines are far behind: training mean 1306.9 and yesterday 1569.8 (MAE 1139.0 and 1307.5). So the weekly pattern is the signal, and a model adds little on top of it.
- **Cross-validation (5 folds x 14 days, April to August, RMSE mean ± std):** SARIMA 600.8 ± 107.9, random forest 625.9 ± 66.3, linear regression 662.3 ± 115.9, weekday baseline 693.9 ± 139.9, gradient boosting 780.5 ± 155.1, training mean 1223.6 ± 167.6, yesterday 1328.7 ± 271.3. The order differs from validation (linear regression, chosen on validation, is third), and the gaps among the top four are within the fold spread, so only "far better than the mean and yesterday baselines" is clear.
- **Test (September, run once, linear regression chosen on validation):** RMSE 840.5, MAE 620.7 against 1018.1 and 766.7 for the weekday baseline, 1514.1 and 1177.4 for yesterday, and 1588.2 and 1428.2 for the training mean. The model beats the baselines on this month, but its error is 52% above its August RMSE. The 80% interval (forecast -809 to +519 listings) covered 63.3% of September's days, so it was too narrow. One month of test data cannot separate a better model from a month that suits it: the model and the weekday baseline tied on validation (552.0 vs. 548.1), and the model was chosen among models only, so this is one comparison with a naive baseline, compared once.
- **SARIMA and weekly seasonality:** the best order by AIC (of 36) is (0, 1, 2) x (1, 0, 1, 7). Weekly seasonality was worth modeling: the weekday baseline and the models are well below the mean and yesterday baselines on RMSE in validation and cross-validation. SARIMA led the cross-validation but was not the model chosen on validation, so it was not run on the test month.
- **Limits:** six months, one crawler, job boards stopping after May: the model describes this period, not a season.

## 5. How long listings stay open, and how often to check

**Survival by month first seen.** Kaplan–Meier on days open, with listings still open on September 30 **censored** (we know they lasted at least that long, not when they close). Later months have less follow-up, so each curve stops at the days every listing in that month could have been followed.

**Cadence.** Someone who checks every N days meets a new listing U days after it appears, with U between 0 and N (equally likely). They **catch** it if it is still open then, and with `k` days left to apply if it is still open at U + k. The catch rate is the average of the closing curve over that window.

*Comment: the curves for April and May are close (74% and 75% still open after 7 days, median 25 and 27 days), and later months close faster (65% after 7 days and a median of 22 days for August). A later month closing faster can be real, or the crawler losing sites; the job-board stop (2e) falls between May and June, so it is one candidate, along with the mix of listings. This notebook does not separate them.*

*Comment: where each line crosses the dashed 80% line is the longest gap between checks that still catches four in five listings: 7 days with no extra days to apply, 3 days with 3 extra days, and none with 7 extra days (daily checking catches 69.7% to 70.0%).*

*Comment: one number is fine for most buckets (the longest gap for 80% is 3 days in seven of the eleven that have one), but not for all: it is 2 days for software development, 1 for database, 8 for marketing specialists and 14 for technical sales, and no gap reaches 80% for quality assurance.*

**What the data shows**
- **Share still open, by month first seen (7 / 14 / 30 days, median days open):** April 74% / 62% / 43%, 25 days; May 75% / 63% / 44%, 27; June 68% / 58% / 42%, 24; July 68% / 58% / 43%, 24; August 65% / 55% / 39%, 22; September 68% / 58%, with no 30-day figure (too little follow-up). The share still open on September 30 (censored) rises from 7.0% for April to 32% for August and 65% for September.
- **Closing speed moved between months:** the April-May curves sit above the later ones, with the drop between May and June (75% to 68% open at 7 days) and August the lowest. Possible reasons, none tested here: the job-board stop (22% and 23% of April and May listings, 0.19% from June), the crawler's coverage, and the mix of listings. September's 68% is back at the June-July level, so the series does not fall steadily.
- **Recommended gap between checks:** to see about four in five new listings with 3 days left to apply, check every 3 days (80.4% for all originals, 80.2% for the app view); longer gaps catch less (7 days 74.1%, 14 days 67.9%, 30 days 57.9%). With no extra days to apply the longest gap for 80% is 7 days, and with 7 extra days no gap reaches 80% (daily checking catches 70.0%).
- **Buckets that differ:** the catch rate for checking every 3 days (3 days to apply) runs from 75% (quality assurance) to 89% (technical sales; median 38 days open), and is 86% for marketing specialists (median 28). The longest gap for 80% with 3 days to apply is 2 for software development, 1 for database, 3 for most others, 8 for marketing specialists and 14 for technical sales; quality assurance has none. Median days open runs from 20 (business and data analysis) to 38.
- **Limits:** `days_open` ends at the crawler's last sighting, which can come before the real closing (and a crawl dropout ends it early), so true time left is a little longer than shown.

## 6. New listings per week by role bucket (listing momentum)

Each bucket's weekly count of new listings, for the "listing momentum" view next to expected job growth (BLS). **Shown beside growth, never blended into it:** six months cannot be backtested against employment, and the full growth blend would take its near-term signal from Indeed and JOLTS, which go back years. Two views per bucket, all sources and without job boards, because the job-board stop after May moves some buckets more than others.

*Comment (after the run): which buckets rise, fall or stay flat, and whether the dashed (no job board) line tells the same story as the solid one.*

*Comment: buckets whose views disagree are not given a momentum label; here six of the twelve agree (all rising) and six disagree.*

**What the data shows**
- **Rising in both views (last 4 full weeks vs. first 4):** software development (+22% all sources, +50% without job boards), networks and systems (+17%, +59%), security (+19%, +65%), quality assurance (+21%, +54%), database (+7%, +47%) and IT project management (+5%, +37%). No bucket falls in both views.
- **Where the job-board stop changes the story:** with all sources, data science (-2%), IT support (-8%), business and data analysis (-10%), marketing specialists (-24%), other computer roles (-8%) and technical sales (-1%) fall; without job boards the same six rise (+31%, +33%, +25%, +9%, +28%, +41%). Their views disagree, so they carry no momentum label; the all-sources fall is consistent with the job boards ending after May, which this notebook does not prove. Marketing specialists is the weakest in both (-24% and +9%).
- **Weekly volume:** software development is the largest bucket (about 9,200 new listings a week, 8,668 without job boards), then networks and systems (6,035), IT project management (4,032) and data science (3,735); the smallest are technical sales (340), other computer roles (588), marketing specialists (1,107) and quality assurance (1,203).
- **Limits:** new listings measure openings posted, not net hiring; a bucket with falling listings can still grow in jobs. These are shown next to the BLS growth, never added to it.

### What was done
- **The series:** 183 days of new original listings (April 1 to September 30), 4,586 a day on average and about 32,000 a week, with no empty day. A weekly rhythm shows on both the crawler's clock (`first_seen`) and the employers' (`posted`): heavy Tuesday to Friday, light on the weekend, and deeper on the weekend for employers (Saturday and Sunday 11.3% of the week, against 19.4% for the crawler). STL puts the weekly component's strength at 61% and the trend's at 16%; the level is inconclusive on ADF and KPSS and one difference makes it stationary (d = 1, D = 0). Job boards stop after May (22% and 23% of listings in April and May, 0.19% in June), but monthly counts do not fall (124K to 156K).
- **Forecast:** on September, run once, the linear regression chosen on validation scores RMSE 840.5 and MAE 620.7 listings a day against 1018.1 and 766.7 for the best baseline (same weekday last week). On validation that baseline was as good (548.1 vs. 552.0 RMSE), and the cross-validation ranking differed (SARIMA 600.8, baseline 693.9, gaps within fold spread). The 80% interval covered 63.3%. The weekday pattern is the signal; models add little on top.
- **Cadence:** check every 3 days to see about four in five new listings with 3 days left to apply (80.4%); 7 days gives 74.1%. Software development (2 days) and database (1) need a shorter gap, quality assurance never reaches 80%, and marketing specialists (8) and technical sales (14) can check less often.
- **Momentum:** software development, networks and systems, security, quality assurance, database and IT project management rise in both views (+5% to +22% all sources, +37% to +65% without job boards); the other six buckets fall with all sources and rise without job boards, so they carry no label.

### Answers to the clarifying questions
| Question | Answer |
|:--|:--|
| How many new scope listings per day and week; is there a weekly rhythm? | 4,586 a day (median 4,686, 653 to 14,314) and about 32,000 a week. Yes: Tuesday to Friday carry 17-19% of the week each on `first_seen`; Sunday is lowest (8.4%). |
| Is the rhythm the employers' or the crawler's? | Both. The weekend dip is deeper on `posted` (Saturday 7.4%, Sunday 3.9%) than on `first_seen` (11%, 8.4%), so it is employer behavior, and the crawler adds weekend arrivals. |
| Is there a trend across April-September, and by role bucket? | Weak overall (STL trend strength 16%; monthly counts 138K, 142K, 124K, 137K, 142K, 156K). By bucket, six rise in both views and six disagree (see momentum); the job-board stop after May affects the all-sources view. Six months only: no seasonal claim. |
| Can next-day volume be forecast better than a naive baseline? | Better than the mean and yesterday baselines, yes; better than "same weekday last week", not clearly. Validation: 552.0 vs. 548.1 RMSE. Cross-validation: SARIMA 600.8 vs. 693.9, within the fold spread. Test (September, once): 840.5 vs. 1018.1. |
| How fast do listings close in each month? | 65-75% are still open after 7 days, 55-63% after 14 and 39-44% after 30; median 22-27 days. Later months close faster (August 65% open at 7 days vs. 75% for May); the cause is not separated here. |
| Do listings open to career changers arrive on a different rhythm? | Not tested here: `career_open` comes from EDA_SDA_3's text features. Add it as a cut now that EDA_SDA_3's table exists. |

### Limitations
- Six months of history: no seasonal claims, and the trend is a trend of this period only.
- The crawler's schedule and collection changes (job boards stop after May; descriptions go missing more from June) shape `first_seen`; the no-job-board view and the `posted` clock are the guards.
- `days_open` ends at the last sighting; crawl dropouts end it early.
- The test month is one month and was used once; its error (RMSE 840.5) is 52% above the validation error, and the 80% interval covered 63.3%. Nothing is to be tuned against it.
- A forecast of arrivals says nothing about quality or fit; it is a timing aid.

### Handoff
- **Saved** (`data/clean/`): `timeseries_daily.parquet` (183 rows, four views), `timeseries_weekly_bucket.parquet` (324 rows, per bucket and week, all sources and without job boards), `timeseries_cadence.parquet` (18 rows, catch rates by gap between checks).
- **To `ML_Rank`:** the closing curve by month tells how much to trust "still open" at different ages (65-75% open at 7 days, 55-63% at 14, 39-44% at 30); the cadence table is the basis of the "apply by" and "check every N days" copy in the app. Features known at first sight stay as in `STANDARDS.md`: weekday of `first_seen` follows the crawler, so it is a feature for when a listing appears to the app, not for employer behavior.
- **To the app:** the listing-momentum view per bucket next to BLS growth (never blended), and, if wanted, the "expect about N new listings" line: the chosen forecast beats the baselines on the test month but not "same weekday last week" on validation, and its 80% interval was too narrow (63.3% covered), so show a weekday-based range, not a precise count.
- **Open:** repeat section 4 on `first_seen_no_board` (2e shows the job-board stop does not step the monthly count but changes the buckets' momentum); add the `career_open` cut from EDA_SDA_3's table.
