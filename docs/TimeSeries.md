<!-- Generated from notebooks/TimeSeries.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: Time Series

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

When do listings in the career-change target appear and close? Answers "when to search" and "how often to check", and creates timing features for the `ML_Rank` notebook.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, and what's the scope? | Final table 1 |
| EDA_SDA_2 | Which listings do we study? The niche cut, additional cleaning, non-text checks, ghost listings | Niche table, non-text features, occupation summary |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Niche table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### Clarifying questions
- How many new niche listings appear per day and week, and is there a weekly rhythm?
- Is that rhythm the employers' or the crawler's? (`posted`, read as Central European dates, vs. `first_seen`)
- Is there a trend across April–September? By role bucket, too: if the team adopts the proposed listing-momentum view (`docs/job_growth.md` 4c), each bucket's new-listings trend is shown next to expected growth, never blended into it (six months can't be backtested against employment).
- Can next-day or next-week volume be forecast better than a naive baseline?
- How fast do listings close in each month (survival by month)?
- Do listings open to career changers arrive on a different rhythm, for example entry-level hiring waves?

**Data rules:** aggregates only. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard (TimeSeries template) | Taken further |
|:--|:--|:--|
| 1. Setup | Load EDA_SDA_4's table | |
| 2. Series | `resample()` to daily and weekly counts of new listings | Separate series from `first_seen` and from `posted` (CET), to separate crawler from employer rhythm |
| 3. Analysis **before** features | Plot, rolling mean and std, `seasonal_decompose` (trend, seasonal, residual; weekly period) | STL decomposition; autocorrelation plots (ACF / PACF) |
| 4. Stationarity | **ADF test** | KPSS test as a cross-check |
| 5. Features | Calendar features (month, day, day of week), **lags**, rolling mean and std **with `.shift()`** (no leakage) | |
| 6. Split | Train / validation / test in time order, **no shuffling**; test used once | Rolling-origin (time series) cross-validation |
| 7. Models | **DummyRegressor** baseline, LinearRegression, RandomForest with a tuning loop, **ARIMA** (`arma_order_select_ic`) | SARIMA with weekly seasonality; gradient boosting with lags |
| 8. Evaluation | **RMSE** on validation; the best model tested once | MAE; prediction intervals |
| 9. Timing patterns | | Survival by month first seen; arrival cadence → "check every N days" |
| 10. Conclusion | What was done, best model, limitations (six months of history) | |

## 2.–10.

_To do, once EDA_SDA_4's table exists._
