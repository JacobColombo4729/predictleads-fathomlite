<!-- Generated from notebooks/EDA_SDA_4.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 4 (Statistics)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

Vets every candidate feature, non-text (EDA_SDA_2) and text (EDA_SDA_3), with statistical evidence, and hands a **modeling-ready table** to `ML_Rank` (`TimeSeries` and `ML_Fit` read the scope and text tables directly).

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Text features, text rules as labels and opt-in filters (keyword rules, not validated), skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### Clarifying questions
- Which features are redundant with each other (collinearity), and which genuinely overlap?
- Which differences between groups are real, and which are large enough to matter to an applicant?
- Do factor effects hold within each other (for example source within company-volume bands), or reverse (Simpson's paradox)?
- **Adjusted check:** do the factors that survived EDA_SDA_2's within-group checks still matter in one **Cox survival regression** that adjusts for all the confounders at once (company volume, source type, staffing agency, role bucket, seniority, month first seen, has a description; state and platform are not loaded here and company type is not in the scope table, so they are not in the model), with errors clustered by company? Report hazard ratios with 95% intervals, and check the proportional-hazards assumption.
- How much does clustering by company inflate apparent certainty?
- Do the career-change flags carry signal beyond seniority, company size and source, and which parts of `career_open` hold up?
- Do factor effects differ by role bucket, or hold across them?
- Are the survival assumptions reasonable (censoring, left truncation)?
- **Is each feature stable?** Does its effect hold in every month (ML_Rank trains on April–July and tests on September), between listings from the same company, and in every role bucket?
- **Is each feature safe?** Is it known when a listing first appears (leakage audit), and does its distribution drift with the collection changes (job boards after May, descriptions from June)?

**Data rules:** aggregates only; never `df.head()` or sample rows. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard | Taken further |
|:--|:--|:--|
| 1. Setup | Load EDA_SDA_3's table | Shared helpers in `src/` |
| 2. Descriptive statistics | Per-group count, mean, median, std, variance, min, max, Q1, Q3; histograms and boxplots by group | Robust summaries, log scales for skewed measures |
| 3. Correlation | **Pearson** `.corr()` with the outcome + heatmap, interpreted by direction and strength | **Spearman**, **Cramér's V** (categories), **point-biserial** (flags); **VIF** for multicollinearity |
| 4. Group rates and interactions | `groupby(feature)[outcome].mean()`; two-way `groupby([A, B])` | Stratified comparisons to check confounding and Simpson's paradox |
| 5. Hypothesis tests | H₀ / H₁, α = 0.05, **Welch's t-test**, assumptions (independence, normality via the central limit theorem, equal variance), sample sizes, practical meaning | **Mann–Whitney (Wilcoxon rank-sum)** / **Kruskal–Wallis** for skewed measures, including EDA_SDA_1's and EDA_SDA_2's gaps, **chi-square** for categories, **log-rank** for survival curves, and its Wilcoxon-weighted version (**Gehan–Breslow–Wilcoxon**) when early differences matter; plain Mann–Whitney never on days open, which is censored; **effect sizes**; **Benjamini–Hochberg** correction across many tests |
| 6. Uncertainty | **Bootstrap** confidence intervals (1,000 resamples, 2.5% / 97.5% quantiles) | **Company-clustered** bootstrap, so listings from one company don't count as independent |
| 7. Time structure | `pivot_table` + heatmap (month × occupation group), `pct_change` | Hands off to `TimeSeries` |
| 8. Survival checks | | Kaplan–Meier with confidence intervals, censoring and left-truncation checks, **competing risks** (filled vs. reposted vs. ghost) described |
| 9. In-depth feature analysis (full data) | | **Stability over months** (each feature's effect per month); **within-company comparisons** (listings from the same company, which removes company-level confounding); **effects by role bucket**; a **leakage audit** (one row per feature: known when the listing first appears?); **drift checks** (distributions over time vs. the collection changes); **information beyond the confounders** (mutual information, likelihood-ratio tests). Priority: stability, within-company and leakage first |
| 10. Feature decisions | Document each decision with data | Each kept / dropped / combined feature recorded with its reason, citing sections 3–9 |
| 11. Final table | | Modeling-ready table, saved for `ML_Rank` |
| 12. Conclusion | What was done, what it means, limitations | |

**Conventions in this notebook.** The outcome is *closing*: `closed` (1 = the listing closed, 0 = still open when collection ended, i.e. censored) with `days_open` as time. `closed7` (closed within 7 days, only for listings first seen early enough to observe 7 days) is the simple rate used in tables, as in EDA_SDA_1 to 3; `closed14` and `closed30` are sensitivity checks, and 14 days is ML_Rank's main window (`CLOSING_DAYS`), so section 4c and 7a show both. Days open is **never** compared with a plain t-test or Mann–Whitney: it is censored, so survival methods (log-rank, Cox) are used. Every test carries an effect size, and p-values across many tests are Benjamini–Hochberg adjusted (section 5).

## 2. Descriptive statistics

Count, mean, median, std, variance, min, Q1, Q3 and max per group, with histograms and boxplots (log scale for skewed measures). Groups: role bucket, source type, company-volume band, month.

**What the data shows**
- **Role buckets differ little in what they ask, and a lot in how long listings stay open.** Among finished listings the median days open is 10.9 (quality assurance) to 14.4 (IT project management) in ten of the twelve buckets, against 20.2 for marketing specialists and 22.5 for technical sales. The median years required is 5 in most buckets, 3 for IT support and marketing specialists and 6 for IT project management; the median description is 524 words (IT support) to 635 (IT project management).
- **Company volume and source are the big splits.** The median days open of finished listings is 28.0 for companies with 1-4 listings, 23.5 (5-49), 16.0 (50-499) and 6.5 (500+); by source, 11.2 for company sites, 15.0 for job boards and 22.1 for hiring platforms. Descriptions are shortest at the smallest companies (median 466 words, against 599 at 50-499) and on job boards (496, against 617 on hiring platforms).
- **By month, finished listings look faster and faster (median 22.1 days in March to May, 2.6 in September), but that is the end of the collection, not faster hiring:** 65% of September listings are still open (8a), and only finished ones are counted here. Days open of finished listings is a lower bound and is never compared across months or groups without the survival methods in section 8.

## 3. Correlation

**Pearson** and **Spearman** of each numeric feature with the 7-day close; **Cramér's V** between categorical factors; **point-biserial** for the 0/1 flags; **VIF** for multicollinearity among the numeric features.

**What the data shows**
- **Nothing numeric is strongly related to closing within 7 days except length.** Spearman with the 7-day close: `words_raw` -0.272, `words_clean` -0.081, `travel_pct` -0.042, `blocker_years_5plus` -0.033, `years_preferred` -0.030, `years_asked` -0.021, `years_required` -0.019 (Pearson is close in every case). `words_raw` covers all 868K listings with a 7-day outcome and `words_clean` the 716K with a description, so the larger value probably mostly reflects listings with no description (8d, `has_desc`) rather than longer text.
- **Every flag is weakly related.** The point-biserial of the 36 flags with the 7-day close is at most 0.027 in size (`degree_mentioned` -0.027, `degree_required` -0.024, `us_citizen_required` +0.021, `evergreen_always_hiring` -0.019). Seven have p above 5%: `evergreen_talent_pool` (10%), `cert_itil` (16%), `cert_comptia` (62%), `changer_wording` (66%), `evergreen_pipeline` (73%), `clearance_required` and `blocker_clearance` (79%).
- **The confounders overlap only in places.** Cramer's V is 31% between source type and month, 30% between role bucket and seniority level, 22% between volume band and month and 20% between source type and volume band; every other pair is 7% or lower. Source and month are tied because job boards stop after May.
- **Collinearity:** `years_asked` has a VIF of 11.4 (above 10), `words_clean` and `words_raw` 6.6 each, `years_preferred` 6.0, `years_required` 5.5, `blocker_years_5plus` 2.2 and `travel_pct` 1.0. `years_asked` overlaps the other two years measures.

_Limit:_ these are associations on 868K listings; with that many rows nearly any non-zero value tests as significant, so the size is what to read.

## 4. Group rates, interactions and Simpson's paradox

Rate of closing within 7 days by factor, two-way by role bucket, and for each flag the crude gap against the gap within strata (company volume, source, role bucket). Opposite signs mean Simpson's paradox: the factor looks one way overall and the other within groups.

**What the data shows**
- **The 7-day close rate is 34% for company-site listings, 29% for job boards and 19% for hiring platforms,** and it climbs with company volume: 15% (1-4 listings), 17% (5-49), 24% (50-499), 43% (500+). The 500+ band is the highest in all 12 role buckets (29% to 49%), so volume is not just a role-bucket effect. By seniority the rate is flat: 28% entry, 28% leadership, 31% manager, 29% mid_senior.
- **By month the rate rises from 25% in April to 33% in September** (17% in March, a partial month of 11K listings), in every bucket but marketing specialists (21% in both months); for example data science 25% to 32%, database 26% to 39%.
- **Within role buckets the source gap holds:** company-site listings close at 20% (technical sales) to 41% (quality assurance), hiring platforms at 9.4% to 20%.
- **Simpson's paradox is rare but real: 13 of the 180 flag-by-confounder pairs reverse sign** between the crude gap and the gap within the strata. The flags involved are `cert_required` and `blocker_cert_required` (within role bucket), `cert_comptia` (source, role bucket), `cert_project` (level), `clearance_required` and `blocker_clearance` (source, month, level), `contract_to_hire` (source) and `evergreen_staffing` (source). For the others the sign holds: `degree_mentioned` is -2.3 points crude and -1.8 to -3.3 within each of the five confounders.

_Limit:_ rates are associations; the saved 4b output shows 10 of its 180 rows (the list of reversals is complete).

### 4c. Sensitivity to the window

Do the rates and flag gaps change if the window is 14 or 30 days instead of 7?

**What the data shows**
- **The role-bucket order does not depend on the window.** At 7, 14 and 30 days the close rate is 16%, 22% and 43% for technical sales and 21%, 30% and 51% for marketing specialists, the two lowest, against 28% to 33% (7 days), 37% to 43% (14) and 55% to 62% (30) for the other ten.
- **Most flag gaps keep their sign across the three windows, but seven do not:** `degree_preferred`, `cert_comptia`, `no_sponsorship` and its twin `blocker_no_sponsorship`, `changer_wording`, `evergreen_talent_pool` and `evergreen_pipeline`. For example `no_sponsorship` goes from -0.9 points (7 days) to +0.4 (14) and +2.8 (30), and `changer_wording` from -0.3 to +1.5 and +5.1.
- **Some gaps grow with the window:** `contract_to_hire` +4.3, +8.5, +10.1 points; `evergreen_always_hiring` -5.6, -7.2, -10.7; `evergreen_no_opening` -4.5, -6.0, -11.5; `sign_on_bonus` -3.7, -4.3, -4.9. `urgency_context` has the largest gap of any flag, +12.7 at 14 days (+7.7 at 7 and +9.4 at 30).
- **Others are steady:** `degree_mentioned` is -2.3, -1.3, -1.5 points, `cert_cisco` -2.8, -2.6, -2.5.

_Read with 14 days, ML_Rank's main window (`CLOSING_DAYS`):_ the 7-day gaps are this notebook's default and the 14-day ones are in the same table; a flag whose sign changes between windows is not safe to read as one effect.

## 5. Hypothesis tests

H₀: the groups do not differ; H₁: they do; α = 0.05. Assumptions: independence (listings of one company are not independent: section 6 widens the intervals for this), large samples (central limit theorem for Welch), and equal variances not assumed. **Welch's t-test** and **Mann–Whitney** compare measures that are fully observed (words, years); **chi-square** compares categories and flags; **log-rank** (and Gehan–Breslow–Wilcoxon, which weights early differences) compares closing curves because days open is censored. Each test reports an effect size (Cohen's d, rank-biserial, Cramér's V, event-rate ratio), and the p-values are Benjamini–Hochberg adjusted across the whole family. This is where EDA_SDA_1's and EDA_SDA_2's descriptive gaps get significance tests.

**What the data shows**
- **All seven numeric features differ between listings that closed within 7 days and the rest, in both tests, after the Benjamini-Hochberg correction, and every difference points the same way:** closed-fast listings have shorter descriptions, ask for fewer years and state less travel. The effects are small except for raw length: Cohen's d is -0.634 for `words_raw`, -0.198 for `words_clean`, -0.089 for `years_preferred`, -0.080 for `blocker_years_5plus`, -0.041 for `years_asked`, -0.035 for `years_required` and -0.033 for `travel_pct`; the rank-biserial values are -0.345, -0.113, -0.042, -0.038, -0.029, -0.027 and -0.061.
- **The five categorical factors are all related to the 7-day close** (chi-square, 868K listings): Cramer's V is 0.2 for volume band, 0.1 for role bucket, source type and month, and 0.0 for level (rounded).
- **The flag tests (log-rank and Gehan-Breslow-Wilcoxon on the full closing curve) are in the 113-row table; the saved output shows ten of its rows,** all significant after correction, with event-rate ratios of 0.9 to 1.1 for the survival tests among them (for example `blocker_no_sponsorship` 1.1, `blocker_clearance` 0.9).
- **Significance is cheap here.** With 868K listings almost every comparison has p near 0, so the effect sizes decide what matters; for an applicant a gap of a point or two is small.

_Limit:_ the tests treat listings as independent; listings of one company are not (section 6), so their p-values are too small.

## 6. Uncertainty: company-clustered bootstrap

1,000 resamples of whole companies (2.5% / 97.5% quantiles), against the plain row-level bootstrap. The ratio of widths is how much clustering by company inflates apparent certainty.

**What the data shows**
- **The overall 7-day close rate is 29% (95% interval 27% to 31%, whole companies resampled, 1,000 draws) across 70K companies.** Resampling companies makes the interval 11.0 times as wide as the plain row-level bootstrap (29% to 29%), so the plain interval overstates certainty by that factor.
- **Per bucket the clustered intervals are wider:** the factor ranges from 3.9 (marketing specialists) to 14.7 (software development), and quality assurance spans 28% to 40%.
- **Only two buckets stand clearly apart:** technical sales (16%, 13% to 19%) and marketing specialists (21%, 19% to 22%) sit below every other bucket, whose lower limits are 25% or higher. The other ten (28% to 33%) overlap each other.

_Limit:_ the bootstrap resamples companies as independent of one another; the plain bootstrap ran on a 200,000-listing sample.

## 7. Time structure

Listings by month and role bucket, and the month-to-month change in the 7-day close rate. The detailed timing analysis is `TimeSeries`.

**What the data shows**
- **Listings per month are steady from April (156K) to September (162K)** with a dip in June (130K); March is a partial month (11K). Software development is the largest bucket every month (38K to 48K from April) and technical sales the smallest (1,219 to 2,024 a month from April).
- **The 7-day close rate rises from 25% in April to 33% in September, mostly in April (+1.1 to +12.9 points per bucket against March) and June (up to +9.2),** then in changes within about 6 points. Marketing specialists are flat (21% in April and September); technical sales reach their highest at 23% in September (+5.9 points).
- **The 30-day rate is flat (54% to 58% from March to August; September has almost no listings with 30 days of follow-up), while the 7-day and 14-day rates rise** (25% to 33% and 35% to 42% from April to September). More quick closes without more closes within 30 days fits a change in how often listings were checked, or in when job boards stopped (after May), as much as a change in hiring; TimeSeries looks at timing, and this notebook does not separate the two.

_Limit:_ the rate for a month depends on how many of its listings had 7 days of follow-up; the last week of September is excluded by construction.

## 8. Survival checks

Censoring (how many listings are still open and when they were first seen), **left truncation** (listings already open when collection began enter the risk set late; ignoring it overstates survival), and **competing risks** (a listing can end by being filled or turn out a ghost; described here, not modeled).

**What the data shows**
- **25% of listings are still open when collection ends (censored), and the share depends on when they were first seen:** 5.9% for March, 6.5% April, 9.0% May, 13% June, 21% July, 32% August and 65% September. By role bucket it is 20% (business and data analysis) to 32% (technical sales). The median days open of finished listings falls from 22.1 (March) to 2.6 (September) for the same reason.
- **Left truncation matters:** listings already open when collection began enter late, and ignoring it overstates survival. The share still open after 7, 14 and 30 days is 86%, 70% and 47% naively, and 76%, 59% and 38% once truncation is accounted for (9 to 11 points lower).
- **Crawl artifacts do not move the main numbers.** Dropping batch first sightings (879K of 908K listings), the first 7 days of collection (876K) or both (848K) leaves the 7-day close rate at 29% to 30% and moves the flag gaps shown by about 0.5 points or less (`degree_mentioned` -2.3 to -2.6, `cert_required` +1.4 to +0.9). The exceptions among the gaps shown are `evergreen_pipeline` (+0.8 to +3.2, +0.1 and +2.6) and `evergreen_staffing` (+4.9 to +6.7, +4.8 and +6.7), rare flags whose gaps are not stable (9a).
- **Competing risks cannot be separated here.** 75% of listings closed and 25% are open; the likely-ghost cause is empty (0.0% at 7, 30 and 90 days) because the cell codes a still-open listing as open first, and ghosts are open listings (EDA_SDA_2). Cumulative incidence of closing is 26% at 7 days, 54% at 30 and 83% at 90 (Aalen-Johansen).

_Limit:_ the censoring is heavy for recent months, so any month comparison needs the survival methods (8d) or the 7-day window.

### 8d. Adjusted check: Cox survival regression

One Cox model with every standard confounder at once (company volume, source, staffing agency, role bucket, seniority, month first seen, company type if available, has a description) plus the features under test, with **company-clustered** standard errors. Hazard ratio above 1 = closes faster. Reported with 95% intervals and a proportional-hazards (Schoenfeld) check. The model is fitted on **all rows** (its hazard ratios are ML_Rank's starting weights) with a small ridge penalizer of 0.01 to keep the fit stable with many dummy columns; with a penalizer the intervals are approximate. The likelihood-ratio tests in 9f use a 300,000-listing sample instead. Which features go in the model is the analyst's call: the lists below are the starting point.

**What the data shows**
- **One Cox model on all rows with 63 covariates (7 confounders as 27 dummy columns, 36 flags) and company-clustered errors.** The saved output shows 10 of its 63 rows; the full table appears once 8d is rerun (display setting added). Hazard ratio above 1 = closes faster.
- **Having a description is the largest effect in the rows shown:** hazard ratio about 0.4 (0.4 to 0.5), p near 0; listings with a description close at less than half the rate of those without, with the other confounders held fixed. This agrees with EDA_SDA_2 (listings without a description close in about 3 days) and with its choice to keep `has_desc` as a standard confounder.
- **Month first seen:** May to September are 0.8 (0.7 to 0.9) against the baseline month (not shown), so given the other factors later months close more slowly, the opposite direction to the crude 7-day rate (25% to 33%, 4a). These outputs do not say why; collection timing and the mix of listings are candidates.
- **The five multi-level factors each matter jointly** (Wald, robust covariance): volume band 58K (3 df), source type 18K (2), month 3,246 (6), role bucket 2,288 (11), level 379 (3).
- **In the flag rows shown the crude degree gaps largely disappear once the confounders are in:** `degree_mentioned` 1.0 (0.9 to 1.1, p 91%), `degree_preferred` 1.0 (0.9 to 1.0, p 30%), `degree_required` 0.9 (0.8 to 1.0, p 3.1%); `is_staffing` is 1.0 (0.9 to 1.2, p 80%).
- **The proportional-hazards assumption fails for company volume:** the Schoenfeld test is 8.6 (p 0.34%) for 5-49, 71.7 for 50-499 and 50.6 for 500+ (p near 0), so those hazard ratios are averages over time. The flag rows shown pass (p 8.1% to 86%).

_Limits:_ the ridge penalizer (0.01) makes the intervals approximate; state, company type and platform are not in the model (this notebook does not load them, or the table lacks them); the other flag hazard ratios are not yet shown, so no claim is made for them.

## 9. In-depth feature analysis (full data)

Priority: 9a stability over months, 9b within-company comparisons, 9d leakage audit first.

### 9a. Stability over months

Does each flag's effect on the 7-day close hold in every month? ML_Rank trains on April–July and tests on September, so a feature that flips sign across months is a risk.

**What the data shows**
- **19 of the 36 flags keep their pooled sign in at least 70% of the months tested, and 6 in every month:** `us_citizen_required` (+1.6 to +7.4 points), `degree_equivalent` (-2.2 to -0.1), `cert_security` (-5.3 to -1.1), `evergreen_no_opening` (-11.0 to -0.5), `evergreen_always_hiring` (-6.9 to -2.3) and `shift_or_oncall` (-2.5 to -0.3).
- **17 are less stable or untestable:** `evergreen_talent_pool` has the pooled sign in 29% of months; `clearance_required`, `cert_itil` and `blocker_clearance` in 43%; `cert_comptia`, `no_sponsorship`, `degree_required`, `degree_preferred`, `blocker_degree_no_equivalent` and `blocker_no_sponsorship` in 57%; `contract_to_hire`, `cert_required`, `blocker_cert_required`, `changer_wording`, `evergreen_staffing` and `hiring_immediately` in 67%. `evergreen_pipeline` has no month with 100+ listings in each group.
- **The rare flags swing the most:** `evergreen_staffing` -4.0 to +15.1 points, `contract_to_hire` -3.7 to +10.0, `urgency_context` -1.1 to +12.6, `changer_wording` -8.2 to +3.9. ML_Rank trains on April to July and tests on September, so these are the features most likely to change meaning between them.

_Limit:_ six or seven monthly gaps per flag; a month with fewer than 100 listings in either group is skipped, so rarer flags are tested on fewer months.

### 9b. Within-company comparisons

Among listings of the same company, does the flag still predict closing? This removes company-level confounding (a company's size, habits, pay).

**What the data shows**
- **For 15 of 36 flags the within-company gap has the opposite sign from the crude gap, or is zero,** which says the crude gap was mostly about which companies use the wording. The largest: `urgency_context` +7.7 points crude, -2.4 within company (289 companies); `evergreen_staffing` +4.9 to -2.1 (226); `contract_to_hire` +4.3 to -2.3 (542); `us_citizen_required` +4.0 to -0.5 (2,074); `hiring_immediately` +2.9 to -1.1 (695); `cert_required` +1.4 to -0.8 (1,927); `relocation_offered` +2.2 to -0.2; `evergreen_many_openings` +1.8 to -1.1. `clearance_required` is 0.0 crude and -2.8 within company.
- **Others hold up inside companies:** `cert_cisco` -2.8 and -2.8, `cert_security` -2.4 and -2.0, `cert_mentioned` -0.8 and -0.9, `entry_wording` +1.2 and +1.3, `evergreen_always_hiring` -5.6 and -3.5, `sign_on_bonus` -3.7 and -6.1, `remote_mention` +0.4 and +0.6 (crude and within company).
- **The degree wording mostly disappears within company:** `degree_mentioned` -2.3 crude, -0.4 within company (19K companies), `degree_equivalent` -1.2 and -0.3.
- Companies used range from 152 (`evergreen_pipeline`) to 21K (`degree_required`); the rare flags rest on a few hundred companies.

_Limit:_ a within-company gap compares listings of the same company, but not of the same role or date; it removes company habits, not everything.

**What the data shows**
- **A few flags point the same way in almost every bucket:** `degree_mentioned` is negative in 10 of 12 (-1.0 to -5.3 points; marketing specialists +0.2 and technical sales +1.4), `evergreen_always_hiring` in 11 of 12 (-0.6 to -9.4; technical sales +2.5) and `us_citizen_required` positive in 10 of 12 (+1.2 to +7.1; data science -1.0 and marketing specialists -3.4).
- **Many change sign by bucket:** `hiring_immediately` -15.0 to +21.2 points, `cert_cisco` +14.1 to -8.2, `cert_security` -7.7 to +6.9, `changer_wording` +15.5 to -3.5, `evergreen_staffing` -3.8 to +12.5. A single pooled effect for these is not a safe feature without the role bucket next to it.
- A bucket is shown only where both groups have 100+ listings; blanks are cells that are too thin, and the technical sales column rests on the fewest listings.

_Limit:_ gaps by bucket are crude within the bucket (not adjusted for volume or source).

### 9d. Leakage audit

One row per feature: is it known when the listing first appears? Anything only known later (how long it stayed open, how many times it was reposted, later edits) must not be a model feature. The answers are filled in by hand; `None` means not yet decided.

**What the data shows**
- **13 of 57 rows are decided and 44 are still "to decide":** the 7 numeric text features, the 36 text flags and `has_desc`. They wait for EDA_SDA_2's 3.9 check (is the description the one seen at first sighting?), which this notebook cannot answer from its outputs; until it is done, text features are provisional for ML_Rank.
- **Known at first sighting (8, marked ok):** `level`, `source_type`, `company_id`, `volume_band`, `first_seen`, `month`, `is_staffing`, `is_repost`. The note on `volume_band` says to check it uses only earlier listings; EDA_SDA_2 builds it from `company_earlier + 1`, which counts earlier listings.
- **Not allowed (5):** `last_seen`, `live`, `closed`, `days_open`, `later_reposts`. `last_seen` and `title` are kept in the final table only as ML_Rank's targets and keyword baseline, never as features.

_Decision for the analyst:_ mark the text rows True or False once the 3.9 check is done; nothing here assumes the answer.

### 9e. Drift checks

Which distributions shift over time, and do they line up with the collection changes (job boards after May, descriptions from June)? Standardized mean difference against the first month, the Kolmogorov–Smirnov distance, and the population stability index (missing counted as its own bin).

**What the data shows**
- **No numeric feature or flag shifts visibly against March (standardized mean difference at most 0.15) except two:** `words_raw` (-0.22, -0.29, -0.25 and -0.33 from June to September) and `has_desc` (-0.51 to -0.56 from June). Both line up with the collection change in June (more listings without a description). The largest other values are `words_clean` +0.15, `remote_mention` +0.15, `travel_pct` -0.14 and `clearance_required` +0.13.
- **The population stability index (blank counted as its own bin) is 24% to 37% from June for the length measures and 28% to 35% for every flag,** about the same as `has_desc` (28% to 34%): that is the blank bin (no description), not a change in the flags' values. Before June it is at most 3.7% (`remote_mention`). The years and travel measures sit lower (0.4% to 9.7% from June).
- **The effect on ML_Rank:** a model trained on April to July sees a different share of blank text than September. `has_desc` should sit next to every text feature, and `words_raw` should stay out.

_Limit:_ March is a partial month (11K listings), so it is a thin reference.

### 9f. Information beyond the confounders

A likelihood-ratio test: does the feature improve a logistic model of the 7-day close that already holds the confounders? Plus mutual information with the outcome, and as a cross-check the **conditional** mutual information within confounder cells, compared with the same statistic after shuffling the outcome inside each cell (a feature whose value is above that noise floor carries information the confounders don't).

**What the data shows**
- **29 of the 36 flags add information beyond the confounders (likelihood-ratio p under 5%, 248K listings); 7 do not:** `cert_cloud` (p 75%), `evergreen_no_opening` (76%), `contract_to_hire` (39%), `changer_wording` (33%), `evergreen_pipeline` (27%), `clearance_required` and `blocker_clearance` (10%).
- **The gains are tiny.** The pseudo-R-squared gain is at most 0.10% (`degree_mentioned`, LR 258.5, odds ratio 0.8), then `degree_required` 0.08%, `evergreen_always_hiring` 0.06% (LR 146.0, odds ratio 0.6) and 0.03% or less for the rest; odds ratios run from 0.6 to 1.4. `urgency_context` has the largest odds ratio (1.4), on few listings.
- **The conditional-information check does not separate features here:** all 43 numeric features and flags are above their shuffle floor, because with this many listings the floor is very small. Read it as "not nothing", not as a ranking.
- **Mutual information with the 7-day close** is highest for `words_clean` (6.6%), `words_raw` (6.5%), `years_preferred` (4.9%), `degree_required` (4.1%) and `degree_mentioned` (4.0%); most flags are under 1%. It fills blanks with the median, so a feature with many blanks partly measures which listings have a description.

_Limit:_ the LR test uses a 300,000-listing sample (248K with an outcome), and all of it is association beyond seven confounders, not an effect.

## 10. Feature decisions

Each feature kept, dropped or combined, with its reason and the section it comes from. Recorded in 10a from sections 3–9: the analyst's call, nothing is decided automatically. The rule used for the flags is stated in the findings below it.

**What the data shows**
- **49 decisions: 22 kept, 1 combined, 26 dropped.** The rule, written after sections 3 to 9 were seen (so a judgment, not a pre-registered test): a flag is **kept** when (1) it adds information beyond the confounders (9f, likelihood-ratio p under 5%), (2) its within-company gap keeps the crude sign (9b) and (3) it has the pooled sign in at least 70% of the months tested (9a). Failing any of these drops it from the model; it stays in EDA_SDA_3's table as a label and opt-in filter, never as a hiding rule. 13 of the 36 flags pass.
- **Kept flags (12):** `degree_mentioned`, `degree_equivalent`, `cert_mentioned`, `cert_cisco`, `cert_security`, `cert_project`, `clearance_obtainable`, `shift_or_oncall`, `entry_wording`, `sign_on_bonus`, `remote_mention`, `evergreen_always_hiring`. **Combined (1):** `evergreen_any`, which stands in for the six evergreen phrase groups.
- **Dropped flags (23):** 7 add nothing beyond the confounders, 15 do not hold within a company and 17 fail on stability (many fail more than one); the three `blocker_` duplicates (`blocker_clearance`, `blocker_no_sponsorship`, `blocker_cert_required`) have identical statistics to the flag they copy. `degree_required`, `no_sponsorship`, `us_citizen_required`, `clearance_required`, `changer_wording` and `contract_to_hire` remain labels for the app.
- **Numeric (7):** kept `words_clean`, `years_required`, `years_preferred`, `travel_pct`, `blocker_years_5plus`; dropped `words_raw` (repeats `has_desc`, drifts) and `years_asked` (VIF 11.4).
- **Controls:** kept `has_desc`, `role_bucket`, `source_type`, `volume_band`, `level`; dropped `is_staffing` (hazard ratio 1.0, p 80%).
- **Carried, not tested here:** `salary_hourly_usd` and `n_subsidiaries` are passed through to the final table for ML_Rank's pay factor and its placeholder features; nothing in sections 3 to 9 tests them. The company-history features EDA_SDA_2 3.13 forwarded (for example the company's past closing speed) are not in this notebook's tables either; ML_Rank computes the company counts it uses itself.

_Limits:_ the cut-offs (5%, 70%, the sign of the within-company gap) are mine; the gaps are associations of a point or two; and 9 of the 36 flags rest on fewer than 1,000 companies in 9b, including a kept one (`sign_on_bonus`, 439).

## 11. Final table

One row per scope listing (`id`) with the kept and combined features, plus the identifiers and outcome columns the next notebooks need (`title` for ML_Rank's keyword baseline, `last_seen` for its targets; neither is a feature). Saved as `modeling_ready.parquet` in `data/clean/` (`PART4_TABLE`) for `ML_Rank` (`TimeSeries` and `ML_Fit` read the scope and text tables); the name is this notebook's choice, since EDA_SDA_3 hands off `text_table.parquet` and the scope table (joined by `id`).

## 12. Conclusion

### What was done
- **Setup:** 908,291 scope listings from 71,919 companies (EDA_SDA_2's table joined to EDA_SDA_3's text features), 75.3% closed; 7 text numeric features and 36 text flags vetted. Backgrounds and the team's own résumés play no part here: the outcome is the listings' own closing.
- **Strongest drivers are the confounders, not the text.** The 7-day close rate is 15% (companies with 1-4 listings) to 43% (500+), 34% for company-site listings against 19% for hiring platforms, and 16% to 33% by role bucket (technical sales and marketing specialists lowest, clearly apart in the company-clustered intervals). In the Cox model all five multi-level factors matter jointly, and having a description is the largest single effect in the rows shown (hazard ratio about 0.4).
- **Text features carry little beyond them.** Flag correlations with the 7-day close are 0.027 or less; the likelihood-ratio gains are at most 0.10% of pseudo-R-squared; 15 of the 36 flags change sign or vanish within a company; 13 of 180 flag-by-confounder checks reverse; 17 flags are unstable across months. The longest description is the strongest numeric feature (Cohen's d -0.198 for `words_clean`).
- **Survival checks:** 25% of listings are censored (65% of September's), left truncation overstates survival by 9 to 11 points if ignored, and crawl artifacts barely move the main rates. Clustering by company widens the overall interval 11-fold.
- **Decisions:** 22 kept, 1 combined, 26 dropped (section 10): 12 flags plus `evergreen_any`, 5 numeric features, `has_desc` and four controls. The table is saved as `modeling_ready.parquet`.

### Answers to the clarifying questions
| Question | Answer |
|:--|:--|
| Which features are redundant? | `years_asked` (VIF 11.4) overlaps the two other years measures; `words_raw` repeats `has_desc`; three `blocker_` flags are identical to the flags they copy. Among the confounders, role bucket and seniority (Cramer's V 30%) and source and month (31%) overlap. |
| Which differences are real and large enough to matter? | Most are real at this size (868K listings) and small: flag gaps of 1 to 6 points and pseudo-R-squared gains of 0.10% or less. Company volume, source and having a description are large. |
| Do effects hold within other factors, or reverse (Simpson)? | Mostly the sign holds: 13 of 180 pairs reverse (`cert_required`, `cert_comptia`, `cert_project`, `clearance_required`, `contract_to_hire`, `evergreen_staffing` and the `blocker_` copies). Within company, 15 flags lose their sign. |
| Adjusted check (Cox): which factors still matter? | Volume, source, role bucket, level and month each matter jointly; `has_desc` is the largest effect. In the rows shown, the degree flags (hazard ratio 0.9 to 1.0) and `is_staffing` (1.0) add little. The other flag hazard ratios need the rerun of 8d. |
| How much does clustering by company inflate certainty? | 11.0 times for the overall rate (29%, 27% to 31%) and 3.9 to 14.7 by role bucket. |
| Do the career-change flags carry signal beyond seniority, size and source? | `entry_wording` does, weakly (LR 4.7, p 3.0%; within-company +1.3 points, same sign in 5 of 7 months): kept. `changer_wording` does not (LR 1.0, p 33%): dropped. `career_open` is loaded in 1b but not tested in any section, so its parts are still open. |
| Do effects differ by role bucket? | For most flags yes: `hiring_immediately`, `cert_cisco`, `cert_security` and `changer_wording` change sign across buckets; `degree_mentioned` and `evergreen_always_hiring` are negative in 10 and 11 of 12. |
| Are the survival assumptions reasonable? | Censoring and left truncation are handled where they matter (8a, 8b); proportional hazards fails for company volume (8d), so its hazard ratios are averages over time. |
| Is each feature stable? | 19 of 36 flags keep their sign in 70% or more of the months; 6 in every month. The 13 kept flags pass this and the other two checks. |
| Is each feature safe? | Structure and source features are known at first sighting. The 44 text-derived rows are still "to decide" in 9d (EDA_SDA_2 3.9 check); `has_desc` and `words_raw` drift with the June collection change. |

### Limits
- Everything is an association on observational data; no row of this notebook shows what causes a listing to close fast. The adjusted checks (Cox, within-company, stability) reduce confounding by the factors they contain, not by others.
- The Cox model adjusts for 7 confounders; state, company type and platform are not in it. Its volume effect violates proportional hazards, and the saved output shows only 10 of its 63 hazard ratios.
- The decision rule (5%, 70% of months, within-company sign) is a judgment made after seeing the results, and the kept effects are of a point or two.
- The text flags are unvalidated keyword rules (EDA_SDA_3); they are labels and opt-in filters, never hiding rules. Dropping a flag here removes it from the model only.
- March is a partial month; September is mostly censored; the 7-day outcome needs 7 days of follow-up.
- This notebook tests the text features and the confounders. The 48 non-text features EDA_SDA_2 3.13 forwarded (company history and the others) are not re-tested here, and only `salary_hourly_usd` and `n_subsidiaries` are carried.

### Handoff
- **Saved:** `modeling_ready.parquet` (`PART4_TABLE`) in `data/clean/`: one row per scope listing with `id`, `company_id`, `title`, `first_seen`, `last_seen`, `month`, `role_bucket`, `days_open`, `live`, `closed` (identifiers and outcome, never features), the kept and combined features of section 10, and `salary_hourly_usd` and `n_subsidiaries` (carried, untested).
- **To `ML_Rank`:** the model features are the kept numeric measures (`words_clean`, `years_required`, `years_preferred`, `travel_pct`, `blocker_years_5plus`), the kept flags (as 0/1 with blank for no description, next to `has_desc`), and the kept controls `role_bucket`, `source_type`, `volume_band`, `level`. Leave out `words_raw`, `years_asked` and the dropped flags. Compute company counts from earlier listings only, as ML_Rank does. The Cox hazard ratios here (closing speed) do not set the priority queue's four weights; those stay starting values tuned on August.
- **`TimeSeries` and `ML_Fit`** read the scope and text tables, not this one; nothing here changes them.
- **Open:** rerun 1a, 1b, 4b, 5b, 8d, 9d, 10a and 11a (display settings and the decision log are new), then refresh sections 4, 5 and 8d from the full tables; mark the 9d text rows once the 3.9 check is done; decide on testing `career_open` and the 48 forwarded features.
