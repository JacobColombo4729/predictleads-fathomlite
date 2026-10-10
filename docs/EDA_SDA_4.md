<!-- Generated from notebooks/EDA_SDA_4.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 4 (Statistics)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

Vets every candidate feature, non-text (EDA_SDA_2) and text (EDA_SDA_3), with statistical evidence, and hands a **modeling-ready table** to the `TimeSeries` and `ML` notebooks.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, and what's the scope? | Final table 1 |
| EDA_SDA_2 | Which listings do we study? Non-text checks, ghost listings, the career-change target | Niche table, non-text features |
| EDA_SDA_3 | What features does the text add? (NLP) | Niche table + text features |
| EDA_SDA_4 | Which features hold up? (statistics) | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing patterns and features |
| ML | How should listings be ranked for a career changer? | The prioritization model |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

### Clarifying questions
- Which features are redundant with each other (collinearity), and which genuinely overlap?
- Which differences between groups are real, and which are large enough to matter to an applicant?
- Do factor effects hold within each other (for example source within company-volume bands), or reverse (Simpson's paradox)?
- **Adjusted check:** do the factors that survived EDA_SDA_2's within-group checks still matter in one **Cox survival regression** that adjusts for all the confounders at once (company volume, source and platform, staffing agency, role bucket, seniority, month first seen), with errors clustered by company? Report hazard ratios with 95% intervals, and check the proportional-hazards assumption.
- How much does clustering by company inflate apparent certainty?
- Do the career-change flags carry signal beyond seniority, company size and source, and which parts of `career_open` hold up?
- Do factor effects differ by role bucket, or hold across them?
- Are the survival assumptions reasonable (censoring, left truncation)?

**Data rules:** aggregates only; never `df.head()` or sample rows. Clear outputs before committing.

### Roadmap and methods

| Section | Course standard | Taken further |
|:--|:--|:--|
| 1. Setup | Load EDA_SDA_3's table | Shared helpers in `src/` |
| 2. Descriptive statistics | Per-group count, mean, median, std, variance, min, max, Q1, Q3; histograms and boxplots by group | Robust summaries, log scales for skewed measures |
| 3. Correlation | **Pearson** `.corr()` with the outcome + heatmap, interpreted by direction and strength | **Spearman**, **Cramér's V** (categories), **point-biserial** (flags); **VIF** for multicollinearity |
| 4. Group rates and interactions | `groupby(feature)[outcome].mean()`; two-way `groupby([A, B])` | Stratified comparisons to check confounding and Simpson's paradox |
| 5. Hypothesis tests | H₀ / H₁, α = 0.05, **Welch's t-test**, assumptions (independence, normality via the central limit theorem, equal variance), sample sizes, practical meaning | **Mann–Whitney** / **Kruskal–Wallis** for skewed measures, **chi-square** for categories, **log-rank** for survival curves; **effect sizes**; **Benjamini–Hochberg** correction across many tests |
| 6. Uncertainty | **Bootstrap** confidence intervals (1,000 resamples, 2.5% / 97.5% quantiles) | **Company-clustered** bootstrap, so listings from one company don't count as independent |
| 7. Time structure | `pivot_table` + heatmap (month × occupation group), `pct_change` | Hands off to `TimeSeries` |
| 8. Survival checks | | Kaplan–Meier with confidence intervals, censoring and left-truncation checks, **competing risks** (filled vs. reposted vs. ghost) described |
| 9. Feature decisions | Document each decision with data | Each kept / dropped / combined feature recorded with its reason |
| 10. Final table | | Modeling-ready table, saved for `TimeSeries` and `ML` |
| 11. Conclusion | What was done, what it means, limitations | |

## 2.–11.

_To do, once EDA_SDA_3's table exists._
