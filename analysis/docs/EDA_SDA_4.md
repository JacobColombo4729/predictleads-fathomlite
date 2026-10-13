<!-- Generated from notebooks/EDA_SDA_4.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 4 (Statistics)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

Vets every candidate feature, non-text (EDA_SDA_2) and text (EDA_SDA_3), with statistical evidence, and hands a **modeling-ready table** to `TimeSeries`, `ML_Fit` and `ML_Rank` notebooks.

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
- **Adjusted check:** do the factors that survived EDA_SDA_2's within-group checks still matter in one **Cox survival regression** that adjusts for all the confounders at once (company volume, source and platform, staffing agency, role bucket, seniority, month first seen, state, company type, has a description), with errors clustered by company? Report hazard ratios with 95% intervals, and check the proportional-hazards assumption.
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
| 11. Final table | | Modeling-ready table, saved for `TimeSeries`, `ML_Fit` and `ML_Rank` |
| 12. Conclusion | What was done, what it means, limitations | |

## 2.–12.

_To do, once EDA_SDA_3's table exists._
