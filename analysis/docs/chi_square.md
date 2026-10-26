# Chi-square test of independence

What the test in `EDA_SDA_4` section 5b asks, and how to read its result. Part 1 is the test on its own, the way
a statistics textbook introduces it. Part 2 applies it to this project. Shorter than its siblings
([`kaplan_meier.md`](kaplan_meier.md), [`gehan_wilcoxon.md`](gehan_wilcoxon.md),
[`simpsons_paradox.md`](simpsons_paradox.md)) because the test itself is short. Every number below was checked
against `scipy.stats.chi2_contingency` on the made-up table it uses.

## Quick summary (for explaining this out loud)

Two yes/no questions - *was this listing flagged?* and *did it close within 7 days?* - give a 2×2 table of
counts. The chi-square test asks: **if the two questions were unrelated, how far from this table would we
expect to land by chance?** It compares the counts you observed with the counts independence predicts, and
returns a p-value.

It says nothing about **size**. On a 2×2 table it is exactly the square of the two-proportion z-test, so the
thing it is testing is the gap between two rates - and that gap, in percentage points, is what to quote. The
effect size that travels with it is **Cramér's V** (0 = unrelated, 1 = perfectly determined).

## Part 1: The test, in general

### The visual: observed against expected

A made-up example. 1,000 listings, 200 of them flagged; 270 closed within a week:

```text
OBSERVED                            EXPECTED  (if the flag made no difference)
              closed   open                        closed   open
flagged           70    130  = 200   flagged           54    146  = 200
not flagged      200    600  = 800   not flagged      216    584  = 800
               =270   =730  =1000                   =270   =730  =1000

closed within 7 days
  flagged      |##################       35%
  not flagged  |#############            25%     gap = 10 points
```

Each expected count is the row total times the column total, divided by the grand total:
$E = 200 \times 270 / 1000 = 54$. The row and column totals of the two tables are identical by construction -
only the way the counts are *distributed inside* them differs. Chi-square measures that difference.

### The arithmetic

$$\chi^2 = \sum \frac{(O - E)^2}{E}, \qquad \text{df} = (\text{rows} - 1)(\text{columns} - 1)$$

| cell | $O$ | $E$ | $(O-E)^2/E$ |
|:--|--:|--:|--:|
| flagged, closed | 70 | 54 | 4.74 |
| flagged, open | 130 | 146 | 1.75 |
| not flagged, closed | 200 | 216 | 1.19 |
| not flagged, open | 600 | 584 | 0.44 |
| | | | **8.12** |

With df = 1, the 5% cutoff is 3.84 and the 0.1% cutoff is 10.83, so $\chi^2 = 8.12$ gives p = 0.0044: a gap this
size would be unlucky if the flag truly made no difference. The matching effect size is

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}} = \sqrt{8.12/1000} = 0.09$$

which is **tiny**. Both statements are true at once, and that is the whole lesson of this test.

Two details that come up in practice. For a 2×2 table `scipy` applies **Yates' continuity correction** by
default, which shrinks each $|O-E|$ by 0.5 and gives 7.62 instead of 8.12 here (p = 0.0058); it matters only for
small tables. And the two-proportion z-test on the same table gives z = 2.849, whose square is 8.118 - the same
number, as promised.

### What it does not tell you

Multiply every count in that table by 100 - same percentages, same 10-point gap, 100,000 listings:

| | $\chi^2$ | p | Cramér's V | gap |
|:--|--:|--:|--:|--:|
| n = 1,000 | 8.1 | 0.004 | 0.090 | 10 points |
| n = 100,000 | 811.8 | 1.5 × 10⁻¹⁷⁸ | 0.090 | 10 points |

$\chi^2$ grows in proportion to $n$; V and the gap do not move at all. **A small p-value is partly a statement
about how much data you have.** This is why the significance of a test on hundreds of thousands of listings
carries almost no information, and why every table in section 5 reports an effect size beside the p-value.

Assumptions worth knowing: the expected count in every cell should be at least about 5 (easily met at this
scale), and the observations must be independent.

## Part 2: Applying it here

In 5b the two questions are a text flag and `closed7`:

```python
chi = stx.chi_square(on, d.closed7)                  # on = the flag as a boolean; effect is Cramer's V
rate_on, rate_off = d.loc[on, "closed7"].mean(), d.loc[~on, "closed7"].mean()
# reported side by side: the two rates, the gap in points, chi-square's p, and V
```

`chi_square` ([`../src/stats_tests.py`](../src/stats_tests.py)) builds the contingency table, calls
`scipy.stats.chi2_contingency`, and returns Cramér's V without the small-sample bias correction.

### How to read it

- **Read the gap first, then V, then p.** The gap in percentage points is what an applicant would feel; V says
  how strongly the two questions are tied; p only says whether a gap that size could be chance, and at this
  sample size it essentially always says no. `EDA_SDA_4`'s run found no flag's correlation with the 7-day close
  above 0.027 in size - flags move the rate by a point or two.
- **The independence assumption is broken here, and knowingly so.** Listings from one company are not
  independent of each other, so these p-values are too small. Section 6's company-clustered bootstrap is the
  honest version of the same uncertainty, and 8d's Cox model clusters its errors by company for the same reason.
- **A significant chi-square is not a reason to keep a feature.** 10a decides by the screening rule `EDA_SDA_2`
  fixed before its run; 5b is evidence placed in front of that rule.

## Left open, worth learning on your own

- **Fisher's exact test** replaces chi-square when expected counts are very small - not a situation this data
  produces.
- **Bigger tables than 2×2** give a single p-value for "something in here is related", with no direction; the
  per-cell contributions above (standardized residuals, properly) are how you find where the relationship sits.
- **McNemar's test** is the version for paired yes/no data, such as the same listing measured twice.
