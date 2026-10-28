# Gehan-Wilcoxon test

What the test in `EDA_SDA_1_Lean` section 5c asks, and how to read its result. Part 1 is the method on its own,
the way a survival-analysis textbook introduces it. Part 2 applies it to this project. It builds on
[`kaplan_meier.md`](kaplan_meier.md): read that first if "survival curve" or "censored" is new. It was run with
`lifelines.statistics.logrank_test(..., weightings="wilcoxon")`, and the worked example below was checked against
it.

## Quick summary (for explaining this out loud)

Kaplan-Meier gives one curve per group (here, the share of listings still open after each day). The Gehan-Wilcoxon
test asks: **could these curves differ this much by chance alone, if the groups really closed at the same rate?**
It answers with a p-value, a probability, not a size. A tiny p-value means "this gap is very unlikely to be luck."
It does not mean the gap is big. The size is the difference between the curves themselves (in 5c: a median of
12 days at companies with 500+ earlier listings vs. 44 at companies with 1 to 4, with 34 and 28 in between).
It can compare two groups, all groups at once ("do any of them differ?"), or every pair of groups ("is each step
real?"); 5c does the last two.

- **Why not just compare the two medians:** a median says nothing about whether a gap that size would turn up by
  chance, and a plain test on days open (a t-test, say) would throw away the still-open listings, the same mistake
  as averaging only closed listings (`kaplan_meier.md`).
- **Why this version of the test:** it is a *log-rank test* that gives the most weight to early differences, when
  the most listings are still being watched. Listings close fast in this data (29% within a week), so early days
  are where most of the information is.
- **What it needs:** the censoring (still open on September 30) shouldn't depend on which group a listing is in. In
  5c it doesn't: every listing is cut off by the same date.
- **What it is here:** a preview. The formal comparison, with the other confounders held fixed, is `EDA_SDA_4`'s
  Cox model.

## Part 1: The method, in general

### The question

Two groups of subjects, each with an event time (some censored). Their Kaplan-Meier curves $S_1(t)$ and $S_2(t)$
differ. The **null hypothesis** is that they are the same curve: $S_1(t) = S_2(t)$ at every $t$. The test asks how
surprising the observed difference would be if the null were true.

### The idea: compare observed events with expected events, day by day

Take every distinct day $t_i$ on which at least one event happened (censoring days don't get a term). On that day:

- $n_i$ = subjects at risk in both groups together, just before $t_i$.
- $n_{1i}$ = how many of those are in group 1.
- $d_i$ = events in both groups on $t_i$, and $d_{1i}$ = events in group 1.

If the groups were the same, the $d_i$ events would fall on group 1 in proportion to its share of the risk set. So
group 1 would be **expected** to have

$$e_{1i} = d_i \cdot \frac{n_{1i}}{n_i}$$

events on that day, and it **actually** had $d_{1i}$. The difference $d_{1i} - e_{1i}$ is positive on a day when
group 1 loses more than its share. Over many days, a group that really closes faster keeps running positive.

Each day's contribution is multiplied by a **weight** $w_i$ and the weighted differences are added:

$$U = \sum_i w_i \,(d_{1i} - e_{1i})$$

Under the null, $U$ is centred on zero. Its variance has a standard form (from counting how the $d_i$ events could
fall by chance among those at risk):

$$V = \sum_i w_i^2 \, d_i \,\frac{n_{1i}}{n_i}\!\left(1 - \frac{n_{1i}}{n_i}\right)\!\frac{n_i - d_i}{n_i - 1}$$

(a day with $n_i = 1$ contributes 0). The test statistic is $U^2 / V$, compared against a chi-square distribution
with 1 degree of freedom. A large value means the observed gap is far from what chance would give, which gives a
small p-value.

**More than two groups.** With $k$ groups the same idea runs on a vector: for each group, observed minus expected
events, weighted and summed, and the statistic is compared against a chi-square distribution with $k - 1$ degrees of
freedom. It tests the null that *all* $k$ curves are the same, so it can reject without saying which group is
different. To see which, run the two-group test on every pair ($k(k-1)/2$ tests). The more tests you run, the more
likely one looks significant by chance alone, which matters when p-values are near 0.05 and hardly at all when they
are far below it.

### The weights are what name the test

| Weights $w_i$ | Test | Gives most say to |
|---|---|---|
| $1$ | log-rank | every event day equally, so late days count as much as early ones |
| $n_i$ | **Gehan-Breslow-Wilcoxon** (this file) | early days, when $n_i$ is largest |

So the Gehan-Wilcoxon test is the log-rank test with each day weighted by how many subjects were at risk that day.
It is the survival version of the Wilcoxon rank-sum test (which compares two groups by ranks and not by averages).
It reads early differences more strongly than the log-rank test and is less affected by the thin tail at long
times, where few subjects remain.

### A small worked example

Group A has events on days 2, 3 and 5. Group B has events on days 4, 6 and 7. Nobody is censored.

| $t_i$ | $n_i$ | $n_{1i}$ | $d_i$ | $d_{1i}$ | expected $e_{1i}$ | $d_{1i}-e_{1i}$ | variance term | weight $w_i=n_i$ |
|---|---|---|---|---|---|---|---|---|
| 2 | 6 | 3 | 1 | 1 | 0.500 | +0.500 | 0.2500 | 6 |
| 3 | 5 | 2 | 1 | 1 | 0.400 | +0.600 | 0.2400 | 5 |
| 4 | 4 | 1 | 1 | 0 | 0.250 | −0.250 | 0.1875 | 4 |
| 5 | 3 | 1 | 1 | 1 | 0.333 | +0.667 | 0.2222 | 3 |
| 6 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| 7 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |

- $U = 6(0.5) + 5(0.6) + 4(-0.25) + 3(0.667) = 7.0$
- $V = 6^2(0.25) + 5^2(0.24) + 4^2(0.1875) + 3^2(0.2222) = 9 + 6 + 3 + 2 = 20.0$
- Statistic $= U^2 / V = 49 / 20 = 2.45$, so $p \approx 0.12$.

Group A tends to close sooner, but with six subjects that is within what chance gives, so the test does not call it.
The same data with equal weights (log-rank) gives 2.56 and $p \approx 0.11$.

## Part 2: Applying it here

In section 5c the groups are four **company volume bands**: listings at companies with 1 to 4, 5 to 49, 50 to 499 and
500 or more listings, counting only the listings the company had *before* each one (`company_earlier`), which is known
when the listing appears. The events are closes, and still-open listings are censored. Two runs:

```python
# 5c-3: do any of the four bands differ? (the chart's question)
all_bands = multivariate_logrank_test(sample.days_open, sample.band, ~sample.live, weightings="wilcoxon")

# 5c-4: is each step real? (all six pairs of bands)
pairs = pairwise_logrank_test(sample.days_open, sample.band, ~sample.live, weightings="wilcoxon").summary
```

- The third argument is `~live`: a listing that is no longer live closed (event); a live one is censored.
- `weightings="wilcoxon"` is the $w_i = n_i$ weighting above.
- `sample` holds up to 500,000 random listings per band (`random_state=0`) to keep it fast; the curves in the chart use
  every listing.

### How to read it

- **The chi-square is the test statistic $U^2/V$.** 0 means identical curves; bigger means a bigger gap, on a scale
  with no units. To read it, compare it with the chi-square cutoff for its degrees of freedom (one fewer than the
  number of groups): for one pair (1 degree of freedom), 3.84 is p = 0.05 and 10.83 is p = 0.001; for four groups (3
  degrees of freedom), 7.81 and 16.27.
- **The chi-square and p-value say whether the gaps are real, not how big they are.** With hundreds of thousands of
  listings per band, even a small gap gives a tiny p-value. What matters is the size: medians of 44, 34, 28 and 12
  days.
- **The all-bands test can say "something differs" without saying what.** The pairwise table fills that in: it shows
  whether each step (1-4 to 5-49, 5-49 to 50-499, 50-499 to 500+) is real, not only the two ends. Six tests means one
  could look significant by chance; the p-values in 5c are far from that worry.
- **It does not say company volume *causes* faster closing.** Large companies differ in other ways (source, role mix,
  how they post). That is what `EDA_SDA_4`'s Cox model, which holds the other factors fixed, is for.

## Left open, worth learning on your own

- **Why the multi-group test is one test, not six.** It controls the chance of a false alarm across all groups at once; the pairwise table does not, which is why it carries the multiple-tests caution.
- **Why the weights matter in practice.** The log-rank test is most powerful when the curves differ by a constant
  ratio over time (proportional hazards). `EDA_SDA_4` finds that assumption fails for company volume, which is one
  reason to also report an early-weighted test like this one.
- **Other weights.** Fleming-Harrington weights let you choose how much to favour early or late days.
