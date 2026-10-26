# Simpson's paradox

What the check in `EDA_SDA_4` section 4b asks, and how to read its result. Part 1 is the paradox on its own, the
way a statistics textbook introduces it. Part 2 applies it to this project. Nothing else needs reading first. The
code it describes is `stratified_gap` in [`../src/stats_tests.py`](../src/stats_tests.py), and the worked example
below is arithmetic you can check by hand.

## Quick summary (for explaining this out loud)

A factor can look **helpful overall and harmful in every single subgroup** - or the reverse. That is Simpson's
paradox. It is not a trick of the data or a mistake in the arithmetic: both statements are true at once, about
different questions.

It happens because a pooled rate is a **weighted average**. When two groups are spread differently across
subgroups that have very different base rates, comparing their pooled rates partly measures *which subgroup
people are in* rather than *which group they belong to*.

- **The crude gap** compares the two groups as they actually are, subgroup mix included.
- **The adjusted gap** compares them as if both had the same subgroup mix: work out the gap inside each subgroup,
  then average those gaps with one common set of weights.
- **When the two have opposite signs, that is the paradox.** `stratified_gap` reports exactly this and calls it
  `reverses`.

What to do about it: the adjusted gap answers "what if the subgroup mix were the same", which is usually the
question you meant to ask - but only when the subgroup is **not a consequence** of the thing being compared (see
the caveat at the end of Part 1).

## Part 1: The paradox, in general

### The question

Treatment A succeeds in 78% of cases, treatment B in 83%. B looks better. Then someone splits the cases by how
hard they were, and A wins in the easy cases *and* in the hard ones. Which treatment is better?

### Why it happens: a pooled rate is a weighted average

Write $r_s^A$ for A's success rate in subgroup $s$, and $w_s^A$ for the share of A's cases that fall in subgroup
$s$. A's pooled rate is

$$r^A = \sum_s w_s^A \, r_s^A$$

and likewise for B. The crude gap is $r^A - r^B$. Two things vary between the groups: the rates $r_s$ and the
weights $w_s$. So the crude gap mixes together "how well A does in each subgroup" and "which subgroups A's cases
came from".

The fix is to use **one common set of weights** for both groups - a *standardized*, or adjusted, gap:

$$\text{adjusted} = \sum_s w_s \,(r_s^A - r_s^B), \qquad w_s = \frac{n_s}{\sum_t n_t}$$

where $n_s$ is the total size of subgroup $s$ across both groups. Every term is now a within-subgroup comparison,
so the subgroup mix cannot leak in.

### A small worked example

The classic case is Charig et al. (1986), two treatments for kidney stones:

| | small stones | large stones | overall |
|:--|:--|:--|:--|
| **A** (open surgery) | 81/87 = **93%** | 192/263 = **73%** | 273/350 = **78%** |
| **B** (keyhole) | 234/270 = **87%** | 55/80 = **69%** | 289/350 = **83%** |

A wins on small stones (93 vs 87) and on large stones (73 vs 69), yet loses overall (78 vs 83).

The weights explain it. Of A's 350 cases, 263 were large stones - the hard ones - so A's pooled 78% is mostly its
large-stone rate. Of B's 350 cases, 270 were small - the easy ones - so B's pooled 83% is mostly its small-stone
rate. The doctors chose the keyhole procedure for easier cases, and the pooled comparison is reading that choice.

Now standardize. Across both treatments there are $87 + 270 = 357$ small-stone cases and $263 + 80 = 343$ large,
so $w_{\text{small}} = 357/700 = 0.51$ and $w_{\text{large}} = 343/700 = 0.49$:

$$\text{adjusted} = 0.51 \times (93 - 87) + 0.49 \times (73 - 69) = 3.1 + 2.0 = +5.0 \text{ points for A}$$

against a crude gap of $78 - 83 = -5.0$ points for A. Same data, opposite sign. The other canonical example is
the 1973 Berkeley graduate admissions figures, where the university appeared to favour men overall while most
individual departments admitted women at a slightly higher rate - because women applied in greater numbers to the
departments that admitted fewest people of any gender.

### When it can happen, and when it cannot

Both of these must hold:

1. the subgroups have **different outcome rates** (easy and hard stones are not equally survivable), and
2. the two groups are **spread differently** across the subgroups (A took the hard cases).

If either fails, the crude and adjusted gaps come out close together. This is why a randomized trial is largely
immune: randomization makes the two groups' subgroup mixes the same by construction, so $w_s^A \approx w_s^B$ and
there is nothing for the weights to distort. Observational data has no such protection.

### The caveat that matters most

Adjusting is not automatically right. Only adjust for a subgroup variable that is **not a consequence** of the
thing you are comparing. If the treatment changes which subgroup you land in - the subgroup sits *downstream* of
the treatment, as a mediator - then holding it fixed removes part of the real effect, and the adjusted number is
the misleading one. "Adjust for everything" is not a safe default; "adjust for what was already true beforehand"
is.

## Part 2: Applying it here

Section 4b runs this check for every candidate flag against each of five confounders. The outcome is `closed7`
(the listing closed within 7 days), the two groups are listings with and without the flag, and the strata are
`volume_band`, `source_type`, `role_bucket`, `month` and `level`. That is five of the eight confounders 8d's Cox
model holds at once; platform, staffing agency and whether the listing has a description are left to the model.
With 36 text flags, five confounders make the 180 checks the conclusion counts.

```python
# 4b Simpson's paradox check: crude vs. stratum-weighted gap for each flag, within each confounder
CONF = [c for c in ["volume_band", "source_type", "role_bucket", "month", "level"] if c in jobs]
rows = []
for f in TEXT_FLAGS:
    for s in CONF:
        _, summ = stx.stratified_gap(jobs, "closed7", f, s)   # summ: crude_gap, adjusted_gap, reverses
        rows.append({"flag": f, "within": s, **summ})
simpson = pd.DataFrame(rows)
```

Condensed from `stratified_gap` itself, one line per term of the formula above:

```python
mean, size = _mean_size_by_flag(d, outcome, flag, strata)   # r_s and n_s, per stratum, for flag on and off
tab["gap"] = mean[True] - mean[False]                       # the within-stratum gaps, r_s^A - r_s^B
w = ok.n_flag + ok.n_other                                  # the common weights n_s
adjusted = (ok.gap * w).sum() / w.sum()                     # the standardized gap
crude = d.loc[d[flag], outcome].mean() - d.loc[~d[flag], outcome].mean()
reverses = np.sign(crude) != np.sign(adjusted)              # the paradox
```

Two details worth knowing. `closed7` is blank for listings first seen too late for seven days to be observed, so
the check runs only on listings old enough to have an answer (`dropna`). And a stratum counts toward the adjusted
gap only if it holds both flagged and unflagged listings - a stratum with one kind has no gap to contribute.

### How to read it

- **Both conditions for the paradox hold in this data,** which is why the check is here rather than decorative.
  The confounders have very different close rates - `EDA_SDA_4`'s run put the 7-day rate at 15% for companies with
  1 to 4 earlier listings against 43% for 500+, and 34% for company-site listings against 19% for hiring
  platforms - and the text flags are spread unevenly across them. That is exactly the kidney-stone setup.
- **A reversal does not mean the flag is useless.** It means the crude gap is not evidence *about the flag*: the
  gap belongs to the confounder. `EDA_SDA_4`'s run found 13 of 180 flag-by-confounder checks reverse.
- **No reversal is not a clean bill of health either.** 4b adjusts one confounder at a time, so another can still
  be mixed in - and three of the eight are not among its strata at all. The stronger versions are 8d's Cox model,
  which holds all eight at once, and 9b, which uses the company itself as the stratum: the narrowest confounder
  available, and the one that absorbs everything about a company without having to name it.
- **4b decides nothing.** Keep and drop calls are made in 10a by the screening rule `EDA_SDA_2` fixed before its
  own run; 4b is evidence put in front of that rule, not a decision.

## Left open, worth learning on your own

- **Continuous confounders have to be binned** before they can be strata, and the bin edges are a choice that can
  change the answer. Here the confounders arrive already categorical (`volume_band`, `month`, `level`).
- **The mirror image is the ecological fallacy:** reading a group-level relationship as if it held for
  individuals. Simpson's paradox is the same arithmetic seen from the other end.
- **Direct standardization versus regression.** The adjusted gap above is direct standardization with one
  confounder. A model does the same job for several at once and borrows strength across strata, at the cost of
  assuming a shape for the relationship - which is the trade 8d makes.
- **Which weights.** Standardizing to the overall population is one choice; standardizing to one group's mix, or
  to an external reference population, answers slightly different questions.
