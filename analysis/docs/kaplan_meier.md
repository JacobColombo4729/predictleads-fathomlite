# Kaplan-Meier survival estimation

What `src/kaplan_meier.py` computes, and why. Part 1 is the method on its own, the way a survival-analysis
textbook introduces it — no listings yet. Part 2 applies it to this project and lines the formula up against the
code. Written for the formula, not for `src/eda_helpers.py`'s `km_curve`/`surv_at`/`km_median`, which stay in use
elsewhere (`EDA_SDA_2`–`4`, `TimeSeries`) and are untouched; the two implementations are checked against each
other on synthetic data in `tests/test_kaplan_meier.py` and agree exactly.

## Quick summary (for explaining this out loud)

Kaplan-Meier estimates the probability that something hasn't happened yet, as a function of time, when you don't
get to watch every subject all the way to an ending. Here, "the thing" is a job listing closing, and "not
watching to an ending" means the listing was still open when data collection stopped (September 30) — we know it
survived at least that long, nothing more. It's the standard method for exactly this mix: some listings whose
outcome you know (closed, on a specific day) and some whose outcome you don't know yet. It produces a curve that
 decreases over time.

- **What it estimates:** $S(t)$, the share of listings still open after $t$ days.
- **Why not just average the listings that already closed:** drop every still-open listing from the calculation,
  then average only the observed durations of the ones that did close. The listings you kept are
  disproportionately the ones that happened to close before you stopped watching — a listing that takes 90 days
  to close hasn't closed yet at day 30, so it's excluded while every faster listing counts. The average reads
  shorter than reality (survivorship bias); imputing that average onto the still-open listings doesn't fix it
  either, since it ignores how long each one has already been open — a listing already open 60 days can't
  plausibly be heading toward a 12-day average, it's already past that.
- **Why not just treat still-open listings as closed today, or as never closing:** both are guesses too, in the
  other direction. Kaplan-Meier uses only the time each listing was actually observed, and nothing beyond that.
- **The one-line mechanics:** at every day something closed, take the share of the still-open listings that
  *didn't* close that day; multiply those daily shares together, in order, to get the curve.
- **Censoring, in this project:** a listing is censored if it's still open on September 30 — not a flaw in the
  data, just the data collection ending while the listing is still running.
- **Scale:** 22.3% of all original listings are censored this way — not a tail case, roughly a quarter of the
  data, and unevenly spread (recently posted listings are far more likely to still be open).
- **Where this shows up beyond this notebook:** `TimeSeries` runs it by month; `EDA_SDA_4` fits a related model
  (Cox regression) for the same reason; `ML_Rank`'s priority queue trains an actual survival forest for its
  "chance still open" factor. None of those work without treating closed and still-open listings together.

## Part 1: The method, in general

### The problem

Say you're observing a group of subjects, and for each one you're waiting for **the event** — death, failure,
relapse, a machine breaking, anything with a time attached. Let $T$ be the (unknown, true) time from the start of
observation to the event, for a given subject. You want $S(t) = P(T > t)$: the probability a subject's event
hasn't happened yet by time $t$. This is the **survival function**, and it's a full curve, not one number.

The complication is **censoring**: for some subjects, the study ends — or the subject drops out, or is lost to
follow-up — before their event happens. You don't know their $T$; you only know $T$ is *at least* however long
you watched them. This is the textbook's classic setup: a clinical trial where some patients are still alive when
the study ends, or are lost to follow-up partway through. A censored subject is not missing data and not a
failure — it's a lower bound on an unknown $T$.

Two tempting shortcuts are both wrong:

- **Drop everyone censored, average the rest.** This biases $S(t)$ low (survival looks shorter than it is),
  because the subjects you kept are disproportionately the ones whose event happened *before* the study ended —
  long survivors are exactly the ones still running when you stopped watching.
- **Treat censored subjects as if they survived forever (or until some arbitrary end date).** This biases $S(t)$
  high, for the opposite reason: you're crediting unknown future survival you haven't actually observed.

Kaplan-Meier (1958) is the standard nonparametric estimator of $S(t)$ that uses a censored subject for exactly as
long as they were actually observed, and no longer, without guessing what happens after.

### The estimator

Order every distinct time at which an event happened, across every subject, as $t_1 < t_2 < \dots$. (Censoring
times don't get their own term in the product below, but they matter — see $n_i$.) At each $t_i$, define:

- $n_i$ = the number of subjects **at risk** just before $t_i$: still under observation, event not yet happened.
- $d_i$ = the number of subjects whose event happened *exactly* at $t_i$.

Among the $n_i$ subjects at risk at $t_i$, the fraction $d_i / n_i$ had their event right then, so $1 - d_i/n_i$
is the conditional probability of making it past $t_i$, *given* you were still at risk going into $t_i$.
Surviving past some time $t$ means surviving past every $t_i \le t$ in turn, and (this is the only real
assumption) each of those conditional probabilities multiplies, by the chain rule:

$$S(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)$$

A few properties worth noticing before writing any code: $S(0) = 1$ (nobody's had the event yet); $S$ is a
step function that only drops at a time with an actual event; and a censoring at $t_i$ never appears in $d_i$,
but it does shrink $n_i$ for every $t_j > t_i$ that follows — censored subjects affect the curve only by leaving
the at-risk pool, never by looking like an event.

### A small worked example

10 subjects, observed until event (E) or censoring (C):

| time | 1 | 1 | 2 | 2 | 2 | 3 | 4 | 5 | 5 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| status | E | C | E | E | C | E | C | E | E | C |

| $t_i$ | at risk $n_i$ | events $d_i$ | factor $1-d_i/n_i$ | $S(t_i)$ |
|---|---|---|---|---|
| 1 | 10 | 1 | 0.900 | 0.900 |
| 2 | 8  | 2 | 0.750 | 0.675 |
| 3 | 5  | 1 | 0.800 | 0.540 |
| 4 | 4  | 0 | 1.000 | 0.540 |
| 5 | 3  | 2 | 0.333 | 0.180 |

Reading the $n_i$ column: it starts at 10, drops by 2 after $t=1$ (1 event + 1 censoring both leave the at-risk
pool), by 3 more after $t=2$, by 1 after $t=3$, by 1 after $t=4$ (the censoring there, with zero events, is why
$S$ doesn't move from $t=3$ to $t=4$ even though the curve is still "stepping"). This exact table is what
`tests/test_kaplan_meier.py` hand-checks `km_curve` against.

## Part 2: Applying it here

In this project, a subject is a job listing, and the event is **closing**. $T$ is `days_open`
(`last_seen - first_seen`). A listing is **censored** when it's still open on September 30 — the date's data
collection ends, not anything about the listing — so we know it survived at least that long and nothing more.
We want $S(t)$ = the share of listings still open after $t$ days, which is exactly what section 3's field
comparison and section 5's priority factors need (e.g. "closed within 7 days" is $1 - S(7)$).

### Step by step, against the code

**1. One row per day: how many listings' observation ended, and how many of those were actual closes.**

```python
def _by_day(days, finished):
    paired = pd.DataFrame({"day": days, "event": finished}).dropna()
    by_day = paired.groupby("day")["event"].agg(total="size", events="sum")
    return by_day.sort_index()
```

`total` is everything that ended that day (closed or censored, i.e. $n_i$'s raw ingredient); `events` is just the
closes ($d_i$). Ties are the normal case, not an edge case — many listings close or get censored on the same
day, which is why this groups by day instead of handling one listing at a time.

**2. Who's still at risk right before each day ($n_i$).**

```python
def _at_risk(by_day):
    n_total = by_day["total"].sum()
    ended_before = by_day["total"].cumsum().shift(fill_value=0)
    return n_total - ended_before
```

Everyone, minus everyone whose observation already ended on a *strictly earlier* day. `cumsum().shift()` is "the
running total, as of the day before this one" — `shift(fill_value=0)` is what makes the first day start from the
full count (nothing has ended "before" the first day).

**3. Each day's survival factor ($1 - d_i/n_i$).**

```python
def _survival_factor(by_day, at_risk):
    return 1 - by_day["events"] / at_risk
```

Directly from the formula. This is the only line that touches $d_i$ — censoring never appears here, only in step
2's $n_i$.

**4. The curve: the running product.**

```python
def km_curve(days, finished):
    by_day = _by_day(days, finished)
    at_risk = _at_risk(by_day)
    factor = _survival_factor(by_day, at_risk)
    return factor.cumprod()
```

`cumprod()` is $\prod_{t_i \le t}$. That's the whole estimator — three small steps and a running product.

**5. Two things built from the curve, not part of the estimator itself.**

```python
def survival_at(curve, x):
    earlier = curve[curve.index <= x]
    return earlier.iloc[-1] if len(earlier) else 1.0
```
$S(t)$ is a step function — flat between event days — so "the value at day $x$" means "the last day-with-an-
event at or before $x$." Before the first event, nobody's closed yet, so it's 1.0 (everyone's still open).

```python
def median_survival(days, finished):
    curve = km_curve(days, finished)
    at_or_below_half = curve[curve <= 0.5]
    return at_or_below_half.index.min()
```
The first day the curve reaches 50% — the typical closing time. `NaN` (an empty `at_or_below_half`) means more
than half of this group is still open at the end of the data; there isn't a median yet, and reporting one would
be a guess dressed as a number.

## Left open, worth learning on your own

- **Why this is "the" estimator, not just a reasonable one.** Kaplan-Meier is the nonparametric maximum-
  likelihood estimator of $S(t)$ under non-informative censoring (censoring that isn't caused by the thing
  you're measuring). Worth reading *why* once you've used the formula a few times — it's not derivable by just
  staring at the code.
- **Uncertainty on the curve.** This file reports $S(t)$ with no interval. Greenwood's formula gives the
  variance of $S(t)$ if a confidence band is ever needed.
- **Comparing two curves.** `EDA_SDA_4` already runs a log-rank test (comparing survival between groups, e.g.
  two role buckets) via `lifelines`; that's a different, related method worth understanding on its own once this
  estimator is solid.
- **The non-informative-censoring assumption, specifically for this data.** A listing is censored because the
  data collection ended on September 30 — not because of anything about the listing itself. That's the good
  case. It would be informative censoring (and bias the curve) if, say, listings were pulled from the crawl
  *because* they were about to close — worth keeping in mind if a future censoring mechanism is less clean than
  "the data just ended."
