# Cox proportional-hazards regression

What the model in `EDA_SDA_4` section 8d does, and how to read its output. Part 1 is the method on its own, the
way a survival-analysis textbook introduces it. Part 2 applies it to this project. It builds on
[`kaplan_meier.md`](kaplan_meier.md): read that first if "censored" or "survival curve" is new. This is the one
genuinely advanced method the Lean rewrite keeps, which is why it has a doc at all
([`lean_conventions.md`](lean_conventions.md) rule 9). The worked example below is solvable by hand and was
checked against `lifelines`.

## Quick summary (for explaining this out loud)

Kaplan-Meier describes **one group's** closing curve. Cox regression compares **many factors at once**, on the
same censored data, and says how each one changes the speed of closing with the others held fixed.

The quantity it models is the **hazard**: among listings still open on a given day, the rate at which they close
that day. Cox says each listing's hazard is a shared, unestimated baseline curve multiplied by a factor that
depends only on its features:

$$h(t \mid x) = h_0(t) \, e^{\beta_1 x_1 + \cdots + \beta_p x_p}$$

- **The output is a hazard ratio,** $e^{\beta}$. Above 1 means closing faster, below 1 slower, 1 means no
  difference. A hazard ratio of 2 means that at any moment, among listings still open, this kind closes at twice
  the rate.
- **It is not a probability** and not "twice as likely to close eventually". It is a ratio of rates.
- **Why it suits this data:** still-open listings are censored rather than discarded, and the eight confounders
  are held at once instead of one at a time, which is what 4b and 9b can only do singly.
- **What the name assumes:** that the ratio stays the same over time. That is testable, and 8d tests it.

## Part 1: The method, in general

### The hazard, and what "proportional" means

Take two listings whose features differ in one covariate, $x = 1$ against $x = 0$. Divide their hazards:

$$\frac{h(t \mid x=1)}{h(t \mid x=0)} = \frac{h_0(t) e^{\beta}}{h_0(t)} = e^{\beta}$$

The baseline $h_0(t)$ cancels. Two consequences, and they are the whole character of the method: the ratio does
not depend on $t$ (that is "proportional hazards"), and **the baseline never has to be estimated** to learn
$\beta$. The shape of the closing curve over time can be arbitrary - fast early, slow later, bumpy - and the
model does not care.

### Why the baseline can be ignored: the partial likelihood

At each moment an event happens, ask a smaller question: *among the listings still at risk right now, what is
the chance that the one which closed is the one that actually did?* Under the model that chance is

$$L_i(\beta) = \frac{e^{\beta x_i}}{\sum_{j \,\in\, \text{at risk at } t_i} e^{\beta x_j}}$$

and the partial likelihood is the product over all event times. The baseline cancels out of every term, which is
why Cox is called *semiparametric*: a parametric factor for the features, nothing assumed about time.

Censoring enters naturally. A listing that is still open at the end contributes no numerator - it never had an
event - but it sits in the **denominators** of every event time before it leaves. That is exactly what we know
about it: it was available to close and didn't yet.

### A small worked example

Four listings, one covariate (`flag` = 1 or 0), one of them still open:

| listing | days | closed? | flag |
|:--|--:|:--|--:|
| A | 2 | yes | 1 |
| B | 3 | yes | 0 |
| C | 4 | **still open** | 1 |
| D | 5 | yes | 0 |

Write $u = e^{\beta}$. Three event times, with the at-risk set shrinking:

| event | at risk | term |
|:--|:--|:--|
| A closes at 2 | A, B, C, D | $u / (u + 1 + u + 1) = u/(2u+2)$ |
| B closes at 3 | B, C, D | $1 / (1 + u + 1) = 1/(u+2)$ |
| D closes at 5 | D | $1/1 = 1$ |

C never appears in a numerator, but it is in the first two denominators and gone by $t=5$. The last term is 1
and carries no information - with one listing left there was nothing to choose between. So

$$\log L = \log\frac{u}{2u+2} + \log\frac{1}{u+2}$$

Setting the derivative to zero gives $\frac{u}{u+1} + \frac{u}{u+2} = 1$, and clearing denominators collapses it
to

$$u^2 = 2 \quad\Longrightarrow\quad e^{\beta} = \sqrt{2} \approx 1.414, \qquad \beta = \tfrac12\log 2 \approx 0.347$$

Flagged listings close about **41% faster per day at risk**. `lifelines` on the same four rows returns
$\beta = 0.346$, a hazard ratio of 1.413 - it stops at its convergence tolerance rather than the exact root, and
agrees to three decimals.

### The assumption in the name, and how it is tested

If a factor's effect changes over time - strong in week one, gone by month two - a single hazard ratio averages
it into something that describes neither period. The standard test uses **Schoenfeld residuals**: for each
covariate, correlate its residuals with time. A small p-value says the effect moves. The responses are to
stratify by that factor (giving it its own baseline curve, and no hazard ratio), to add an interaction with time,
or to report the ratio as a time-average and say so.

### What Cox does not give you

- **No absolute risk.** Hazard ratios are relative. Turning them into "this listing has a 30% chance of closing
  this week" needs the baseline hazard back, estimated separately.
- **Ratios are conditional on the rest of the model.** A hazard ratio of 1.3 means "holding the other covariates
  fixed", so it is not comparable with a crude ratio from a two-way table, and it changes when the covariate set
  changes. That is a property of the model, not a bug.
- **A hazard ratio is not an odds ratio** and not a risk ratio, though people quote all three as if
  interchangeable.

## Part 2: Applying it here

The duration is `days_open`, the event is `closed`, and the covariates are the confounders plus whichever
features are under test. Listings with `days_open == 0` are dropped - a duration of zero has no at-risk period:

```python
CONFOUNDERS = [c for c in ["volume_band", "source_type", "platform", "is_staffing", "role_bucket",
                           "level", "month", "company_type", "has_desc"] if c in jobs]
UNDER_TEST = TEXT_FLAGS  # edit: the features whose adjusted effect is wanted
cph, design, hr = stx.cox_adjusted(cox_df, "days_open", "closed", CONFOUNDERS + UNDER_TEST,
                                   cluster="company_id", categorical=CAT_CONF, penalizer=0.01)
```

Four choices in that call worth being able to defend:

- **`cluster="company_id"`** switches on robust, company-clustered standard errors. Listings from one company are
  not independent - the same employer, the same hiring habits - and ignoring that makes every interval too
  narrow. It widens intervals; it does not move the hazard ratios.
- **`categorical=CAT_CONF`** expands each categorical confounder into indicator columns, so every level gets its
  own hazard ratio against that factor's reference level.
- **`penalizer=0.01`** is a small ridge penalty. With this many indicator levels a few are nearly collinear, and
  without it the fit can wander or fail to converge. It pulls coefficients slightly toward zero, so hazard ratios
  are a touch closer to 1 than an unpenalized fit would give.
- **Flags are filled with 0 where a listing has no description,** because a missing description means the text
  rules never ran, not that they found nothing. `has_desc` is in the model to carry that difference itself.

8d prints three things: the hazard-ratio table with 95% intervals, a joint Wald test per multi-level factor, and
the Schoenfeld check. Rule 9 keeps the first and the last and drops the joint Wald, which refines p-values that
carry little information at this sample size.

### How to read it

- **Above 1 closes faster, below 1 slower.** In `EDA_SDA_4`'s run the largest single effects were having a
  description, at about 0.4, and belonging to a company with 500+ earlier listings, at about 2.5 - so a listing
  with a description closes at well under half the rate of one without, and a big company's listings close about
  two and a half times as fast. Every text flag landed between 0.79 and 1.28.
- **Read the interval, not the p-value.** With clustered errors the interval is the honest one, and a hazard
  ratio whose interval spans 1 has not shown its direction. A wide interval is also where multicollinearity
  surfaces - which is why 3c's pairwise screen is enough and VIF was dropped
  ([`lean_conventions.md`](lean_conventions.md) rule 9).
- **"Closes faster" is not "better".** For a career changer it means *less time to apply*, which is why ML_Rank
  uses closing speed as urgency in the priority queue rather than as a quality score.
- **A small p in the Schoenfeld check is information, not failure.** It says that factor's effect changes over
  the window, so its single hazard ratio is a time-average - worth saying out loud rather than hiding.
- **8d is still evidence, not a decision.** Keep and drop calls are made in 10a by the rule `EDA_SDA_2` fixed
  before its run.

## Left open, worth learning on your own

- **Time-varying covariates and stratified Cox** - the two standard repairs when proportional hazards fails.
- **Competing risks.** A listing that ends as a likely ghost did not "close" in the same sense as one that
  disappeared for another reason. Cause-specific hazards and the Fine-Gray subdistribution model are the two ways to handle that; 8c
  looks at it descriptively instead.
- **The baseline hazard** (Breslow's estimator) is what turns these ratios into predicted curves - the step
  ML_Rank needs when it scores a listing rather than compares two.
- **Frailty (random-effects) models** are the alternative to clustered errors: model the company-level variation
  instead of correcting the errors for it.
