"""Kaplan-Meier survival estimation, written step by step to match analysis/docs/kaplan_meier.md.

For EDA_SDA_1_Lean. src/eda_helpers.py's km_curve/surv_at/km_median are untouched (EDA_SDA_2-4 and TimeSeries
still call them); this is a separate, explicit implementation of the same formula, not a replacement.

    km_curve(days, finished)         the survival curve S(t): share of listings still open after t days
    survival_at(curve, x)            the curve's value at day x
    median_survival(days, finished)  the day S(t) first drops to or below 0.5 (NaN if it never does)
"""
import pandas as pd


def _by_day(days, finished):
    """One row per distinct day with an event or a censoring, sorted by day:
    total   how many listings' observation ended that day, closed or still open (censored) alike
    events  how many of those actually closed (finished=True) that day
    """
    paired = pd.DataFrame({"day": days, "event": finished}).dropna()
    by_day = paired.groupby("day")["event"].agg(total="size", events="sum")
    return by_day.sort_index()


def _at_risk(by_day):
    """How many listings are still "at risk" (not yet closed or censored) just before each day: everyone, minus
    everyone whose observation already ended on an earlier day."""
    n_total = by_day["total"].sum()
    ended_before = by_day["total"].cumsum().shift(fill_value=0)
    return n_total - ended_before


def _survival_factor(by_day, at_risk):
    """The Kaplan-Meier step at each day: the share of those at risk who did NOT close that day (1 - d_i / n_i)."""
    return 1 - by_day["events"] / at_risk


def km_curve(days, finished):
    """S(t): the share of listings still open after t days, for every day an event or censoring happened.
    S(t) = the product, over every day up to and including t, of that day's survival factor."""
    by_day = _by_day(days, finished)
    at_risk = _at_risk(by_day)
    factor = _survival_factor(by_day, at_risk)
    return factor.cumprod()


def survival_at(curve, x):
    """The curve's value at day x: it only changes on days with an event, so step forward from the last one at
    or before x (1.0 if x is before the curve's first day)."""
    earlier = curve[curve.index <= x]
    return earlier.iloc[-1] if len(earlier) else 1.0


def median_survival(days, finished):
    """Days until half of listings have closed; NaN if more than half are still open at the end of the data."""
    curve = km_curve(days, finished)
    at_or_below_half = curve[curve <= 0.5]
    return at_or_below_half.index.min()
