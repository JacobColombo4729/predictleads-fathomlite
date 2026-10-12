"""Small helpers for TimeSeries.ipynb: counts per day and week, leakage-safe features, time-ordered split, scores,
and the "check every N days" arithmetic. Pure functions on pandas objects; no data paths in here."""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CET = "Europe/Berlin"  # `posted` is midnight Central European time: read its calendar date there, not in UTC


def calendar_date(ts, tz=None):
    """Calendar dates (midnight, no timezone) of a timestamp column. tz=None reads UTC dates (the crawler's clock);
    tz=CET reads Central European dates (the employer's posting day)."""
    ts = pd.to_datetime(ts, utc=True)
    if tz:
        ts = ts.dt.tz_convert(tz)
    return ts.dt.tz_localize(None).dt.normalize()


def daily_counts(dates, start, end, name="listings"):
    """Listings per calendar day from `start` to `end`, with 0 on days that have none (so gaps show up as zeros)."""
    days = pd.date_range(start, end, freq="D")
    counts = dates.dropna().value_counts().reindex(days, fill_value=0).sort_index()
    counts.index.name = "date"
    return counts.rename(name)


def weekly_counts(daily):
    """Weeks run Monday to Sunday, labelled by the Monday. A week is `full` only if all 7 days are inside the data:
    partial first and last weeks are flagged, not dropped, so plots can mark them and rates can skip them."""
    weekly = daily.resample("W-MON", closed="left", label="left").sum().to_frame(daily.name or "listings")
    first, last = daily.index.min(), daily.index.max()
    weekly["full_week"] = (weekly.index >= first) & (weekly.index + pd.Timedelta(days=6) <= last)
    return weekly


def holiday_flag(index):
    """US federal holidays (known in advance, so safe as a feature)."""
    days = USFederalHolidayCalendar().holidays(index.min(), index.max())
    return pd.Series(index.isin(days), index=index, name="is_holiday")


def lag_features(y, lags=(1, 2, 3, 7, 14), windows=(7, 14)):
    """Features for predicting day t from days before t only. Every lag and rolling statistic is shifted by one day
    first (`.shift(1)` before `.rolling`), so day t's own count never leaks into its features. Calendar features
    (weekday, holiday) are known ahead of time. Rows without a full history are dropped."""
    X = pd.DataFrame(index=y.index)
    for k in lags:
        X[f"lag_{k}"] = y.shift(k)
    past = y.shift(1)
    for w in windows:
        X[f"roll_mean_{w}"] = past.rolling(w).mean()
        X[f"roll_std_{w}"] = past.rolling(w).std()
    for d in range(1, 7):  # Monday is the baseline weekday
        X[f"dow_{d}"] = (y.index.dayofweek == d).astype(int)
    X["is_holiday"] = holiday_flag(y.index).astype(int).values
    X = X.dropna()
    return X, y.loc[X.index]


def time_split(index, val_start, test_start):
    """Boolean masks (train, validation, test) in time order; no shuffling."""
    index = pd.DatetimeIndex(index)
    train = index < pd.Timestamp(val_start)
    val = (index >= pd.Timestamp(val_start)) & (index < pd.Timestamp(test_start))
    test = index >= pd.Timestamp(test_start)
    return train, val, test


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def survival_at(curve, t):
    """Share still open after t days from a Kaplan–Meier curve (a step function indexed by day); 1 before the first
    day with a closure. Accepts a single value or an array."""
    t = np.atleast_1d(np.asarray(t, float))
    pos = np.searchsorted(curve.index.values.astype(float), t, side="right") - 1
    return np.where(pos < 0, 1.0, curve.values[np.clip(pos, 0, None)])


def catch_rate(curve, every_n, days_left=0, step=0.25):
    """Check every `every_n` days at a fixed time of day. A listing appears at a random moment, so you meet it
    U days later, U uniform between 0 and N. You 'catch' it if it is still open at U, with at least `days_left`
    days still to apply: the average of survival(U + days_left) over U. Returns a share between 0 and 1."""
    u = np.arange(0, every_n, step) + step / 2
    return float(survival_at(curve, u + days_left).mean())


def longest_cadence(curve, target, days_left=0, max_n=60):
    """The longest N (in days) whose catch rate is still at least `target`; None if even daily checks fall short."""
    ok = [n for n in range(1, max_n + 1) if catch_rate(curve, n, days_left) >= target]
    return max(ok) if ok else None
