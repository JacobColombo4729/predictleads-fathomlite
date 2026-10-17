"""Date picker (as-of date): the listings live on a date, with features computed as of that date.

Rule (PROJECT_OUTLINE.md, How we keep it honest): a listing is live on a date if it was first seen on or before that day and last
seen on or after it. `last_seen` only decides who was live; it never feeds a feature. Every count uses only listings
first seen before the listing (or on or before the chosen day). The model for a date trains on `training_as_of`:
listings first seen by that day, with the ones still open that day censored at it.

Input: one row per listing (listing_rows.lean_part columns): id, company_id, first_seen, last_seen, desc_hash.
"""
import pandas as pd

NEEDED = ["id", "company_id", "first_seen", "last_seen", "desc_hash"]  # load only these (plus any you pass through)
DAY = pd.Timedelta(days=1)
LIVE_TOLERANCE_DAYS = 2  # last seen within this many days of a date = still open then (same rule as `live` at the data's end)


def _day(as_of, like):
    """The chosen day as a midnight timestamp in the same timezone as the data."""
    t = pd.Timestamp(as_of).normalize()
    tz = getattr(like.dt, "tz", None)
    return t.tz_localize(tz) if tz is not None and t.tzinfo is None else t


def known_by(listings, as_of):
    """Listings first seen on or before the as-of day: all anyone could know that day. Later rows are dropped."""
    day = _day(as_of, listings["first_seen"])
    return listings[listings["first_seen"] < day + DAY].copy()


def known_features(known):
    """Features that use only earlier listings. `known` must already be cut at the as-of day (see known_by)."""
    out = known.copy()
    by_co = out.groupby("company_id")["first_seen"]
    out["company_prior_listings"] = by_co.rank(method="min").sub(1).astype(int)  # strictly earlier, same company
    if "desc_hash" in out:
        key = out.groupby(["company_id", "desc_hash"])["first_seen"]
        out["repost_prior_copies"] = key.rank(method="min").sub(1).fillna(0).astype(int)  # earlier copies of the text
    return out


def live_on(listings, as_of):
    """Listings live on the as-of day, with features as of it: age_days, company_prior_listings,
    company_listings_to_date (all of the company's listings seen by that day), repost_prior_copies."""
    day = _day(as_of, listings["first_seen"])
    known = known_features(known_by(listings, as_of))
    known["company_listings_to_date"] = known.groupby("company_id")["id"].transform("size")
    live = known[known["last_seen"] >= day].copy()
    live["age_days"] = ((day - live["first_seen"]) / DAY).clip(lower=0)
    return live.drop(columns=["last_seen"])  # last_seen picked who was live; it is not a feature


def training_as_of(listings, as_of):
    """Rows to train the model for a date: listings first seen by the day, their features, and the outcome as known
    then. `days_observed` stops at the end of the day; `finished` is False (censored) for listings last seen
    within LIVE_TOLERANCE_DAYS of the day (still open then, as at the end of the data)."""
    day = _day(as_of, listings["first_seen"])
    known = known_features(known_by(listings, as_of))
    end = known["last_seen"].clip(upper=day + DAY)
    known["finished"] = known["last_seen"] < day - pd.Timedelta(days=LIVE_TOLERANCE_DAYS)  # more than 2 days before
    known["days_observed"] = ((end - known["first_seen"]) / DAY).clip(lower=0)
    return known.drop(columns=["last_seen"])


def count_check(listings, dates=("2026-07-01", "2026-09-01")):
    """Counts only (no rows, no values): live listings, their median age in days, companies, and the training rows
    (and how many are censored) for each date. EDA_SDA_1 8b shows dates before July are skewed young."""
    rows = []
    for d in dates:
        live, train = live_on(listings, d), training_as_of(listings, d)
        rows.append({"as_of": d, "live": len(live), "median_age_days": live["age_days"].median(),
                     "companies_live": live["company_id"].nunique(), "training_rows": len(train),
                     "censored": int((~train["finished"]).sum())})
    return pd.DataFrame(rows).set_index("as_of")
