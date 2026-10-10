"""Hiring-momentum and buying-intent signals from observed job openings.

Built only from structured fields (company, title, role-family reasons, `first_seen_at`, `last_seen_at`, status, the
provider's technology tags), so the same code can later run over the full snapshot; it needs no embeddings.

Everything here is **observed** activity. `first_seen_at` and `last_seen_at` are crawl observation times, not posting
or closing dates (see docs/signal-semantics.md): a listing is "observed from first to last sighting", and it counts as
*still observed* when it was last seen within `GRACE_DAYS` of the snapshot and is not marked closed.

Pipeline:
  load_listings    stream the pool JSONL into a slim table, merging reposts of the same opening into one
  flag_artifacts   mark batch-onboarding days (a company's openings discovered at once) and mass-disappearance days
  company_metrics  per company at a cutoff day T, using only what was known at T: inflow, active, outflow, baseline, surge
  intent_signals   role-mix and technology signals with evidence
  backtest         do the signals at an earlier cutoff predict later openings better than simple baselines?
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EPOCH = date(2026, 1, 1)
GRACE_DAYS = 7              # last seen within this many days of the cutoff counts as still observed
HISTORY_DAYS = 90           # a baseline needs at least this much observed history
BATCH_MIN = 6               # a day with at least this many first sightings can be a batch discovery...
BATCH_SHARE = 0.4           # ...if it is also at least this share of the company's listings
SURGE_Z = 2.0               # surge z-score at which a company is called "surging"
SURGE_MIN_NEW = 3           # ...with at least this many new observed openings in 30 days
SUSTAINED_MIN_BASELINE = 3.0   # steady hiring: at least this many new openings a month, on average...
SUSTAINED_BAND = (0.5, 1.6)    # ...and this month within this multiple of that average
FAMILIES = ("ai_ml", "data", "technical_sales", "other")

_TITLE_REASONS = (("ai_ml_title", "ai_ml"), ("technical_sales_title", "technical_sales"), ("data_title", "data"))
_ONET_FAMILY = {
    "Data Scientists": "data", "Statisticians": "data", "Business Intelligence Analysts": "data",
    "Computer and Information Research Scientists": "ai_ml", "Sales Engineers": "technical_sales",
}
# provider tags that are not technologies a vendor could sell into
_NOT_TECH = {
    "internship", "contractor", "customer success", "marketing campaigns", "business development", "microsoft", "google",
    "amazon", "microsoft office", "microsoft word", "microsoft powerpoint", "microsoft excel", "c",
}


def day_number(d: date | datetime | str | None) -> int | None:
    """Days since 2026-01-01 for a date, datetime or ISO string; None if missing or unparsable."""
    if d is None or d == "":
        return None
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d[:10])
        except ValueError:
            return None
    if isinstance(d, datetime):
        d = d.date()
    return (d - EPOCH).days


def day_to_date(day: int) -> date:
    return date.fromordinal(EPOCH.toordinal() + int(day))


def primary_family(reasons: Iterable[str]) -> str:
    reasons = list(reasons or [])
    for reason, family in _TITLE_REASONS:
        if reason in reasons:
            return family
    for reason in reasons:
        if reason.startswith("onet:"):
            return _ONET_FAMILY.get(reason[5:], "other")
    return "other"


def _dedupe_key(company: str, title: str, description: str | None, location: str | None) -> str:
    body = " ".join((description or "").split())[:2000] or f"|{location or ''}"
    return hashlib.sha1(f"{company}\0{title.casefold()}\0{body}".encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------- loading

def load_listings(pool_path: Path, limit: int | None = None) -> pd.DataFrame:
    """Stream the pool into a slim table. Reposts (same company, title and description) become one opening that runs from
    its earliest first sighting to its latest last sighting, with `reposts` counting the extra copies."""
    rows: dict[str, dict[str, Any]] = {}
    with pool_path.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                break
            record = json.loads(line)
            dates = record.get("dates_utc") or {}
            first, last = day_number(dates.get("first_seen_at")), day_number(dates.get("last_seen_at"))
            reference = record.get("company_reference")
            company = str(reference.get("id")) if isinstance(reference, Mapping) and reference.get("id") else None
            if first is None or last is None or company is None:
                continue
            title = str(record.get("title_searchable") or record.get("title_normalized") or record.get("title_raw") or "")
            key = _dedupe_key(company, title, record.get("description_text"), record.get("location_raw"))
            closed = str(record.get("status_derived") or "").casefold() == "closed"
            tags = tuple(t for t in (record.get("tags") or []) if isinstance(t, str) and t.strip())
            existing = rows.get(key)
            if existing is not None:
                existing["first_day"] = min(existing["first_day"], first)
                existing["last_day"] = max(existing["last_day"], last)
                existing["closed"] = existing["closed"] and closed  # reopened by a later copy: not closed
                existing["reposts"] += 1
                continue
            salary = record.get("salary_data_raw") or {}
            rows[key] = {
                "job_id": record.get("job_id"), "company_id": company,
                "company_name": record.get("company_name_included"), "domain": record.get("company_domain_included"),
                "title": title, "family": primary_family(record.get("targeted_pool_reasons")),
                "first_day": first, "last_day": last, "closed": closed, "reposts": 0,
                "seniority": record.get("seniority"),
                "has_salary": bool(isinstance(salary, Mapping) and (salary.get("salary_low") or salary.get("salary_high"))),
                "tags": tags,
            }
    frame = pd.DataFrame(list(rows.values()))
    if frame.empty:
        return frame
    frame["company_code"] = pd.factorize(frame["company_id"])[0]
    return frame


# --------------------------------------------------------------------------- artifacts

def flag_artifacts(frame: pd.DataFrame, upto: int | None = None) -> pd.DataFrame:
    """Add `batch_start` and `mass_end` flags using only listings known by day `upto` (no look-ahead; None = everything).

    batch_start: first sighted on a day when the company's openings were discovered together (a crawl onboarding or a
                 site rebuild): at least BATCH_MIN openings that day and at least BATCH_SHARE of the company's listings.
    mass_end:    last sighted on a day when most of the company's vanished openings disappeared together (a crawl
                 dropout or site change rather than individual closures).
    """
    out = frame.copy()
    known = out["first_day"] <= upto if upto is not None else pd.Series(True, index=out.index)
    seen = out[known]
    totals = seen.groupby("company_code").size()
    per_day = seen.groupby(["company_code", "first_day"]).size().rename("n").reset_index()
    per_day["total"] = per_day["company_code"].map(totals)
    batch = per_day[(per_day["n"] >= BATCH_MIN) & (per_day["n"] >= BATCH_SHARE * per_day["total"])]
    batch_keys = set(zip(batch["company_code"], batch["first_day"]))
    out["batch_start"] = [(c, d) in batch_keys for c, d in zip(out["company_code"], out["first_day"])]

    cutoff = upto if upto is not None else int(out["last_day"].max())
    gone = known & (out["last_day"] < cutoff - GRACE_DAYS)
    gone_frame = out[gone]
    gone_totals = gone_frame.groupby("company_code").size()
    per_end = gone_frame.groupby(["company_code", "last_day"]).size().rename("n").reset_index()
    per_end["total"] = per_end["company_code"].map(gone_totals)
    mass = per_end[(per_end["n"] >= BATCH_MIN) & (per_end["n"] >= 0.5 * per_end["total"])]
    mass_keys = set(zip(mass["company_code"], mass["last_day"]))
    out["mass_end"] = [(c, d) in mass_keys for c, d in zip(out["company_code"], out["last_day"])]
    return out


# --------------------------------------------------------------------------- metrics at a cutoff

def _count(frame: pd.DataFrame, mask: pd.Series, n_codes: int) -> np.ndarray:
    return np.bincount(frame.loc[mask, "company_code"].to_numpy(), minlength=n_codes)


def company_metrics(frame: pd.DataFrame, cutoff: int, family: str | None = None) -> pd.DataFrame:
    """Signals per company as of day `cutoff`, using only what was observable then.

    A listing is known at `cutoff` if first sighted by then. It is *active* at the cutoff if last sighted on or after the
    cutoff, and it is known to have *disappeared* only once it has been unseen for GRACE_DAYS.
    """
    data = flag_artifacts(frame, upto=cutoff)
    if family is not None:
        data = data[data["family"] == family]
    data = data[data["first_day"] <= cutoff]
    n_codes = int(frame["company_code"].max()) + 1
    first, last = data["first_day"], data["last_day"]
    clean_start = ~data["batch_start"]

    def inflow(lo: int, hi: int) -> np.ndarray:  # new observed openings first sighted in (lo, hi]
        return _count(data, (first > lo) & (first <= hi) & clean_start, n_codes)

    new_30 = inflow(cutoff - 30, cutoff)
    new_prev_90 = inflow(cutoff - 120, cutoff - 30)
    active = _count(data, last >= cutoff, n_codes)
    active_30_ago = _count(data, (first <= cutoff - 30) & (last >= cutoff - 30), n_codes)
    gone_30 = _count(data, (last > cutoff - 30 - GRACE_DAYS) & (last <= cutoff - GRACE_DAYS) & ~data["mass_end"], n_codes)
    seen = _count(data, pd.Series(True, index=data.index), n_codes)
    batch_recent = _count(data, (first > cutoff - 30) & data["batch_start"], n_codes) > 0
    first_obs = data.groupby("company_code")["first_day"].min().reindex(range(n_codes)).to_numpy()

    out = pd.DataFrame({
        "company_code": np.arange(n_codes), "seen": seen, "new_30": new_30, "new_prev_90": new_prev_90,
        "active": active, "active_30_ago": active_30_ago, "gone_30": gone_30, "batch_recent": batch_recent,
        "history_days": cutoff - first_obs,
    })
    out = out[out["seen"] > 0].copy()
    out["baseline"] = np.maximum(out["new_prev_90"] / 3.0, 0.25)       # monthly rate over the prior three months
    out["net_30"] = out["new_30"] - out["gone_30"]
    out["ratio"] = (out["new_30"] + 1.0) / (out["baseline"] + 1.0)
    out["surge_z"] = (out["new_30"] - out["baseline"]) / np.sqrt(out["baseline"] + 1.0)
    out["enough_history"] = out["history_days"] >= HISTORY_DAYS
    out["surging"] = (
        out["enough_history"] & ~out["batch_recent"] & (out["surge_z"] >= SURGE_Z) & (out["new_30"] >= SURGE_MIN_NEW)
    )
    out["cooling"] = out["enough_history"] & (out["net_30"] <= -SURGE_MIN_NEW) & (out["new_30"] < out["baseline"])
    steady = out["new_30"] / out["baseline"]
    out["sustained"] = (
        out["enough_history"] & ~out["batch_recent"] & ~out["surging"] & (out["baseline"] >= SUSTAINED_MIN_BASELINE)
        & (out["new_30"] >= SURGE_MIN_NEW) & (steady >= SUSTAINED_BAND[0]) & (steady <= SUSTAINED_BAND[1])
    )
    out["pattern"] = np.select([out["surging"], out["sustained"], out["cooling"]], ["burst", "sustained", "cooling"], default="quiet")
    return out


# --------------------------------------------------------------------------- intent signals

INTENT_FAMILIES = {
    "ai_ml": ("AI/ML buildout", "hiring ML and AI roles: likely buying data, compute, MLOps and labeling tools"),
    "data": ("Data stack buildout", "hiring analytics and data roles: likely buying warehouse, BI and pipeline tools"),
    "technical_sales": ("Technical go-to-market buildout", "hiring solutions and sales engineers: likely buying sales-tech and enablement tools"),
}


def intent_signals(frame: pd.DataFrame, cutoff: int, min_new: int = SURGE_MIN_NEW) -> pd.DataFrame:
    """One row per company and role family that is surging, with the newest openings as evidence."""
    rows: list[dict[str, Any]] = []
    flagged = flag_artifacts(frame, upto=cutoff)
    names = frame.drop_duplicates("company_code").set_index("company_code")[["company_id", "company_name", "domain"]]
    for family, (label, meaning) in INTENT_FAMILIES.items():
        metrics = company_metrics(frame, cutoff, family=family)
        hot = metrics[metrics["surging"] & (metrics["new_30"] >= min_new)]
        for _, row in hot.iterrows():
            code = int(row["company_code"])
            recent = flagged[(flagged["company_code"] == code) & (flagged["family"] == family) & ~flagged["batch_start"]
                             & (flagged["first_day"] > cutoff - 30) & (flagged["first_day"] <= cutoff)]
            recent = recent.sort_values("first_day", ascending=False)
            tags = pd.Series([t for tags in recent["tags"] for t in tags if t.casefold() not in _NOT_TECH]).value_counts()
            rows.append({
                "company_code": code, "company_id": names.loc[code, "company_id"], "company_name": names.loc[code, "company_name"],
                "domain": names.loc[code, "domain"], "family": family, "signal": label, "meaning": meaning,
                "new_30": int(row["new_30"]), "baseline_monthly": round(float(row["baseline"]), 2), "surge_z": round(float(row["surge_z"]), 2),
                "active": int(row["active"]), "top_tech": list(tags.index[:5]),
                "evidence": [{"job_id": r.job_id, "title": r.title, "first_seen": day_to_date(int(r.first_day)).isoformat()}
                             for r in recent.head(3).itertuples()],
            })
    return pd.DataFrame(rows).sort_values("surge_z", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(rows)


def technology_leads(frame: pd.DataFrame, cutoff: int, top_tags: int = 60) -> pd.DataFrame:
    """Companies ranked, per technology tag, by new observed openings that mention it in the last 30 days."""
    data = flag_artifacts(frame, upto=cutoff)
    data = data[(data["first_day"] > cutoff - 30) & (data["first_day"] <= cutoff) & ~data["batch_start"]]
    exploded = data[["company_code", "company_id", "company_name", "domain", "tags", "last_day"]].explode("tags").dropna(subset=["tags"])
    exploded = exploded[~exploded["tags"].str.casefold().isin(_NOT_TECH)]
    popular = exploded["tags"].value_counts().head(top_tags).index
    exploded = exploded[exploded["tags"].isin(popular)]
    grouped = exploded.groupby(["tags", "company_code", "company_id", "company_name", "domain"]).agg(
        new_30=("last_day", "size"), still_observed=("last_day", lambda s: int((s >= cutoff).sum()))).reset_index()
    return grouped.rename(columns={"tags": "technology"}).sort_values(["technology", "new_30"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- weekly series (for charts)

def weekly_series(frame: pd.DataFrame, cutoff: int, weeks: int = 26) -> pd.DataFrame:
    """New (excluding batch days), disappeared and still-observed openings per company per week, ending at `cutoff`."""
    data = flag_artifacts(frame, upto=cutoff)
    n_codes = int(frame["company_code"].max()) + 1
    rows = []
    for w in range(weeks):
        end = cutoff - 7 * w
        start = end - 7
        new = _count(data, (data["first_day"] > start) & (data["first_day"] <= end) & ~data["batch_start"], n_codes)
        gone = _count(data, (data["last_day"] > start - GRACE_DAYS) & (data["last_day"] <= end - GRACE_DAYS) & ~data["mass_end"], n_codes)
        active = _count(data, (data["first_day"] <= end) & (data["last_day"] >= end), n_codes)
        keep = np.flatnonzero((new + gone + active) > 0)
        rows.append(pd.DataFrame({"company_code": keep, "week_end": end, "new": new[keep], "gone": gone[keep], "active": active[keep]}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- backtest

FEATURES = ["new_30", "baseline", "active", "active_30_ago", "gone_30", "net_30", "ratio", "surge_z", "seen_log", "history_days"]


def _design(frame: pd.DataFrame, cutoff: int, horizon: int = 30) -> pd.DataFrame:
    """Features known at `cutoff` and the target: new observed openings in (cutoff, cutoff + horizon]."""
    features = company_metrics(frame, cutoff)
    features = features[features["enough_history"]].copy()
    features["seen_log"] = np.log1p(features["seen"])
    future = flag_artifacts(frame, upto=cutoff + horizon)
    n_codes = int(frame["company_code"].max()) + 1
    target = _count(future, (future["first_day"] > cutoff) & (future["first_day"] <= cutoff + horizon) & ~future["batch_start"], n_codes)
    features["target"] = target[features["company_code"].to_numpy()]
    return features


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho = spearmanr(a, b).statistic
    return float(rho) if rho == rho else 0.0


def _lift(pred: np.ndarray, actual: np.ndarray, top: int) -> float:
    """Mean actual openings among the `top` highest-scored companies, divided by the mean over everyone."""
    order = np.argsort(-pred, kind="stable")[:top]
    return float(actual[order].mean() / max(actual.mean(), 1e-9))


def backtest(frame: pd.DataFrame, train_cutoffs: Sequence[int], test_cutoff: int, horizon: int = 30, top: int = 200) -> dict[str, Any]:
    """Temporal holdout: fit on earlier cutoffs, test at a later one whose target window does not overlap training."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    if max(train_cutoffs) + horizon > test_cutoff:
        raise ValueError("A training target window overlaps the test cutoff; choose earlier training cutoffs.")
    train = pd.concat([_design(frame, c, horizon) for c in train_cutoffs], ignore_index=True)
    test = _design(frame, test_cutoff, horizon)
    y_train, y_test = np.log1p(train["target"].to_numpy()), np.log1p(test["target"].to_numpy())
    actual = test["target"].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4, random_state=0).fit(train[FEATURES], y_train)
    # predicts the CHANGE from last month's count, so it can only beat the naive baseline if momentum explains the change
    last_train, last_test = np.log1p(train["new_30"].to_numpy()), np.log1p(test["new_30"].to_numpy())
    delta_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4, random_state=0).fit(train[FEATURES], y_train - last_train)
    predictions = {
        "baseline: still-observed openings (size only)": test["active"].to_numpy(dtype=float),
        "baseline: new openings in the last 30 days": test["new_30"].to_numpy(dtype=float),
        "baseline: average monthly rate over 4 months": ((test["new_prev_90"] + test["new_30"]) / 4.0).to_numpy(),
        "surge score only (z against own baseline)": test["surge_z"].to_numpy(),
        "model: momentum features (gradient boosting)": np.expm1(model.predict(test[FEATURES])),
        "model: change from last month (gradient boosting)": np.expm1(last_test + delta_model.predict(test[FEATURES])),
    }
    results = []
    for name, pred in predictions.items():
        mae = float(np.mean(np.abs(np.log1p(np.clip(pred, 0, None)) - y_test)))
        results.append({"method": name, "spearman": round(_spearman(pred, actual), 3), "mae_log1p": round(mae, 3),
                        f"lift_top{top}": round(_lift(pred, actual, top), 2)})

    # does the surge flag add information beyond the company's own baseline?
    flagged = test["surging"].to_numpy()
    beat = (actual > test["baseline"].to_numpy())
    surge_report = {
        "companies_tested": int(len(test)), "flagged_surging": int(flagged.sum()),
        "share_flagged_above_own_baseline_next_month": round(float(beat[flagged].mean()), 3) if flagged.any() else None,
        "share_unflagged_above_own_baseline_next_month": round(float(beat[~flagged].mean()), 3),
        "mean_next_30_flagged": round(float(actual[flagged].mean()), 2) if flagged.any() else None,
        "mean_next_30_unflagged": round(float(actual[~flagged].mean()), 2),
    }
    # Surges mean-revert: how much of this month's level carries into the next?
    surge_report["flagged_mean_new_30_at_cutoff"] = round(float(test.loc[flagged, "new_30"].mean()), 2) if flagged.any() else None
    surge_report["flagged_mean_own_baseline"] = round(float(test.loc[flagged, "baseline"].mean()), 2) if flagged.any() else None
    # Does the surge flag add anything beyond "this company is hiring a lot right now"? Compare flagged and unflagged
    # companies that had the SAME number of new openings this month.
    level_bins = pd.cut(test["new_30"], [2, 4, 7, 12, 25, 10_000], labels=["3-4", "5-7", "8-12", "13-25", "26+"])
    level_matched = []
    for label in level_bins.cat.categories:
        in_bin = (level_bins == label).to_numpy()
        f, u = in_bin & flagged, in_bin & ~flagged
        if f.sum() >= 5 and u.sum() >= 20:
            level_matched.append({
                "new_30_this_month": str(label), "flagged": int(f.sum()), "unflagged": int(u.sum()),
                "mean_new_30_flagged": round(float(test.loc[f, "new_30"].mean()), 2), "mean_new_30_unflagged": round(float(test.loc[u, "new_30"].mean()), 2),
                "mean_next_30_flagged": round(float(actual[f].mean()), 2), "mean_next_30_unflagged": round(float(actual[u].mean()), 2),
                "share_next_ge_this_month_flagged": round(float((actual[f] >= test.loc[f, "new_30"].to_numpy()).mean()), 3),
                "share_next_ge_this_month_unflagged": round(float((actual[u] >= test.loc[u, "new_30"].to_numpy()).mean()), 3),
            })
    # size-matched comparison: among companies with similar activity, are flagged ones higher next month?
    bins = pd.qcut(test["active"].rank(method="first"), 5, labels=False)
    matched = []
    for b in range(5):
        in_bin = (bins == b).to_numpy()
        f, u = in_bin & flagged, in_bin & ~flagged
        if f.sum() >= 5 and u.sum() >= 5:
            matched.append({"size_quintile": b + 1, "flagged": int(f.sum()), "mean_next_30_flagged": round(float(actual[f].mean()), 2),
                            "mean_next_30_unflagged": round(float(actual[u].mean()), 2)})
    # do bursts and sustained hirers really behave differently? (next month's openings relative to this month's)
    pattern = test["pattern"].to_numpy()
    by_pattern = []
    for name in ("burst", "sustained", "cooling", "quiet"):
        sel = pattern == name
        if sel.sum() >= 5:
            this_month = test.loc[sel, "new_30"].to_numpy(dtype=float)
            by_pattern.append({
                "pattern": name, "companies": int(sel.sum()), "mean_new_30_this_month": round(float(this_month.mean()), 2),
                "mean_next_30": round(float(actual[sel].mean()), 2),
                "next_as_share_of_this_month": round(float(actual[sel].sum() / max(this_month.sum(), 1e-9)), 2),
                "share_still_hiring_next_month": round(float((actual[sel] >= 1).mean()), 3),
                "share_next_at_least_half_of_this_month": round(float((actual[sel] >= 0.5 * this_month).mean()), 3),
            })
    importance = dict(sorted(zip(FEATURES, np.round(_permutation(model, test[FEATURES], y_test), 4)), key=lambda p: -p[1]))
    return {
        "train_cutoffs": [day_to_date(c).isoformat() for c in train_cutoffs], "test_cutoff": day_to_date(test_cutoff).isoformat(),
        "horizon_days": horizon, "train_rows": int(len(train)), "test_rows": int(len(test)),
        "results": results, "surge_flag": surge_report, "surge_flag_by_size": matched, "surge_flag_vs_same_hiring_level": level_matched, "by_pattern": by_pattern, "feature_importance_mae_increase": importance,
        "notes": ["Targets are new *observed* openings (crawl first sightings, batch days excluded), not true postings.",
                  "Features at the cutoff use only listings first sighted by then; batch flags are recomputed from the same data.",
                  "Test companies need at least 90 days of observed history at the cutoff."],
    }


def _permutation(model: Any, X: pd.DataFrame, y: np.ndarray, repeats: int = 3) -> list[float]:
    rng = np.random.default_rng(0)
    base = float(np.mean(np.abs(model.predict(X) - y)))
    scores = []
    for column in X.columns:
        increase = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            increase.append(float(np.mean(np.abs(model.predict(shuffled) - y))) - base)
        scores.append(float(np.mean(increase)))
    return scores


# --------------------------------------------------------------------------- run everything

def build(pool_path: Path, out_dir: Path, snapshot: str = "2026-09-30", limit: int | None = None) -> dict[str, Any]:
    cutoff = day_number(snapshot)
    out_dir.mkdir(parents=True, exist_ok=False)
    frame = load_listings(pool_path, limit=limit)
    names = frame.drop_duplicates("company_code").set_index("company_code")[["company_id", "company_name", "domain"]]
    metrics = company_metrics(frame, cutoff)
    # still-observed openings by family now
    for family in FAMILIES[:3]:
        part = company_metrics(frame, cutoff, family=family).set_index("company_code")
        metrics[f"new_30_{family}"] = metrics["company_code"].map(part["new_30"]).fillna(0).astype(int)
        metrics[f"active_{family}"] = metrics["company_code"].map(part["active"]).fillna(0).astype(int)
    metrics = metrics.join(names, on="company_code")
    metrics.sort_values("surge_z", ascending=False).to_csv(out_dir / "company_signals.csv", index=False)
    intents = intent_signals(frame, cutoff)
    intents.assign(top_tech=intents["top_tech"].map(lambda t: "|".join(t)) if len(intents) else None,
                   evidence=intents["evidence"].map(json.dumps) if len(intents) else None).to_csv(out_dir / "intent_signals.csv", index=False)
    technology_leads(frame, cutoff).to_csv(out_dir / "technology_leads.csv", index=False)
    weekly_series(frame, cutoff).merge(names, left_on="company_code", right_index=True).to_csv(out_dir / "company_weekly.csv", index=False)
    summary = {
        "snapshot": snapshot, "pool": str(pool_path), "listings_after_merging_reposts": int(len(frame)),
        "reposts_merged": int(frame["reposts"].sum()), "companies": int(frame["company_code"].nunique()),
        "companies_with_signals": int(len(metrics)), "surging_companies": int(metrics["surging"].sum()),
        "cooling_companies": int(metrics["cooling"].sum()), "intent_rows": int(len(intents)),
        "batch_start_listings": int(flag_artifacts(frame, upto=cutoff)["batch_start"].sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write company signals, intent signals, technology leads and weekly series")
    b.add_argument("--pool", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--snapshot", default="2026-09-30")
    b.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("backtest", help="temporal backtest of the momentum signals")
    t.add_argument("--pool", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.pool, args.out, args.snapshot, args.limit), indent=1))
    else:
        frame = load_listings(args.pool, limit=args.limit)
        test = day_number("2026-08-31")
        result = backtest(frame, [day_number("2026-07-01"), day_number("2026-07-15"), day_number("2026-08-01")], test)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
