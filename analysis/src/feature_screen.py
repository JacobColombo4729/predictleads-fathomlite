"""Feature screening: the rule EDA_SDA_2 3.13 fixed before its run, shared so EDA_SDA_4 screens the text features the
same way (one pre-set test for every feature, never a rule chosen after seeing the results).

The outcome is "closed within 7 days" on original listings with a full week of follow-up. For each candidate:
  - levels: true / false, quintiles, or the 10 most common values, with "missing" as its own level;
  - signal within the confounders: within each confounder cell (company volume x source type x has a description by
    default), the 7-day close rate's spread across levels with MIN_WITHIN+ listings, averaged over cells weighted by
    listings; minus the same spread with the outcome shuffled inside each cell (the noise floor), in points;
  - information within the confounders: conditional mutual information as a share of the outcome's uncertainty, above
    the same shuffled floor (per 10,000);
  - stability: the spread in April-June and in July-September, and whether the raw level order is kept;
  - redundancy: Cramer's V between levels (listings where both have a value), on a sample.
`verdict` applies the rules in EDA_SDA_2's order. Same numbers as EDA_SDA_2's own cells, which predate this module.
"""
import numpy as np
import pandas as pd

MIN_WITHIN, MIN_RAW = 400, 500   # listings per level: within a confounder cell / overall
MIN_SIGNAL = 2.0                 # points above the noise floor
MIN_COVERAGE = 0.05              # share of listings with a value
V_REDUNDANT = 0.7                # Cramer's V at which two features say the same thing
HALVES = {"Apr–Jun": ("2026-04", "2026-05", "2026-06"), "Jul–Sep": ("2026-07", "2026-08", "2026-09")}


def outcome_7d(base):
    """Closed within 7 days (base must have a full week of follow-up)."""
    return (~base.live.astype(bool) & base.days_open.le(7)).astype(float)


def halves(month):
    m = pd.Series(month).astype(str)
    return pd.Series(np.select([m.isin(HALVES["Apr–Jun"]), m.isin(HALVES["Jul–Sep"])], list(HALVES), "other"), index=m.index)


def levels(s):
    """Cut a feature into levels: true / false, quintiles, or the 10 most common values; "missing" is a level."""
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean" or set(s.dropna().unique()) <= {True, False}:
        out = s.map({True: "yes", False: "no"})
    elif pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
        out = pd.Series(np.nan, index=s.index, dtype=object)
        known = s.notna()
        out[known] = pd.qcut(s[known].rank(method="first"), 5, labels=["q1", "q2", "q3", "q4", "q5"]).astype(str)
    else:
        top = s.astype(str).where(s.notna()).value_counts().index[:10]
        out = s.astype(str).where(s.astype(str).isin(top), "other").where(s.notna())
    return out.fillna("missing").astype(str)


def shuffled_within(y, cell, seed=0):
    """The outcome shuffled within each confounder cell: what spreads look like when a feature carries nothing."""
    r = np.random.default_rng(seed)
    out = y.copy()
    for _, idx in y.groupby(cell).groups.items():
        out.loc[idx] = r.permutation(y.loc[idx].to_numpy())
    return out


def within_spread(lv, y, cell, mask=None):
    d = pd.DataFrame({"l": lv, "y": y, "c": cell})
    if mask is not None:
        d = d[mask]
    g = d.groupby(["c", "l"]).y.agg(["mean", "size"])
    g = g[g["size"] >= MIN_WITHIN]
    per = g.groupby(level="c")["mean"].agg(lambda m: m.max() - m.min() if len(m) >= 2 else np.nan)
    w = g.groupby(level="c")["size"].sum()[per.notna()]
    return float((per[w.index] * w).sum() / w.sum()) if len(w) else np.nan


def cond_info(lv, y, cell):
    """Conditional mutual information of the levels and the outcome within the cells, as a share of the outcome's
    uncertainty within the cells."""
    d = pd.DataFrame({"l": lv, "y": y.astype(int), "c": cell})
    n = d.groupby(["c", "l", "y"]).size().rename("n").reset_index()
    n_cl = n.groupby(["c", "l"]).n.transform("sum")
    n_cy = n.groupby(["c", "y"]).n.transform("sum")
    n_c = n.groupby("c").n.transform("sum")
    N = n.n.sum()
    cmi = (n.n / N * np.log(n.n * n_c / (n_cl * n_cy))).sum()
    cy = n.groupby(["c", "y"]).n.sum()
    c_tot = cy.groupby(level="c").transform("sum")
    h = -(cy / N * np.log(cy / c_tot)).sum()
    return cmi / h if h > 0 else np.nan


def raw_rates(lv, y, mask=None):
    d = pd.DataFrame({"l": lv, "y": y})
    if mask is not None:
        d = d[mask]
    g = d.groupby("l").y.agg(["mean", "size"])
    return g.loc[g["size"] >= MIN_RAW, "mean"]


def min_info(p0):
    """The information bar, fixed before looking: what a yes/no feature splitting listings in half, with a MIN_SIGNAL-point
    gap at the overall close rate p0, would carry (per 10,000)."""
    h = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))
    d = MIN_SIGNAL / 200
    return 1e4 * (h(p0) - 0.5 * h(p0 + d) - 0.5 * h(p0 - d)) / h(p0)


def screen(base, values, meta, cell, y=None):
    """Screen candidates on one population. base: the listings (index used throughout; needs month, live, days_open,
    source_type). values: {feature: Series indexed like base}. meta: {feature: (family, known at first sighting)}.
    cell: the confounder cell per listing. Returns (table, levels per feature, outcome)."""
    y = outcome_7d(base) if y is None else y
    half = halves(base.month)
    y_shuffled = shuffled_within(y, cell)
    rows, lvls = {}, {}
    for f, s in values.items():
        family, known = meta[f]
        s = s.loc[base.index]
        lv = levels(s)
        lvls[f] = lv
        present = s.notna()
        cov_by = [present.groupby(base.source_type).mean().min(), present.groupby(base.month.astype(str)).mean().min()]
        raw = raw_rates(lv, y)
        h1, h2 = (raw_rates(lv, y, half == h) for h in HALVES)
        common = h1.index.intersection(h2.index)
        spread, floor = 100 * within_spread(lv, y, cell), 100 * within_spread(lv, y_shuffled, cell)
        rows[f] = {
            "family": family, "known at first sighting": known, "coverage": present.mean(), "lowest coverage (source, month)": min(cov_by),
            "spread, raw (points)": 100 * (raw.max() - raw.min()) if len(raw) > 1 else np.nan,
            "spread within confounders (points)": spread, "noise floor (points)": floor,
            "signal above noise (points)": spread - floor,
            "information above noise (per 10,000)": 1e4 * (cond_info(lv, y, cell) - cond_info(lv, y_shuffled, cell)),
            "within, Apr–Jun": 100 * within_spread(lv, y, cell, half == "Apr–Jun"),
            "within, Jul–Sep": 100 * within_spread(lv, y, cell, half == "Jul–Sep"),
            "level order kept (rank correlation)": h1[common].corr(h2[common], method="spearman") if len(common) > 2 else np.nan,
        }
    table = pd.DataFrame(rows).T
    for c in table.columns.drop(["family", "known at first sighting"]):
        table[c] = pd.to_numeric(table[c])
    return table, lvls, y


def redundancy(table, lvls, sample_index, min_pairs=1_000):
    """Each screened feature's most similar other feature (Cramer's V between levels, listings where both have a
    value: shared blanks aren't redundancy). `lvls` may hold features beyond `table` (partners only)."""
    from scipy.stats.contingency import association
    feats = list(lvls)
    codes = {f: lvls[f].loc[sample_index] for f in feats}
    V = pd.DataFrame(np.nan, index=feats, columns=feats)
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if a not in table.index and b not in table.index:
                continue
            both = codes[a].ne("missing") & codes[b].ne("missing")
            if both.sum() < min_pairs:
                continue
            t = pd.crosstab(codes[a][both], codes[b][both]).to_numpy()
            V.loc[a, b] = V.loc[b, a] = association(t, method="cramer") if t.shape[0] > 1 and t.shape[1] > 1 else np.nan
    out = table.copy()
    partner, partner_v = {}, {}
    for f in table.index:
        best = V.loc[f].drop(f).dropna()
        partner[f], partner_v[f] = (best.idxmax(), best.max()) if len(best) else ("", np.nan)
    out["most similar feature"] = pd.Series(partner)
    out["Cramér's V with it"] = pd.Series(partner_v)
    return out


def verdict(f, r, signal_of, info_bar, confounders=(), hold_text=True):
    """EDA_SDA_2's rules, in its order. signal_of: Series of 'signal above noise (points)' for every feature a partner
    may be (the screened ones and any others). hold_text=False drops weak text features instead of holding them for
    EDA_SDA_3 (EDA_SDA_4: EDA_SDA_3 is done; a dropped text feature stays a label or opt-in filter in the app)."""
    if f in confounders:
        return "keep: standard confounder"
    if r["known at first sighting"] == "no":
        return "drop: not known when the listing is first seen"
    if r["coverage"] < MIN_COVERAGE:
        return "drop: too rare (under 5%)"
    w = r["signal above noise (points)"]
    if pd.isna(w) or w < MIN_SIGNAL:
        if r["family"] == "6 text" and hold_text:
            return "hold: EDA_SDA_3 validates the text"
        return "drop: no signal beyond the confounders"
    info_low = pd.notna(r["information above noise (per 10,000)"]) and r["information above noise (per 10,000)"] < info_bar
    a, b = r["within, Apr–Jun"], r["within, Jul–Sep"]
    unstable = (pd.notna(a) and pd.notna(b) and min(a, b) < max(a, b) / 3) or (r["level order kept (rank correlation)"] < 0)
    p = r.get("most similar feature", "")
    if pd.notna(r.get("Cramér's V with it")) and r["Cramér's V with it"] >= V_REDUNDANT and p in signal_of.index and signal_of[p] > w:
        return f"keep one: redundant with {p}"
    out = ("keep, watch: unstable across months" if unstable
           else "keep, watch: the two measures disagree (little information: few listings affected)" if info_low else "keep")
    if r["known at first sighting"] == "partly":
        out += " (recompute on training months)"
    elif r["known at first sighting"] == "provisional":
        out += " (provisional: the leakage audit is open)"
    return out


def to_decision(v):
    """A verdict as EDA_SDA_4's decision vocabulary: kept / dropped (stx.DECISIONS)."""
    return "kept" if v.startswith("keep") and not v.startswith("keep one") else "dropped"
