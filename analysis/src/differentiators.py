"""Differentiation ideas 1, 3 and 5 (analysis/DIFFERENTIATORS.md, Planned additions).

Self-contained: pandas/numpy only, column names are parameters, nothing is read from disk.
Claim rules: counts of listings and of people who moved; never causation, hires or success rates.

1. skill_unlock_table      which skill added to the user's own would make the most listings reachable
3. stepping_stone_routes   routes of 2+ consecutive moves within tech, people per step, openings per stop
5. backtest_top_k          precision@k of a model's top k against newest-first, trained earlier, tested later
"""
import numpy as np
import pandas as pd


def _isna(v):
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------- 1. which skill unlocks the most jobs
def skill_unlock_table(listings, user_skills, skills_col="skills", hidden_col=None, top=None):
    """For each skill the user lacks, how many more listings become reachable if they add it.

    listings: one row per listing; `skills_col` holds the skills the listing asks for (list/set/tuple).
    hidden_col: optional boolean column, True for listings the hiding rules (EDA_SDA_2 3.11) hide; they are left out.
    A listing is reachable when every skill it asks for is one the user has. It is "unlocked" by skill s
    when s is the only skill it asks for that the user lacks. Listings asking for no skills are always reachable.
    Returns: skill, unlocks (listings that become reachable), asked_by (visible listings asking for it),
    reachable_now, visible. Sorted by unlocks. Counts only; says nothing about who gets hired.
    """
    have = {str(s).lower() for s in user_skills}
    df = listings if hidden_col is None else listings[~listings[hidden_col].fillna(False).astype(bool)]
    asked = [set() if (v is None or _isna(v)) else {str(s).lower() for s in v} for v in df[skills_col]]
    missing = [a - have for a in asked]
    reachable_now = sum(1 for m in missing if not m)
    unlocks, asked_by = {}, {}
    for m in missing:
        for s in m:
            asked_by[s] = asked_by.get(s, 0) + 1
        if len(m) == 1:
            (s,) = m
            unlocks[s] = unlocks.get(s, 0) + 1
    out = pd.DataFrame({"skill": sorted(asked_by)})
    out["unlocks"] = out["skill"].map(unlocks).fillna(0).astype(int)
    out["asked_by"] = out["skill"].map(asked_by).astype(int)
    out["reachable_now"] = reachable_now
    out["visible"] = len(df)
    out = out.sort_values(["unlocks", "asked_by", "skill"], ascending=[False, False, True]).reset_index(drop=True)
    return out if top is None else out.head(top)


# ----------------------------------------------------------------- 3. stepping-stone planner
def stepping_stone_routes(moves, openings, person_col="person", from_col="from_role", to_col="to_role",
                          order_col="step", min_people=5, min_moves=2, max_moves=3, roles=None):
    """Routes of `min_moves`..`max_moves` consecutive moves within tech.

    moves: one row per move (a person's job change between roles); `order_col` orders each person's moves
        (a date or a counter). Only chains where each move starts where the previous one ended count.
    openings: dict or Series role -> number of 2026 listings (from the compass).
    roles: optional set of roles to keep (tech buckets); moves touching other roles are dropped first.
    A route is shown only if at least `min_people` people took exactly that chain of moves back to back.
    Returns: route ("A > B > C"), n_moves, people (took the whole route), step_people (fewest people on any
    single step of it, over everyone who took that step), openings_at_stops ("A: n | B: n"), last_stop_openings.
    Counts of people who moved; not odds of success.
    """
    m = moves[[person_col, from_col, to_col, order_col]].dropna()
    m = m[m[from_col] != m[to_col]]
    if roles is not None:
        m = m[m[from_col].isin(roles) & m[to_col].isin(roles)]
    m = m.sort_values([person_col, order_col], kind="stable")
    step_people = m.groupby([from_col, to_col])[person_col].nunique().to_dict()
    openings = dict(openings)

    chains = {}  # route tuple -> set of people
    for person, g in m.groupby(person_col, sort=False):
        f, t = g[from_col].tolist(), g[to_col].tolist()
        for i in range(len(f)):
            route = [f[i], t[i]]
            for j in range(i + 1, len(f)):
                if f[j] != route[-1] or len(route) - 1 >= max_moves:
                    break
                route.append(t[j])
                if len(route) - 1 >= min_moves:
                    chains.setdefault(tuple(route), set()).add(person)
    rows = []
    for route, people in chains.items():
        if len(people) < min_people:
            continue
        rows.append({"route": " > ".join(route), "n_moves": len(route) - 1, "people": len(people),
                     "step_people": min(step_people[(a, b)] for a, b in zip(route, route[1:])),
                     "openings_at_stops": " | ".join(f"{r}: {int(openings.get(r, 0))}" for r in route),
                     "last_stop_openings": int(openings.get(route[-1], 0))})
    cols = ["route", "n_moves", "people", "step_people", "openings_at_stops", "last_stop_openings"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["people", "last_stop_openings", "route"], ascending=[False, False, True]).reset_index(drop=True)


# ----------------------------------------------------------------- 5. backtested ranking claim
def precision_at_k(relevant_in_rank_order, k=10):
    """Share of the first k items (already in rank order, 0/1) that are relevant; fewer than k items use what exists."""
    top = np.asarray(relevant_in_rank_order)[:k]
    return float(top.mean()) if len(top) else np.nan


def split_train_test_months(df, date_col, train_months, test_months):
    """Rows whose month ('YYYY-MM') is in train_months / test_months; the two sets must not overlap."""
    if set(train_months) & set(test_months):
        raise ValueError("train and test months overlap")
    month = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")
    return df[month.isin(train_months)], df[month.isin(test_months)]


def backtest_top_k(test, score_col, label_col, date_col, group_col, k=10, n_boot=1000, random_state=0):
    """Precision@k of the model's top k against newest-first, on test rows only.

    Scores must come from a model fitted on earlier months only (see split_train_test_months).
    `group_col` is the list being ranked (e.g. one user profile or target role); `label_col` is 0/1 relevance.
    Groups with fewer than k rows are skipped. Ties break by row order for both rankings.
    Returns a dict: model, newest_first, lift (model / newest_first, nan if baseline is 0), difference,
    ci_low/ci_high (95% bootstrap over groups, for the difference), n_groups, k.
    """
    rng = np.random.default_rng(random_state)
    mp, bp = [], []
    for _, g in test.groupby(group_col, sort=True):
        if len(g) < k:
            continue
        mp.append(precision_at_k(g.sort_values(score_col, ascending=False, kind="stable")[label_col], k))
        bp.append(precision_at_k(g.sort_values(date_col, ascending=False, kind="stable")[label_col], k))
    if not mp:
        raise ValueError(f"no group has at least {k} listings")
    mp, bp = np.array(mp), np.array(bp)
    diff = mp - bp
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"model": float(mp.mean()), "newest_first": float(bp.mean()),
            "lift": float(mp.mean() / bp.mean()) if bp.mean() > 0 else float("nan"),
            "difference": float(diff.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n_groups": int(len(mp)), "k": int(k)}


def claim_sentence(result, train_label="April to July", test_label="September", unit="relevant"):
    """The pitch line, with its test months and baseline stated (DIFFERENTIATORS.md rule for idea 5)."""
    r = result
    return (f"Trained on {train_label} and tested on {test_label}, {r['model']:.0%} of the model's top {r['k']} were "
            f"{unit}, against {r['newest_first']:.0%} for newest-first (difference {r['difference']:+.1%}, "
            f"95% interval {r['ci_low']:+.1%} to {r['ci_high']:+.1%}, {r['n_groups']} lists).")
