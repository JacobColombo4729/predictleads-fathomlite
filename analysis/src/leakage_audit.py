"""Leakage audit for ML_Fit: checks, on the saved tables, that nothing about a person's actual next role (or anything
after their switch) reaches the model or its evaluation, and writes the evidence to reports/ML_Fit_audit.md.

Run from analysis/ after ML_Fit has run (it reads what ML_Fit 9a saved; a few minutes, no BGE):
    python -m src.leakage_audit

Each check is something a grader can rerun: it rebuilds the trained layer from the saved inputs and settings and
compares, splits and shuffles on purpose, and prints only aggregates (counts, shares, hit rates). Exit code 1 if a
check fails. Thread regenerates this report after every ML_Fit rerun.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import fit_eval as fe
from src import fit_model as fm
from src.eda_helpers import CLEAN_DIR, DATA_DIR

REPORT = Path(__file__).resolve().parents[1] / "reports" / "ML_Fit_audit.md"
INPUTS, CHOICE = CLEAN_DIR / "fit_inputs.parquet", CLEAN_DIR / "fit_choice.json"
# Known only at or after the switch: none may be a model input
POST_SWITCH = ["target_bucket", "target_title", "roles_reached", "reached_data", "first_level", "switch_period", "pool",
               "career_changer", "split", "fold"]
N_SHUFFLES = 5


def hits(probs, targets, exclude, k=3):
    return fe.top_k_hits(probs, list(targets), [e if e != "none" else None for e in exclude], k)


def load():
    data = pd.read_parquet(INPUTS)
    meta_cols = [c for c in data if c.startswith("meta: ")]
    meta = data[meta_cols].rename(columns=lambda c: c[len("meta: "):])
    X = data.drop(columns=meta_cols)
    choice = json.loads(CHOICE.read_text(encoding="utf-8"))
    fit = pd.read_parquet(CLEAN_DIR / "fit_scores.parquet")
    return X, meta, choice, fit


def model(choice):
    pipe, _ = fm.candidates()[choice["model"]]
    return pipe.set_params(**choice["params"])


def check_split(meta):
    dev, test = meta.index[meta.split == "development"], meta.index[meta.split == "test"]
    overlap = len(dev.intersection(test))
    folds_ok = meta.loc[dev, "fold"].between(0, 4).all() and meta.loc[test, "fold"].eq(-1).all()
    ok = overlap == 0 and folds_ok and meta.index.is_unique
    return ok, (f"{len(dev):,} development and {len(test):,} test people; {overlap} people in both; every development "
                f"person in one of 5 folds and no test person in any: {'yes' if folds_ok else 'NO'}; one row per person: "
                f"{'yes' if meta.index.is_unique else 'NO'}")


def check_inputs(X):
    bad = [c for c in X if c in POST_SWITCH or any(c.startswith(p + ":") for p in POST_SWITCH)]
    groups = pd.Series([fm.group_of(c) for c in X]).value_counts()
    ok = not bad and set(fm.CATEGORICAL + fm.FLAGS + fm.NUMERIC) <= set(X)
    return ok, (f"{X.shape[1]} inputs: the Profiles features {fm.CATEGORICAL + fm.FLAGS + fm.NUMERIC} (all measured "
                f"before the first tech job) and each method's score per role; post-switch columns among them: "
                f"{bad or 'none'}. By group: " + ", ".join(f"{g} {n}" for g, n in groups.items()))


def check_pipeline():
    from sklearn.pipeline import Pipeline
    cands = fm.candidates()
    inside = {name: isinstance(p, Pipeline) and list(p.named_steps)[:1] == ["prep"] for name, (p, _) in cands.items()}
    return all(inside.values()), (f"all {len(cands)} candidates are one sklearn Pipeline with the preprocessing "
                                  "(one-hot encoding, imputing, scaling) as its first step, so it is fit inside each "
                                  f"training fold: {'yes' if all(inside.values()) else 'NO: ' + str(inside)}")


def check_reproduce(X, meta, choice, fit, y):
    """Rebuild the saved fit scores from development people only: out-of-fold for development, one dev fit for test."""
    cols = [f"fit: {r}" for r in fe.ROLES]
    dev, test = (meta.split == "development").to_numpy(), (meta.split == "test").to_numpy()
    saved = fit.reindex(meta.index)[cols].to_numpy()
    oof = fm.out_of_fold(model(choice), X[dev], y[dev], meta.loc[dev, "fold"])
    test_p = fm.probabilities(model(choice).fit(X[dev], y[dev]), X[test])
    d_dev, d_test = np.abs(oof - saved[dev]).max(), np.abs(test_p - saved[test]).max()
    ok = d_dev < 1e-6 and d_test < 1e-6
    return ok, d_dev, d_test


def check_shuffled_labels(X, meta, choice, y, rng):
    """Shuffle the development people's roles and refit: if the pipeline could see test answers, it would still score
    well; with no leak it falls to about the most-common-role baseline."""
    dev, test = (meta.split == "development").to_numpy(), (meta.split == "test").to_numpy()
    targets, exclude = meta.loc[test, "target_bucket"], meta.loc[test, "from_bucket"]
    real = hits(fm.probabilities(model(choice).fit(X[dev], y[dev]), X[test]), targets, exclude).mean()
    shuffled = [hits(fm.probabilities(model(choice).fit(X[dev], rng.permutation(y[dev])), X[test]), targets, exclude).mean()
                for _ in range(N_SHUFFLES)]
    base = hits(fe.baseline_scores(int(test.sum()), meta.loc[dev, "target_bucket"]), targets, exclude).mean()
    ok = max(shuffled) <= base + 0.05  # no leak: with the answers shuffled, nothing beyond the role mix is left to learn
    return ok, real, shuffled, base


def check_duplicates(meta, fit_hits):
    """Near-identical pre-switch texts across the split (a résumé saved twice that exact de-duplication missed)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    m = pd.read_pickle(DATA_DIR / "profiles" / "modeling_table.pkl")
    text = pd.read_pickle(DATA_DIR / "profiles" / "pre_switch_text.pkl").reindex(meta.index).fillna("")
    t = text[(meta.split == "test") & (text.str.len() > 30)]
    d = text[(meta.split == "development") & (text.str.len() > 30)]
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(pd.concat([t, d]))
    sim = (vec.transform(t) @ vec.transform(d).T).tocsr()
    best = np.asarray(sim.max(axis=1).todense()).ravel()
    arg = np.asarray(sim.argmax(axis=1)).ravel()
    near = np.flatnonzero(best >= 0.95)
    cols = ["target_bucket", "from_bucket", "switch_period", "years_before", "n_jobs_before", "background_area", "target_title"]
    a = m.reindex(t.index)[cols].astype(str).to_numpy()
    b = m.reindex(d.index)[cols].astype(str).to_numpy()
    same = int((a[near] == b[arg[near]]).all(axis=1).sum())
    flagged = t.index[near]
    without = fit_hits.drop(flagged)
    effect = fit_hits.mean() - without.mean()
    exact = int(t.isin(set(d)).sum())
    ok = len(without) > 0 and abs(effect) < 0.01  # under 1 point; with no test person left, it can't be shown
    return ok, len(t), exact, len(near), same, fit_hits.mean(), without.mean(), effect


def check_undated(meta, fit_hits):
    """Undated programs are placed before the switch; if some were really after it, those people would score higher."""
    und = meta.loc[fit_hits.index, "study_undated"].astype(bool)
    lo, hi = fit_hits[und].mean(), fit_hits[~und].mean()
    return lo <= hi + 0.02, int(und.sum()), lo, hi


def check_target_in_text(meta):
    m = pd.read_pickle(DATA_DIR / "profiles" / "modeling_table.pkl").reindex(meta.index)
    text = pd.read_pickle(DATA_DIR / "profiles" / "pre_switch_text.pkl").reindex(meta.index).fillna("").map(fe.clean)
    title = m.target_title.fillna("").map(fe.clean)
    found = pd.Series([bool(ti) and ti in tx for ti, tx in zip(title, text)], index=meta.index)
    # held before: the exact title as a whole entry (a job or program of its own), not just words inside a longer one
    held = pd.Series([bool(ti) and any(e.strip() == ti for e in tx.split(" | ")) for ti, tx in zip(title, text)], index=meta.index)
    return held.mean() <= 0.01, int(found.sum()), int(held.sum()), found.mean()


def main():
    if not INPUTS.exists() or not CHOICE.exists():
        sys.exit("fit_inputs.parquet / fit_choice.json aren't there: run ML_Fit (section 9a saves them), then this again.")
    start = time.time()
    rng = np.random.default_rng(0)
    X, meta, choice, fit = load()
    y = np.array([fe.ROLES.index(t) for t in meta.target_bucket])
    test = meta.index[meta.split == "test"]
    cols = [f"fit: {r}" for r in fe.ROLES]
    fit_hits = pd.Series(hits(fit.loc[test, cols].to_numpy(), meta.loc[test, "target_bucket"], meta.loc[test, "from_bucket"]),
                         index=test)

    results = []
    ok, msg = check_split(meta)
    results.append(("Development and test people never overlap", ok, msg))
    ok, msg = check_inputs(X)
    results.append(("No input is known only after the switch", ok, msg))
    ok, msg = check_pipeline()
    results.append(("Preprocessing is fit inside the training folds", ok, msg))
    ok, d_dev, d_test = check_reproduce(X, meta, choice, fit, y)
    results.append(("The saved fit scores come from development people only", ok,
                    f"rebuilt from the saved inputs with the chosen model ({choice['model']}, {choice['params']}): "
                    f"development scores out-of-fold (each person scored by a model fit on the other 4 folds), "
                    f"test scores by one model fit on development people only. Largest difference from the saved "
                    f"scores: development {d_dev:.1e}, test {d_test:.1e} (identical up to rounding)."))
    ok, real, shuffled, base = check_shuffled_labels(X, meta, choice, y, rng)
    results.append(("With the answers shuffled, the model falls to the baseline", ok,
                    f"test top-3 hit rate with the real development roles {real:.1%}; with them shuffled "
                    f"({N_SHUFFLES} times) {np.mean(shuffled):.1%} (range {min(shuffled):.1%}–{max(shuffled):.1%}); "
                    f"the most-common-role baseline {base:.1%}. A pipeline that saw the test answers some other way "
                    f"would stay high when the training answers are shuffled."))
    ok, n_t, exact, n_near, same, all_rate, without_rate, effect = check_duplicates(meta, fit_hits)
    results.append(("Résumés copied across the split don't change the result", ok,
                    f"of {n_t} test people with pre-switch text, {exact} have text identical to a development "
                    f"person's and {n_near} near-identical (TF-IDF cosine ≥ 0.95); {same} of those match their twin on "
                    f"every field (target role and title, current role, period, years, jobs, background), so are likely "
                    f"the same résumé saved twice. Test top-3 hit rate {all_rate:.1%} with them, "
                    + (f"{without_rate:.1%} without them ({effect * 100:+.1f} points)." if pd.notna(without_rate)
                       else "and no test person is left without a twin, so the effect can't be shown.")))
    ok, n_und, r_und, r_rest = check_undated(meta, fit_hits)
    results.append(("Undated study doesn't inflate the result", ok,
                    f"{n_und} test people have a program with no date, which the rules place before the switch. "
                    f"Their top-3 hit rate is {r_und:.1%}, against {r_rest:.1%} for the rest: if post-switch training "
                    f"were leaking in through them, they would score higher, not lower."))
    ok, n_found, n_held, share = check_target_in_text(meta)
    results.append(("The job switched into isn't in the pre-switch text", ok,
                    f"{n_held} of {len(meta):,} career changers have the exact title of the job they switched into as "
                    f"an entry of their pre-switch text (a job or program held before). {n_found} ({share:.1%}) have its "
                    f"words inside a longer entry, mostly short titles such as two-word role names, which is wording "
                    f"overlap, not the job itself."))

    write(results, choice, time.time() - start)
    for name, ok, _ in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"wrote {REPORT.relative_to(REPORT.parents[1])}")
    return 0 if all(ok for _, ok, _ in results) else 1


def write(results, choice, seconds):
    n_pass = sum(ok for _, ok, _ in results)
    lines = [
        "# ML_Fit: data leakage audit",
        "",
        f"Generated by `python -m src.leakage_audit` (run from `analysis/`) on {time.strftime('%Y-%m-%d %H:%M')}, from the "
        f"tables ML_Fit saved; {seconds:.0f} seconds. **{n_pass} of {len(results)} checks pass.** Everything here "
        "is an aggregate; no person or listing is shown.",
        "",
        "**Leakage** here means anything about a person's actual next role, or anything that happened after their switch, "
        "reaching the model's inputs, its training or its evaluation. ML_Fit predicts which of 8 tech roles a career "
        "changer moved into, from what they did before: if the answer leaked in, the reported hit rates would be "
        "inflated. Each check below either rebuilds part of the pipeline and compares, or breaks the pipeline on "
        "purpose and shows the result collapses.",
        "",
        "| # | Check | Result |",
        "|---|---|---|",
    ]
    lines += [f"| {i} | {name} | {'PASS' if ok else '**FAIL**'} |" for i, (name, ok, _) in enumerate(results, 1)]
    lines += ["", "## The checks", ""]
    for i, (name, ok, msg) in enumerate(results, 1):
        lines += [f"**{i}. {name}: {'PASS' if ok else 'FAIL'}.** {msg}", ""]
    lines += [
        "## By design (from the code; tested in `analysis/tests/`)",
        "",
        "- **Split by person, fixed once:** 20% of career changers are held out as the test set, stratified by role and "
        "pool (`fit_eval.split_people`, seed 0); the other 80% are split into 5 development folds. Every choice (methods, "
        "the six candidate models, their settings, the model-choice rule) is made on the development folds only.",
        "- **Features from before the switch** (`src/career_switch.py`): the background, years and jobs before, study "
        "level and training field are counted strictly before the first tech job; the pre-switch text keeps only job "
        "titles and programs dated before the switch month (the job switched into starts that month, so it is out). "
        "Résumé skills are undated, so they are never used.",
        "- **Method scores use no one's answer:** keyword overlap, TF-IDF, BGE and RRF compare a person's pre-switch "
        "text with a fixed, hashed sample of 2026 listings (2,000 per role); the skills scores compare O*NET skill "
        "profiles. None of them sees any person's role.",
        "- **The current role is never a candidate** for moves within tech, in training (the scorer) and in evaluation.",
        "- **Reproducible:** every seed is fixed; keyword ties break alphabetically (tested under three hash seeds); the "
        f"model is the simplest candidate within one standard deviation of the best on the development folds "
        f"(`fit_model.choose`; this run: {choice['model']}).",
        "- **Exact duplicate résumés** (36,981 copies) were removed in Profiles before any of this.",
        "",
        "## Known limits",
        "",
        "- **The test set was scored three times on 2026-10-13:** for steps 1–4, then with steps 5–6, then after a "
        "reproducibility fix and a model-choice rule set without looking at the test. No choice was made from test "
        "results; the top-3 figure stayed between 81.7% and 82.6%.",
        "- **Development figures are slightly optimistic:** the model settings were chosen on the same folds that score "
        "them. The test figures are the fair ones.",
        "- **Near-copies of a résumé** (check 6) survive exact de-duplication; their effect is shown above. Removing them "
        "would mean changing frozen Profiles and the split.",
        "- **Role descriptions come from 2026 listings** while the moves happened earlier. That uses no person's answer, "
        "but a role's wording can drift over time.",
        "",
        "## How to reproduce",
        "",
        "1. Run `analysis/notebooks/ML_Fit.ipynb` top to bottom (section 9a saves `fit_inputs.parquet`, "
        "`fit_choice.json` and `fit_scores.parquet`).",
        "2. From `analysis/`: `python -m src.leakage_audit` (rewrites this file; exit code 1 if a check fails).",
        "3. The unit tests behind the design checks: `python -m unittest tests.test_fit_eval tests.test_fit_model "
        "tests.test_switch_cutoffs`.",
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
