"""Step 7: the app's rankers scored on real career moves, with ML_Fit's own evaluation (analysis/STANDARDS.md,
Validation design). Run from notebooks/Evaluate.ipynb.

The answer key is ML_Fit's: Profiles' career changers, the same split (fit_eval.split_people, seed 0), the test people
scored once. Each ranker gets only a test person's pre-switch background, ranks the serving corpus, and its top
TOP_N listings become a role ranking: a role's score is how many of those listings are in it, ties broken by the best
position (role_scores). fit_eval.evaluate then scores hit rate @3 and @5 against the same most-common-role baseline,
per pool, with bootstrap intervals, leaving the current role out for moves within tech, as ML_Fit does.

The rankers, all over the same serving corpus (combined/corpus.py), as the apps serve them:
  - rrf:      Jesse's TF-IDF + BGE fusion (app/backend/rrf_ranker.py), unchanged;
  - mlrank:   Jacob's app: ML_Rank's queue over the text match (app/backend/ml_rank_ranker.py), unchanged;
  - combined: the combined model (steps 3-5), its trained-layer role fit from the test person's Profiles features,
    blended with the role mix of their text matches (v2; no desired roles: Profiles has none).
Each test person's background is given as a profile of their pre-switch entries (titles and programs, as in ML_Fit).
Prints and returns aggregates only.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from combined._paths import CORPUS_DIR, DATA_DIR, REPO_ROOT

if str(REPO_ROOT / "app") not in sys.path:
    sys.path.append(str(REPO_ROOT / "app"))  # Jesse's backend (rrf_ranker, ml_rank_ranker), imported unchanged

TOP_N = 20  # listings per person turned into roles: the app's first page
PROFILES_DIR = DATA_DIR / "profiles"

RankOne = Callable[[str, str, pd.Series], Sequence[int]]  # (person id, pre-switch text, person row) -> listing positions


def load_people(profiles_dir=PROFILES_DIR):
    """ML_Fit 2a: the career changers moving into a fit role, their pre-switch text and the split.
    Returns (people, dev index, test index)."""
    from src import fit_eval as fe

    modeling = pd.read_pickle(profiles_dir / "modeling_table.pkl")
    pre_text = pd.read_pickle(profiles_dir / "pre_switch_text.pkl")
    people = modeling[modeling.career_changer & modeling.target_bucket.isin(fe.ROLES)].copy()
    people["text"] = pre_text.reindex(people.index).fillna("")
    dev, test, _ = fe.split_people(people, test_share=0.2, folds=5, seed=0)
    return people, dev, test


def role_scores(listing_roles: Sequence[Sequence[Any]], roles: Sequence[str], n: int = TOP_N) -> np.ndarray:
    """people x roles: how many of each person's top n listings are in each role, plus a tie-break under 1 for the best
    position (earlier = higher). Listings outside `roles` don't count; a role with none scores 0 (fit_eval.ranks then
    orders those by the fixed role order, never by the data)."""
    out = np.zeros((len(listing_roles), len(roles)))
    col = {r: j for j, r in enumerate(roles)}
    for i, top in enumerate(listing_roles):
        best = {}
        for pos, role in enumerate(list(top)[:n]):
            if role in col:
                out[i, col[role]] += 1
                best.setdefault(role, pos)
        for role, pos in best.items():
            out[i, col[role]] += (n - pos) / (n + 1) / n  # < 1/n, so a count always outweighs any tie-break
    return out


def text_profile(person_id: str, text: str):
    """The app's candidate profile for a background given as text: one experience entry per title or program."""
    import job_matching_engine as matching

    parts = tuple(p.strip() for p in str(text or "").split(" | ") if p.strip())
    return matching.CandidateProfile(profile_id=str(person_id), experience=parts)


def score(people: pd.DataFrame, rankers: Mapping[str, RankOne], listing_roles: Sequence[Any], roles: Sequence[str],
          n: int = TOP_N, progress_every: int = 100) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, list]]:
    """Run every ranker on every person; returns ({ranker: people x roles}, {ranker: seconds per person},
    {ranker: each person's top-n listing roles}, kept so a smaller n can be scored without running again)."""
    scores, seconds, kept = {}, {}, {}
    for name, rank_one in rankers.items():
        started, tops = time.time(), []
        for k, (pid, row) in enumerate(people.iterrows(), 1):
            tops.append([listing_roles[i] for i in rank_one(str(pid), row["text"], row)][:n])
            if progress_every and k % progress_every == 0:
                print(f"{name}: {k:,} of {len(people):,} people, {time.time() - started:.0f}s", flush=True)
        scores[name] = role_scores(tops, roles, n)
        seconds[name] = (time.time() - started) / max(len(people), 1)
        kept[name] = tops
    return scores, seconds, kept


class AppRankers:
    """The three app rankers over the serving corpus, sharing one RRF ranker (loading it fits TF-IDF over the corpus)."""

    def __init__(self, corpus_dir=CORPUS_DIR, bge: Any = None):
        import job_matching_engine as matching
        from backend.ml_rank_ranker import MlRankRankingBackend
        from backend.rrf_ranker import RrfRankingBackend

        from combined.candidates import ListingIndex
        from combined.queue import Queue
        from combined.model import CombinedModel
        from combined.role_fit import RoleFit

        jobs, _ = matching.load_prepared_jobs(corpus_dir / "jobs_prepared_corpus.jsonl")
        self.base = RrfRankingBackend(jobs, corpus_dir)
        self.mlrank = MlRankRankingBackend.from_handoff(self.base)
        from combined.candidates import titles_and_companies

        titles, companies = titles_and_companies(self.base.jobs)
        self.listings = ListingIndex.from_handoff(self.base.job_ids, titles=titles, companies=companies)
        self.queue = Queue.from_handoff(self.listings)
        self.role_fit = RoleFit.load(bge=bge)
        self.model = CombinedModel(self.role_fit, self.listings, self.queue)
        self.everyone = list(range(len(self.base.job_ids)))
        self.position = {j: i for i, j in enumerate(self.base.job_ids)}

    def trained_layer(self, people: pd.DataFrame) -> np.ndarray:
        """people x roles: the saved trained layer's probabilities (ML_Fit's hand-off), the anchor: on ML_Fit's test
        people it must reproduce ML_Fit 6b's hit rates."""
        from src import fit_handoff as fh

        rf = self.role_fit
        return fh.probabilities(rf.layer, rf.scorer.features(people, rf.bge), rf.manifest["feature_columns"])

    def fit_ratings(self, people: pd.DataFrame, probs: np.ndarray | None = None) -> pd.DataFrame:
        """people x roles: the combined model's role fit (step 3) from each person's Profiles features, best = 100."""
        from src import rank_model as rm

        probs = self.trained_layer(people) if probs is None else probs
        return pd.DataFrame(rm.fit_ratings(probs).to_numpy(), index=people.index, columns=self.role_fit.roles_order)

    def rankers(self, people: pd.DataFrame, n: int = TOP_N, probs: np.ndarray | None = None) -> dict[str, RankOne]:
        from combined.role_fit import RoleFitResult

        ratings = self.fit_ratings(people, probs)

        def positions(results):
            return [self.position[str(r["job_id"])] for r in results]

        def rrf(pid, text, row):
            return positions(self.base.rank(text_profile(pid, text), n, self.everyone))

        def mlrank(pid, text, row):
            return positions(self.mlrank.rank(text_profile(pid, text), n, self.everyone))

        model_probs = pd.DataFrame(self.trained_layer(people) if probs is None else probs, index=people.index,
                                   columns=self.role_fit.roles_order)

        def combined(pid, text, row):
            profile = text_profile(pid, text)
            p = model_probs.loc[pid]
            table = pd.DataFrame({"probability": p, "score": p, "rating": ratings.loc[pid]})
            model_roles = RoleFitResult(table.sort_values("score", ascending=False, kind="stable"), [], "profiles", 0)
            result = self.model.recommend_from(model_roles, lambda m, idx: self.base.rank(profile, m, idx), limit=n)
            return list(result.queue["index"])

        return {"rrf (Jesse's app)": rrf, "mlrank (Jacob's app)": mlrank, "combined model": combined}
