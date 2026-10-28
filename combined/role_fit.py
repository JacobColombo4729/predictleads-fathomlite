"""The combined model's step 3: a person's fit for every role, at serving time, as ML_Fit and ML_Rank define it.

    fit = RoleFit.load()                         ML_Fit's hand-off (data/clean/ml_fit/, written by ML_Fit 9b) and BGE
    fit.roles(entries, growth=None, growth_weight=0.0, as_of=None)
        -> RoleFitResult: per role, the trained layer's probability, the growth-adjusted score, the fit rating (best role
           = 100, ML_Rank's rm.fit_ratings: what the queue's weights were tuned on) and why (ML_Fit 7a's grouped SHAP);
           plus the person's case and how many entries had no readable date.

Every step is the analysis code itself: src.fit_serving (résumé -> Profiles' features), src.fit_handoff (the saved
scorer and layer), src.fit_model (growth, SHAP groups), src.rank_model (the fit rating). Growth is applied after the
model, never trained in (ML_Fit 7b); its weight is 0 until BLS projections exist, and the app shows it as a slider.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combined._paths import DATA_DIR

FIT_DIR = DATA_DIR / "clean" / "ml_fit"
TEXT_WEIGHT = 0.5  # the text match's share of the role choice (v2): an equal blend, fixed before any result was seen


def role_of_title(title: str) -> str | None:
    """A job title's role bucket by Profiles' own title rules (career_switch RULES["cm_buckets"], first match wins)."""
    import re

    from src import career_switch as cs

    low = str(title or "").lower()
    return next((bucket for bucket, pattern in cs.RULES["cm_buckets"].items() if re.search(pattern, low)), None)


@dataclass
class RoleFitResult:
    table: pd.DataFrame  # one row per role, best first: probability, score, rating, and one column per reason group
    reason_groups: list[str]
    case: str
    undated_entries: int

    def top(self, k: int = 3) -> list[str]:
        return list(self.table.index[:k])

    def ratings(self) -> dict[str, float]:
        return self.table["rating"].to_dict()

    def with_evidence(self, text_mix: Mapping[str, float], desired_roles: Sequence[str] = (),
                      text_weight: float = TEXT_WEIGHT) -> "RoleFitResult":
        """The roles from the whole profile, not the trained layer alone (v2). The trained layer reads only titles,
        programs and dates (all Profiles' resumes have), so a profile's skills and descriptions never reach it. Here:
          score = (1 - w) x the trained layer's score + w x the role's share of the person's top text matches (which read
          the whole profile: titles, descriptions, skills), w = 0.5, fixed before any result was seen;
        and the roles the person asked for (desired roles, matched to a role by Profiles' title rules) always come first.
        The rating is recomputed (best = 100); the trained layer's own score stays in "model score"."""
        from src import rank_model as rm

        t = self.table.copy()
        t["model score"] = t["score"]
        t["text match"] = [float(text_mix.get(r, 0.0)) for r in t.index]
        wanted = {role_of_title(d) for d in desired_roles} - {None}
        t["desired"] = [r in wanted for r in t.index]
        t["score"] = (1 - text_weight) * t["model score"] + text_weight * t["text match"] + t["desired"].astype(float)
        t["rating"] = rm.fit_ratings(t["score"].to_numpy()[None, :]).to_numpy()[0]
        t = t.sort_values("score", ascending=False, kind="stable")
        return RoleFitResult(t, self.reason_groups, self.case, self.undated_entries)


class RoleFit:
    def __init__(self, scorer: Any, layer: Any, manifest: Mapping[str, Any], bge: Any):
        self.scorer, self.layer, self.manifest, self.bge = scorer, layer, manifest, bge

    @classmethod
    def load(cls, directory: Path = FIT_DIR, bge: Any = None) -> "RoleFit":
        from src import fit_eval as fe
        from src import fit_handoff as fh

        if not (Path(directory) / "manifest.json").exists():
            raise FileNotFoundError("ML_Fit's hand-off is missing: run ML_Fit 9b first.")
        scorer, layer, manifest = fh.load(directory)
        return cls(scorer, layer, manifest, bge if bge is not None else fe.bge_model())

    @property
    def roles_order(self) -> list[str]:
        return list(self.manifest["roles"])

    def roles(self, entries: Sequence[Mapping[str, Any]], growth: Mapping[str, float] | None = None,
              growth_weight: float = 0.0, as_of: Any = None) -> RoleFitResult:
        from src import fit_handoff as fh
        from src import fit_model as fm
        from src import fit_serving as fs
        from src import rank_model as rm

        person = fs.person_features(entries, as_of=as_of)
        X = self.scorer.features(person, self.bge)
        probs = fh.probabilities(self.layer, X, self.manifest["feature_columns"])
        score = fm.growth_adjust(probs, growth or {}, growth_weight) if growth else probs
        rating = rm.fit_ratings(score).to_numpy()[0]
        by_group, groups = fh.reasons(self.layer, X, self.manifest)
        table = pd.DataFrame({"probability": probs[0], "score": score[0], "rating": rating}, index=self.roles_order)
        table = table.join(pd.DataFrame(by_group[0], index=self.roles_order, columns=[f"why: {g}" for g in groups]))
        table.index.name = "role"
        return RoleFitResult(table.sort_values("score", ascending=False, kind="stable"), groups,
                             str(person.iloc[0]["case"]), int(person.iloc[0]["undated_entries"]))
