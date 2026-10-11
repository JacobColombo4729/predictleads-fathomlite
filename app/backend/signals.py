"""Read-only access to the precomputed hiring signals (see docs/hiring-signals.md).

The tables are built offline by code/hiring_signals.py and loaded once at startup. Nothing here is computed per request.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_ROOT = ROOT / "data" / "signals"
EPOCH = date(2026, 1, 1)  # day numbers in the tables count from here
FAMILIES = ("ai_ml", "data", "technical_sales")
PATTERNS = ("burst", "sustained", "cooling", "quiet")
SORTS = ("new_30", "active", "surge_z")


def default_signals_dir() -> Path | None:
    override = os.environ.get("PREDICTLEADS_SIGNALS_DIR")
    if override:
        return Path(override).expanduser()
    found = sorted(SIGNALS_ROOT.glob("hiring_signals_*")) if SIGNALS_ROOT.exists() else []
    return found[-1] if found else None


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


@dataclass
class SignalStore:
    companies: pd.DataFrame
    weekly: dict[str, list[int]] = field(default_factory=dict)
    week_ends: list[str] = field(default_factory=list)
    intent: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tech: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    carry_over: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path) -> "SignalStore":
        companies = pd.read_csv(directory / "company_signals.csv").dropna(subset=["company_id"])
        companies = companies.set_index("company_id", drop=False)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8")) if (directory / "summary.json").exists() else {}
        carry: list[dict[str, Any]] = []
        if (directory / "backtest.json").exists():
            carry = json.loads((directory / "backtest.json").read_text(encoding="utf-8")).get("by_pattern", [])

        # Sparklines are only needed for companies that carry a signal; quiet ones are skipped to keep memory small.
        interesting = set(companies.index[companies["pattern"] != "quiet"])
        intent: dict[str, list[dict[str, Any]]] = {}
        if (directory / "intent_signals.csv").exists():
            for row in pd.read_csv(directory / "intent_signals.csv").to_dict("records"):
                cid = str(row["company_id"])
                interesting.add(cid)
                evidence = json.loads(row["evidence"]) if isinstance(row.get("evidence"), str) else []
                intent.setdefault(cid, []).append({
                    "family": row["family"], "signal": row["signal"], "meaning": row["meaning"],
                    "new_30": _clean(row["new_30"]), "baseline_monthly": _clean(row["baseline_monthly"]),
                    "top_tech": [t for t in str(row["top_tech"]).split("|") if t and t != "nan"] if isinstance(row.get("top_tech"), str) else [],
                    "evidence": evidence,
                })

        weekly: dict[str, list[int]] = {}
        week_ends: list[str] = []
        if (directory / "company_weekly.csv").exists():
            frame = pd.read_csv(directory / "company_weekly.csv", usecols=["week_end", "new", "company_id"])
            days = sorted(frame["week_end"].unique())
            week_ends = [(EPOCH + timedelta(days=int(d))).isoformat() for d in days]
            frame = frame[frame["company_id"].isin(interesting)]
            wide = frame.pivot_table(index="company_id", columns="week_end", values="new", aggfunc="sum", fill_value=0).reindex(columns=days, fill_value=0)
            weekly = {cid: [int(v) for v in row] for cid, row in zip(wide.index, wide.to_numpy())}

        tech: dict[str, list[dict[str, Any]]] = {}
        if (directory / "technology_leads.csv").exists():
            leads = pd.read_csv(directory / "technology_leads.csv")
            leads = leads[leads["company_id"].isin(interesting)].sort_values("new_30", ascending=False)
            for row in leads.to_dict("records"):
                tech.setdefault(str(row["company_id"]), []).append({"technology": row["technology"], "new_30": int(row["new_30"]), "still_observed": int(row["still_observed"])})
        return cls(companies, weekly, week_ends, intent, tech, summary, carry)

    # ----- views -----

    def _row(self, row: pd.Series, detail: bool = False) -> dict[str, Any]:
        cid = str(row["company_id"])
        out: dict[str, Any] = {
            "company_id": cid, "name": row["company_name"] if isinstance(row["company_name"], str) else None,
            "domain": row["domain"] if isinstance(row["domain"], str) else None,
            "pattern": row["pattern"], "new_30": int(row["new_30"]), "active": int(row["active"]),
            "gone_30": int(row["gone_30"]), "baseline_monthly": _clean(round(float(row["baseline"]), 2)),
            "surge_z": _clean(round(float(row["surge_z"]), 1)), "enough_history": bool(row["enough_history"]),
            "families": {f: {"new_30": int(row[f"new_30_{f}"]), "active": int(row[f"active_{f}"])} for f in FAMILIES},
            "weekly": self.weekly.get(cid, []),
            "intent": [{k: v for k, v in s.items() if k != "evidence"} for s in self.intent.get(cid, [])],
        }
        if detail:
            out["evidence"] = [e for s in self.intent.get(cid, []) for e in s["evidence"]]
            out["technologies"] = self.tech.get(cid, [])[:15]
            out["week_ends"] = self.week_ends
        return out

    def leaderboard(self, pattern: str | None, family: str | None, query: str | None, sort: str, limit: int, offset: int) -> dict[str, Any]:
        data = self.companies
        if pattern:
            data = data[data["pattern"] == pattern]
        if family:
            data = data[data[f"new_30_{family}"] > 0]
        if query:
            q = query.casefold()
            data = data[data["company_name"].fillna("").str.casefold().str.contains(q, regex=False) | data["domain"].fillna("").str.casefold().str.contains(q, regex=False)]
        sort_col = {"new_30": f"new_30_{family}" if family else "new_30", "active": f"active_{family}" if family else "active", "surge_z": "surge_z"}[sort]
        data = data.sort_values(sort_col, ascending=False)
        page = data.iloc[offset: offset + limit]
        return {"total": int(len(data)), "companies": [self._row(r) for _, r in page.iterrows()]}

    def company(self, company_id: str) -> dict[str, Any] | None:
        if company_id not in self.companies.index:
            return None
        return self._row(self.companies.loc[company_id], detail=True)

    def patterns(self) -> dict[str, str]:
        """company_id -> pattern for the companies worth a badge on a job card (burst, sustained or cooling)."""
        sub = self.companies[self.companies["pattern"].isin(["burst", "sustained", "cooling"])]
        return dict(zip(sub.index.astype(str), sub["pattern"]))

    def overview(self) -> dict[str, Any]:
        counts = self.companies["pattern"].value_counts().to_dict()
        return {
            "snapshot": self.summary.get("snapshot"), "companies": int(len(self.companies)),
            "pattern_counts": {p: int(counts.get(p, 0)) for p in PATTERNS},
            "week_ends": self.week_ends, "carry_over": self.carry_over,
        }
