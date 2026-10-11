"""Questions about the hiring market, answered only from our own tables (see docs/ask-the-market.md).

A question becomes a MarketQuery (by a language model, or by a preset), the query is validated, and this module runs it on the
per-listing market table and the company signals. Nothing here guesses: a filter that cannot be resolved (an unknown technology,
an unknown company) is reported back instead of being replaced by a similar one, and every number is a count or a statistic of
real listings. Dates are when the crawler saw a listing, not when it was posted (docs/signal-semantics.md).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))
import hiring_signals as hs  # noqa: E402

EPOCH = date(2026, 1, 1)
FAMILIES = ("ai_ml", "data", "technical_sales")
FAMILY_LABEL = {"ai_ml": "AI / ML", "data": "Data", "technical_sales": "Technical sales"}
SENIORITIES = ("junior", "mid_senior", "manager", "director", "head", "vice_president", "c_level", "founder")
SENIORITY_LABEL = {"junior": "Junior", "mid_senior": "Mid / senior", "manager": "Manager", "director": "Director", "head": "Head of",
                   "vice_president": "Vice president", "c_level": "C-level", "founder": "Founder"}
PATTERNS = ("burst", "sustained", "cooling")
TASKS = ("companies", "technologies", "listings", "trend", "pay", "compare")
WEEKS = 26
MAX_ROWS = 50

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
# Common short names for tools, matched only when the full name exists in the data. Nothing else is matched by similarity.
TECH_ALIASES = {
    "aws": "amazon web services", "gcp": "google cloud platform", "azure": "microsoft azure", "powerbi": "power bi",
    "sklearn": "scikit-learn", "k8s": "kubernetes", "postgres": "postgresql", "js": "javascript", "ts": "typescript",
    "tf": "tensorflow", "excel": "microsoft excel", "dotnet": ".net",
}
_COMPANY_SUFFIX = re.compile(r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|gmbh|sa|plc|ag|bv|lp|llp|the)\b\.?", re.IGNORECASE)


# Questions we refuse before any AI is used, because the data cannot answer them honestly: forecasts and advice.
_FORECAST = re.compile(r"\b(will|won't|going to|gonna|predict\w*|forecast\w*|projected|expected to|likely to|probably|next (week|month|quarter|year)|in the future|coming (weeks|months)|by (next|the end of))\b", re.IGNORECASE)
_ADVICE = re.compile(r"\b(should i|best compan\w+|worth applying|good place to work|is it a good|recommend\w*|who would hire me|chances? of)\b", re.IGNORECASE)


def refuse_in_advance(question: str) -> str | None:
    """A reason this question cannot be answered from the data, or None. Forecasts and advice are never answered."""
    if _FORECAST.search(question):
        return "We can't predict future hiring. Our data shows what has already been seen; ask who is hiring now, or how hiring has moved so far."
    if _ADVICE.search(question):
        return "We don't give advice or rankings of employers. Our data shows listings and hiring patterns; ask about those."
    return None


_NAMED_FIELDS = ("states", "cities", "countries", "technologies", "companies", "title_words")
_STATE_CODE = {name.casefold(): code for code, name in US_STATES.items()}


def ground_in_question(fields: Mapping[str, Any], question: str) -> tuple[dict[str, Any], list[str]]:
    """Keep only the places, technologies, companies and title words that the question itself names.

    A language model sometimes adds a filter that sounds sensible (a state for a city, a company it assumes you mean). A search
    is only as honest as its filters, so anything not literally in the question is removed, and we say so.
    """
    text = question.casefold()
    words = set(re.findall(r"[a-z0-9+#.]+", text))
    cleaned = dict(fields)
    dropped: list[str] = []
    for key in _NAMED_FIELDS:
        kept: list[str] = []
        for item in cleaned.get(key) or []:
            name = str(item).casefold().strip()
            code = _STATE_CODE.get(name) if key == "states" else None
            named = name in text or (code is not None and code.casefold() in words)
            if not named and key == "title_words":  # "engineer" is named by "engineers"
                named = any(w.startswith(name) or name.startswith(w) for w in words if len(w) >= 4 and len(name) >= 4)
            (kept if named else dropped).append(item)
        cleaned[key] = kept
    notes = [f"Left out {', '.join(repr(str(d)) for d in dropped)} because your question did not name it."] if dropped else []
    return cleaned, notes


def day_to_date(day: int) -> date:
    return EPOCH + timedelta(days=int(day))


def _clean_list(values: Iterable[str], limit: int = 8, length: int = 60) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        text = " ".join(str(value).split())[:length]
        if text and text.casefold() not in seen:
            seen[text.casefold()] = text
    return list(seen.values())[:limit]


class MarketQuery(BaseModel):
    """What a question asks for, in a form we can check. Everything is optional except the task."""

    model_config = ConfigDict(extra="forbid")
    task: Literal["companies", "technologies", "listings", "trend", "pay", "compare"] = "companies"
    role_family: list[Literal["ai_ml", "data", "technical_sales"]] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    title_words: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    technology_match: Literal["any", "all"] = "any"
    seniority: list[Literal["junior", "mid_senior", "manager", "director", "head", "vice_president", "c_level", "founder"]] = Field(default_factory=list)
    patterns: list[Literal["burst", "sustained", "cooling"]] = Field(default_factory=list)
    window_days: int = 30
    first_time_hiring: bool = False
    companies: list[str] = Field(default_factory=list)
    only_my_saved: bool = False
    sort: Literal["new", "active", "surge"] = "new"
    limit: int = 20

    @field_validator("role_family", mode="after")
    @classmethod
    def _all_families_is_no_filter(cls, value: list[str]) -> list[str]:
        return [] if set(value) >= set(FAMILIES) else value

    @field_validator("states", mode="before")
    @classmethod
    def _states(cls, value: Any) -> list[str]:
        out = []
        for item in _clean_list(value or [], 10):
            out.append(US_STATES.get(item.upper(), item) if len(item) == 2 else item)
        return out

    @field_validator("cities", "countries", "title_words", "technologies", "companies", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> list[str]:
        return _clean_list(value or [], 10)

    @field_validator("window_days", mode="before")
    @classmethod
    def _window(cls, value: Any) -> int:
        return max(7, min(90, int(value or 30)))

    @field_validator("limit", mode="before")
    @classmethod
    def _limit(cls, value: Any) -> int:
        return max(1, min(MAX_ROWS, int(value or 20)))


def _normal_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _COMPANY_SUFFIX.sub(" ", name.casefold())).strip()


class MarketEngine:
    def __init__(self, listings: pd.DataFrame, signals: pd.DataFrame, cutoff: int):
        self.cutoff = int(cutoff)
        self.listings = listings.reset_index(drop=True)
        self.signals = signals.set_index("company_id", drop=False)
        t = self.listings
        self._tags = ("|" + t["tags"].fillna("").str.casefold() + "|")
        self._title = t["title"].fillna("").str.casefold()
        self._state = t["state"].fillna("").str.casefold()
        self._city = t["city"].fillna("").str.casefold()
        self._country = t["country"].fillna("").str.casefold()
        self.new_mask = lambda window: (t["first_day"] > self.cutoff - window) & (t["first_day"] <= self.cutoff) & ~t["batch_start"]
        self.active_mask = (t["last_day"] >= self.cutoff) & (t["first_day"] <= self.cutoff)
        # vocabulary of technology tags: lowercase key -> (display spelling, listings)
        counts = t["tags"].dropna().str.split("|").explode().value_counts()
        spelling: dict[str, tuple[str, int]] = {}  # key -> (most common spelling, its count)
        totals: Counter[str] = Counter()
        for tag, n in counts.items():
            key = str(tag).casefold()
            if key in hs._NOT_TECH:
                continue
            totals[key] += int(n)
            if key not in spelling or n > spelling[key][1]:
                spelling[key] = (str(tag), int(n))
        self.vocab = {key: (spelling[key][0], totals[key]) for key in totals}
        # company names, normalised, for matching names the way people write them
        names = self.signals[["company_id", "company_name", "domain"]].dropna(subset=["company_name"])
        sizes = t.groupby("company_id").size()
        self._company_index: dict[str, list[tuple[int, str]]] = {}
        for row in names.itertuples():
            self._company_index.setdefault(_normal_name(str(row.company_name)), []).append((int(sizes.get(row.company_id, 0)), row.company_id))
            if isinstance(row.domain, str) and row.domain:
                self._company_index.setdefault(row.domain.casefold().split(".")[0], []).append((int(sizes.get(row.company_id, 0)), row.company_id))

    # ---- resolving words to things that exist -------------------------------------------------------------------------

    def resolve_technologies(self, words: Sequence[str]) -> tuple[list[str], dict[str, list[str]]]:
        """Technology names that exist in the data, and for the others the closest real names (never substituted silently)."""
        found: list[str] = []
        missing: dict[str, list[str]] = {}
        for word in words:
            key = word.casefold().strip()
            key = TECH_ALIASES.get(key, key)
            if key in self.vocab:
                found.append(self.vocab[key][0])
                continue
            squashed = re.sub(r"[^a-z0-9+#.]", "", key)
            match = next((v for k, v in self.vocab.items() if re.sub(r"[^a-z0-9+#.]", "", k) == squashed), None)
            if match:
                found.append(match[0])
                continue
            near = sorted((v for k, v in self.vocab.items() if key and key in k), key=lambda v: -v[1])[:3]
            missing[word] = [n[0] for n in near]
        return list(dict.fromkeys(found)), missing

    def resolve_companies(self, names: Sequence[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
        """Company ids for the names people wrote, the names that match nothing, and notes where several companies shared a name."""
        ids: list[str] = []
        missing: dict[str, list[str]] = {}
        notes: list[str] = []
        for name in names:
            key = _normal_name(name)
            hits = self._company_index.get(key, [])
            if hits:
                hits = sorted(set(hits), reverse=True)
                ids.append(hits[0][1])
                if len(hits) > 1:
                    notes.append(f"More than one company is called “{name}”; showing the one with the most listings.")
                continue
            near = [k for k in self._company_index if key and k.startswith(key)][:3]
            missing[name] = [self.signals.loc[self._company_index[k][0][1], "company_name"] for k in near]
        return list(dict.fromkeys(ids)), missing, notes

    # ---- filters ---------------------------------------------------------------------------------------------------------------

    def _mask(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None, family_only: bool = False) -> pd.Series:
        t = self.listings
        mask = pd.Series(True, index=t.index)
        if q.role_family:
            mask &= t["family"].isin(q.role_family)
        if family_only:
            return mask
        if q.states:
            mask &= self._state.isin([s.casefold() for s in q.states])
        if q.cities:
            mask &= self._city.isin([c.casefold() for c in q.cities])
        if q.countries:
            mask &= self._country.isin([c.casefold() for c in q.countries])
        for word in q.title_words:
            mask &= self._title.str.contains(re.escape(word.casefold()), regex=True)
        if technologies:
            hits = [self._tags.str.contains("|" + tech.casefold() + "|", regex=False) for tech in technologies]
            mask &= (np.logical_and.reduce(hits) if q.technology_match == "all" else np.logical_or.reduce(hits))
        if q.seniority:
            mask &= t["seniority"].isin(q.seniority)
        if company_ids is not None:
            mask &= t["company_id"].isin(company_ids)
        return mask

    # ---- the answer -----------------------------------------------------------------------------------------------------------

    def run(self, q: MarketQuery, saved_company_ids: Sequence[str] | None = None) -> dict[str, Any]:
        notes: list[str] = []
        technologies, missing_tech = self.resolve_technologies(q.technologies)
        for word, near in missing_tech.items():
            notes.append(f"No listing carries the technology “{word}”." + (f" Closest names in the data: {', '.join(near)}." if near else ""))
        if q.technologies and not technologies:
            return self._empty(q, notes + ["Nothing was searched, because none of the technologies you named exist in the data."], [])
        company_ids: list[str] | None = None
        if q.task == "compare" or q.companies:
            company_ids, missing_names, name_notes = self.resolve_companies(q.companies)
            notes += name_notes
            for name, near in missing_names.items():
                notes.append(f"No company named “{name}” in the data." + (f" Did you mean {', '.join(near)}?" if near else ""))
            if not company_ids:
                return self._empty(q, notes + ["Nothing was searched, because none of the companies you named are in the data."], [])
        if q.only_my_saved:
            mine = list(saved_company_ids or [])
            if not mine:
                return self._empty(q, notes + ["You have no saved jobs yet, so there are no saved companies to look at."], [])
            company_ids = [c for c in company_ids if c in mine] if company_ids is not None else mine
            if not company_ids:
                return self._empty(q, notes + ["None of the companies you named are among your saved jobs."], [])

        handler = {"companies": self._companies, "compare": self._compare, "technologies": self._technologies,
                   "listings": self._listings, "trend": self._trend, "pay": self._pay}[q.task]
        result = handler(q, technologies, company_ids)
        result["notes"] = notes + result.get("notes", [])
        result["task"] = q.task
        result["interpretation"] = self.interpretation(q, technologies)
        result["as_of"] = day_to_date(self.cutoff).isoformat()
        return result

    def interpretation(self, q: MarketQuery, technologies: list[str]) -> list[dict[str, Any]]:
        """How the question was read, as chips. `fields` names the query fields a chip controls, so a chip can be removed to loosen the search."""
        items: list[dict[str, Any]] = []

        def add(label: str, value: str, *fields: str) -> None:
            items.append({"label": label, "value": value, "fields": list(fields)})

        task = {"companies": "Companies hiring", "technologies": "Technologies in listings", "listings": "Example openings",
                "trend": "New openings per week", "pay": "Pay stated in listings", "compare": "Compare companies"}[q.task]
        add("Showing", task)
        if q.role_family:
            add("Role family", ", ".join(FAMILY_LABEL[f] for f in q.role_family), "role_family")
        place = q.cities + q.states + q.countries
        if place:
            add("Place", ", ".join(place), "cities", "states", "countries")
        if q.title_words:
            add("Title contains", ", ".join(q.title_words), "title_words")
        if technologies:
            add("Technology", (" and " if q.technology_match == "all" else " or ").join(technologies), "technologies")
        if q.seniority:
            add("Level", ", ".join(SENIORITY_LABEL[s] for s in q.seniority), "seniority")
        if q.patterns:
            add("Company pattern", ", ".join(q.patterns), "patterns")
        if q.first_time_hiring:
            add("Only", "companies whose first opening in this role family appeared in the window", "first_time_hiring")
        if q.companies:
            add("Companies", ", ".join(q.companies))
        if q.only_my_saved:
            add("Limited to", "companies of your saved jobs", "only_my_saved")
        if q.task in ("companies", "technologies", "listings", "compare"):
            add("Window", f"last {q.window_days} days")
        return items

    def _empty(self, q: MarketQuery, notes: list[str], rows: list[Any]) -> dict[str, Any]:
        return {"task": q.task, "rows": rows, "summary": {}, "notes": notes, "interpretation": self.interpretation(q, []),
                "as_of": day_to_date(self.cutoff).isoformat()}

    # ---- helpers --------------------------------------------------------------------------------------------------------------

    def _weekly(self, subset: pd.DataFrame, by: str | None = None) -> Any:
        clean = subset[~subset["batch_start"] & (subset["first_day"] <= self.cutoff)]
        age = (self.cutoff - clean["first_day"]) // 7
        clean = clean[(age >= 0) & (age < WEEKS)]
        weeks = (WEEKS - 1 - ((self.cutoff - clean["first_day"]) // 7)).astype(int)
        if by is None:
            return np.bincount(weeks, minlength=WEEKS).tolist()
        out: dict[str, list[int]] = {}
        for key, group in clean.assign(_w=weeks).groupby(by):
            out[str(key)] = np.bincount(group["_w"], minlength=WEEKS).tolist()
        return out

    def week_labels(self) -> list[str]:
        return [day_to_date(self.cutoff - 7 * (WEEKS - 1 - i)).isoformat() for i in range(WEEKS)]

    def _pay_text(self, row: Any) -> str | None:
        low, high, unit = row.salary_low_usd, row.salary_high_usd, row.salary_unit
        if pd.isna(low) and pd.isna(high):
            return None
        figures = [f"${v:,.0f}" for v in (low, high) if not pd.isna(v)]
        return f"{' - '.join(dict.fromkeys(figures))} per {unit}" if isinstance(unit, str) else " - ".join(figures)

    def _place_text(self, row: Any) -> str | None:
        parts = [p for p in (row.city, row.state, row.country) if isinstance(p, str) and p]
        return ", ".join(parts[:2]) if parts else None

    def _company_rows(self, subset: pd.DataFrame, q: MarketQuery) -> pd.DataFrame:
        """New openings in the window, openings still observed, and listings, per company, for companies with any of the first two."""
        new = self.new_mask(q.window_days).loc[subset.index]
        active = self.active_mask.loc[subset.index]
        grouped = pd.DataFrame({"company_id": subset["company_id"], "new": new, "active": active, "total": 1}).groupby("company_id").sum()
        return grouped[(grouped["new"] > 0) | (grouped["active"] > 0)]

    def _decorate(self, grouped: pd.DataFrame, subset: pd.DataFrame, q: MarketQuery, technologies: list[str]) -> list[dict[str, Any]]:
        ids = list(grouped.index)
        sub = subset[subset["company_id"].isin(ids)]
        recent = sub.sort_values("first_day", ascending=False)
        titles = recent.groupby("company_id")["title"].apply(lambda s: list(dict.fromkeys(s))[:3]).to_dict()
        places = sub.assign(_p=sub.apply(lambda r: r["state"] if isinstance(r["state"], str) else None, axis=1)).dropna(subset=["_p"]) \
            .groupby("company_id")["_p"].apply(lambda s: [f"{k} ({v})" for k, v in Counter(s).most_common(2)]).to_dict()
        weekly = self._weekly(sub, by="company_id")
        tech_counts: dict[str, dict[str, int]] = {}
        if technologies:
            for tech in technologies:
                has = sub[self._tags.loc[sub.index].str.contains("|" + tech.casefold() + "|", regex=False)]
                for cid, n in has.groupby("company_id").size().items():
                    tech_counts.setdefault(cid, {})[tech] = int(n)
        rows = []
        for cid, r in grouped.iterrows():
            sig = self.signals.loc[cid] if cid in self.signals.index else None
            rows.append({
                "company_id": cid, "name": sig["company_name"] if sig is not None and isinstance(sig["company_name"], str) else None,
                "domain": sig["domain"] if sig is not None and isinstance(sig["domain"], str) else None,
                "new": int(r["new"]), "active": int(r["active"]), "listings": int(r["total"]),
                "pattern": sig["pattern"] if sig is not None else None,
                "surge_z": round(float(sig["surge_z"]), 1) if sig is not None and pd.notna(sig["surge_z"]) else None,
                "company_new_30": int(sig["new_30"]) if sig is not None else None,
                "example_titles": titles.get(cid, []), "places": places.get(cid, []), "technologies": tech_counts.get(cid, {}),
                "weekly": weekly.get(cid, [0] * WEEKS),
            })
        return rows

    # ---- tasks ------------------------------------------------------------------------------------------------------------------

    def _companies(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        notes: list[str] = []
        mask = self._mask(q, technologies, company_ids)
        subset = self.listings[mask]
        grouped = self._company_rows(subset, q)
        if q.first_time_hiring:
            family_rows = self.listings[self._mask(q, [], None, family_only=True) & ~self.listings["batch_start"]]
            earliest = family_rows.groupby("company_id")["first_day"].min()
            fresh = set(earliest[earliest > self.cutoff - q.window_days].index)
            grouped = grouped[grouped.index.isin(fresh)]
            notes.append("“First time” means the company's first opening in this role family that our data has seen; the data starts in spring 2026, so some of these may have hired before then.")
        if q.patterns:
            patterns = self.signals["pattern"].reindex(grouped.index)
            grouped = grouped[patterns.isin(q.patterns).to_numpy()]
        if q.sort == "surge":
            key = self.signals["surge_z"].reindex(grouped.index).fillna(-1e9)
            grouped = grouped.assign(_k=key.to_numpy()).sort_values(["_k", "new"], ascending=False).drop(columns="_k")
        elif q.sort == "active":
            grouped = grouped.sort_values(["active", "new"], ascending=False)
        else:
            grouped = grouped.sort_values(["new", "active"], ascending=False)
        total = int(len(grouped))
        top = grouped.head(q.limit)
        rows = self._decorate(top, subset, q, technologies)
        return {"rows": rows, "week_labels": self.week_labels(), "summary": {"companies": total, "new_openings": int(grouped["new"].sum()), "shown": len(rows)},
                "notes": notes}

    def _compare(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        mask = self._mask(q, technologies, company_ids)
        subset = self.listings[mask]
        grouped = pd.DataFrame({"company_id": subset["company_id"], "new": self.new_mask(q.window_days).loc[subset.index],
                                "active": self.active_mask.loc[subset.index], "total": 1}).groupby("company_id").sum()
        grouped = grouped.reindex(company_ids).fillna(0).astype(int)
        rows = self._decorate(grouped, subset, q, technologies)
        for row in rows:
            mine = subset[subset["company_id"] == row["company_id"]]
            exploded = mine["tags"].dropna().str.split("|").explode()
            exploded = exploded[~exploded.str.casefold().isin(hs._NOT_TECH)]
            row["top_technologies"] = [{"name": k, "listings": int(v)} for k, v in exploded.value_counts().head(8).items()]
            row["families"] = {FAMILY_LABEL.get(k, k): int(v) for k, v in mine[self.active_mask.loc[mine.index]]["family"].value_counts().items() if k in FAMILY_LABEL}
        return {"rows": rows, "week_labels": self.week_labels(), "summary": {"companies": len(rows)}, "notes": []}

    def _technologies(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        subset = self.listings[self._mask(q, technologies, company_ids)]
        exploded = subset.assign(_t=subset["tags"].fillna("").str.split("|")).explode("_t")
        exploded = exploded[(exploded["_t"] != "") & ~exploded["_t"].str.casefold().isin(hs._NOT_TECH)]
        window = self.cutoff - q.window_days
        recent = (exploded["first_day"] > window) & (exploded["first_day"] <= self.cutoff) & ~exploded["batch_start"]
        before = (exploded["first_day"] > window - q.window_days) & (exploded["first_day"] <= window) & ~exploded["batch_start"]
        table = pd.DataFrame({"tag": exploded["_t"], "company_id": exploded["company_id"], "recent": recent, "before": before}).groupby("tag") \
            .agg(listings=("company_id", "size"), companies=("company_id", "nunique"), new=("recent", "sum"), earlier=("before", "sum"))
        table = table.sort_values(["new", "listings"] if q.sort != "active" else ["listings", "new"], ascending=False)
        if technologies:  # when a technology was named, do not list it in its own results
            table = table[~table.index.str.casefold().isin([t.casefold() for t in technologies])]
        rows = [{"technology": tag, "listings": int(r.listings), "companies": int(r.companies), "new": int(r.new), "earlier": int(r.earlier)}
                for tag, r in table.head(q.limit).iterrows()]
        return {"rows": rows, "summary": {"listings": int(len(subset)), "companies": int(subset["company_id"].nunique())},
                "notes": ["Technologies are the tags on each listing; “new” counts listings first seen in the window, “earlier” the same-length window before it."]}

    def _listings(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        subset = self.listings[self._mask(q, technologies, company_ids) & self.new_mask(q.window_days)]
        top = subset.sort_values("first_day", ascending=False).head(q.limit)
        rows = [{"job_id": r.job_id, "title": r.title, "company_id": r.company_id, "company": r.company_name, "place": self._place_text(r),
                 "first_seen": day_to_date(r.first_day).isoformat(), "pay": self._pay_text(r),
                 "level": SENIORITY_LABEL.get(r.seniority, None) if isinstance(r.seniority, str) else None,
                 "technologies": [t for t in (r.tags.split("|") if isinstance(r.tags, str) else []) if t.casefold() not in hs._NOT_TECH][:5]}
                for r in top.itertuples()]
        return {"rows": rows, "summary": {"openings": int(len(subset)), "shown": len(rows)}, "notes": []}

    def _trend(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        subset = self.listings[self._mask(q, technologies, company_ids)]
        weekly = self._weekly(subset)
        last4, prev4 = sum(weekly[-4:]), sum(weekly[-8:-4])
        notes = ["Counts are openings first seen by the crawler each week (bulk-import days removed). A change can also come from the crawler covering more or fewer sites."]
        return {"rows": [{"week_ending": w, "new": n} for w, n in zip(self.week_labels(), weekly)],
                "summary": {"last_4_weeks": int(last4), "previous_4_weeks": int(prev4), "openings_in_range": int(sum(weekly))}, "notes": notes}

    def _pay(self, q: MarketQuery, technologies: list[str], company_ids: list[str] | None) -> dict[str, Any]:
        subset = self.listings[self._mask(q, technologies, company_ids)]
        stated = subset[subset["salary_low_usd"].notna() | subset["salary_high_usd"].notna()]
        yearly = stated[stated["salary_unit"] == "year"].copy()
        yearly["mid"] = yearly[["salary_low_usd", "salary_high_usd"]].mean(axis=1)
        notes = [f"Only listings that state a yearly pay are used ({len(yearly):,} of {len(subset):,} matching listings); {len(stated) - len(yearly):,} state pay per hour, month, week or day and are left out, not converted."]
        if len(yearly) < 5:
            return {"rows": [], "summary": {"matching_listings": int(len(subset)), "with_yearly_pay": int(len(yearly))},
                    "notes": notes + ["Too few listings state pay to give a range."]}
        pct = lambda p: float(np.percentile(yearly["mid"], p))  # noqa: E731
        by_level = []
        for level, group in yearly.groupby("seniority"):
            if len(group) >= 5 and level in SENIORITY_LABEL:
                by_level.append({"level": SENIORITY_LABEL[level], "listings": int(len(group)), "median": round(float(group["mid"].median())),
                                 "p25": round(float(np.percentile(group["mid"], 25))), "p75": round(float(np.percentile(group["mid"], 75)))})
        return {"rows": by_level, "summary": {"matching_listings": int(len(subset)), "with_yearly_pay": int(len(yearly)), "median": round(pct(50)),
                                              "p25": round(pct(25)), "p75": round(pct(75)), "low": round(float(yearly["mid"].min())), "high": round(float(yearly["mid"].max()))},
                "notes": notes + ["Each figure is the midpoint of a listing's stated range. These are the listings' own numbers, not an estimate for any one job."]}


def load_engine(directory: Path) -> MarketEngine | None:
    listings_file, signals_file = directory / "market_listings.csv.gz", directory / "company_signals.csv"
    if not listings_file.exists() or not signals_file.exists():
        return None
    listings = pd.read_csv(listings_file)
    signals = pd.read_csv(signals_file)
    summary = directory / "summary.json"
    snapshot = "2026-09-30"
    if summary.exists():
        import json

        snapshot = json.loads(summary.read_text(encoding="utf-8")).get("snapshot", snapshot)
    return MarketEngine(listings, signals, hs.day_number(snapshot))
