"""Compass analysis: what a set of picked jobs has in common, how a profile fits, and what else might fit.

Given 2-20 listings a person picked, their saved profile and the stored BGE embeddings, this builds:

* **themes**: groups of similar picks (at most 3), named from the listings' own occupation labels;
* **requirement groups**: requirement and duty sentences from the picks, clustered by meaning, with how many of
  the picks mention each group and the closest line of the person's own profile as evidence;
* **fit** for each pick: similarity of the profile to the listing, scaled against all listings;
* **role axes**: up to 4 axes of O*NET role families, the least alike roles facing each other; each job points
  along its role's spoke, or between the most similar spokes when its role has none;
* **scope**: the direction with the most well-fitting jobs, and how wide that scope is at every heading;
* **suggestions**: listings the person did not pick that sit near the picks and the profile, diversified and
  filtered so they are not near-copies of a pick's title;
* **skill areas** for each job: its requirement lines sorted into a few broad areas, with how much the job asks for
  each area and how closely the profile's lines match it (the strengths-and-gaps radar);
* **next steps**: the requirement groups that appear often and are least supported.

Everything is local (no outside AI call). Evidence is always a verbatim line from the profile or the listing, never
generated text. Scores are similarities, not probabilities of being hired, and "not mentioned" means the profile
does not say, which is unknown, not a verdict. Thresholds below are heuristics and are not validated.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

import bge_text
import job_matching_engine as matching
from role_family_classifier import FAMILY_OF_OCCUPATION

# ---- heuristics (documented, tunable, unvalidated) ----
FIT_LOW_PERCENTILE = 5.0       # a listing among the least similar 5% to the profile scores 0
FIT_HIGH_PERCENTILE = 100.0    # the single most similar listing scores 100
COVERAGE_LOW = 0.50            # sentence-to-sentence cosine that counts as no evidence
COVERAGE_HIGH = 0.80           # cosine that counts as strong evidence
SUPPORTED_AT = 70              # coverage >= 70 -> supported
PARTIAL_AT = 40                # coverage >= 40 -> partial, else not mentioned
WHY_MIN_SIMILARITY = 0.66      # minimum cosine to show a profile line next to a requirement
CLUSTER_DISTANCE = 0.22        # cosine distance under which requirement sentences are merged
BUILDS_MIN_SIMILARITY = 0.74   # a suggested job "builds toward" a requirement group above this cosine
MAX_SENTENCES_PER_JOB = 18
MAX_REQUIREMENTS = 14
MIN_SENTENCE_WORDS = 5         # shorter lines are usually titles or headings, not requirements
MAX_THEMES = 3
SUGGESTION_COUNT = 5
SUGGESTION_POOL = 150
STRETCH_MAX_SKIP = 14000       # at stretch 100, suggestions come from ranks this far below the closest matches (capped to a quarter of the listings)
MMR_LAMBDA = 0.7
REUSE_PENALTY = 0.04           # per earlier use of the same profile line when choosing explanations
NOVELTY_WEIGHT = 0.35          # how strongly a suggestion is pushed away from near-copies of the picks
MAX_PROFILE_UNITS = 90
MIN_PICKS = 2
MAX_PICKS = 20
MAX_ROLE_POLES = 8             # 4 axes; each axis pairs two roles that sit opposite each other on the map
ROLE_CENTROID_SAMPLE = 400     # listings per role used to place that role
OFF_AXIS_SHARPNESS = 20.0      # how strongly a job whose role has no spoke leans towards the most similar spoke
OTHER_ROLE = "Other roles"
# Scope: the slice of the map the person faces. Heuristics, not validated.
SCOPE_KAPPA = 4.0              # how sharply jobs near a heading count (von Mises kernel)
SCOPE_WINDOW = 90.0            # jobs further than this from the heading are ignored
SCOPE_MIN_HALF = 8.0           # narrowest scope: a tight cluster of good fits at your strongest direction
SCOPE_MAX_HALF = 60.0          # widest scope: scattered, weak or no jobs
SCOPE_STEP = 5                 # degrees between precomputed scope widths
SCOPE_SUPPORT_HALF = 0.4       # fit-weighted job mass that takes the evidence term halfway to full
SUGGESTION_SCOPE_WEIGHT = 0.6  # a suggestion counts less than a job the person picked

# Skill areas for the strengths radar, in the order they go round it. A requirement line joins the area whose
# description it reads most like; years, degrees, certifications and eligibility (citizenship, location, travel)
# always go to Credentials & eligibility, which is never scored.
SKILL_AREAS = [
    ("tools", "Technical tools", "Hands-on experience with programming languages, software, platforms, databases and "
     "tools such as Python, SQL, Excel, cloud services and APIs."),
    ("analysis", "Analysis & data", "Analyze data, statistics, models, metrics, research and reporting to find insights "
     "and evaluate results."),
    ("delivery", "Building & delivery", "Design, build, test, deploy and maintain products, systems, pipelines and "
     "processes; troubleshoot and improve them."),
    ("domain", "Domain knowledge", "Knowledge of the industry, business, products, customers, markets and regulations "
     "in this field."),
    ("communication", "Communication", "Communicate clearly, write documentation, present to stakeholders, partner with "
     "customers and collaborate across teams."),
    ("leadership", "Leadership & ownership", "Lead projects, manage and mentor people, own outcomes, set priorities and "
     "plan work independently."),
    ("credentials", "Credentials & eligibility", None),
]
CREDENTIALS_AREA = "credentials"

_BOILERPLATE = re.compile(
    r"equal opportunity|\beeo\b|reasonable accommodation|background check|drug[- ]free|e-verify|salary range|base pay|"
    r"\bbenefits?\b|401\s?\(?k\)?|paid time off|\bpto\b|health insurance|visa sponsorship|click apply|apply now|"
    r"about us|our mission|we are an? |\bperks\b|compensation", re.IGNORECASE)
_VERIFY_YOURSELF = re.compile(
    r"\b\d+\+?\s*(?:years?|yrs)\b|\bdegree\b|\bbachelor|\bmaster|\bmba\b|\bcertifi|\bclearance\b|\blicen[sc]e",
    re.IGNORECASE)
_ELIGIBILITY = re.compile(
    r"\bcitizen|permanent resident|work authori[sz]|authori[sz]ed to work|sponsorship|\brelocat|\bbased in\b|"
    r"\btravel\b|\bon-?site\b|\bin-office\b|\bcommut|\bshifts?\b|\bweekends?\b|\bdriver'?s licen",
    re.IGNORECASE)
_TITLE_TOKEN = re.compile(r"[a-z0-9+#.]+")
_STOP_TITLE = {"the", "and", "of", "for", "to", "in", "a", "an", "at", "with", "senior", "junior", "sr", "jr", "i", "ii", "iii", "lead", "associate"}
_KIND_LABEL = {"job": "Experience", "project": "Project", "education": "Education"}


class CompassBackend(Protocol):
    """What the analysis needs from the ranker (implemented by RrfRankingBackend; faked in tests)."""

    def job_vectors(self, job_ids: Sequence[str]) -> np.ndarray: ...
    def query_vector(self, profile: matching.CandidateProfile) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def score_all(self, vector: np.ndarray) -> np.ndarray: ...


class CompassError(ValueError):
    """The request cannot be analysed; `code` is an API error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Unit:
    """One line of the person's own profile that can serve as evidence."""
    text: str
    source: str


# ---------------------------------------------------------------- text extraction

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t-*•|:#")


def requirement_sentences(description: str | None) -> list[str]:
    """Requirement and duty sentences of a listing (benefits, EEO and company-pitch sections are skipped)."""
    if not description:
        return []
    buckets = bge_text._job_sections(description)
    requirement, duty, other = buckets["requirement"], buckets["duty"], buckets["other"]
    ordered: list[str] = []
    for index in range(max(len(requirement), len(duty))):
        if index < len(requirement):
            ordered.append(requirement[index])
        if index < len(duty):
            ordered.append(duty[index])
    if len(ordered) < 3:  # unstructured description: fall back to its first plain sentences
        ordered.extend(other[: 6 - len(ordered)])
    result: list[str] = []
    seen: set[str] = set()
    for raw in ordered:
        sentence = _clean(raw)
        words = len(sentence.split())
        if not (MIN_SENTENCE_WORDS <= words <= 45) or not (25 <= len(sentence) <= 280) or _BOILERPLATE.search(sentence):
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
        if len(result) >= MAX_SENTENCES_PER_JOB:
            break
    return result


def profile_units(profile_data: Mapping[str, Any]) -> list[Unit]:
    """Split a stored profile into citeable lines: experience, projects, education and skills."""
    units: list[Unit] = []
    for entry in profile_data.get("experience", []) or []:
        title = str(entry.get("title") or "").strip()
        source = " · ".join(part for part in (_KIND_LABEL.get(str(entry.get("kind") or "job"), "Experience"), title) if part)
        head = " at ".join(part for part in (title, str(entry.get("organization") or "").strip()) if part)
        if head:
            units.append(Unit(head[:300], source))
        for sentence in bge_text._sentences(str(entry.get("description") or "")):
            sentence = sentence.strip()
            if len(sentence) >= 15:
                units.append(Unit(sentence[:300], source))
    for skill in profile_data.get("skills", []) or []:
        name = str(skill.get("name") or "").strip()
        evidence = str(skill.get("evidence") or "").strip()
        if name:
            units.append(Unit((f"{name}: {evidence}" if evidence else name)[:300], f"Skills · {name}"))
    # the headline is deliberately not an evidence line: it is a summary, and a generic one matches almost anything
    seen: set[str] = set()
    unique: list[Unit] = []
    for unit in units:
        key = unit.text.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(unit)
    return unique[:MAX_PROFILE_UNITS]


def _job_title(source: Mapping[str, Any]) -> str:
    return str(source.get("title_searchable") or source.get("title_normalized") or source.get("title_raw") or "Unknown title")


def _occupation(source: Mapping[str, Any]) -> str | None:
    onet = source.get("onet_data")
    name = onet.get("occupation_name") if isinstance(onet, Mapping) else None
    return str(name) if name else None


def _role_family(source: Mapping[str, Any]) -> str:
    occupation = _occupation(source)
    return FAMILY_OF_OCCUPATION.get(occupation, OTHER_ROLE) if occupation else OTHER_ROLE


_role_centroid_cache: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}


def _role_centroids(backend: CompassBackend, all_jobs: Sequence[matching.JobRecord]) -> dict[str, np.ndarray]:
    """Mean embedding of each role family across the corpus (a fixed sample per role), cached per corpus."""
    key = (id(backend), id(all_jobs), len(all_jobs))
    if key not in _role_centroid_cache:
        ids_by_role: dict[str, list[str]] = {}
        for job in all_jobs:
            role = _role_family(job.source)
            ids = ids_by_role.setdefault(role, [])
            if role != OTHER_ROLE and len(ids) < ROLE_CENTROID_SAMPLE:
                ids.append(job.job_id)
        _role_centroid_cache[key] = {
            role: _normalize(np.asarray(backend.job_vectors(ids), dtype=np.float32).mean(axis=0))
            for role, ids in ids_by_role.items() if ids
        }
    return _role_centroid_cache[key]


def _pairings(items: list[str]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for k, other in enumerate(rest):
        for tail in _pairings(rest[:k] + rest[k + 1:]):
            yield [(first, other), *tail]


def _pole_layout(roles: Sequence[str], centroids: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Pair roles into axes so each role faces the one least like it, then order the axes so neighbouring spokes
    are alike. roles[0] points north. An even number of roles gives len(roles) / 2 axes."""
    if len(roles) == 1:
        return {roles[0]: 0.0}

    def sim(a: str, b: str) -> float:
        return float(centroids[a] @ centroids[b])

    pairs = min(_pairings(list(roles)), key=lambda m: (sum(sim(a, b) for a, b in m), m))
    best: list[str] = []
    best_score = -math.inf
    for rest in itertools.permutations(pairs[1:]):
        for flips in itertools.product((False, True), repeat=len(rest)):
            axes = [pairs[0], *[(b, a) if flip else (a, b) for (a, b), flip in zip(rest, flips)]]
            ring = [a for a, _ in axes] + [b for _, b in axes]
            score = sum(sim(ring[n], ring[(n + 1) % len(ring)]) for n in range(len(ring)))
            if score > best_score:
                best, best_score = ring, score
    step = 360.0 / len(best)
    north = best.index(roles[0])
    return {role: round(((n - north) * step) % 360.0, 2) for n, role in enumerate(best)}


def _signed_turn(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _scope(points: Sequence[tuple[float, int, bool]]) -> dict[str, Any]:
    """Where the person's scope points and how wide it is when each map heading faces north.

    points are (angle, fit, picked). Well-fitting picks count most. The heading is the direction with the most
    fit-weighted jobs around it, so it can fall between two role spokes. The width at a heading starts at its widest
    and narrows toward the weighted angular spread of the jobs near it as support builds up. Support is the product of
    how much fit-weighted evidence is there, how many jobs share it (one centred job, however good, counts half), and
    how strong this direction is against the strongest one, so the scope is tightest on the best-fitting cluster.
    """
    weights = [(fit / 100.0) ** 2 * (1.0 if pick else SUGGESTION_SCOPE_WEIGHT) for _, fit, pick in points]

    def kernel(offset: float) -> float:
        return math.exp(SCOPE_KAPPA * (math.cos(math.radians(offset)) - 1.0))

    def density(h: int) -> float:
        return sum(w * kernel(a - h) for (a, _, _), w in zip(points, weights))

    heading = max(range(360), key=lambda h: (density(h), -h))
    stats: list[tuple[float, float, float, float]] = []  # (mass, spread, fit, sum of squared weights) per step
    for h in range(0, 360, SCOPE_STEP):
        mass = spread = fit = squares = 0.0
        for (a, f, _), w in zip(points, weights):
            offset = _signed_turn(a - h)
            if abs(offset) > SCOPE_WINDOW:
                continue
            k = w * kernel(offset)
            mass += k
            spread += k * offset * offset
            fit += k * f / 100.0
            squares += k * k
        stats.append((mass, spread, fit, squares))
    peak = max((m for m, _, _, _ in stats), default=0.0)
    half_widths: list[float] = []
    for mass, spread, fit, squares in stats:
        if mass < 1e-3:
            half_widths.append(SCOPE_MAX_HALF)
            continue
        cluster = min(SCOPE_MAX_HALF, max(SCOPE_MIN_HALF, math.sqrt(spread / mass) * (1.4 - fit / mass)))  # better fit narrows
        evidence = mass / (mass + SCOPE_SUPPORT_HALF)
        sharing = 1.0 - 0.5 ** (mass * mass / squares)  # effective job count: 1 job -> 0.5, 3 -> 0.875, 6 -> 0.98
        strength = math.sqrt(mass / peak)                # 1 at your strongest direction
        support = evidence * sharing * strength
        half_widths.append(round(SCOPE_MAX_HALF - (SCOPE_MAX_HALF - cluster) * support, 1))
    return {"heading": heading, "step": SCOPE_STEP, "half_widths": half_widths}


def _circular_mean(degrees: Sequence[float], weights: Sequence[float] | None = None) -> float:
    w = weights if weights is not None else [1.0] * len(degrees)
    x = sum(wi * math.sin(math.radians(d)) for d, wi in zip(degrees, w))
    y = sum(wi * math.cos(math.radians(d)) for d, wi in zip(degrees, w))
    return round(math.degrees(math.atan2(x, y)) % 360.0, 2)


def _skill_areas(sentences: Sequence[str], vectors: np.ndarray, unit_vectors: np.ndarray,
                 area_vectors: np.ndarray) -> list[dict[str, Any]]:
    """Sort one job's requirement lines into skill areas and score each area, in SKILL_AREAS order.

    `asked` is how much of the job's requirement text falls in the area, scaled so the most-asked scored area is 100;
    `shown` is the mean coverage of the area's lines by the closest profile line, or None when the area has no lines
    or is credentials, which similarity cannot judge.
    """
    scored = [area[0] for area in SKILL_AREAS if area[2] is not None]
    lines: dict[str, list[np.ndarray]] = {area[0]: [] for area in SKILL_AREAS}
    for text, vector in zip(sentences, vectors):
        if _VERIFY_YOURSELF.search(text) or _ELIGIBILITY.search(text):
            lines[CREDENTIALS_AREA].append(vector)
        else:
            lines[scored[int(np.argmax(area_vectors @ vector))]].append(vector)
    most = max([len(lines[a]) for a in scored] + [1])
    out = []
    for area_id, name, _ in SKILL_AREAS:
        own = lines[area_id]
        shown = None
        if own and area_id != CREDENTIALS_AREA and len(unit_vectors):
            best = (np.vstack(own) @ unit_vectors.T).max(axis=1)
            shown = int(round(float(np.mean([_coverage(float(c)) for c in best]))))
        out.append({"id": area_id, "name": name, "asked": int(round(100 * len(own) / most)), "lines": len(own), "shown": shown})
    return out


def _title_tokens(title: str) -> set[str]:
    return {t for t in _TITLE_TOKEN.findall(title.casefold()) if t not in _STOP_TITLE}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _coverage(cosine: float) -> int:
    return int(round(float(np.clip((cosine - COVERAGE_LOW) / (COVERAGE_HIGH - COVERAGE_LOW), 0.0, 1.0)) * 100))


def _status(coverage: int) -> str:
    return "supported" if coverage >= SUPPORTED_AT else "partial" if coverage >= PARTIAL_AT else "unknown"


def _shorten(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------- themes

def _choose_theme_labels(vectors: np.ndarray) -> list[int]:
    """Cluster the picks into 1-3 themes; keep more than one only when the split is clearly better than chance."""
    n = len(vectors)
    if n < 4:
        return [0] * n
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    best_labels, best_score = [0] * n, 0.05  # a split must beat this silhouette to be used
    for k in range(2, min(MAX_THEMES, n // 2) + 1):
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(vectors)
        if min(np.bincount(labels)) < 2:
            continue
        score = float(silhouette_score(vectors, labels, metric="cosine"))
        if score > best_score:
            best_labels, best_score = [int(x) for x in labels], score
    return best_labels


def _short_occupation(name: str) -> str:
    """O*NET labels can be very long ("Sales Representatives of Services, Except Advertising, ..."); keep the first clause."""
    return name.split(",")[0].strip()


def _theme_name(sources: Sequence[Mapping[str, Any]]) -> str:
    occupations = [_short_occupation(o) for o in (_occupation(s) for s in sources) if o]
    if occupations:
        top, count = max(((o, occupations.count(o)) for o in set(occupations)), key=lambda p: (p[1], p[0]))
        if count / len(sources) >= 0.4 and len(top) <= 40:
            return top
    tokens: dict[str, int] = {}
    for source in sources:
        for token in _title_tokens(_job_title(source)):
            tokens[token] = tokens.get(token, 0) + 1
    common = [t for t, _ in sorted(tokens.items(), key=lambda p: (-p[1], p[0]))[:2]]
    return " / ".join(t.title() for t in common) if common else "Mixed roles"


def _spread_offsets(raw: list[float], half_width: float, min_gap: float = 8.0) -> list[float]:
    """Scale raw 1-D coordinates into +/- half_width degrees and keep neighbours at least min_gap apart."""
    if not raw:
        return []
    scale = max(abs(x) for x in raw) or 1.0
    offsets = [half_width * x / scale for x in raw]
    order = sorted(range(len(offsets)), key=lambda i: offsets[i])
    for _ in range(6):
        for a, b in zip(order, order[1:]):
            if offsets[b] - offsets[a] < min_gap:
                push = (min_gap - (offsets[b] - offsets[a])) / 2
                offsets[a] -= push
                offsets[b] += push
    return [float(np.clip(x, -half_width, half_width)) for x in offsets]


# ---------------------------------------------------------------- main analysis

def analyze(
    backend: CompassBackend,
    *,
    picks: Sequence[matching.JobRecord],
    all_jobs: Sequence[matching.JobRecord],
    profile: matching.CandidateProfile,
    profile_data: Mapping[str, Any],
    excluded_job_ids: Sequence[str] = (),
    stretch: int = 0,
) -> dict[str, Any]:
    """`stretch` (0-100) sets how far beyond the closest matches the suggested roles reach; 0 is the original behaviour."""
    stretch = max(0, min(100, int(stretch)))
    # 1. collapse duplicate listings among the picks
    distinct: list[matching.JobRecord] = []
    seen_keys: set[str] = set()
    seen_company_titles: set[tuple[str, str]] = set()
    for job in picks:
        key = str(job.source.get("duplicate_key") or job.job_id)
        reference = job.source.get("company_reference")
        company = str(reference.get("id") or "") if isinstance(reference, Mapping) else ""
        company_title = (company, _job_title(job.source).casefold())
        if key in seen_keys or (company and company_title in seen_company_titles):
            continue  # a repost of the same posting, or the same company's same-titled opening
        seen_keys.add(key)
        seen_company_titles.add(company_title)
        distinct.append(job)
    duplicates_collapsed = len(picks) - len(distinct)
    if len(distinct) < MIN_PICKS:
        raise CompassError("too_few_distinct_jobs", "Pick at least two jobs that are not copies of each other.")
    units = profile_units(profile_data)
    if not units:
        raise CompassError("profile_too_thin", "Add some experience or skills to your profile first.")

    pick_ids = [job.job_id for job in distinct]
    pick_vectors = np.asarray(backend.job_vectors(pick_ids), dtype=np.float32)

    # 2. fit scale: where the profile sits among all listings
    query = _normalize(np.asarray(backend.query_vector(profile), dtype=np.float32))
    cosine_all = np.asarray(backend.score_all(query), dtype=np.float32)
    low = float(np.percentile(cosine_all, FIT_LOW_PERCENTILE))
    high = float(np.percentile(cosine_all, FIT_HIGH_PERCENTILE))
    if high - low < 1e-6:
        high = low + 1e-6

    def fit_of(cosine: float) -> int:
        return int(round(float(np.clip((cosine - low) / (high - low), 0.0, 1.0)) * 100))

    pick_fits = [fit_of(float(c)) for c in (pick_vectors @ query)]

    # 3. suggestions: near the picks and the profile, diversified, not a copy of a pick
    index_of = {job.job_id: i for i, job in enumerate(all_jobs)}
    blended = _normalize(0.5 * query + 0.5 * _normalize(pick_vectors.mean(axis=0)))
    blended_scores = np.asarray(backend.score_all(blended), dtype=np.float32)
    blocked = set(pick_ids) | set(excluded_job_ids)
    pick_titles = [_title_tokens(_job_title(job.source)) for job in distinct]
    pick_dup_keys = set(seen_keys)
    skip = int(round(min(STRETCH_MAX_SKIP, 0.25 * len(all_jobs)) * stretch / 100))
    pool_order = np.argsort(-blended_scores)[skip: skip + SUGGESTION_POOL * 3]
    pool: list[int] = []
    pool_relaxed: list[int] = []
    seen_title_company: set[tuple[str, str]] = set()
    for idx in pool_order:
        job = all_jobs[int(idx)]
        if job.job_id in blocked or str(job.source.get("duplicate_key") or "") in pick_dup_keys:
            continue
        title = _job_title(job.source)
        company = str((job.source.get("company_reference") or {}).get("id") or "") if isinstance(job.source.get("company_reference"), Mapping) else ""
        key = (title.casefold(), company)
        if key in seen_title_company:
            continue
        seen_title_company.add(key)
        tokens = _title_tokens(title)
        pool_relaxed.append(int(idx))
        if all(_jaccard(tokens, p) < 0.75 for p in pick_titles):
            pool.append(int(idx))
        if len(pool) >= SUGGESTION_POOL:
            break
    candidates = pool if len(pool) >= SUGGESTION_COUNT else pool_relaxed[:SUGGESTION_POOL]
    suggestions: list[matching.JobRecord] = []
    if candidates:
        cand_vectors = np.asarray(backend.job_vectors([all_jobs[i].job_id for i in candidates]), dtype=np.float32)
        relevance = np.clip((blended_scores[candidates] - low) / (high - low), 0.0, 1.0)
        nearest_pick = (cand_vectors @ pick_vectors.T).max(axis=1)  # how close each candidate is to a job already picked
        chosen: list[int] = []
        while len(chosen) < min(SUGGESTION_COUNT, len(candidates)):
            best, best_value = -1, -math.inf
            for pos in range(len(candidates)):
                if pos in chosen:
                    continue
                redundancy = max((float(cand_vectors[pos] @ cand_vectors[c]) for c in chosen), default=0.0)
                value = (MMR_LAMBDA * float(relevance[pos]) - (1 - MMR_LAMBDA) * redundancy
                         - NOVELTY_WEIGHT * max(0.0, float(nearest_pick[pos]) - 0.8))
                if value > best_value:
                    best, best_value = pos, value
            chosen.append(best)
        suggestions = [all_jobs[candidates[pos]] for pos in chosen]
        suggestion_vectors = cand_vectors[chosen]
    else:
        suggestion_vectors = np.zeros((0, pick_vectors.shape[1]), dtype=np.float32)
    suggestion_fits = [fit_of(float(c)) for c in (suggestion_vectors @ query)] if len(suggestions) else []

    # 4. one batch encode of every sentence we need
    pick_sentences = [requirement_sentences(job.source.get("description_text")) for job in distinct]
    suggestion_sentences = [requirement_sentences(job.source.get("description_text")) for job in suggestions]
    flat_texts: list[str] = [u.text for u in units]
    for group in (*pick_sentences, *suggestion_sentences):
        flat_texts.extend(group)
    area_texts = [text for _, _, text in SKILL_AREAS if text is not None]
    flat_texts.extend(area_texts)
    encoded = _unit_rows(np.asarray(backend.encode_documents(flat_texts), dtype=np.float32)) if flat_texts else np.zeros((0, 1), dtype=np.float32)
    unit_vectors = encoded[: len(units)]
    cursor = len(units)
    sentence_vectors: list[np.ndarray] = []
    for group in (*pick_sentences, *suggestion_sentences):
        sentence_vectors.append(encoded[cursor: cursor + len(group)])
        cursor += len(group)
    area_vectors = encoded[cursor: cursor + len(area_texts)]
    pick_sentence_vectors = sentence_vectors[: len(distinct)]
    suggestion_sentence_vectors = sentence_vectors[len(distinct):]

    # 5. requirement groups across the picks
    members: list[tuple[int, str, np.ndarray]] = []  # (pick index, sentence, vector)
    for j, (sentences, vectors) in enumerate(zip(pick_sentences, pick_sentence_vectors)):
        for sentence, vector in zip(sentences, vectors):
            members.append((j, sentence, vector))
    requirements: list[dict[str, Any]] = []
    group_of_member: dict[int, int] = {}
    group_centroids: list[np.ndarray] = []
    if len(members) >= 2:
        from sklearn.cluster import AgglomerativeClustering

        matrix = np.vstack([m[2] for m in members])
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=CLUSTER_DISTANCE).fit_predict(matrix)
        min_support = 2  # a group must appear in at least two of the picks
        groups: list[dict[str, Any]] = []
        for label in sorted(set(int(x) for x in labels)):
            idx = [i for i, l in enumerate(labels) if int(l) == label]
            job_set = sorted({members[i][0] for i in idx})
            if len(job_set) < min_support:
                continue
            centroid = _normalize(matrix[idx].mean(axis=0))
            order = sorted(idx, key=lambda i: (-(float(matrix[i] @ centroid)) if len(members[i][1]) <= 200 else 1.0))
            groups.append({"idx": idx, "jobs": job_set, "centroid": centroid, "rep": members[order[0]][1]})
        groups.sort(key=lambda g: (-len(g["jobs"]), -len(g["idx"]), g["rep"]))
        for gid, group in enumerate(groups[:MAX_REQUIREMENTS], start=1):
            sims = matrix[group["idx"]] @ unit_vectors.T
            m_best, u_best = np.unravel_index(int(np.argmax(sims)), sims.shape)
            cosine = float(sims[m_best, u_best])
            coverage = _coverage(cosine)
            status = _status(coverage)
            # years of experience, degrees and certifications cannot be confirmed by similarity: never "supported" automatically
            verify = bool(_VERIFY_YOURSELF.search(group["rep"]))
            if verify and status == "supported":
                status = "partial"
            evidence = None
            if cosine >= WHY_MIN_SIMILARITY:
                evidence = {"text": units[int(u_best)].text, "source": units[int(u_best)].source,
                            "matched_requirement": members[group["idx"][int(m_best)]][1]}
            requirements.append({
                "id": f"r{gid}", "text": _shorten(group["rep"], 160), "n": len(group["jobs"]),
                "coverage": coverage, "status": status, "evidence": evidence, "verify": verify,
                "job_ids": [pick_ids[j] for j in group["jobs"]],
            })
            group_centroids.append(group["centroid"])
            for i in group["idx"]:
                group_of_member[i] = gid - 1
    total_picks = len(distinct)

    def importance(req: Mapping[str, Any]) -> float:
        return req["n"] * (1 - req["coverage"] / 100)

    # 6. themes and map directions
    labels = _choose_theme_labels(pick_vectors)
    theme_members = {t: [i for i, l in enumerate(labels) if l == t] for t in sorted(set(labels))}
    mean_fit = {t: float(np.mean([pick_fits[i] for i in idx])) for t, idx in theme_members.items()}
    ordered_themes = sorted(theme_members, key=lambda t: (-mean_fit[t], t))
    theme_info: dict[int, dict[str, Any]] = {}
    for rank, t in enumerate(ordered_themes):
        idx = theme_members[t]
        theme_info[t] = {
            "id": f"t{rank + 1}", "name": _theme_name([distinct[i].source for i in idx]), "count": len(idx),
            "mean_fit": int(round(mean_fit[t])),
            "centroid": _normalize(pick_vectors[idx].mean(axis=0)), "mean_vec": pick_vectors[idx].mean(axis=0),
        }
    names_seen: dict[str, int] = {}
    for t in ordered_themes:  # keep names distinct
        name = theme_info[t]["name"]
        names_seen[name] = names_seen.get(name, 0) + 1
        if names_seen[name] > 1:
            theme_info[t]["name"] = f"{name} ({names_seen[name]})"
    suggestion_theme = [int(max(theme_info, key=lambda t: float(theme_info[t]["centroid"] @ v))) for v in suggestion_vectors]

    # direction is role: up to 4 axes of opposite roles; a job sits on its role's spoke, or leans towards the most
    # similar spokes when its role has none
    map_keys = [("p", i) for i in range(len(distinct))] + [("s", i) for i in range(len(suggestions))]
    map_vectors = np.vstack([pick_vectors, suggestion_vectors]) if len(suggestions) else pick_vectors
    map_roles = [_role_family(job.source) for job in (*distinct, *suggestions)]
    role_of = dict(zip(map_keys, map_roles))
    centroids = _role_centroids(backend, all_jobs)
    role_counts: dict[str, int] = {}
    for role in map_roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    poles = sorted((r for r in role_counts if r in centroids), key=lambda r: (-role_counts[r], r))[:MAX_ROLE_POLES]
    if poles and len(poles) < MAX_ROLE_POLES:  # fill with the roles closest to everything on the map
        middle = _normalize(map_vectors.mean(axis=0))
        nearby = sorted((r for r in centroids if r not in poles), key=lambda r: (-float(centroids[r] @ middle), r))
        poles += nearby[: MAX_ROLE_POLES - len(poles)]
    if len(poles) > 1 and len(poles) % 2:
        poles = poles[:-1]
    spoke_angle = _pole_layout(poles, centroids) if poles else {OTHER_ROLE: 0.0}
    if not poles:
        role_of = {key: OTHER_ROLE for key in map_keys}
    role_axes = [{"name": r, "angle": a, "count": sum(1 for x in role_of.values() if x == r)}
                 for r, a in sorted(spoke_angle.items(), key=lambda p: p[1])]
    slice_half = min(60.0, 180.0 / len(spoke_angle) - 3.0)
    angles: dict[str, float] = {}
    for role, axis_angle in spoke_angle.items():
        on_spoke = [n for n, key in enumerate(map_keys) if role_of[key] == role]
        if not on_spoke:
            continue
        mats = map_vectors[on_spoke]
        center = mats.mean(axis=0)
        if len(on_spoke) >= 2:
            _, _, vt = np.linalg.svd(mats - center, full_matrices=False)
            coords = [float(v) for v in (mats - center) @ vt[0]]
        else:
            coords = [0.0]
        spread = _spread_offsets(coords, slice_half, min_gap=min(8.0, slice_half / 2))
        for n, off in zip(on_spoke, spread):
            kind, i = map_keys[n]
            angles[f"{kind}{i}"] = round((axis_angle + off) % 360.0, 2)
    pole_names = list(spoke_angle)
    for n, key in enumerate(map_keys):
        if role_of[key] in spoke_angle:
            continue
        sims = np.array([float(centroids[r] @ _normalize(map_vectors[n])) for r in pole_names])
        weights = np.exp(OFF_AXIS_SHARPNESS * (sims - sims.max()))
        angles[f"{key[0]}{key[1]}"] = _circular_mean([spoke_angle[r] for r in pole_names], list(weights))
    scope = _scope([(angles[f"{kind}{i}"], (pick_fits if kind == "p" else suggestion_fits)[i], kind == "p")
                    for kind, i in map_keys])

    # 7. per-job explanations from sentence/profile-line matches
    unit_use: dict[int, int] = {}  # how often each profile line has already been shown, so explanations vary

    def why_for(sent_vectors: np.ndarray, sentences: Sequence[str]) -> list[dict[str, str]]:
        if not len(sentences):
            return []
        sims = sent_vectors @ unit_vectors.T
        used_units: set[int] = set()
        pairs: list[dict[str, str]] = []
        flat = sorted(
            ((float(sims[s, u]) - REUSE_PENALTY * unit_use.get(u, 0), s, u) for s in range(sims.shape[0]) for u in range(sims.shape[1])),
            reverse=True,
        )
        used_sentences: set[int] = set()
        for value, s, u in flat:
            if float(sims[s, u]) < WHY_MIN_SIMILARITY or len(pairs) >= 2:
                if len(pairs) >= 2:
                    break
                continue
            if u in used_units or s in used_sentences:
                continue
            used_units.add(u)
            used_sentences.add(s)
            unit_use[u] = unit_use.get(u, 0) + 1
            pairs.append({"evidence": units[u].text, "source": units[u].source, "requirement": sentences[s]})
        return pairs

    jobs_out: list[dict[str, Any]] = []
    for j, job in enumerate(distinct):
        t = labels[j]
        job_groups = sorted({group_of_member[i] for i, m in enumerate(members) if m[0] == j and i in group_of_member})
        gaps = [requirements[g] for g in job_groups if requirements[g]["status"] != "supported"]
        gaps.sort(key=importance, reverse=True)
        jobs_out.append({
            "job_id": job.job_id, "pick": True, "theme_id": theme_info[t]["id"], "fit": pick_fits[j],
            "role": role_of[("p", j)], "angle": angles[f"p{j}"],
            "why": why_for(pick_sentence_vectors[j], pick_sentences[j]),
            "stretch": [g["id"] for g in gaps[:3]],
            "areas": _skill_areas(pick_sentences[j], pick_sentence_vectors[j], unit_vectors, area_vectors),
        })
    for i, job in enumerate(suggestions):
        t = suggestion_theme[i]
        builds = []
        if len(suggestion_sentences[i]) and group_centroids:
            sims = suggestion_sentence_vectors[i] @ np.vstack(group_centroids).T
            reached = [g for g in range(len(group_centroids)) if float(sims[:, g].max()) >= BUILDS_MIN_SIMILARITY and requirements[g]["status"] != "supported"]
            reached.sort(key=lambda g: importance(requirements[g]), reverse=True)
            builds = [requirements[g]["id"] for g in reached[:2]]
        jobs_out.append({
            "job_id": job.job_id, "pick": False, "theme_id": theme_info[t]["id"], "fit": suggestion_fits[i],
            "role": role_of[("s", i)], "angle": angles[f"s{i}"],
            "why": why_for(suggestion_sentence_vectors[i], suggestion_sentences[i]),
            "builds": builds,
            "areas": _skill_areas(suggestion_sentences[i], suggestion_sentence_vectors[i], unit_vectors, area_vectors),
        })

    # 8. next steps
    open_reqs = sorted((r for r in requirements if r["status"] != "supported"), key=importance, reverse=True)[:6]
    next_steps = [{"requirement_id": r["id"], "text": r["text"], "n": r["n"], "coverage": r["coverage"], "status": r["status"],
                   "verify": r["verify"],
                   "action": ("Check this one yourself: years, degrees and certifications are not judged automatically."
                              if r["verify"] else "Add or build one concrete example that shows this, then describe it in your profile.")}
                  for r in open_reqs]

    themes_out = [{k_: v for k_, v in info.items() if k_ not in {"centroid", "mean_vec"}} for info in (theme_info[t] for t in ordered_themes)]
    with_evidence = sum(1 for r in requirements if r["status"] != "unknown")
    return {
        "stretch": stretch,
        "picks_analyzed": total_picks,
        "duplicates_collapsed": duplicates_collapsed,
        "themes": themes_out,
        "strongest_theme_id": themes_out[0]["id"],
        "roles": role_axes,
        "scope": scope,
        "jobs": jobs_out,
        "requirements": requirements,
        "next_steps": next_steps,
        "stats": {
            "jobs": total_picks, "themes": len(themes_out), "requirement_groups": len(requirements),
            "groups_with_evidence": with_evidence, "suggestions": len(suggestions),
            "sentences_read": sum(len(s) for s in pick_sentences),
        },
        "fit_scale": {"low_cosine": round(low, 4), "high_cosine": round(high, 4),
                      "meaning": "0 is among the least similar 5% of listings to your profile; 100 is the most similar listing in the whole set."},
        "method": {
            "evidence": "Each piece of evidence is a verbatim line from your profile matched to a requirement line by embedding similarity.",
            "thresholds": {"supported_at": SUPPORTED_AT, "partial_at": PARTIAL_AT, "coverage_cosine_low": COVERAGE_LOW,
                           "coverage_cosine_high": COVERAGE_HIGH},
            "limits": ["Scores are similarities, not hiring probabilities.",
                       "Years of experience, degrees and certifications are never judged automatically; they are marked to check yourself.",
                       "Not mentioned means your profile does not say, which is unknown, not a verdict.",
                       "Thresholds are heuristics and have not been validated.",
                       "Listings are historical and not verified as open."],
        },
    }

