"""Compass analysis: what a set of picked jobs has in common, how a profile fits, and what else might fit.

Given 2-20 listings a person picked, their saved profile and the stored BGE embeddings, this builds:

* **themes**: groups of similar picks (at most 3), named from the listings' own occupation labels;
* **requirement groups**: requirement and duty sentences from the picks, clustered by meaning, with how many of
  the picks mention each group and the closest line of the person's own profile as evidence;
* **fit** for each pick: similarity of the profile to the listing, scaled against all listings;
* **suggestions**: listings the person did not pick that sit near the picks and the profile, diversified and
  filtered so they are not near-copies of a pick's title;
* **next steps**: the requirement groups that appear often and are least supported.

Everything is local (no outside AI call). Evidence is always a verbatim line from the profile or the listing, never
generated text. Scores are similarities, not probabilities of being hired, and "not mentioned" means the profile
does not say, which is unknown, not a verdict. Thresholds below are heuristics and are not validated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

import bge_text
import job_matching_engine as matching

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
MMR_LAMBDA = 0.7
REUSE_PENALTY = 0.04           # per earlier use of the same profile line when choosing explanations
NOVELTY_WEIGHT = 0.35          # how strongly a suggestion is pushed away from near-copies of the picks
MAX_PROFILE_UNITS = 90
MIN_PICKS = 2
MAX_PICKS = 20

_BOILERPLATE = re.compile(
    r"equal opportunity|\beeo\b|reasonable accommodation|background check|drug[- ]free|e-verify|salary range|base pay|"
    r"\bbenefits?\b|401\s?\(?k\)?|paid time off|\bpto\b|health insurance|visa sponsorship|click apply|apply now|"
    r"about us|our mission|we are an? |\bperks\b|compensation", re.IGNORECASE)
_VERIFY_YOURSELF = re.compile(
    r"\b\d+\+?\s*(?:years?|yrs)\b|\bdegree\b|\bbachelor|\bmaster|\bmba\b|\bcertifi|\bclearance\b|\blicen[sc]e",
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
) -> dict[str, Any]:
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
    pool_order = np.argsort(-blended_scores)[: SUGGESTION_POOL * 3]
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
    encoded = _unit_rows(np.asarray(backend.encode_documents(flat_texts), dtype=np.float32)) if flat_texts else np.zeros((0, 1), dtype=np.float32)
    unit_vectors = encoded[: len(units)]
    cursor = len(units)
    sentence_vectors: list[np.ndarray] = []
    for group in (*pick_sentences, *suggestion_sentences):
        sentence_vectors.append(encoded[cursor: cursor + len(group)])
        cursor += len(group)
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

    # 6. themes and angles
    labels = _choose_theme_labels(pick_vectors)
    k = len(set(labels))
    theme_members = {t: [i for i, l in enumerate(labels) if l == t] for t in sorted(set(labels))}
    mean_fit = {t: float(np.mean([pick_fits[i] for i in idx])) for t, idx in theme_members.items()}
    ordered_themes = sorted(theme_members, key=lambda t: (-mean_fit[t], t))
    base_angles = {1: [40], 2: [50, 230], 3: [40, 160, 280]}[k]
    half_width = {1: 70.0, 2: 82.0, 3: 52.0}[k]
    theme_info: dict[int, dict[str, Any]] = {}
    for rank, t in enumerate(ordered_themes):
        idx = theme_members[t]
        theme_info[t] = {
            "id": f"t{rank + 1}", "name": _theme_name([distinct[i].source for i in idx]), "count": len(idx),
            "angle": base_angles[rank], "half_width": half_width, "mean_fit": int(round(mean_fit[t])),
            "centroid": _normalize(pick_vectors[idx].mean(axis=0)), "mean_vec": pick_vectors[idx].mean(axis=0),
        }
    names_seen: dict[str, int] = {}
    for t in ordered_themes:  # keep names distinct
        name = theme_info[t]["name"]
        names_seen[name] = names_seen.get(name, 0) + 1
        if names_seen[name] > 1:
            theme_info[t]["name"] = f"{name} ({names_seen[name]})"
    suggestion_theme = [int(max(theme_info, key=lambda t: float(theme_info[t]["centroid"] @ v))) for v in suggestion_vectors]

    offsets: dict[str, float] = {}
    for t in ordered_themes:
        member_idx = theme_members[t]
        sug_idx = [i for i, st in enumerate(suggestion_theme) if st == t]
        mats = pick_vectors[member_idx]
        center = theme_info[t]["mean_vec"]
        if len(member_idx) >= 2:
            _, _, vt = np.linalg.svd(mats - center, full_matrices=False)
            axis = vt[0]
        else:
            axis = np.zeros(pick_vectors.shape[1], dtype=np.float32)
        coords = [float((pick_vectors[i] - center) @ axis) for i in member_idx] + [float((suggestion_vectors[i] - center) @ axis) for i in sug_idx]
        spread = _spread_offsets(coords, theme_info[t]["half_width"] - 14)
        keys = [("p", i) for i in member_idx] + [("s", i) for i in sug_idx]
        for (kind, i), off in zip(keys, spread):
            offsets[f"{kind}{i}"] = off

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
            "offset": round(offsets[f"p{j}"], 2),
            "why": why_for(pick_sentence_vectors[j], pick_sentences[j]),
            "stretch": [g["id"] for g in gaps[:3]],
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
            "offset": round(offsets[f"s{i}"], 2),
            "why": why_for(suggestion_sentence_vectors[i], suggestion_sentences[i]),
            "builds": builds,
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
        "picks_analyzed": total_picks,
        "duplicates_collapsed": duplicates_collapsed,
        "themes": themes_out,
        "strongest_theme_id": themes_out[0]["id"],
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

