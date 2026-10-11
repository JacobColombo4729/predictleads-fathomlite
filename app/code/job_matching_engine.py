"""Local-only first milestone for PredictLeads job matching.

This module reads already-prepared JSONL records. It never accesses AWS or
creates relevance labels. Similarity scores are retrieval signals, not hiring
probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent  # data/, models/ and .env are shared with analysis/ at the top
DEFAULT_SAMPLE = REPO_ROOT / (
    "data/discovery/role_exploration/roles_corrected_20261006T000000Z/"
    "jobs_prepared_random_sample.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/matching/job_matching_baseline"
DEFAULT_PRIOR_BASELINE = REPO_ROOT / "data/matching/job_matching_engine_baseline/recommendations.csv"
TOP_K = 10
DESCRIPTION_EVIDENCE_LIMIT = 500
DESCRIPTION_EVIDENCE_RULE_VERSION = "substantive_description_v1"
DESCRIPTION_EVIDENCE_MIN_DUTY_WORDS = 3
DESCRIPTION_EVIDENCE_MIN_QUALIFICATION_WORDS = 3

_DESCRIPTION_METADATA_LINE = re.compile(
    r"(?i)^(?:[-*]\s*)?(?:home|fashion jobs|share|facebook|linkedin|twitter|apply|new|"
    r"published on|posted(?: on)?|reference|internal number|job requisition id|job id|"
    r"location(?: name)?|address|time type|job type|position type|job function|position|profile|"
    r"employment type|store|common location|pay type|pay rate|start rate|max rate|bonus eligible|"
    r"full time|part time|beginner|internship|regular|permanent job|"
    r"application window|primary location|posting date|salary starting at)\s*[:\-]?.*$"
)
_DESCRIPTION_BOILERPLATE = re.compile(
    r"(?i)(?:equal opportunity employer|reasonable accommodations?|protected (?:category|class)|"
    r"discrimination|eeo statement|benefits (?:package|include|offered)|"
    r"competitive compensation|health, dental and vision|401\s*\(?k\)?|"
    r"our mission is|our vision is|founded in \d{4}|about the company|"
    r"we are proud to be|we value a diverse|we believe in (?:our|the) (?:power|future)|"
    r"our culture (?:thrives|is)|our company (?:is|offers)|"
    r"we offer (?:a )?(?:competitive|comprehensive|wide variety of) (?:benefits|compensation)|"
    r"apply today|join our team today)"
)
_DESCRIPTION_QUALIFICATION_SIGNAL = re.compile(
    r"(?i)(?:\b\d+\s*(?:\+|\-\s*\d+)?\s*years?\b|\b(?:degree|diploma|ged|certification|certificate|license|"
    r"proficiency|proficient|familiarity|familiar with|knowledge of|ability to|must be|must have|"
    r"experience with|experience in|experience using|experience required|required experience|"
    r"experience preferred|qualifications? include|skills include|high school|bachelor(?:'s)?|"
    r"associate(?:'s)? degree|minimum qualifications|preferred qualifications)\b)"
)
_DESCRIPTION_ACTION_VERB = re.compile(
    r"(?i)\b(?:manage|perform|support|assist|coordinate|maintain|process|provide|develop|ensure|"
    r"operate|analy[sz]e|troubleshoot|lead|deliver|handle|monitor|prepare|review|create|conduct|"
    r"build|implement|execute|receive|stock|count|schedule|sell|resolve|communicate|track|"
    r"reconcile|inspect|train|supervise|load|unload|restock|ship|purchase|order|greet|record|"
    r"investigate|audit|replenish|organize|organise|assist|respond|install|test|design|write)\b"
)
_DESCRIPTION_SECTION_HEADING = re.compile(
    r"(?i)^(?:job description|job summary|position summary|about the role|what you(?:'|’)ll do|"
    r"what you(?:'|’)ll bring|essential (?:functions|duties)|key responsibilities|responsibilities|"
    r"duties|qualifications?|requirements?|required qualifications|preferred qualifications|"
    r"knowledge, skills(?: and abilities)?|skills and qualifications|education(?:/experience/skills/qualifications)?|"
    r"experience requirements|minimum qualifications|working conditions|benefits)\s*:?$"
)


@dataclass(frozen=True)
class DescriptionEvidenceAssessment:
    classification: str
    eligible: bool
    reason: str
    duty_evidence: tuple[str, ...] = ()
    qualification_evidence: tuple[str, ...] = ()


def assess_substantive_description(record: Mapping[str, Any]) -> DescriptionEvidenceAssessment:
    """Apply a transparent v1 heuristic; not a validated description-quality classifier.

    Eligibility requires at least one non-boilerplate line with a concrete duty
    action and object, or a non-heading line with an explicit qualification
    signal. Length and standard section headings are never eligibility rules.
    """
    description = _nonblank(record.get("description_text"))
    if not description:
        return DescriptionEvidenceAssessment(
            "missing_or_blank", False, "description_text_missing_or_blank"
        )

    source_title = {
        _nonblank(record.get(field)).casefold()
        for field in ("title_raw", "title_normalized", "title_translated", "title_searchable")
        if _nonblank(record.get(field))
    }
    informative_lines: list[str] = []
    metadata_count = 0
    boilerplate_count = 0
    for raw_line in description.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" \t-*•")
        if not line:
            continue
        if (line.casefold() in source_title or _DESCRIPTION_METADATA_LINE.match(line)
                or _DESCRIPTION_SECTION_HEADING.match(line)):
            metadata_count += 1
            continue
        if _DESCRIPTION_BOILERPLATE.search(line):
            boilerplate_count += 1
            continue
        informative_lines.append(line)

    duty_lines: list[str] = []
    qualification_lines: list[str] = []
    for line in informative_lines:
        words = re.findall(r"[A-Za-z0-9+#.]+", line)
        if (_DESCRIPTION_QUALIFICATION_SIGNAL.search(line)
                and len(words) >= DESCRIPTION_EVIDENCE_MIN_QUALIFICATION_WORDS):
            qualification_lines.append(line)
        if (_DESCRIPTION_ACTION_VERB.search(line)
                and len(words) >= DESCRIPTION_EVIDENCE_MIN_DUTY_WORDS):
            duty_lines.append(line)

    if duty_lines and qualification_lines:
        return DescriptionEvidenceAssessment(
            "substantive_duties_and_qualifications", True,
            "non_boilerplate_duty_and_qualification_evidence",
            tuple(duty_lines[:3]), tuple(qualification_lines[:3]),
        )
    if duty_lines:
        return DescriptionEvidenceAssessment(
            "substantive_duties", True, "non_boilerplate_duty_evidence",
            tuple(duty_lines[:3]), (),
        )
    if qualification_lines:
        return DescriptionEvidenceAssessment(
            "substantive_qualifications", True, "non_boilerplate_qualification_evidence",
            (), tuple(qualification_lines[:3]),
        )
    if not informative_lines and boilerplate_count and metadata_count:
        reason = "metadata_and_boilerplate_only"
    elif not informative_lines and metadata_count:
        reason = "metadata_only"
    elif not informative_lines and boilerplate_count:
        reason = "boilerplate_only"
    else:
        reason = "no_detected_duty_or_qualification_evidence"
    return DescriptionEvidenceAssessment("insufficient", False, reason)

# Match common technical forms before the general token expression. Lowercase
# normalization retains C++, C#, and .NET as distinct tokens.
_TOKEN_PATTERN = re.compile(r"(?i)\.net\b|c\+\+|c#|[a-z0-9]+(?:[.+#/-][a-z0-9+#/-]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize searchable text while retaining common technical names."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text or "")]


def _nonblank(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _occupation(record: Mapping[str, Any]) -> str:
    onet = record.get("onet_data")
    if isinstance(onet, Mapping):
        return _nonblank(onet.get("occupation_name"))
    return ""


def _location_text(record: Mapping[str, Any]) -> str:
    parts = [_nonblank(record.get("location_raw"))]
    structured = record.get("location_data_raw")
    if isinstance(structured, list):
        parts.extend(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in structured)
    elif isinstance(structured, Mapping):
        parts.append(json.dumps(structured, ensure_ascii=False, sort_keys=True))
    return " ".join(part for part in parts if part)


def _job_search_fields(record: Mapping[str, Any]) -> tuple[str, str, str]:
    title = (
        _nonblank(record.get("title_searchable"))
        or _nonblank(record.get("title_normalized"))
        or _nonblank(record.get("title_translated"))
        or _nonblank(record.get("title_raw"))
    )
    occupation = _occupation(record)
    description = _nonblank(record.get("description_text"))
    return title, occupation, description


def _weighted_job_text(record: Mapping[str, Any]) -> str:
    """Repeat fields to approximate explicit title/occupation/description weights."""
    title, occupation, description = _job_search_fields(record)
    return " ".join(part for part in (title, title, title, occupation, occupation, description) if part)


def _candidate_query_terms(profile: CandidateProfile) -> list[str]:
    sections = [*profile.experience, *profile.evidenced_skills, *profile.desired_roles]
    if profile.location:
        sections.append(profile.location)
    sections.extend(_preference_text(profile.preferences))
    return tokenize(" ".join(sections))


def _candidate_keyword_weights(profile: CandidateProfile) -> Counter[str]:
    """Give target roles more weight than evidence, and preferences less."""
    weights: Counter[str] = Counter()
    for role in profile.desired_roles:
        weights.update({term: 3 for term in tokenize(role)})
    for evidence in (*profile.experience, *profile.evidenced_skills):
        weights.update({term: 2 for term in tokenize(evidence)})
    preferences = ([profile.location] if profile.location else []) + _preference_text(profile.preferences)
    for preference in preferences:
        weights.update({term: 1 for term in tokenize(preference)})
    return weights


def _preference_text(preferences: Mapping[str, Any]) -> list[str]:
    text: list[str] = []
    for key in ("keywords", "work_arrangement", "industry_preferences"):
        value = preferences.get(key)
        if isinstance(value, str) and value.strip():
            text.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            text.extend(str(item).strip() for item in value if str(item).strip())
    return text


@dataclass(frozen=True)
class CandidateProfile:
    """Reusable candidate input; all profiles in this prototype are synthetic."""

    profile_id: str
    experience: tuple[str, ...] = ()
    evidenced_skills: tuple[str, ...] = ()
    desired_roles: tuple[str, ...] = ()
    location: str | None = None
    preferences: Mapping[str, Any] = field(default_factory=dict)

    def tfidf_text(self) -> str:
        """Candidate query with role targets weighted above evidence/preferences."""
        target_text = " ".join(self.desired_roles)
        evidence_text = " ".join((*self.experience, *self.evidenced_skills))
        preference_text = " ".join(
            part for part in (self.location or "", *_preference_text(self.preferences)) if part
        )
        return " ".join(part for part in (target_text, target_text, target_text,
                                           evidence_text, evidence_text, preference_text) if part)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobRecord:
    """A prepared job with its original prepared fields preserved verbatim."""

    job_id: str
    source: Mapping[str, Any]
    usable_description: bool

    @property
    def searchable_title(self) -> str:
        return _job_search_fields(self.source)[0]

    @property
    def occupation(self) -> str:
        return _occupation(self.source)


@dataclass
class LoadReport:
    path: str
    lines_seen: int = 0
    records_loaded: int = 0
    usable_descriptions: int = 0
    excluded_by_reason: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "lines_seen": self.lines_seen,
            "records_loaded": self.records_loaded,
            "usable_descriptions": self.usable_descriptions,
            "excluded_records": sum(self.excluded_by_reason.values()),
            "excluded_by_reason": dict(sorted(self.excluded_by_reason.items())),
        }


def load_prepared_jobs(path: Path) -> tuple[list[JobRecord], LoadReport]:
    """Read local JSONL, retaining source data and explaining excluded lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared sample not found: {path}")

    report = LoadReport(path=str(path))
    jobs: list[JobRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            report.lines_seen += 1
            if not line.strip():
                report.excluded_by_reason["blank_line"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                report.excluded_by_reason["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                report.excluded_by_reason["record_not_object"] += 1
                continue

            raw_id = record.get("job_id")
            job_id = str(raw_id).strip() if raw_id is not None else ""
            if not job_id:
                report.excluded_by_reason["missing_stable_job_id"] += 1
                continue
            if job_id in seen_ids:
                report.excluded_by_reason["duplicate_job_id"] += 1
                continue
            if not tokenize(" ".join(_job_search_fields(record))):
                report.excluded_by_reason["no_searchable_text"] += 1
                continue

            seen_ids.add(job_id)
            description_usable = bool(
                record.get("description_usable") and _nonblank(record.get("description_text"))
            )
            jobs.append(JobRecord(job_id=job_id, source=record, usable_description=description_usable))
            report.records_loaded += 1
            report.usable_descriptions += int(description_usable)

    return jobs, report


@dataclass(frozen=True)
class FilterDecision:
    included: bool
    reason: str | None = None


def filter_job(job: JobRecord, profile: CandidateProfile) -> FilterDecision:
    """Apply only explicit status/location constraints with defined unknown handling."""
    if profile.preferences.get("exclude_explicitly_closed") is True:
        status = _nonblank(job.source.get("status_derived")).casefold()
        if status == "closed":
            return FilterDecision(False, "explicitly_closed")
        # Unknown or absent status is retained and remains labelled unknown.

    if profile.preferences.get("require_location_match") is True:
        location_terms = tokenize(profile.location or "")
        if not location_terms:
            return FilterDecision(False, "location_filter_missing_candidate_location")
        location = _location_text(job.source).casefold()
        if not location:
            return FilterDecision(False, "unknown_location")
        if not all(term in tokenize(location) for term in location_terms):
            return FilterDecision(False, "location_no_match")

    return FilterDecision(True)


@dataclass
class MatchingCorpus:
    jobs: list[JobRecord]
    vectorizer: TfidfVectorizer
    job_matrix: Any
    description_assessments: list[DescriptionEvidenceAssessment] | None = None

    def assess_descriptions(self) -> list[DescriptionEvidenceAssessment]:
        if self.description_assessments is None:
            self.description_assessments = [
                assess_substantive_description(job.source) for job in self.jobs
            ]
        return self.description_assessments


def fit_tfidf(jobs: Sequence[JobRecord]) -> MatchingCorpus:
    """Fit TF-IDF once on the job corpus; candidate profiles are transformed later."""
    if not jobs:
        raise ValueError("Cannot fit TF-IDF: no usable job records were loaded.")
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        max_features=250_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    texts = [_weighted_job_text(job.source) for job in jobs]
    matrix = vectorizer.fit_transform(texts)
    return MatchingCorpus(list(jobs), vectorizer, matrix)


def load_candidate_profiles(path: Path) -> list[CandidateProfile]:
    """Load reusable profile definitions from a local JSON file."""
    with Path(path).open("r", encoding="utf-8") as source_file:
        raw_profiles = json.load(source_file)
    if not isinstance(raw_profiles, list):
        raise ValueError("Candidate profile file must contain a JSON array.")
    profiles: list[CandidateProfile] = []
    for index, raw in enumerate(raw_profiles, 1):
        if not isinstance(raw, dict) or not isinstance(raw.get("profile_id"), str):
            raise ValueError(f"Candidate profile {index} must be an object with a string profile_id.")
        for field_name in ("experience", "evidenced_skills", "desired_roles"):
            value = raw.get(field_name, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"Candidate profile {index} field {field_name} must be a string array.")
        preferences = raw.get("preferences", {})
        if not isinstance(preferences, dict):
            raise ValueError(f"Candidate profile {index} preferences must be an object.")
        location = raw.get("location")
        if location is not None and not isinstance(location, str):
            raise ValueError(f"Candidate profile {index} location must be a string or null.")
        profiles.append(CandidateProfile(
            profile_id=raw["profile_id"],
            experience=tuple(raw.get("experience", [])),
            evidenced_skills=tuple(raw.get("evidenced_skills", [])),
            desired_roles=tuple(raw.get("desired_roles", [])),
            location=location,
            preferences=preferences,
        ))
    ids = [profile.profile_id for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate profile IDs must be unique.")
    return profiles


def _job_keyword_parts(job: JobRecord) -> tuple[str, str, str]:
    return _job_search_fields(job.source)


def _keyword_score(job: JobRecord, query_weights: Mapping[str, int]) -> tuple[float, list[str]]:
    title, occupation, description = _job_keyword_parts(job)
    fields = (tokenize(title), tokenize(occupation), tokenize(description))
    weights = (3.0, 2.0, 1.0)
    denominator = sum(query_weights.values())
    matched: list[str] = []
    weighted_hits = 0.0
    for term, profile_weight in query_weights.items():
        source_weight = max(
            (weight for tokens, weight in zip(fields, weights) if term in tokens), default=0.0
        )
        if source_weight:
            matched.append(term)
            weighted_hits += profile_weight * source_weight / max(weights)
    score = weighted_hits / denominator if denominator else 0.0
    return score, matched


def condition_job_indices(corpus: MatchingCorpus, condition: str) -> list[int]:
    """Return eligible-corpus row indices; TF-IDF rows are never refitted here."""
    if condition == "all_eligible":
        return list(range(len(corpus.jobs)))
    if condition == "description_available":
        return [index for index, job in enumerate(corpus.jobs) if job.usable_description]
    if condition == "substantive_description":
        return [index for index, assessment in enumerate(corpus.assess_descriptions())
            if assessment.eligible]
    raise ValueError(f"Unknown retrieval condition: {condition}")


def rank_description_coverage_on_corpus(
    corpus: MatchingCorpus, profiles: Sequence[CandidateProfile], top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """Rank both eligibility conditions from one already-fitted corpus/matrix."""
    recommendations: list[dict[str, Any]] = []
    for condition in ("description_available", "substantive_description"):
        candidate_indices = condition_job_indices(corpus, condition)
        for profile in profiles:
            recommendations.extend(rank_keyword(
                corpus, profile, top_k, candidate_indices=candidate_indices, condition=condition
            ))
            recommendations.extend(rank_tfidf(
                corpus, profile, top_k, candidate_indices=candidate_indices, condition=condition
            ))
    return recommendations


def run_description_coverage_rankings(
    jobs: Sequence[JobRecord], profiles: Sequence[CandidateProfile], top_k: int = TOP_K
) -> tuple[MatchingCorpus, list[dict[str, Any]], dict[str, int]]:
    """Fit TF-IDF once on all eligible jobs, then compare two eligibility subsets."""
    corpus = fit_tfidf(jobs)
    condition_counts = {
        "description_available": sum(job.usable_description for job in corpus.jobs),
        "substantive_description": sum(
            assessment.eligible for assessment in corpus.assess_descriptions()
        ),
    }
    recommendations = rank_description_coverage_on_corpus(corpus, profiles, top_k)
    return corpus, recommendations, condition_counts


def _eligible_jobs(
    corpus: MatchingCorpus,
    profile: CandidateProfile,
    candidate_indices: Sequence[int] | None = None,
) -> tuple[list[int], Counter[str]]:
    included: list[int] = []
    excluded: Counter[str] = Counter()
    indices = range(len(corpus.jobs)) if candidate_indices is None else candidate_indices
    for index in indices:
        job = corpus.jobs[index]
        decision = filter_job(job, profile)
        if decision.included:
            included.append(index)
        else:
            excluded[decision.reason or "filtered"] += 1
    return included, excluded


def rank_keyword(
    corpus: MatchingCorpus,
    profile: CandidateProfile,
    top_k: int = TOP_K,
    *,
    candidate_indices: Sequence[int] | None = None,
    condition: str = "all_eligible",
) -> list[dict[str, Any]]:
    query_weights = _candidate_keyword_weights(profile)
    eligible, _ = _eligible_jobs(corpus, profile, candidate_indices)
    scored = []
    for index in eligible:
        score, matched = _keyword_score(corpus.jobs[index], query_weights)
        scored.append((score, corpus.jobs[index].job_id, index, matched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [_recommendation(corpus.jobs[index], profile, "keyword", rank, score, matched, condition)
            for rank, (score, _, index, matched) in enumerate(scored[:top_k], 1)]


def rank_tfidf(
    corpus: MatchingCorpus,
    profile: CandidateProfile,
    top_k: int = TOP_K,
    *,
    candidate_indices: Sequence[int] | None = None,
    condition: str = "all_eligible",
) -> list[dict[str, Any]]:
    eligible, _ = _eligible_jobs(corpus, profile, candidate_indices)
    if not eligible:
        return []
    candidate_vector = corpus.vectorizer.transform([profile.tfidf_text()])
    similarities = (corpus.job_matrix[eligible] @ candidate_vector.T).toarray().ravel()
    ranked = sorted(zip(similarities, eligible), key=lambda item: (-float(item[0]), corpus.jobs[item[1]].job_id))
    return [_recommendation(corpus.jobs[index], profile, "tfidf_cosine", rank, float(score), [], condition)
            for rank, (score, index) in enumerate(ranked[:top_k], 1)]


def _recommendation(job: JobRecord, profile: CandidateProfile, method: str, rank: int,
                    score: float, matched_terms: Sequence[str], condition: str = "all_eligible") -> dict[str, Any]:
    source = job.source
    title, occupation, description = _job_search_fields(source)
    description_assessment = assess_substantive_description(source)
    evidence = description[:DESCRIPTION_EVIDENCE_LIMIT] if description else title
    status = _nonblank(source.get("status_derived")) or "unknown"
    return {
        "profile_id": profile.profile_id,
        "method": method,
        "condition": condition,
        "rank": rank,
        "job_id": job.job_id,
        "title": _nonblank(source.get("title_raw")) or title,
        "searchable_title": title,
        "normalized_title": _nonblank(source.get("title_normalized")) or None,
        "searchable_title_source": source.get("title_searchable_source"),
        "occupation": occupation or None,
        "location_raw": source.get("location_raw"),
        "location_data_raw": source.get("location_data_raw"),
        "company_name": source.get("company_name_included"),
        "company_domain": source.get("company_domain_included"),
        "company_reference": source.get("company_reference"),
        "score": round(float(score), 8),
        "score_type": "cosine_similarity" if method == "tfidf_cosine" else "weighted_keyword_coverage",
        "source_url": source.get("url"),
        "historical_snapshot": bool(source.get("historical_snapshot", True)),
        "snapshot_date": source.get("snapshot_date"),
        "source_line": source.get("source_line"),
        "status_raw": source.get("status_raw"),
        "status_derived": status,
        "description_available": job.usable_description,
        "description_length": len(description),
        "description_evidence_classification": description_assessment.classification,
        "description_evidence_reason": description_assessment.reason,
        "matched_terms": list(matched_terms),
        "source_evidence_excerpt": evidence[:DESCRIPTION_EVIDENCE_LIMIT],
        "quality_flags": source.get("quality_flags", {}),
    }


def recommend_all(corpus: MatchingCorpus, profiles: Sequence[CandidateProfile], top_k: int = TOP_K
                  ) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for profile in profiles:
        recommendations.extend(rank_keyword(corpus, profile, top_k))
        recommendations.extend(rank_tfidf(corpus, profile, top_k))
    return recommendations


def pooled_review_rows(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pooled: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recommendations:
        key = (str(row["profile_id"]), str(row["job_id"]))
        entry = pooled.setdefault(key, {
            "profile_id": key[0], "job_id": key[1], "condition_ranks": {},
            "source_title": row.get("title"), "company_name": row.get("company_name"),
            "source_url": row.get("source_url"), "relevance_score": "", "reviewer": "", "notes": "",
        })
        condition = str(row.get("condition", "all_eligible"))
        method = str(row["method"])
        entry["condition_ranks"][f"{condition}|{method}"] = int(row["rank"])
    return [
        {**entry, "condition_ranks": json.dumps(entry["condition_ranks"], sort_keys=True)}
        for _, entry in sorted(pooled.items())
    ]


def reference_pools_from_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Union every retrieved job across methods/conditions into a profile pool."""
    pools: dict[str, set[str]] = {}
    for row in recommendations:
        pools.setdefault(str(row["profile_id"]), set()).add(str(row["job_id"]))
    return {profile_id: sorted(job_ids) for profile_id, job_ids in sorted(pools.items())}


def _dcg(labels: Sequence[int]) -> float:
    return sum(((2 ** label) - 1) / math.log2(rank + 1) for rank, label in enumerate(labels, 1))


def evaluate_reviewed_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[str, str], int],
    *,
    reference_pools: Mapping[str, Sequence[str]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Evaluate rankings against a shared, fully judged per-profile reference pool.

    Precision@k requires k returned and judged results. nDCG's ideal ranking is
    formed from the k highest labels in the shared pool, never from each method's
    own retrieved list. Missing reference-pool judgments withhold nDCG.
    """
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in recommendations:
        grouped.setdefault((str(row["profile_id"]), str(row["method"]),
                            str(row.get("condition", "all_eligible"))), []).append(row)

    metrics: list[dict[str, Any]] = []
    for (profile_id, method, condition), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["rank"]))[:k]
        judged: list[int] = []
        unjudged: list[int] = []
        for row in ordered:
            key = (profile_id, str(row["job_id"]))
            if key not in labels:
                unjudged.append(int(row["rank"]))
            else:
                judged.append(labels[key])
        precision_complete = len(ordered) == k and not unjudged
        precision = sum(label > 0 for label in judged) / k if precision_complete else None
        precision_status = "reported" if precision_complete else (
            "fewer_than_k_results" if len(ordered) < k else "incomplete_top_k_judgments"
        )
        reference_ids = list(dict.fromkeys(
            str(job_id) for job_id in reference_pools.get(profile_id, ())
        ))
        missing_reference_ids = sorted(
            job_id for job_id in reference_ids if (profile_id, job_id) not in labels
        )
        pool_set = set(reference_ids)
        outside_pool_ids = sorted({str(row["job_id"]) for row in ordered} - pool_set)
        ndcg: float | None = None
        if missing_reference_ids:
            ndcg_status = "incomplete_shared_reference_judgments"
        elif outside_pool_ids:
            ndcg_status = "recommendations_outside_reference_pool"
        elif not reference_ids:
            ndcg_status = "empty_reference_pool"
        else:
            # The ideal order is shared across every method/condition for this profile.
            ideal_labels = sorted((labels[(profile_id, job_id)] for job_id in reference_ids), reverse=True)[:k]
            actual_labels = [labels[(profile_id, str(row["job_id"]))] for row in ordered]
            ideal_dcg = _dcg(ideal_labels)
            actual_dcg = _dcg(actual_labels)
            if ideal_dcg == 0:
                ndcg = 0.0
                ndcg_status = "reported_shared_pool_all_zero_relevance"
            else:
                ndcg = actual_dcg / ideal_dcg
                ndcg_status = "reported_shared_pool"
        metrics.append({
            "profile_id": profile_id,
            "method": method,
            "condition": condition,
            "k": k,
            "result_count_at_k": len(ordered),
            "judged_at_k": len(judged),
            "unjudged_recommendation_ranks": unjudged,
            "precision_at_k": precision,
            "precision_status": precision_status,
            "reference_pool_size": len(reference_ids),
            "missing_reference_job_ids": missing_reference_ids,
            "outside_reference_pool_job_ids": outside_pool_ids,
            "ndcg_ideal_pool_items": min(k, len(reference_ids)),
            "ndcg_zero_padded_result_slots": max(0, k - len(ordered)),
            "ndcg_at_k": ndcg,
            "ndcg_status": ndcg_status,
        })
    return metrics


def read_review_labels(path: Path) -> dict[tuple[str, str], int]:
    """Read genuine 0/1/2 human labels; blanks remain unjudged."""
    labels: dict[tuple[str, str], int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source_file:
        for row_number, row in enumerate(csv.DictReader(source_file), 2):
            raw = (row.get("relevance_score") or "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError(f"Invalid relevance_score at CSV row {row_number}: use 0, 1, or 2") from error
            if value not in (0, 1, 2):
                raise ValueError(f"Invalid relevance_score at CSV row {row_number}: use 0, 1, or 2")
            key = ((row.get("profile_id") or "").strip(), (row.get("job_id") or "").strip())
            if not all(key):
                raise ValueError(f"Missing profile_id or job_id at CSV row {row_number}")
            labels[key] = value
    return labels


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def description_coverage_counts(jobs: Sequence[JobRecord], short_threshold: int = 250) -> dict[str, Any]:
    lengths = [len(_nonblank(job.source.get("description_text"))) for job in jobs]
    usable_lengths = [length for job, length in zip(jobs, lengths) if job.usable_description]
    return {
        "eligible_records": len(jobs),
        "description_usable": len(usable_lengths),
        "description_missing_or_blank": len(jobs) - len(usable_lengths),
        "short_description_threshold_chars": short_threshold,
        "short_usable_descriptions": sum(0 < length <= short_threshold for length in usable_lengths),
        "description_length_chars": {
            "minimum_usable": min(usable_lengths) if usable_lengths else None,
            "median_usable": float(np.median(usable_lengths)) if usable_lengths else None,
            "p90_usable": float(np.percentile(usable_lengths, 90)) if usable_lengths else None,
            "maximum_usable": max(usable_lengths) if usable_lengths else None,
        },
        "coverage_filter_note": "Short descriptions are counted for review but are not excluded by either condition.",
    }


def reconcile_previous_top_fives(
    recommendations_path: Path,
    jobs: Sequence[JobRecord],
    profile_ids: set[str],
    *,
    short_threshold: int = 250,
) -> dict[str, Any]:
    """Recount saved top-five descriptions for the previously reported profiles."""
    if not Path(recommendations_path).is_file():
        return {"status": "prior_recommendations_not_found", "path": str(recommendations_path)}
    by_id = {job.job_id: job for job in jobs}
    selected: list[tuple[str, bool, int]] = []
    with Path(recommendations_path).open("r", encoding="utf-8-sig", newline="") as source_file:
        for row in csv.DictReader(source_file):
            if row.get("profile_id") not in profile_ids or int(row.get("rank") or 0) > 5:
                continue
            job = by_id.get(str(row.get("job_id") or ""))
            if job is None:
                continue
            text = _nonblank(job.source.get("description_text"))
            selected.append((job.job_id, job.usable_description, len(text)))
    missing = [job_id for job_id, usable, _ in selected if not usable]
    short = [job_id for job_id, usable, length in selected if usable and 0 < length <= short_threshold]
    return {
        "status": "recounted_from_saved_recommendations_and_prepared_sample",
        "recommendations_path": str(recommendations_path),
        "profiles": sorted(profile_ids),
        "methods": ["keyword", "tfidf_cosine"],
        "top_k_per_method": 5,
        "recommendation_rows_matched_to_sample": len(selected),
        "missing_or_unusable_description_count": len(missing),
        "missing_or_unusable_job_ids": missing,
        "short_usable_description_threshold_chars": short_threshold,
        "short_usable_description_count": len(short),
        "short_usable_job_ids": short,
    }


def _candidate_evidence(profile: CandidateProfile) -> dict[str, Any]:
    return {
        "experience": list(profile.experience),
        "evidenced_skills": list(profile.evidenced_skills),
        "desired_roles": list(profile.desired_roles),
        "location": profile.location,
        "preferences": dict(profile.preferences),
    }


def _coverage_review_rows(
    corpus: MatchingCorpus,
    profiles: Sequence[CandidateProfile],
    recommendations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    pooled = pooled_review_rows(recommendations)
    profile_map = {profile.profile_id: profile for profile in profiles}
    job_map = {job.job_id: job for job in corpus.jobs}
    pools = reference_pools_from_recommendations(recommendations)
    blinded: list[dict[str, Any]] = []
    for row in pooled:
        job = job_map[row["job_id"]]
        source = job.source
        blinded.append({
            "profile_id": row["profile_id"],
            "candidate_evidence": json.dumps(_candidate_evidence(profile_map[row["profile_id"]]), ensure_ascii=False),
            "job_id": job.job_id,
            "source_title": _nonblank(source.get("title_raw")) or job.searchable_title,
            "searchable_title": job.searchable_title,
            "occupation": job.occupation,
            "location_raw": source.get("location_raw"),
            "company_name": source.get("company_name_included"),
            "source_url": source.get("url"),
            "description_available": job.usable_description,
            "description_length": len(_nonblank(source.get("description_text"))),
            "description_raw_length": len(_nonblank(source.get("description_raw"))),
            "description_full_prepared": source.get("description_text") or "",
            "status_raw": source.get("status_raw"),
            "status_derived": _nonblank(source.get("status_derived")) or "unknown",
            "quality_flags": json.dumps(source.get("quality_flags", {}), ensure_ascii=False),
            "relevance_score": "",
            "reviewer": "",
            "notes": "",
        })
    # Shuffle pool rows deterministically without exposing method, condition, rank, or score.
    blinded.sort(key=lambda row: hashlib.sha256(
        f"{row['profile_id']}\0{row['job_id']}\0review-v1".encode("utf-8")
    ).hexdigest())
    return pooled, blinded, pools


def _description_evidence_diagnostics(corpus: MatchingCorpus, limit_per_group: int = 10
                                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assessments = list(zip(corpus.jobs, corpus.assess_descriptions()))
    class_counts = Counter(assessment.classification for _, assessment in assessments)
    reason_counts = Counter(assessment.reason for _, assessment in assessments)
    groups = {
        "accepted": [(job, assessment) for job, assessment in assessments if assessment.eligible],
        "rejected": [(job, assessment) for job, assessment in assessments if not assessment.eligible],
    }
    rows: list[dict[str, Any]] = []
    for group, items in groups.items():
        chosen = sorted(items, key=lambda item: hashlib.sha256(
            f"{DESCRIPTION_EVIDENCE_RULE_VERSION}\0{item[0].job_id}".encode("utf-8")
        ).hexdigest())[:limit_per_group]
        for job, assessment in chosen:
            source = job.source
            rows.append({
                "selection_group": group,
                "job_id": job.job_id,
                "title": _nonblank(source.get("title_raw")) or job.searchable_title,
                "source_line": source.get("source_line"),
                "description_length": len(_nonblank(source.get("description_text"))),
                "classification": assessment.classification,
                "reason": assessment.reason,
                "duty_evidence": json.dumps(assessment.duty_evidence, ensure_ascii=False),
                "qualification_evidence": json.dumps(assessment.qualification_evidence, ensure_ascii=False),
                "description_excerpt": _nonblank(source.get("description_text"))[:DESCRIPTION_EVIDENCE_LIMIT],
                "review_note": (
                    "Possible false exclusion: inspect whether concrete duties or qualifications are present "
                    "despite the v1 rule rejecting this description."
                    if not assessment.eligible else ""
                ),
            })
    rows.sort(key=lambda row: (row["selection_group"], row["job_id"]))
    counts = {
        "rule_version": DESCRIPTION_EVIDENCE_RULE_VERSION,
        "eligible_count": sum(assessment.eligible for _, assessment in assessments),
        "ineligible_count": sum(not assessment.eligible for _, assessment in assessments),
        "classification_counts": dict(sorted(class_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "diagnostic_limit_per_group": limit_per_group,
        "diagnostic_rows_by_group": {key: min(len(value), limit_per_group) for key, value in groups.items()},
        "diagnostic_sampling": "Deterministic SHA-256 ordering by rule version and stable job ID.",
        "warning": "The v1 evidence heuristic is unvalidated; rejected examples may be false exclusions.",
    }
    return rows, counts


def _top_five_changes(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in recommendations:
        grouped.setdefault((str(row["profile_id"]), str(row["method"]),
                            str(row["condition"])), []).append(row)
    keys = sorted({(profile_id, method) for profile_id, method, _ in grouped})
    changes: list[dict[str, Any]] = []
    for profile_id, method in keys:
        available = sorted(grouped.get((profile_id, method, "description_available"), ()),
                           key=lambda row: int(row["rank"]))[:5]
        substantive = sorted(grouped.get((profile_id, method, "substantive_description"), ()),
                             key=lambda row: int(row["rank"]))[:5]
        available_ranks = {str(row["job_id"]): int(row["rank"]) for row in available}
        substantive_ranks = {str(row["job_id"]): int(row["rank"]) for row in substantive}
        changes.append({
            "profile_id": profile_id,
            "method": method,
            "description_available_top5_job_ids": json.dumps(list(available_ranks)),
            "substantive_description_top5_job_ids": json.dumps(list(substantive_ranks)),
            "entered_substantive_top5_job_ids": json.dumps(
                [job_id for job_id in substantive_ranks if job_id not in available_ranks]
            ),
            "left_description_available_top5_job_ids": json.dumps(
                [job_id for job_id in available_ranks if job_id not in substantive_ranks]
            ),
            "common_job_rank_shifts": json.dumps([
                {"job_id": job_id, "description_available_rank": available_ranks[job_id],
                 "substantive_description_rank": substantive_ranks[job_id],
                 "rank_change": substantive_ranks[job_id] - available_ranks[job_id]}
                for job_id in available_ranks if job_id in substantive_ranks
            ]),
        })
    return changes


def _new_unlabeled_retail_jobs(
    corpus: MatchingCorpus,
    profiles: Sequence[CandidateProfile],
    recommendations: Sequence[Mapping[str, Any]],
    existing_labels: Mapping[tuple[str, str], int],
    *,
    retail_profile_id: str = "synthetic_retail_inventory",
) -> list[dict[str, Any]]:
    profile = next((item for item in profiles if item.profile_id == retail_profile_id), None)
    if profile is None:
        return []
    retrieved: dict[str, set[str]] = {"description_available": set(), "substantive_description": set()}
    for row in recommendations:
        if row["profile_id"] == retail_profile_id:
            retrieved[str(row["condition"])].add(str(row["job_id"]))
    new_ids = (retrieved["substantive_description"] - retrieved["description_available"])
    new_ids = {job_id for job_id in new_ids if (retail_profile_id, job_id) not in existing_labels}
    jobs = {job.job_id: job for job in corpus.jobs}
    rows: list[dict[str, Any]] = []
    for job_id in new_ids:
        job = jobs[job_id]
        source = job.source
        rows.append({
            "profile_id": retail_profile_id,
            "candidate_evidence": json.dumps(_candidate_evidence(profile), ensure_ascii=False),
            "job_id": job_id,
            "source_title": _nonblank(source.get("title_raw")) or job.searchable_title,
            "searchable_title": job.searchable_title,
            "occupation": job.occupation,
            "location_raw": source.get("location_raw"),
            "company_name": source.get("company_name_included"),
            "source_url": source.get("url"),
            "description_length": len(_nonblank(source.get("description_text"))),
            "description_full_prepared": source.get("description_text") or "",
            "relevance_score": "",
            "reviewer": "",
            "notes": "",
        })
    rows.sort(key=lambda row: hashlib.sha256(
        f"{retail_profile_id}\0{row['job_id']}\0new-retail-review-v1".encode("utf-8")
    ).hexdigest())
    return rows


def write_description_coverage_outputs(
    output_dir: Path,
    corpus: MatchingCorpus,
    profiles: Sequence[CandidateProfile],
    recommendations: Sequence[Mapping[str, Any]],
    condition_counts: Mapping[str, int],
    sample_path: Path,
    profiles_path: Path,
    timings: Mapping[str, float],
    prior_baseline_path: Path,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if labels_path is not None:
        labels_path = Path(labels_path)
        if not labels_path.is_absolute():
            labels_path = PROJECT_ROOT / labels_path
    output_dir.mkdir(parents=True, exist_ok=False)
    rec_path = output_dir / "description_coverage_recommendations.csv"
    pool_path = output_dir / "pooled_review_template.csv"
    blinded_path = output_dir / "blinded_review_view.csv"
    diagnostics_path = output_dir / "description_evidence_diagnostics.csv"
    top_five_path = output_dir / "top5_changes.csv"
    fields = [
        "profile_id", "method", "condition", "rank", "job_id", "title", "searchable_title",
        "normalized_title", "searchable_title_source", "occupation", "location_raw", "location_data_raw",
        "company_name", "company_domain", "company_reference", "score", "score_type", "source_url",
        "historical_snapshot", "description_available", "description_length", "description_raw_length",
        "description_evidence_classification", "description_evidence_reason",
        "matched_terms", "source_evidence_excerpt", "quality_flags", "status_raw", "status_derived",
        "snapshot_date", "source_line",
    ]
    with rec_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in recommendations:
            writer.writerow({**row,
                             "location_data_raw": json.dumps(row["location_data_raw"], ensure_ascii=False),
                             "company_reference": json.dumps(row["company_reference"], ensure_ascii=False),
                             "matched_terms": json.dumps(row["matched_terms"], ensure_ascii=False),
                             "quality_flags": json.dumps(row["quality_flags"], ensure_ascii=False)})

    pooled = pooled_review_rows(recommendations)
    reference_pools = reference_pools_from_recommendations(recommendations)
    pool_fields = ["profile_id", "job_id", "condition_ranks", "source_title", "company_name", "source_url",
                   "relevance_score", "reviewer", "notes"]
    with pool_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=pool_fields)
        writer.writeheader()
        writer.writerows(pooled)

    labels: dict[tuple[str, str], int] = {}
    if labels_path is not None:
        labels = read_review_labels(labels_path)
    blinded = _new_unlabeled_retail_jobs(corpus, profiles, recommendations, labels)
    blind_fields = [
        "profile_id", "candidate_evidence", "job_id", "source_title", "searchable_title", "occupation",
        "location_raw", "company_name", "source_url", "description_length", "description_full_prepared",
        "relevance_score", "reviewer", "notes",
    ]
    with blinded_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=blind_fields)
        writer.writeheader()
        writer.writerows(blinded)

    metrics = evaluate_reviewed_recommendations(recommendations, labels, reference_pools=reference_pools)
    metrics_path = output_dir / "shared_pool_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    diagnostic_rows, evidence_counts = _description_evidence_diagnostics(corpus)
    diagnostic_fields = [
        "selection_group", "job_id", "title", "source_line", "description_length", "classification",
        "reason", "duty_evidence", "qualification_evidence", "description_excerpt", "review_note",
    ]
    with diagnostics_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    top_five_changes = _top_five_changes(recommendations)
    with top_five_path.open("w", encoding="utf-8-sig", newline="") as output:
        fields = list(top_five_changes[0]) if top_five_changes else [
            "profile_id", "method", "description_available_top5_job_ids",
            "substantive_description_top5_job_ids", "entered_substantive_top5_job_ids",
            "left_description_available_top5_job_ids", "common_job_rank_shifts",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(top_five_changes)

    profile_counts: dict[str, dict[str, int]] = {}
    for profile in profiles:
        profile_counts[profile.profile_id] = {}
        for condition in ("description_available", "substantive_description"):
            indices = condition_job_indices(corpus, condition)
            eligible, _ = _eligible_jobs(corpus, profile, indices)
            profile_counts[profile.profile_id][condition] = len(eligible)

    summary = {
        "experiment": "substantive_description_evidence_comparison",
        "description_evidence_rule": {
            "version": DESCRIPTION_EVIDENCE_RULE_VERSION,
            "decision": "Require at least one non-boilerplate line with a concrete duty action and object, or an explicit qualification signal.",
            "minimum_signal_words": {
                "duty_line": DESCRIPTION_EVIDENCE_MIN_DUTY_WORDS,
                "qualification_line": DESCRIPTION_EVIDENCE_MIN_QUALIFICATION_WORDS,
            },
            "not_used_as_rules": ["overall description length", "mandatory section headings"],
            "validation_status": "heuristic, unvalidated; rejected cases may be false exclusions",
        },
        "sample_path": str(sample_path),
        "sample_sha256": sha256_file(sample_path),
        "profiles_path": str(profiles_path),
        "profiles_sha256": sha256_file(profiles_path),
        "profile_ids": [profile.profile_id for profile in profiles],
        "ranking_configuration": {
            "keyword_source_weights": {"title": 3, "onet_occupation": 2, "description": 1},
            "keyword_profile_weights": {"desired_roles": 3, "experience_and_skills": 2, "preferences_and_location": 1},
            "tfidf_job_text_weights": {"title_repetitions": 3, "onet_occupation_repetitions": 2, "description_repetitions": 1},
            "tfidf_profile_text_weights": {"desired_roles_repetitions": 3, "experience_and_skills_repetitions": 2, "preferences_and_location_repetitions": 1},
            "tfidf_ngram_range": [1, 2],
            "tfidf_max_features": 250_000,
            "tokenizer_preserves": ["C++", "C#", ".NET"],
        },
        "tfidf_fit_count": 1,
        "tfidf_reused_for_both_conditions": True,
        "candidate_profiles_and_ranking_controls_unchanged": True,
        "conditions": {
            "description_available": {
                "description_eligibility": "description_usable is true and description_text is nonblank.",
                "corpus_records": condition_counts["description_available"],
            },
            "substantive_description": {
                "description_eligibility": "Rule v1 accepts explicit non-boilerplate duty or qualification evidence.",
                "corpus_records": condition_counts["substantive_description"],
            },
        },
        "description_evidence_assessment_counts": evidence_counts,
        "independent_label_provenance": {
            "path": str(labels_path) if labels_path else None,
            "sha256": sha256_file(labels_path) if labels_path else None,
            "source_note": "Existing independent retail evaluation only; conversational judgments were not loaded.",
            "numeric_label_pairs_loaded": len(labels),
            "provenance": "model-based independent review" if labels_path else None,
            "matched_pairs_in_shared_retrieval_pool": sum(
                1 for profile_id, job_ids in reference_pools.items()
                for job_id in job_ids if (profile_id, job_id) in labels
            ),
            "new_recommendation_pairs_labeled": 0,
        },
        "new_blinded_retail_review_queue": {
            "selection": "substantive-condition top-10 union minus description-available top-10 union and minus existing retail labels",
            "row_count": len(blinded),
            "profile_id": "synthetic_retail_inventory",
        },
        "profile_condition_eligible_counts_after_existing_filters": profile_counts,
        "description_coverage_and_length": description_coverage_counts(corpus.jobs),
        "previous_top_five_description_reconciliation": reconcile_previous_top_fives(
            prior_baseline_path, corpus.jobs,
            {"synthetic_customer_support_technical_training", "synthetic_retail_inventory"},
        ),
        "reference_pool_job_counts": {profile_id: len(job_ids) for profile_id, job_ids in reference_pools.items()},
        "metrics_status": "computed_from_supplied_human_labels" if labels else "not_reported_no_genuine_human_labels",
        "metrics_file": metrics_path.name,
        "recommendation_count": len(recommendations),
        "pooled_review_rows": len(pooled),
        "blinded_review_rows": len(blinded),
        "timings_seconds": dict(timings),
        "outputs": {
            "recommendations": rec_path.name,
            "pooled_review_template": pool_path.name,
            "blinded_review_view": blinded_path.name,
            "shared_pool_metrics": metrics_path.name,
            "description_evidence_diagnostics": diagnostics_path.name,
            "top5_changes": top_five_path.name,
        },
        "review_instructions": {
            "labels": {"0": "unsuitable", "1": "plausible", "2": "strong fit"},
            "blank": "Leave relevance_score blank when available evidence is insufficient.",
            "blinding": "blinded_review_view.csv contains only previously unlabeled retail jobs newly appearing in the substantive condition; it omits method, condition, rank, and score.",
            "pool": "The shared per-profile pool for metrics is the union of jobs retrieved by both methods and both conditions.",
            "ndcg": "Label every job in a profile's shared pool before nDCG is reported. Missing pool judgments withhold nDCG and are listed in shared_pool_metrics.json.",
            "precision": "Precision@5 is reported only when five results are returned and all five have genuine labels.",
            "short_results": "Fewer than five returned results use zero gain for missing ranks in nDCG; Precision@5 is withheld.",
            "short_pool": "If the pool has fewer than five jobs, its available labels define the ideal and remaining ideal ranks have zero gain.",
            "all_zero_pool": "A fully judged pool with all-zero relevance has nDCG 0.",
            "new_blind_queue": "Do not add or infer labels for any other profile or any previously labeled pair in this experiment.",
        },
    }
    summary_path = output_dir / "description_coverage_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol_path = output_dir / "review_instructions.md"
    protocol_path.write_text(
        "# Blinded Recommendation Review\n\n"
        "Use `blinded_review_view.csv`; it contains only previously unlabeled retail jobs that enter "
        "the substantive-description condition's top-10 union but not the description-available "
        "condition's top-10 union. It contains no method, condition, rank, or score fields. "
        "Review candidate evidence beside the full prepared job description. Label relevance_score "
        "0 = unsuitable, 1 = plausible, or 2 = strong fit. Leave blank when the available evidence "
        "is insufficient. Do not infer an unmentioned skill is absent or unnecessary. Preserve the "
        "completed file as a review copy; do not edit generated rankings. Do not add labels for "
        "other profiles or reuse a label for a different profile/job pair. The separate "
        "`pooled_review_template.csv` records condition/method ranks for audit and should not be used "
        "as the blinded judging view. The shared pool for reported metrics is the union of jobs retrieved by both methods "
        "under both conditions. Label the entire per-profile pool for comparable nDCG; missing pool "
        "judgments withhold nDCG. Precision@5 requires five returned results and five genuine labels. "
        "Fewer than five results have zero gain at missing ranks for nDCG. If the pool has fewer than "
        "five jobs, its available labels define the ideal and remaining ideal ranks are zero-gain. "
        "An all-zero judged pool yields nDCG 0.\n",
        encoding="utf-8",
    )
    return summary


def write_outputs(output_dir: Path, recommendations: Sequence[Mapping[str, Any]],
                  load_report: LoadReport, elapsed_seconds: float,
                  profiles: Sequence[CandidateProfile], labels_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation_path = output_dir / "recommendations.csv"
    review_path = output_dir / "relevance_review_template.csv"
    fields = [
        "profile_id", "method", "condition", "rank", "job_id", "title", "searchable_title", "normalized_title",
        "searchable_title_source", "occupation", "company_name", "company_domain", "company_reference",
        "score", "score_type", "source_url", "historical_snapshot", "snapshot_date", "source_line",
        "status_raw", "status_derived", "description_available", "description_length", "matched_terms",
        "source_evidence_excerpt", "quality_flags",
    ]
    with recommendation_path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in recommendations:
            writer.writerow({**row,
                             "matched_terms": json.dumps(row["matched_terms"], ensure_ascii=False),
                             "company_reference": json.dumps(row["company_reference"], ensure_ascii=False),
                             "quality_flags": json.dumps(row["quality_flags"], ensure_ascii=False)})

    review_rows = pooled_review_rows(recommendations)
    review_fields = ["profile_id", "job_id", "condition_ranks", "source_title", "company_name",
                     "source_url", "relevance_score", "reviewer", "notes"]
    with review_path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)

    summary: dict[str, Any] = {
        "sample_path": load_report.path,
        "load_report": load_report.to_dict(),
        "synthetic_profile_count": len(profiles),
        "profile_ids": [profile.profile_id for profile in profiles],
        "recommendation_count": len(recommendations),
        "methods": sorted({str(row["method"]) for row in recommendations}),
        "top_k": TOP_K,
        "ranking_configuration": {
            "keyword_source_weights": {"title": 3, "onet_occupation": 2, "description": 1},
            "keyword_profile_weights": {"desired_roles": 3, "experience_and_skills": 2, "preferences_and_location": 1},
            "tfidf_job_text_weights": {"title_repetitions": 3, "onet_occupation_repetitions": 2, "description_repetitions": 1},
            "tfidf_profile_text_weights": {"desired_roles_repetitions": 3, "experience_and_skills_repetitions": 2, "preferences_and_location_repetitions": 1},
            "tfidf_ngram_range": [1, 2],
            "tfidf_max_features": 250_000,
            "tokenizer_preserves": ["C++", "C#", ".NET"],
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "recommendations_csv": recommendation_path.name,
        "review_template_csv": review_path.name,
        "metrics_status": "not_reported_no_genuine_review_labels",
    }
    if labels_path is not None:
        labels = read_review_labels(labels_path)
        if labels:
            summary["metrics_status"] = "reported_only_for_complete_judged_top_5"
            summary["metrics"] = evaluate_reviewed_recommendations(
                recommendations, labels,
                reference_pools=reference_pools_from_recommendations(recommendations),
            )
            metrics_path = output_dir / "review_metrics.json"
            metrics_path.write_text(json.dumps(summary["metrics"], indent=2), encoding="utf-8")
            summary["metrics_file"] = metrics_path.name
        else:
            summary["metrics_status"] = "not_reported_review_file_has_no_labels"
    summary_path = output_dir / "matching_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _self_test() -> None:
    import tempfile

    assert tokenize("C++ C# .NET") == ["c++", "c#", ".net"]
    profile = CandidateProfile(
        profile_id="synthetic_test",
        experience=("customer support troubleshooting",),
        evidenced_skills=("C++", "network diagnostics"),
        desired_roles=("technical support engineer",),
        location="Chicago",
    )
    assert "c++" in _candidate_query_terms(profile)
    with tempfile.TemporaryDirectory() as directory:
        sample = Path(directory) / "jobs.jsonl"
        records = [
            {"job_id": "job-a", "title_raw": "Technical Support Engineer", "title_searchable": "Technical Support Engineer",
             "description_text": "Troubleshoot C++ network devices", "description_usable": True,
             "onet_data": {"occupation_name": "Computer User Support Specialists"}, "location_raw": "Chicago, Illinois",
             "status_derived": "unknown", "quality_flags": {"description": "present"}, "url": "https://example.invalid/a"},
            {"job_id": "job-b", "title_raw": "Retail Associate", "title_searchable": "Retail Associate",
             "description_text": None, "description_usable": False, "onet_data": {"occupation_name": "Retail Salespersons"},
             "location_raw": None, "status_derived": "closed", "quality_flags": {"description": "null"}},
            {"job_id": "job-a", "title_raw": "Duplicate ID", "description_text": "ignored"},
            {"title_raw": "No stable ID", "description_text": "excluded"},
            {"job_id": "job-empty", "title_raw": "!!! ###"},
        ]
        with sample.open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record) + "\n")
            out.write("not-json\n")
        jobs, report = load_prepared_jobs(sample)
        assert [job.job_id for job in jobs] == ["job-a", "job-b"]
        assert report.records_loaded == 2 and report.usable_descriptions == 1
        assert report.excluded_by_reason["duplicate_job_id"] == 1
        assert report.excluded_by_reason["missing_stable_job_id"] == 1
        assert report.excluded_by_reason["no_searchable_text"] == 1
        assert report.excluded_by_reason["invalid_json"] == 1
        corpus = fit_tfidf(jobs)
        keyword = rank_keyword(corpus, profile)
        tfidf = rank_tfidf(corpus, profile)
        assert keyword[0]["job_id"] == "job-a" and tfidf[0]["job_id"] == "job-a"
        assert keyword[0]["rank"] == 1 and tfidf[0]["rank"] == 1
        assert jobs[1].source["description_text"] is None
        assert filter_job(jobs[1], profile).included
        excluded_closed = CandidateProfile("exclude_closed", preferences={"exclude_explicitly_closed": True})
        assert filter_job(jobs[1], excluded_closed) == FilterDecision(False, "explicitly_closed")
        missing_location = CandidateProfile("location_required", location="Seattle",
                                            preferences={"require_location_match": True})
        assert filter_job(jobs[1], missing_location) == FilterDecision(False, "unknown_location")
        recs = [
            {"profile_id": "p", "method": "m", "rank": rank, "job_id": f"j{rank}"}
            for rank in range(1, 6)
        ]
        pool = {"p": [f"j{rank}" for rank in range(1, 6)]}
        metrics = evaluate_reviewed_recommendations(
            recs, {("p", "j1"): 2}, reference_pools=pool
        )[0]
        assert metrics["precision_at_k"] is None
        assert metrics["unjudged_recommendation_ranks"] == [2, 3, 4, 5]
        complete = evaluate_reviewed_recommendations(
            recs,
            {("p", f"j{rank}"): (2 if rank == 1 else 0) for rank in range(1, 6)},
            reference_pools=pool,
        )[0]
        assert complete["precision_at_k"] == 0.2 and complete["ndcg_at_k"] == 1.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE,
                        help="local prepared JSONL sample; no remote source is used")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", type=Path, default=Path(__file__).with_name("synthetic_candidate_profiles.json"),
                        help="local JSON array of candidate profiles")
    parser.add_argument("--labels", type=Path, default=None,
                        help="optional human-reviewed CSV; metrics require fully judged top five")
    parser.add_argument("--description-coverage-experiment", action="store_true",
                        help="compare description-available and substantive-description conditions on one fitted TF-IDF corpus")
    parser.add_argument("--experiment-output-dir", type=Path, default=None,
                        help="fresh output directory for the substantive-description experiment")
    parser.add_argument("--prior-baseline-recommendations", type=Path,
                        default=DEFAULT_PRIOR_BASELINE,
                        help="saved baseline CSV used only to reconcile prior top-five description counts")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        print("Job matching offline checks passed.")
        return 0

    sample_path = args.sample if args.sample.is_absolute() else PROJECT_ROOT / args.sample
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    total_started = time.perf_counter()
    sample_load_started = time.perf_counter()
    try:
        jobs, load_report = load_prepared_jobs(sample_path)
    except FileNotFoundError as error:
        print(f"{error}\nObtain the approved prepared sample and place it at this path; no scan was run.", file=sys.stderr)
        return 2
    if not jobs:
        print(json.dumps(load_report.to_dict(), indent=2), file=sys.stderr)
        return 2

    sample_load_seconds = time.perf_counter() - sample_load_started
    profile_load_started = time.perf_counter()
    profiles_path = args.profiles if args.profiles.is_absolute() else PROJECT_ROOT / args.profiles
    try:
        profiles = load_candidate_profiles(profiles_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Cannot load candidate profiles from {profiles_path}: {error}", file=sys.stderr)
        return 2

    profile_load_seconds = time.perf_counter() - profile_load_started
    if args.description_coverage_experiment:
        output_dir = args.experiment_output_dir
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("description_coverage_%Y%m%dT%H%M%SZ")
            output_dir = DEFAULT_OUTPUT.parent / stamp
        elif not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        if output_dir.exists():
            parser.error(f"Refusing to overwrite existing experiment output directory: {output_dir}")

        fit_started = time.perf_counter()
        corpus = fit_tfidf(jobs)
        tfidf_fit_seconds = time.perf_counter() - fit_started
        condition_counts = {
            "description_available": sum(job.usable_description for job in corpus.jobs),
            "substantive_description": sum(
                assessment.eligible for assessment in corpus.assess_descriptions()
            ),
        }
        ranking_started = time.perf_counter()
        recommendations = rank_description_coverage_on_corpus(corpus, profiles)
        candidate_ranking_seconds = time.perf_counter() - ranking_started
        timings = {
            "sample_loading_seconds": round(sample_load_seconds, 6),
            "profile_loading_seconds": round(profile_load_seconds, 6),
            "tfidf_fit_seconds": round(tfidf_fit_seconds, 6),
            "candidate_ranking_seconds": round(candidate_ranking_seconds, 6),
        }
        output_started = time.perf_counter()
        summary = write_description_coverage_outputs(
            output_dir, corpus, profiles, recommendations, condition_counts,
            sample_path, profiles_path, timings,
            args.prior_baseline_recommendations if args.prior_baseline_recommendations.is_absolute()
            else PROJECT_ROOT / args.prior_baseline_recommendations,
            args.labels if args.labels is None or args.labels.is_absolute() else PROJECT_ROOT / args.labels,
        )
        timings["output_writing_seconds"] = round(time.perf_counter() - output_started, 6)
        timings["total_seconds_excluding_process_startup"] = round(time.perf_counter() - total_started, 6)
        summary["timings_seconds"] = timings
        (output_dir / "description_coverage_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0

    started = time.monotonic()
    corpus = fit_tfidf(jobs)
    recommendations = recommend_all(corpus, profiles)
    elapsed = time.monotonic() - started
    summary = write_outputs(output_dir, recommendations, load_report, elapsed,
                            profiles, args.labels)
    print(json.dumps(summary, indent=2))
    print("Similarity scores are retrieval scores, not hiring probabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
