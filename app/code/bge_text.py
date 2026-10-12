"""BGE text representation used by the live ranker: job documents, candidate queries, model location.

This is the part of the original BGE transition experiment that the app depends on, extracted so the app no
longer imports experiment code. The logic is a faithful copy: job and profile text must stay byte-identical to
what produced the saved listing embeddings, otherwise queries and listing vectors stop matching. Change it only
together with re-encoding the corpus (`code/rank_candidates.py encode`).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import job_matching_engine as engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent  # data/, models/ and .env are shared with analysis/ at the top

BGE_ID = "BAAI/bge-base-en-v1.5"
BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_REPRESENTATION = "bge_section_priority_v1"
MAX_BGE_TOKENS = 512

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, for fetch/ (shared with analysis/)
from fetch.paths import DATA_DIR, MODELS_DIR  # noqa: E402  shared with analysis/: one definition of where data lives
MODEL_DIR_ENV = "PREDICTLEADS_BGE_MODEL_DIR"
DEFAULT_MODEL_DIR = MODELS_DIR / "bge-base-en-v1.5"
_LEGACY_MODEL_DIR = DATA_DIR / "matching" / "bge_transition_20261008T142818Z" / "bge_model_files"


def model_dir() -> Path:
    """Local BGE model files: $PREDICTLEADS_BGE_MODEL_DIR, else models/bge-base-en-v1.5, else the legacy experiment copy."""
    configured = os.environ.get(MODEL_DIR_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.is_dir() or not _LEGACY_MODEL_DIR.is_dir() else _LEGACY_MODEL_DIR


# ---------- section-aware parsing of job descriptions ----------

_HEADING = re.compile(r"^[\s\-*•#|:]{0,6}([^:|]{2,100}?)[\s:|]*$", re.IGNORECASE)
_REQUIREMENT_HEADING = re.compile(
    r"\b(requirements?|qualifications?|what you(?:'|’)ll bring|minimum qualifications|"
    r"required qualifications|education and experience|experience requirements)\b", re.IGNORECASE
)
_DUTY_HEADING = re.compile(
    r"\b(responsibilities|duties|essential functions|what you(?:'|’)ll do|job functions|"
    r"key accountabilities|role summary|job summary)\b", re.IGNORECASE
)
_NOISE_HEADING = re.compile(
    r"\b(company overview|about us|who we are|why join|benefits|compensation|equal opportunity|"
    r"eeo statement|our culture|total rewards|privacy notice)\b", re.IGNORECASE
)
_REQUIREMENT_SIGNAL = re.compile(
    r"\b(?:required|must have|must be|minimum|at least|preferred|qualifications?|"
    r"\d+\+?\s+years?|license required|certification required)\b", re.IGNORECASE
)
_DUTY_SIGNAL = re.compile(
    r"\b(?:manage|perform|support|assist|coordinate|maintain|process|provide|develop|ensure|"
    r"operate|troubleshoot|lead|deliver|handle|monitor|prepare|review|create|conduct|build|"
    r"implement|execute|receive|stock|count|schedule|resolve|communicate|track|reconcile|"
    r"inspect|train|supervise|install|test|design|write|analyze|analyse|record|collect)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=(?:[-*•]\s*)?[A-Z0-9])")


def _token_count(tokenizer: Any, text: str, *, special: bool = True) -> int:
    return len(tokenizer(text, add_special_tokens=special, truncation=False)["input_ids"])


def _sentences(text: str) -> list[str]:
    output: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" \t-*•|")
        if not cleaned:
            continue
        output.extend(part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip())
    return output


def _job_sections(description: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"requirement": [], "duty": [], "other": []}
    state = "other"
    for line in description.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip(" \t-*•|:#")
        heading_match = _HEADING.match(cleaned) if len(cleaned) <= 100 else None
        if heading_match:
            heading = heading_match.group(1).strip()
            if _NOISE_HEADING.search(heading):
                state = "noise"
                continue
            if _REQUIREMENT_HEADING.search(heading):
                state = "requirement"
                continue
            if _DUTY_HEADING.search(heading):
                state = "duty"
                continue
        if state == "noise":
            continue
        for sentence in _sentences(cleaned):
            if state == "requirement" or _REQUIREMENT_SIGNAL.search(sentence):
                category = "requirement"
            elif state == "duty" or _DUTY_SIGNAL.search(sentence):
                category = "duty"
            else:
                category = "other"
            sections[category].append(sentence)
    for category in sections:
        seen: set[str] = set()
        sections[category] = [sentence for sentence in sections[category]
                              if sentence.casefold() not in seen and not seen.add(sentence.casefold())]
    return sections


class _TokenBudget:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_count = 2
        self.pieces: list[str] = []

    def append(self, piece: str) -> bool:
        piece_tokens = _token_count(self.tokenizer, piece, special=False)
        if self.token_count + piece_tokens > self.max_length:
            return False
        self.pieces.append(piece)
        self.token_count += piece_tokens
        return True

    def text(self) -> str:
        return " ".join(self.pieces)


# ---------- candidate query text ----------

def _candidate_parts(profile: engine.CandidateProfile, fixture: Mapping[str, Any]) -> list[tuple[str, str]]:
    sections: list[tuple[str, Sequence[str]]] = [
        ("Target roles", profile.desired_roles),
        ("Experience level", [str(fixture.get("experience_level", ""))]),
        ("Transition goal", [str(fixture.get("career_transition_goal", ""))]),
        ("Work history", profile.experience[:len(fixture.get("work_history", []))]),
        ("Accomplishments", fixture.get("accomplishments", [])),
        ("Demonstrated skills", profile.evidenced_skills),
        ("Education and certifications", fixture.get("education_certifications", [])),
        ("Soft preferences", fixture.get("soft_preferences", [])),
        ("Work arrangement preferences", fixture.get("work_arrangement_preferences", [])),
    ]
    return [(label, "{}: {}".format(label, value.strip()))
            for label, values in sections for value in values if str(value).strip()]


def build_bge_candidate_text(
    profile: engine.CandidateProfile, fixture: Mapping[str, Any], tokenizer: Any,
) -> tuple[str, dict[str, Any]]:
    parts = _candidate_parts(profile, fixture)
    prefix_tokens = _token_count(tokenizer, BGE_QUERY_PREFIX, special=False)
    budget = _TokenBudget(tokenizer, MAX_BGE_TOKENS - prefix_tokens)
    retained: list[tuple[str, str]] = []
    omitted: list[tuple[str, str]] = []
    for label, part in parts:
        (retained if budget.append(part) else omitted).append((label, part))
    content = budget.text()
    encoded_length = _token_count(tokenizer, BGE_QUERY_PREFIX + content)
    original = " ".join(part for _, part in parts)
    original_length = _token_count(tokenizer, BGE_QUERY_PREFIX + original)
    labels = sorted({label for label, _ in parts})
    kept_labels = {label for label, _ in retained}
    report = {
        "original_tokens_including_prefix_and_specials": original_length,
        "encoded_tokens_including_prefix_and_specials": encoded_length,
        "omitted_tokens_estimate": max(0, original_length - encoded_length),
        "original_sections": labels,
        "retained_sections": sorted(kept_labels),
        "omitted_sections": sorted(set(labels) - kept_labels),
        "omitted_items": len(omitted),
    }
    if encoded_length > MAX_BGE_TOKENS:
        raise AssertionError("BGE candidate input exceeds the 512-token limit.")
    return content, report


# ---------- job document text ----------

def build_bge_job_text(record: Mapping[str, Any], tokenizer: Any) -> tuple[str, dict[str, Any]]:
    title, occupation, description = engine._job_search_fields(record)
    header = "Job title: {}. Occupation: {}.".format(title, occupation)
    buckets = _job_sections(description)
    requirement, duty, other = buckets["requirement"], buckets["duty"], buckets["other"]
    tagged: list[tuple[str, str]] = [("header", header)]
    for index in range(max(len(requirement), len(duty))):
        if index < len(requirement):
            tagged.append(("requirements", "Requirement: " + requirement[index]))
        if index < len(duty):
            tagged.append(("duties", "Duty: " + duty[index]))
    tagged.extend(("other", "Description: " + sentence) for sentence in other)
    budget = _TokenBudget(tokenizer, MAX_BGE_TOKENS)
    retained: list[tuple[str, str]] = []
    omitted: list[tuple[str, str]] = []
    for section, part in tagged:
        (retained if budget.append(part) else omitted).append((section, part))
    content = budget.text()
    encoded_length = _token_count(tokenizer, content)
    original = " ".join(part for _, part in tagged)
    original_length = _token_count(tokenizer, original)
    all_sections = {section for section, _ in tagged}
    retained_sections = {section for section, _ in retained}
    report = {
        "original_tokens_including_specials": original_length,
        "encoded_tokens_including_specials": encoded_length,
        "omitted_tokens_estimate": max(0, original_length - encoded_length),
        "original_sections": sorted(all_sections),
        "retained_sections": sorted(retained_sections),
        "omitted_sections": sorted(all_sections - retained_sections),
        "omitted_items": len(omitted),
        "noise_section_handling": "recognized noise-heading sections excluded by the existing section-aware parser before token budgeting",
    }
    if encoded_length > MAX_BGE_TOKENS:
        raise AssertionError("BGE document input exceeds the 512-token limit.")
    return content, report


def eligible_indices(
    corpus: engine.MatchingCorpus,
    profile: engine.CandidateProfile,
    candidate_indices: Sequence[int],
) -> list[int]:
    """Single hard-filter entry point shared by embedding scoring."""
    included, _ = engine._eligible_jobs(corpus, profile, candidate_indices)
    return included
