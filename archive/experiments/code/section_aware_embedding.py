"""Deterministic, tokenizer-budgeted text representation for embedding experiments."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import job_matching_engine as engine


REPRESENTATION_ID = "embedding_section_aware_v1"

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


def _token_count(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"])


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
        piece_tokens = _token_count(self.tokenizer, piece, add_special_tokens=False)
        if self.token_count + piece_tokens > self.max_length:
            return False
        self.pieces.append(piece)
        self.token_count += piece_tokens
        return True

    def text(self) -> str:
        return " ".join(self.pieces)


def section_aware_job_text(
    record: Mapping[str, Any], tokenizer: Any, max_length: int = 256
) -> str:
    """Keep identifying fields, then alternate complete requirement and duty sentences."""
    title, occupation, description = engine._job_search_fields(record)
    budget = _TokenBudget(tokenizer, max_length)
    header = "Job title: {}. Occupation: {}.".format(title, occupation)
    if not budget.append(header):
        return header
    buckets = _job_sections(description)
    requirement = buckets["requirement"]
    duty = buckets["duty"]
    other = buckets["other"]
    tagged = []
    for index in range(max(len(requirement), len(duty))):
        if index < len(requirement):
            tagged.append(("Requirement: ", requirement[index]))
        if index < len(duty):
            tagged.append(("Duty: ", duty[index]))
    tagged.extend(("Description: ", sentence) for sentence in other)
    for prefix, sentence in tagged:
        budget.append(prefix + sentence)
    return budget.text()


def section_aware_candidate_text(
    profile: engine.CandidateProfile,
    fixture: Mapping[str, Any],
    tokenizer: Any,
    max_length: int = 256,
) -> str:
    """Include candidate evidence by explicit section priority without repeat weighting."""
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
    budget = _TokenBudget(tokenizer, max_length)
    for label, values in sections:
        clean_values = [str(value).strip() for value in values if str(value).strip()]
        if not clean_values:
            continue
        for value in clean_values:
            piece = "{}: {}".format(label, value)
            budget.append(piece)
    return budget.text()