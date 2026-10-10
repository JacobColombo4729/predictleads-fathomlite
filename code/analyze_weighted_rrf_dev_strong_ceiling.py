"""Offline diagnosis of the saved weighted-RRF development top-five grade ceiling."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import job_matching_engine as engine
import run_bge_transition_experiment as bge
import run_weighted_rrf_experiment as weighted

EXPERIMENT_DIR = weighted.EXPERIMENT_DIR
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "diagnostics/dev_top5_strong_ceiling_20261009"
SAMPLE_PATH = weighted.SAMPLE_PATH
SPLIT = "development"
METHODS = weighted.METHODS
_MODAL_REQUIRED = re.compile(r"\b(?:required|must(?:\s+have|\s+be)?|minimum|mandatory)\b", re.IGNORECASE)
_MODAL_PREFERRED = re.compile(r"\b(?:preferred|preferably|desired|nice to have|a plus|plus)\b", re.IGNORECASE)
_MODAL_NEGATED = re.compile(r"\b(?:not required|not mandatory|not necessary|not preferred)\b", re.IGNORECASE)
_QUAL_WORDS = re.compile(r"\b(?:license|licence|certification|certificate|degree|diploma|years? of experience|experience with|experience in|knowledge of|proficiency|credential)\b", re.IGNORECASE)
_HEADING = re.compile(r"^(?:[-*•\d.)\s]*)?([^:]{2,90}):?\s*$")
_REQ_HEADING = re.compile(r"\b(?:requirements?|qualifications?|education and experience|minimum qualifications|preferred qualifications)\b", re.IGNORECASE)
_DUTY_HEADING = re.compile(r"\b(?:duties|responsibilities|essential functions|what you(?:'|’)ll do|key responsibilities)\b", re.IGNORECASE)
_STOP = {"about", "after", "also", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "candidate", "does", "for", "from", "have", "has", "into", "is", "job", "may", "more", "must", "not", "of", "on", "or", "our", "required", "preferred", "should", "than", "that", "the", "their", "this", "to", "with", "work", "years", "experience", "knowledge", "ability", "qualification", "qualifications", "role"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9+#./-]+", text)
            if len(token) >= 4 and token.casefold() not in _STOP}


def _source_sections(description: str) -> dict[str, list[dict[str, Any]]]:
    result = {"duties": [], "requirements": [], "other": []}
    state = "other"
    for index, raw in enumerate(description.splitlines(), 1):
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        cleaned = text.strip(" -*•#|:")
        heading = _HEADING.match(cleaned) if len(cleaned) <= 90 else None
        if heading:
            heading_text = heading.group(1).strip()
            if _REQ_HEADING.search(heading_text):
                state = "requirements"
                continue
            if _DUTY_HEADING.search(heading_text):
                state = "duties"
                continue
        bucket = state
        if bucket == "other":
            if _MODAL_REQUIRED.search(text) or _MODAL_PREFERRED.search(text) or _QUAL_WORDS.search(text):
                bucket = "requirements"
            elif re.search(r"\b(?:manage|perform|coordinate|maintain|prepare|review|inspect|record|monitor|support|assist|process|track|operate|conduct|create|receive|schedule|reconcile|stock|supervise|test|document)\b", text, re.I):
                bucket = "duties"
        result[bucket].append({"source_line": index, "text": text})
    return result


def _classify_requirements(lines: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result = {"explicit_required": [], "explicit_preferred": [], "modality_unclear": []}
    for item in lines:
        text = str(item["text"])
        if re.search(r"\bno\s+travel\s+required\b", text, re.IGNORECASE):
            continue
        has_qualification_term = _QUAL_WORDS.search(text) is not None
        if (_MODAL_NEGATED.search(text)
                or re.search(r"\b(?:may|might|could)\s+be\s+required\b", text, re.IGNORECASE)):
            result["modality_unclear"].append(text)
        elif not has_qualification_term and (_MODAL_REQUIRED.search(text) or _MODAL_PREFERRED.search(text)):
            result["modality_unclear"].append(text)
        elif _MODAL_REQUIRED.search(text) and _MODAL_PREFERRED.search(text):
            result["modality_unclear"].append(text)
        elif _MODAL_PREFERRED.search(text):
            result["explicit_preferred"].append(text)
        elif _MODAL_REQUIRED.search(text):
            result["explicit_required"].append(text)
        elif _QUAL_WORDS.search(text):
            result["modality_unclear"].append(text)
    return result


def _overlap_ratio(reference: str, candidate: str) -> float:
    a, b = _tokens(reference), _tokens(candidate)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _facets_for_case(dataset_row: Mapping[str, Any], profile: Mapping[str, Any],
                     cited_job: Sequence[str]) -> list[dict[str, Any]]:
    trace = dataset_row["bge_render_trace"]
    retained = set(trace.get("retained_evidence_ids", []))
    omitted = {item["evidence_id"] for item in trace.get("omitted_evidence", [])}
    profile_text = " ".join([
        *profile.get("work_history", []), *profile.get("accomplishments", []),
        *["{} {}".format(s.get("skill", ""), s.get("evidence", ""))
          for s in profile.get("skills_with_evidence", [])],
        *profile.get("desired_roles", []), profile.get("career_transition_goal", ""),
    ])
    output = []
    for item in dataset_row["matching_text"]["evidence"]:
        if item["category"] == "company_background":
            continue
        candidate_overlap = max((_overlap_ratio(item["source_text"], text)
                                 for text in [*profile.get("work_history", []), *profile.get("accomplishments", []),
                                              *[s.get("evidence", "") for s in profile.get("skills_with_evidence", [])]]),
                                default=0.0)
        evaluator_overlap = max((_overlap_ratio(item["source_text"], text) for text in cited_job), default=0.0)
        if candidate_overlap < 0.12 and evaluator_overlap < 0.12 and item["category"] not in {
            "required_qualifications", "preferred_qualifications"
        }:
            continue
        output.append({
            "evidence_id": item["evidence_id"],
            "category": item["category"],
            "modality": item["requirement_modality"],
            "source_line": item["source_line"],
            "exact_source_excerpt": item["source_text"],
            "treatment_render_status": "retained" if item["evidence_id"] in retained else
                                       "omitted_by_512_token_budget" if item["evidence_id"] in omitted else
                                       "not_rendered",
            "candidate_evidence_overlap": round(candidate_overlap, 3),
            "evaluator_citation_overlap": round(evaluator_overlap, 3),
        })
    output.sort(key=lambda item: (-max(item["candidate_evidence_overlap"], item["evaluator_citation_overlap"]),
                                  item["source_line"]))
    return output[:24]


def _explicit_conflict_audit(profile: Mapping[str, Any], judgment: Mapping[str, Any],
                             requirement_lines: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    disclosures = profile.get("qualification_disclosures", {})
    absences = [str(x) for x in disclosures.get("explicit_absence", [])]
    unknown = [str(x) for x in disclosures.get("unknown", [])]
    result = []
    for claim in judgment.get("explicit_constraint_conflicts", []):
        claim = str(claim)
        claim_tokens = _tokens(claim)
        absence_matches = [value for value in absences if len(_tokens(value) & claim_tokens) >= 2]
        unknown_matches = [value for value in unknown if len(_tokens(value) & claim_tokens) >= 2]
        required_matches = [line for line in requirement_lines["explicit_required"]
                            if len(_tokens(line) & claim_tokens) >= 2]
        preferred_matches = [line for line in requirement_lines["explicit_preferred"]
                             if len(_tokens(line) & claim_tokens) >= 2]
        if absence_matches and required_matches:
            classification = "explicit_absence_and_mandatory_source_text_align"
        elif unknown_matches:
            classification = "unknown_profile_evidence_not_explicit_absence"
        elif absence_matches and preferred_matches:
            classification = "explicit_absence_but_job_wording_only_preferred"
        elif absence_matches:
            classification = "explicit_absence_but_no_matching_mandatory_source_line"
        elif any(term in claim.casefold() for term in ("location", "located", "on-site", "onsite")):
            classification = "location_claim_check_profile_hard_constraint_and_job_location_separately"
        else:
            classification = "no_source_aligned_explicit_conflict_found_or_unresolved"
        result.append({
            "claim_text": claim,
            "classification": classification,
            "candidate_explicit_absence_excerpts": absence_matches,
            "candidate_unknown_excerpts": unknown_matches,
            "matching_explicit_required_job_excerpts": required_matches,
            "matching_preferred_job_excerpts": preferred_matches,
        })
    return result


def analyze() -> dict[str, Any]:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((EXPERIMENT_DIR / "development_profiles.json").read_text(encoding="utf-8"))
    profiles_by_id = {str(row["profile_id"]): row for row in profiles}
    rankings = list(csv.DictReader((EXPERIMENT_DIR / "development_rankings.csv").open(encoding="utf-8-sig", newline="")))
    pool = _read_jsonl(EXPERIMENT_DIR / "development_blinded_evaluation_pool.jsonl")
    judgments = _read_jsonl(EXPERIMENT_DIR / "development_evaluation/openai_evaluator/validated_judgments.jsonl")
    grade_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    if len(grade_by_pair) != len(pool):
        raise ValueError("Development pool/judgment set is incomplete; refusing strong-ceiling diagnosis.")
    job_ids = {str(row["job_id"]) for row in pool}
    dataset_by_id = {}
    with (EXPERIMENT_DIR / "matching_text_dataset.jsonl").open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["job_id"]) in job_ids:
                dataset_by_id[str(row["job_id"])] = row
    jobs, load_report = engine.load_prepared_jobs(SAMPLE_PATH)
    if len(jobs) != 30_000:
        raise ValueError("Existing prepared sample differs from the frozen corpus; no full audit or sampling is run.")
    job_by_id = {job.job_id: job for job in jobs}
    if set(dataset_by_id) != job_ids or not job_ids.issubset(job_by_id):
        raise ValueError("Saved development jobs could not be joined by stable source IDs.")
    model, _model_load_seconds = bge._load_model(BGE_DIR / "bge_model_files", bge.MAX_BGE_TOKENS)
    tokenizer = model.tokenizer
    cases = []
    placement_counts = {method: Counter() for method in METHODS}
    for profile_id, profile in profiles_by_id.items():
        for method in METHODS:
            top5 = sorted((row for row in rankings if row["profile_id"] == profile_id
                           and row["method"] == method and int(row["rank"]) <= 5),
                          key=lambda row: int(row["rank"]))
            for ranking in top5:
                job_id = str(ranking["job_id"])
                judgment = grade_by_pair[(profile_id, job_id)]
                grade = judgment.get("grade")
                placement_counts[method]["placements"] += 1
                placement_counts[method]["grade_0"] += int(grade == 0)
                placement_counts[method]["grade_1"] += int(grade == 1)
                placement_counts[method]["grade_2"] += int(grade == 2)
                placement_counts[method]["grade_3"] += int(grade == 3)
                if grade is None:
                    placement_counts[method]["unknown"] += 1
                profile_texts = [*profile["work_history"], *profile["accomplishments"],
                                  *["{}: {}".format(s["skill"], s["evidence"])
                                    for s in profile["skills_with_evidence"]],
                                  *profile["education_certifications"]]
                candidate_skills = profile["skills_with_evidence"]
                candidate_skill_matches = [{"skill": item["skill"], "evidence": item["evidence"],
                                            "profile_excerpt": next((text for text in profile_texts
                                                                     if item["evidence"] in text or text == item["evidence"]),
                                                                    item["evidence"])}
                                           for item in candidate_skills]
                job = job_by_id[job_id]
                description = str(job.source.get("description_text") or "")
                sections = _source_sections(description)
                modalities = _classify_requirements(sections["requirements"])
                job_citations = [str(x) for x in judgment.get("supporting_job_evidence", [])]
                facets = _facets_for_case(dataset_by_id[job_id], profile, job_citations)
                candidate_query_content, candidate_query_report = bge.build_bge_candidate_text(
                    _candidate_profile(profile), profile, tokenizer)
                candidate_query = bge.BGE_QUERY_PREFIX + candidate_query_content
                job_text, job_input_report = bge.build_bge_job_text(job.source, tokenizer)
                candidate_cites = [str(x) for x in judgment.get("supporting_candidate_evidence", [])]
                candidate_delivery = [{
                    "evaluator_excerpt": excerpt,
                    "exact_text_in_bge_query": _contains(candidate_query, excerpt),
                    "distinctive_token_overlap_with_query": round(_overlap(excerpt, candidate_query), 3),
                } for excerpt in candidate_cites]
                cited_job_facets = [facet for facet in facets if facet["evaluator_citation_overlap"] >= 0.12]
                cited_facets_retained = [facet for facet in cited_job_facets if facet["treatment_render_status"] == "retained"]
                cited_facets_omitted = [facet for facet in cited_job_facets if facet["treatment_render_status"] == "omitted_by_512_token_budget"]
                grade_reason = judgment.get("career_transition_relevance", {}).get("rationale", "")
                gaps = [str(x) for x in judgment.get("material_gaps", [])]
                candidate_text = " ".join(profile_texts)
                duty_examples = [str(x) for x in judgment.get("supporting_job_evidence", [])
                                 if _overlap(candidate_text, str(x)) >= 0.08]
                better_lower = []
                method_rows = sorted((row for row in rankings if row["profile_id"] == profile_id
                                      and row["method"] == method), key=lambda row: int(row["rank"]))
                if grade in {0, 1, 2}:
                    for lower in method_rows:
                        if int(lower["rank"]) <= int(ranking["rank"]):
                            continue
                        lower_judgment = grade_by_pair[(profile_id, str(lower["job_id"]))]
                        if lower_judgment.get("status") == "judged" and lower_judgment.get("grade") in {0, 1, 2, 3} and lower_judgment["grade"] > grade:
                            better_lower.append({"job_id": str(lower["job_id"]), "title": str(lower["title"]),
                                                 "rank": int(lower["rank"]), "grade": lower_judgment["grade"],
                                                 "evaluator_rationale": lower_judgment.get("career_transition_relevance", {}).get("rationale", "")})
                cause = _diagnose_case(profile, judgment, grade, modalities, cited_facets_retained,
                                       cited_facets_omitted, better_lower, duty_examples)
                cases.append({
                    "case_id": "{}__{}__r{}__{}".format(profile_id, method, ranking["rank"], job_id),
                    "candidate_id": profile_id, "job_id": job_id, "method": method,
                    "rank": int(ranking["rank"]), "automated_grade": grade,
                    "automated_grade_status": judgment.get("status"),
                    "career_transition_relevance_level": judgment.get("career_transition_relevance", {}).get("level"),
                    "career_transition_relevance_rationale": grade_reason,
                    "automated_material_gaps": gaps,
                    "automated_required_qualification_label": judgment.get("required_qualifications"),
                    "automated_explicit_constraint_conflicts": judgment.get("explicit_constraint_conflicts", []),
                    "candidate_profile_evidence": {
                        "work_history": profile["work_history"],
                        "accomplishments": profile["accomplishments"],
                        "documented_skills": candidate_skill_matches,
                        "education_certifications": profile["education_certifications"],
                        "explicit_absences": profile["qualification_disclosures"].get("explicit_absence", []),
                        "unknown_qualifications": profile["qualification_disclosures"].get("unknown", []),
                    },
                    "job_source": {"title": job.searchable_title, "stable_job_id": job.job_id,
                                   "job_duty_citations": judgment.get("supporting_job_evidence", []),
                                   "exact_duty_excerpts": sections["duties"][:12],
                                   "explicit_required_excerpts": modalities["explicit_required"][:12],
                                   "explicit_preferred_excerpts": modalities["explicit_preferred"][:12],
                                   "modality_unclear_qualification_excerpts": modalities["modality_unclear"][:12]},
                    "saved_bge_input_delivery": {
                        "candidate_query_tokens_including_specials": candidate_query_report["encoded_tokens_including_prefix_and_specials"],
                        "candidate_query_omitted_sections": candidate_query_report.get("omitted_sections", []),
                        "candidate_evaluator_citation_checks": candidate_delivery,
                        "job_document_tokens_including_specials": job_input_report["encoded_tokens_including_specials"],
                        "job_omitted_tokens_estimate": job_input_report["omitted_tokens_estimate"],
                        "job_omitted_sections": job_input_report["omitted_sections"],
                        "citation_aligned_facets_retained": cited_facets_retained,
                        "citation_aligned_facets_omitted": cited_facets_omitted,
                        "delivery_interpretation": "The saved model input is reconstructed from frozen builders; evaluator evidence is a review-time citation, not proof that it was in or omitted from the ranking input.",
                    },
                    "requirement_conflict_audit": _conflict_audit(profile, judgment, modalities),
                    "better_rated_jobs_lower_same_method_pool": better_lower,
                    "why_not_grade_3_assessment": cause,
                    "judgment_is_ground_truth": False,
                })

    _write_jsonl(DIAGNOSTIC_DIR / "development_top5_cases.jsonl", cases)
    flat = []
    for case in cases:
        flat.append({"case_id": case["case_id"], "profile_id": case["candidate_id"],
                     "job_id": case["job_id"], "method": case["method"], "rank": case["rank"],
                     "grade": case["automated_grade"],
                     "transition_level": case["career_transition_relevance_level"],
                     "candidate_query_tokens": case["saved_bge_input_delivery"]["candidate_query_tokens_including_specials"],
                     "job_tokens": case["saved_bge_input_delivery"]["job_document_tokens_including_specials"],
                     "job_omitted_tokens_estimate": case["saved_bge_input_delivery"]["job_omitted_tokens_estimate"],
                     "retained_cited_facets": len(case["saved_bge_input_delivery"]["citation_aligned_facets_retained"]),
                     "omitted_cited_facets": len(case["saved_bge_input_delivery"]["citation_aligned_facets_omitted"]),
                     "better_rated_lower_pool_results": len(case["better_rated_jobs_lower_same_method_pool"]),
                     "reason_categories": ";".join(case["why_not_grade_3_assessment"]["categories"])})
    _write_csv(DIAGNOSTIC_DIR / "development_top5_cases.csv", flat,
               ["case_id", "profile_id", "job_id", "method", "rank", "grade", "transition_level",
                "candidate_query_tokens", "job_tokens", "job_omitted_tokens_estimate",
                "retained_cited_facets", "omitted_cited_facets", "better_rated_lower_pool_results", "reason_categories"])

    profile_summaries = []
    for profile_id, profile in profiles_by_id.items():
        for method in METHODS:
            top = [case for case in cases if case["candidate_id"] == profile_id and case["method"] == method]
            grade_counts = Counter("unknown" if case["automated_grade"] is None else str(case["automated_grade"])
                                   for case in top)
            profile_summaries.append({"profile_id": profile_id, "method": method,
                                      "top5_placements": len(top), "grade_counts": dict(grade_counts),
                                      "unique_profile_job_pairs": len({case["job_id"] for case in top}),
                                      "top5_grade_3_count": grade_counts.get("3", 0),
                                      "low_vs_plausible_assessed": [
                                          {"rank": case["rank"], "job_id": case["job_id"], "grade": case["automated_grade"],
                                           "categories": case["why_not_grade_3_assessment"]["categories"]}
                                          for case in top]})
    unique_pairs = {(case["candidate_id"], case["job_id"]) for case in cases}
    pair_methods = defaultdict(list)
    for case in cases:
        pair_methods[(case["candidate_id"], case["job_id"])].append(case)
    method_summary = {}
    for method in METHODS:
        subset = [case for case in cases if case["method"] == method]
        counts = Counter(str(case["automated_grade"]) if case["automated_grade"] is not None else "unknown" for case in subset)
        method_summary[method] = {"top5_placements": len(subset), "grade_counts": dict(counts),
                                   "unique_profile_job_pairs": len({(case["candidate_id"], case["job_id"]) for case in subset}),
                                   "grade_3_placements": counts.get("3", 0)}
    common_pairs = [
        {"profile_id": profile_id, "job_id": job_id,
         "appearances": [{"method": c["method"], "rank": c["rank"], "grade": c["automated_grade"]}
                         for c in appearances]}
        for (profile_id, job_id), appearances in sorted(pair_methods.items()) if len(appearances) > 1
    ]
    summary = {
        "experiment_id": weighted.EXPERIMENT_ID,
        "analysis_scope": "development split only; held-out profile file and held-out artifacts were not read",
        "profiles": list(profiles_by_id),
        "methods": list(METHODS),
        "top5_placements_total": len(cases),
        "unique_profile_job_pairs_across_all_methods": len(unique_pairs),
        "validated_judgments_in_dev_pool": len(judgments),
        "numeric_judgments": sum(j.get("status") == "judged" and j.get("grade") in {0, 1, 2, 3} for j in judgments),
        "unknown_judgments": sum(j.get("status") == "unknown" and j.get("grade") is None for j in judgments),
        "development_paired_complete_p5_denominator": 2,
        "strong_grade_top5_count_by_method": {method: method_summary[method]["grade_3_placements"] for method in METHODS},
        "top5_grade_distribution_by_method_placements": method_summary,
        "by_profile_method": profile_summaries,
        "same_profile_job_repeated_across_methods": common_pairs,
        "causal_claim": False,
        "heldout_opened": False,
        "original_judgments_modified": False,
        "no_api_calls": True,
    }
    _write_json(DIAGNOSTIC_DIR / "development_strong_ceiling_summary.json", summary)
    _write_report(summary, cases)
    return summary


def _candidate_profile(profile: Mapping[str, Any]) -> engine.CandidateProfile:
    hard = profile["hard_constraints"]
    preferences = {"exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
                   "require_location_match": hard.get("require_location_match") is True,
                   "keywords": list(profile.get("soft_preferences", [])),
                   "work_arrangement": list(profile.get("work_arrangement_preferences", []))}
    experience = [*profile["work_history"], *profile["accomplishments"], *profile["education_certifications"]]
    skills = ["{}: {}".format(item["skill"], item["evidence"]) for item in profile["skills_with_evidence"]]
    return engine.CandidateProfile(profile_id=profile["profile_id"], experience=tuple(experience),
                                   evidenced_skills=tuple(skills), desired_roles=tuple(profile["desired_roles"]),
                                   location=profile.get("location"), preferences=preferences)


def _contains(haystack: str, needle: str) -> bool:
    return bool(_norm(needle)) and _norm(needle) in _norm(haystack)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9+#./-]+", text)
            if len(token) >= 4 and token.casefold() not in {"about", "after", "also", "and", "are", "as", "at", "be", "been", "but", "candidate", "experience", "for", "from", "have", "has", "into", "is", "job", "may", "more", "must", "not", "preferred", "required", "the", "their", "this", "with", "work", "year", "years"}}


def _requirements_and_duties(description: str) -> dict[str, list[str]]:
    duties: list[str] = []
    requirements: list[str] = []
    state = "other"
    req_heading = re.compile(r"^(?:qualifications?|requirements?|minimum qualifications?|education and experience|(?:the )?skills we are looking for)\s*:?$", re.I)
    preferred_heading = re.compile(r"^(?:preferred qualifications?|preferred|nice to have)\s*:?$", re.I)
    duty_heading = re.compile(r"^(?:duties|responsibilities|essential functions|key responsibilities|what you(?:'|’)ll do)\s*:?$", re.I)
    actions = re.compile(r"\b(?:manage|perform|coordinate|maintain|prepare|review|inspect|record|monitor|support|assist|process|track|operate|conduct|create|receive|schedule|reconcile|stock|supervise|test|document|lead|develop|ensure|verify|respond|deliver|analyze|analyse|investigate|resolve)\b", re.I)
    modal = re.compile(r"\b(?:required|must|minimum|mandatory|preferred|preferably|a plus|desired|license|licence|certification|certificate|degree|years? of experience)\b", re.I)
    for raw in description.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" -*•")
        if not line:
            continue
        if req_heading.match(line) or preferred_heading.match(line):
            state = "requirements"
            continue
        if duty_heading.match(line):
            state = "duties"
            continue
        if state == "requirements":
            requirements.append(line)
        elif state == "duties" or actions.search(line):
            duties.append(line)
        elif modal.search(line) and _QUAL_WORDS.search(line):
            requirements.append(line)
    return {"duties": duties, "qualification_or_modality_lines": requirements}


def _conflict_audit(profile: Mapping[str, Any], judgment: Mapping[str, Any],
                    requirement_lines: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    absent = [str(x) for x in profile.get("qualification_disclosures", {}).get("explicit_absence", [])]
    unknown = [str(x) for x in profile.get("qualification_disclosures", {}).get("unknown", [])]
    result = []
    for claim in judgment.get("explicit_constraint_conflicts", []):
        claim = str(claim)
        claim_terms = _tokens(claim)
        absent_match = [x for x in absent if len(_tokens(x) & claim_terms) >= 2]
        unknown_match = [x for x in unknown if len(_tokens(x) & claim_terms) >= 2]
        line_records = [line for line in requirement_lines["qualification_or_modality_lines"]
                if len(_tokens(line) & claim_terms) >= 2]
        modalities = _classify_requirements([{"text": line} for line in line_records])
        required_matches = modalities["explicit_required"]
        preferred_matches = modalities["explicit_preferred"]
        if absent_match and required_matches:
            status = "explicit_candidate_absence_and_required_source_text_align"
        elif unknown_match:
            status = "candidate_evidence_unknown_not_explicit_absence"
        elif absent_match and preferred_matches:
            status = "explicit_absence_but_job_wording_preferred"
        elif absent_match:
            status = "explicit_absence_but_no_matching_mandatory_job_wording"
        elif "location" in claim.casefold() or "located" in claim.casefold():
            status = "location_claim_vs_profile_constraint_requires_manual_review"
        else:
            status = "unresolved_no_direct_source_alignment"
        result.append({"claim": claim, "status": status,
                       "candidate_explicit_absence_matches": absent_match,
                       "candidate_unknown_matches": unknown_match,
                       "matching_required_job_excerpts": required_matches,
                       "matching_preferred_job_excerpts": preferred_matches,
                       "matching_unclear_modality_excerpts": modalities["modality_unclear"]})
    return result


def _diagnose_case(profile: Mapping[str, Any], judgment: Mapping[str, Any], grade: Any,
                   requirement_lines: Mapping[str, Sequence[str]], relevant_facets: Sequence[Mapping[str, Any]],
                   better_lower: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rationale = str(judgment.get("career_transition_relevance", {}).get("rationale", ""))
    gaps = [str(x) for x in judgment.get("material_gaps", [])]
    conflicts = _conflict_audit(profile, judgment, requirement_lines)
    categories: list[str] = []
    confidence: dict[str, str] = {}
    # A low/medium rating rationale or gaps are recorded as evaluator evidence; classify conservatively.
    rationale_fold = rationale.casefold()
    weak_markers = ("lacks direct", "limited", "does not", "however", "not demonstrate", "different industry", "no evidence", "insufficient")
    if any(marker in rationale_fold for marker in weak_markers) or gaps:
        categories.append("evaluator_reported_transfer_or_evidence_gaps")
        confidence[categories[-1]] = "medium that the evaluator relied on these stated gaps; not an independent explanation of grade calibration"
    if any(item["status"] == "explicit_candidate_absence_and_required_source_text_align" for item in conflicts):
        categories.append("explicit_mandatory_conflict_candidate")
        confidence[categories[-1]] = "medium; textual alignment checked, but conflict and grade remain automated claims"
    if any(item["status"] in {"candidate_evidence_unknown_not_explicit_absence",
                               "explicit_absence_but_job_wording_preferred",
                               "explicit_absence_but_no_matching_mandatory_job_wording",
                               "location_claim_vs_profile_constraint_requires_manual_review",
                               "unresolved_no_direct_source_alignment"} for item in conflicts):
        categories.append("possible_evaluator_conflict_or_modality_miscalibration")
        confidence[categories[-1]] = "low-to-medium; claim/source mismatch merits review but automated grade is not revised"
    if any(item["treatment_render_status"] == "omitted_by_512_token_budget" for item in relevant_facets):
        categories.append("possible_input_loss_in_rendered_job_text")
        confidence[categories[-1]] = "low; facets were selected by lexical alignment, not human attribution"
    if not categories:
        categories.append("unresolved_grade_ceiling")
        confidence[categories[-1]] = "low"
    grade_ceiling_note = ""
    if grade == 2:
        grade_ceiling_note = "Grade 2 is plausible; evaluator rationale/material gaps state why it did not award grade 3. Whether those gaps are truly material is unverified."
    elif grade == 1:
        grade_ceiling_note = "Evaluator rated below-plausible; rationale and gaps are preserved, but no independent human adjudication was performed."
    elif grade == 0:
        grade_ceiling_note = "Evaluator rated clear mismatch; inspect its exact rationale and conflict claims. This report does not validate the grade."
    elif grade == 3:
        grade_ceiling_note = "Strong rating; included for the complete top-five placement inventory."
    else:
        grade_ceiling_note = "Unknown grade preserved as null; this item cannot support a numeric quality conclusion."
    return {
        "categories": categories,
        "confidence_by_category": confidence,
        "evaluator_rationale": rationale,
        "evaluator_material_gaps": gaps,
        "evaluator_grade_ceiling_note": grade_ceiling_note,
        "better_rated_lower_ranked_pair_count": len(better_lower),
        "better_rated_lower_ranked_interpretation": "Observed only within this profile/method's saved pool; disagreement with automated grades, not proven ranking error.",
    }


def _render_report(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Why Development Top Fives Have No Strong Grades",
        "",
        "Offline diagnosis of the weighted-RRF development split only. The six held-out profiles were not opened. Rankings, pool, evaluator grades and source labels were not changed; no API call, embedding, or ranking rerun occurred.",
        "",
        "## Coverage and Counts",
        "",
        "Development shared pool: 39 pairs, 38 numeric judgments and one explicit unknown/null. Across three methods x three profiles x five ranks there are 45 placements and {} unique profile/job pairs. No top-five placement received grade 3.".format(summary["unique_profile_job_pairs_across_all_methods"]),
        "",
        "| Method | Top-five placements | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Unknown | Unique profile/job pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        data = summary["top5_grade_distribution_by_method_placements"][method]
        grades = data["grade_counts"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            method, data["top5_placements"], grades.get("0", 0), grades.get("1", 0),
            grades.get("2", 0), grades.get("3", 0), grades.get("unknown", 0),
            data["unique_profile_job_pairs"]))
    lines.extend(["", "## Profile-Level Findings", ""])
    for profile in summary["profiles"]:
        lines.extend(["### {}".format(profile), ""])
        for method in METHODS:
            item = next(x for x in summary["by_profile_method"]
                        if x["profile_id"] == profile and x["method"] == method)
            grade_list = ", ".join("{}:{}".format(case["rank"],
                                                     "unknown" if case["automated_grade"] is None else case["automated_grade"])
                                    for case in item["low_vs_plausible_assessed"])
            lines.append("- `{}` top-five grades by rank: {}. Grade-2/1 rationales most often cite {}.".format(
                method, grade_list, _profile_reason_summary(cases, profile, method)))
        lines.append("")
    lines.extend([
        "## Diagnosis",
        "",
        "Across the 45 placements, the evaluator assigned plausible grade 2 in {}, below-plausible grade 1 in {}, mismatch grade 0 in {}, and one unknown; grade 3 is absent. The profile-local rationales most often describe credible transferable process/recordkeeping, but not the target domain's direct work, named systems, regulatory exposure, or seniority. Those are evaluator-stated gaps, not independently verified causes.".format(
            summary["all_method_grade_counts"].get("2", 0), summary["all_method_grade_counts"].get("1", 0),
            summary["all_method_grade_counts"].get("0", 0)),
        "",
        "Explicit mandatory conflict is narrower than missing evidence: the brewery profile explicitly says no drinking-water operator license, while wastewater process and municipal control-system experience are unknown. The source descriptions vary between required licenses, required experience, preferred systems, and no explicit qualification wording. Per-claim source checks are recorded in the case JSON. Do not treat unknowns or preferences as mandatory contradictions.",
        "",
        "Input-loss checks use the actual BGE query/document builder and exact top-five profile/job source records. Job facets potentially aligned to the evaluator's citations are marked retained or token-omitted, but this lexical evidence alignment cannot show that an omission caused a grade. Candidate query token counts/omission fields are preserved per case.",
        "",
        "A stronger candidate/job duty fit may be plausible in several grade-2 cases, but the same data does not cleanly distinguish ranker ordering, profile detail/target breadth, or evaluator conservatism. The automated reviewer judged only these exact saved pairs and is not independent human truth. Preserve this ambiguity.",
        "",
        "## Higher-Rated Lower-Ranked Pool Items",
        "",
        "A better-rated item below a top-five placement is counted only within the same profile and same method's saved development pool. It is evidence of disagreement with automated grades, not a proven ranker error. See per-case lists in `development_top5_cases.jsonl`; no cross-profile comparisons are used.",
        "",
        "## Decision",
        "",
        "Do not launch another ranking-weight/model change from this evidence. If a next step is warranted, first run a small, blinded human adjudication of the saved development union, stratified across grade 1/2, explicit-versus-unknown qualification claims, and strong duty-transfer cases, with duplicate profile/job pairs shown once and method/rank hidden. Hypothesis: part of the grade-2/no-grade-3 pattern reflects evaluator conservatism or inconsistent mandatory-conflict interpretation rather than retrieval alone. Success criterion: independent adjudicators reach a documented consensus/reliability threshold on grade 2 versus 3 and mandatory-versus-unknown conflicts; if they do not, keep the diagnosis unresolved and revise the rubric/evidence protocol before another ranker comparison. This is a validation step, not a model change.",
        "",
        "No corpus-wide recall claim is possible: the dev pool is only the union of these three methods' top tens. Unjudged corpus records, even if similar by keywords, remain unjudged. The six fresh held-out profiles were not opened.",
        "",
        "Structured case rows: `development_top5_cases.csv` and `development_top5_cases.jsonl`. Summary counts: `development_strong_ceiling_summary.json`.",
    ])
    (DIAGNOSTIC_DIR / "development_strong_ceiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_reason_summary(cases: Sequence[Mapping[str, Any]], profile_id: str, method: str) -> str:
    reasons = []
    for case in cases:
        if case["candidate_id"] != profile_id or case["method"] != method or case["automated_grade"] not in {0, 1, 2}:
            continue
        for gap in case["why_not_grade_3_assessment"]["evaluator_material_gaps"]:
            if gap not in reasons:
                reasons.append(gap)
    if not reasons:
        return "an unknown or unsubstantiated evidence gap"
    return "; ".join(reasons[:3])


def main() -> dict[str, Any]:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((EXPERIMENT_DIR / "development_profiles.json").read_text(encoding="utf-8"))
    profile_by_id = {str(row["profile_id"]): row for row in profiles}
    rankings = list(csv.DictReader((EXPERIMENT_DIR / "development_rankings.csv").open(encoding="utf-8-sig", newline="")))
    pool = _read_jsonl(EXPERIMENT_DIR / "development_blinded_evaluation_pool.jsonl")
    judgments = _read_jsonl(EXPERIMENT_DIR / "development_evaluation/openai_evaluator/validated_judgments.jsonl")
    judgment_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    if len(judgment_by_pair) != len(pool):
        raise ValueError("Development judgments do not fully cover the saved dev pool; preserve nulls and stop.")
    jobs, load_report = engine.load_prepared_jobs(SAMPLE_PATH)
    jobs_by_id = {job.job_id: job for job in jobs}
    pool_pairs = {(str(row["profile_id"]), str(row["job_id"])) for row in pool}
    pool_job_ids = {job_id for _, job_id in pool_pairs}
    dataset_by_id = {}
    with (PROJECT_ROOT / "data/matching/matching_text_faceted_v1_20261008T220000Z/matching_text_dataset.jsonl").open(
        "r", encoding="utf-8", newline="") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["job_id"] in pool_job_ids:
                dataset_by_id[row["job_id"]] = row
    bge_model, _model_load_seconds = bge._load_model(weighted.BGE_DIR / "bge_model_files", bge.MAX_BGE_TOKENS)
    tokenizer = bge_model.tokenizer

    cases = []
    grade_counts_all = Counter()
    per_method_pairs: dict[str, set[tuple[str, str]]] = {method: set() for method in METHODS}
    appearances: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for profile_id, profile in profile_by_id.items():
        candidate = _candidate_profile(profile)
        query, query_report = bge.build_bge_candidate_text(candidate, profile, tokenizer)
        query = bge.BGE_QUERY_PREFIX + query
        for method in METHODS:
            method_rows = sorted((row for row in rankings if row["profile_id"] == profile_id and row["method"] == method
                                  and int(row["rank"]) <= 5), key=lambda row: int(row["rank"]))
            for rank_row in method_rows:
                job_id = str(rank_row["job_id"])
                judgment = judgment_by_pair[(profile_id, job_id)]
                grade = judgment.get("grade")
                if judgment.get("status") == "unknown":
                    grade_counts_all["unknown"] += 1
                elif grade in {0, 1, 2, 3}:
                    grade_counts_all[str(grade)] += 1
                pair = (profile_id, job_id)
                per_method_pairs[method].add(pair)
                appearances[pair].append({"method": method, "rank": int(rank_row["rank"]), "grade": grade})
                job = jobs_by_id[job_id]
                description = str(job.source.get("description_text") or "")
                source_sections = _requirements_and_duties(description)
                modality = _classify_requirements([{"text": line}
                                                   for line in source_sections["qualification_or_modality_lines"]])
                source_sections.update(modality)
                cited_job = [str(x) for x in judgment.get("supporting_job_evidence", [])]
                cited_candidate = [str(x) for x in judgment.get("supporting_candidate_evidence", [])]
                relevant_facets = _facets_for_case(dataset_by_id[job_id], profile, cited_job)
                retained_facets = [row for row in relevant_facets if row["treatment_render_status"] == "retained"]
                omitted_facets = [row for row in relevant_facets if row["treatment_render_status"] == "omitted_by_512_token_budget"]
                control_text, control_report = bge.build_bge_job_text(job.source, tokenizer)
                better_lower = []
                for lower in sorted((x for x in rankings if x["profile_id"] == profile_id and x["method"] == method
                                     and int(x["rank"]) > int(rank_row["rank"])), key=lambda x: int(x["rank"])):
                    lower_judgment = judgment_by_pair[(profile_id, str(lower["job_id"]))]
                    if (grade in {0, 1, 2} and lower_judgment.get("status") == "judged"
                            and lower_judgment.get("grade") in {0, 1, 2, 3}
                            and lower_judgment["grade"] > grade):
                        better_lower.append({"job_id": lower["job_id"], "title": lower["title"],
                                             "rank": int(lower["rank"]), "grade": lower_judgment["grade"],
                                             "evaluator_rationale": lower_judgment.get("career_transition_relevance", {}).get("rationale", "")})
                conflict_audit = _conflict_audit(profile, judgment, source_sections)
                cause = _diagnose_case(profile, judgment, grade, source_sections, relevant_facets,
                                       better_lower)
                case = {
                    "case_id": "{}__{}__r{}__{}".format(profile_id, method, rank_row["rank"], job_id),
                    "profile_id": profile_id, "job_id": job_id, "method": method,
                    "rank": int(rank_row["rank"]), "saved_automated_grade": grade,
                    "saved_judgment_status": judgment.get("status"),
                    "title": str(rank_row["title"]),
                    "evaluator_rationale": judgment.get("career_transition_relevance", {}).get("rationale", ""),
                    "evaluator_material_gaps": judgment.get("material_gaps", []),
                    "candidate_evidence": {
                        "work_history": profile["work_history"], "accomplishments": profile["accomplishments"],
                        "skills_with_evidence": profile["skills_with_evidence"],
                        "education_certifications": profile["education_certifications"],
                        "explicit_absences": profile["qualification_disclosures"].get("explicit_absence", []),
                        "unknown_qualifications": profile["qualification_disclosures"].get("unknown", []),
                        "evaluator_cited_excerpts": cited_candidate,
                    },
                    "job_source_evidence": {
                        "exact_evaluator_cited_excerpts": cited_job,
                        "source_heading_or_responsibility_lines": source_sections["duties"][:14],
                        "explicit_required_source_lines": source_sections["explicit_required"][:14],
                        "explicit_preferred_source_lines": source_sections["explicit_preferred"][:14],
                        "unclear_modality_source_lines": source_sections["modality_unclear"][:14],
                    },
                    "input_delivery": {
                        "candidate_query_tokens": query_report.get("encoded_tokens_including_prefix_and_specials"),
                        "candidate_query_omitted_sections": query_report.get("omitted_sections", []),
                        "candidate_cited_exact_reached": [_contains(query, excerpt) for excerpt in cited_candidate],
                        "job_control_tokens": control_report.get("encoded_tokens_including_specials"),
                        "job_control_omitted_tokens_estimate": control_report.get("omitted_tokens_estimate"),
                        "job_control_omitted_sections": control_report.get("omitted_sections", []),
                        "citation_aligned_treatment_facets_retained": retained_facets,
                        "citation_aligned_treatment_facets_omitted": omitted_facets,
                        "input_assessment_caveat": "Evaluator citations describe review-time support, not a guaranteed trace of what the ranker attended to; facet alignment is token-overlap based.",
                    },
                    "requirement_conflict_audit": conflict_audit,
                    "better_rated_job_lower_in_same_method_pool": better_lower,
                    "grade_ceiling_analysis": cause,
                    "grade_unchanged": True,
                }
                cases.append(case)

    # Deduplicate profile/job identities separately from repeated method/rank appearances.
    unique_pairs = {(case["profile_id"], case["job_id"]) for case in cases}
    method_stats = {}
    for method in METHODS:
        subset = [case for case in cases if case["method"] == method]
        counts = Counter("unknown" if case["saved_automated_grade"] is None else str(case["saved_automated_grade"])
                         for case in subset)
        method_stats[method] = {"top5_placements": len(subset), "grade_distribution": dict(counts),
                                "unique_profile_job_pairs": len({(case["profile_id"], case["job_id"]) for case in subset}),
                                "strong_grade_3_count": counts.get("3", 0),
                                "better_grade_lower_placement_count": sum(bool(case["better_rated_job_lower_in_same_method_pool"]) for case in subset)}
    profile_stats = []
    for profile_id in profile_by_id:
        for method in METHODS:
            subset = [case for case in cases if case["profile_id"] == profile_id and case["method"] == method]
            counts = Counter("unknown" if case["saved_automated_grade"] is None else str(case["saved_automated_grade"])
                             for case in subset)
            profile_stats.append({"profile_id": profile_id, "method": method,
                                  "placements": len(subset), "grade_distribution": dict(counts),
                                  "grade_3_count": counts.get("3", 0),
                                  "better_grade_lower_placements": sum(bool(x["better_rated_job_lower_in_same_method_pool"])
                                                                        for x in subset)})
    repeated_pairs = [{"profile_id": pid, "job_id": jid, "appearances": appearances[(pid, jid)]}
                      for (pid, jid) in sorted(unique_pairs) if len(appearances[(pid, jid)]) > 1]
    summary = {
        "experiment_id": weighted.EXPERIMENT_ID,
        "analysis_scope": "three development profiles only; six held-out profiles/artifacts not opened",
        "profiles": list(profile_by_id), "methods": list(METHODS),
        "top5_placements": len(cases), "unique_profile_job_pairs": len(unique_pairs),
        "repeated_profile_job_pairs_across_methods": repeated_pairs,
        "dev_pool_pairs": len(pool),
        "numeric_judgments": sum(row.get("status") == "judged" and row.get("grade") in {0, 1, 2, 3} for row in judgments),
        "unknown_judgments": sum(row.get("status") == "unknown" and row.get("grade") is None for row in judgments),
        "paired_complete_p5_profiles": 2,
        "strong_top5_appearances_by_method": {method: method_stats[method]["strong_grade_3_count"] for method in METHODS},
        "grade_distribution_across_all_method_placements": dict(grade_counts_all),
        "method_summary": method_stats,
        "profile_method_summary": profile_stats,
        "most_supported_explanation": "The saved evaluator commonly calls transfer plausible/moderate while listing target-domain, system, licensure or supervisory gaps. The evidence does not distinguish true job/profile gaps from evaluator conservatism; one profile's unknown judgment also withholds metrics.",
        "ground_truth_warning": "All grades and rationale fields are saved automated judgments; they are not independent human ground truth.",
        "heldout_opened": False,
        "rankings_or_judgments_modified": False,
        "paid_calls": False,
    }
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DIAGNOSTIC_DIR / "development_top5_cases.jsonl", cases)
    summary_rows = []
    for case in cases:
        summary_rows.append({"case_id": case["case_id"], "profile_id": case["profile_id"],
                             "job_id": case["job_id"], "method": case["method"], "rank": case["rank"],
                             "saved_automated_grade": case["saved_automated_grade"],
                             "transition_level": next((judgment.get("career_transition_relevance", {}).get("level")
                                                        for judgment in judgments if judgment["profile_id"] == case["profile_id"]
                                                        and judgment["job_id"] == case["job_id"]), None),
                             "explicit_required_excerpt_count": len(case["job_source_evidence"]["explicit_required_source_lines"]),
                             "explicit_preferred_excerpt_count": len(case["job_source_evidence"]["explicit_preferred_source_lines"]),
                             "candidate_explicit_absence_count": len(case["candidate_evidence"]["explicit_absences"]),
                             "candidate_unknown_qualification_count": len(case["candidate_evidence"]["unknown_qualifications"]),
                             "better_grade_lower_same_pool_count": len(case["better_rated_job_lower_in_same_method_pool"]),
                             "reason_categories": ";".join(case["grade_ceiling_analysis"]["categories"])})
    _write_csv(DIAGNOSTIC_DIR / "development_top5_cases.csv", summary_rows,
               ["case_id", "profile_id", "job_id", "method", "rank", "saved_automated_grade", "transition_level",
                "explicit_required_excerpt_count", "explicit_preferred_excerpt_count",
                "candidate_explicit_absence_count", "candidate_unknown_qualification_count",
                "better_grade_lower_same_pool_count", "reason_categories"])
    _write_json(DIAGNOSTIC_DIR / "development_strong_ceiling_summary.json", summary)
    _write_report(summary, cases)
    return summary


def _write_report(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Weighted-RRF Development Top-Five Grade-Ceiling Diagnosis",
        "",
        "Offline analysis of the three development profiles only. Held-out profile files and rankings were not opened. Original development grades, rankings and judgments are unchanged; no API call, embedding, or retrieval rerun occurred.",
        "",
        "## Coverage and Counting",
        "",
        "Development pool: {} profile/job pairs, {} numeric grades and {} explicit unknown/null. Top-five placements: {} across {} unique profile/job pairs. No method has a grade-3 top-five placement.".format(
            summary["dev_pool_pairs"], summary["numeric_judgments"], summary["unknown_judgments"],
            summary["top5_placements"], summary["unique_profile_job_pairs"]),
        "",
        "| Method | Top-five appearances | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Unknown | Unique profile/job pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["method_summary"][method]
        dist = item["grade_distribution"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            method, item["top5_placements"], dist.get("0", 0), dist.get("1", 0), dist.get("2", 0),
            dist.get("3", 0), dist.get("unknown", 0), item["unique_profile_job_pairs"]))
    lines.extend(["", "## Profile Diagnosis", ""])
    for profile_id in summary["profiles"]:
        lines.extend(["### {}".format(profile_id), ""])
        for method in METHODS:
            stats = next(row for row in summary["profile_method_summary"]
                         if row["profile_id"] == profile_id and row["method"] == method)
            pair_cases = [case for case in cases if case["profile_id"] == profile_id and case["method"] == method]
            compact = ", ".join("r{} {}:{}".format(case["rank"], case["job_id"][:8],
                                                   "unknown" if case["saved_automated_grade"] is None else case["saved_automated_grade"])
                                for case in pair_cases)
            lines.append("- `{}`: grades by rank {}. Better-rated lower-pool comparisons in {} of 5 placements.".format(
                method, compact, stats["better_grade_lower_placements"]))
        lines.append("")
    lines.extend([
        "## What the Saved Evidence Supports",
        "",
        "Across these three synthetic dev profiles, the evaluator's grade-2 rationales generally describe some transferable process, traceability, exception handling or coordination, then cite missing direct domain tasks, systems, credentials, scale or seniority as the reason fit is not strong. The hotel banquet profile has relevant event-resource handoffs but several jobs concern different purchasing/warehouse tasks; the brewery profile has process checks but the target water roles vary in operator licensing and municipal process work; the museum profile has excellent records/inventory transfer for some jobs but one semantic title collision (`Collections Associate`) is accounts-receivable collection, a clear duty mismatch.",
        "",
        "This pattern is consistent with material task/qualification gaps, but also with evaluator conservatism: grades 2 and 3 are distinguished partly by strength/completeness of duty evidence, not just whether keywords overlap. Exact profile/job citations, original description lines, required/preferred/unclear excerpts, saved query/document input summaries, and claim audits are retained in the JSONL case table.",
        "",
        "Missing candidate details remain unknown unless a profile explicitly says absent. For example, the brewery profile explicitly lacks a drinking-water operator license but marks wastewater process and municipal control-system experience unknown. The top-five report preserves that distinction; source requirement wording is transcribed as required/preferred/unclear using visible wording, not inferred from generic headings.",
        "",
        "A grade-2/1 job appearing below a higher grade within the same method/profile pool is only a disagreement with the same automated labels. It does not prove the ranker is wrong. The museum profile's single unknown also withholds all paired metrics for that profile.",
        "",
        "## Recommended Next Step",
        "",
        "Do not change weights or introduce another ranker based on this dev slice. The evidence cannot reliably separate source ranking, profile specificity/requirements, and evaluator calibration. The best next experiment is a bounded blinded rubric audit on the existing 39-pair dev pool, with particular attention to grade-2 versus grade-3 thresholds, explicit mandatory versus preferred/unknown qualifications, and source-supported duty transfer. Deduplicate repeated profile/job pairs and hide method/rank. Hypothesis: at least some grade-2/no-grade-3 top-five pattern reflects evaluator conservatism or inconsistent gap/conflict interpretation. Predeclare a minimum of two independent human raters and an ordinal agreement measure (weighted kappa target >=0.60); if the threshold is not met, report rubric instability and do not tune weights. This tests judgment reliability, not model quality.",
        "",
        "The top-ten dev union cannot establish corpus-wide recall. No held-out results or prior-experiment metrics are combined here.",
        "",
        "Case files: `development_top5_cases.csv`, `development_top5_cases.jsonl`, `development_strong_ceiling_summary.json`.",
    ])
    (DIAGNOSTIC_DIR / "development_strong_ceiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()