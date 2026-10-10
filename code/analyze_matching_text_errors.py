"""Create a read-only case analysis from the saved faceted-text experiment."""

from __future__ import annotations

import csv
import hashlib
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

EXPERIMENT_ID = "matching_text_faceted_v1_20261008T220000Z"
EXPERIMENT_DIR = PROJECT_ROOT / "data/matching" / EXPERIMENT_ID
DIAGNOSTIC_DIR = EXPERIMENT_DIR / "diagnostics/saved_error_analysis_20261008"
SAMPLE_PATH = PROJECT_ROOT / "data/discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl"
CONTROL_METHOD = "tfidf_bge_rrf_control"
TREATMENT_METHOD = "tfidf_bge_rrf_faceted_v1"
METHODS = (CONTROL_METHOD, TREATMENT_METHOD)

CORPUS_SEARCH_CUES = {
    "heldout_dental_cadcam_to_device_quality": (
        "quality technician", "quality inspector", "manufacturing quality", "production quality",
        "medical device quality", "quality control technician", "inspection technician",
        "device manufacturing",
    ),
    "heldout_shelter_intake_to_emergency_supply": (
        "supply coordinator", "inventory coordinator", "inventory clerk", "receiving clerk",
        "logistics assistant", "emergency supply", "materials coordinator", "donation coordinator",
    ),
    "heldout_facilities_to_energy_audit_support": (
        "energy auditor", "energy efficiency coordinator", "building efficiency", "retrofit coordinator",
        "facilities coordinator", "inspection scheduler", "building performance", "work order coordinator",
    ),
}
_WORD = re.compile(r"[a-z0-9]+(?:[.+#/-][a-z0-9+#/-]+)*", re.IGNORECASE)
_STOP_WORDS = {
    "about", "after", "also", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "candidate", "could", "does", "for", "from", "have", "has", "into", "is", "job",
    "may", "more", "must", "not", "our", "required", "preferred", "should", "that", "the",
    "their", "them", "then", "there", "these", "this", "those", "through", "to", "was", "were",
    "will", "with", "work", "years", "year", "experience", "knowledge", "ability", "lack", "lacks",
    "lacking", "candidate", "position", "role", "degree", "license", "credential", "certification",
    "standards", "requirement", "qualifications", "qualification",
}


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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _contains_excerpt(container: str, excerpt: str) -> bool:
    normalized = _normalize(excerpt)
    return bool(normalized) and normalized in _normalize(container)


def _line_excerpt(text: str, terms: Sequence[str], limit: int = 5) -> list[str]:
    lowered_terms = [term.casefold() for term in terms if term]
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 15:
            continue
        if any(term in line.casefold() for term in lowered_terms):
            key = line.casefold()
            if key not in seen:
                lines.append(line[:900])
                seen.add(key)
        if len(lines) >= limit:
            break
    return lines


def _profile_terms(profile: Mapping[str, Any]) -> list[str]:
    values = [*profile.get("desired_roles", []), profile.get("career_transition_goal", "")]
    values.extend(row.get("skill", "") for row in profile.get("skills_with_evidence", []))
    terms = []
    for value in values:
        for token in _WORD.findall(str(value).casefold()):
            if len(token) >= 5 and token not in terms:
                terms.append(token)
    return terms


def _distinctive_terms(text: str) -> set[str]:
    return {token.casefold() for token in _WORD.findall(text)
            if len(token) >= 4 and token.casefold() not in _STOP_WORDS}


def _align_citation_to_facets(excerpt: str, facets: Sequence[Mapping[str, Any]],
                              source_description: str, control_text: str, treatment_text: str,
                              treatment_trace: Mapping[str, Any]) -> dict[str, Any]:
    excerpt_terms = _distinctive_terms(excerpt)
    retained = set(treatment_trace.get("retained_evidence_ids", []))
    omitted = {item["evidence_id"] for item in treatment_trace.get("omitted_evidence", [])}
    matches = []
    for item in facets:
        facet_text = str(item.get("source_text", ""))
        facet_terms = _distinctive_terms(facet_text)
        if not facet_terms:
            continue
        intersection = facet_terms & excerpt_terms
        coverage = len(intersection) / len(facet_terms)
        if _contains_excerpt(excerpt, facet_text) or _contains_excerpt(facet_text, excerpt) or coverage >= 0.72:
            matches.append({
                "evidence_id": item["evidence_id"],
                "category": item["category"],
                "requirement_modality": item["requirement_modality"],
                "source_line": item["source_line"],
                "source_text": facet_text,
                "citation_distinctive_term_coverage": round(coverage, 3),
                "present_in_control_bge_document": _contains_excerpt(control_text, facet_text),
                "treatment_input_status": "retained" if item["evidence_id"] in retained else
                                          "omitted_by_token_budget" if item["evidence_id"] in omitted else
                                          "not_rendered",
            })
    matches.sort(key=lambda item: (-item["citation_distinctive_term_coverage"], item["source_line"]))
    return {
        "source_citation": excerpt,
        "facet_alignment_method": "distinctive source-term coverage >=0.72 or normalized containment; exploratory alignment only",
        "aligned_source_facets": matches[:12],
        "source_excerpt_verbatim_in_model_input": {
            "control": _contains_excerpt(control_text, excerpt),
            "treatment": _contains_excerpt(treatment_text, excerpt),
        },
        "source_excerpt_verbatim_in_original_description": _contains_excerpt(source_description, excerpt),
        "citation_scope": "whole_or_long_description" if len(excerpt) > 1200 else "excerpt",
        "alignment_status": "matched_to_source_facets" if matches else "no_facet_alignment_found",
        "warning": "A long citation can span many source facets; facet alignment is exploratory and is not causal evidence about relevance.",
    }


def _explicit_requirement_review(profile: Mapping[str, Any], judgment: Mapping[str, Any],
                                 facets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    disclosures = profile.get("qualification_disclosures", {})
    explicit_absence = [str(value) for value in disclosures.get("explicit_absence", [])]
    unknown = [str(value) for value in disclosures.get("unknown", [])]
    required = [item for item in facets if item.get("requirement_modality") == "required"]
    preferred = [item for item in facets if item.get("requirement_modality") == "preferred"]
    claims = [str(value) for value in judgment.get("explicit_constraint_conflicts", [])]
    assessments = []
    for claim in claims:
        claim_folded = claim.casefold()
        claim_terms = _distinctive_terms(claim)
        absent_match = next((item for item in explicit_absence
                             if _distinctive_terms(item)
                             and len(_distinctive_terms(item) & claim_terms)
                             >= min(2, len(_distinctive_terms(item)))), None)
        unknown_match = next((item for item in unknown
                              if _distinctive_terms(item)
                              and len(_distinctive_terms(item) & claim_terms)
                              >= min(2, len(_distinctive_terms(item)))), None)
        matched_required = [item for item in required
                            if _distinctive_terms(item["source_text"])
                            and len(_distinctive_terms(item["source_text"]) & claim_terms)
                            >= min(2, len(_distinctive_terms(item["source_text"])))
                            and len(_distinctive_terms(item["source_text"]) & claim_terms) >= 2]
        matched_preferred = [item for item in preferred
                             if _distinctive_terms(item["source_text"])
                             and len(_distinctive_terms(item["source_text"]) & claim_terms)
                             >= min(2, len(_distinctive_terms(item["source_text"])))
                             and len(_distinctive_terms(item["source_text"]) & claim_terms) >= 2]
        if absent_match and matched_required:
            status = "textually_supported_mandatory_conflict_candidate"
        elif absent_match:
            status = "candidate_explicit_absence_without_matching_mandatory_job_text"
        elif unknown_match:
            status = "profile_disclosure_is_unknown_not_explicit_absence"
        elif matched_preferred and not matched_required:
            status = "matching_job_text_is_preferred_not_mandatory"
        elif not matched_required:
            status = "location_or_other_constraint_not_classified_as_qualification" if any(
                term in claim_folded for term in ("located", "location", "on-site", "onsite", "city", "miles")
            ) else "no_matching_explicit_required_source_span_found"
        else:
            status = "unresolved_source_or_candidate_alignment"
        assessments.append({
            "evaluator_claim": claim,
            "status": status,
            "candidate_explicit_absence_match": absent_match,
            "candidate_unknown_disclosure_match": unknown_match,
            "matching_explicit_required_source_excerpts": [item["source_text"] for item in matched_required],
            "matching_preferred_source_excerpts": [item["source_text"] for item in matched_preferred],
        })
    return {
        "candidate_explicit_absences": explicit_absence,
        "candidate_qualifications_unknown": unknown,
        "explicit_required_source_excerpts": [item["source_text"] for item in required],
        "explicit_preferred_source_excerpts": [item["source_text"] for item in preferred],
        "evaluator_claim_assessments": assessments,
        "assessment_note": "These are source-text checks for explicitness and disclosure state, not independent qualification or legal determinations.",
    }


def _important_facets(dataset_row: Mapping[str, Any], query_terms: Sequence[str], max_items: int = 20) -> list[dict[str, Any]]:
    trace = dataset_row["bge_render_trace"]
    retained = set(trace["retained_evidence_ids"])
    omitted = {row["evidence_id"] for row in trace["omitted_evidence"]}
    selected = []
    priority = {"required_qualifications": 0, "preferred_qualifications": 1,
                "duties": 2, "skills_tools": 3, "working_conditions": 4,
                "unclassified": 5, "company_background": 6}
    for item in dataset_row["matching_text"]["evidence"]:
        if item["category"] == "company_background":
            continue
        text = item["source_text"].casefold()
        overlap = sum(term in text for term in query_terms if len(term) >= 5)
        if overlap or item["category"] in {"required_qualifications", "preferred_qualifications"}:
            selected.append({
                "evidence_id": item["evidence_id"],
                "category": item["category"],
                "requirement_modality": item["requirement_modality"],
                "source_line": item["source_line"],
                "source_text": item["source_text"],
                "treatment_input_status": "retained" if item["evidence_id"] in retained else
                                          "omitted_by_token_budget" if item["evidence_id"] in omitted else
                                          "not_rendered_or_not_eligible",
                "lexical_cue_overlap_count": overlap,
            })
    selected.sort(key=lambda item: (-item["lexical_cue_overlap_count"], priority[item["category"]],
                                    item["source_line"]))
    return selected[:max_items]


def _search_corpus(profile: Mapping[str, Any], pool_ids: set[str], jobs_by_id: Mapping[str, engine.JobRecord],
                   top_n: int = 15) -> tuple[int, list[dict[str, Any]]]:
    profile_id = str(profile["profile_id"])
    cues = CORPUS_SEARCH_CUES[profile_id]
    matches = []
    match_count = 0
    for job in jobs_by_id.values():
        if job.job_id in pool_ids or not job.usable_description:
            continue
        source = job.source
        description = str(source.get("description_text") or "")
        title = str(source.get("title_searchable") or source.get("title_normalized")
                    or source.get("title_raw") or "")
        title_hits = [cue for cue in cues if cue in title.casefold()]
        description_hits = [cue for cue in cues if cue in description.casefold()]
        if not title_hits and not description_hits:
            continue
        match_count += 1
        snippet_terms = [*title_hits, *description_hits[:3]]
        matches.append({
            "job_id": job.job_id,
            "title": title,
            "title_cues_matched": title_hits,
            "description_cues_matched": description_hits,
            "source_excerpt": _line_excerpt(description, snippet_terms, limit=3),
            "judgment_status_for_this_profile": "unjudged_not_in_saved_shared_pool",
            "grade": None,
            "interpretation": "Literal cue-search lead only; not ranked, evaluated, or asserted suitable.",
        })
    matches.sort(key=lambda item: (-len(item["title_cues_matched"]),
                                   -len(item["description_cues_matched"]),
                                   item["title"].casefold(), item["job_id"]))
    return match_count, matches[:top_n]


def analyze() -> dict[str, Any]:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    profiles = _read_json(EXPERIMENT_DIR / "heldout_profiles.json")
    profiles_by_id = {str(row["profile_id"]): row for row in profiles}
    rankings = []
    with (EXPERIMENT_DIR / "heldout_rankings.csv").open(encoding="utf-8-sig", newline="") as source:
        rankings = list(csv.DictReader(source))
    pool = _read_jsonl(EXPERIMENT_DIR / "blinded_evaluation_pool.jsonl")
    pool_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in pool}
    judgments = _read_jsonl(EXPERIMENT_DIR / "openai_evaluator/validated_judgments.jsonl")
    judgment_by_pair = {(str(row["profile_id"]), str(row["job_id"])): row for row in judgments}
    dataset_by_id: dict[str, dict[str, Any]] = {}
    relevant_job_ids = {job_id for _, job_id in pool_by_pair}
    with (EXPERIMENT_DIR / "matching_text_dataset.jsonl").open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["job_id"]) in relevant_job_ids:
                dataset_by_id[str(row["job_id"])] = row
    jobs, load_report = engine.load_prepared_jobs(SAMPLE_PATH)
    jobs_by_id = {job.job_id: job for job in jobs}
    if len(dataset_by_id) != len({job_id for _, job_id in pool_by_pair}):
        raise ValueError("Some blinded-pool jobs are missing from the prepared source/dataset.")
    if len(judgment_by_pair) != len(pool_by_pair):
        raise ValueError("Validated judgment and pool cardinalities differ; refusing partial error analysis.")

    bge_experiment_dir = PROJECT_ROOT / "data/matching/bge_transition_20261008T142818Z"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(bge_experiment_dir / "bge_model_files"),
                                               local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 1_000_000
    query_inputs = {str(row["profile_id"]): row for row in _read_jsonl(EXPERIMENT_DIR / "heldout_query_inputs.jsonl")}
    cases: list[dict[str, Any]] = []
    case_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    placement_totals: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    pair_sets: dict[str, set[tuple[str, str]]] = {method: set() for method in METHODS}
    for method in METHODS:
        for profile_id, profile in profiles_by_id.items():
            method_rows = sorted((row for row in rankings
                                  if row["profile_id"] == profile_id and row["method"] == method),
                                 key=lambda row: int(row["rank"]))
            pool_ids = {job_id for pair_profile, job_id in pool_by_pair if pair_profile == profile_id}
            query_text = str(query_inputs[profile_id]["bge_query_text"])
            query_terms = _profile_terms(profile)
            for rank_row in method_rows[:5]:
                job_id = str(rank_row["job_id"])
                pair = (profile_id, job_id)
                judgment = judgment_by_pair[pair]
                grade = judgment.get("grade")
                if judgment.get("status") != "judged" or grade not in {0, 1}:
                    continue
                placement_totals[method]["grade_{}".format(grade)] += 1
                placement_totals[method]["low_grade_placements"] += 1
                pair_sets[method].add(pair)
                source_job = jobs_by_id[job_id]
                original_description = str(source_job.source.get("description_text") or "")
                treatment = dataset_by_id[job_id]
                control_text, _ = bge.build_bge_job_text(source_job.source, tokenizer)
                treatment_text = str(treatment["bge_document_text"])
                job_facets = treatment["matching_text"]["evidence"]
                relevant_facets = _important_facets(treatment, query_terms)
                conflict_check = _explicit_requirement_review(profile, judgment, job_facets)
                candidate_refs = [str(value) for value in judgment.get("supporting_candidate_evidence", [])]
                job_refs = [str(value) for value in judgment.get("supporting_job_evidence", [])]
                job_evidence_delivery = [
                    _align_citation_to_facets(excerpt, job_facets, original_description,
                                              control_text, treatment_text,
                                              treatment["bge_render_trace"])
                    for excerpt in job_refs
                ]
                candidate_delivery = [{
                    "source_excerpt": excerpt,
                    "in_saved_candidate_bge_query": _contains_excerpt(query_text, excerpt),
                } for excerpt in candidate_refs]
                better_lower = []
                for lower in method_rows:
                    lower_rank = int(lower["rank"])
                    if lower_rank <= int(rank_row["rank"]):
                        continue
                    lower_pair = (profile_id, str(lower["job_id"]))
                    lower_judgment = judgment_by_pair[lower_pair]
                    if lower_judgment.get("status") != "judged" or lower_judgment.get("grade") not in {0, 1, 2, 3}:
                        continue
                    if int(lower_judgment["grade"]) <= int(grade):
                        continue
                    lower_job = jobs_by_id[str(lower["job_id"])]
                    lower_refs = [str(value) for value in lower_judgment.get("supporting_job_evidence", [])]
                    better_lower.append({
                        "job_id": str(lower["job_id"]),
                        "title": str(lower["title"]),
                        "method": method,
                        "rank": lower_rank,
                        "grade": int(lower_judgment["grade"]),
                        "evaluator_candidate_evidence": lower_judgment.get("supporting_candidate_evidence", []),
                        "evaluator_job_evidence": lower_refs[:5],
                        "source_title": lower_job.searchable_title,
                    })
                better_lower.sort(key=lambda item: (item["rank"], -item["grade"], item["job_id"]))
                missed = [facet for item in job_evidence_delivery
                          for facet in item["aligned_source_facets"]
                          if facet["treatment_input_status"] == "omitted_by_token_budget"]
                categories = []
                if better_lower:
                    categories.append({
                        "category": "within-pool ordering error relative to existing automated grades",
                        "support": "One or more higher-grade jobs occur below this low-grade placement in the same profile, method, and shared pool.",
                        "confidence": "high for the ordering observation; not a claim that the automated grades are ground truth",
                    })
                if missed:
                    categories.append({
                        "category": "possible input loss or extraction/rendering mismatch",
                        "support": "One or more source facets lexically aligned to evaluator-cited job text were omitted by the treatment token budget.",
                        "confidence": "low-to-moderate; automated citation scope can be broad and lexical alignment is not semantic evidence attribution",
                    })
                if any(item["status"] in {"matching_job_text_is_preferred_not_mandatory",
                                          "profile_disclosure_is_unknown_not_explicit_absence",
                                          "no_matching_explicit_required_source_span_found"}
                       for item in conflict_check["evaluator_claim_assessments"]):
                    categories.append({
                        "category": "questionable evaluator conflict claim",
                        "support": "At least one conflict claim is not supported by an explicit candidate absence paired with matching mandatory source text, or is associated with preferred/unknown evidence.",
                        "confidence": "moderate as an evidence-alignment discrepancy; the evaluator grade itself is not independently judged",
                    })
                if not better_lower:
                    categories.append({
                        "category": "possible pool coverage or evaluator disagreement",
                        "support": "No higher grade appears at a lower rank within this method's saved top-ten pool.",
                        "confidence": "low; this does not establish a corpus-wide retrieval miss or absence of suitable jobs",
                    })
                case = {
                    "case_id": "{}__{}__r{}__{}".format(profile_id, method, rank_row["rank"], job_id),
                    "candidate_id": profile_id,
                    "job_id": job_id,
                    "method": method,
                    "rank": int(rank_row["rank"]),
                    "automated_grade": int(grade),
                    "automated_grade_status": judgment.get("status"),
                    "automated_evaluator_required_qualification_label": judgment.get("required_qualifications"),
                    "automated_explicit_constraint_conflicts": judgment.get("explicit_constraint_conflicts", []),
                    "candidate_source_evidence": {
                        "supporting_excerpts_from_validated_judgment": candidate_refs,
                        "profile_work_history": profile.get("work_history", []),
                        "profile_accomplishments": profile.get("accomplishments", []),
                        "profile_demonstrated_skills": profile.get("skills_with_evidence", []),
                        "profile_education_certifications": profile.get("education_certifications", []),
                        "explicit_absences": profile.get("qualification_disclosures", {}).get("explicit_absence", []),
                        "unknown_qualifications": profile.get("qualification_disclosures", {}).get("unknown", []),
                    },
                    "job_source_evidence": {
                        "title": source_job.searchable_title,
                        "occupation": source_job.occupation,
                        "supporting_excerpts_from_validated_judgment": job_refs,
                        "source_description_excerpts_matching_profile_cues": _line_excerpt(
                            original_description, query_terms, limit=10),
                        "important_facets_and_treatment_delivery": relevant_facets,
                    },
                    "model_input_delivery": {
                        "candidate_excerpt_delivery": candidate_delivery,
                        "job_excerpt_delivery": job_evidence_delivery,
                        "saved_control_bge_document_text": control_text,
                        "saved_treatment_bge_document_text": treatment_text,
                        "saved_candidate_bge_query_text": query_text,
                        "control_cache_reused": True,
                        "treatment_cache_separate": True,
                    },
                    "explicit_requirement_assessment": conflict_check,
                    "better_rated_jobs_lower_in_same_method_pool": better_lower,
                    "likely_failure_categories": categories,
                    "judgment_caveat": "Grade is the existing automated evaluator label and is preserved unchanged; not ground truth.",
                }
                cases.append(case)
                case_lookup[(profile_id, method, job_id)] = case

    _write_jsonl(DIAGNOSTIC_DIR / "low_grade_top5_cases.jsonl", cases)
    flat_rows = []
    for case in cases:
        flat_rows.append({
            "case_id": case["case_id"],
            "candidate_id": case["candidate_id"],
            "job_id": case["job_id"],
            "method": case["method"],
            "rank": case["rank"],
            "automated_grade": case["automated_grade"],
            "higher_grade_lower_rank_count": len(case["better_rated_jobs_lower_in_same_method_pool"]),
            "better_lower_job_ids": ";".join(item["job_id"] for item in case["better_rated_jobs_lower_in_same_method_pool"]),
            "candidate_cited_excerpts_reached_query": sum(
                item["in_saved_candidate_bge_query"] for item in case["model_input_delivery"]["candidate_excerpt_delivery"]),
            "candidate_cited_excerpt_count": len(case["model_input_delivery"]["candidate_excerpt_delivery"]),
            "job_citations_aligned_to_omitted_facets": sum(
                facet["treatment_input_status"] == "omitted_by_token_budget"
                for citation in case["model_input_delivery"]["job_excerpt_delivery"]
                for facet in citation["aligned_source_facets"]),
            "job_citations_aligned_to_retained_facets": sum(
                facet["treatment_input_status"] == "retained"
                for citation in case["model_input_delivery"]["job_excerpt_delivery"]
                for facet in citation["aligned_source_facets"]),
            "job_cited_excerpt_count": len(case["model_input_delivery"]["job_excerpt_delivery"]),
            "failure_categories": ";".join(item["category"] for item in case["likely_failure_categories"]),
        })
    _write_csv(DIAGNOSTIC_DIR / "low_grade_top5_cases.csv", flat_rows,
               ["case_id", "candidate_id", "job_id", "method", "rank", "automated_grade",
                "higher_grade_lower_rank_count", "better_lower_job_ids", "candidate_cited_excerpts_reached_query",
                "candidate_cited_excerpt_count", "job_cited_excerpts_found_in_control",
                "job_cited_excerpts_found_in_treatment", "job_cited_excerpt_count", "failure_categories"])

    profile_method_summary = []
    for profile_id in profiles_by_id:
        for method in METHODS:
            top5 = sorted((row for row in rankings if row["profile_id"] == profile_id
                           and row["method"] == method and int(row["rank"]) <= 5),
                          key=lambda row: int(row["rank"]))
            failures = [row for row in top5 if judgment_by_pair[(profile_id, row["job_id"])].get("grade") in {0, 1}]
            unique_bad = {(profile_id, row["job_id"]) for row in failures}
            better_pairs = {(profile_id, lower["job_id"])
                            for bad in failures
                            for lower in case_lookup[(profile_id, method, bad["job_id"])]
                            ["better_rated_jobs_lower_in_same_method_pool"]}
            profile_method_summary.append({
                "profile_id": profile_id,
                "method": method,
                "top5_placements": len(top5),
                "grade_0_1_placements": len(failures),
                "unique_grade_0_1_profile_job_pairs": len(unique_bad),
                "low_grade_placements_with_higher_grade_lower_ranked_pool_jobs": sum(
                    bool(case_lookup[(profile_id, method, row["job_id"])]
                         ["better_rated_jobs_lower_in_same_method_pool"]) for row in failures),
                "distinct_higher_grade_lower_ranked_pairs": len(better_pairs),
                "grades_by_rank": [{"rank": int(row["rank"]), "job_id": row["job_id"],
                                    "grade": judgment_by_pair[(profile_id, row["job_id"])].get("grade")}
                                   for row in top5],
            })

    search_summaries = []
    search_case_ids = defaultdict(set)
    sample_jobs, sample_report = engine.load_prepared_jobs(SAMPLE_PATH)
    if len(sample_jobs) != 30_000:
        raise ValueError("Frozen prepared sample row count changed; local cue search stopped.")
    for profile_id, profile in profiles_by_id.items():
        profile_pool_ids = {job_id for p_id, job_id in pool_by_pair if p_id == profile_id}
        match_count, candidates = _search_corpus(profile, profile_pool_ids,
                                                  {job.job_id: job for job in sample_jobs})
        for candidate in candidates:
            search_case_ids[profile_id].add(candidate["job_id"])
        search_summaries.append({
            "profile_id": profile_id,
            "exact_cue_candidate_count_outside_saved_pool": match_count,
            "candidates": candidates,
            "search_cues": list(CORPUS_SEARCH_CUES[profile_id]),
            "eligibility_source": "existing 30,000 prepared records where description_usable is true and description_text is nonblank",
            "ranking_or_judgment_run": False,
            "interpretation": "Literal term-search leads only. Every result remains unjudged; neither a matching cue nor lack of a result establishes suitability or corpus-wide recall.",
        })
    _write_jsonl(DIAGNOSTIC_DIR / "unjudged_corpus_search_leads.jsonl", search_summaries)

    unique_case_pairs = {}
    for case in cases:
        unique_case_pairs.setdefault((case["candidate_id"], case["job_id"]), case)
    method_summary = {}
    for method in METHODS:
        placements = [case for case in cases if case["method"] == method]
        unique_pairs = {(case["candidate_id"], case["job_id"]) for case in placements}
        conflict_statuses = Counter(
            assessment["status"]
            for case in placements
            for assessment in case["explicit_requirement_assessment"]["evaluator_claim_assessments"]
        )
        unique_pairs_with_omitted_facets = {
            (case["candidate_id"], case["job_id"])
            for case in placements
            if any(facet["treatment_input_status"] == "omitted_by_token_budget"
                   for citation in case["model_input_delivery"]["job_excerpt_delivery"]
                   for facet in citation["aligned_source_facets"])
        }
        method_summary[method] = {
            "grade_0_1_top5_placements": len(placements),
            "unique_profile_job_pairs": len(unique_pairs),
            "grade_0_placements": sum(case["automated_grade"] == 0 for case in placements),
            "grade_1_placements": sum(case["automated_grade"] == 1 for case in placements),
            "placements_with_higher_grade_lower_in_same_method_pool": sum(
                bool(case["better_rated_jobs_lower_in_same_method_pool"]) for case in placements),
            "unique_pairs_with_higher_grade_lower_in_same_method_pool": len({
                (case["candidate_id"], case["job_id"]) for case in placements
                if case["better_rated_jobs_lower_in_same_method_pool"]
            }),
            "unique_pairs_with_cited_facets_omitted_in_treatment": len(unique_pairs_with_omitted_facets),
            "evaluator_conflict_claim_status_counts_by_placement": dict(conflict_statuses),
        }
    unique_summary = []
    for (profile_id, job_id), case in sorted(unique_case_pairs.items()):
        unique_summary.append({
            "profile_id": profile_id,
            "job_id": job_id,
            "title": jobs_by_id[job_id].searchable_title,
            "top5_placements": [{"method": c["method"], "rank": c["rank"], "grade": c["automated_grade"]}
                                for c in cases if c["candidate_id"] == profile_id and c["job_id"] == job_id],
            "method_count": len({c["method"] for c in cases if c["candidate_id"] == profile_id and c["job_id"] == job_id}),
        })
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "created_offline": True,
        "paid_calls_made": False,
        "original_validated_judgments_modified": False,
        "profiles": list(profiles_by_id),
        "methods": list(METHODS),
        "shared_pool_pairs": len(pool),
        "validated_numeric_judgments": len(judgment_by_pair),
        "top5_low_grade_method_summary": method_summary,
        "top5_low_grade_placement_count_all_methods": len(cases),
        "top5_low_grade_unique_profile_job_pair_count": len(unique_case_pairs),
        "unique_profile_job_pairs_with_higher_grade_lower_in_same_method_pool": len({
            (case["candidate_id"], case["job_id"]) for case in cases
            if case["better_rated_jobs_lower_in_same_method_pool"]
        }),
        "unique_profile_job_pairs_with_source_facets_omitted_in_treatment": len({
            (case["candidate_id"], case["job_id"]) for case in cases
            if any(facet["treatment_input_status"] == "omitted_by_token_budget"
                   for citation in case["model_input_delivery"]["job_excerpt_delivery"]
                   for facet in citation["aligned_source_facets"])
        }),
        "unique_profile_job_cases": unique_summary,
        "profile_method_details": profile_method_summary,
        "same_profile_pool_ordering_observation": "A lower-rank higher grade is a within-pool ordering discrepancy relative to the existing automated labels; it does not prove human relevance.",
        "mandatory_conflict_observation": "Each evaluator conflict claim is checked against explicit absence versus unknown profile disclosures and explicit required versus preferred source spans; lexical/source association is not a legal or semantic adjudication.",
        "corpus_scope_warning": "The shared top-ten pool does not establish corpus-wide recall. Literal term-search leads are unjudged, and failure to find a suitable listing does not prove none exists.",
        "local_search": search_summaries,
        "source_sample_hash": _file_hash(SAMPLE_PATH),
        "source_rows_loaded_for_local_cue_search": len(sample_jobs),
    }
    _write_json(DIAGNOSTIC_DIR / "error_analysis_summary.json", summary)
    markdown = _render_markdown(summary, cases)
    (DIAGNOSTIC_DIR / "error_analysis.md").write_text(markdown, encoding="utf-8")
    return summary


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_markdown(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Saved Matching Error Analysis",
        "",
        "Offline diagnosis of the saved faceted-text experiment. Original rankings, pool, grades, and judgments were not changed; no model/API call or embedding was run.",
        "",
        "## Scope and Counting",
        "",
        "Profiles were analyzed separately. Counts distinguish method/rank placements from unique profile/job pairs. Grades are existing automated evaluator outputs, not ground truth. Comparisons never combine these profiles with the earlier hybrid experiment.",
        "",
        "| Method | Grade 0/1 top-five placements | Unique profile/job pairs | Grade 0 | Grade 1 | Placements with a higher-grade lower-ranked pool job |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["top5_low_grade_method_summary"][method]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            method, item["grade_0_1_top5_placements"], item["unique_profile_job_pairs"],
            item["grade_0_placements"], item["grade_1_placements"],
            item["placements_with_higher_grade_lower_in_same_method_pool"]))
    lines.extend([
        "",
        "There are {} low-grade top-five placements across {} unique profile/job pairs. Same-pool, same-profile, same-method higher-grade lower results are direct ordering discrepancies relative to the automated labels only.".format(
            summary["top5_low_grade_placement_count_all_methods"],
            summary["top5_low_grade_unique_profile_job_pair_count"]),
        "",
        "## Profile Patterns",
        "",
    ])
    for profile_id in summary["profiles"]:
        lines.append("### {}".format(profile_id))
        lines.append("")
        for method in METHODS:
            detail = next(row for row in summary["profile_method_details"]
                          if row["profile_id"] == profile_id and row["method"] == method)
            low = [row for row in detail["grades_by_rank"] if row["grade"] in {0, 1}]
            lines.append("- `{}`: {}/5 low-grade top-five placements; {} placements have a higher-grade result lower in this method's saved pool. Top-five grades by rank: {}.".format(
                method, len(low), detail["low_grade_placements_with_higher_grade_lower_ranked_pool_jobs"],
                ", ".join("{}:{}".format(row["rank"], row["grade"]) for row in detail["grades_by_rank"])))
        lines.append("")
    lines.extend([
        "## Case Findings",
        "",
        "Every low-grade placement is listed separately below. Exact candidate/job citations and original excerpts, both saved BGE document inputs, relevant treatment facets, explicit-required/preferred checks, and any better-rated lower-ranked jobs are in `low_grade_top5_cases.jsonl`. The compact join table is `low_grade_top5_cases.csv`.",
        "",
        "| Profile | Method | Rank | Grade | Job title | Higher-rated lower-ranked jobs | Most supported category | Confidence |",
        "|---|---|---:|---:|---|---:|---|---|",
    ])
    for case in cases:
        top_category = case["likely_failure_categories"][0] if case["likely_failure_categories"] else {
            "category": "unresolved", "confidence": "low"}
        job_title = next((item.get("title") for item in summary["unique_profile_job_cases"]
                          if item["profile_id"] == case["candidate_id"] and item["job_id"] == case["job_id"]), case["job_id"])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            case["candidate_id"], case["method"], case["rank"], case["automated_grade"],
            job_title, len(case["better_rated_jobs_lower_in_same_method_pool"]),
            top_category["category"], top_category["confidence"]))
    lines.extend([
        "",
        "## Cross-Case Diagnosis",
        "",
        "- **Within-pool ranking errors:** Higher automated grades appear below {} of 16 low-grade placements and for {} unique profile/job cases. This is the strongest observable pattern because both alternatives have saved pair labels in the same profile-specific union; it remains relative to automated grades.".format(
            sum(bool(case["better_rated_jobs_lower_in_same_method_pool"]) for case in cases),
            summary["unique_profile_job_pairs_with_higher_grade_lower_in_same_method_pool"]),
        "- **Input/extraction loss:** Citation-aligned source facets omitted by the faceted token budget appear in {} unique low-grade profile/job cases. A whole-paragraph citation can span relevant and irrelevant facets; lexical alignment is approximate, so this is a lead for inspection, not evidence that omitted text caused a low grade. Candidate cited evidence is also checked against the saved BGE query.".format(
            summary["unique_profile_job_pairs_with_source_facets_omitted_in_treatment"]),
        "- **Explicit requirements versus unknowns:** Per-claim source alignment distinguishes explicit absences from unknown profile disclosures and required from preferred job wording. Conflict status counts are saved by method in the structured summary. These checks do not determine actual qualification or legal eligibility.",
        "- **Possible retrieval misses:** The offline sample scan below is a literal cue search outside each profile's judged pool. These jobs remain unjudged and are only leads for future review.",
        "- **Sample coverage:** The saved union contains only top-ten outputs from two methods for three profiles. It cannot estimate corpus-wide recall or establish that no suitable job exists.",
        "- **Questionable evaluator judgments:** Grades and explicit-conflict claims are model outputs. Contradictions between a claim and the cited mandatory/preferred source wording are recorded for audit, not adjudicated as truth.",
        "",
        "## Unjudged Local Search",
        "",
        "A literal cue scan over the existing eligible prepared sample found the following count of candidates outside each profile's saved shared pool. Candidate excerpts are preserved in `unjudged_corpus_search_leads.jsonl`; these were not ranked or judged.",
        "",
        "| Profile | Cue matches outside saved pool | Returned leads |",
        "|---|---:|---:|",
    ])
    for result in summary["local_search"]:
        lines.append("| {} | {} | {} |".format(result["profile_id"],
                                               result["exact_cue_candidate_count_outside_saved_pool"],
                                               len(result["candidates"])))
    lines.extend([
        "",
        "## Recommendation",
        "",
        "Run one controlled **within-shortlist RRF calibration experiment** next: retain the same two retrieval sources and frozen eligibility, but compare the current equal reciprocal-rank sum with a small predeclared dev-only BGE-vs-TF-IDF rank-weight grid, then freeze before fresh held-out profiles. Hypothesis: some low-grade placements result from source-rank imbalance, since higher-grade jobs already exist lower in the saved union. Success criteria: on at least six fresh paired profiles with complete top fives and shared pools, improve plausible/strong P@5 over equal-weight RRF without reducing strong-only P@5, and improve or maintain nDCG@10; report incomplete coverage as unassessable. This is a proposed diagnostic only; no setting was changed or tuned here.",
        "",
        "The separate experiment's aggregate metrics are in `../metrics_and_coverage.json`; the prior hybrid report is supporting context only and uses different profiles/pools, so its metrics are not combined here.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))