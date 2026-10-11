from __future__ import annotations

import csv
import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_SPEC = importlib.util.spec_from_file_location(
    "job_matching_engine", PROJECT_ROOT / "code" / "job_matching_engine.py"
)
if ENGINE_SPEC is None or ENGINE_SPEC.loader is None:
    raise ImportError("Could not load code/job_matching_engine.py")
job_matching_engine = importlib.util.module_from_spec(ENGINE_SPEC)
sys.modules[ENGINE_SPEC.name] = job_matching_engine
ENGINE_SPEC.loader.exec_module(job_matching_engine)

CandidateProfile = job_matching_engine.CandidateProfile
FilterDecision = job_matching_engine.FilterDecision
JobRecord = job_matching_engine.JobRecord
assess_substantive_description = job_matching_engine.assess_substantive_description
DESCRIPTION_EVIDENCE_RULE_VERSION = job_matching_engine.DESCRIPTION_EVIDENCE_RULE_VERSION
evaluate_reviewed_recommendations = job_matching_engine.evaluate_reviewed_recommendations
filter_job = job_matching_engine.filter_job
fit_tfidf = job_matching_engine.fit_tfidf
load_prepared_jobs = job_matching_engine.load_prepared_jobs
load_candidate_profiles = job_matching_engine.load_candidate_profiles
pooled_review_rows = job_matching_engine.pooled_review_rows
condition_job_indices = job_matching_engine.condition_job_indices
rank_description_coverage_on_corpus = job_matching_engine.rank_description_coverage_on_corpus
reference_pools_from_recommendations = job_matching_engine.reference_pools_from_recommendations
read_review_labels = job_matching_engine.read_review_labels
rank_keyword = job_matching_engine.rank_keyword
rank_tfidf = job_matching_engine.rank_tfidf
write_description_coverage_outputs = job_matching_engine.write_description_coverage_outputs
run_description_coverage_rankings = job_matching_engine.run_description_coverage_rankings
tokenize = job_matching_engine.tokenize


class JobMatchingEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_path = Path(self.temp_dir.name) / "prepared.jsonl"
        self.records = [
            {
                "job_id": "stable-technical",
                "job_type": "job_opening",
                "title_raw": "Technical Support Engineer",
                "title_searchable": "Technical Support Engineer",
                "description_text": "Troubleshoot C++ networking devices and customer tickets.",
                "description_usable": True,
                "onet_data": {"occupation_name": "Computer User Support Specialists"},
                "location_raw": "Chicago, Illinois, United States",
                "location_data_raw": [],
                "status_raw": None,
                "status_derived": "unknown",
                "company_name_included": "Example Components",
                "company_domain_included": "example.invalid",
                "url": "https://example.invalid/jobs/technical",
                "quality_flags": {"description": "present", "company_relationship": "matched"},
            },
            {
                "job_id": "stable-retail",
                "job_type": "job_opening",
                "title_raw": "Retail Sales Associate",
                "title_searchable": "Retail Sales Associate",
                "description_text": None,
                "description_usable": False,
                "onet_data": {"occupation_name": "Retail Salespersons"},
                "location_raw": None,
                "location_data_raw": [],
                "status_raw": "closed",
                "status_derived": "closed",
                "company_name_included": None,
                "company_domain_included": None,
                "url": None,
                "quality_flags": {"description": "null", "company_relationship": "missing_reference"},
            },
            {
                "job_id": "stable-developer",
                "job_type": "job_opening",
                "title_raw": "C# .NET Application Developer",
                "title_searchable": "C# .NET Application Developer",
                "description_text": "Build C# and .NET services with SQL.",
                "description_usable": True,
                "onet_data": {"occupation_name": "Software Developers"},
                "location_raw": "Remote, United States",
                "location_data_raw": [],
                "status_raw": "closed",
                "status_derived": "closed",
                "company_name_included": "Example Software",
                "company_domain_included": "software.invalid",
                "url": "https://example.invalid/jobs/developer",
                "quality_flags": {"description": "present"},
            },
        ]
        with self.sample_path.open("w", encoding="utf-8") as output:
            for record in self.records:
                output.write(json.dumps(record) + "\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_technical_tokens_are_preserved(self) -> None:
        self.assertEqual(tokenize("C++ C# .NET"), ["c++", "c#", ".net"])
        self.assertEqual(DESCRIPTION_EVIDENCE_RULE_VERSION, "substantive_description_v1")

    def test_description_evidence_rule_handles_missing_metadata_short_and_unheaded_signals(self) -> None:
        missing = assess_substantive_description({"description_text": "  "})
        self.assertEqual((missing.classification, missing.reason, missing.eligible),
                         ("missing_or_blank", "description_text_missing_or_blank", False))

        metadata = assess_substantive_description({
            "title_raw": "Inventory Associate",
            "description_text": "Inventory Associate\nPublished on 08/27/2026\nReference: 12345\n- Full Time\n- Beginner",
        })
        self.assertEqual((metadata.classification, metadata.reason, metadata.eligible),
                         ("insufficient", "metadata_only", False))

        short_duties = assess_substantive_description({
            "description_text": "Count stock accurately.\nReconcile inventory daily."
        })
        self.assertTrue(short_duties.eligible)
        self.assertEqual(short_duties.classification, "substantive_duties")

        unheaded_qualification = assess_substantive_description({
            "description_text": "Forklift certification required. One year inventory experience preferred."
        })
        self.assertTrue(unheaded_qualification.eligible)
        self.assertEqual(unheaded_qualification.classification, "substantive_qualifications")

        boilerplate = "\n".join([
            "We are proud to be a supportive and inclusive workplace.",
            "Benefits package includes health, dental and vision coverage.",
            "Equal opportunity employer committed to reasonable accommodations.",
        ] * 80)
        self.assertGreater(len(boilerplate), 5000)
        long_boilerplate = assess_substantive_description({"description_text": boilerplate})
        self.assertFalse(long_boilerplate.eligible)
        self.assertEqual(long_boilerplate.reason, "boilerplate_only")

    def test_description_evidence_reasons_are_stable_and_old_condition_is_unchanged(self) -> None:
        missing_record = {"job_id": "x", "title_searchable": "Retail Associate",
                          "description_text": None, "description_usable": False}
        substantive_record = {"job_id": "y", "title_searchable": "Retail Associate",
                              "description_text": "Count stock accurately.", "description_usable": True}
        self.assertEqual(assess_substantive_description(missing_record).reason,
                         "description_text_missing_or_blank")
        self.assertEqual(assess_substantive_description(substantive_record).reason,
                         "non_boilerplate_duty_evidence")
        jobs = [JobRecord("x", missing_record, False), JobRecord("y", substantive_record, True)]
        corpus = fit_tfidf(jobs)
        self.assertEqual(condition_job_indices(corpus, "description_available"), [1])
        self.assertEqual(condition_job_indices(corpus, "substantive_description"), [1])
        profile = CandidateProfile("p", desired_roles=("retail associate",))
        legacy_rank = rank_keyword(corpus, profile, top_k=2)
        self.assertEqual(legacy_rank[0]["job_id"], "x")
        self.assertEqual(legacy_rank[0]["condition"], "all_eligible")

    def test_top_five_change_report_tracks_entries_exits_and_rank_shifts(self) -> None:
        recommendations = [
            {"profile_id": "p", "method": "keyword", "condition": "description_available",
             "rank": 1, "job_id": "shared"},
            {"profile_id": "p", "method": "keyword", "condition": "description_available",
             "rank": 2, "job_id": "leaving"},
            {"profile_id": "p", "method": "keyword", "condition": "substantive_description",
             "rank": 1, "job_id": "entering"},
            {"profile_id": "p", "method": "keyword", "condition": "substantive_description",
             "rank": 2, "job_id": "shared"},
        ]
        change = job_matching_engine._top_five_changes(recommendations)[0]
        self.assertEqual(json.loads(change["entered_substantive_top5_job_ids"]), ["entering"])
        self.assertEqual(json.loads(change["left_description_available_top5_job_ids"]), ["leaving"])
        shifts = json.loads(change["common_job_rank_shifts"])
        self.assertEqual(shifts, [{"job_id": "shared", "description_available_rank": 1,
                                  "substantive_description_rank": 2, "rank_change": 1}])

    def test_synthetic_profile_json_is_valid_and_extensible(self) -> None:
        profile_path = PROJECT_ROOT / "code" / "synthetic_candidate_profiles.json"
        profiles = load_candidate_profiles(profile_path)
        self.assertEqual(len(profiles), 5)
        customer_profile = next(profile for profile in profiles if "customer_support" in profile.profile_id)
        self.assertTrue(customer_profile.experience)
        self.assertTrue(customer_profile.evidenced_skills)
        self.assertTrue(customer_profile.desired_roles)
        self.assertEqual(customer_profile.location, "Chicago")
        self.assertIn("C++", " ".join(customer_profile.evidenced_skills))

    def test_loader_preserves_source_and_counts_missing_descriptions(self) -> None:
        jobs, report = load_prepared_jobs(self.sample_path)
        self.assertEqual([job.job_id for job in jobs], ["stable-technical", "stable-retail", "stable-developer"])
        self.assertEqual(report.records_loaded, 3)
        self.assertEqual(report.usable_descriptions, 2)
        self.assertEqual(jobs[0].source["company_name_included"], "Example Components")
        self.assertEqual(jobs[0].source["quality_flags"]["company_relationship"], "matched")
        self.assertIsNone(jobs[1].source["description_text"])

    def test_exclusions_are_counted_by_reason(self) -> None:
        with self.sample_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"title_raw": "No ID"}) + "\n")
            output.write(json.dumps({"job_id": "stable-technical", "title_raw": "duplicate"}) + "\n")
            output.write("not-json\n")
            output.write("\n")
        jobs, report = load_prepared_jobs(self.sample_path)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(report.excluded_by_reason["missing_stable_job_id"], 1)
        self.assertEqual(report.excluded_by_reason["duplicate_job_id"], 1)
        self.assertEqual(report.excluded_by_reason["invalid_json"], 1)
        self.assertEqual(report.excluded_by_reason["blank_line"], 1)

    def test_keyword_and_tfidf_ordering_and_profile_transform(self) -> None:
        jobs, _ = load_prepared_jobs(self.sample_path)
        corpus = fit_tfidf(jobs)
        profile = CandidateProfile(
            "synthetic_support",
            experience=("customer support troubleshooting",),
            evidenced_skills=("C++ networking",),
            desired_roles=("technical support engineer",),
        )
        self.assertEqual(rank_keyword(corpus, profile)[0]["job_id"], "stable-technical")
        self.assertLess(rank_keyword(corpus, profile)[0]["score"], 1.0)
        self.assertEqual(rank_tfidf(corpus, profile)[0]["job_id"], "stable-technical")
        self.assertEqual(rank_tfidf(corpus, profile)[0]["rank"], 1)

    def test_unknown_status_is_retained_closed_filter_is_explicit(self) -> None:
        jobs, _ = load_prepared_jobs(self.sample_path)
        unknown = CandidateProfile("include_unknown")
        closed_job = jobs[1]
        self.assertTrue(filter_job(jobs[0], unknown).included)
        self.assertTrue(filter_job(closed_job, unknown).included)
        exclude_closed = CandidateProfile("closed_filter", preferences={"exclude_explicitly_closed": True})
        self.assertEqual(filter_job(closed_job, exclude_closed), FilterDecision(False, "explicitly_closed"))
        require_location = CandidateProfile("location_filter", location="Seattle",
                                            preferences={"require_location_match": True})
        self.assertEqual(filter_job(closed_job, require_location), FilterDecision(False, "unknown_location"))

    def test_metrics_leave_unjudged_results_unjudged(self) -> None:
        recommendations = [
            {"profile_id": "p", "method": "keyword", "rank": rank, "job_id": f"j{rank}"}
            for rank in range(1, 6)
        ]
        pool = {"p": [f"j{rank}" for rank in range(1, 6)]}
        partial = evaluate_reviewed_recommendations(
            recommendations, {("p", "j1"): 2}, reference_pools=pool
        )[0]
        self.assertIsNone(partial["precision_at_k"])
        self.assertEqual(partial["unjudged_recommendation_ranks"], [2, 3, 4, 5])
        self.assertIsNone(partial["ndcg_at_k"])
        self.assertEqual(partial["missing_reference_job_ids"], ["j2", "j3", "j4", "j5"])
        complete = evaluate_reviewed_recommendations(
            recommendations,
            {("p", f"j{rank}"): (2 if rank == 1 else 0) for rank in range(1, 6)},
            reference_pools=pool,
        )[0]
        self.assertEqual(complete["precision_at_k"], 0.2)
        self.assertEqual(complete["ndcg_at_k"], 1.0)

    def test_ndcg_uses_shared_pool_ideal_not_each_method_top_five(self) -> None:
        recommendations = []
        for method, job_ids in (
            ("strong", ["a", "b", "c", "d", "e"]),
            ("weak", ["f", "g", "h", "i", "j"]),
        ):
            recommendations.extend({
                "profile_id": "p", "method": method, "condition": "all_eligible",
                "rank": rank, "job_id": job_id,
            } for rank, job_id in enumerate(job_ids, 1))
        pool = {"p": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]}
        labels = {("p", job_id): 2 for job_id in "abcde"}
        labels.update({("p", job_id): 0 for job_id in "fghij"})

        metrics = evaluate_reviewed_recommendations(recommendations, labels, reference_pools=pool)
        by_method = {row["method"]: row for row in metrics}
        self.assertEqual(by_method["strong"]["ndcg_at_k"], 1.0)
        self.assertEqual(by_method["weak"]["ndcg_at_k"], 0.0)
        self.assertEqual(by_method["strong"]["precision_at_k"], 1.0)
        self.assertEqual(by_method["weak"]["precision_at_k"], 0.0)

    def test_shared_pool_incomplete_labels_and_short_results_are_explicit(self) -> None:
        recommendations = [
            {"profile_id": "p", "method": "short", "condition": "all_eligible", "rank": 1, "job_id": "a"},
            {"profile_id": "p", "method": "short", "condition": "all_eligible", "rank": 2, "job_id": "b"},
        ]
        result = evaluate_reviewed_recommendations(
            recommendations,
            {("p", "a"): 2, ("p", "b"): 1},
            reference_pools={"p": ["a", "b", "c"]},
        )[0]
        self.assertIsNone(result["precision_at_k"])
        self.assertIsNone(result["ndcg_at_k"])
        self.assertEqual(result["precision_status"], "fewer_than_k_results")
        self.assertEqual(result["missing_reference_job_ids"], ["c"])

        complete_pool = evaluate_reviewed_recommendations(
            recommendations,
            {("p", "a"): 0, ("p", "b"): 0, ("p", "c"): 0},
            reference_pools={"p": ["a", "b", "c"]},
        )[0]
        self.assertEqual(complete_pool["ndcg_at_k"], 0.0)
        self.assertEqual(complete_pool["ndcg_status"], "reported_shared_pool_all_zero_relevance")
        short_but_nonzero = evaluate_reviewed_recommendations(
            recommendations,
            {("p", "a"): 2, ("p", "b"): 0, ("p", "c"): 1},
            reference_pools={"p": ["a", "b", "c"]},
        )[0]
        self.assertAlmostEqual(short_but_nonzero["ndcg_at_k"], 3 / (3 + 1 / 1.584962500721156))
        self.assertEqual(short_but_nonzero["ndcg_zero_padded_result_slots"], 3)
        self.assertIsNone(short_but_nonzero["precision_at_k"])

    def test_coverage_conditions_reuse_same_fitted_tfidf_rows(self) -> None:
        from unittest.mock import patch

        jobs, _ = load_prepared_jobs(self.sample_path)
        profile = CandidateProfile("p", desired_roles=("technical support engineer",))
        with patch.object(job_matching_engine, "fit_tfidf", wraps=fit_tfidf) as fit_spy:
            corpus, recommendations, condition_counts = run_description_coverage_rankings(jobs, [profile])
        self.assertEqual(fit_spy.call_count, 1)
        matrix_before = corpus.job_matrix
        vectorizer_before = corpus.vectorizer
        condition_b_indices = condition_job_indices(corpus, "description_available")
        rank_tfidf(corpus, profile, candidate_indices=condition_b_indices, condition="description_available")
        self.assertIs(corpus.job_matrix, matrix_before)
        self.assertIs(corpus.vectorizer, vectorizer_before)
        self.assertEqual(condition_counts["description_available"], sum(job.usable_description for job in jobs))
        self.assertEqual(condition_counts["substantive_description"], sum(
            assess_substantive_description(job.source).eligible for job in jobs
        ))
        described_ids = {row["job_id"] for row in recommendations if row["condition"] == "description_available"}
        substantive_ids = {row["job_id"] for row in recommendations if row["condition"] == "substantive_description"}
        self.assertEqual(described_ids, substantive_ids)
        self.assertTrue(all(jobs[index].usable_description for index in range(len(jobs))
                            if jobs[index].job_id in described_ids))
        self.assertEqual({row["condition"] for row in recommendations},
                 {"description_available", "substantive_description"})

    def test_coverage_writer_pools_ranks_and_blinds_review_view(self) -> None:
        jobs, load_report = load_prepared_jobs(self.sample_path)
        profile_path = Path(self.temp_dir.name) / "profiles.json"
        profile_path.write_text(json.dumps([{
            "profile_id": "synthetic_writer_test",
            "experience": ["customer support"],
            "evidenced_skills": ["C++"],
            "desired_roles": ["technical support engineer"],
            "location": None,
            "preferences": {},
        }]), encoding="utf-8")
        profile = load_candidate_profiles(profile_path)[0]
        corpus, recommendations, counts = run_description_coverage_rankings(jobs, [profile], top_k=2)
        output_dir = Path(self.temp_dir.name) / "coverage-output"
        summary = write_description_coverage_outputs(
            output_dir, corpus, [profile], recommendations, counts,
            self.sample_path, profile_path, {"sample_loading_seconds": 0.01},
            Path(self.temp_dir.name) / "no-prior-baseline.csv",
        )
        with (output_dir / "pooled_review_template.csv").open(encoding="utf-8-sig", newline="") as source:
            pooled = list(csv.DictReader(source))
        with (output_dir / "blinded_review_view.csv").open(encoding="utf-8-sig", newline="") as source:
            blinded_reader = csv.DictReader(source)
            blinded = list(blinded_reader)
            blinded_fields = set(blinded_reader.fieldnames or [])
        self.assertGreaterEqual(len(pooled), 1)
        self.assertIn("description_available|tfidf_cosine", pooled[0]["condition_ranks"])
        self.assertNotIn("method", blinded_fields)
        self.assertNotIn("condition", blinded_fields)
        self.assertNotIn("rank", blinded_fields)
        self.assertNotIn("score", blinded_fields)
        self.assertIn("description_full_prepared", blinded_fields)
        self.assertTrue(all(row["relevance_score"] == "" for row in blinded))
        self.assertEqual(summary["tfidf_fit_count"], 1)
        self.assertTrue(summary["tfidf_reused_for_both_conditions"])
        self.assertIn("substantive_description", summary["conditions"])
        self.assertNotIn("all_eligible", summary["conditions"])
        self.assertEqual(summary["independent_label_provenance"]["numeric_label_pairs_loaded"], 0)
        self.assertTrue((output_dir / "description_evidence_diagnostics.csv").is_file())
        self.assertTrue((output_dir / "top5_changes.csv").is_file())

    def test_review_template_pools_methods_without_creating_labels(self) -> None:
        recommendations = [
            {"profile_id": "p1", "job_id": "j1", "method": "keyword", "rank": 2,
             "title": "Source title", "company_name": "Company", "source_url": "https://example.invalid"},
            {"profile_id": "p1", "job_id": "j1", "method": "tfidf_cosine", "rank": 1,
             "title": "Source title", "company_name": "Company", "source_url": "https://example.invalid"},
        ]
        pooled = pooled_review_rows(recommendations)
        self.assertEqual(len(pooled), 1)
        self.assertEqual(json.loads(pooled[0]["condition_ranks"]), {"all_eligible|keyword": 2, "all_eligible|tfidf_cosine": 1})
        self.assertEqual(pooled[0]["relevance_score"], "")
        template = Path(self.temp_dir.name) / "review.csv"
        import csv
        with template.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=pooled[0].keys())
            writer.writeheader()
            writer.writerows(pooled)
        self.assertEqual(read_review_labels(template), {})


if __name__ == "__main__":
    unittest.main()
