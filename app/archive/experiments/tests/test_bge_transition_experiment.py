import csv
import gc
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import job_matching_engine as engine
from run_bge_transition_experiment import (
    BGE_REPRESENTATION,
    AMENDED_ENCODING_BUDGET_SECONDS,
    EncodingBudgetExceeded,
    _bge_cache_key,
    _build_blinded_pool,
    _cache_manifest_valid,
    _encode_bge_cache,
    _ranking_identity,
    build_bge_candidate_text,
    build_bge_job_text,
    compute_metrics,
    eligible_indices,
    sha256_file,
    select_length_stratified,
    save_pending_judgments,
    validate_amended_runtime_gate,
)


class WordTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False):
        words = str(text).split()
        return {"input_ids": (["[CLS]"] + words + ["[SEP]"]) if add_special_tokens else words}


def _profile_record(profile_id="test_profile"):
    return {
        "profile_id": profile_id,
        "fixture_type": "career_transition",
        "synthetic_fixture": True,
        "work_history": ["Checked pump readings and recorded each route visit."],
        "accomplishments": ["Created a handoff list for unresolved readings."],
        "experience_level": "three years",
        "skills_with_evidence": [{"skill": "field records", "evidence": "recorded readings by route"}],
        "education_certifications": ["High school diploma", "Commercial driver's license status unknown"],
        "qualification_disclosures": {"explicit_absence": [], "unknown": ["Commercial driver's license"]},
        "desired_roles": ["utility field coordinator"],
        "career_transition_goal": "Move into utility coordination.",
        "location": "Toledo, Ohio",
        "work_arrangement_preferences": ["field work"],
        "soft_preferences": ["safety procedures"],
        "hard_constraints": {"exclude_explicitly_closed": True, "require_location_match": False,
                             "location_requirement": None},
    }


def _job(job_id, status="open", location="Toledo, Ohio", description="Read meters and record results."):
    return engine.JobRecord(job_id, {
        "job_id": job_id, "title_raw": "Meter Technician", "title_searchable": "Meter Technician",
        "location_raw": location, "status_derived": status, "description_text": description,
    }, True)


class BgeTransitionExperimentTests(unittest.TestCase):
    def test_candidate_prefix_is_included_in_512_token_budget(self):
        fixture = _profile_record()
        fixture["work_history"] = ["field reading record " * 120 for _ in range(5)]
        fixture["accomplishments"] = ["documented route handoff " * 100 for _ in range(3)]
        fixture["skills_with_evidence"] = [{"skill": "field notes", "evidence": "route readings " * 80}]
        profile = engine.CandidateProfile(
            profile_id=fixture["profile_id"], experience=tuple(fixture["work_history"] + fixture["accomplishments"]),
            evidenced_skills=("field notes: route readings",), desired_roles=tuple(fixture["desired_roles"]),
            location=fixture["location"], preferences={},
        )
        text, report = build_bge_candidate_text(profile, fixture, WordTokenizer())
        self.assertTrue(text.startswith("Target roles:"))
        self.assertLessEqual(report["encoded_tokens_including_prefix_and_specials"], 512)
        self.assertIn("Target roles", report["retained_sections"])
        self.assertGreater(report["omitted_tokens_estimate"], 0)

    def test_job_document_has_no_query_prefix_and_reports_budget_omissions(self):
        requirements = " ".join("Minimum experience required for department {}.".format(i)
                                for i in range(220))
        duties = " ".join("Inspect equipment {} and record each reading.".format(i)
                          for i in range(220))
        record = _job("long", description=(
            "Requirements:\n" + requirements
            + "\nResponsibilities:\n" + duties
        ))
        text, report = build_bge_job_text(record.source, WordTokenizer())
        self.assertFalse(text.startswith("Represent this sentence"))
        self.assertLessEqual(report["encoded_tokens_including_specials"], 512)
        self.assertGreater(report["omitted_tokens_estimate"], 0)
        self.assertIn("requirements", report["retained_sections"])

    def test_shared_hard_filter_retains_unknown_status_and_excludes_closed(self):
        corpus = engine.MatchingCorpus([
            _job("open", "open"), _job("unknown", "unknown"), _job("closed", "closed"),
            _job("other_city", "open", location="Buffalo, New York"),
        ], None, None)
        profile = engine.CandidateProfile(
            profile_id="filter", location="Toledo", preferences={
                "exclude_explicitly_closed": True, "require_location_match": True,
            })
        self.assertEqual(eligible_indices(corpus, profile, [0, 1, 2, 3]), [0, 1])

    def test_cache_key_depends_on_ordered_ids_and_exact_text(self):
        baseline = _bge_cache_key(["a", "b"], ["document one", "document two"])
        self.assertNotEqual(baseline, _bge_cache_key(["b", "a"], ["document two", "document one"]))
        self.assertNotEqual(baseline, _bge_cache_key(["a", "b"], ["document one", "changed text"]))

    def test_sample_hash_comparison_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"frozen input")
            digest = sha256_file(path)
            self.assertEqual(digest.casefold(), digest.upper().casefold())

    def test_benchmark_selection_spans_original_prebudget_lengths_deterministically(self):
        rows = [{"job_id": "j{:03d}".format(i), "original_length": i * 10,
                 "encoded_length": min(512, i * 10)} for i in range(1000)]
        selected = select_length_stratified(rows, 20)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, select_length_stratified(rows, 20))
        self.assertLess(selected[0]["original_length"], selected[-1]["original_length"])
        self.assertEqual(selected[-1]["original_length"], 9750)

    def test_amended_gate_accepts_projected_106_minutes_without_mutating_original(self):
        benchmark = {"model_id": "BAAI/bge-base-en-v1.5",
                     "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
                     "device": "cpu", "description_available_records": 24514,
                     "full_encoding_time_estimate_seconds": 6396.48,
                     "memory": {"stable_under_gate": True},
                     "gate": {"limit_seconds": 3600, "projected_within_limit": False,
                              "proceed_to_full_encoding": False}}
        original = json.dumps(benchmark, sort_keys=True)
        amendment = {"amended_cumulative_cpu_corpus_encoding_limit_seconds": 7200}
        gate = validate_amended_runtime_gate(benchmark, amendment)
        self.assertTrue(gate["proceed_to_encoding"])
        self.assertEqual(gate["original_benchmark_gate_seconds"], 3600)
        self.assertEqual(AMENDED_ENCODING_BUDGET_SECONDS, 7200)
        self.assertEqual(json.dumps(benchmark, sort_keys=True), original)
        benchmark["memory"]["stable_under_gate"] = False
        self.assertFalse(validate_amended_runtime_gate(benchmark, amendment)["proceed_to_encoding"])

    def test_partial_bge_cache_resumes_from_verified_checkpoint(self):
        class FakeModel:
            def __init__(self, delay):
                self.delay = delay

            def encode(self, texts, **kwargs):
                if self.delay:
                    time.sleep(self.delay)
                return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            final = cache_dir / "bge.npy"
            manifest = cache_dir / "bge.json"
            partial = cache_dir / "bge.partial.npy"
            progress = cache_dir / "bge.partial.progress.json"
            paths = (final, manifest, partial, progress)
            job_ids = ["j{:02d}".format(i) for i in range(64)]
            texts = ["document {}".format(i) for i in range(64)]
            with patch("run_bge_transition_experiment._new_model_cache_paths", return_value=paths):
                with self.assertRaises(EncodingBudgetExceeded) as stopped:
                    _encode_bge_cache(cache_dir, FakeModel(0.03), job_ids, texts,
                                      "frozen-cache-key", 3, budget_seconds=0.01)
                self.assertEqual(stopped.exception.completed_rows, 16)
                cumulative_after_first = stopped.exception.cumulative_seconds
                checkpoint = json.loads(progress.read_text(encoding="utf-8"))
                self.assertEqual(checkpoint["completed_rows"], 16)
                self.assertEqual(len(checkpoint["completed_prefix_sha256"]), 64)
                with self.assertRaises(EncodingBudgetExceeded) as stopped_again:
                    _encode_bge_cache(cache_dir, FakeModel(0.03), job_ids, texts,
                                      "frozen-cache-key", 3,
                                      budget_seconds=cumulative_after_first + 0.01)
                self.assertEqual(stopped_again.exception.completed_rows, 32)
                self.assertGreater(stopped_again.exception.cumulative_seconds, cumulative_after_first)
                matrix, info = _encode_bge_cache(cache_dir, FakeModel(0), job_ids, texts,
                                                 "frozen-cache-key", 3, budget_seconds=60)
                self.assertEqual(matrix.shape, (64, 3))
                self.assertEqual(info["resumed_at_row"], 32)
                self.assertEqual(info["rows_encoded_this_run"], 32)
                self.assertTrue(final.is_file())
                self.assertTrue(manifest.is_file())
                self.assertFalse(progress.exists())
                _, hit = _encode_bge_cache(cache_dir, FakeModel(0), job_ids, texts,
                                           "frozen-cache-key", 3, budget_seconds=60)
                self.assertTrue(hit["cache_hit"])
                del matrix
                del _
                gc.collect()

    def test_cache_validation_rejects_manifest_or_file_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npy"
            np.save(path, np.ones((2, 3), dtype=np.float32))
            from run_bge_transition_experiment import _write_json, sha256_file
            expected = {"cache_key": "frozen", "model_id": "bge", "dimension": 3, "job_count": 2}
            manifest_path = path.with_suffix(".json")
            _write_json(manifest_path, {**expected, "embedding_file_sha256": sha256_file(path)})
            self.assertIsNotNone(_cache_manifest_valid(path, manifest_path, expected))
            self.assertIsNone(_cache_manifest_valid(path, manifest_path, {**expected, "model_id": "other"}))
            path.write_bytes(path.read_bytes() + b"tampered")
            self.assertIsNone(_cache_manifest_valid(path, manifest_path, expected))

    def test_union_deduplication_is_stable_and_blinds_ranker_fields(self):
        fixtures = {"p": _profile_record("p")}
        jobs = {"j": _job("j")}
        rows = [
            {"profile_id": "p", "job_id": "j", "method": "tfidf_cosine", "rank": 1, "score": 0.4},
            {"profile_id": "p", "job_id": "j", "method": BGE_REPRESENTATION, "rank": 2, "score": 0.9},
        ]
        pool = _build_blinded_pool(rows, fixtures, jobs, 5)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]["job_id"], "j")
        self.assertFalse({"method", "rank", "score", "score_type"} & set(pool[0]))
        self.assertEqual(pool, _build_blinded_pool(rows, fixtures, jobs, 5))

    def test_ranking_mapping_identity_preserves_pair_rank(self):
        rows = [{"profile_id": "p", "method": "m", "rank": 1, "job_id": "j", "score": 0.2}]
        mapping = [{"profile_id": "p", "method": "m", "rank": 1, "job_id": "j", "score": 0.9}]
        self.assertEqual(_ranking_identity(rows), _ranking_identity(mapping))
        mapping[0]["job_id"] = "other"
        self.assertNotEqual(_ranking_identity(rows), _ranking_identity(mapping))

    def test_unknown_top_five_is_preserved_and_withholds_p5_and_ndcg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = _profile_record("p")
            (root / "synthetic_profiles.json").write_text(json.dumps([profile]), encoding="utf-8")
            pool = [{"profile_id": "p", "job_id": "j{}".format(i)} for i in range(10)]
            (root / "blinded_evaluation_pool.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in pool), encoding="utf-8")
            fields = ["profile_id", "method", "rank", "job_id"]
            rankings = [{"profile_id": "p", "method": method, "rank": rank + 1,
                         "job_id": "j{}".format(rank)}
                        for method in ("tfidf_cosine", "embedding_section_aware_v1", BGE_REPRESENTATION)
                        for rank in range(10)]
            with (root / "rankings.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rankings)
            pending = {"profile_id": "p", "job_id": "j0", "status": "pending_evaluator",
                       "grade": None, "unknown_reason": "insufficient evidence"}
            numeric = [{"profile_id": "p", "job_id": "j{}".format(i), "grade": 2,
                        "status": "judged", "unknown_reason": None,
                        "supporting_candidate_evidence": [], "supporting_job_evidence": [],
                        "material_gaps": [], "explicit_constraint_conflicts": [],
                        "required_qualifications": "unknown",
                        "career_transition_relevance": {"level": "moderate", "rationale": "transferable duties"},
                        "title_outside_stated_targets": True, "evidence_sufficiency": "sufficient"}
                       for i in range(1, 10)]
            (root / "raw_judgments.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in [pending, *numeric]), encoding="utf-8")
            result = compute_metrics(root)
            methods = result["per_profile"][0]["methods"]
            for method in methods.values():
                self.assertIsNone(method["plausible_or_strong_p5"])
                self.assertIsNone(method["ndcg_at_10"])
                self.assertEqual(method["unknown_top5_ranks"], [1])
            self.assertEqual(result["judgment_coverage"]["unknown_or_pending"], 1)
            self.assertEqual(result["predeclared_criterion"]["status"], "unassessable")

    def test_unavailable_evaluator_saves_pending_null_rows_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = [{"profile_id": "p", "job_id": "j1", "presentation_id": "B0001"},
                    {"profile_id": "p", "job_id": "j2", "presentation_id": "B0002"}]
            (root / "blinded_evaluation_pool.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in pool), encoding="utf-8")
            provenance = save_pending_judgments(root)
            raw = [json.loads(line) for line in (root / "raw_judgments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(raw), 2)
            self.assertTrue(all(row["grade"] is None and row["status"] == "pending_evaluator" for row in raw))
            self.assertEqual(provenance["paid_api_calls"], 0)
            with self.assertRaises(FileExistsError):
                save_pending_judgments(root)


if __name__ == "__main__":
    unittest.main()