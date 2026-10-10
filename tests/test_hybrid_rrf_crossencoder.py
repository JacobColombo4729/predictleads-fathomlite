import sys
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import job_matching_engine as engine
import run_hybrid_rrf_crossencoder_experiment as hybrid


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def __call__(self, first, second=None, add_special_tokens=True, truncation=False):
        count = len(self.encode(first))
        if second is None:
            count += 2 if add_special_tokens else 0
        else:
            count += len(self.encode(second)) + (3 if add_special_tokens else 0)
        return {"input_ids": list(range(count))}


class HybridExperimentTests(unittest.TestCase):
    def test_profile_loader_accepts_split_sizes_without_bge_eight_profile_assumption(self):
        row = {
            "profile_id": "dev-profile", "fixture_type": "career_transition",
            "synthetic_fixture": True, "work_history": ["Logged field observations."],
            "accomplishments": ["Created a route checklist."], "experience_level": "3 years",
            "skills_with_evidence": [{"skill": "documentation", "evidence": "logged observations"}],
            "education_certifications": ["Associate degree"],
            "qualification_disclosures": {"explicit_absence": [], "unknown": []},
            "desired_roles": ["field coordinator"], "career_transition_goal": "Move into field coordination.",
            "soft_preferences": ["written procedures"],
            "hard_constraints": {"exclude_explicitly_closed": True, "require_location_match": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            fixtures, profiles = hybrid._load_profiles(path, ["dev-profile"])
        self.assertEqual(list(fixtures), ["dev-profile"])
        self.assertEqual(profiles[0].desired_roles, ("field coordinator",))

    def test_rrf_combines_complementary_rankings_deterministically(self):
        lexical = [{"job_id": "a", "rank": 1}, {"job_id": "b", "rank": 2}]
        dense = [{"job_id": "b", "rank": 1}, {"job_id": "c", "rank": 2}]
        scores = hybrid._rrf_scores(lexical, dense, 60)
        self.assertGreater(scores["b"], scores["a"])
        self.assertGreater(scores["b"], scores["c"])
        self.assertEqual(scores, hybrid._rrf_scores(lexical, dense, 60))

    def test_pair_builder_retains_candidate_transfer_skills_and_job_requirements(self):
        fixture = {
            "profile_id": "dev",
            "desired_roles": ["quality technician"],
            "career_transition_goal": "Move to documented quality work.",
            "work_history": ["Inspected assembled parts and recorded defects."],
            "accomplishments": ["Created a handoff checklist."],
            "skills_with_evidence": [{"skill": "inspection", "evidence": "checked parts against a measurement guide"}],
            "education_certifications": ["Associate degree"],
        }
        profile = engine.CandidateProfile(
            profile_id="dev", desired_roles=("quality technician",),
            experience=("Inspected assembled parts and recorded defects.",),
            evidenced_skills=("inspection: checked parts against a measurement guide",),
        )
        source = {
            "job_id": "job", "title_raw": "Production Quality Assistant",
            "title_searchable": "Production Quality Assistant", "title_normalized": "quality assistant",
            "description_text": (
                "Requirements:\nMust have experience documenting inspections and measurement results.\n"
                "Responsibilities:\nInspect production components and record dimensional defects."
            ),
        }
        job = engine.JobRecord("job", source, True)
        candidate, passage, report = hybrid._cross_pair_text(profile, fixture, job, WordTokenizer())
        self.assertIn("quality technician", candidate)
        self.assertIn("inspection: checked parts against a measurement guide", candidate)
        self.assertIn("Job requirement", passage)
        self.assertIn("Job duty", passage)
        self.assertLessEqual(report["job"]["pair_tokens_including_specials"], hybrid.CROSS_ENCODER_MAX_LENGTH)

    def test_pair_builder_reports_omitted_candidate_evidence(self):
        fixture = {
            "profile_id": "long", "desired_roles": ["technician"],
            "career_transition_goal": "Transition to field work.",
            "work_history": ["Recorded field conditions and coordinated a route handoff. " * 100],
            "accomplishments": ["Created a checklist. " * 100],
            "skills_with_evidence": [{"skill": "records", "evidence": "logged observations"}],
            "education_certifications": ["Diploma"],
        }
        profile = engine.CandidateProfile(profile_id="long", desired_roles=("technician",))
        job = engine.JobRecord("job", {
            "job_id": "job", "title_raw": "Field Technician", "title_searchable": "Field Technician",
            "description_text": "Responsibilities:\nRecord inspections and report hazards.",
        }, True)
        _, _, report = hybrid._cross_pair_text(profile, fixture, job, WordTokenizer())
        self.assertGreater(report["candidate"]["omitted_items"], 0)
        self.assertLessEqual(report["job"]["pair_tokens_including_specials"], 512)


if __name__ == "__main__":
    unittest.main()
