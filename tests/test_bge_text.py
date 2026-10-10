"""Golden tests for the live BGE text builders.

The expected strings were captured from the original experiment implementation. The saved listing embeddings were
made from exactly this text, so changing it silently would make profile queries stop matching the stored vectors.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import bge_text  # noqa: E402
import job_matching_engine as engine  # noqa: E402


class WordTokenizer:
    """Deterministic stand-in for the BGE tokenizer: one token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": list(range(len(text.split()) + (2 if add_special_tokens else 0)))}


JOB = {
    "title_searchable": "Sales Engineer", "title_raw": "Sales Engineer",
    "onet_data": {"occupation_name": "Sales Engineers"},
    "description_text": (
        "About us\nWe are a great company.\nResponsibilities\nDemonstrate the product to prospects.\n"
        "Work with engineers.\nRequirements\n3+ years of technical sales experience.\n"
        "Preferred: SQL knowledge.\nBenefits\nFree snacks."
    ),
}

PROFILE = engine.CandidateProfile(
    profile_id="p", experience=("Support lead: handled tickets", "Built a dashboard"),
    evidenced_skills=("SQL: reporting queries",), desired_roles=("solutions engineer",), location=None, preferences={},
)
FIXTURE = {
    "work_history": ["Support lead: handled tickets"], "accomplishments": ["Built a dashboard"],
    "experience_level": "early career", "career_transition_goal": "move into technical sales",
    "education_certifications": ["Data certificate"], "soft_preferences": ["remote"],
    "work_arrangement_preferences": [],
}


class BgeTextTests(unittest.TestCase):
    def test_constants_match_the_saved_embeddings(self) -> None:
        self.assertEqual(bge_text.BGE_ID, "BAAI/bge-base-en-v1.5")
        self.assertEqual(bge_text.BGE_REVISION, "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
        self.assertEqual(bge_text.BGE_REPRESENTATION, "bge_section_priority_v1")
        self.assertEqual(bge_text.MAX_BGE_TOKENS, 512)
        self.assertEqual(bge_text.BGE_QUERY_PREFIX, "Represent this sentence for searching relevant passages: ")

    def test_job_text_is_unchanged(self) -> None:
        content, report = bge_text.build_bge_job_text(JOB, WordTokenizer())
        self.assertEqual(
            content,
            "Job title: Sales Engineer. Occupation: Sales Engineers. Requirement: 3+ years of technical sales experience. "
            "Duty: Demonstrate the product to prospects. Requirement: Preferred: SQL knowledge. Duty: Work with engineers.",
        )
        self.assertEqual(report["encoded_tokens_including_specials"], 30)
        self.assertEqual(report["omitted_sections"], [])
        self.assertNotIn("snacks", content)  # the benefits section is dropped as noise

    def test_candidate_text_is_unchanged(self) -> None:
        content, report = bge_text.build_bge_candidate_text(PROFILE, FIXTURE, WordTokenizer())
        self.assertEqual(
            content,
            "Target roles: solutions engineer Experience level: early career Transition goal: move into technical sales "
            "Work history: Support lead: handled tickets Accomplishments: Built a dashboard Demonstrated skills: SQL: "
            "reporting queries Education and certifications: Data certificate Soft preferences: remote",
        )
        self.assertEqual(report["encoded_tokens_including_prefix_and_specials"], 46)
        self.assertEqual(report["omitted_items"], 0)

    def test_token_budget_drops_whole_pieces_and_stays_within_limit(self) -> None:
        long_description = "Requirements\n" + "\n".join(f"Must have skill number {i} in depth." for i in range(400))
        job = {**JOB, "description_text": long_description}
        content, report = bge_text.build_bge_job_text(job, WordTokenizer())
        self.assertLessEqual(report["encoded_tokens_including_specials"], bge_text.MAX_BGE_TOKENS)
        self.assertGreater(report["omitted_items"], 0)
        self.assertTrue(content.endswith("."))  # never cut mid-sentence

    def test_model_dir_prefers_environment_override(self) -> None:
        import os
        previous = os.environ.get(bge_text.MODEL_DIR_ENV)
        os.environ[bge_text.MODEL_DIR_ENV] = "somewhere/else"
        try:
            self.assertEqual(bge_text.model_dir(), Path("somewhere/else"))
        finally:
            if previous is None:
                del os.environ[bge_text.MODEL_DIR_ENV]
            else:
                os.environ[bge_text.MODEL_DIR_ENV] = previous


if __name__ == "__main__":
    unittest.main()
