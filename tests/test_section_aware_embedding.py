from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_name, relative in (
    ("job_matching_engine", Path("code/job_matching_engine.py")),
    ("section_aware_embedding", Path("code/section_aware_embedding.py")),
):
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

engine = sys.modules["job_matching_engine"]
representation = sys.modules["section_aware_embedding"]


class WhitespaceTokenizer:
    model_max_length = 256

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> dict[str, list[int]]:
        length = len(text.split()) + (2 if add_special_tokens else 0)
        return {"input_ids": list(range(length))}


class SectionAwareEmbeddingTests(unittest.TestCase):
    def test_job_text_prioritizes_requirements_and_duties_and_excludes_marketing(self) -> None:
        tokenizer = WhitespaceTokenizer()
        record = {
            "title_raw": "Maintenance Technician",
            "title_searchable": "Maintenance Technician",
            "onet_data": {"occupation_name": "Maintenance Workers"},
            "description_text": (
                "Company Overview\nA celebrated company with a vibrant culture and excellent benefits.\n"
                "Responsibilities\nInspect equipment and record service results.\n"
                "Requirements\nMust have a valid driver license and two years of maintenance experience.\n"
                "Benefits\nMedical, dental and retirement benefits."
            ),
        }
        text = representation.section_aware_job_text(record, tokenizer, max_length=256)
        self.assertIn("Requirement: Must have a valid driver license", text)
        self.assertIn("Duty: Inspect equipment", text)
        self.assertNotIn("celebrated company", text)
        self.assertNotIn("Medical, dental", text)

    def test_job_and_candidate_text_respect_token_budget_and_candidate_priority(self) -> None:
        tokenizer = WhitespaceTokenizer()
        profile = engine.CandidateProfile(
            "p", experience=("Maintained intake records.",),
            evidenced_skills=("record review: reconciled files",),
            desired_roles=("claims examiner",), preferences={"keywords": ["x " * 100]},
        )
        fixture = {
            "experience_level": "mid-level",
            "career_transition_goal": "Continue claims work.",
            "work_history": ["Maintained intake records."],
            "accomplishments": ["Reduced incomplete claim packets."],
            "education_certifications": ["Associate degree"],
            "soft_preferences": ["x " * 100],
            "work_arrangement_preferences": [],
        }
        text = representation.section_aware_candidate_text(profile, fixture, tokenizer, max_length=24)
        self.assertLessEqual(len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]), 24)
        self.assertIn("Target roles", text)
        self.assertIn("Experience level", text)
        self.assertIn("Transition goal", text)
        self.assertNotIn("Soft preferences", text)

    def test_sparse_job_description_is_not_fabricated_or_expanded(self) -> None:
        tokenizer = WhitespaceTokenizer()
        record = {"title_raw": "Field Technician", "title_searchable": "Field Technician",
                  "description_text": "Posted today. Apply now."}
        text = representation.section_aware_job_text(record, tokenizer, max_length=256)
        self.assertIn("Field Technician", text)
        self.assertNotIn("Requirement:", text)
        self.assertNotIn("Duty:", text)
        self.assertNotIn("experience", text.lower())


if __name__ == "__main__":
    unittest.main()