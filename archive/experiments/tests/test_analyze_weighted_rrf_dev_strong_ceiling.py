import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analyze_weighted_rrf_dev_strong_ceiling import (
    _classify_requirements,
    _requirements_and_duties,
)


class WeightedRrfStrongCeilingAnalysisTests(unittest.TestCase):
    def test_explicit_required_and_preferred_qualifications_are_separated(self):
        result = _classify_requirements([
            {"text": "A valid state operator license is required."},
            {"text": "Forklift certification preferred."},
        ])

        self.assertEqual(result["explicit_required"], ["A valid state operator license is required."])
        self.assertEqual(result["explicit_preferred"], ["Forklift certification preferred."])
        self.assertEqual(result["modality_unclear"], [])

    def test_conditional_or_mixed_qualification_modality_stays_unclear(self):
        result = _classify_requirements([
            {"text": "Possess or have ability to obtain an operator certification."},
            {"text": "Certification may be required; a diploma is strongly preferred."},
            {"text": "No travel required."},
        ])

        self.assertEqual(result["explicit_required"], [])
        self.assertEqual(result["explicit_preferred"], [])
        self.assertEqual(result["modality_unclear"], [
            "Possess or have ability to obtain an operator certification.",
            "Certification may be required; a diploma is strongly preferred.",
        ])

    def test_duty_phrase_as_required_is_not_a_qualification(self):
        sections = _requirements_and_duties(
            "Responsibilities\nRespond to alarms and emergencies as required."
        )

        self.assertIn("Respond to alarms and emergencies as required.", sections["duties"])
        self.assertEqual(sections["qualification_or_modality_lines"], [])


if __name__ == "__main__":
    unittest.main()