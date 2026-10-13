"""BACKGROUND_SOC (src/career_switch.py) covers every Profiles background area. No data."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.career_switch import BACKGROUND_AREAS, BACKGROUND_SOC


class BackgroundSocTests(unittest.TestCase):
    def test_every_area_has_codes(self):
        self.assertEqual(set(BACKGROUND_SOC), set(BACKGROUND_AREAS))

    def test_codes_are_soc_prefixes_outside_the_scope(self):
        for area, prefixes in BACKGROUND_SOC.items():
            self.assertTrue(prefixes, area)
            for p in prefixes:
                self.assertRegex(p, r"^\d{2}-\d{0,4}$", area)
                self.assertFalse(p.startswith("15-"), area)  # computer and mathematical is the target, never a background


if __name__ == "__main__":
    unittest.main()
