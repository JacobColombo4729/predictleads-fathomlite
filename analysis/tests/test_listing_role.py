"""A listing's role checked against its title (career_switch.listing_role) and the per-company cap
(rank_model.cap_per_company): serving-time rules, made-up titles only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import career_switch as cs
from src import rank_model as rm


class ListingRole(unittest.TestCase):
    def test_titles_that_are_no_tech_role_have_no_role(self):
        for title in ["Data Entry Specialist", "Data Contributor - Remote", "AI Trainer (Python)", "Search Quality Rater",
                      "Data Annotator", "Data Labeler", "Medical Transcriptionist", "Billing Clerk"]:
            self.assertIsNone(cs.listing_role(title, "data science"), title)

    def test_profiles_title_rules_move_a_listing_to_its_title_bucket(self):
        self.assertEqual(cs.listing_role("Senior Data Analyst", "database"), "data science")
        self.assertEqual(cs.listing_role("Software Engineer", "data science"), "software development")
        self.assertEqual(cs.listing_role("Help Desk Technician", "networks and systems"), "it support")

    def test_an_unrecognised_title_keeps_its_occupation_code(self):
        self.assertEqual(cs.listing_role("AI Engineer", "data science"), "data science")
        self.assertEqual(cs.listing_role("Operations Associate", "data science"), "data science")
        self.assertIsNone(cs.listing_role("Operations Associate", None))

    def test_title_bucket_is_profiles_first_match(self):
        self.assertEqual(cs.title_bucket("Data Scientist"), "data science")
        self.assertIsNone(cs.title_bucket("Store Manager"))


class CapPerCompany(unittest.TestCase):
    def test_at_most_cap_per_company_first_then_the_rest(self):
        order = rm.cap_per_company([0, 1, 2, 3, 4, 5], ["a", "a", "a", "b", None, "a"], cap=2)
        self.assertEqual(order, [0, 1, 3, 4, 2, 5])

    def test_limit_and_listings_without_a_company(self):
        self.assertEqual(rm.cap_per_company([0, 1, 2], [None, None, None], cap=1, limit=2), [0, 1])
        self.assertEqual(rm.cap_per_company([], [], cap=2), [])


if __name__ == "__main__":
    unittest.main()
