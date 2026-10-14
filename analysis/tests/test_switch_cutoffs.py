"""The career-switch rules (src/career_switch.py) count only what happened before the switch: made-up timelines with
jobs and programs after the switch, which must never reach the features or the pre-switch text. No real data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src.career_switch import find_switches, person_timeline, prepare_education, tag_jobs

EXPERIENCE = pd.DataFrame([
    # a teacher who retrained and moved into data, then kept going after the switch
    ("teacher", "High School Teacher", "Jan 2012", "Dec 2018"),
    ("teacher", "Data Analyst", "Jun 2020", "Dec 2021"),
    ("teacher", "Software Developer", "Jan 2022", "Present"),
    # a data analyst who moved within tech to databases
    ("mover", "Data Analyst", "Jan 2015", "Dec 2018"),
    ("mover", "Database Administrator", "Jan 2019", "Present"),
], columns=["person_id", "title", "start_date", "end_date"])
EDUCATION = pd.DataFrame([
    ("teacher", "Bachelor of Arts in English", "Sep 2008"),
    ("teacher", "Data Analytics Certificate", "Jan 2020"),
    ("teacher", "PhD in Computer Science", "Sep 2022"),  # after the switch
    ("mover", "PhD in Database Systems", "Sep 2020"),     # after the switch
], columns=["person_id", "program", "start_date"])


def pre_switch_text(jobs, edu, row, pid):
    """As Profiles 5c builds it: entries keyed before the switch month, internships and high school left out."""
    tl = person_timeline(jobs[jobs.person_id == pid], edu.get(pid), row.switch_month)
    return " | ".join(tl[(tl.key < row.switch_month.ordinal) & tl.label.ne("skip")].text.astype(str)).lower()


class BeforeTheSwitchOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = tag_jobs(EXPERIENCE)
        cls.edu = prepare_education(EDUCATION)
        cls.out = find_switches(cls.jobs, EDUCATION)

    def test_the_switch_is_the_first_tech_job(self):
        t = self.out.loc["teacher"]
        self.assertEqual(t.target_bucket, "data science")
        self.assertEqual(str(t.switch_month), "2020-06")
        self.assertEqual(t.n_jobs_before, 1)  # the teaching job only; the later developer job doesn't count

    def test_training_is_what_came_before_the_switch(self):
        t = self.out.loc["teacher"]
        self.assertTrue(t.retrained)
        self.assertIn("data analytics", t.retrain_study)
        self.assertLess(t.study_level_before, 4)  # the PhD started after the switch

    def test_pre_switch_text_never_holds_the_switch_or_later(self):
        text = pre_switch_text(self.jobs, self.edu, self.out.loc["teacher"], "teacher")
        self.assertIn("high school teacher", text)
        for later in ("data analyst", "software developer", "phd"):
            self.assertNotIn(later, text)

    def test_within_tech_moves_use_the_later_move_as_the_switch(self):
        m = self.out.loc["mover"]
        self.assertEqual((m.from_bucket, m.target_bucket), ("data science", "database"))
        text = pre_switch_text(self.jobs, self.edu, m, "mover")
        self.assertIn("data analyst", text)  # the earlier tech role is the background
        self.assertNotIn("database administrator", text)
        self.assertNotIn("phd", text)


if __name__ == "__main__":
    unittest.main()
