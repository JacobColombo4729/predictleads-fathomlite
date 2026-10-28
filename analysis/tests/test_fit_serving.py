"""A new person's résumé to ML_Fit's inputs (src/fit_serving.py), on made-up résumés in the app's profile format."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import career_switch as cs
from src import fit_model as fm
from src import fit_serving as fs
from src.fit_eval import ROLES

AS_OF = "2026-10"


def entry(kind, title, dates, rest="Did the work."):
    return {"kind": kind, "title": title, "organization": None, "description": f"{dates}. {rest}" if dates else rest}


TEACHER = [
    entry("job", "Middle School Teacher", "Aug 2015 - Jun 2023"),
    entry("job", "Summer Camp Counselor Intern", "Jun 2014 - Aug 2014"),
    entry("project", "Grade Tracker", "2024", "Built a spreadsheet."),
    entry("education", "Data Analytics Bootcamp", "Jan 2024 - Aug 2024"),
    entry("education", "B.A. English", "2011 - 2015"),
]
SUPPORT = [
    entry("job", "Retail Associate", "03/2015 - 12/2018"),
    entry("job", "IT Support Specialist", "Jan 2019 - Jun 2021"),
    entry("job", "Senior Help Desk Technician", "Jul 2021 - Present"),
    entry("education", "Associate of Applied Science, Information Technology", "2017 - 2019"),
]


class EntryDates(unittest.TestCase):
    def test_reads_the_readers_date_formats(self):
        self.assertEqual(fs.entry_dates("Jan 2020 - Present. Did things."), ("Jan 2020", "Present"))
        self.assertEqual(fs.entry_dates("Sept. 2015 – June 2018. x"), ("Sept 2015", "June 2018"))
        self.assertEqual(fs.entry_dates("06 / 2015 to 07/2016. x"), ("06/2015", "07/2016"))
        self.assertEqual(fs.entry_dates("2015. x"), ("2015", None))
        self.assertEqual(fs.entry_dates("Led a team of five."), (None, None))
        self.assertEqual(fs.entry_dates(None), (None, None))
        for start, end in [fs.entry_dates("Sept. 2015 – June 2018."), fs.entry_dates("06 / 2015 to 07/2016.")]:
            self.assertIsNot(cs.parse_month(start, pd.Period(AS_OF, "M")), pd.NaT)
            self.assertIsNot(cs.parse_month(end, pd.Period(AS_OF, "M")), pd.NaT)

    def test_projects_are_left_out(self):
        jobs, edu = fs.profile_tables(TEACHER)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(edu), 2)
        self.assertNotIn("Grade Tracker", list(jobs.title) + list(edu.program))


class PersonFeatures(unittest.TestCase):
    def test_a_career_changer_is_described_at_a_switch_now(self):
        row = fs.person_features(TEACHER, as_of=AS_OF).iloc[0]
        self.assertEqual(row.case, "career change")
        self.assertEqual(row.origin, "other-field job")
        self.assertEqual(row.background_area, "education")
        self.assertTrue(row.retrained)
        self.assertEqual(row.retrain_field, "data science and analytics")
        self.assertEqual(row.years_before, round(((2023 - 2015) * 12 + (6 - 8)) / 12, 1))
        self.assertEqual(row.n_jobs_before, 2)  # the internship is a dated job, though not a background
        self.assertEqual(row.study_level_before, 3)  # a bootcamp counts as a certificate
        self.assertIsNone(row.from_bucket)
        self.assertNotIn("Intern", row.text)  # internships stay out of the text, as in Profiles 5c
        self.assertNotIn("Grade Tracker", row.text)
        self.assertIn("Middle School Teacher", row.text)

    def test_someone_in_tech_moves_from_their_current_role(self):
        row = fs.person_features(SUPPORT, as_of=AS_OF).iloc[0]
        self.assertEqual(row.case, "within tech")
        self.assertEqual(row.from_bucket, "it support")
        self.assertEqual(row.background_area, "tech: it support")
        self.assertEqual(row.from_level, "senior")
        self.assertEqual(row.years_before, round((29 + 63) / 12, 1))  # both it-support jobs: Jan 2019-Jun 2021, Jul 2021-now

    def test_undated_entries_are_counted_not_placed(self):
        row = fs.person_features(TEACHER + [entry("job", "Tutor", None)], as_of=AS_OF).iloc[0]
        self.assertEqual(row.undated_entries, 1)
        self.assertNotIn("Tutor", row.text)

    def test_an_empty_profile_still_gives_a_row(self):
        row = fs.person_features([], as_of=AS_OF).iloc[0]
        self.assertEqual(row.case, "career change")
        self.assertEqual(row.text, "")
        self.assertEqual(row.background_area, "unknown")

    def test_the_trained_layers_inputs_accept_it(self):
        people = fs.person_features(TEACHER, as_of=AS_OF)
        X = fm.features(people, {"tf-idf": np.zeros((1, len(ROLES)))})
        for col in fm.CATEGORICAL + fm.FLAGS + fm.NUMERIC:
            self.assertIn(col, X)
        self.assertFalse(X[fm.CATEGORICAL].isna().any().any())


class SameAsProfiles(unittest.TestCase):
    """A Profiles person who switched at month M gets the same features from find_switches as an app user with the
    same résumé (minus the tech job) gets from person_features as of M."""

    def check(self, entries, switch_title, switch_month):
        as_of = pd.Period(switch_month, "M")
        jobs, edu = fs.profile_tables(entries)
        switch = pd.DataFrame([{"person_id": fs.USER, "title": switch_title, "start_date": switch_month.replace("-", "/")[5:] + "/" + switch_month[:4],
                                "end_date": "Present"}])
        rules = {**cs.RULES, "as_of": as_of}
        table = cs.find_switches(cs.tag_jobs(pd.concat([jobs, switch], ignore_index=True), rules), edu, rules)
        s = table.loc[fs.USER]
        expected_area = cs.background_area(s.source_title if s.origin_type == "job" else s.source_study)
        tl = cs.person_timeline(cs.tag_jobs(jobs, rules), cs.prepare_education(edu, rules).get(fs.USER), s.switch_month, rules)
        expected_text = " | ".join(tl[(tl.key < s.switch_month.ordinal) & tl.label.ne("skip")].text.astype(str))
        row = fs.person_features(entries, as_of=switch_month).iloc[0]
        self.assertEqual(row.origin, s.origin)
        self.assertEqual(row.background_area, expected_area)
        self.assertEqual(row.retrained, bool(s.retrained))
        self.assertEqual(row.retrain_field, s.retrain_field)
        self.assertEqual(row.years_before, round(s.background_months / 12, 1))
        self.assertEqual(row.n_jobs_before, s.n_jobs_before)
        self.assertEqual(row.study_undated, bool(s.study_undated))
        self.assertTrue((pd.isna(row.study_level_before) and pd.isna(s.study_level_before)) or row.study_level_before == s.study_level_before)
        self.assertEqual(row.text, expected_text)

    def test_teacher_who_retrained(self):
        self.check(TEACHER, "Data Analyst", "2025-03")

    def test_retail_worker_without_training(self):
        self.check([entry("job", "Store Manager", "2012 - 2020"), entry("education", "B.S. Biology", "2008 - 2012")],
                   "Junior Software Developer", "2021-05")

    def test_undated_late_program(self):
        self.check([entry("job", "Registered Nurse", "05/2010 - 02/2022"), entry("education", "Certificate in Data Analytics", None)],
                   "Data Analyst", "2023-01")


if __name__ == "__main__":
    unittest.main()
