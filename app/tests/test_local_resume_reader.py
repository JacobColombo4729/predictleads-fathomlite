"""The local resume reader (Jacob's app) on made-up resumes in common layouts. Fictional people only; no data, no OpenAI."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import local_resume_reader as reader  # noqa: E402
import resume_profile_extraction as extraction  # noqa: E402

CLASSIC = """Jane Doe
Data Analyst
Austin, TX | jane.doe@example.com | (512) 555-0100

SUMMARY
Former math teacher moving into analytics, with three years of reporting work.

EXPERIENCE
Data Analyst
Acme Corp, Austin, TX
Jan 2022 – Present
- Built dashboards in Tableau and SQL for the sales team.
- Automated weekly reports with Python, saving five hours
  a week for the team.
High School Math Teacher
Lincoln High School | Aug 2015 - Dec 2021
- Taught algebra to 150 students a year.

PROJECTS
Customer Churn Model
- Trained a logistic regression in Python on public data.

EDUCATION
B.S. Mathematics, State University, 2015
TripleTen Data Analyst Bootcamp | Jun 2021 - Dec 2021

SKILLS
Python, SQL, Tableau, Excel
"""

PIPES = """JOHN SMITH
Customer Success Manager
Work History
Globex Inc. | Sales Associate | Mar 2018 – Jun 2020
• Grew key accounts by 20% in two years.
Initech LLC — Support Specialist — 2016 - 2018
• Answered 60 support tickets a day.
Education
Associate of Arts, Lakeside Community College, May 2014
Technical Skills
Languages: Python, R | Tools: Salesforce, Zendesk
"""

COMPANY_FIRST = """Maria Garcia
Operations professional
Professional Experience
Acme Corp    Jan 2020 - Present
Product Manager
- Led the roadmap for a scheduling product used by 3,000 clinics.
Big Hospital    05/2016 - 12/2019
Operations Coordinator
- Coordinated schedules for 40 nurses across three units in Excel.
Education
Master of Business Administration
Northern University, 2016
"""

TITLE_AT = """Sam Lee
Experience:
Operations Coordinator at Big Hospital (2019 – 2021)
Coordinated schedules for 40 nurses and tracked supplies in Excel.
Full-Time Student at TripleTen (2021 – 2022)
Completed a data science program with weekly Python projects.
Skills:
Excel; scheduling; Python
"""


# As pypdf's layout mode gives a resume whose dates sit at the right margin (wide gaps, kept as four spaces)
WIDE_GAPS = """ALEX RIVERA    Denver, CO
Data Analyst
SUMMARY
Analyst with a background in retail operations and five years of inventory planning.
EXPERIENCE
Inventory Planner    Northwind Traders    Mar 2019 - Present
  • Forecast weekly demand for 400 products with Excel and SQL.
Store Supervisor    Contoso Retail    Jun 2015 - Feb 2019
  • Scheduled a team of 12 and cut overtime by a third.
EDUCATION
B.A. Economics    Mountain State University    2015
SKILLS
SQL    Excel    Power BI
"""


class FakeMatcher:
    """Stands in for ML_Fit's O*NET matcher: finds a few tool names in a line."""

    TOOLS = {"tableau": "Tableau", "python": "Python", "sql": "Structured query language SQL", "excel": "Microsoft Excel"}

    def find(self, text):
        words = set(re.findall(r"[a-z0-9]+", text.casefold()))
        return {(name, "tool", "x") for key, name in self.TOOLS.items() if key in words}


def entries(text):
    return reader.read_resume(text, matcher=FakeMatcher())["experience"]


def pairs(text):
    return [(e["kind"], e["title"], e["organization"]) for e in entries(text)]


class LocalResumeReaderTests(unittest.TestCase):
    def test_classic_layout(self) -> None:
        r = reader.read_resume(CLASSIC, matcher=FakeMatcher())
        self.assertEqual((r["name"], r["headline"], r["location"]), ("Jane Doe", "Data Analyst", "Austin, TX"))
        self.assertEqual(pairs(CLASSIC), [
            ("job", "Data Analyst", "Acme Corp"),
            ("job", "High School Math Teacher", "Lincoln High School"),
            ("project", "Customer Churn Model", None),
            ("education", "B.S. Mathematics", "State University"),
            ("education", "TripleTen Data Analyst Bootcamp", None),
        ])
        first = r["experience"][0]["description"]
        self.assertTrue(first.startswith("Jan 2022 – Present."))  # the dates open the description, as written
        self.assertIn("saving five hours a week for the team.", first)  # the wrapped bullet is joined back into one sentence
        self.assertNotIn("Taught algebra", first)

    def test_company_title_dates_on_one_line(self) -> None:
        self.assertEqual(pairs(PIPES), [
            ("job", "Sales Associate", "Globex Inc."),
            ("job", "Support Specialist", "Initech LLC"),
            ("education", "Associate of Arts", "Lakeside Community College"),
        ])
        names = [s["name"] for s in reader.read_resume(PIPES, matcher=FakeMatcher())["skills"]]
        self.assertEqual(names[:4], ["Python", "R", "Salesforce", "Zendesk"])  # "Languages:" / "Tools:" labels dropped

    def test_title_on_the_line_after_the_dates(self) -> None:
        self.assertEqual(pairs(COMPANY_FIRST), [
            ("job", "Product Manager", "Acme Corp"),
            ("job", "Operations Coordinator", "Big Hospital"),
            ("education", "Master of Business Administration", "Northern University"),
        ])

    def test_title_at_company_and_students_count_as_education(self) -> None:
        self.assertEqual(pairs(TITLE_AT), [
            ("job", "Operations Coordinator", "Big Hospital"),
            ("education", "Full-Time Student", "TripleTen"),
        ])

    def test_wide_gaps_from_pdf_layout_mode_separate_columns(self) -> None:
        r = reader.read_resume(WIDE_GAPS, matcher=FakeMatcher())
        self.assertEqual((r["name"], r["headline"], r["location"]), ("ALEX RIVERA", "Data Analyst", "Denver, CO"))
        self.assertEqual(pairs(WIDE_GAPS), [
            ("job", "Inventory Planner", "Northwind Traders"),
            ("job", "Store Supervisor", "Contoso Retail"),
            ("education", "B.A. Economics", "Mountain State University"),
        ])
        self.assertEqual([s["name"] for s in r["skills"]][:3], ["SQL", "Excel", "Power BI"])

    def test_skills_from_the_skills_section_and_from_the_entries(self) -> None:
        skills = {s["name"]: s for s in reader.read_resume(CLASSIC, matcher=FakeMatcher())["skills"]}
        self.assertEqual(skills["Tableau"]["evidence"], "used in the Data Analyst role")  # listed and used in a job
        self.assertEqual(skills["Excel"]["evidence"], "")  # listed only
        self.assertNotIn("Microsoft Excel", skills)  # already listed as "Excel": not added twice
        listed = [(s["name"], s["evidence"]) for s in reader.read_resume(TITLE_AT, matcher=FakeMatcher())["skills"]]
        self.assertEqual(listed[0], ("Excel", "used in the Operations Coordinator role"))
        found = [(s["name"], s["evidence"]) for s in reader.read_resume(COMPANY_FIRST, matcher=FakeMatcher())["skills"]]
        self.assertEqual(found, [("Microsoft Excel", "used in the Operations Coordinator role")])  # no Skills section: O*NET's name

    def test_every_value_passes_the_apps_grounding_check(self) -> None:
        class Provider:
            def generate(self, payload):
                return reader.read_resume(payload["resume_text"], matcher=FakeMatcher())

        for text in (CLASSIC, PIPES, COMPANY_FIRST, TITLE_AT + "\nExtra line to pass the minimum length of a resume text." * 3):
            result = extraction.extract_profile(text, Provider())
            self.assertEqual(result["dropped"], 0, text.splitlines()[0])
            self.assertGreater(len(result["profile"]["experience"]), 1)

    def test_contact_details_never_reach_the_reader(self) -> None:
        seen = {}

        class Provider:
            def generate(self, payload):
                seen["text"] = payload["resume_text"]
                return reader.read_resume(payload["resume_text"], matcher=FakeMatcher())

        extraction.extract_profile(CLASSIC, Provider())
        self.assertNotIn("example.com", seen["text"])
        self.assertNotIn("555-0100", seen["text"])

    def test_provider_says_it_is_local(self) -> None:
        p = reader.LocalResumeProvider()
        self.assertEqual(p.reader, "local")
        self.assertTrue(p.model.startswith("local-resume-reader"))


if __name__ == "__main__":
    unittest.main()
