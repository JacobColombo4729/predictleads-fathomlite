"""src/nlp_requirements.py: years and degree read from the sentence's grammar. Made-up sentences, no real data.

Why this test exists: these two numbers are what `EDA_SDA_3_Lean` exists to produce, and the claim behind the
rewrite is that grammar answers the question a character pattern could only approximate. The cases below are
chosen to pin that claim where it actually bites - the negatives especially. "Founded in 1998, Northwind has 25
years of history" and "we have been in business for 30 years" are sentences the old keyword rule had to be taught
to ignore; here they are rejected because the parse says the years belong to *history* and to *business*, not to
anything an applicant could have.

The spaCy model is loaded once for the whole class: it is a statistical model, so a change in model version can
change a parse, and that is exactly what these tests are for.
"""
import sys
import unittest
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from src import nlp_requirements as nr


class Years(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = nr.pipeline()

    def years(self, text):
        return nr.years_asked(self.nlp(text))

    def test_a_plain_requirement(self):
        self.assertEqual(self.years("3+ years of experience with Python and SQL"), 3)

    def test_a_minimum(self):
        self.assertEqual(self.years("Minimum of 5 years of professional experience"), 5)

    def test_an_at_least(self):
        self.assertEqual(self.years("At least 2 years' experience in a data role"), 2)

    def test_a_range_takes_its_lower_end(self):
        """"3-5 years" asks for 3: the parse hangs the low end off the high end, not off the noun."""
        self.assertEqual(self.years("3-5 years of experience required"), 3)
        self.assertEqual(self.years("4 to 6 years of relevant experience"), 4)

    def test_a_spelled_out_number(self):
        self.assertEqual(self.years("Five years of relevant experience"), 5)

    def test_the_largest_requirement_in_the_text_wins(self):
        """Two bars stated in one piece of text: the applicant has to clear the higher one."""
        self.assertEqual(self.years("2 years of experience with SQL and 7 years of experience with Python"), 7)

    def test_company_history_is_not_a_requirement(self):
        """The case a keyword rule gets wrong: the years belong to "history", and the parse says so."""
        self.assertIsNone(self.years("Founded in 1998, Northwind has 25 years of history in the midwest"))
        self.assertIsNone(self.years("We have been in business for 30 years"))

    def test_years_of_something_else_is_not_experience(self):
        self.assertIsNone(self.years("A 3 year degree programme"))
        self.assertIsNone(self.years("This is a 2 year fixed term contract"))

    def test_a_number_that_is_not_about_years(self):
        """What the cheap screen passes and the parse has to reject: a number modifying something else entirely."""
        self.assertIsNone(self.years("We match 401k contributions up to 6%"))
        self.assertIsNone(self.years("Lead a team of 5 engineers"))

    def test_an_implausible_number_is_ignored(self):
        """Above MAX_YEARS it is the company talking about itself, whatever the grammar says."""
        self.assertIsNone(self.years("We bring 50 years of combined experience to every project"))

    def test_no_years_at_all(self):
        for text in ["Build dashboards for the sales team", "Bachelor's degree in a technical field", ""]:
            self.assertIsNone(self.years(text), text)

    def test_ner_agrees_that_there_is_a_year_span(self):
        """The cross-check the notebook reports: an independent statistical reading of the same sentence."""
        self.assertTrue(nr.date_entities(self.nlp("3+ years of experience with Python")))
        self.assertFalse(nr.date_entities(self.nlp("Build dashboards for the sales team")))


class Degree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = nr.pipeline()

    def degree(self, text):
        return nr.degree_asked(self.nlp(text))

    def test_the_level_asked_for(self):
        self.assertEqual(self.degree("Bachelor's degree in a technical field")[0], "bachelor")
        self.assertEqual(self.degree("Associate degree in information technology")[0], "associate")
        self.assertEqual(self.degree("Master's degree in statistics")[0], "master")

    def test_the_highest_level_wins(self):
        self.assertEqual(self.degree("Master's degree preferred, PhD a plus")[0], "phd")
        self.assertEqual(self.degree("Bachelor's or Master's degree in a related field")[0], "master")

    def test_a_degree_with_no_level_named(self):
        """Named but unlevelled: no level comes back, and `mentions_degree` is what keeps the fact at all."""
        level, equivalent = self.degree("A degree is required")
        self.assertIsNone(level)
        self.assertFalse(equivalent)
        self.assertTrue(nr.mentions_degree(self.nlp("A degree is required")))
        self.assertFalse(nr.mentions_degree(self.nlp("3 years of experience with Python")))

    def test_equivalent_experience_is_recognised(self):
        """"or equivalent experience" - the equivalence modifies the alternative."""
        self.assertTrue(self.degree("Bachelor's degree in computer science or equivalent experience")[1])
        self.assertTrue(self.degree("Associate degree or equivalent work experience")[1])

    def test_an_equivalence_coordinated_onto_the_degree_itself(self):
        """"High school diploma or equivalent" - no alternative is named, the degree word is the anchor."""
        self.assertTrue(self.degree("High school diploma or equivalent")[1])

    def test_in_lieu_of_a_degree(self):
        """The equivalence heads its own phrase, and the degree sits two hops down inside it."""
        self.assertTrue(self.degree("Experience in lieu of a degree will be considered")[1])

    def test_an_equivalence_that_is_not_connected_to_the_degree(self):
        """Both words are in the text and nothing in the parse joins them, so no alternative is offered. This is
        the case a phrase rule cannot tell from "degree or equivalent"."""
        self.assertFalse(self.degree("Bachelor's degree required. Equivalent roles are a plus.")[1])
        self.assertFalse(self.degree("A degree in a technical field. We offer equivalent pay to competitors.")[1])

    def test_a_degree_with_no_alternative_offered(self):
        self.assertFalse(self.degree("Bachelor's degree in a technical field")[1])
        self.assertFalse(self.degree("Master's degree required")[1])

    def test_no_degree_at_all(self):
        self.assertEqual(self.degree("3 years of experience with Python"), (None, False))
        self.assertEqual(self.degree("Build dashboards for the sales team"), (None, False))


class Screen(unittest.TestCase):
    """The screen decides what gets parsed, so the only property that matters is that it loses nothing. The first
    test takes every sentence the two classes above read a fact out of and insists the screen passes all of them:
    if a rule can find something, the screen must not have thrown it away. The rest only document that the screen
    is a coarse superset, which is what makes it cheap."""

    POSITIVES = [
        "3+ years of experience with Python and SQL", "Minimum of 5 years of professional experience",
        "At least 2 years' experience in a data role", "3-5 years of experience required",
        "4 to 6 years of relevant experience", "Five years of relevant experience",
        "2 years of experience with SQL and 7 years of experience with Python",
        "Bachelor's degree in a technical field", "Associate degree in information technology",
        "Master's degree in statistics", "Master's degree preferred, PhD a plus",
        "Bachelor's or Master's degree in a related field", "A degree is required",
        "Bachelor's degree in computer science or equivalent experience",
        "Associate degree or equivalent work experience", "High school diploma or equivalent",
        "Experience in lieu of a degree will be considered", "Masters degree preferred",
    ]

    @classmethod
    def setUpClass(cls):
        cls.nlp = nr.tokenizer()

    def screen(self, texts):
        return nr.may_state_requirement(texts, self.nlp)

    def test_nothing_a_rule_could_find_is_screened_out(self):
        got = self.screen(self.POSITIVES)
        for text, passed in zip(self.POSITIVES, got):
            self.assertTrue(passed, text)

    def test_text_with_neither_a_number_nor_a_degree_word_is_skipped(self):
        texts = ["Build dashboards for the sales team", "Collaborate with product managers and designers",
                 "You will own the reporting stack end to end", ""]
        self.assertEqual(self.screen(texts), [False] * len(texts))

    def test_the_screen_over_includes_and_says_so(self):
        """These pass the screen on a number or a degree word and are thrown out afterwards by the grammar (see
        `Years.test_a_number_that_is_not_about_years`). That division of labour is the point of the screen."""
        self.assertEqual(self.screen(["We match 401k contributions up to 6%", "Lead a team of 5 engineers"]),
                         [True, True])

    def test_one_answer_per_text_in_order(self):
        got = self.screen(["Build dashboards", "5 years of experience", "Build dashboards"])
        self.assertEqual(got, [False, True, False])


class FactsFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = nr.pipeline()

    def test_one_row_per_text_in_order(self):
        texts = ["5 years of experience", "Bachelor's degree or equivalent experience", "Build dashboards"]
        got = nr.facts_frame(texts, self.nlp)
        self.assertEqual(got.text.tolist(), texts)
        self.assertEqual(got.years.fillna(-1).tolist(), [5, -1, -1])   # Int64 holds pd.NA, not None
        self.assertEqual(got.degree_named.tolist(), [False, True, False])
        # degree_level is categorical, so it is compared as plain objects ("-" is not one of its categories)
        self.assertEqual(got.degree_level.astype(object).fillna("-").tolist(), ["-", "bachelor", "-"])
        self.assertEqual(got.degree_equivalent.tolist(), [False, True, False])

    def test_years_is_nullable_integer_not_float(self):
        """A missing requirement must stay missing, and a present one must stay a whole number of years."""
        got = nr.facts_frame(["5 years of experience", "Build dashboards"], self.nlp)
        self.assertEqual(str(got.years.dtype), "Int64")

    def test_no_texts_is_fine(self):
        got = nr.facts_frame([], self.nlp)
        self.assertEqual(len(got), 0)
        self.assertIn("degree_level", got.columns)


if __name__ == "__main__":
    unittest.main()
