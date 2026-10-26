"""src/nlp_text.py: a description's HTML turned into the blocks a reader sees. Made-up HTML only, no real data.

Why this test exists: `EDA_SDA_3_Lean` is the project's NLP notebook and holds no regular expressions (decided
2026-10-25), so the structure it reads has to come from the HTML parser instead. These tests pin the two things the
rest of the notebook depends on: the blocks come out in reading order with the tag that produced each one, and the
`kind` of a block is taken from that tag rather than from its words. A heading's *meaning* is never decided here -
`src/nlp_sections.py` does that from the heading's own embedding - so no test here asserts anything about which
words a heading contains.
"""
import sys
import unittest
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from src import nlp_text as nt

LISTING = """<h2>About Northwind Analytics</h2>
<p>Founded in 1998, Northwind serves the midwest.</p>
<h3>Requirements</h3>
<ul><li>Bachelor&#39;s degree in a technical field, or equivalent experience</li>
<li>3+ years of experience with <b>Python</b> and SQL</li></ul>
<p><strong>Nice to have</strong></p>
<ul><li>Tableau</li></ul>
<h3>Responsibilities</h3>
<ul><li>Build dashboards for the sales team</li></ul>"""


class Blocks(unittest.TestCase):
    def test_reading_order_is_kept(self):
        texts = [b.text for b in nt.blocks(LISTING)]
        self.assertEqual(texts, [
            "About Northwind Analytics",
            "Founded in 1998, Northwind serves the midwest.",
            "Requirements",
            "Bachelor's degree in a technical field, or equivalent experience",
            "3+ years of experience with Python and SQL",
            "Nice to have",
            "Tableau",
            "Responsibilities",
            "Build dashboards for the sales team",
        ])

    def test_each_block_carries_the_tag_that_produced_it(self):
        by_text = {b.text: b.tag for b in nt.blocks(LISTING)}
        self.assertEqual(by_text["About Northwind Analytics"], "h2")
        self.assertEqual(by_text["Requirements"], "h3")
        self.assertEqual(by_text["Tableau"], "li")
        self.assertEqual(by_text["Founded in 1998, Northwind serves the midwest."], "p")

    def test_a_list_does_not_collapse_into_its_ul(self):
        """The <ul> holds no text of its own; each <li> is a block, which is where requirements are written."""
        items = [b.text for b in nt.blocks(LISTING) if b.kind == "item"]
        self.assertEqual(len(items), 4)
        self.assertIn("Tableau", items)

    def test_an_inline_tag_belongs_to_the_block_around_it(self):
        """<b>Python</b> is part of its bullet, not a block of its own, and the words stay apart."""
        texts = [b.text for b in nt.blocks(LISTING)]
        self.assertIn("3+ years of experience with Python and SQL", texts)
        self.assertNotIn("Python", texts)

    def test_entities_come_back_as_characters(self):
        self.assertIn("Bachelor's degree in a technical field, or equivalent experience",
                      [b.text for b in nt.blocks(LISTING)])

    def test_a_br_breaks_a_block_in_two(self):
        got = [b.text for b in nt.blocks("<p>5 years of experience<br>Bachelor's degree</p>")]
        self.assertEqual(got, ["5 years of experience", "Bachelor's degree"])

    def test_script_style_and_comments_are_not_text(self):
        html = ("<h3>Requirements</h3><script>var years = 5;</script><style>li{color:red}</style>"
                "<!-- req id 4471 --><ul><li>2 years of experience</li></ul>")
        self.assertEqual([b.text for b in nt.blocks(html)], ["Requirements", "2 years of experience"])

    def test_nothing_in_gives_nothing_out(self):
        for empty in [None, "", "   ", "\n\n", 3.5, b"bytes"]:
            self.assertEqual(nt.blocks(empty), [], repr(empty))

    def test_plain_text_with_no_markup_becomes_one_block_per_line(self):
        plain = "Requirements:\n5 years of experience\nA degree is preferred"
        got = nt.blocks(plain)
        self.assertEqual([b.text for b in got],
                         ["Requirements:", "5 years of experience", "A degree is preferred"])
        self.assertEqual(got[0].kind, "heading")

    def test_a_table_keeps_its_cells_apart(self):
        html = "<table><tr><td>Experience</td><td>4+ years in a data role</td></tr></table>"
        self.assertEqual([b.text for b in nt.blocks(html)], ["Experience", "4+ years in a data role"])


class Kinds(unittest.TestCase):
    def kinds(self, html):
        return {b.text: b.kind for b in nt.blocks(html)}

    def test_heading_tags_are_headings(self):
        for tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.assertEqual(self.kinds(f"<{tag}>Requirements</{tag}>")["Requirements"], "heading", tag)

    def test_a_bold_paragraph_is_a_heading(self):
        """Listings that style their headings instead of tagging them - very common."""
        self.assertEqual(self.kinds("<p><strong>Nice to have</strong></p>")["Nice to have"], "heading")
        self.assertEqual(self.kinds("<div><em>Qualifications</em></div>")["Qualifications"], "heading")

    def test_a_partly_bold_paragraph_is_not_a_heading(self):
        """Emphasis inside a sentence is emphasis, not a heading."""
        html = "<p>You will need <strong>5 years</strong> of experience</p>"
        self.assertEqual(self.kinds(html)["You will need 5 years of experience"], "text")

    def test_a_block_ending_in_a_colon_is_a_heading(self):
        self.assertEqual(self.kinds("<p>Qualifications:</p>")["Qualifications:"], "heading")

    def test_bullets_and_cells_are_items(self):
        self.assertEqual(self.kinds("<ul><li>2 years of experience</li></ul>")["2 years of experience"], "item")
        self.assertEqual(self.kinds("<dl><dd>A degree</dd></dl>")["A degree"], "item")

    def test_an_ordinary_paragraph_is_text(self):
        self.assertEqual(self.kinds("<p>We are a growing team.</p>")["We are a growing team."], "text")

    def test_kind_does_not_depend_on_the_words(self):
        """The whole point of reading structure: the same sentence keeps its kind whatever it says."""
        for sentence in ["Requirements", "Tableau experience is a plus", "Build dashboards", "Airflow"]:
            self.assertEqual(self.kinds(f"<h3>{sentence}</h3>")[sentence], "heading", sentence)
            self.assertEqual(self.kinds(f"<ul><li>{sentence}</li></ul>")[sentence], "item", sentence)


class SplitLong(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = nt.sentencizer()

    def test_a_wall_of_text_becomes_its_sentences(self):
        wall = "<p>" + ("We are a growing team. " * 40) + "You must have 5 years of experience.</p>"
        got = nt.split_long(nt.blocks(wall), self.nlp)
        self.assertGreater(len(got), 1)
        self.assertIn("You must have 5 years of experience.", [b.text for b in got])

    def test_the_tag_and_kind_survive_the_split(self):
        wall = "<li>" + ("We are a growing team. " * 40) + "A degree is required.</li>"
        got = nt.split_long(nt.blocks(wall), self.nlp)
        self.assertTrue(all(b.tag == "li" and b.kind == "item" for b in got))

    def test_short_blocks_are_left_alone(self):
        given = nt.blocks(LISTING)
        self.assertEqual(nt.split_long(given, self.nlp), given)

    def test_no_blocks_is_fine(self):
        self.assertEqual(nt.split_long([], self.nlp), [])


class Rows(unittest.TestCase):
    def test_rows_number_the_blocks_in_order(self):
        rows = nt.as_rows("abc", nt.blocks(LISTING))
        self.assertEqual([r[1] for r in rows], list(range(9)))
        self.assertTrue(all(r[0] == "abc" for r in rows))
        self.assertEqual(rows[2][2:], ("h3", "heading", "Requirements"))


if __name__ == "__main__":
    unittest.main()
