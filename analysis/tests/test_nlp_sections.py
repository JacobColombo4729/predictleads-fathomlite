"""src/nlp_sections.py: what a section of a listing is about, from its heading's meaning. No real data, no model.

The encoder is injected, so these tests use a stub that places texts in a space we control. That keeps the test
about the *mechanics* - closest anchor wins, the margin guards a close call, a section carries down to the blocks
under it - and not about how good BGE is, which is not something a unit test can judge. The quality of the real
labels is checked in the notebook, where every distinct heading above a frequency is printed with its label for a
reader to look over.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from src import nlp_sections as ns

# A stub space: three unit axes, one per class. A text is placed by the word it starts with, so the test author
# decides what is "close" to what, instead of a sentence model deciding it.
AXES = {"required": [1.0, 0.0, 0.0], "preferred": [0.0, 1.0, 0.0], "other": [0.0, 0.0, 1.0]}
PLACED = {
    "Minimum Qualifications": "required",
    "What You'll Need": "required",
    "Nice to have": "preferred",
    "Bonus Points": "preferred",
    "Responsibilities": "other",
    "About Us": "other",
}


def stub_encode(texts):
    """Anchor sentences land on their own class axis; the placed headings land on theirs; anything else lands exactly
    between required and preferred, which is the close call the margin is supposed to catch."""
    out = []
    for text in texts:
        owner = next((name for name, sentences in ns.ANCHORS.items() if text in sentences), None)
        owner = owner or PLACED.get(text)
        if owner:
            out.append(AXES[owner])
        else:
            out.append([np.sqrt(0.5), np.sqrt(0.5), 0.0])   # equally close to two classes
    return np.asarray(out, dtype="float32")


class Scores(unittest.TestCase):
    def test_a_text_scores_its_closest_anchor_in_each_class(self):
        table = ns.scores(["Minimum Qualifications"], stub_encode)
        self.assertEqual(list(table.columns), list(ns.CLASSES))
        self.assertAlmostEqual(table.loc["Minimum Qualifications", "required"], 1.0, places=5)
        self.assertAlmostEqual(table.loc["Minimum Qualifications", "preferred"], 0.0, places=5)

    def test_anchors_are_kept_separate_not_averaged(self):
        """Two shapes of one class must both score 1.0; averaging the anchors would put neither at 1.0."""
        owners, vecs = ns.anchor_matrix(stub_encode)
        self.assertEqual(len(owners), sum(len(v) for v in ns.ANCHORS.values()))
        self.assertEqual(vecs.shape[0], len(owners))


class Label(unittest.TestCase):
    def test_each_heading_joins_its_closest_class(self):
        got = ns.label(list(PLACED), stub_encode).set_index("text").section
        for heading, expected in PLACED.items():
            self.assertEqual(got[heading], expected, heading)

    def test_a_close_call_is_unclear_rather_than_guessed(self):
        got = ns.label(["Your Toolkit"], stub_encode).iloc[0]
        self.assertEqual(got.section, ns.UNCLEAR)
        self.assertLess(got.margin, ns.MIN_MARGIN)

    def test_the_margin_is_the_gap_to_the_runner_up(self):
        got = ns.label(["Minimum Qualifications"], stub_encode).iloc[0]
        self.assertAlmostEqual(got.margin, 1.0, places=5)

    def test_a_lower_threshold_lets_a_close_call_through(self):
        got = ns.label(["Your Toolkit"], stub_encode, min_margin=-1).iloc[0]
        self.assertIn(got.section, ns.CLASSES)

    def test_every_text_comes_back_once_in_order(self):
        texts = ["Nice to have", "Responsibilities", "Nice to have"]
        got = ns.label(texts, stub_encode)
        self.assertEqual(got.text.tolist(), texts)


class Propagate(unittest.TestCase):
    BLOCKS = pd.DataFrame([
        ("job1", "text", "We are hiring at Northwind."),
        ("job1", "heading", "Minimum Qualifications"),
        ("job1", "item", "3 years of experience with Python"),
        ("job1", "item", "A degree or equivalent experience"),
        ("job1", "heading", "Nice to have"),
        ("job1", "item", "Tableau"),
        ("job1", "heading", "Responsibilities"),
        ("job1", "item", "Build dashboards"),
        ("job2", "item", "5 years of experience"),
    ], columns=["id", "kind", "text"])
    SECTIONS = {"Minimum Qualifications": "required", "Nice to have": "preferred", "Responsibilities": "other"}

    def test_a_block_takes_the_section_of_the_heading_above_it(self):
        got = ns.propagate(self.BLOCKS, self.SECTIONS).tolist()
        self.assertEqual(got[1:4], ["required", "required", "required"])
        self.assertEqual(got[4:6], ["preferred", "preferred"])
        self.assertEqual(got[6:8], ["other", "other"])

    def test_text_above_the_first_heading_is_left_unknown(self):
        """Nothing has been said about it yet, so it is not assumed to be anything."""
        self.assertTrue(pd.isna(ns.propagate(self.BLOCKS, self.SECTIONS).iloc[0]))

    def test_a_section_does_not_leak_between_listings(self):
        """job2 has no heading at all, so its bullet stays unknown even though job1 ended in a section."""
        self.assertTrue(pd.isna(ns.propagate(self.BLOCKS, self.SECTIONS).iloc[-1]))

    def test_an_unlabelled_heading_does_not_change_the_section(self):
        """A heading the classifier could not place leaves the section as it was, rather than resetting it."""
        got = ns.propagate(self.BLOCKS, {k: v for k, v in self.SECTIONS.items() if k != "Nice to have"})
        self.assertEqual(got.tolist()[4:6], ["required", "required"])


if __name__ == "__main__":
    unittest.main()
