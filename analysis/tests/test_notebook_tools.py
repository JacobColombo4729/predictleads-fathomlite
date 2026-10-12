"""The notebook checks Thread uses (src/notebook_tools.py), on a small made-up notebook. No real data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import notebook_tools as nt


def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def cell_code(table_html, tags=None):
    return {"cell_type": "code", "execution_count": 1, "metadata": {"tags": tags or []}, "source": ["# 1a A table\n"],
            "outputs": [{"output_type": "display_data", "data": {"text/html": [table_html]}, "metadata": {}}]}


class NotebookTools(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "notebooks").mkdir()
        self.path = root / "notebooks" / "Demo.ipynb"
        nb = {"cells": [
            cell_md("## 1. Results\n\n**What the data shows**\n- 908K listings, 24.7% still open, median $64.62 an hour.\n"
                    "- Old figure: 1,000,000 listings and 31.5% open. Checked in 2026, section 3."),
            cell_code("<table><tr><td>listings</td><td>908K</td></tr><tr><td>open</td><td>24.7%</td></tr>"
                      "<tr><td>pay</td><td>$64.62</td></tr></table>"),
            cell_code("<table><tr><td>secret</td><td>31.5%</td></tr></table>", tags=["data-rows"]),
        ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        self.path.write_text(json.dumps(nb), encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_stale_finds_only_numbers_missing_from_outputs(self):
        found = {num for _, num in nt.stale(self.path)}
        self.assertIn("1,000,000", found)
        self.assertIn("31.5%", found)  # only in a row-preview cell, which doesn't count
        self.assertNotIn("24.7%", found)
        self.assertNotIn("64.62", found)
        self.assertNotIn("2026", found)  # a year, not a result

    def test_outputs_skips_row_previews(self):
        text = nt.outputs(self.path, 0, 2)
        self.assertIn("908K", text)
        self.assertIn("data-rows, skipped", text)
        self.assertNotIn("secret", text)

    def test_status_counts_and_missing_reports(self):
        s = nt.status(self.path)
        self.assertIn("2/2 code cells ran", s)
        self.assertIn("missing", s)  # no reports built yet


if __name__ == "__main__":
    unittest.main()
