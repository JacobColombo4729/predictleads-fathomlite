"""The scope corpus builder on a made-up export (synthetic listings only; no data/, no S3, no BGE).

    python -m unittest discover -s combined/tests -t .
"""
import json
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from combined import corpus

DESCRIPTION = "<p>Help customers fix printers and laptops. Explain each step clearly.</p>"


def job(job_id, title="IT Support Specialist", description=DESCRIPTION, company="c1"):
    return {"id": job_id, "type": "job_opening",
            "attributes": {"title": title, "normalized_title": title, "description": description, "status": "open",
                           "first_seen_at": "2026-09-01T00:00:00Z", "last_seen_at": "2026-09-20T00:00:00Z"},
            "relationships": {"company": {"data": {"id": company, "type": "company"}}}}


COMPANY = {"id": "c1", "type": "company", "attributes": {"company_name": "Example Co", "domain": "example.test"}}


def write_export(path, documents):
    text = "".join(json.dumps(doc) + "\n" for doc in documents).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(text))


class CorpusBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.export = self.tmp / "job_openings.jsonl.zst"
        write_export(self.export, [
            {"data": [job("a"), job("not_in_scope")], "included": [COMPANY]},
            {"data": [job("b", description=None)], "included": [COMPANY]},           # no usable description
            {"data": [job("c")], "included": [COMPANY]},                              # duplicate of "a"
            {"data": [job("d", title="Network Administrator", description="<p>Keep the office network up.</p>")],
             "included": [COMPANY]},
            "not a document",                                                         # data not an array
        ])

    def build(self, ids, limit=None):
        out = self.tmp / ("slice" if limit else "corpus")
        report = corpus.build(out, ids, local=self.export, limit=limit)
        rows = [json.loads(line) for line in (out / "jobs_prepared_corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        return report, rows

    def test_keeps_only_scope_ids_with_jesses_rules(self):
        report, rows = self.build({"a", "b", "c", "d", "missing"})
        self.assertEqual([r["job_id"] for r in rows], ["a", "d"])
        self.assertEqual(report["counts"]["excluded_no_usable_description"], 1)
        self.assertEqual(report["counts"]["collapsed_duplicate_listing"], 1)
        self.assertEqual(report["errors"], {"data_not_array": 1})
        self.assertEqual(report["ids_not_found"], 1)
        self.assertTrue(report["complete_file_scanned"])

    def test_records_are_in_the_apps_corpus_format(self):
        _, rows = self.build({"a"})
        record = rows[0]
        for field in ("job_id", "title_searchable", "description_text", "description_usable", "status_derived",
                      "company_reference", "quality_flags", "duplicate_key"):
            self.assertIn(field, record)
        self.assertNotIn("description_raw", record)
        self.assertNotIn("<p>", record["description_text"])
        self.assertEqual(record["corpus_sources"], ["combined_scope"])

    def test_limit_writes_a_partial_slice_and_says_so(self):
        report, rows = self.build({"a", "d"}, limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["stop_reason"], "limit")
        self.assertFalse(report["complete_file_scanned"])
        self.assertIsNone(report["ids_not_found"])

    def test_never_overwrites_a_corpus(self):
        self.build({"a"})
        with self.assertRaises(FileExistsError):
            corpus.build(self.tmp / "corpus", {"a"}, local=self.export)

    def test_serving_ids_are_the_open_listings(self):
        import pandas as pd

        handoff = self.tmp / "ml_rank"
        handoff.mkdir()
        pd.DataFrame({"id": [1, 2, 3], "open_at_end": [True, False, None]}).to_parquet(handoff / "listing_scores.parquet")
        self.assertEqual(corpus.serving_ids(handoff), {"1"})

    def test_serving_ids_needs_ml_ranks_handoff(self):
        with self.assertRaises(FileNotFoundError):
            corpus.serving_ids(self.tmp / "nowhere")


if __name__ == "__main__":
    unittest.main()
