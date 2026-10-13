"""EDA_SDA_3's text rules and batch pipeline (src/text_features.py) on made-up descriptions. No real data."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import text_features as tf

A = """<h2>About Acme Corp</h2><p>We build widgets for everyone and love our customers.</p>
<h2>Requirements</h2><ul><li>Bachelor's degree in Computer Science or a related field, or equivalent experience</li>
<li>3+ years of experience with Python</li><li>Active Secret clearance required</li>
<li>CompTIA Security+ certification required</li></ul>
<h2>Preferred Qualifications</h2><ul><li>5 years of experience in cloud</li><li>Master's degree preferred</li></ul>
<p>We are an equal opportunity employer and value diversity.</p><p>Up to 25% travel.</p>
<p>We are unable to sponsor visas for this role. Sign-on bonus available. This is a contract-to-hire role.</p>"""

B = """<p>Entry level role. No experience required, we will train you. Career changers and bootcamp graduates welcome.</p>
<p>Join our talent community for future opportunities. We are always accepting applications.</p>
<p>Ability to obtain a security clearance. Must be willing to work on-call rotations. Remote position.</p>"""

C = "Help desk analyst. Minimum of two years experience supporting users. Remote patient monitoring tools."


def kept(html):
    units = tf.to_lines(html)
    return [x for x, r in zip(units, tf.classify(units, set())) if not r]


class Rules(unittest.TestCase):
    def test_years_split_required_from_preferred(self):
        f = tf.features(kept(A))
        self.assertEqual((f["years_required"], f["years_preferred"], f["years_asked"]), (3, 5, 5))

    def test_years_in_words(self):
        self.assertEqual(tf.features(kept(C))["years_asked"], 2)

    def test_degree(self):
        f = tf.features(kept(A))
        self.assertTrue(f["degree_required"] and f["degree_equivalent"] and f["degree_preferred"])
        self.assertEqual(f["degree_level"], "bachelor")
        self.assertFalse(tf.features(kept(C))["degree_mentioned"])

    def test_blockers(self):
        f = tf.features(kept(A))
        for k in ("clearance_required", "cert_required", "cert_comptia", "no_sponsorship", "sign_on_bonus", "contract_to_hire"):
            self.assertTrue(f[k], k)
        self.assertEqual(f["travel_pct"], 25)

    def test_clearance_obtainable_is_not_required(self):
        f = tf.features(kept(B))
        self.assertTrue(f["clearance_obtainable"])
        self.assertFalse(f["clearance_required"])

    def test_newcomer_evergreen_and_shift(self):
        f = tf.features(kept(B))
        for k in ("entry_wording", "changer_wording", "evergreen_talent_pool", "evergreen_always_hiring", "evergreen_any",
                  "shift_or_oncall", "remote_mention"):
            self.assertTrue(f[k], k)
        self.assertNotEqual(f["years_asked"], f["years_asked"])  # NaN: no years stated

    def test_remote_patient_monitoring_is_not_remote(self):
        self.assertFalse(tf.features(kept(C))["remote_mention"])

    def test_no_description_is_unknown(self):
        self.assertEqual(tf.features([]), {})


class Cleanup(unittest.TestCase):
    def test_about_section_and_legal_line_removed(self):
        text = " ".join(kept(A))
        self.assertNotIn("Acme", text)
        self.assertNotIn("equal opportunity", text)
        self.assertIn("3+ years", text)

    def test_template_lines(self):
        shared = "Our people make the difference in everything that we proudly do together every day"
        req = "Bachelor's degree in Computer Science or a related field is required for this role"
        units = [[shared, req, f"unique line {i}"] for i in range(6)]
        templates = tf.template_keys(units, [f"c{i}" for i in range(6)])
        reasons = tf.classify(units[0], templates)
        self.assertEqual(reasons[0], "shared template text")
        self.assertEqual(reasons[1], "")  # a requirement line is never cut as template text
        self.assertEqual(tf.line_key(shared), tf.line_key(shared + "  "))  # stable keys


class Pipeline(unittest.TestCase):
    def test_build_writes_fixed_schema_in_batches(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            desc = pd.DataFrame({"id": ["a", "b", "c", "x"], "description": [A, B, C, None]})
            pq.write_table(pa.Table.from_pandas(desc, preserve_index=False), d / "desc.parquet", row_group_size=2)
            meta = pd.DataFrame(index=pd.Index(["a", "b", "c", "x"], name="id"))
            removed = tf.build(d / "desc.parquet", meta, set(), d / "f.parquet", d / "c.parquet", batch_size=1)
            f = pd.read_parquet(d / "f.parquet").set_index("id")
            self.assertEqual(len(f), 4)
            self.assertEqual(len(pd.read_parquet(d / "c.parquet")), 4)
            self.assertEqual(str(pq.read_schema(d / "f.parquet").field("clearance_required").type), "bool")
            self.assertTrue(pd.isna(f.loc["x", "clearance_required"]))  # no description: unknown, not False
            self.assertGreater(removed.get("equal opportunity and legal", 0), 0)

    def test_failed_build_leaves_no_files_and_bad_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            desc = pd.DataFrame({"id": ["a", "b", "c"], "description": [A, B, C]})
            pq.write_table(pa.Table.from_pandas(desc, preserve_index=False), d / "desc.parquet")
            meta = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="id"))

            def boom(n, k):
                raise RuntimeError("killed")

            with self.assertRaises(RuntimeError):
                tf.build(d / "desc.parquet", meta, set(), d / "f.parquet", d / "c.parquet", batch_size=1, progress=boom, workers=1)
            self.assertEqual(list(d.glob("f.parquet*")) + list(d.glob("c.parquet*")), [])  # no half-written file
            (d / "f.parquet").write_bytes(b"PAR1 not a real file")  # what a killed kernel leaves behind
            self.assertFalse(tf.cache_valid(d / "f.parquet", d / "desc.parquet"))
            tf.build(d / "desc.parquet", meta, set(), d / "f.parquet", d / "c.parquet", workers=1)
            self.assertTrue(tf.cache_valid(d / "f.parquet", d / "desc.parquet", d / "c.parquet"))

    def test_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            desc = pd.DataFrame({"id": list("abcdef"), "description": [A, B, C, None, A, B]})
            pq.write_table(pa.Table.from_pandas(desc, preserve_index=False), d / "desc.parquet")
            meta = pd.DataFrame(index=pd.Index(list("abcdef"), name="id"))
            r1 = tf.build(d / "desc.parquet", meta, set(), d / "f1.parquet", d / "c1.parquet", batch_size=2, workers=1)
            r2 = tf.build(d / "desc.parquet", meta, set(), d / "f2.parquet", d / "c2.parquet", batch_size=2, workers=2)
            self.assertEqual(r1, r2)
            self.assertTrue(pd.read_parquet(d / "f1.parquet").equals(pd.read_parquet(d / "f2.parquet")))
            self.assertTrue(pd.read_parquet(d / "c1.parquet").equals(pd.read_parquet(d / "c2.parquet")))


if __name__ == "__main__":
    unittest.main()
