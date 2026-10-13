"""Run log (src/run_log.py) on made-up values. No real data."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import run_log as rl


class RunLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "runs" / "run_log.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_appends_rows_and_widens_header(self):
        rl.log_run("ML_Fit", params={"model": "logreg"}, seeds={"random_state": 42},
                   metrics={"cv_mean": np.float64(0.81)}, path=self.path)
        rl.log_run("ML_Fit", metrics={"cv_mean": 0.8, "cv_std": 0.02}, path=self.path)
        rows = rl.read_log(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["seed_random_state"], "42")
        self.assertEqual(rows[0]["metric_cv_std"], "")
        self.assertEqual(rows[1]["metric_cv_std"], "0.02")

    def test_rejects_free_text(self):
        long = "We are looking for a data analyst to join our growing team and build dashboards"
        for bad in (long, "two\nlines", pd.DataFrame({"a": [1]}), pd.Series([1, 2]), [1, 2], {"a": 1}):
            with self.assertRaises(ValueError):
                rl.log_run("x", params={"p": bad}, path=self.path)
            with self.assertRaises(ValueError):
                rl.log_run("x", metrics={"m": bad}, path=self.path)
        self.assertFalse(self.path.exists())

    def test_default_path_under_data_dir_and_ignored(self):
        from fetch import paths
        p = rl.default_path()
        self.assertEqual(paths.DATA_DIR, p.parents[1])
        if paths.DATA_DIR.is_relative_to(paths.REPO_ROOT):
            r = subprocess.run(["git", "check-ignore", "-q", str(p.relative_to(paths.REPO_ROOT))],
                               cwd=paths.REPO_ROOT)
            self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
