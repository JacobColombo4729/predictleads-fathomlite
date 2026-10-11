"""The data folder setting: app/data by default, $PREDICTLEADS_DATA_DIR when set, used by the backend and the pipeline.

Each check runs in a fresh interpreter because the paths are fixed when the modules are imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import json, sys
sys.path.insert(0, "code")
import data_paths, rank_candidates, job_matching_engine
from backend import api, signals
print(json.dumps({
    "data_dir": str(data_paths.DATA_DIR),
    "serving_sample": str(api.DEFAULT_SAMPLE),
    "signals_root": str(signals.SIGNALS_ROOT),
    "corpus_dir": str(rank_candidates.DEFAULT_CORPUS_DIR),
    "engine_output": str(job_matching_engine.DEFAULT_OUTPUT),
}))
"""


def probe(data_dir: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PREDICTLEADS_DATA_DIR"}
    if data_dir is not None:
        env["PREDICTLEADS_DATA_DIR"] = data_dir
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


class DataPathsTest(unittest.TestCase):
    def test_default_is_app_data(self):
        paths = probe(None)
        default = PROJECT_ROOT / "data"
        self.assertEqual(Path(paths["data_dir"]), default)
        for key in ("serving_sample", "signals_root", "corpus_dir", "engine_output"):
            self.assertTrue(Path(paths[key]).is_relative_to(default), key)

    def test_setting_moves_every_data_path(self):
        with tempfile.TemporaryDirectory() as shared:
            paths = probe(shared)
            self.assertEqual(Path(paths["data_dir"]), Path(shared))
            self.assertEqual(Path(paths["serving_sample"]),
                             Path(shared) / "matching/ai_ml_serving_v1/jobs_prepared_corpus.jsonl")
            self.assertEqual(Path(paths["signals_root"]), Path(shared) / "signals")
            self.assertEqual(Path(paths["corpus_dir"]), Path(shared) / "matching/ai_ml_corpus_v1")
            self.assertEqual(Path(paths["engine_output"]), Path(shared) / "matching/job_matching_baseline")


if __name__ == "__main__":
    unittest.main()
