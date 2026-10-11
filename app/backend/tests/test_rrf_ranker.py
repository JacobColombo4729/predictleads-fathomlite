"""Offline tests for the optional RRF ranker (fake encoder, tiny corpus; no model download)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import job_matching_engine as matching  # noqa: E402
from backend.rrf_ranker import RrfRankingBackend  # noqa: E402


def _job(job_id: str, title: str, description: str) -> matching.JobRecord:
    return matching.JobRecord(job_id=job_id, usable_description=True, source={
        "job_id": job_id, "title_searchable": title, "description_text": description,
        "status_derived": "unknown", "snapshot_date": "2026-09-30"})


class RrfRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = [
            _job("a", "Solutions Engineer", "customer demos and technical discovery for software"),
            _job("b", "Data Scientist", "build machine learning models in python"),
            _job("c", "Forklift Operator", "move pallets in a warehouse"),
        ]
        self.matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        np.save(self.dir / "bge_embeddings.npy", self.matrix)
        ids_hash = hashlib.sha256("\0".join(j.job_id for j in self.jobs).encode()).hexdigest()
        (self.dir / "bge_embeddings_manifest.json").write_text(json.dumps({"ordered_job_ids_sha256": ids_hash}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _profile(self) -> matching.CandidateProfile:
        return matching.CandidateProfile(profile_id="p", experience=("customer demos",), evidenced_skills=(),
                                         desired_roles=(), preferences={})

    def test_fuses_both_signals_and_respects_limit(self) -> None:
        backend = RrfRankingBackend(self.jobs, self.dir, encoder=lambda text: np.array([0, 1, 0], dtype=np.float32))
        ranked = backend.rank(self._profile(), 2, [0, 1, 2])
        self.assertEqual(len(ranked), 2)
        self.assertEqual({row["job_id"] for row in ranked}, {"a", "b"})  # a from TF-IDF, b from the embedding
        self.assertEqual([row["rank"] for row in ranked], [1, 2])
        self.assertTrue(all(row["score"] > 0 for row in ranked))
        self.assertEqual(backend.score_type, "reciprocal_rank_fusion")

    def test_rejects_mismatched_row_order(self) -> None:
        with self.assertRaises(ValueError):
            RrfRankingBackend(list(reversed(self.jobs)), self.dir, encoder=lambda text: np.zeros(3, dtype=np.float32))

    def test_empty_candidate_set(self) -> None:
        backend = RrfRankingBackend(self.jobs, self.dir, encoder=lambda text: np.zeros(3, dtype=np.float32))
        self.assertEqual(backend.rank(self._profile(), 5, []), [])


if __name__ == "__main__":
    unittest.main()
