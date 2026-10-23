"""assert_bulk_safe() and build()'s guard against explicit bulk mode on too little RAM (src/stream_full.py).
machine() is mocked throughout; nothing here touches real hardware or data."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/, so `src` imports
from src import stream_full as sf

LAPTOP = (8.0, 3.0, 4)  # total GB, available GB, threads: fails choose_mode()'s bar
WORKSTATION = (16.0, 10.0, 8)  # exactly at the bar


class BulkSafety(unittest.TestCase):
    def test_refuses_bulk_on_too_little_ram(self):
        with mock.patch.object(sf, "machine", return_value=LAPTOP):
            with self.assertRaises(RuntimeError):
                sf.assert_bulk_safe(force=False)

    def test_force_bypasses_the_check(self):
        with mock.patch.object(sf, "machine", return_value=LAPTOP):
            sf.assert_bulk_safe(force=True)  # must not raise

    def test_allows_bulk_on_a_capable_machine_without_force(self):
        with mock.patch.object(sf, "machine", return_value=WORKSTATION):
            sf.assert_bulk_safe(force=False)  # must not raise

    def test_build_bulk_raises_before_touching_run_bulk(self):
        with mock.patch.object(sf, "machine", return_value=LAPTOP), mock.patch.object(sf, "run_bulk") as run_bulk:
            with self.assertRaises(RuntimeError):
                sf.build(mode="bulk")
            run_bulk.assert_not_called()

    def test_build_bulk_with_force_calls_run_bulk(self):
        with mock.patch.object(sf, "machine", return_value=LAPTOP), mock.patch.object(sf, "run_bulk") as run_bulk:
            sf.build(mode="bulk", force=True)
            run_bulk.assert_called_once()

    def test_build_auto_falls_back_to_stream_without_raising(self):
        with mock.patch.object(sf, "machine", return_value=LAPTOP), mock.patch.object(sf, "run") as run:
            sf.build(mode="auto")
            self.assertEqual(run.call_count, 3)  # run(), run(rows=True), run(extra=True)


if __name__ == "__main__":
    unittest.main()
