"""fetch.get_onet on a made-up zip in a temporary folder: nothing is downloaded, no real data."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, so `fetch` imports
from fetch import get_onet as go

HEADER = "O*NET-SOC Code\tTitle\tDescription\n"


def fake_zip(path, names, extra=()):
    """A zip shaped like the O*NET release: every file inside db_31_0_text/, one made-up line each."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("db_31_0_text/", "")
        for name in [*names, *extra]:
            z.writestr(name if name.startswith(("/", "..")) or "/" in name else f"db_31_0_text/{name}", HEADER)


class GetOnetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "external" / "onet"
        patches = [mock.patch.object(go, "ONET_DIR", self.dir), mock.patch.object(go, "DATA_DIR", Path(self.tmp.name)),
                   mock.patch.object(go, "confirm_space", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def fake_download(self, names, extra=()):
        def download(target, yes=False):
            self.calls += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            fake_zip(target, names, extra)
        return mock.patch.object(go, "download", download)

    def test_unpacks_only_the_kept_tables_and_records_the_source(self):
        with self.fake_download(go.KEEP, extra=["Work Context.txt", "Task Ratings.txt"]):
            go.ensure()
        self.assertEqual({p.name for p in self.dir.glob("*.txt")}, go.KEEP)
        source = json.loads((self.dir / "SOURCE.json").read_text())
        self.assertEqual(source["release"], go.RELEASE)
        self.assertEqual(source["license"], "CC BY 4.0")
        self.assertIn("USDOL/ETA", source["credit"])
        self.assertFalse((self.dir.parent / "onet.zip").exists())  # the zip is deleted

    def test_skips_when_already_there_and_force_downloads_again(self):
        with self.fake_download(go.KEEP):
            go.ensure()
            go.ensure()
            self.assertEqual(self.calls, 1)
            go.ensure(force=True)
            self.assertEqual(self.calls, 2)

    def test_missing_required_table_stops(self):
        with self.fake_download(go.KEEP - {"Related Occupations.txt"}):
            with self.assertRaises(SystemExit):
                go.ensure()
        self.assertFalse((self.dir / "SOURCE.json").exists())

    def test_unsafe_path_in_zip_is_refused(self):
        with self.fake_download(go.KEEP, extra=["../Occupation Data.txt"]):
            with self.assertRaises(ValueError):
                go.ensure()
        self.assertFalse((self.dir.parent / "Occupation Data.txt").exists())
        self.assertFalse((self.dir.parent / "onet.zip").exists())

    def test_url_is_the_pinned_public_release(self):
        self.assertEqual(go.URL, "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip")
        self.assertTrue(go.REQUIRED <= go.KEEP)


if __name__ == "__main__":
    unittest.main()
