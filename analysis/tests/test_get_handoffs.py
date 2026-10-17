"""fetch.get_handoffs between two made-up machines and an in-memory S3: nothing is uploaded or downloaded, no real data."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, so `fetch` imports
from fetch import get_handoffs as gh

BUCKET, PREFIX = "made-up", "handoffs/" + gh.VERSION + "/"
REAL_CLEAN, REAL_HANDOFFS = gh.CLEAN_DIR, dict(gh.HANDOFFS)  # before any test points them elsewhere


class FakeS3:
    """The four S3 calls the script makes, on a dict."""
    def __init__(self):
        self.objects = {}

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def upload_file(self, filename, bucket, key):
        self.objects[key] = Path(filename).read_bytes()

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key])


class GetHandoffsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.s3 = FakeS3()
        patches = [mock.patch.object(gh, "confirm_space", lambda *a, **k: None),
                   mock.patch.object(gh, "REPO_ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.machine("jacob")

    def tearDown(self):
        self.tmp.cleanup()

    def machine(self, who):
        """Point the script at one made-up machine's data folder."""
        clean = self.root / who / "clean"
        handoffs = {name: (clean / folder.relative_to(REAL_CLEAN), required)
                    for name, (folder, required) in REAL_HANDOFFS.items()}
        for name, value in [("CLEAN_DIR", clean), ("STAGING", clean / ".handoff_staging"), ("HANDOFFS", handoffs)]:
            p = mock.patch.object(gh, name, value)
            p.start()
            self.addCleanup(p.stop)
        return handoffs

    def write_outputs(self, extra=b""):
        """Made-up notebook outputs: every required file, plus a nested one."""
        for folder, required in gh.HANDOFFS.values():
            for name in required:
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(b"made-up " + name.encode() + extra)
        (gh.HANDOFFS["combined_corpus"][0] / "sub").mkdir(exist_ok=True)
        (gh.HANDOFFS["combined_corpus"][0] / "sub" / "extra.json").write_bytes(b"{}")

    def contents(self, folder):
        return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*")
                if p.is_file() and p.name != gh.MARKER}

    def test_upload_then_download_gives_the_same_files_and_skips_on_a_rerun(self):
        self.write_outputs()
        jacob = {name: self.contents(folder) for name, (folder, _) in gh.HANDOFFS.items()}
        gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        self.machine("jesse")
        gh.download(self.s3, BUCKET, PREFIX, yes=True)
        for name, (folder, _) in gh.HANDOFFS.items():
            self.assertEqual(self.contents(folder), jacob[name])
            self.assertEqual(gh.marker_version(folder), gh.VERSION)
        self.assertFalse(gh.STAGING.exists())
        with mock.patch.object(self.s3, "download_file", side_effect=AssertionError("downloaded again")):
            gh.download(self.s3, BUCKET, PREFIX, yes=True)

    def test_upload_refuses_an_incomplete_set_and_an_existing_version(self):
        self.write_outputs()
        (gh.HANDOFFS["combined_corpus"][0] / "bge_embeddings.npy").unlink()
        with self.assertRaisesRegex(SystemExit, "bge_embeddings.npy"):
            gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        self.assertEqual(self.s3.objects, {})
        self.write_outputs()
        gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        with self.assertRaisesRegex(SystemExit, "already on S3"):
            gh.upload(self.s3, BUCKET, PREFIX, yes=True)

    def test_a_version_without_its_index_is_never_downloaded(self):
        self.write_outputs()
        gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        del self.s3.objects[PREFIX + gh.INDEX]
        self.machine("jesse")
        with self.assertRaisesRegex(SystemExit, "isn't on S3"):
            gh.download(self.s3, BUCKET, PREFIX, yes=True)

    def test_a_corrupt_file_leaves_every_folder_untouched(self):
        self.write_outputs()
        gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        self.s3.objects[PREFIX + "ml_rank/queue_config.json"] = b"changed"
        self.machine("jesse")
        with self.assertRaisesRegex(SystemExit, "doesn't match"):
            gh.download(self.s3, BUCKET, PREFIX, yes=True)
        for folder, _ in gh.HANDOFFS.values():
            self.assertFalse(folder.exists())
        self.assertFalse(gh.STAGING.exists())

    def test_the_notebooks_own_outputs_are_replaced_only_with_force(self):
        self.write_outputs()
        gh.upload(self.s3, BUCKET, PREFIX, yes=True)
        self.machine("jesse")
        self.write_outputs(extra=b" from a local run")
        with self.assertRaisesRegex(SystemExit, "--force"):
            gh.download(self.s3, BUCKET, PREFIX, yes=True)
        self.assertIsNone(gh.marker_version(gh.HANDOFFS["ml_fit"][0]))
        gh.download(self.s3, BUCKET, PREFIX, yes=True, force=True)
        self.assertEqual(gh.marker_version(gh.HANDOFFS["ml_fit"][0]), gh.VERSION)

    def test_location_needs_the_env_setting_and_appends_the_version(self):
        with mock.patch.dict(gh.os.environ, {"HANDOFF_S3_PATH": ""}):
            with self.assertRaisesRegex(SystemExit, "HANDOFF_S3_PATH"):
                gh.location()
        with mock.patch.dict(gh.os.environ, {"HANDOFF_S3_PATH": "s3:" + "//made-up/team/handoffs/"}):
            self.assertEqual(gh.location(), ("made-up", "team/handoffs/" + gh.VERSION + "/"))

    def test_the_folders_are_the_ones_the_readers_use(self):
        from combined import _paths
        from combined.role_fit import FIT_DIR
        self.assertEqual({name: folder for name, (folder, _) in REAL_HANDOFFS.items()},
                         {"ml_fit": FIT_DIR, "ml_rank": _paths.ML_RANK_DIR, "combined_corpus": _paths.CORPUS_DIR})


if __name__ == "__main__":
    unittest.main()
