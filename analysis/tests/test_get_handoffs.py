"""fetch.get_handoffs between two made-up machines in a temporary folder: made-up files, no real data."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, so `fetch` imports
from fetch import get_handoffs as gh

REAL_CLEAN, REAL_HANDOFFS = gh.CLEAN_DIR, dict(gh.HANDOFFS)  # before any test points them elsewhere


class GetHandoffsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for p in [mock.patch.object(gh, "confirm_space", lambda *a, **k: None),
                  mock.patch.object(gh, "REPO_ROOT", self.root)]:
            p.start()
            self.addCleanup(p.stop)
        self.machine("jacob")

    def tearDown(self):
        self.tmp.cleanup()

    def machine(self, who):
        """Point the script at one made-up machine's data folder."""
        data = self.root / who
        handoffs = {name: (data / "clean" / folder.relative_to(REAL_CLEAN), required)
                    for name, (folder, required) in REAL_HANDOFFS.items()}
        for name, value in [("CLEAN_DIR", data / "clean"), ("STAGING", data / "clean" / ".handoff_staging"),
                            ("ZIP_DIR", data / "handoffs"), ("HANDOFFS", handoffs)]:
            p = mock.patch.object(gh, name, value)
            p.start()
            self.addCleanup(p.stop)

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

    def packed(self):
        """Pack Jacob's made-up outputs, then switch to Jesse's machine; returns the zip and Jacob's files."""
        self.write_outputs()
        jacob = {name: self.contents(folder) for name, (folder, _) in gh.HANDOFFS.items()}
        zip_file = gh.pack(yes=True)
        self.machine("jesse")
        return zip_file, jacob

    def test_pack_then_unpack_gives_the_same_files_and_skips_on_a_rerun(self):
        zip_file, jacob = self.packed()
        gh.unpack(zip_file, yes=True)
        for name, (folder, _) in gh.HANDOFFS.items():
            self.assertEqual(self.contents(folder), jacob[name])
            self.assertEqual(gh.marker_version(folder), gh.VERSION)
        self.assertFalse(gh.STAGING.exists())
        with mock.patch.object(gh.shutil, "copyfileobj", side_effect=AssertionError("unpacked again")):
            gh.unpack(zip_file, yes=True)

    def test_unpack_finds_the_zip_in_its_standard_folder(self):
        zip_file, _ = self.packed()
        gh.ZIP_DIR.mkdir(parents=True)
        zip_file.replace(gh.zip_path())
        gh.unpack(yes=True)
        self.assertEqual(gh.marker_version(gh.HANDOFFS["ml_rank"][0]), gh.VERSION)

    def test_pack_refuses_an_incomplete_set(self):
        self.write_outputs()
        (gh.HANDOFFS["combined_corpus"][0] / "bge_embeddings.npy").unlink()
        with self.assertRaisesRegex(SystemExit, "bge_embeddings.npy"):
            gh.pack(yes=True)
        self.assertFalse(gh.zip_path().exists())

    def test_a_changed_file_leaves_every_folder_untouched(self):
        zip_file, _ = self.packed()
        with zipfile.ZipFile(zip_file) as z:
            members = {n: z.read(n) for n in z.namelist()}
        members["ml_rank/queue_config.json"] = b"changed"
        with zipfile.ZipFile(zip_file, "w") as z:
            for n, data in members.items():
                z.writestr(n, data)
        with self.assertRaisesRegex(SystemExit, "doesn't match"):
            gh.unpack(zip_file, yes=True)
        for folder, _ in gh.HANDOFFS.values():
            self.assertFalse(folder.exists())
        self.assertFalse(gh.STAGING.exists())

    def test_a_zip_without_an_index_or_with_an_unsafe_path_is_refused(self):
        bare = self.root / "bare.zip"
        with zipfile.ZipFile(bare, "w") as z:
            z.writestr("ml_rank/queue_config.json", "{}")
        with self.assertRaisesRegex(SystemExit, "no INDEX.json"):
            gh.unpack(bare, yes=True)
        unsafe = self.root / "unsafe.zip"
        index = {"version": "v1", "handoffs": {"ml_rank": {"files": {"../escape.txt": {"size": 1, "sha256": "x"}}}}}
        with zipfile.ZipFile(unsafe, "w") as z:
            z.writestr(gh.INDEX, json.dumps(index))
        with self.assertRaisesRegex(SystemExit, "unsafe path"):
            gh.unpack(unsafe, yes=True)
        self.assertFalse((gh.CLEAN_DIR / "escape.txt").exists())

    def test_the_notebooks_own_outputs_are_replaced_only_with_force(self):
        zip_file, _ = self.packed()
        self.write_outputs(extra=b" from a local run")
        with self.assertRaisesRegex(SystemExit, "--force"):
            gh.unpack(zip_file, yes=True)
        self.assertIsNone(gh.marker_version(gh.HANDOFFS["ml_fit"][0]))
        gh.unpack(zip_file, yes=True, force=True)
        self.assertEqual(gh.marker_version(gh.HANDOFFS["ml_fit"][0]), gh.VERSION)

    def test_the_folders_are_the_ones_the_readers_use(self):
        from combined import _paths
        from combined.role_fit import FIT_DIR
        self.assertEqual({name: folder for name, (folder, _) in REAL_HANDOFFS.items()},
                         {"ml_fit": FIT_DIR, "ml_rank": _paths.ML_RANK_DIR, "combined_corpus": _paths.CORPUS_DIR})


if __name__ == "__main__":
    unittest.main()
