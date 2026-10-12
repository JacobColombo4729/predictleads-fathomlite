"""Guards the data rules across the whole repository (analysis/ and app/), not just the app.

Data locations and credentials come only from the top-level .env (S3_PATH, AWS_PROFILE) and the download scripts
(analysis/src/get_predictleads.py, get_resumes.py). This test fails if a tracked file hard-codes an S3 location, an AWS
profile or access key, or a personal folder path. It reports the file and line, never the matched text.
"""
import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES = {
    "an S3 location (use S3_PATH from .env)": re.compile(r"s3://[a-z0-9][a-z0-9.-]{2,}"),
    "an AWS profile name (use AWS_PROFILE from .env)": re.compile(r"profile_name\s*=\s*\\?[\"']"),
    "an S3 bucket name (use S3_PATH from .env)": re.compile(r"\bBucket\s*=\s*\\?[\"']"),
    "an AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
CODE_ONLY = {  # personal paths are allowed in docs as examples, not in code
    "a personal folder path (use paths relative to the repository)": re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}|/Users/[A-Za-z]|/home/[a-z]"),
}
CODE_SUFFIXES = {".py", ".ipynb", ".ps1", ".sh", ".ts", ".tsx", ".js"}
HOW_TO_FIX = """

How to fix: take the value out of the file and read it from the setup instead.
  - S3 location: S3_PATH from the top-level .env (in app/code, import BUCKET, PREFIX and KEY from prepare_jobs;
    in analysis/src, use src.pl_data.connect_s3()).
  - AWS profile: AWS_PROFILE from .env (prepare_jobs.s3_session() in the app, connect_s3() in the analysis).
  - The raw files on disk: download them once with `python -m src.get_predictleads` (from analysis/); every reader
    finds them in data/raw/. Never point code at your own folders: use REPO_ROOT / "data" or DATA_DIR.
Why: one setup for both halves, no bucket names or credentials in committed files (our data agreement), and no
duplicate downloads. Details: README step 9 and the "Getting data" rules in CLAUDE.md.
"""


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    return [REPO_ROOT / name for name in out.decode("utf-8").split("\0") if name]


class NoHardcodedDataTest(unittest.TestCase):
    def test_no_hardcoded_data_locations_or_credentials(self):
        try:
            files = tracked_files()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git is not available")
        problems = []
        for path in files:
            if path == Path(__file__).resolve() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary files
            rules = dict(RULES, **(CODE_ONLY if path.suffix in CODE_SUFFIXES else {}))
            for number, line in enumerate(text.splitlines(), 1):
                for what, pattern in rules.items():
                    if pattern.search(line):
                        problems.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number}: {what}")
        self.assertEqual(problems, [], "Hard-coded data settings found:\n" + "\n".join(problems) + HOW_TO_FIX)


if __name__ == "__main__":
    unittest.main()
