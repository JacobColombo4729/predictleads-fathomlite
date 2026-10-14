"""The data guard and the audit (safety/) on made-up settings, notebooks and transcripts. No real data.

Run from the repository root: python -m unittest discover -s safety/tests -t .
"""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from safety import audit, rules

ROOT = Path(__file__).resolve().parents[2]
S3 = "s3" + "://made-up-bucket/exports"  # split so the hard-coding test doesn't flag this made-up location


class Folder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "made_up.env").write_text(f"S3_PATH={S3}\nAWS_PROFILE=made-up-profile\nAWS_REGION=us-east-2\n"
                                              "PREDICTLEADS_ASSESSMENT=openai\n", encoding="utf-8")
        self.secrets = rules.load_secrets(self.dir / "made_up.env")

    def tearDown(self):
        self.tmp.cleanup()


class Guard(Folder):
    def blocked(self, name, tool_input):
        return rules.check_tool(name, tool_input, str(self.dir), self.secrets)

    def test_secrets_are_the_location_and_bucket_not_the_profile_or_region(self):
        self.assertEqual(self.secrets, {S3, "made-up-bucket"})

    def test_reading_data_or_env_is_blocked(self):
        for cmd in ["cat data/clean/table.parquet", "ls data/clean", "head -5 ../data/raw/x.jsonl.zst", "type .env",
                    "Get-Content C:\\repo\\.env", "cat ~/.aws/credentials", "grep -r foo pretrained/bge/"]:
            self.assertTrue(self.blocked("Bash", {"command": cmd}), cmd)
        for name, inp in [("Read", {"file_path": "C:\\repo\\data\\clean\\x.pkl"}), ("Grep", {"pattern": "a", "path": "data/"}),
                          ("Glob", {"pattern": "data/**/*.parquet"}), ("Read", {"file_path": "/repo/.env"})]:
            self.assertTrue(self.blocked(name, inp), (name, inp))

    def test_existence_checks_code_searches_and_git_messages_are_allowed(self):
        for cmd in ["test -e data/clean/fit_scores.parquet && echo yes", "Test-Path data\\clean", "cat .env.example",
                    "grep -rn 'fit_scores.parquet' analysis/src", "python -m unittest discover -s tests",
                    "git add JACOB.md && git commit -m \"$(cat <<'EOF'\nML_Fit saves data/clean/fit_scores.parquet\nEOF\n)\"",
                    "gh pr create --title t --body \"reads data/clean/x.pkl\""]:
            self.assertEqual(self.blocked("Bash", {"command": cmd}), [], cmd)

    def test_prose_loops_counts_and_exclusions_are_not_reads(self):
        allowed = [("Bash", {"command": "for p in .venv .env pretrained/bge data/matching/x SETUP.md; do test -e \"$p\" && echo ok; done"}),
                   ("Bash", {"command": "python - <<'EOF'\nline = ('- saves 10 tables to `data/external/onet/`: occupations')\n"
                                        "nb = md(\"\"\"Needs `OPENAI_API_KEY` in .env and data/clean/x.pkl\"\"\")\nEOF"}),
                   ("Bash", {"command": "Remove-Item -Confirm:$false data\\clean\\old.parquet"}),
                   ("Grep", {"pattern": "x", "path": ".", "glob": "!data/**", "output_mode": "content"})]
        for name, inp in allowed:
            self.assertEqual(self.blocked(name, inp), [], inp)
        for cmd in ["python - <<'EOF'\nprint(open('data/clean/x.csv').read())\nEOF", "echo $(cat data/raw/x.csv)",
                    "for f in data/clean/*; do cat \"$f\"; done", "cd data/clean"]:
            self.assertTrue(self.blocked("Bash", {"command": cmd}), cmd)

    def test_printing_rows_or_settings_is_blocked(self):
        for cmd in ["python - <<'EOF'\nimport pandas as pd\nfrom src.eda_helpers import CLEAN_DIR\n"
                    "df = pd.read_parquet(CLEAN_DIR / 'x.parquet')\nprint(df.head())\nEOF", "printenv", "echo $S3_PATH",
                    "aws s3 ls", "Get-ChildItem env:"]:
            self.assertTrue(self.blocked("Bash", {"command": cmd}), cmd)
        shapes = "python - <<'EOF'\nimport pandas as pd\ndf = pd.read_parquet(CLEAN_DIR / 'x.parquet')\nprint(df.shape, df.dtypes)\nEOF"
        self.assertEqual(self.blocked("Bash", {"command": shapes}), [])

    def test_a_data_file_outside_data_is_blocked_only_when_it_exists(self):
        (self.dir / "sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        self.assertTrue(self.blocked("Bash", {"command": "cat sample.csv"}))
        self.assertEqual(self.blocked("Bash", {"command": "cat missing.csv"}), [])

    def test_notebooks_with_row_previews_are_read_only_through_notebook_tools(self):
        cell = {"cell_type": "code", "metadata": {"tags": ["data-rows"]}, "outputs": [{"output_type": "stream", "text": "x"}], "source": []}
        (self.dir / "with_rows.ipynb").write_text(json.dumps({"cells": [cell]}), encoding="utf-8")
        (self.dir / "cleared.ipynb").write_text(json.dumps({"cells": [dict(cell, outputs=[])]}), encoding="utf-8")
        self.assertTrue(self.blocked("Read", {"file_path": str(self.dir / "with_rows.ipynb")}))
        self.assertTrue(self.blocked("Bash", {"command": "grep -n foo with_rows.ipynb"}))
        self.assertEqual(self.blocked("Bash", {"command": "grep -c foo with_rows.ipynb && git diff with_rows.ipynb"}), [])
        self.assertEqual(self.blocked("Read", {"file_path": str(self.dir / "cleared.ipynb")}), [])

    def test_outbound_calls_never_carry_secrets_paths_or_project_details(self):
        self.assertTrue(self.blocked("WebSearch", {"query": "TripleTen career changers pandas"}))
        self.assertTrue(self.blocked("WebSearch", {"query": "boto3 made-up-bucket permissions"}))
        self.assertTrue(self.blocked("WebFetch", {"url": "https://example.com", "prompt": "read " + "C:" + "\\Users\\someone\\x"}))
        self.assertTrue(self.blocked("mcp__claude_ai_Slack__slack_send_message", {"text": f"it's at {S3}"}))
        self.assertEqual(self.blocked("WebSearch", {"query": "sklearn PredefinedSplit GridSearchCV"}), [])

    def test_the_hook_blocks_with_exit_code_2_and_never_echoes_the_value(self):
        call = json.dumps({"tool_name": "Bash", "tool_input": {"command": "cat data/raw/secret_name.csv"}, "cwd": str(self.dir)})
        r = subprocess.run([sys.executable, str(ROOT / "safety" / "guard.py")], input=call, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("secret_name", r.stderr)
        ok = subprocess.run([sys.executable, str(ROOT / "safety" / "guard.py")], input="not json", capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0)  # a broken input lets the call through; the audit checks afterwards


class Audit(Folder):
    def transcript(self, records):
        project = self.dir / "projects" / "C--made-up-predictleads-fathomlite"
        project.mkdir(parents=True, exist_ok=True)
        with (project / "session-1.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return audit.transcript_files(self.dir / "projects")

    def test_finds_rule_breaks_whether_they_ran_and_secrets_the_model_saw(self):
        use = lambda i, name, inp: {"type": "assistant", "timestamp": "2026-10-01T10:00", "cwd": str(self.dir),
                                    "message": {"content": [{"type": "tool_use", "id": i, "name": name, "input": inp}]}}
        result = lambda i, text, err=False: {"type": "user", "timestamp": "2026-10-01T10:01",
                                             "message": {"content": [{"type": "tool_result", "tool_use_id": i, "content": text, "is_error": err}]}}
        files = self.transcript([use("a", "Bash", {"command": "ls data/clean"}), result("a", "denied", True),
                                 use("b", "Bash", {"command": "cat data/raw/x.csv"}), result("b", "rows..."),
                                 use("c", "Bash", {"command": "git status"}), result("c", "listing made-up-bucket/exports"),
                                 {"type": "file-history-snapshot", "content": S3}])  # not sent to the model
        found = audit.audit_transcripts(files, self.secrets)
        outcomes = {(f[0], f[-1]) for f in found}
        self.assertIn(("access", "blocked or failed"), outcomes)
        self.assertIn(("access", "RAN"), outcomes)
        self.assertEqual(sum(f[0] == "exposure" for f in found), 1)
        self.assertFalse(any("made-up" in " ".join(map(str, f)) for f in found))  # never the value itself

    def test_fingerprints_find_data_identifiers_in_tool_results(self):
        ids = {audit._hash("job-7f3a9c21e4")}
        files = self.transcript([{"type": "user", "timestamp": "2026-10-02",
                                  "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "row job-7f3a9c21e4 ..."}]}}])
        found = audit.audit_transcripts(files, set(), ids)
        self.assertEqual([f[0] for f in found], ["exposure"])
        self.assertEqual(audit.audit_transcripts(files, set(), ids, since="2026-10-03"), [])

    def test_runs_end_to_end_and_reports_the_app_switches(self):
        self.transcript([{"type": "user", "timestamp": "2026-10-02", "message": {"content": "hello"}}])
        out = io.StringIO()
        with redirect_stdout(out):
            audit.main(["--projects-dir", str(self.dir / "projects"), "--env-file", str(self.dir / "made_up.env")])
        text = out.getvalue()
        self.assertIn("1 files", text)
        self.assertIn("PREDICTLEADS_ASSESSMENT is ON in .env", text)
        self.assertNotIn("made-up", text)


if __name__ == "__main__":
    unittest.main()
