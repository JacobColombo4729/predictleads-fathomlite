"""Audit of the AI tools against the data rules (CLAUDE.md, "Data & AI rules"). Run from the repository root:

    python -m safety.audit                 # Claude Code transcripts of this project, the repository, the app's AI calls
    python -m safety.audit --fingerprints  # also look for the data's own identifiers in everything Claude was sent
    python -m safety.audit --since 2026-10-01 --all-projects

1. Claude Code transcripts (~/.claude/projects): every session of this project (and its worktrees, the two earlier
   repositories and the agents' sandboxes), including Eve, Thread, Pace, Foreman and other subagents. A transcript holds
   everything the model was sent and every tool it ran, so it is the record to audit:
   - tool calls that broke the rules (the guard's checks, safety/rules.py), and whether they ran or were blocked;
   - outbound calls (web search and fetch, connectors, artifacts) carrying a .env value, an S3 location, a data path or
     project details;
   - the real .env values (S3 location, bucket, profile, keys) anywhere in what the model was sent or wrote;
   - with --fingerprints: the data's identifiers (listing ids from the scope table, person ids from Profiles), hashed,
     anywhere in what the model was sent. A hit means data values reached the model.
2. The repository: committed notebooks with outputs, .env values in tracked files, the hard-coding test, the guard on.
3. The app's calls to outside AI services: which files call one, what each sends, and whether its .env switch is on.

Prints locations, rules and counts only, never a matched value. Exit code 1 if anything needs attention.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety import rules  # noqa: E402

PROJECT_NAMES = ("predictleads", "project-leads", "project_leads")
SENT_TYPES = {"user", "assistant", "attachment", "system"}  # transcript records that reach the model
TOKEN = re.compile(r"[A-Za-z0-9_-]{8,64}")
# Every app file that calls an outside AI service: its switch, what it sends, and whether that includes PredictLeads data
AI_CALLS = {
    "app/backend/market_llm.py": ("PREDICTLEADS_MARKET", "the user's market question (and a company name) to OpenAI", False),
    "app/code/grounded_match_explanations.py": ("PREDICTLEADS_ASSESSMENT", "profile evidence and PredictLeads listing text to OpenAI", True),
    "app/code/resume_profile_extraction.py": ("PREDICTLEADS_RESUME_EXTRACTION", "the user's résumé text (contacts redacted) to OpenAI", False),
}
# A call that sends PredictLeads data passes only once the team has confirmed the data agreement allows it: record the
# confirmation here (who, when, where it's written down)
CONFIRMED_SENDS = {}
# What app/tests/test_no_hardcoded_data.py forbids, as one git grep pattern: S3 locations, profile and bucket arguments, AWS keys
HARD_CODED = r"s3://[a-z0-9][a-z0-9.-]{2,}|profile_name[[:space:]]*=[[:space:]]*[\"']|Bucket[[:space:]]*=[[:space:]]*[\"']|AKIA[0-9A-Z]{16}"
AI_ENDPOINT = re.compile(r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|api\.cohere\.|api\.mistral\.ai|"
                         r"api\.groq\.com|api-inference\.huggingface\.co|openai\.azure\.com|bedrock-runtime|\bfrom openai import|"
                         r"\bimport (?:openai|anthropic)\b")


def _hash(token):
    return hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()


def load_fingerprints(data_dir):
    """Hashed identifiers from the data, kept in memory only: listing ids (scope table) and person ids (Profiles)."""
    import pandas as pd
    sources = {"listing ids (scope table)": data_dir / "clean" / "scope_table.parquet",
               "person ids (Profiles)": data_dir / "profiles" / "modeling_table.pkl"}
    found, counts = set(), {}
    for what, path in sources.items():
        if not path.exists():
            counts[what] = "not there"
            continue
        values = (pd.read_parquet(path, columns=["id"]).id if path.suffix == ".parquet" else pd.read_pickle(path).index.to_series())
        keep = {v for v in values.astype(str).unique() if TOKEN.fullmatch(v) and (any(c.isdigit() for c in v) or len(v) >= 16)}
        found |= {_hash(v) for v in keep}
        counts[what] = f"{len(keep):,} distinctive ids"
    return found, counts


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def is_review(rule):
    """A command that only named a protected path: the guard blocks it, but on its own it shows nothing (often an edit of
    code that mentions data/ or an existence check), so the audit lists it for review rather than as a problem."""
    return rule.startswith(("the command names", "the script opens"))


def transcript_files(projects_dir, all_projects=False):
    dirs = [d for d in sorted(Path(projects_dir).iterdir())
            if d.is_dir() and (all_projects or any(n in d.name.lower() for n in PROJECT_NAMES))]
    return [f for d in dirs for f in sorted(d.rglob("*.jsonl"))]


def audit_transcripts(files, secrets, fingerprints=frozenset(), since=None):
    """Findings from the transcripts: (kind, session, time, tool, rule, outcome) tuples."""
    findings = []
    for f in files:
        session = f.stem[:8] if f.parent.name != "subagents" else f"{f.parent.parent.name[:8]}/{f.stem[:14]}"
        calls, errors, exposed = {}, set(), Counter()
        try:
            lines = f.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with lines:
            for line in lines:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                stamp = str(d.get("timestamp", ""))[:16]
                if since and stamp and stamp < since:
                    continue
                content = (d.get("message") or {}).get("content")
                for b in content if isinstance(content, list) else []:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        calls[b.get("id")] = (stamp, b.get("name", ""), b.get("input"), d.get("cwd"))
                    elif b.get("type") == "tool_result" and b.get("is_error"):
                        errors.add(b.get("tool_use_id"))
                if d.get("type") not in SENT_TYPES:
                    continue
                text = "\n".join(_strings(d.get("message") or d.get("attachment") or d.get("content") or ""))
                if secrets and rules.secret_hits(text, secrets):
                    exposed[("a .env value (S3 location, bucket or key)", stamp, d.get("type"))] += 1
                if fingerprints:
                    hits = sum(_hash(t) in fingerprints for t in set(TOKEN.findall(text)))
                    if hits:
                        exposed[(f"data identifiers ({hits})", stamp, d.get("type"))] += 1
        for use_id, (stamp, name, tool_input, cwd) in calls.items():
            for reason in rules.check_tool(name, tool_input, cwd, secrets):
                kind = "outbound" if rules.OUTBOUND.match(name) else "access"
                findings.append((kind, session, stamp, name, reason, "blocked or failed" if use_id in errors else "RAN"))
        for (what, stamp, rtype), n in exposed.items():
            findings.append(("exposure", session, stamp, rtype, f"{what} in what the model was sent", "RAN"))
    return findings


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def audit_repo(repo, secrets):
    """(ok, message) checks of the repository itself."""
    checks = []
    with_outputs = []
    for path in _git(repo, "ls-files", "*.ipynb").split("\n"):
        if path:
            try:
                nb = json.loads(_git(repo, "show", f"HEAD:{path}"))
                if any(c.get("outputs") for c in nb.get("cells", [])):
                    with_outputs.append(path)
            except ValueError:
                pass
    checks.append((not with_outputs, "committed notebooks have no outputs" if not with_outputs
                   else f"committed notebooks with outputs: {', '.join(with_outputs)}"))
    # The committed tree (HEAD), not the copies on disk: notebooks on disk keep their outputs until nbstripout commits them
    leaked = sorted({line.split(":", 2)[1] for line in _git(repo, "grep", "-l", "-I", "-F", *[a for s in secrets for a in ("-e", s)], "HEAD")
                     .splitlines() if line.count(":") >= 1}) if secrets else []
    checks.append((not leaked, "no .env value in any committed file" if not leaked else f".env values in committed files: {', '.join(leaked)}"))
    exclude = [":!app/tests/test_no_hardcoded_data.py", ":!safety/tests/*"]
    hard = sorted({line.split(":", 2)[1] for line in _git(repo, "grep", "-l", "-I", "-E", "-e", HARD_CODED, "HEAD", "--", ".", *exclude)
                   .splitlines() if line.count(":") >= 1})
    checks.append((not hard, "no S3 location, bucket, profile or AWS key written into committed files" if not hard
                   else f"S3 locations, profiles or AWS keys written into committed files: {', '.join(hard)}"))
    try:
        hooked = "safety/guard.sh" in (Path(repo) / ".claude" / "settings.json").read_text(encoding="utf-8")
    except OSError:
        hooked = False
    checks.append((hooked, "the data guard is on for Claude Code (.claude/settings.json)" if hooked
                   else "the data guard isn't registered in .claude/settings.json"))
    return checks


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def audit_ai_calls(repo, env_file):
    """(ok, message) for each app file that calls an outside AI service."""
    env = {}
    try:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name in {c[0] for c in AI_CALLS.values()}:
                env[name] = value.split(" #")[0].strip().strip("'\"").lower()
    except OSError:
        pass
    checks = []
    code = [p for p in _git(repo, "ls-files", "*.py", "*.ts", "*.tsx", "*.js").split("\n") if p and "/tests/" not in p and "test_" not in p
            and not p.startswith("safety/")]  # the audit names the services it looks for
    for path in code:
        try:
            calls = AI_ENDPOINT.search((Path(repo) / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if not calls:
            continue
        if path not in AI_CALLS:
            checks.append((False, f"{path} calls an outside AI service and isn't reviewed (add it to AI_CALLS in safety/audit.py)"))
            continue
        switch, sends, listing_data = AI_CALLS[path]
        scripts = [n for n in ("app/run_demo_backend.ps1", "app/run_demo_backend.sh")
                   if re.search(switch + r"\s*=\s*[\"']?openai", _read(Path(repo) / n))]
        state = ("ON in .env" if env.get(switch) == "openai" else "off in .env") + (
            f", ON by default in {' and '.join(scripts)}" if scripts else "")
        on = env.get(switch) == "openai" or bool(scripts)
        if listing_data and on and path not in CONFIRMED_SENDS:
            checks.append((False, f"{path}: sends {sends}; {switch} is {state}. Confirm the PredictLeads data agreement allows "
                                  "this (then record it in CONFIRMED_SENDS in safety/audit.py), or keep it off"))
        else:
            checks.append((True, f"{path}: sends {sends}; {switch} is {state}"))
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", help="only transcript records from this date (YYYY-MM-DD)")
    ap.add_argument("--all-projects", action="store_true", help="every Claude Code project on this machine, not just this one")
    ap.add_argument("--fingerprints", action="store_true", help="also look for the data's identifiers (reads data/, needs pandas)")
    ap.add_argument("--show", type=int, default=40, help="how many findings that ran to list (default 40)")
    ap.add_argument("--projects-dir", default=str(Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"))
    ap.add_argument("--repo", default=str(rules.REPO_ROOT))
    ap.add_argument("--env-file", default=str(rules.ENV_FILE))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    secrets = rules.load_secrets(args.env_file)
    problems = 0
    print(f"Data audit of the AI tools ({'.env found' if secrets else 'no .env values: the secret checks are skipped'})\n")

    fingerprints = set()
    if args.fingerprints:
        sys.path.insert(0, args.repo)
        from fetch.paths import DATA_DIR
        fingerprints, counts = load_fingerprints(DATA_DIR)
        print("Fingerprints: " + "; ".join(f"{k}: {v}" for k, v in counts.items()) + "\n")

    files = transcript_files(args.projects_dir, args.all_projects) if Path(args.projects_dir).is_dir() else []
    findings = audit_transcripts(files, secrets, fingerprints, args.since)
    ran = [f for f in findings if f[-1] == "RAN" and not is_review(f[4])]
    print(f"1. Claude Code transcripts: {len(files)} files")
    table = Counter((k, rule, "review" if outcome == "RAN" and is_review(rule) else outcome) for k, _, _, _, rule, outcome in findings)
    if not table:
        print("   nothing found")
    order = {"RAN": 0, "review": 1}
    for (kind, rule, outcome), n in sorted(table.items(), key=lambda kv: (order.get(kv[0][2], 2), -kv[1])):
        print(f"   {n:>5}  {kind:<9} {outcome:<17} {rule}")
    print("   (RAN: needs attention; review: a command named a protected path, usually editing code or checking it exists; "
          "blocked or failed: stopped before anything was shown)")
    if ran:
        problems += len(ran)
        print(f"\n   Calls that ran (newest first, up to {args.show}): session, time, tool, rule")
        for kind, session, stamp, tool, rule, _ in sorted(ran, key=lambda f: f[2], reverse=True)[:args.show]:
            print(f"   {session:<24} {stamp:<17} {tool:<12} {rule}")

    print("\n2. The repository")
    for ok, msg in audit_repo(args.repo, secrets):
        problems += not ok
        print(f"   {'ok  ' if ok else 'FAIL'} {msg}")

    print("\n3. The app's calls to outside AI services")
    ai = audit_ai_calls(args.repo, args.env_file)
    for ok, msg in ai or [(True, "none")]:
        problems += not ok
        print(f"   {'ok  ' if ok else 'FAIL'} {msg}")

    print(f"\n{'Nothing needs attention.' if not problems else f'{problems} item(s) need attention.'} "
          "Not covered: AI tools outside Claude Code (other assistants, chat sites), and what a person pastes by hand.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
