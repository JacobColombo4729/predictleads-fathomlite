"""The data rules (CLAUDE.md, "Data & AI rules") as checks on what an AI tool does, shared by the guard and the audit.

guard.py runs these before each Claude Code tool call and blocks a call that breaks them; audit.py runs the same checks
over the saved Claude Code transcripts afterwards, and also looks for the real secrets and data identifiers in
everything the model was sent. Every check returns reasons only: a reason never contains the matched value.
Standard library only, so the guard starts fast with any Python.
"""
import glob
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

# Paths Claude never reads (CLAUDE.md): data/, pretrained/ (at any depth), .env (not .env.example), ~/.aws
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
PROTECTED_DIR = re.compile(r"(?<![\w.-])(?:[\w.~:$%{}-]*/)*(?:data|pretrained)/(?=[\w.*?\[{-])")
ENV_FILE_REF = re.compile(r"(?<![\w-])\.env(?:\.(?!example\b)[\w-]+)?(?![\w.-])")
AWS_DIR = re.compile(r"(?:[/~]|\$HOME|%USERPROFILE%)/?\.aws\b|(?<![\w.])\.aws/")
DATA_EXT = re.compile(r"[^\s'\"=<>|;&()]*\.(?:parquet|pkl|pickle|joblib|csv|tsv|jsonl|zst|xlsx|sqlite3?|db|feather|arrow|npy|npz)\b",
                      re.IGNORECASE)
# Commands that show a file's contents (wc only counts, so it's left out)
READER = re.compile(r"(?:^|[\s;&|(`])(?:cat|head|tail|less|more|type|bat|gc|Get-Content|strings|xxd|od|hexdump|sed|awk|cut|sort|"
                    r"uniq|jq|zstd|zstdcat|zcat|unzstd|duckdb|sqlite3|grep|egrep|rg|Select-String|parquet-tools|nl|tac|"
                    r"python[\d.]*(?:\.exe)?\s+-c)\b", re.IGNORECASE)
COUNT_ONLY = re.compile(r"\s-(?:[a-zA-Z]*[clLq][a-zA-Z]*)\b|--count\b|--files-with(?:out)?-match")
# Shell commands that may name a protected path without showing what's in it: existence checks, git (data is never
# committed; nbstripout keeps notebook outputs out of diffs), messages, and deleting or making folders
HARMLESS_FIRST = re.compile(r"^\s*(?:test\s+-[edfs]\s|\[\s+-[edfs]\s|Test-Path\b|git\s+(?!diff\s+.*--no-index|show\b|cat-file\b)|gh\s|"
                            r"echo\b|printf\b|Write-(?:Host|Output)\b|rm\b|rmdir\b|Remove-Item\b|del\b|mkdir\b|New-Item\b|"
                            r"done\b|do\s+(?:test|\[)|then\b|fi\b|else\b)", re.IGNORECASE)
EXISTENCE_ONLY = HARMLESS_FIRST  # kept for older imports
# Showing the environment or the AWS settings prints the S3 location, profile and keys
ENV_DUMP = re.compile(r"^\s*(?:printenv|env|set)\s*$|\b(?:Get-ChildItem|gci|dir|ls)\s+env:|"
                      r"\$(?:env:)?\{?(?:S3_PATH|AWS_[A-Z_]+|[A-Z_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*)\b|"
                      r"environ(?:\.get\(|\[)\s*[\"'](?:S3_PATH|AWS_\w+|\w*(?:API_KEY|SECRET|TOKEN|PASSWORD)\w*)|"
                      r"^\s*aws\s+(?:configure|s3|s3api|sts)\b")
FILE_ACCESS = re.compile(r"open\(|read_\w+\(|Path\(|read_text|read_bytes|load\(|glob\(|listdir|iterdir|scandir|os\.walk|"
                         r"shutil\.|subprocess|exists\(|is_file\(|is_dir\(|stat\(")
_QUOTED = re.compile(r"\"([^\"\s]*)\"|'([^'\s]*)'")
_TRIPLE = re.compile(r'"""[\s\S]*?"""' + r"|'''[\s\S]*?'''")
# A script that loads a data table and prints rows of it (shapes, dtypes and column names are fine)
LOADS_DATA = re.compile(r"\b(?:DATA_DIR|CLEAN_DIR|RAW_DIR|PROFILES_DIR|PART2_TABLE|EXTERNAL_DIR|FLAT_DIR)\b|"
                        r"\bexternal\(|\braw_file\(|\bfetch\.paths\b")  # the project's data folders (not synthetic or temporary files)
SHOWS_ROWS = re.compile(r"\.(?:head|tail|sample|to_string|to_markdown|to_dict|tolist|iterrows|itertuples|unique)\(|"
                        r"\.values\b|\.iloc\[|\bprint\(\s*(?:df|rows?|sample|table|data)\s*[)\[]|display\(")
# Words that name the project: never in a web search or a fetched URL
PROJECT_TERMS = re.compile(r"predict\s*leads|triple\s*ten|fathom\s*lite|project[_ -]leads|retraining compass", re.IGNORECASE)
PERSONAL_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+\w|/Users/\w|/home/\w", re.IGNORECASE)
# Tools that send something out of the machine (besides the model itself)
OUTBOUND = re.compile(r"^(?:WebSearch|WebFetch|Artifact|ArtifactComments|ArtifactData|SendUserFile|PushNotification|RemoteTrigger|"
                      r"mcp__.+)$")
# The AWS profile is a name the README gives, not a credential, so it is left out
SECRET_NAMES = re.compile(r"S3|BUCKET|KEY|SECRET|TOKEN|PASSWORD|COOKIE", re.IGNORECASE)


def load_secrets(env_file=ENV_FILE):
    """The .env values that must never reach a model: the S3 location (and its bucket), keys and tokens.
    Kept in memory only; nothing here is printed."""
    secrets = set()
    try:
        lines = Path(env_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return secrets
    for line in lines:
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        value = value.split(" #")[0].strip().strip("'\"")
        if not sep or name.startswith("#") or not SECRET_NAMES.search(name) or len(value) < 4:
            continue
        secrets.add(value)
        if value.startswith("s3://"):
            bucket = value.removeprefix("s3://").split("/")[0]
            if len(bucket) >= 4:
                secrets.add(bucket)
    return secrets


def secret_hits(text, secrets):
    return any(s in text for s in secrets)


_GIT_MESSAGE = re.compile(r"((?:git|gh)\s[^\n]*?(?:-m|--message|--body|--title)\s+)(\"(?:[^\"\\]|\\.)*\"|'[^']*')", re.DOTALL)
_HEREDOC = re.compile(r"(<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n).*?\n\s*\2[ \t]*(?=\n|$|\))", re.DOTALL)


def _segments(command):
    """(text, is_script) parts of a command: shell parts split at ; && || |, and each line of a heredoc script. The text
    of git and gh messages, and triple-quoted strings (prose and cell text that scripts write), are dropped first."""
    command = _GIT_MESSAGE.sub(r"\1''", command)

    def drop_git_body(m):  # only a heredoc that starts on a git or gh line
        line = m.string[m.string.rfind("\n", 0, m.start()) + 1:m.start()]
        return m.group(1) if re.search(r"\b(?:git|gh)\s", line) else m.group(0)
    command = _TRIPLE.sub('""', _HEREDOC.sub(drop_git_body, command))
    parts, end_tag = [], None
    for line in command.split("\n"):
        if end_tag is not None:
            if line.strip().rstrip(")") == end_tag:
                end_tag = None
            elif line.strip():
                parts.append((line, True))
            continue
        tag = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        parts += [(seg, False) for seg in re.split(r"&&|\|\||;|\|", line) if seg.strip()]
        end_tag = tag.group(1) if tag else None
    return parts


def _strip_messages(command):
    """The command without git/gh message text and triple-quoted strings (for whole-command checks)."""
    return _TRIPLE.sub('""', _GIT_MESSAGE.sub(r"\1''", command))


def _protected(text):
    """Which protected location a piece of text names, or None (URLs are ignored)."""
    text = _URL.sub(" ", text.replace("\\", "/"))
    if PROTECTED_DIR.search(text):
        return "the data/ or pretrained/ folder"
    if ENV_FILE_REF.search(text):
        return "the .env file"
    if AWS_DIR.search(text):
        return "the AWS settings folder (~/.aws)"
    return None


def notebook_has_row_previews(path):
    """True if a notebook on disk has outputs in a cell tagged data-rows (row previews)."""
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any("data-rows" in c.get("metadata", {}).get("tags", []) and c.get("outputs") for c in nb.get("cells", []))


def _existing_data_files(segment, cwd):
    """Data-file names in a command segment that exist on disk (so a code search for a file name isn't flagged)."""
    found = []
    for token in DATA_EXT.findall(segment.replace("\\", "/")):
        token = token.strip("'\"")
        path = token if os.path.isabs(token) else os.path.join(cwd or ".", token)
        if glob.glob(path) if any(c in token for c in "*?[") else os.path.exists(path):
            found.append(token)
    return found


def check_command(command, cwd=None):
    """Reasons a shell command (Bash or PowerShell) breaks the data rules."""
    reasons = []
    existence_loop = bool(re.search(r"\b(?:test\s+-[edfs]|\[\s+-[edfs]|Test-Path)\b", command)) and not READER.search(command)
    for seg, is_script in _segments(command):
        where = _protected(seg)
        if is_script and seg.lstrip().startswith("#"):
            continue
        if is_script:  # a line of a heredoc script: a bare path in quotes, or a path on a line that opens files
            bare = any(_protected(a or b) for a, b in _QUOTED.findall(seg))
            if where and (bare or FILE_ACCESS.search(seg)):
                reasons.append(f"the script opens {where}")
        elif where and not (HARMLESS_FIRST.match(seg) and "$(" not in seg and "`" not in seg) and not (
                existence_loop and re.match(r"\s*for\s", seg)):
            reasons.append(f"the command names {where} (only existence checks like `test -e` are allowed)")
        if not is_script and READER.search(seg):
            if _existing_data_files(seg, cwd):
                reasons.append("the command shows the contents of a data file")
            if not COUNT_ONLY.search(seg) and not re.match(r"\s*git\s", seg):
                for nb in re.findall(r"[^\s'\"]+\.ipynb\b", seg.replace("\\", "/")):
                    if notebook_has_row_previews(nb if os.path.isabs(nb) else os.path.join(cwd or ".", nb)):
                        reasons.append("the command reads a notebook whose row previews (data-rows cells) have outputs; "
                                       "use `python -m src.notebook_tools` from analysis/")
        if not is_script and ENV_DUMP.search(seg) or is_script and "environ" in seg and ENV_DUMP.search(seg):
            reasons.append("the command shows environment settings or AWS configuration (S3 location, profile or keys)")
    code = _strip_messages(command)
    if LOADS_DATA.search(code) and SHOWS_ROWS.search(code):
        reasons.append("the script loads a data table and prints rows or values from it (shapes, dtypes and column names only)")
    return list(dict.fromkeys(reasons))


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def check_tool(name, tool_input, cwd=None, secrets=frozenset()):
    """Reasons a Claude Code tool call breaks the data rules (used before the call by the guard, after it by the audit)."""
    tool_input = tool_input or {}
    reasons = []
    if name in ("Bash", "PowerShell"):
        reasons += check_command(tool_input.get("command", ""), cwd)
    elif name in ("Read", "Edit", "Write", "NotebookEdit", "Grep", "Glob"):
        targets = [tool_input.get(k, "") for k in ("file_path", "notebook_path", "path", "glob")]
        targets = [t for t in targets if not str(t).startswith("!")]  # a Grep glob that leaves files out
        if name == "Glob":
            targets.append(tool_input.get("pattern", ""))
        for t in filter(None, targets):
            where = _protected(t) or (re.search(r"(?:^|/)(?:data|pretrained)/?$", t.replace("\\", "/")) and "the data/ or pretrained/ folder")
            if where:
                reasons.append(f"{name} on {where}")
            elif DATA_EXT.fullmatch(t.replace("\\", "/")) or (name in ("Grep", "Glob") and DATA_EXT.search(t)):
                reasons.append(f"{name} on a data file")
        nb = tool_input.get("file_path") or tool_input.get("path") or ""
        shows_text = name == "Read" or tool_input.get("output_mode") == "content"
        if shows_text and name in ("Read", "Grep") and nb.endswith(".ipynb") and notebook_has_row_previews(nb if os.path.isabs(nb) else os.path.join(cwd or ".", nb)):
            reasons.append("the notebook's row previews (data-rows cells) have outputs; read outputs with "
                           "`python -m src.notebook_tools` from analysis/")
    if OUTBOUND.match(name):
        text = "\n".join(_strings(tool_input))
        if name == "Artifact" and tool_input.get("file_path"):
            try:
                text += "\n" + Path(tool_input["file_path"]).read_text(encoding="utf-8")[:5_000_000]
            except (OSError, UnicodeDecodeError):
                pass
        if secret_hits(text, secrets):
            reasons.append(f"{name} would send a value from .env (S3 location, profile or key)")
        if re.search(r"s3://[a-z0-9][a-z0-9.-]{2,}", text):
            reasons.append(f"{name} would send an S3 location")
        if _protected(text):
            reasons.append(f"{name} would send a path into {_protected(text)}")
        if name in ("WebSearch", "WebFetch") and (PROJECT_TERMS.search(text) or PERSONAL_PATH.search(text)):
            reasons.append(f"{name} would put project details or a local path into a web request")
    return list(dict.fromkeys(reasons))
