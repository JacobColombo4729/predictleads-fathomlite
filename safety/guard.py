"""Claude Code PreToolUse hook: blocks a tool call that would break the data rules (safety/rules.py) before it runs.

Claude Code passes the call as JSON on stdin; exit code 2 blocks it and the reason on stderr goes back to Claude, which
then does it another way. Reasons never contain the matched value. If the hook itself fails, it lets the call through
(exit 0) rather than stop the session: the audit (safety/audit.py) checks afterwards.
Registered in .claude/settings.json through safety/guard.sh, which finds a Python.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety import rules  # noqa: E402


def main():
    try:
        call = json.load(sys.stdin)
        reasons = rules.check_tool(call.get("tool_name", ""), call.get("tool_input"), call.get("cwd"), rules.load_secrets())
    except Exception:  # never stop a session because the guard broke
        return 0
    if not reasons:
        return 0
    print("Blocked by the data guard (CLAUDE.md, Data & AI rules): " + "; ".join(reasons) + ". "
          "Work from code, column names, dtypes, shapes and aggregates; do it another way.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
