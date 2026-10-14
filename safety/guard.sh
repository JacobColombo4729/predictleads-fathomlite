#!/bin/sh
# Runs safety/guard.py with a Python that works: the project's venv (Windows or Mac), else python3, python or py.
# With none, it blocks (exit 2) and says why, rather than let every call through unchecked.
d="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
for py in "$d/.venv/Scripts/python.exe" "$d/.venv/bin/python" python3 python py; do
  if "$py" -c "import sys; sys.exit(sys.version_info < (3, 8))" >/dev/null 2>&1; then
    exec "$py" "$d/safety/guard.py"
  fi
done
echo "The data guard (safety/guard.sh) found no working Python, so it blocks every checked call: set up .venv (SETUP.md)." >&2
exit 2
