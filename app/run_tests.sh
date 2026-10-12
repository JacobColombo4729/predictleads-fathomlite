#!/usr/bin/env bash
# macOS / Linux version of run_tests.ps1. Runs the project's tests from one place.
#   ./run_tests.sh              live Python tests (tests/, backend/tests/ and analysis/tests/)
#   ./run_tests.sh --frontend   also type-check and build the React app
set -uo pipefail

frontend=0
for arg in "$@"; do
  case "$arg" in
    --frontend) frontend=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="$(dirname "$root")/.venv/bin/python"  # the repository's .venv, at the top
export PYTHONPATH="$root/code"
cd "$root"
failed=0

run_step() {
  local name="$1"; shift
  printf '\n\033[36m=== %s ===\033[0m\n' "$name"
  if ! "$@"; then printf '\033[31mFAILED: %s\033[0m\n' "$name"; failed=1; fi
}

run_step "tests/ (engine, explanations, BGE text)" "$py" -m unittest discover -s tests -q
run_step "backend/tests/ (API, ranker, feedback, resume reader)" "$py" -m unittest discover -s backend/tests -t . -q
run_step "analysis/tests/ (role discovery evaluation)" "$py" -m unittest discover -s ../analysis/tests -t ../analysis -q
if [[ $frontend -eq 1 ]]; then
  run_step "frontend type-check" bash -c 'cd frontend && npx tsc -b'
  run_step "frontend build" bash -c 'cd frontend && npm run build --silent'
fi

if [[ $failed -ne 0 ]]; then printf '\n\033[31mSome steps failed.\033[0m\n'; exit 1; fi
printf '\n\033[32mAll steps passed.\033[0m\n'
