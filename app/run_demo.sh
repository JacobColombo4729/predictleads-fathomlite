#!/usr/bin/env bash
# macOS / Linux version of run_demo.ps1: starts the whole demo with every OpenAI feature off.
# Runs the backend (run_demo_backend.sh --no-ai) and the frontend in this terminal, waits for the backend to load
# (about 3.5 minutes, about 2.5 GB of memory; ~20 s with --preview), then opens http://127.0.0.1:5173.
# Ctrl+C stops both. Needs what run_demo_backend.sh needs and Node.js for the frontend.
# The AI features (Magic fill, the AI gap analysis, Ask the market's typed questions) show as off; everything else works.
#   ./app/run_demo.sh                  the full ranker (TF-IDF + BGE)
#   ./app/run_demo.sh --ranker tfidf   the lighter TF-IDF-only ranker
#   ./app/run_demo.sh --preview        a light preview for a laptop (see run_demo_backend.sh)
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    --port) echo "run_demo.sh always uses port 8000 (the frontend expects it); use run_demo_backend.sh for another port." >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url="http://127.0.0.1:5173"

if curl -fs --max-time 2 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
  echo "A backend is already running on port 8000. Stop it first, so this demo starts with the AI features off." >&2
  exit 1
fi

if [[ ! -d "$root/frontend/node_modules" ]]; then
  echo "Installing the frontend's packages (first time only)..."
  (cd "$root/frontend" && npm install)
fi

trap 'trap - EXIT; kill 0' INT TERM EXIT  # Ctrl+C or exit stops the backend and the frontend with this script

"$root/run_demo_backend.sh" --no-ai "$@" &
backend=$!
(cd "$root/frontend" && npm run dev) &

echo "Demo starting with the OpenAI features OFF. Waiting for the backend..."
for _ in $(seq 1 120); do  # up to 10 minutes
  if curl -fs --max-time 3 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    if command -v open >/dev/null 2>&1; then open "$url"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"; fi
    echo "Ready: $url (AI features off). Press Ctrl+C to stop."
    wait
    exit 0
  fi
  if ! kill -0 "$backend" 2>/dev/null; then
    echo "The backend stopped: see the error above." >&2
    exit 1
  fi
  sleep 5
done
echo "The backend didn't come up within 10 minutes: see the messages above." >&2
exit 1
