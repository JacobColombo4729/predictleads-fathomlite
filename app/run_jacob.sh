#!/usr/bin/env bash
# macOS / Linux version of run_jacob.ps1: Jacob's app, the same app as run_demo.sh ranked by ML_Rank's priority queue
# (backend/ml_rank_ranker.py), with every OpenAI feature off. Runs next to the demo: backend on port 8001, frontend on
# http://127.0.0.1:5174 (the demo keeps 8000/5173). Ctrl+C stops both. Needs what run_demo.sh needs plus ML_Rank's
# hand-off (ML_Rank section 13, data/clean/ml_rank/). Both apps share the demo's session database.
#   ./app/run_jacob.sh                              ports 8001 and 5174
#   ./app/run_jacob.sh --backend-port 8002 --frontend-port 5175
set -euo pipefail

backend_port=8001
frontend_port=5174
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-port) backend_port="$2"; shift 2 ;;
    --frontend-port) frontend_port="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$root")"
url="http://127.0.0.1:$frontend_port"

if [[ ! -f "$repo/data/clean/ml_rank/listing_scores.parquet" ]]; then
  echo "ML_Rank's hand-off isn't there yet: run ML_Rank section 13 (13a) first." >&2
  exit 1
fi
if curl -fs --max-time 2 "http://127.0.0.1:$backend_port/healthz" >/dev/null 2>&1; then
  echo "A backend is already running on port $backend_port. Stop it first." >&2
  exit 1
fi
if [[ ! -d "$root/frontend/node_modules" ]]; then
  echo "Installing the frontend's packages (first time only)..."
  (cd "$root/frontend" && npm install)
fi

# Every OpenAI feature off, whatever this shell or .env says (only "openai" turns one on); Magic fill reads
# the resume on this computer instead (code/local_resume_reader.py)
export PREDICTLEADS_RESUME_EXTRACTION="local" PREDICTLEADS_ASSESSMENT="off" PREDICTLEADS_MARKET="off"
export PREDICTLEADS_BACKEND_PORT="$backend_port" PREDICTLEADS_FRONTEND_PORT="$frontend_port"
export PREDICTLEADS_API_ORIGINS="http://127.0.0.1:$frontend_port,http://localhost:$frontend_port"  # the backend accepts this page's port

trap 'trap - EXIT; kill 0' INT TERM EXIT  # Ctrl+C or exit stops the backend and the frontend with this script

"$root/run_demo_backend.sh" --no-ai --ranker mlrank --port "$backend_port" &
backend=$!
(cd "$root/frontend" && npm run dev) &

echo "Jacob's app starting (ML_Rank queue, OpenAI features OFF). Waiting for the backend..."
for _ in $(seq 1 120); do  # up to 10 minutes
  if curl -fs --max-time 3 "http://127.0.0.1:$backend_port/healthz" >/dev/null 2>&1; then
    if command -v open >/dev/null 2>&1; then open "$url"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"; fi
    echo "Ready: $url (ML_Rank queue, AI features off). Press Ctrl+C to stop."
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
