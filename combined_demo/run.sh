#!/usr/bin/env bash
# macOS / Linux version of run.ps1: the combined demo, the same app ranked by the combined model (combined/). Backend
# on port 8002, the page on http://127.0.0.1:5175 (Jesse's app keeps 8000/5173, Jacob's 8001/5174), with its own
# session database. Ctrl+C stops both. Needs the combined corpus, ML_Fit's hand-off (9b) and ML_Rank's (section 13).
#   ./combined_demo/run.sh           every OpenAI feature off; Magic fill reads resumes on this computer
#   ./combined_demo/run.sh --ai      Jesse's OpenAI features on (each asks for consent in the app before sending anything)
set -euo pipefail

ai=0
backend_port=8002
frontend_port=5175
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ai) ai=1; shift ;;
    --backend-port) backend_port="$2"; shift 2 ;;
    --frontend-port) frontend_port="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend="$repo/app/frontend"
python="$repo/.venv/bin/python"
url="http://127.0.0.1:$frontend_port"

if curl -fs --max-time 2 "http://127.0.0.1:$backend_port/healthz" >/dev/null 2>&1; then
  echo "A backend is already running on port $backend_port. Stop it first." >&2
  exit 1
fi
if [[ ! -d "$frontend/node_modules" ]]; then
  echo "Installing the frontend's packages (first time only)..."
  (cd "$frontend" && npm install)
fi

if [[ $ai -eq 1 ]]; then mode="openai"; reader="openai"; label="OpenAI features ON"
else mode="off"; reader="local"; label="OpenAI features OFF"; fi
export PREDICTLEADS_RESUME_EXTRACTION="$reader" PREDICTLEADS_ASSESSMENT="$mode" PREDICTLEADS_MARKET="$mode"
export PREDICTLEADS_API_COOKIE_SECURE="0" PYTHONUNBUFFERED="1"
export PREDICTLEADS_BACKEND_PORT="$backend_port" PREDICTLEADS_FRONTEND_PORT="$frontend_port"
export PREDICTLEADS_API_ORIGINS="http://127.0.0.1:$frontend_port,http://localhost:$frontend_port"

trap 'trap - EXIT; kill 0' INT TERM EXIT  # Ctrl+C or exit stops the backend and the frontend with this script

(cd "$repo" && "$python" -m uvicorn combined_demo.server:app --host 127.0.0.1 --port "$backend_port") &
backend=$!
(cd "$frontend" && npm run dev) &

echo "Combined demo starting ($label). Waiting for the backend (a few minutes)..."
for _ in $(seq 1 180); do  # up to 15 minutes
  if curl -fs --max-time 3 "http://127.0.0.1:$backend_port/healthz" >/dev/null 2>&1; then
    if command -v open >/dev/null 2>&1; then open "$url"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"; fi
    echo "Ready: $url (combined model, $label). Press Ctrl+C to stop."
    wait
    exit 0
  fi
  if ! kill -0 "$backend" 2>/dev/null; then
    echo "The backend stopped: see the error above." >&2
    exit 1
  fi
  sleep 5
done
echo "The backend didn't come up within 15 minutes: see the messages above." >&2
exit 1
