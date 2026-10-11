#!/usr/bin/env bash
# macOS / Linux version of run_demo_backend.ps1.
# Starts the demo API on http://127.0.0.1:8000 with the TF-IDF + BGE (RRF) ranker on the serving corpus.
# Needs: data/matching/ai_ml_serving_v1 (built by `python code/rank_candidates.py export-serving`).
# The data folder is app/data, or $PREDICTLEADS_DATA_DIR when set (for example one folder shared with analysis/).
# Uses the single project .venv (pip install -r requirements.txt) and the BGE model in models/bge-base-en-v1.5.
# First start takes a few minutes (it fits TF-IDF over the corpus once). Stop it with Ctrl+C.
#   --ranker tfidf   start the lighter TF-IDF-only ranker instead
#   --port N         listen on another port
#   --preview        light preview for a laptop: TF-IDF on a small job sample in data/preview/jobs.jsonl
#                    (no BGE model or embeddings needed; starts in ~20 s; Compass is unavailable)
#   --assess         enable the Compass "skills gap analysis" (sends the profile's experience and skills and the job descriptions to OpenAI after a consent; needs OPENAI_API_KEY in .env; off by default)
#   --market         enable typed questions in "Ask the market" and the optional "Research this company" web step (sends the question, or a company name and website, to OpenAI; needs OPENAI_API_KEY; off by default)
#   --resume-fill    enable the Profile page's "Magic fill" (sends the resume text you pick to OpenAI; needs OPENAI_API_KEY in .env; off by default)
set -euo pipefail

ranker="rrf"
port=8000
resume_fill=0
assess=0
market=0
preview=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ranker) ranker="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --resume-fill) resume_fill=1; shift ;;
    --assess) assess=1; shift ;;
    --market) market=1; shift ;;
    --preview) preview=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${PREDICTLEADS_DATA_DIR:-$root/data}"
serving="$data/matching/ai_ml_serving_v1"
if [[ $preview -eq 1 ]]; then
  jobs="$data/preview/jobs.jsonl"
  if [[ ! -f "$jobs" ]]; then
    echo "Missing $jobs. Build it from the serving corpus, e.g. from the shared zip:" >&2
    echo "  mkdir -p data/preview && unzip -p <zip> data/matching/ai_ml_serving_v1/jobs_prepared_corpus.jsonl | head -n 5000 > data/preview/jobs.jsonl" >&2
    echo "  unzip -n <zip> 'data/signals/*'   # optional: Signals tab" >&2
    exit 1
  fi
  export PREDICTLEADS_API_JOBS_PATH="$jobs"
  # With matching embeddings and the BGE model present, use the full ranker (Compass works); otherwise TF-IDF.
  if [[ -f "$data/preview/bge_embeddings.npy" && -d "$root/models/bge-base-en-v1.5" ]]; then
    export PREDICTLEADS_API_CORPUS_DIR="$data/preview"
    export PREDICTLEADS_API_RANKER="rrf"
  else
    export PREDICTLEADS_API_RANKER="tfidf"
  fi
  export PREDICTLEADS_API_DB_PATH="$root/backend/data/preview.sqlite3"
else
  export PREDICTLEADS_API_JOBS_PATH="$serving/jobs_prepared_corpus.jsonl"
  export PREDICTLEADS_API_CORPUS_DIR="$serving"
  export PREDICTLEADS_API_RANKER="$ranker"
  export PREDICTLEADS_API_DB_PATH="$root/backend/data/predictleads-demo.sqlite3"
fi
export PREDICTLEADS_API_COOKIE_SECURE="0"
if [[ $resume_fill -eq 1 ]]; then export PREDICTLEADS_RESUME_EXTRACTION="openai"; fi
if [[ $assess -eq 1 ]]; then export PREDICTLEADS_ASSESSMENT="openai"; fi
if [[ $market -eq 1 ]]; then export PREDICTLEADS_MARKET="openai"; fi
export PYTHONUNBUFFERED="1"
mkdir -p "$root/backend/data"
cd "$root"
exec "$root/.venv/bin/python" -m uvicorn backend.api:app --host 127.0.0.1 --port "$port"
