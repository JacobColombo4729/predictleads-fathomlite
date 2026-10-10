#!/usr/bin/env bash
# macOS / Linux version of run_demo_backend.ps1.
# Starts the demo API on http://127.0.0.1:8000 with the TF-IDF + BGE (RRF) ranker on the serving corpus.
# Needs: data/matching/ai_ml_serving_v1 (built by `python code/rank_candidates.py export-serving`).
# Uses the single project .venv (pip install -r requirements.txt) and the BGE model in models/bge-base-en-v1.5.
# First start takes a few minutes (it fits TF-IDF over the corpus once). Stop it with Ctrl+C.
#   --ranker tfidf   start the lighter TF-IDF-only ranker instead
#   --port N         listen on another port
#   --preview        light preview for a laptop: TF-IDF on a small job sample in data/preview/jobs.jsonl
#                    (no BGE model or embeddings needed; starts in ~20 s; Compass is unavailable)
#   --resume-fill    enable the Profile page's "Magic fill" (sends the resume text you pick to OpenAI; needs OPENAI_API_KEY in .env; off by default)
set -euo pipefail

ranker="rrf"
port=8000
resume_fill=0
preview=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ranker) ranker="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --resume-fill) resume_fill=1; shift ;;
    --preview) preview=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
serving="$root/data/matching/ai_ml_serving_v1"
if [[ $preview -eq 1 ]]; then
  jobs="$root/data/preview/jobs.jsonl"
  if [[ ! -f "$jobs" ]]; then
    echo "Missing $jobs. Build it from the serving corpus, e.g. from the shared zip:" >&2
    echo "  mkdir -p data/preview && unzip -p <zip> data/matching/ai_ml_serving_v1/jobs_prepared_corpus.jsonl | head -n 5000 > data/preview/jobs.jsonl" >&2
    echo "  unzip -n <zip> 'data/signals/*'   # optional: Signals tab" >&2
    exit 1
  fi
  export PREDICTLEADS_API_JOBS_PATH="$jobs"
  # With matching embeddings and the BGE model present, use the full ranker (Compass works); otherwise TF-IDF.
  if [[ -f "$root/data/preview/bge_embeddings.npy" && -d "$root/models/bge-base-en-v1.5" ]]; then
    export PREDICTLEADS_API_CORPUS_DIR="$root/data/preview"
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
export PYTHONUNBUFFERED="1"
mkdir -p "$root/backend/data"
cd "$root"
exec "$root/.venv/bin/python" -m uvicorn backend.api:app --host 127.0.0.1 --port "$port"
