# Starts the demo API on http://127.0.0.1:8000 with the TF-IDF + BGE (RRF) ranker on the serving corpus.
# Needs: data/matching/ai_ml_serving_v1 (built by `python code/rank_candidates.py export-serving`).
# Uses the repository's .venv (pip install -r requirements.txt at the top) and the BGE model in the top-level pretrained/bge-base-en-v1.5.
# First start takes a few minutes (it fits TF-IDF over the corpus once). Stop it with Ctrl+C.
# Use -Ranker tfidf to start the lighter TF-IDF-only ranker instead.
# Use -Market to enable typed questions in "Ask the market" (the question text goes to OpenAI to become a search) and the optional "Research this company" web step (sends only a company name and website); needs OPENAI_API_KEY in .env; off by default. The preset example questions work without it.
# Use -Assess to enable the job details "AI gap analysis" tab (it sends the profile's experience and skills and the job description to OpenAI after a per-use consent; needs OPENAI_API_KEY in .env; off by default).
# Use -ResumeFill to enable the Profile page's "Magic fill" (it sends the resume text you pick to OpenAI; needs OPENAI_API_KEY in .env; off by default).
param([string]$Ranker = "rrf", [int]$Port = 8000, [switch]$ResumeFill, [switch]$Assess, [switch]$Market)

$root = $PSScriptRoot
$repo = Split-Path $root -Parent  # data/, pretrained/, .env and .venv are shared with analysis/ at the top
$serving = Join-Path $repo "data\matching\ai_ml_serving_v1"
$env:PREDICTLEADS_API_JOBS_PATH = Join-Path $serving "jobs_prepared_corpus.jsonl"
$env:PREDICTLEADS_API_CORPUS_DIR = $serving
$env:PREDICTLEADS_API_RANKER = $Ranker
$env:PREDICTLEADS_API_DB_PATH = Join-Path $root "backend\data\predictleads-demo.sqlite3"
$env:PREDICTLEADS_API_COOKIE_SECURE = "0"
if ($ResumeFill) { $env:PREDICTLEADS_RESUME_EXTRACTION = "openai" }
if ($Assess) { $env:PREDICTLEADS_ASSESSMENT = "openai" }
if ($Market) { $env:PREDICTLEADS_MARKET = "openai" }
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force (Join-Path $root "backend\data") | Out-Null
Set-Location $root
& (Join-Path $repo ".venv\Scripts\python.exe") -m uvicorn backend.api:app --host 127.0.0.1 --port $Port
