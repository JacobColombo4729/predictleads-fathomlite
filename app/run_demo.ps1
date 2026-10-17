# Starts the whole demo with every OpenAI feature off: nothing leaves the computer except what the browser loads.
# Opens two windows (the backend, started with -NoAI, and the frontend), waits for the backend to load (about
# 3.5 minutes, about 2.5 GB of memory), then opens http://127.0.0.1:5173. Stop the demo by closing both windows.
# Needs what run_demo_backend.ps1 needs (the serving corpus, the BGE model, the .venv) and Node.js for the frontend.
# The AI features (Magic fill, the AI gap analysis, Ask the market's typed questions) show as off; everything else works.
#   .\app\run_demo.ps1                 the full ranker (TF-IDF + BGE)
#   .\app\run_demo.ps1 -Ranker tfidf   the lighter TF-IDF-only ranker
param([string]$Ranker = "rrf")

$root = $PSScriptRoot
$frontend = Join-Path $root "frontend"
$backendScript = Join-Path $root "run_demo_backend.ps1"
$url = "http://127.0.0.1:5173"

try {
    if ((Invoke-WebRequest "http://127.0.0.1:8000/healthz" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
        Write-Host "A backend is already running on port 8000. Close its window first, so this demo starts with the AI features off."
        exit 1
    }
} catch { }  # nothing on port 8000: good

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing the frontend's packages (first time only)..."
    Push-Location $frontend
    npm install
    Pop-Location
}

$backend = Start-Process powershell -PassThru -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$backendScript`"", "-NoAI", "-Ranker", $Ranker)
# Inside -Command, PowerShell strips double quotes, so a folder name with a space would split: single quotes, apostrophes doubled.
$frontendQ = $frontend.Replace("'", "''")
Start-Process powershell -ArgumentList @("-NoExit", "-Command", "Set-Location -LiteralPath '$frontendQ'; npm run dev") | Out-Null

Write-Host "Demo starting with the OpenAI features OFF. Waiting for the backend (about 3.5 minutes)..."
$deadline = (Get-Date).AddMinutes(10)
$ready = $false
while (-not $ready -and (Get-Date) -lt $deadline -and -not $backend.HasExited) {
    Start-Sleep -Seconds 5
    try { $ready = (Invoke-WebRequest "http://127.0.0.1:8000/healthz" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { }
}
if ($ready) {
    Start-Process $url
    Write-Host "Ready: $url (AI features off). Close the backend and frontend windows to stop."
} else {
    Write-Host "The backend didn't come up within 10 minutes: look at its window for the error."
    exit 1
}
