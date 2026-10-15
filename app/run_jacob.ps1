# Jacob's app: the same app as run_demo.ps1, ranked by ML_Rank's priority queue (backend/ml_rank_ranker.py), with every
# OpenAI feature off. Runs next to the demo: backend on port 8001, frontend on http://127.0.0.1:5174 (the demo keeps 8000/5173).
# Opens two windows (the backend and the frontend), waits for the backend to load (about 3.5 minutes), then opens the page.
# Stop it by closing both windows. Needs what run_demo.ps1 needs plus ML_Rank's hand-off (ML_Rank section 13, data/clean/ml_rank/).
# Both apps share the demo's session database, so a profile made in one shows in the other.
param([int]$BackendPort = 8001, [int]$FrontendPort = 5174)

$root = $PSScriptRoot
$repo = Split-Path $root -Parent
$frontend = Join-Path $root "frontend"
$backendScript = Join-Path $root "run_demo_backend.ps1"
$url = "http://127.0.0.1:$FrontendPort"

if (-not (Test-Path (Join-Path $repo "data\clean\ml_rank\listing_scores.parquet"))) {
    Write-Host "ML_Rank's hand-off isn't there yet: run ML_Rank section 13 (13a) first."
    exit 1
}
try {
    if ((Invoke-WebRequest "http://127.0.0.1:$BackendPort/healthz" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) {
        Write-Host "A backend is already running on port $BackendPort. Close its window first."
        exit 1
    }
} catch { }  # nothing on that port: good

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing the frontend's packages (first time only)..."
    Push-Location $frontend
    npm install
    Pop-Location
}

# Every OpenAI feature off, whatever this shell or .env says (only "openai" turns one on); the windows below inherit these
$env:PREDICTLEADS_RESUME_EXTRACTION = "off"
$env:PREDICTLEADS_ASSESSMENT = "off"
$env:PREDICTLEADS_MARKET = "off"
$env:PREDICTLEADS_BACKEND_PORT = "$BackendPort"
$env:PREDICTLEADS_FRONTEND_PORT = "$FrontendPort"

$backend = Start-Process powershell -PassThru -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$backendScript`"", "-NoAI", "-Ranker", "mlrank", "-Port", $BackendPort)
Start-Process powershell -ArgumentList @("-NoExit", "-Command", "Set-Location `"$frontend`"; npm run dev") | Out-Null

Write-Host "Jacob's app starting (ML_Rank queue, OpenAI features OFF). Waiting for the backend (about 3.5 minutes)..."
$deadline = (Get-Date).AddMinutes(10)
$ready = $false
while (-not $ready -and (Get-Date) -lt $deadline -and -not $backend.HasExited) {
    Start-Sleep -Seconds 5
    try { $ready = (Invoke-WebRequest "http://127.0.0.1:$BackendPort/healthz" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { }
}
if ($ready) {
    Start-Process $url
    Write-Host "Ready: $url (ML_Rank queue, AI features off). Close the backend and frontend windows to stop."
} else {
    Write-Host "The backend didn't come up within 10 minutes: look at its window for the error."
    exit 1
}
