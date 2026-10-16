# The combined demo: the same app ranked by the combined model (combined/; ML_Fit's role fit as the fit factor of
# ML_Rank's queue over Jesse's text match, on the scope's listings). Runs next to the other two apps: backend on port
# 8002, the page on http://127.0.0.1:5175 (Jesse's app keeps 8000/5173, Jacob's 8001/5174), with its own session database.
# Opens two windows (the backend and the frontend), waits for the backend (a few minutes), then opens the page.
# Stop it by closing both windows. Needs the combined corpus (python -m combined.corpus build, then encode), ML_Fit's
# hand-off (ML_Fit 9b) and ML_Rank's (ML_Rank section 13); the backend says which is missing.
#   .\combined_demo\run.ps1          every OpenAI feature off; Magic fill reads resumes on this computer
#   .\combined_demo\run.ps1 -AI      Jesse's OpenAI features on (each asks for consent in the app before sending anything)
param([switch]$AI, [int]$BackendPort = 8002, [int]$FrontendPort = 5175)

$repo = Split-Path $PSScriptRoot -Parent
$frontend = Join-Path $repo "app\frontend"
$python = Join-Path $repo ".venv\Scripts\python.exe"
$url = "http://127.0.0.1:$FrontendPort"

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

$mode = if ($AI) { "openai" } else { "off" }
$env:PREDICTLEADS_RESUME_EXTRACTION = if ($AI) { "openai" } else { "local" }
$env:PREDICTLEADS_ASSESSMENT = $mode
$env:PREDICTLEADS_MARKET = $mode
$env:PREDICTLEADS_API_COOKIE_SECURE = "0"
$env:PREDICTLEADS_BACKEND_PORT = "$BackendPort"
$env:PREDICTLEADS_FRONTEND_PORT = "$FrontendPort"
$env:PREDICTLEADS_API_ORIGINS = "http://127.0.0.1:$FrontendPort,http://localhost:$FrontendPort"
$env:PYTHONUNBUFFERED = "1"

$backend = Start-Process powershell -PassThru -WorkingDirectory $repo -ArgumentList @(
    "-NoExit", "-Command", "& `"$python`" -m uvicorn combined_demo.server:app --host 127.0.0.1 --port $BackendPort")
Start-Process powershell -ArgumentList @("-NoExit", "-Command", "Set-Location `"$frontend`"; npm run dev") | Out-Null

$label = if ($AI) { "OpenAI features ON" } else { "OpenAI features OFF" }
Write-Host "Combined demo starting ($label). Waiting for the backend (a few minutes)..."
$deadline = (Get-Date).AddMinutes(15)
$ready = $false
while (-not $ready -and (Get-Date) -lt $deadline -and -not $backend.HasExited) {
    Start-Sleep -Seconds 5
    try { $ready = (Invoke-WebRequest "http://127.0.0.1:$BackendPort/healthz" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { }
}
if ($ready) {
    Start-Process $url
    Write-Host "Ready: $url (combined model, $label). Close the backend and frontend windows to stop."
} else {
    Write-Host "The backend didn't come up within 15 minutes: look at its window for the error."
    exit 1
}
