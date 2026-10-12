# Runs the project's tests from one place.
#   .\run_tests.ps1             live Python tests (tests/, backend/tests/ and analysis/tests/)
#   .\run_tests.ps1 -Frontend   also type-check and build the React app
param([switch]$Frontend)

$root = $PSScriptRoot
$py = Join-Path (Split-Path $root -Parent) ".venv\Scripts\python.exe"  # the repository's .venv, at the top
$env:PYTHONPATH = Join-Path $root "code"
Set-Location $root
$failed = $false

function Run-Step($name, [scriptblock]$step) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $step
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name" -ForegroundColor Red; $script:failed = $true }
}

Run-Step "tests/ (engine, explanations, BGE text)" { & $py -m unittest discover -s tests -q }
Run-Step "backend/tests/ (API, ranker, feedback, resume reader)" { & $py -m unittest discover -s backend/tests -t . -q }
Run-Step "analysis/tests/ (role discovery evaluation)" { & $py -m unittest discover -s ../analysis/tests -t ../analysis -q }
if ($Frontend) {
    Run-Step "frontend type-check" { Push-Location frontend; npx tsc -b; $code = $LASTEXITCODE; Pop-Location; $global:LASTEXITCODE = $code }
    Run-Step "frontend build" { Push-Location frontend; npm run build --silent; $code = $LASTEXITCODE; Pop-Location; $global:LASTEXITCODE = $code }
}

if ($failed) { Write-Host "`nSome steps failed." -ForegroundColor Red; exit 1 }
Write-Host "`nAll steps passed." -ForegroundColor Green
