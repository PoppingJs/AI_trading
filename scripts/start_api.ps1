$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found. Creating it under D drive project path..."
    python -m venv (Join-Path $ProjectRoot ".venv")
}

& $Python -m pip show ai-trading *> $null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install -e "$ProjectRoot"
}

& $Python -m uvicorn ai_trading.api:create_app --factory --host 127.0.0.1 --port 8000
