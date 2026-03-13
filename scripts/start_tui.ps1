$ErrorActionPreference = 'Stop'

$InvokeDir = (Get-Location).Path
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Error "Missing virtual environment at .venv\Scripts\python.exe"
}

$env:WORKSPACE_ROOT = if ($env:WORKSPACE_ROOT) { $env:WORKSPACE_ROOT } else { $InvokeDir }

& ".venv\Scripts\python.exe" -m app.tui --root $env:WORKSPACE_ROOT
