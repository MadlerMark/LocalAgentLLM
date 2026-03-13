$ErrorActionPreference = 'Stop'

$InvokeDir = (Get-Location).Path
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Error "Missing virtual environment at .venv\Scripts\python.exe"
}

$hostVar = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$portVar = if ($env:PORT) { $env:PORT } else { "8011" }
$env:WORKSPACE_ROOT = if ($env:WORKSPACE_ROOT) { $env:WORKSPACE_ROOT } else { $InvokeDir }

while (Get-NetTCPConnection -State Listen -LocalPort $portVar -ErrorAction SilentlyContinue) {
  $portVar = [string]([int]$portVar + 1)
}

Write-Host "Starting LocalAgentLLM on http://$hostVar`:$portVar"
Write-Host "workspace_root=$($env:WORKSPACE_ROOT)"

& ".venv\Scripts\python.exe" -m app.server_entry --host $hostVar --port $portVar --workspace-root $env:WORKSPACE_ROOT
