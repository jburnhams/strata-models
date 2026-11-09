# Build the project environment from scratch on Windows (Python 3.13).
#   scripts\setup_env.ps1
# Prerequisites: winget install Python.Python.3.13 OpenJS.NodeJS
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot; $Venv = Join-Path $Root '.venv'; $Py = Join-Path $Venv 'Scripts\python.exe'
function Say($m) { Write-Host "`n==> $m" }
Say "virtual environment: $Venv"
py -3.13 -m venv $Venv
& $Py -m pip install --upgrade pip | Out-Null
Say 'requirements'
& $Py -m pip install -r (Join-Path $Root 'requirements.txt')
$env:PYTHONPATH = Join-Path $Root 'src'
Say 'checking'
& $Py -m app doctor
if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location (Join-Path $Root 'web'); npm install --silent; npm run build; Pop-Location; Write-Host 'web app built' }
else { Write-Host 'npm not found: winget install OpenJS.NodeJS, then cd web; npm install; npm run build' }
