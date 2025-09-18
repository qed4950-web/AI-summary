param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv"
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPath = Join-Path $Root $VenvDir
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPath)) {
    Write-Host "[setup] Creating virtual environment at $VenvPath"
    & $Python -m venv $VenvPath
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment python executable not found at $PythonExe"
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host "[setup] Environment ready. Activate with: `n  $VenvPath\Scripts\Activate.ps1"
