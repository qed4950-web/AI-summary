# InfoPilot Desktop UI 실행 파일 빌드 스크립트
# 사용 예시:
#   powershell -ExecutionPolicy Bypass -File scripts\build_desktop_ui.ps1 -OutputName InfoPilotDesktop

param(
    [string]$Python = "python",
    [string]$OutputName = "InfoPilotDesktop",
    [switch]$CreateZip,
    [string]$SignCommand = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not (Test-Path "$repoRoot\ui\app.py")) {
    Write-Error "ui/app.py 파일을 찾을 수 없습니다. 저장소 루트에서 실행해주세요."
}

Write-Host "🚧 PyInstaller로 InfoPilot 데스크톱 UI를 빌드합니다..." -ForegroundColor Cyan
Write-Host "   Python  : $Python"
Write-Host "   Name    : $OutputName"

$cmd = @(
    $Python, "-m", "PyInstaller",
    "--clean",
    "--noconfirm",
    "--windowed",
    "--onedir",
    "--name", $OutputName,
    "--collect-data", "customtkinter",
    "--hidden-import", "tkinter",
    "--hidden-import", "PIL",
    "--hidden-import", "core.conversation.lnp_chat",
    "--hidden-import", "core.agents.meeting.pipeline",
    "--hidden-import", "core.agents.photo.pipeline",
    "--paths", ".",
    "ui\\app.py"
)

Write-Host "   Command : $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length-1)]

$exePath = Join-Path $repoRoot "dist" | Join-Path -ChildPath $OutputName | Join-Path -ChildPath ("$OutputName.exe")
if ($SignCommand) {
    $expanded = $SignCommand.Replace("{file}", $exePath)
    Write-Host "🖊️ 코드 서명 실행: $expanded" -ForegroundColor Cyan
    Invoke-Expression $expanded
}

Write-Host "✅ 빌드가 완료되었습니다. dist\$OutputName\$OutputName.exe 파일을 확인하세요." -ForegroundColor Green

if ($CreateZip) {
    $distPath = Join-Path $repoRoot "dist" | Join-Path -ChildPath $OutputName
    $zipPath = Join-Path $repoRoot "dist" | Join-Path -ChildPath ("$OutputName.zip")
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "📦 설치용 ZIP 패키지를 생성합니다..." -ForegroundColor Cyan
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($distPath, $zipPath)
    Write-Host "✅ ZIP 생성 완료: $zipPath" -ForegroundColor Green
}
