@echo off
setlocal

set PYTHON=python
set OUTPUT_NAME=InfoPilotDesktop

:parse_args
if "%~1"=="" goto after_parse
if "%~1"=="--python" (
    set PYTHON=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--name" (
    set OUTPUT_NAME=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

after_parse:

if not exist "%~dp0..\ui\app.py" (
    echo [ERROR] ui\app.py 파일을 찾지 못했습니다. 저장소 루트에서 실행하세요.
    exit /b 1
)

pushd "%~dp0..\"

%PYTHON% -m PyInstaller ^
    --clean ^
    --noconfirm ^
    --windowed ^
    --onedir ^
    --name %OUTPUT_NAME% ^
    --collect-data customtkinter ^
    --hidden-import tkinter ^
    --hidden-import PIL ^
    --hidden-import core.conversation.lnp_chat ^
    --hidden-import core.agents.meeting.pipeline ^
    --hidden-import core.agents.photo.pipeline ^
    --paths . ^
    ui\app.py

if errorlevel 1 (
    echo [ERROR] PyInstaller 빌드가 실패했습니다.
    popd
    exit /b 1
)

echo.
echo [INFO] 빌드 성공. dist\%OUTPUT_NAME%\%OUTPUT_NAME%.exe 를 실행하세요.
echo.

popd
exit /b 0
