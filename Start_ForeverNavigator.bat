@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "INSTALL_ONLY=0"
set "FORCE_UPDATE="
set "USE_CONSOLE=0"
for %%A in (%*) do (
    if /i "%%~A"=="--install-only" set "INSTALL_ONLY=1"
    if /i "%%~A"=="--update" set "FORCE_UPDATE=1"
    if /i "%%~A"=="--console" set "USE_CONSOLE=1"
)

if "%FORCE_UPDATE%"=="" (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -c "import tkinter, numpy, PIL, cv2, mss, onnxruntime, moderngl, pyrr, websockets, PySide6" >nul 2>&1
        if !ERRORLEVEL!==0 (
            if "%INSTALL_ONLY%"=="1" (
                echo All ForeverNavigator runtime dependencies are installed.
                exit /b 0
            )
            goto :LAUNCH
        )
    )
)

echo ============================================================
echo   ForeverNavigator - First-Time Setup / Dependency Update
echo ============================================================

where uv >nul 2>&1
if %ERRORLEVEL%==0 (
    if not exist ".venv\Scripts\python.exe" (
        uv venv --python 3.12 --python-preference only-managed .venv || uv venv --python 3.11 --python-preference only-managed .venv || uv venv .venv
    )
    uv pip install --python ".venv\Scripts\python.exe" -r requirements-runtime.txt
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
    if "%INSTALL_ONLY%"=="1" (
        echo All ForeverNavigator runtime dependencies are installed.
        exit /b 0
    )
    goto :LAUNCH
)

set "PY_CMD="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3.12 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.12"
    if "!PY_CMD!"=="" py -3.11 -c "import sys" >nul 2>&1 && set "PY_CMD=py -3.11"
    if "!PY_CMD!"=="" set "PY_CMD=py -3"
)
if "%PY_CMD%"=="" where python >nul 2>&1 && set "PY_CMD=python"
if "%PY_CMD%"=="" (
    echo [ERROR] Python 3 not found. Install Python 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" %PY_CMD% -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements-runtime.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)
if "%INSTALL_ONLY%"=="1" exit /b 0

:LAUNCH
if "%USE_CONSOLE%"=="1" (
    ".venv\Scripts\python.exe" -m src.ui.app
    if %ERRORLEVEL% NEQ 0 pause
    exit /b %ERRORLEVEL%
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m src.ui.app
    exit /b 0
)
".venv\Scripts\python.exe" -m src.ui.app
if %ERRORLEVEL% NEQ 0 pause
