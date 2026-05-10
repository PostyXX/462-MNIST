@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Failed to create the virtual environment.
        echo Make sure Python 3.10+ is installed and on your PATH.
        echo Download: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo [setup] Installing dependencies — this can take a few minutes the first time...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency install failed. See the messages above.
        pause
        exit /b 1
    )
    echo [setup] Done.
    echo.
)

REM Python opens the browser itself (after the server is actually listening),
REM so we don't need a fragile timeout-based approach in the batch file.
".venv\Scripts\python.exe" app.py

echo.
echo Server stopped.
pause
