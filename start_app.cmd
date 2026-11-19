@echo off
:: ============================================================================
:: start_app.cmd  –  Launch the Synthetic APAC Face Dataset Generator
::
:: ALWAYS use this script (or manually activate .venv) to start the server.
::
:: Why this matters
:: ----------------
:: The system Python 3.13 has kaggle 2.x installed (via pip install kaggle).
:: kaggle 2.x has an incompatible authentication flow for the blob-upload API
:: and causes HTTP 401 Unauthorized during manifest upload → FAILED datasets.
::
:: The project .venv has kaggle 1.7.x which works correctly.
:: This script explicitly invokes uvicorn from the .venv so the right version
:: is always used, regardless of what is on the system PATH.
:: ============================================================================

setlocal

:: Resolve the directory this script lives in (project root)
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Verify the .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] .venv not found at %PROJECT_ROOT%.venv\Scripts\python.exe
    echo         Create it with:  python -m venv .venv
    echo         Then install:    .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Verify the .venv has the correct kaggle version (1.x, not 2.x)
".venv\Scripts\python.exe" -c "import importlib.metadata; v=importlib.metadata.version('kaggle'); major=int(v.split('.')[0]); exit(1 if major>=2 else 0)" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] The .venv has kaggle 2.x installed, but this application requires
    echo         kaggle 1.7.x.  Fix with:
    echo         .venv\Scripts\pip install "kaggle>=1.7,<2"
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Using .venv Python: %PROJECT_ROOT%.venv\Scripts\python.exe
echo [OK] Kaggle SDK 1.x confirmed in .venv
echo.
echo Starting Uvicorn on http://127.0.0.1:8000  (Ctrl+C to stop)
echo.

".venv\Scripts\uvicorn" backend.app.main:app --host 127.0.0.1 --port 8000 --reload --app-dir "%PROJECT_ROOT%"

endlocal
