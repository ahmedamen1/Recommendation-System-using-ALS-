@echo off
setlocal
title Movie Recommender - Movie recommender
pushd "%~dp0"
set "DASHBOARD_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%DASHBOARD_PYTHON%" goto check_dependencies
set "DASHBOARD_PYTHON=python"
where python >nul 2>&1
if errorlevel 1 goto missing_python

:check_dependencies
"%DASHBOARD_PYTHON%" -c "import numpy" >nul 2>&1
if errorlevel 1 goto missing_numpy
echo Starting Movie Recommender with your saved movie model...
echo Keep this window open. Press Ctrl+C to stop the server.
echo.
"%DASHBOARD_PYTHON%" "%~dp0dashboard\server.py" --open-browser %*
if errorlevel 1 pause
popd
exit /b

:missing_python
echo Python was not found. Install Python or restore the project's .venv folder.
goto finish_error

:missing_numpy
echo NumPy is missing. Run this command, then double-click this file again:
echo "%DASHBOARD_PYTHON%" -m pip install -r "%~dp0dashboard\requirements.txt"

:finish_error
pause
popd
exit /b 1
