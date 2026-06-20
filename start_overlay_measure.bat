@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "D:\python\python.exe" set "PYTHON_EXE=D:\python\python.exe"

if not defined PYTHON_EXE (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 (
        py -3 main.py
        goto :after_run
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo Python was not found. Install Python, then run:
    echo python -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" main.py

:after_run
if errorlevel 1 (
    echo.
    echo Overlay Measure failed to start.
    echo If dependencies are missing, run:
    echo python -m pip install -r requirements.txt
    pause
)
