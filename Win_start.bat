@echo off
chcp 65001 > nul
title 4×4×4 Cube Analysis Simulator - Web GUI

echo ========================================
echo 4×4×4 Cube Analysis Simulator - Web GUI
echo Version: Release 1.10
echo Creator: Haruki Shimodaira
echo ========================================
echo.

cd /d "%~dp0"

python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10 or later from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Checking Python version...
python --version
echo.

if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Clean Python cache to avoid import errors
echo Cleaning Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc > nul 2>&1

python -c "import flask" > nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take a few minutes.
    echo.
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        echo.
        echo Note: pandas is optional and may fail on some systems.
        echo The application will work without it.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully.
    echo Note: Some optional packages may have been skipped.
    echo.
)

echo Starting Web GUI...
echo.
echo Web interface available at:
echo   http://localhost:8888
echo.
echo Press Ctrl+C to stop the server.
echo ========================================
echo.

python run_web_gui.py

echo.
echo Shutting down...
pause
