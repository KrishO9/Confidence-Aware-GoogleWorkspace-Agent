@echo off
REM Quick start script for Email Assistant (Windows)

echo ========================================
echo Email Assistant - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install/upgrade dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Run the assistant
echo Starting Email Assistant...
echo.
python main.py

REM Deactivate virtual environment
deactivate






