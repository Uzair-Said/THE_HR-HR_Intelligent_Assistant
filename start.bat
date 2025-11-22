@echo off
echo ======================================
echo    THE HR - HR Intelligent Assistant  
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install or update requirements
echo Installing/updating dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Clear any previous Streamlit cache
echo Clearing cache...
if exist ".streamlit\cache" rd /s /q ".streamlit\cache"

REM Launch the application
echo.
echo Starting THE HR Assistant...
echo ================================
echo Admin Login Credentials:
echo Username: hradmin
echo Password: hrpass123
echo ================================
echo.
echo Opening in your default browser...
echo If browser doesn't open, navigate to: http://localhost:8501
echo.

REM Run Streamlit
streamlit run hr_assistant.py --server.port=8501 --server.headless=true

pause
