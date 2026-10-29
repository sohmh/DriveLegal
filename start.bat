    @echo off
title DriveLegal Launcher
color 0A

echo.
echo ============================================
echo     DriveLegal - Starting App
echo ============================================
echo.

:: Check venv exists
if not exist venv (
    echo [ERROR] venv not found. Run setup.bat first.
    pause
    exit /b
)

:: Check .env exists
if not exist .env (
    echo [ERROR] .env not found. Run setup.bat first.
    pause
    exit /b
)

echo Starting FastAPI backend on http://localhost:8000 ...
start "DriveLegal Backend" cmd /k "venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait 3 seconds for backend to start before launching frontend
timeout /t 3 /nobreak >nul

echo Starting Streamlit frontend on http://localhost:8501 ...
start "DriveLegal Frontend" cmd /k "venv\Scripts\activate && streamlit run frontend\app.py"

:: Wait 2 seconds then open browser
timeout /t 2 /nobreak >nul
set "CHROME_PATH="
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "CHROME_PATH=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_PATH if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "CHROME_PATH=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined CHROME_PATH if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "CHROME_PATH=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if defined CHROME_PATH (
    echo Opening Chrome...
    start "" "%CHROME_PATH%" "http://localhost:8501"
) else (
    echo Chrome not found, opening default browser...
    start "" "http://localhost:8501"
)

echo.
echo ============================================
echo   DriveLegal is running!
echo   Frontend: http://localhost:8501
echo   API Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Close the two terminal windows to stop the app.
pause
    