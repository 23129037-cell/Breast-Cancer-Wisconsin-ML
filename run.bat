
@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo BREAST CANCER ML + FEATURE SELECTION + XAI
echo ==============================================

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Khong tim thay Python trong .venv
    echo Hay kiem tra moi truong Python cua du an.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo ERROR: main.py chay that bai.
    pause
    exit /b 1
)

echo.
echo Hoan tat chay main.py.
pause
