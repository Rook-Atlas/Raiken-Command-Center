@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe main.py
echo.
echo [launcher] Raiken exited with code %errorlevel%.
pause
