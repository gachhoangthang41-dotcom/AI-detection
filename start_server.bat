@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay .venv\Scripts\python.exe
    echo Hay tao virtual environment va cai requirements truoc.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" api.py
pause
