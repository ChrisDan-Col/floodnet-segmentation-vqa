@echo off
REM  FloodNet app - LAUNCHER (uses the already-installed .venv_app environment)
cd /d "%~dp0"
call ".venv_app\Scripts\activate.bat"
python app.py
