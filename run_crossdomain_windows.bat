@echo off
REM Uses the project virtual environment, which is where torch is installed.
"%~dp0.venv\Scripts\python.exe" "%~dp0crossdomain_app.py"
pause
