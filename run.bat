@echo off
cd /d "%~dp0"
start "" http://localhost:8000
python server.py
pause
