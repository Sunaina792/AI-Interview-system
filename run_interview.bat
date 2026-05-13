@echo off
echo ================================================
echo  AI Interview Agent - Starting...
echo ================================================
cd /d "%~dp0"
.venv\Scripts\python.exe gradio_app.py
pause
