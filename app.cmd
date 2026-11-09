@echo off
rem Launcher for Windows: runs the app CLI with the project environment. Override the interpreter with APP_PYTHON.
set "ROOT=%~dp0"
if "%APP_PYTHON%"=="" set "APP_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%src"
set "PYTHONWARNINGS=ignore"
"%APP_PYTHON%" -m app %*
