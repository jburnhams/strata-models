@echo off
rem Launcher for Windows: runs the strata-models CLI with the project environment. Override the interpreter with STRATAMODELS_PYTHON.
set "ROOT=%~dp0"
if "%STRATAMODELS_PYTHON%"=="" set "STRATAMODELS_PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PYTHONPATH=%ROOT%src"
set "PYTHONWARNINGS=ignore"
"%STRATAMODELS_PYTHON%" -m stratamodels %*
