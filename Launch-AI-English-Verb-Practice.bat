@echo off
title AI English Verb Practice Agent - Launcher

echo ===================================================
echo     AI English Verb Practice Agent - Launcher
echo ===================================================
echo.

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% equ 0 goto python_exists

echo [ERROR] Python was not found in your system PATH!
echo Please download and install Python 3.10+ from https://www.python.org/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b

:python_exists
:: Check if venv folder exists
if not exist venv goto create_venv

echo [INFO] Virtual environment found. Activating...
call .\venv\Scripts\activate.bat
goto start_app

:create_venv
echo [INFO] Creating virtual environment (venv)...
python -m venv venv
if %errorlevel% neq 0 goto venv_fail

echo [INFO] Virtual environment created successfully!
echo.
echo [INFO] Installing requirements.txt. Please wait 1-2 minutes...
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 goto install_fail
echo [INFO] Requirements installed successfully!
echo.
goto start_app

:venv_fail
echo [ERROR] Failed to create virtual environment!
pause
exit /b

:install_fail
echo [ERROR] Failed to install requirements! Please check your network.
pause
exit /b

:start_app
echo.
echo ===================================================
echo     Starting Streamlit Web Application...
echo     Opening browser at http://localhost:8501
echo     Please do NOT close this window.
echo ===================================================
echo.

:: Start Streamlit (forces opening exactly one browser page)
streamlit run app.py --server.headless false

if %errorlevel% neq 0 goto app_fail
exit /b

:app_fail
echo.
echo [INFO] Application exited or closed.
pause
