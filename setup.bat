@echo off
setlocal
title Automation Project Setup

echo ==========================================
echo      SETTING UP AUTOMATION PROJECT
echo ==========================================

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python is already installed.
    goto :create_venv
)

:: --- PYTHON AUTO-INSTALLER SECTION ---
echo [!] Python is NOT found on this computer.
echo.
echo Downloading Python 3.11 installer...
echo (This may take a minute depending on internet speed)

:: Download Python 3.11 (Standard stable version)
curl -o python_installer.exe https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe

if not exist python_installer.exe (
    echo [ERROR] Failed to download Python. Check your internet connection.
    pause
    exit /b
)

echo.
echo [INSTALLING] Launching Python Installer...
echo [IMPORTANT] A Windows security popup will appear. Please click 'YES'.
echo.

:: Run Installer
:: /passive = Show progress bar but don't ask questions
:: PrependPath=1 = CRITICAL: Adds Python to Windows environment variables
python_installer.exe /passive InstallAllUsers=1 PrependPath=1 Include_test=0

:: Cleanup
del python_installer.exe

echo.
echo [REQUIRED] Python has been installed.
echo.
echo ************************************************************
echo * PLEASE CLOSE THIS WINDOW AND RUN 'setup.bat' AGAIN!      *
echo * (Windows needs a restart of the script to see Python)    *
echo ************************************************************
pause
exit
:: -------------------------------------

:create_venv
:: 2. Create Virtual Environment
if exist venv (
    echo [OK] Virtual environment already exists.
) else (
    echo [1/3] Creating Virtual Environment (venv)...
    python -m venv venv
)

:: 3. Activate and Install
echo [2/3] Installing/Updating Dependencies...
call venv\Scripts\activate

:: Upgrade pip first to avoid warnings
python -m pip install --upgrade pip >nul 2>&1

:: Install packages
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install requirements.
    pause
    exit /b
)

:: 4. Finish
echo.
echo [3/3] Setup Complete!
echo ==========================================
echo You can now run the project using 'run.bat'
echo ==========================================
pause