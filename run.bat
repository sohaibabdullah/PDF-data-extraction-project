@echo off
echo Starting Automation Pipeline...

:: 1. Activate the environment
call venv\Scripts\activate

:: 2. Run the main script
python main.py

:: 3. Keep window open if it crashes so they can see the error
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The program crashed. See message above.
    pause
) else (
    echo.
    echo Program finished successfully.
    pause
)