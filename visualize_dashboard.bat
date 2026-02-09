@echo off
REM FastTracker - Performance Dashboard Visualizer
REM Generates a comprehensive performance dashboard from simulation results

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - Performance Dashboard Generator
echo ======================================================
echo.

REM Check if results.csv exists
if not exist "results.csv" (
    echo ERROR: results.csv not found!
    echo Please run a simulation first using fasttracker.exe
    echo.
    pause
    exit /b 1
)

echo Loading data and generating dashboard...
echo.

REM Run the dashboard generator
set PYTHONPATH=python
py -3.13 -m visualization.performance_dashboard ^
    --path . ^
    --results results.csv ^
    --eval evaluation_results.csv ^
    --output dashboard.png

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Dashboard generation failed!
    echo Make sure Python and required libraries are installed:
    echo   pip install pandas numpy matplotlib
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Dashboard generated successfully: dashboard.png
echo ======================================================
echo.

REM Open the generated image
if exist "dashboard.png" (
    start dashboard.png
)

pause
