@echo off
REM FastTracker - Track Quality Report Generator
REM Generates detailed track quality analysis report

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - Track Quality Report Generator
echo ======================================================
echo.

REM Check if required files exist
if not exist "results.csv" (
    echo ERROR: results.csv not found!
    echo Please run a simulation first using fasttracker.exe
    echo.
    pause
    exit /b 1
)

if not exist "track_details.csv" (
    echo WARNING: track_details.csv not found!
    echo Some detailed analysis may not be available.
    echo.
)

echo Loading data and generating quality report...
echo.

REM Run the quality report generator
set PYTHONPATH=python
py -3.13 -m visualization.track_quality_report ^
    --path . ^
    --results results.csv ^
    --eval evaluation_results.csv ^
    --tracks track_details.csv ^
    --output-plot quality_report.png ^
    --output-text quality_report.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Quality report generation failed!
    echo Make sure Python and required libraries are installed:
    echo   pip install pandas numpy matplotlib
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo Quality report generated successfully:
echo   - quality_report.png
echo   - quality_report.txt
echo ======================================================
echo.

REM Open the generated files
if exist "quality_report.png" (
    start quality_report.png
)

if exist "quality_report.txt" (
    type quality_report.txt
    echo.
)

pause
