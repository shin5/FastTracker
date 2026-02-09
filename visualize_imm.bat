@echo off
REM FastTracker - IMM Filter Analyzer
REM Analyzes IMM (Interacting Multiple Model) filter behavior

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - IMM Filter Analyzer
echo ======================================================
echo.

REM Check if track_details.csv exists
if not exist "track_details.csv" (
    echo ERROR: track_details.csv not found!
    echo Please run a simulation first using fasttracker.exe
    echo.
    pause
    exit /b 1
)

echo Analyze specific track or all tracks?
echo   1. All tracks - Default
echo   2. Specific track ID
echo.
set /p choice="Enter choice (1-2) [default: 1]: "

set TRACK_FLAG=
if "%choice%"=="2" (
    set /p track_id="Enter track ID: "
    set TRACK_FLAG=--track-id !track_id!
)

echo.
echo Loading data and generating IMM analysis...
echo.

REM Run the IMM analyzer
set PYTHONPATH=python
py -3.13 -m visualization.imm_analyzer ^
    --path . ^
    --tracks track_details.csv ^
    --output-plot imm_analysis.png ^
    --output-stats imm_stats.txt ^
    %TRACK_FLAG%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: IMM analysis generation failed!
    echo Make sure Python and required libraries are installed:
    echo   pip install pandas numpy matplotlib
    echo.
    echo NOTE: IMM analysis requires track_details.csv with model probability data.
    echo Make sure the simulation was run with IMM filter enabled.
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo IMM analysis generated successfully:
echo   - imm_analysis.png
echo   - imm_stats.txt
echo ======================================================
echo.

REM Display statistics
if exist "imm_stats.txt" (
    echo Statistics summary:
    echo.
    type imm_stats.txt
    echo.
)

REM Open the generated plot
if exist "imm_analysis.png" (
    start imm_analysis.png
)

pause
