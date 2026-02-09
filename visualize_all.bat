@echo off
REM FastTracker - Generate All Visualizations
REM Generates all static visualizations (dashboard, quality report, 3D, IMM)

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - Generate All Visualizations
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

echo This will generate:
echo   1. Performance Dashboard (dashboard.png)
echo   2. Track Quality Report (quality_report.png + .txt)
echo   3. 3D Trajectory Visualization (trajectory_3d.html)
echo   4. IMM Analysis (imm_analysis.png + imm_stats.txt)
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

REM Create timestamp for this batch
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

REM Set Python path for all visualizations
set PYTHONPATH=python

echo.
echo ======================================================
echo [1/4] Generating Performance Dashboard...
echo ======================================================
py -3.13 -m visualization.performance_dashboard ^
    --path . ^
    --results results.csv ^
    --eval evaluation_results.csv ^
    --output dashboard_%timestamp%.png ^
    --no-show

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Dashboard generation failed, continuing...
) else (
    echo SUCCESS: dashboard_%timestamp%.png created
)

echo.
echo ======================================================
echo [2/4] Generating Track Quality Report...
echo ======================================================
py -3.13 -m visualization.track_quality_report ^
    --path . ^
    --results results.csv ^
    --eval evaluation_results.csv ^
    --tracks track_details.csv ^
    --output-plot quality_report_%timestamp%.png ^
    --output-text quality_report_%timestamp%.txt ^
    --no-show

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Quality report generation failed, continuing...
) else (
    echo SUCCESS: quality_report_%timestamp%.png and .txt created
)

echo.
echo ======================================================
echo [3/4] Generating 3D Trajectory Visualization...
echo ======================================================
py -3.13 -m visualization.trajectory_3d ^
    --path . ^
    --tracks track_details.csv ^
    --results results.csv ^
    --color-by state ^
    --output trajectory_3d_%timestamp%.html ^
    --no-show

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: 3D visualization generation failed, continuing...
) else (
    echo SUCCESS: trajectory_3d_%timestamp%.html created
)

echo.
echo ======================================================
echo [4/4] Generating IMM Analysis...
echo ======================================================
py -3.13 -m visualization.imm_analyzer ^
    --path . ^
    --tracks track_details.csv ^
    --output-plot imm_analysis_%timestamp%.png ^
    --output-stats imm_stats_%timestamp%.txt ^
    --no-show

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: IMM analysis generation failed, continuing...
) else (
    echo SUCCESS: imm_analysis_%timestamp%.png and .txt created
)

echo.
echo ======================================================
echo All visualizations completed!
echo ======================================================
echo.
echo Generated files (with timestamp %timestamp%):
echo   - dashboard_%timestamp%.png
echo   - quality_report_%timestamp%.png
echo   - quality_report_%timestamp%.txt
echo   - trajectory_3d_%timestamp%.html
echo   - imm_analysis_%timestamp%.png
echo   - imm_stats_%timestamp%.txt
echo.
echo Opening dashboard...
if exist "dashboard_%timestamp%.png" (
    start dashboard_%timestamp%.png
)

echo.
pause
