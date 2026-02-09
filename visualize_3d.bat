@echo off
REM FastTracker - 3D Trajectory Visualizer
REM Creates interactive 3D trajectory visualization

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - 3D Trajectory Visualizer
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

echo Select color scheme:
echo   1. State (Tentative/Confirmed/Lost) - Default
echo   2. Track ID
echo   3. IMM Model Probability
echo.
set /p choice="Enter choice (1-3) [default: 1]: "

set COLOR_MODE=state
if "%choice%"=="2" set COLOR_MODE=track_id
if "%choice%"=="3" set COLOR_MODE=model_prob

echo.
echo Options:
echo   - Include ground truth? (y/n) [default: y]
set /p show_gt="Enter choice: "

set GT_FLAG=
if /i "%show_gt%"=="n" set GT_FLAG=--no-ground-truth

echo.
echo Loading data and generating 3D visualization...
echo This may take a few seconds for large datasets...
echo.

REM Run the 3D trajectory visualizer
set PYTHONPATH=python
py -3.13 -m visualization.trajectory_3d ^
    --path . ^
    --tracks track_details.csv ^
    --results results.csv ^
    --color-by %COLOR_MODE% ^
    --output trajectory_3d.html ^
    %GT_FLAG%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: 3D visualization generation failed!
    echo Make sure Python and required libraries are installed:
    echo   pip install pandas numpy plotly
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo 3D visualization generated successfully:
echo   - trajectory_3d.html
echo ======================================================
echo.
echo Opening in default browser...

REM Open the HTML file in default browser
if exist "trajectory_3d.html" (
    start trajectory_3d.html
)

pause
