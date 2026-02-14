@echo off
setlocal enabledelayedexpansion
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

echo Select visualization mode:
echo   1. Static (full trajectories) - Default
echo   2. Animated (time progression)
echo.
set /p mode_choice="Enter choice (1-2) [default: 1]: "

set ANIMATE_FLAG=
set FRAME_STEP=5
set ANIM_SPEED=100

if "%mode_choice%"=="2" (
    set ANIMATE_FLAG=--animate
    echo.
    echo Animation settings:
    echo   Frame step: higher = faster but less smooth [default: 5]
    set /p frame_step_input="Enter frame step (1-10): "
    if not "%frame_step_input%"=="" set FRAME_STEP=%frame_step_input%

    echo   Animation speed: milliseconds per frame [default: 100]
    set /p speed_input="Enter speed (50-500): "
    if not "%speed_input%"=="" set ANIM_SPEED=%speed_input%
)

echo.
if "%ANIMATE_FLAG%"=="" (
    echo Select color scheme:
    echo   1. State (Tentative/Confirmed/Lost) - Default
    echo   2. Track ID
    echo   3. IMM Model Probability
    echo.
    set /p choice="Enter choice (1-3) [default: 1]: "

    set COLOR_MODE=state
    if "!choice!"=="2" set COLOR_MODE=track_id
    if "!choice!"=="3" set COLOR_MODE=model_prob

    echo.
    echo Options:
    echo   - Include ground truth? (y/n) [default: y]
    set /p show_gt="Enter choice: "

    set GT_FLAG=
    if /i "!show_gt!"=="n" set GT_FLAG=--no-ground-truth
) else (
    set COLOR_MODE=state
    set GT_FLAG=
)

echo.
echo Loading data and generating 3D visualization...
echo This may take a few seconds for large datasets...
echo.

REM Run the 3D trajectory visualizer
set PYTHONPATH=python
if "%ANIMATE_FLAG%"=="--animate" (
    py -3.13 -m visualization.trajectory_3d ^
        --path . ^
        --tracks track_details.csv ^
        --results results.csv ^
        --animate ^
        --frame-step %FRAME_STEP% ^
        --speed %ANIM_SPEED% ^
        --output trajectory_3d_animated.html
) else (
    py -3.13 -m visualization.trajectory_3d ^
        --path . ^
        --tracks track_details.csv ^
        --results results.csv ^
        --color-by %COLOR_MODE% ^
        --output trajectory_3d.html ^
        %GT_FLAG%
)

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
if "%ANIMATE_FLAG%"=="--animate" (
    echo Animated 3D visualization generated successfully:
    echo   - trajectory_3d_animated.html
    echo ======================================================
    echo.
    echo Use Play/Pause buttons and time slider to control animation.
) else (
    echo 3D visualization generated successfully:
    echo   - trajectory_3d.html
    echo ======================================================
)
echo.
echo Opening in default browser...

REM Open the HTML file in default browser
if "%ANIMATE_FLAG%"=="--animate" (
    if exist "trajectory_3d_animated.html" (
        start trajectory_3d_animated.html
    )
) else (
    if exist "trajectory_3d.html" (
        start trajectory_3d.html
    )
)

pause
