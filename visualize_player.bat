@echo off
REM FastTracker - Real-time Trajectory Player
REM Interactive GUI for replaying track trajectories

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ======================================================
echo FastTracker - Real-time Trajectory Player
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

echo Select playback speed:
echo   1. 0.25x (Slow motion)
echo   2. 0.5x
echo   3. 1.0x (Real-time) - Default
echo   4. 2.0x
echo   5. 5.0x (Fast forward)
echo.
set /p speed_choice="Enter choice (1-5) [default: 3]: "

set SPEED=1.0
if "%speed_choice%"=="1" set SPEED=0.25
if "%speed_choice%"=="2" set SPEED=0.5
if "%speed_choice%"=="3" set SPEED=1.0
if "%speed_choice%"=="4" set SPEED=2.0
if "%speed_choice%"=="5" set SPEED=5.0

echo.
echo Trail length (number of past positions to show):
set /p trail="Enter trail length [default: 30]: "
if "%trail%"=="" set trail=30

echo.
echo Starting trajectory player...
echo.
echo Controls:
echo   - Play/Pause: Toggle playback
echo   - Stop: Reset to beginning
echo   - Slider: Jump to specific frame
echo   - Speed: Change playback speed during playback
echo.

REM Run the trajectory player
set PYTHONPATH=python
py -3.13 -m visualization.trajectory_player ^
    --path . ^
    --results results.csv ^
    --tracks track_details.csv ^
    --speed %SPEED% ^
    --trail %trail%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Trajectory player failed to start!
    echo Make sure Python and required libraries are installed:
    echo   pip install pandas numpy pyqtgraph PyQt5
    echo.
    echo NOTE: pyqtgraph and PyQt5 are required for the GUI player.
    echo Install with: pip install pyqtgraph PyQt5
    echo.
    pause
    exit /b 1
)

pause
