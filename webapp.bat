@echo off
REM FastTracker Web GUI
REM Launches the web application for trajectory generation and visualization

cd /d "%~dp0"

echo ======================================================
echo FastTracker Web GUI
echo ======================================================
echo.

set PYTHONPATH=python

echo Starting web server...
echo.
echo Open your browser at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

start http://localhost:5000

py -3.13 -m webapp.app

pause
