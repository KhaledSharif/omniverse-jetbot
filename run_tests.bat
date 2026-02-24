@echo off
REM Convenience wrapper to run pytest tests using Isaac Sim's Python interpreter
REM
REM Usage:
REM   run_tests.bat                    Run all tests
REM   run_tests.bat -v                 Run with verbose output
REM   run_tests.bat -k test_name       Run specific test
REM   run_tests.bat --cov              Run with coverage

set "ISAAC_PYTHON=C:\isaac-sim\python.bat"

REM Check if Isaac Sim Python exists
if not exist "%ISAAC_PYTHON%" (
    echo Error: Isaac Sim Python not found at %ISAAC_PYTHON%
    echo Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation.
    exit /b 1
)

REM Run pytest with all passed arguments
echo Running tests with Isaac Sim's Python...
call "%ISAAC_PYTHON%" -m pytest src %*
