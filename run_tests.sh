#!/bin/bash
# Convenience wrapper to run pytest tests using Isaac Sim's Python interpreter
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh -v                 # Run with verbose output
#   ./run_tests.sh -k test_name       # Run specific test
#   ./run_tests.sh --cov              # Run with coverage

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Run pytest with all passed arguments
echo "Running tests with Isaac Sim's Python..."
$ISAAC_PYTHON -m pytest \
    src \
    "$@"
