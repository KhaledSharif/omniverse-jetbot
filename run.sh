#!/bin/bash
# Universal Isaac Sim Python script launcher
#
# Usage:
#   ./run.sh                             # Run jetbot_keyboard_control.py
#   ./run.sh --enable-recording          # Run with recording enabled
#   ./run.sh replay.py demos/file.npz    # Run replay script
#   ./run.sh train_bc.py demos/file.npz  # Run training script

ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script to point to your Isaac Sim installation."
    exit 1
fi

# Determine which script to run
if [ $# -gt 0 ] && [[ "$1" == *.py ]]; then
    # First arg is a .py file - run that script with remaining args
    SCRIPT="$1"
    shift

    # If script doesn't exist as-is, try src/ directory
    if [ ! -f "$SCRIPT" ]; then
        SCRIPT="src/$SCRIPT"
    fi

    echo "Running $SCRIPT with Isaac Sim's Python..."
    $ISAAC_PYTHON "$SCRIPT" "$@"
else
    # No .py file specified - run main app with all args
    echo "Running Jetbot keyboard control with Isaac Sim's Python..."
    $ISAAC_PYTHON src/jetbot_keyboard_control.py "$@"
fi
