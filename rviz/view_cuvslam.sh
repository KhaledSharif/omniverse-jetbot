#!/bin/bash
# Launch RViz2 for cuVSLAM visualization
# Run this in a separate terminal AFTER starting:
#   1. Isaac Sim: ./run_slam.sh
#   2. cuVSLAM:   ./docker/run_cuvslam.sh
#
# Prerequisites:
#   sudo apt install -y ros-jazzy-rviz2

source /opt/ros/jazzy/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="$SCRIPT_DIR/cuvslam.rviz"

if [ ! -f "$RVIZ_CONFIG" ]; then
    echo "Error: RViz config not found at $RVIZ_CONFIG"
    exit 1
fi

echo "=========================================="
echo "Starting RViz2 for cuVSLAM Visualization"
echo "=========================================="
echo ""
echo "Make sure these are running first:"
echo "  Terminal 1: ./run_slam.sh        (Isaac Sim)"
echo "  Terminal 2: ./docker/run_cuvslam.sh  (cuVSLAM)"
echo ""
echo "What you'll see in RViz:"
echo "  - Green points:   SLAM Landmarks (3D map)"
echo "  - Yellow points:  Current observations"
echo "  - Magenta path:   SLAM trajectory"
echo "  - Cyan path:      Visual odometry path"
echo "  - Orange arrow:   SLAM pose estimate"
echo "  - Red arrow:      Ground truth (Isaac Sim)"
echo "  - TF frames:      Robot coordinate frames"
echo ""
echo "Drive the Jetbot with W/A/S/D in Isaac Sim to build the map!"
echo ""

# Start RViz2
rviz2 -d "$RVIZ_CONFIG" --ros-args -p use_sim_time:=true
