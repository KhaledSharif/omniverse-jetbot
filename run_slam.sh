#!/bin/bash
# Launch Isaac Sim Jetbot with Stereo Cameras and IMU for cuVSLAM
#
# This script starts Isaac Sim with the ROS2 bridge configured for cuVSLAM:
# - Stereo cameras (10cm baseline) publishing to /camera/left/* and /camera/right/*
# - IMU sensor publishing to /jetbot/imu
# - TF, odometry, and clock topics
#
# Usage:
#   ./run_slam.sh                    # Run Isaac Sim with stereo + IMU
#   ./run_slam.sh --help             # Show help
#
# After starting, run cuVSLAM in a separate terminal:
#   ./docker/run_cuvslam.sh
#
# And RViz in another terminal:
#   source /opt/ros/jazzy/setup.bash
#   rviz2 -d rviz/cuvslam.rviz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Isaac Sim Python path
ISAAC_PYTHON=~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if Isaac Sim Python exists
if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "Error: Isaac Sim Python not found at $ISAAC_PYTHON"
    echo "Please update the ISAAC_PYTHON variable in this script."
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected."
    exit 1
fi

# IMPORTANT: Do NOT source system ROS2
# Isaac Sim has its own internal ROS2 implementation

# Set ROS_DOMAIN_ID to match your system ROS2 (default 0)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Fast DDS configuration for cross-process communication
if [ -f ~/.ros/fastdds.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml
    echo -e "${GREEN}Using FastDDS SHM config${NC}"
else
    echo -e "${YELLOW}WARNING: ~/.ros/fastdds.xml not found${NC}"
    echo "Run Phase 3 setup to create FastDDS config for optimal performance."
fi

echo -e "${GREEN}=== Isaac Sim Jetbot with cuVSLAM Support ===${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Stereo baseline: ${CYAN}10cm${NC}"
echo -e "  Camera resolution: ${CYAN}640x480${NC}"
echo -e "  IMU: ${CYAN}Enabled${NC}"
echo -e "  ROS_DOMAIN_ID: ${CYAN}$ROS_DOMAIN_ID${NC}"
echo ""
echo -e "${GREEN}Published Topics:${NC}"
echo "  /camera/left/image_raw"
echo "  /camera/left/camera_info"
echo "  /camera/right/image_raw"
echo "  /camera/right/camera_info"
echo "  /jetbot/imu"
echo "  /jetbot/odom"
echo "  /tf"
echo "  /clock"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  Terminal 2: ./docker/run_cuvslam.sh"
echo "  Terminal 3: source /opt/ros/jazzy/setup.bash && rviz2 -d rviz/cuvslam.rviz"
echo ""

# Run Isaac Sim with ROS2 and stereo/IMU enabled
"$ISAAC_PYTHON" src/jetbot_keyboard_control.py --enable-ros2 "$@"
