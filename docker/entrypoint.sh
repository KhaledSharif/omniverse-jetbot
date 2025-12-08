#!/bin/bash
# Entrypoint script for cuVSLAM Docker container

set -e

# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Source Isaac ROS if available
if [ -f /opt/ros/isaac_ros/setup.bash ]; then
    source /opt/ros/isaac_ros/setup.bash
fi

# Add Isaac ROS GXF library paths for cuVSLAM
GXF_LIB_BASE=/opt/ros/humble/share/isaac_ros_gxf/gxf/lib
export LD_LIBRARY_PATH="${GXF_LIB_BASE}:${GXF_LIB_BASE}/core:${GXF_LIB_BASE}/std:${GXF_LIB_BASE}/cuda:${GXF_LIB_BASE}/serialization:${GXF_LIB_BASE}/multimedia:${GXF_LIB_BASE}/network:${GXF_LIB_BASE}/npp:${GXF_LIB_BASE}/behavior_tree:${LD_LIBRARY_PATH}"

# Set FastDDS as RMW implementation
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Use FastDDS config if available (mounted from host)
if [ -f /opt/ros/humble/.ros/fastdds.xml ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/ros/humble/.ros/fastdds.xml
    echo "[cuVSLAM] Using FastDDS config from /opt/ros/humble/.ros/fastdds.xml"
fi

# Inherit ROS_DOMAIN_ID from environment (default 0)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
echo "[cuVSLAM] ROS_DOMAIN_ID: $ROS_DOMAIN_ID"

# Execute the command
exec "$@"
