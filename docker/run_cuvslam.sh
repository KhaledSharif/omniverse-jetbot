#!/bin/bash
# Run cuVSLAM Docker container for Isaac Sim Jetbot
#
# This script launches the cuVSLAM container with proper GPU and network configuration
# for communicating with Isaac Sim running on the host.
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit
#   - Isaac Sim running with ./run_slam.sh (publishes stereo + IMU topics)
#   - FastDDS config at ~/.ros/fastdds.xml
#
# Usage:
#   ./docker/run_cuvslam.sh              # Run cuVSLAM
#   ./docker/run_cuvslam.sh bash         # Interactive shell
#   ./docker/run_cuvslam.sh build        # Build the image first

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="jetbot-cuvslam:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for NVIDIA GPU
check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: NVIDIA GPU not detected. cuVSLAM requires NVIDIA GPU.${NC}"
        exit 1
    fi
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

# Build the Docker image
build_image() {
    echo -e "${GREEN}Building cuVSLAM Docker image...${NC}"
    cd "$PROJECT_DIR"
    docker build -f docker/Dockerfile.cuvslam -t "$IMAGE_NAME" .
    echo -e "${GREEN}Build complete: $IMAGE_NAME${NC}"
}

# Run the container
run_container() {
    local CMD="${1:-}"

    # Check FastDDS config
    if [ ! -f ~/.ros/fastdds.xml ]; then
        echo -e "${YELLOW}WARNING: ~/.ros/fastdds.xml not found. SHM transport may not work.${NC}"
    fi

    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo -e "${YELLOW}Image $IMAGE_NAME not found. Building...${NC}"
        build_image
    fi

    echo -e "${GREEN}Starting cuVSLAM container...${NC}"
    echo -e "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
    echo ""

    # Docker run command
    if [ "$CMD" = "bash" ]; then
        # Interactive shell
        docker run --rm -it \
            --gpus all \
            --privileged \
            --network host \
            --ipc host \
            --pid host \
            -v ~/.ros/fastdds.xml:/opt/ros/humble/.ros/fastdds.xml:ro \
            -v /dev/shm:/dev/shm \
            -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
            -e DISPLAY="$DISPLAY" \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            "$IMAGE_NAME" \
            bash
    else
        # Run cuVSLAM
        docker run --rm -it \
            --gpus all \
            --privileged \
            --network host \
            --ipc host \
            --pid host \
            -v ~/.ros/fastdds.xml:/opt/ros/humble/.ros/fastdds.xml:ro \
            -v /dev/shm:/dev/shm \
            -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
            "$IMAGE_NAME"
    fi
}

# Main
case "${1:-}" in
    build)
        check_gpu
        build_image
        ;;
    bash)
        check_gpu
        run_container bash
        ;;
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (none)   Run cuVSLAM node"
        echo "  build    Build the Docker image"
        echo "  bash     Start interactive shell in container"
        echo "  help     Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  ROS_DOMAIN_ID  ROS 2 domain ID (default: 0)"
        ;;
    *)
        check_gpu
        run_container
        ;;
esac
