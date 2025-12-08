# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**isaac-sim-jetbot-keyboard** is a keyboard-controlled Jetbot teleoperation system with ROS2 integration, RViz visualization, and RL training for Isaac Sim 5.0.0.

## Architecture

### Core Components

1. **JetbotKeyboardController** (`src/jetbot_keyboard_control.py`)
   - Main application with pynput keyboard input and Rich TUI
   - Ctrl+K toggle for keyboard capture

2. **ROS2Bridge** (`src/ros2_bridge.py`)
   - OmniGraph-based ROS2 topic publishing
   - Topics: camera (RGB/depth), odometry, TF, clock
   - Uses Isaac Sim's internal ROS2 libraries

3. **JetbotNavigationEnv** (`src/jetbot_rl_env.py`)
   - Gymnasium-compatible RL environment
   - Dense/sparse reward navigation task

### Key Classes

- **TUIRenderer**: Rich terminal UI for robot state
- **SceneManager**: Goal markers and obstacle spawning
- **ROS2Bridge**: OmniGraph ROS2 publishers
- **ActionMapper**: Keyboard to velocity mapping
- **ObservationBuilder**: State to observation vector
- **CameraStreamer**: GStreamer H264 RTP UDP streaming

## Keyboard Controls

```
Ctrl+K - Toggle keyboard capture (must enable first)

Movement (when capture enabled):
  W/S - Forward/Backward
  A/D - Turn left/right
  Space - Stop

System:
  R - Reset robot
  G - New random goal
  C - Toggle camera viewer
  Esc - Exit (always active)

Recording (--enable-recording):
  ` - Toggle recording
  [ / ] - Mark success/failure
```

## State & Action Spaces

### Observation Space (10D)
```
[0:2]  - Robot position (x, y)
[2]    - Robot heading (theta)
[3]    - Linear velocity
[4]    - Angular velocity
[5:7]  - Goal position (x, y)
[7]    - Distance to goal
[8]    - Angle to goal
[9]    - Goal reached flag
```

### Action Space (2D)
```
[0] - Linear velocity command (m/s)
[1] - Angular velocity command (rad/s)
```

## Running the Project

```bash
# Basic teleoperation
./run.sh

# With ROS2 bridge (for RViz)
./run_ros2.sh

# RViz visualization (separate terminal, requires ROS2 Jazzy)
./rviz/view_jetbot.sh

# With recording
./run.sh --enable-recording

# Training
./run.sh train_rl.py --headless --timesteps 500000

# Tests
./run_tests.sh
```

## File Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── ros2_bridge.py                # ROS2 OmniGraph bridge
│   ├── camera_streamer.py            # GStreamer streaming
│   ├── jetbot_rl_env.py              # RL environment
│   └── test_*.py                     # Test files
├── rviz/
│   ├── jetbot.rviz                   # RViz configuration
│   └── view_jetbot.sh                # RViz launch script
├── run.sh                            # Basic launcher
├── run_ros2.sh                       # ROS2-enabled launcher
└── run_tests.sh                      # Test runner
```

## Isaac Sim Integration

- `WheeledRobot` and `DifferentialController` for Jetbot control
- OmniGraph nodes for ROS2 bridge (isaacsim.ros2.bridge extension)
- SimulationApp must be instantiated FIRST before any Isaac imports

## ROS2 Topics

| Topic | Type |
|-------|------|
| `/jetbot/camera/rgb/image_raw` | sensor_msgs/Image |
| `/jetbot/camera/depth/image_raw` | sensor_msgs/Image |
| `/jetbot/camera/camera_info` | sensor_msgs/CameraInfo |
| `/jetbot/camera/depth/camera_info` | sensor_msgs/CameraInfo |
| `/jetbot/odom` | nav_msgs/Odometry |
| `/tf` | tf2_msgs/TFMessage |
| `/clock` | rosgraph_msgs/Clock |

## Testing

Tests use pytest with mocked Isaac Sim imports:
```bash
./run_tests.sh           # All tests (104 tests)
./run_tests.sh -v        # Verbose
./run_tests.sh -k name   # Specific test
```

## Dependencies

- Isaac Sim 5.0.0, pynput, rich, numpy
- Optional: GStreamer + PyGObject (camera streaming)
- Optional: stable-baselines3, gymnasium (RL)
- Optional: ROS2 Jazzy + ros-jazzy-rviz2, ros-jazzy-depth-image-proc (RViz)
