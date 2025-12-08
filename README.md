# Isaac Sim Jetbot Keyboard Control

Keyboard-controlled Jetbot mobile robot teleoperation with ROS2 integration, RViz visualization, demonstration recording, and reinforcement learning training for NVIDIA Isaac Sim 5.0.0.

## Features

- **Keyboard Teleoperation**: Control the Jetbot using WASD keys with Ctrl+K capture toggle
- **ROS2 Integration**: Publishes camera, odometry, TF, and clock topics via Isaac Sim's internal ROS2 bridge
- **RViz Visualization**: Pre-configured RViz setup with RGB/depth cameras, TF tree, odometry, and point cloud
- **Rich Terminal UI**: Real-time robot state display with visual feedback
- **Camera Streaming**: GStreamer H264 RTP UDP streaming from Jetbot camera
- **Random Obstacles**: Configurable static obstacles for navigation challenge
- **Demonstration Recording**: Record navigation trajectories for imitation learning
- **RL Training Pipeline**: Train PPO agents with optional behavioral cloning warmstart

## Requirements

- NVIDIA Isaac Sim 5.0.0 standalone
- NVIDIA RTX GPU
- Python 3.11 (bundled with Isaac Sim)

### Python Dependencies

```bash
~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install numpy pynput rich stable-baselines3 tensorboard gymnasium imitation
```

### Camera Streaming Dependencies (Optional)

For camera streaming functionality, install GStreamer and PyGObject:

```bash
# System packages (Ubuntu/Debian)
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
  libcairo2-dev libgirepository-2.0-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# Python package (in Isaac Sim environment)
~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install PyGObject
```

## Quick Start

### Basic Teleoperation

```bash
./run.sh
```

### With ROS2 Bridge (for RViz visualization)

```bash
# Terminal 1: Start Isaac Sim with ROS2 bridge
./run_ros2.sh

# Terminal 2: Start RViz (requires ROS2 Jazzy)
./rviz/view_jetbot.sh
```

### With Recording Enabled

```bash
./run.sh --enable-recording
```

### Train RL Agent

```bash
./run.sh train_rl.py --headless --timesteps 500000
```

## Controls

| Key | Action |
|-----|--------|
| Ctrl+K | Toggle keyboard capture (must enable first) |
| W | Move forward |
| S | Move backward |
| A | Turn left |
| D | Turn right |
| Space | Stop (emergency brake) |
| R | Reset robot position |
| G | Spawn new goal |
| C | Toggle camera viewer |
| Esc | Exit (always active) |

### Recording Controls (when enabled with `--enable-recording`)

| Key | Action |
|-----|--------|
| ` (backtick) | Toggle recording |
| [ | Mark episode success |
| ] | Mark episode failure |

## Project Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── ros2_bridge.py                # ROS2 OmniGraph bridge
│   ├── camera_streamer.py            # GStreamer camera streaming
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── eval_policy.py                # Policy evaluation
│   ├── train_bc.py                   # Behavioral cloning
│   ├── replay.py                     # Demo playback
│   └── test_*.py                     # Test files
├── rviz/
│   ├── jetbot.rviz                   # RViz configuration
│   └── view_jetbot.sh                # RViz launch script
├── demos/                            # Recorded demonstrations
├── models/                           # Trained models
├── runs/                             # TensorBoard logs
├── run.sh                            # Basic teleoperation launcher
├── run_ros2.sh                       # ROS2-enabled launcher
└── run_tests.sh                      # Test runner
```

## ROS2 Integration

The ROS2 bridge uses Isaac Sim's internal ROS2 libraries via OmniGraph nodes (no system ROS2 sourcing needed for Isaac Sim).

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/jetbot/camera/rgb/image_raw` | sensor_msgs/Image | RGB camera |
| `/jetbot/camera/depth/image_raw` | sensor_msgs/Image | Depth camera |
| `/jetbot/camera/camera_info` | sensor_msgs/CameraInfo | RGB camera intrinsics |
| `/jetbot/camera/depth/camera_info` | sensor_msgs/CameraInfo | Depth camera intrinsics |
| `/jetbot/odom` | nav_msgs/Odometry | Robot odometry |
| `/tf` | tf2_msgs/TFMessage | Transform tree (world → chassis) |
| `/clock` | rosgraph_msgs/Clock | Simulation clock |

### RViz Prerequisites (ROS2 Jazzy)

```bash
sudo apt install -y ros-jazzy-rviz2 ros-jazzy-depth-image-proc
```

### Running with RViz

```bash
# Terminal 1: Isaac Sim (don't source system ROS2)
./run_ros2.sh

# Terminal 2: RViz (source system ROS2)
./rviz/view_jetbot.sh
```

## Usage Examples

```bash
# Teleoperation
./run.sh
./run.sh --num-obstacles 10 --enable-recording

# ROS2 mode
./run_ros2.sh

# Training
./run.sh train_rl.py --headless --timesteps 500000

# Evaluation
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Demo playback
./run.sh replay.py demos/recording.npz --info

# Tests
./run_tests.sh
```

## Observation Space (10D)

| Index | Description | Range |
|-------|-------------|-------|
| 0-1 | Robot position (x, y) | meters |
| 2 | Robot heading (theta) | radians |
| 3 | Linear velocity | m/s |
| 4 | Angular velocity | rad/s |
| 5-6 | Goal position (x, y) | meters |
| 7 | Distance to goal | meters |
| 8 | Angle to goal | radians |
| 9 | Goal reached flag | 0 or 1 |

## Action Space (2D)

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Linear velocity | [-1, 1] -> [-0.3, 0.3] m/s |
| 1 | Angular velocity | [-1, 1] -> [-1.0, 1.0] rad/s |

## Reward Function

### Dense Mode (default)
- Distance reward: Positive for getting closer to goal
- Heading bonus: Reward for facing the goal
- Goal reached: +10.0 bonus

### Sparse Mode
- Goal reached: +10.0
- Otherwise: 0.0

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/
```

## Architecture

### Main Components

- **JetbotKeyboardController**: Main app with keyboard input and Rich TUI
- **ROS2Bridge**: OmniGraph-based ROS2 topic publishing
- **JetbotNavigationEnv**: Gymnasium-compatible RL environment
- **SceneManager**: Goal markers and obstacle spawning
- **CameraStreamer**: GStreamer H264 RTP UDP streaming

### Isaac Sim Integration

- `WheeledRobot` and `DifferentialController` for Jetbot control
- OmniGraph nodes for ROS2 bridge (camera, TF, odometry, clock)
- SimulationApp must be instantiated before any Isaac imports

## License

MIT License
