# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**isaac-sim-jetbot-keyboard** is a keyboard-controlled Jetbot mobile robot teleoperation system with demonstration recording and reinforcement learning training pipeline for Isaac Sim 5.0.0.

The Jetbot is a differential-drive mobile robot with two wheels, controlled via keyboard for navigation tasks.

## Architecture

### Core Components

1. **JetbotKeyboardController** (`src/jetbot_keyboard_control.py`)
   - Main application orchestrating keyboard input, simulation, and TUI
   - Thread-safe keyboard handling via pynput
   - Rich terminal UI for real-time telemetry

2. **JetbotNavigationEnv** (`src/jetbot_rl_env.py`)
   - Gymnasium-compatible RL environment
   - Navigation task: drive to goal position
   - Dense/sparse reward modes

3. **Supporting Scripts**
   - `train_rl.py` - PPO training with optional BC warmstart
   - `eval_policy.py` - Policy evaluation and metrics
   - `train_bc.py` - Behavioral cloning from demonstrations
   - `replay.py` - Demo playback and inspection

### Key Classes

- **TUIRenderer**: Rich-based terminal UI for robot state display
- **SceneManager**: Manages goal markers and scene objects
- **DemoRecorder**: Records (obs, action, reward, done) tuples to NPZ
- **DemoPlayer**: Loads and replays recorded demonstrations
- **ActionMapper**: Maps keyboard keys to velocity commands
- **ObservationBuilder**: Builds observation vectors from robot state
- **RewardComputer**: Computes navigation rewards

## Keyboard Controls

```
Movement:
  W - Forward
  S - Backward
  A - Turn left
  D - Turn right
  Space - Stop (emergency brake)

Recording:
  ` (backtick) - Toggle recording
  [ - Mark episode success
  ] - Mark episode failure

System:
  R - Reset robot to start
  G - Spawn new random goal
  Esc - Exit application
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
# Teleoperation
./run.sh

# With recording enabled
./run.sh --enable-recording

# Training
./run.sh train_rl.py --headless --timesteps 500000

# Evaluation
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Tests
./run_tests.sh
```

## File Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── eval_policy.py                # Policy evaluation
│   ├── train_bc.py                   # Behavioral cloning
│   ├── replay.py                     # Demo playback
│   ├── test_jetbot_keyboard_control.py
│   └── test_jetbot_rl_env.py
├── demos/                            # Recorded demonstrations
├── models/                           # Trained models
├── runs/                             # TensorBoard logs
├── run.sh                            # Isaac Sim Python launcher
├── run_tests.sh                      # Test runner
└── README.md
```

## Isaac Sim Integration

- Uses `isaacsim.robot.wheeled_robots` for Jetbot control
- `DifferentialController` for wheel velocity conversion
- SimulationApp must be instantiated FIRST before any Isaac imports

## Testing

Tests use pytest with mocked Isaac Sim imports:
```bash
./run_tests.sh           # All tests
./run_tests.sh -v        # Verbose
./run_tests.sh -k name   # Specific test
```

## Dependencies

- Isaac Sim 5.0.0
- pynput (keyboard input)
- rich (terminal UI)
- numpy

Optional for RL:
- stable-baselines3
- gymnasium
- tensorboard
