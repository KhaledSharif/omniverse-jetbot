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

3. **Training Pipeline** (`src/train_rl.py`)
   - PPO training with BC warmstart from demonstrations
   - VecNormalize pre-warming from demo data (critical for BC→RL transfer)
   - Critic pretraining on Monte Carlo returns
   - Pipeline: validate → prewarm VecNormalize → BC warmstart → critic pretrain → PPO

4. **SAC/TQC + RLPD Pipeline** (`src/train_sac.py`)
   - TQC (sb3-contrib) with SAC fallback
   - RLPD-style demo/online replay buffer sampling (configurable via `--demo-ratio`)
   - LayerNorm in critics (replaces VecNormalize)
   - UTD ratio configurable via `--utd-ratio` (default 20, recommended 5 for speed)
   - No pretraining phases — demos sampled continuously
   - `--resume` to continue training from a checkpoint (step counter, weights preserved)

5. **Supporting Scripts**
   - `eval_policy.py` - Policy evaluation and metrics (auto-detects TQC/SAC/PPO)
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
- **CameraStreamer**: GStreamer H264 RTP UDP camera streaming (`src/camera_streamer.py`)
- **LidarSensor**: Analytical 2D raycasting for obstacle detection (no physics dependency)
- **OccupancyGrid**: 2D boolean grid from obstacle geometry, inflated by robot radius for C-space planning
- **AutoPilot**: A*-based expert controller with privileged scene access for collision-free demo collection

### Training Pipeline Functions (`src/train_rl.py`)

- **prewarm_vecnormalize()**: Seeds VecNormalize's `obs_rms` and `ret_rms` running statistics from demo data so BC/critic pretraining operates on the same normalized scale as PPO
- **normalize_obs()**: Applies `(obs - mean) / std` clipped to `[-clip_obs, clip_obs]`, matching VecNormalize's transform
- **normalize_returns()**: Applies `returns / std` clipped to `[-clip_reward, clip_reward]`, matching VecNormalize's reward normalization (no mean subtraction)
- **bc_warmstart()**: Pretrains PPO actor on normalized demo observations via MSE loss
- **pretrain_critic()**: Pretrains PPO critic on normalized observations and scaled MC returns

## Keyboard Controls

```
Movement:
  W - Forward
  S - Backward
  A - Turn left
  D - Turn right
  Space - Stop (emergency brake)

Camera:
  C - Toggle camera viewer (starts/stops streaming)

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

### Observation Space (34D for RL env, 10D base for keyboard control)
```
[0:2]   - Robot position (x, y)
[2]     - Robot heading (theta)
[3]     - Linear velocity
[4]     - Angular velocity
[5:7]   - Goal position (x, y)
[7]     - Distance to goal
[8]     - Angle to goal
[9]     - Goal reached flag
[10:34] - LiDAR: 24 normalized distances (0=touching, 1=max range), 180° FOV
```

The RL environment (`JetbotNavigationEnv`) always uses 34D observations with LiDAR.
The keyboard controller uses 10D by default; pass `--use-lidar` for 34D.

### Action Space (2D)
```
[0] - Linear velocity command (m/s)
[1] - Angular velocity command (rad/s)
```

## Running the Project

```bash
# Teleoperation (camera streaming enabled by default)
./run.sh

# Disable camera streaming
./run.sh --no-camera

# Custom camera port
./run.sh --camera-port 5601

# With recording enabled
./run.sh --enable-recording

# Automatic demo collection (100 episodes)
./run.sh --enable-recording --automatic

# Custom episode count
./run.sh --enable-recording --automatic --num-episodes 200

# Continuous mode (run until Esc)
./run.sh --enable-recording --automatic --continuous

# Headless TUI (console progress prints)
./run.sh --enable-recording --automatic --num-episodes 200 --headless-tui

# Collect 34D demos with LiDAR observations
# Note: --automatic now forces --use-lidar automatically
./run.sh --enable-recording --automatic --use-lidar

# PPO Training (from scratch)
./run.sh train_rl.py --headless --timesteps 500000

# PPO Training with BC warmstart
./run.sh train_rl.py --headless --bc-warmstart demos/recording.npz --timesteps 1000000

# SAC/TQC + RLPD Training (recommended — demos in replay buffer, no pretraining needed)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000

# SAC/TQC with custom UTD ratio and buffer size
./run.sh train_sac.py --demos demos/recording.npz --headless --utd-ratio 20 --buffer-size 300000

# SAC/TQC with custom demo ratio (75% demo, 25% online)
./run.sh train_sac.py --demos demos/recording.npz --headless --demo-ratio 0.75

# Resume SAC/TQC training from checkpoint
./run.sh train_sac.py --demos demos/recording.npz --headless --resume models/checkpoints/tqc_jetbot_50000_steps.zip --timesteps 500000

# Evaluation (auto-detects TQC/SAC/PPO)
./run.sh eval_policy.py models/tqc_jetbot.zip --episodes 100
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Tests
./run_tests.sh
```

## File Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── camera_streamer.py            # Camera streaming module
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── train_sac.py                  # SAC/TQC + RLPD training script
│   ├── eval_policy.py                # Policy evaluation (auto-detects TQC/SAC/PPO)
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

## BC → PPO Normalization (Critical Design Constraint)

The RL training pipeline uses SB3's `VecNormalize` to z-score normalize observations and scale rewards at runtime. When using BC warmstart, the demo data must be normalized to the **same scale** the policy will see during PPO training. Failing to do this destroys the BC-learned policy at the RL transition.

**Required pipeline order:**
1. Create VecNormalize-wrapped env
2. `prewarm_vecnormalize()` — seed `obs_rms`/`ret_rms` from demo data
3. `bc_warmstart()` — train actor on **normalized** demo observations
4. `pretrain_critic()` — train critic on **normalized** observations and **scaled** MC returns
5. `model.learn()` — VecNormalize stats are consistent with pretraining

**What goes wrong without pre-warming:**
- BC trains on raw obs (e.g., position ±2m), but PPO feeds normalized obs (~±1σ) — policy outputs garbage
- Critic predicts raw-scale values (~40) but sees normalized rewards (~±1) — advantage estimates explode
- KL divergence spikes → early stopping throttles learning → exploration freezes

**Key rule:** Any function that feeds observations/returns to the policy or critic network must normalize them using VecNormalize's running stats first.

## Training Performance & Bottlenecks

### UTD Ratio is the Dominant Cost (Not Physics)

The primary bottleneck in SAC/TQC training is **gradient computation, not `world.step()`**. With UTD=20 (default), each env step triggers 20 backward passes through TQC's 5 critic networks + actor. Measured on RTX 3090 Ti with 50 obstacles:

| UTD Ratio | Steps/s | Time for 1M steps |
|-----------|---------|-------------------|
| 20        | ~2.9    | ~4 days            |
| 5         | ~10-12  | ~24 hours          |
| 1         | ~39     | ~7 hours           |

The env step itself (`world.step()`) takes only ~25ms. At UTD=20, the 20 gradient updates add ~330ms on top.

**Recommendation:** Use `--utd-ratio 5` for a good speed/sample-efficiency tradeoff. UTD=20 (RLPD paper default) is only worth it if you can afford multi-day runs.

### Obstacle Types: Visual vs Fixed

Obstacles use `Visual*` primitives (VisualCuboid, VisualCylinder, etc.) instead of `Fixed*`. This removes them from PhysX broadphase collision. The analytical LiDAR (`LidarSensor`) only reads `obstacle_metadata` (2D position + radius), not physics colliders, so Visual obstacles work identically. This provides a small speedup by reducing PhysX overhead.

### Headless SimulationApp Optimizations

In headless mode, `jetbot_rl_env.py` applies several optimizations:
- `disable_viewport_updates=True` — skips viewport render products
- `anti_aliasing=0` — disables AA if any rendering leaks through
- `rendering_dt=1.0` — decouples render scheduling from physics (effectively never renders)
- `numThreads=0` — single-threaded physics (faster for single-robot scenes)
- Reduced solver iterations and CPU broadphase (MBP) via `_optimize_physics_scene()`

These collectively save a few ms per step but are minor compared to UTD ratio.

### What Does NOT Help Much

- **`disable_viewport_updates`**: Small effect since `--no-window` already suppresses most rendering
- **CPU vs GPU physics**: Marginal for single-robot scenes — PhysX auto-selects well
- **Reducing reset settle steps**: Saves 8 steps per episode (~200ms/reset), negligible over 1M steps

## Testing

Tests use pytest with mocked Isaac Sim imports:
```bash
./run_tests.sh           # All tests
./run_tests.sh -v        # Verbose
./run_tests.sh -k name   # Specific test
```

## Reward Function

### Dense Mode (default)
- **Goal reached**: +10.0 (terminal)
- **Collision**: -10.0 (terminal, LiDAR distance < 0.08m)
- **Distance shaping**: `(prev_dist - curr_dist) * 1.0`
- **Heading bonus**: `((pi - |angle_to_goal|) / pi) * 0.1`
- **Proximity penalty**: `0.1 * (1.0 - min_lidar / 0.3)` when min_lidar < 0.3m
- **Time penalty**: -0.005 per step

### Sparse Mode
- **Goal reached**: +10.0
- **Collision**: -10.0
- **Otherwise**: 0.0

## Dependencies

- Isaac Sim 5.0.0
- pynput (keyboard input)
- rich (terminal UI)
- numpy

Optional for camera streaming:
- GStreamer 1.0 and plugins (gstreamer1.0-tools, gstreamer1.0-plugins-base/good/bad, gstreamer1.0-libav)
- PyGObject (python3-gi)

Optional for RL:
- stable-baselines3
- gymnasium
- tensorboard
