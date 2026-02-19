# Isaac Sim Jetbot Keyboard Control

Keyboard-controlled Jetbot mobile robot teleoperation with demonstration recording and reinforcement learning training pipeline for NVIDIA Isaac Sim 5.0.0.

## Features

- **Keyboard Teleoperation**: Control the Jetbot using WASD keys
- **Rich Terminal UI**: Real-time robot state display with visual feedback
- **Camera Streaming**: GStreamer H264 RTP UDP streaming from Jetbot camera
- **Random Obstacles**: Configurable static obstacles for navigation challenge
- **Demonstration Recording**: Record navigation trajectories for imitation learning
- **Automatic Demo Collection**: Autonomous A*-based data collection with collision-free expert controller
- **RL Training Pipeline**: PPO with BC warmstart; SAC/TQC with RLPD-style demo replay
- **LiDAR MLP-VAE**: Optional DreamerV3-style VAE for compressed LiDAR latent representations
- **Gymnasium Integration**: Standard RL environment compatible with Stable-Baselines3
- **LiDAR Sensing**: 24-ray analytical raycasting (180 FOV) for obstacle detection

## Requirements

- NVIDIA Isaac Sim 5.0.0 standalone
- NVIDIA RTX GPU
- Python 3.11 (bundled with Isaac Sim)

### Python Dependencies

```bash
~/Downloads/isaac-sim-standalone-5.0.0-linux-x86_64/python.sh -m pip install numpy pynput rich stable-baselines3 tensorboard gymnasium
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

### With Recording Enabled

```bash
./run.sh --enable-recording
```

### With Custom Obstacle Count

```bash
./run.sh --num-obstacles 10
```

### Train RL Agent

```bash
./run.sh train_rl.py --headless --timesteps 500000
```

## Controls

| Key | Action |
|-----|--------|
| W | Move forward |
| S | Move backward |
| A | Turn left |
| D | Turn right |
| Space | Stop (emergency brake) |
| R | Reset robot position |
| G | Spawn new goal |
| C | Toggle camera viewer |
| Esc | Exit |

### Recording Controls

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
│   ├── jetbot_config.py              # Shared robot constants & quaternion_to_yaw()
│   ├── demo_utils.py                 # Shared demo loading/validation & VerboseEpisodeCallback
│   ├── camera_streamer.py            # Camera streaming module
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── train_sac.py                  # SAC/TQC + RLPD training script
│   ├── eval_policy.py                # Policy evaluation (auto-detects TQC/SAC/PPO)
│   ├── train_bc.py                   # Behavioral cloning
│   ├── replay.py                     # Demo playback
│   ├── test_jetbot_keyboard_control.py
│   ├── test_jetbot_rl_env.py
│   ├── test_train_rl.py
│   ├── test_train_sac.py
│   ├── test_train_bc.py
│   ├── test_eval_policy.py
│   └── test_replay.py
├── demos/                            # Recorded demonstrations
├── models/                           # Trained models
├── runs/                             # TensorBoard logs
├── run.sh                            # Isaac Sim Python launcher
├── run_tests.sh                      # Test runner
├── CLAUDE.md                         # AI assistant guidance
└── README.md
```

## Usage Examples

### Teleoperation

```bash
# Basic teleoperation (5 obstacles by default)
./run.sh

# With custom obstacle count
./run.sh --num-obstacles 10

# With recording enabled
./run.sh --enable-recording

# Disable camera streaming
./run.sh --no-camera

# Custom camera port
./run.sh --camera-port 5601

# Combine options
./run.sh --enable-recording --num-obstacles 8 --demo-path demos/my_demo.npz
```

### Automatic Demo Collection

```bash
# Collect 100 episodes automatically (default)
./run.sh --enable-recording --automatic

# Custom episode count
./run.sh --enable-recording --automatic --num-episodes 200

# Continuous mode (run until Esc)
./run.sh --enable-recording --automatic --continuous

# Headless TUI (console progress prints only)
./run.sh --enable-recording --automatic --num-episodes 200 --headless-tui
```

### Training

```bash
# SAC/TQC + RLPD (recommended — demos in replay buffer, no pretraining needed)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 1000000

# SAC/TQC with UTD=5 for faster training (~10 steps/s vs ~3 at UTD=20)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 1000000 --utd-ratio 5

# SAC/TQC with obstacles and arena config
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 1000000 \
  --num-obstacles 50 --arena-size 8 --max-steps 1000 --utd-ratio 5

# SAC/TQC with LiDAR MLP-VAE (DreamerV3-style latent compression)
./run.sh train_sac.py --demos demos/recording.npz --headless --lidar-mlp-vae

# LiDAR MLP-VAE with custom VAE hyperparameters
./run.sh train_sac.py --demos demos/recording.npz --headless --lidar-mlp-vae \
  --vae-epochs 200 --vae-beta 0.05 --vae-aux-freq 5 --vae-aux-lr 1e-4

# PPO with BC warmstart
./run.sh train_rl.py --headless --bc-warmstart demos/recording.npz --timesteps 1000000

# Train BC model only
./run.sh train_bc.py demos/recording.npz --epochs 100
```

#### SAC/TQC + RLPD Pipeline (Recommended)

The `train_sac.py` script uses TQC (Truncated Quantile Critics) with RLPD-style 50/50 demo/online replay buffer sampling. Unlike the PPO pipeline, no VecNormalize pre-warming or critic pretraining is needed — demos are sampled continuously during training. A BC warmstart on the actor is still performed. LayerNorm in critics replaces VecNormalize.

Key parameter: `--utd-ratio` controls gradient steps per env step (default 20). Higher UTD = better sample efficiency but slower wall-clock time. See [Training Performance](#training-performance) below.

#### LiDAR MLP-VAE (Optional)

When `--lidar-mlp-vae` is enabled, a DreamerV3-style MLP-VAE preprocesses LiDAR observations into a compressed 16D latent space before feeding into the actor/critic networks:

```
34D obs -> split -> [state 0:10]  -> symlog -> MLP(10->64->32)  -> concat -> 48D -> actor/critic
                    [lidar 10:34] -> symlog -> VAE enc(24->128->64->16D) /
```

The pipeline: pretrain VAE on demo LiDAR data, inject pretrained weights into the SB3 model, BC warmstart with feature extractor, then train with an auxiliary VAE reconstruction+KL loss. TensorBoard logs `vae/recon_loss`, `vae/kl_loss`, and `vae/total_loss`.

| Flag | Default | Description |
|------|---------|-------------|
| `--lidar-mlp-vae` | off | Enable VAE feature extractor |
| `--vae-epochs` | 100 | VAE pretraining epochs |
| `--vae-beta` | 0.1 | KL divergence weight |
| `--vae-aux-freq` | 10 | Auxiliary loss frequency (steps) |
| `--vae-aux-lr` | 1e-4 | Auxiliary loss learning rate |

#### PPO + BC Warmstart Pipeline

When `--bc-warmstart` is provided, the training script runs:

1. **Validate** demo data (minimum episodes, transitions, successes)
2. **Pre-warm VecNormalize** — seeds observation and reward normalization stats from demo data
3. **BC warmstart** — pretrains the PPO actor on normalized demo observations
4. **Critic pretraining** — pretrains the PPO value network on normalized Monte Carlo returns
5. **PPO training** — fine-tunes with on-policy RL, VecNormalize stats consistent with pretraining

The VecNormalize pre-warming step is critical: without it, the BC-learned policy sees differently-scaled observations during RL and loses its learned behavior.

### Evaluation

```bash
# Evaluate trained policy
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Headless evaluation
./run.sh eval_policy.py models/ppo_jetbot.zip --headless --episodes 100
```

### Demo Inspection

```bash
# Show demo statistics (no simulation needed)
./run.sh replay.py demos/recording.npz --info

# Visual playback
./run.sh replay.py demos/recording.npz

# Replay specific episode
./run.sh replay.py demos/recording.npz --episode 0

# Replay successful episodes only
./run.sh replay.py demos/recording.npz --successful
```

### Testing

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh -v

# Run specific test
./run_tests.sh -k test_action_mapper
```

## Observation Space (34D for RL, 10D base for teleoperation)

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
| 10-33 | LiDAR distances (24 rays, 180 FOV) | 0.0 (touching) to 1.0 (max range) |

The RL environment (`JetbotNavigationEnv`) always uses 34D observations with LiDAR.
The keyboard controller uses 10D by default; pass `--use-lidar` for 34D.

## Action Space (2D)

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Linear velocity | [-1, 1] -> [-0.3, 0.3] m/s |
| 1 | Angular velocity | [-1, 1] -> [-1.0, 1.0] rad/s |

## Reward Function

### Dense Mode (default)
- **Goal reached**: +10.0 (terminal)
- **Collision**: -10.0 (terminal, LiDAR distance < 0.08m)
- **Distance shaping**: `(prev_dist - curr_dist) * 1.0` — reward for getting closer
- **Heading bonus**: `((pi - |angle_to_goal|) / pi) * 0.1` — reward for facing goal
- **Proximity penalty**: `0.1 * (1.0 - min_lidar / 0.3)` when near obstacles
- **Time penalty**: -0.005 per step

### Sparse Mode
- **Goal reached**: +10.0
- **Collision**: -10.0
- **Otherwise**: 0.0

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/
```

## Training Performance

The dominant cost in SAC/TQC training is **gradient computation**, not physics simulation. Each env step takes ~25ms (`world.step()`), but TQC's 5 critic networks + actor require significant GPU time per gradient update.

| UTD Ratio | Steps/s | 1M Steps Wall Time | Sample Efficiency |
|-----------|---------|---------------------|-------------------|
| 20 (default) | ~3 | ~4 days | Best |
| 5 | ~10-12 | ~24 hours | Good |
| 1 | ~39 | ~7 hours | Lower |

*Measured on RTX 3090 Ti, 50 obstacles, headless, batch size 256.*

**Recommendation:** `--utd-ratio 5` gives a good balance of speed and sample efficiency. Use `--utd-ratio 20` only if you can afford multi-day runs.

Additional optimizations applied in headless mode:
- Obstacles use `Visual*` primitives (no PhysX collision — LiDAR is analytical)
- Viewport updates disabled, rendering decoupled from physics
- Reduced PhysX solver iterations for single-robot scene

## Architecture

### Main Components

- **JetbotKeyboardController**: Main application with keyboard input and Rich TUI
- **JetbotNavigationEnv**: Gymnasium-compatible RL environment
- **DifferentialController**: Converts velocity commands to wheel speeds
- **SceneManager**: Manages goal markers, obstacles, and scene objects
- **DemoRecorder/DemoPlayer**: Recording and playback of demonstrations
- **CameraStreamer**: GStreamer H264 RTP UDP camera streaming
- **AutoPilot**: A*-based expert controller with privileged scene access for collision-free demo collection

### Obstacle System

The SceneManager randomly spawns visual-only obstacles (VisualCylinder) with:
- **Configurable count**: Set via `--num-obstacles` parameter (default: 5)
- **Random sizes**: Varied radius and height for each cylinder
- **Safe placement**: Maintains minimum distance from goal (0.5m) and robot start (1.0m)
- **Automatic respawn**: Obstacles regenerate when goal is reset (pressing 'G' or new episode)

### Isaac Sim Integration

The project uses Isaac Sim's:
- `WheeledRobot` class for the Jetbot
- `DifferentialController` for wheel velocity control
- `World` for simulation management

## License

MIT License

## Acknowledgments

- NVIDIA Isaac Sim team
- Based on the [isaac-sim-franka-keyboard](https://github.com/example/isaac-sim-franka-keyboard) project structure
