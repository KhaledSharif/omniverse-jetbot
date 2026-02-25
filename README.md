# Isaac Sim Jetbot Keyboard Control

Keyboard-controlled Jetbot mobile robot teleoperation with demonstration recording and reinforcement learning training pipeline for NVIDIA Isaac Sim 5.1.0.

## Features

- **Keyboard Teleoperation**: Control the Jetbot using WASD keys
- **Rich Terminal UI**: Real-time robot state display with visual feedback
- **Camera Streaming**: GStreamer H264 RTP UDP streaming from Jetbot camera
- **Random Obstacles**: Configurable static obstacles for navigation challenge
- **Demonstration Recording**: Record navigation trajectories to HDF5 (incremental O(delta) checkpoints) or NPZ
- **Automatic Demo Collection**: Autonomous A*-based data collection with collision-free expert controller
- **RL Training Pipeline**: PPO with BC warmstart; CrossQ/TQC/SAC with Chunk CVAE + Q-Chunking
- **CrossQ (Default)**: BatchRenorm critics, no target networks, UTD=1 — ~13x faster than TQC@UTD=20
- **SafeTQC**: Constrained RL with dual cost critic + learned Lagrange multiplier for obstacle avoidance
- **Ego-Centric Observations**: Normalized workspace coords, sin/cos heading, body-frame goal — improves generalization
- **Action Chunking**: k-step action chunks reduce compounding errors from single-step BC
- **Chunk CVAE**: Conditional VAE pretraining handles multimodal demonstrations
- **Q-Chunking**: Critic evaluates chunk-level Q-values via `ChunkedEnvWrapper`
- **Gymnasium Integration**: Standard RL environment compatible with Stable-Baselines3
- **LiDAR Sensing**: 24-ray analytical raycasting (180 FOV) for obstacle detection
- **Solvability Checks**: A* path verification on reset ensures navigable goal placements

## Requirements

- NVIDIA Isaac Sim 5.1.0 standalone
- NVIDIA RTX GPU
- Python 3.11 (bundled with Isaac Sim)

### Python Dependencies

```bash
~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh -m pip install numpy pynput rich stable-baselines3 sb3-contrib tensorboard gymnasium tqdm h5py lark
```

HDF5 enables incremental O(delta) checkpoint saves during recording — checkpoint cost stays ~2ms regardless of dataset size, compared to 200-500ms+ for full NPZ rewrites. Falls back to NPZ automatically if `h5py` is not installed.

For camera streaming functionality (optional, Linux only), install GStreamer and PyGObject:

```bash
# System packages (Ubuntu/Debian)
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
  libcairo2-dev libgirepository-2.0-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0

# Python package (in Isaac Sim environment)
~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh -m pip install PyGObject
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

### Train RL Agent (CrossQ + Chunk CVAE)

```bash
# Both .npz and .hdf5 demo files are accepted
./run.sh train_sac.py --demos demos/recording.hdf5 --headless --timesteps 500000

# Legacy TQC mode (slower but proven)
./run.sh train_sac.py --demos demos/recording.hdf5 --headless --legacy-tqc --utd-ratio 20

# With SafeTQC (constrained RL)
./run.sh train_sac.py --demos demos/recording.hdf5 --headless --safe
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
│   ├── demo_io.py                    # Unified demo I/O: open_demo(), HDF5DemoWriter, NPZ↔HDF5 converter
│   ├── camera_streamer.py            # Camera streaming module
│   ├── jetbot_rl_env.py              # Gymnasium RL environment
│   ├── train_rl.py                   # PPO training script
│   ├── train_sac.py                  # CrossQ/TQC/SAC + RLPD training script
│   ├── eval_policy.py                # Policy evaluation (auto-detects CrossQ/TQC/SAC/PPO)
│   ├── train_bc.py                   # Behavioral cloning
│   ├── replay.py                     # Demo playback
│   ├── test_jetbot_keyboard_control.py
│   ├── test_jetbot_rl_env.py
│   ├── test_train_rl.py
│   ├── test_train_sac.py
│   ├── test_train_bc.py
│   ├── test_eval_policy.py
│   ├── test_replay.py
│   └── test_demo_io.py
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
# CrossQ + Chunk CVAE + Q-Chunking (recommended, ~35-39 steps/s)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 1000000

# Legacy TQC mode (slower but proven, ~2.9 steps/s at UTD=20)
./run.sh train_sac.py --demos demos/recording.npz --headless --legacy-tqc --utd-ratio 20

# Custom chunk size and UTD ratio
./run.sh train_sac.py --demos demos/recording.npz --headless --chunk-size 5 --utd-ratio 5

# With obstacles and arena config
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 1000000 \
  --num-obstacles 50 --arena-size 8 --max-steps 1000

# Custom CVAE hyperparameters
./run.sh train_sac.py --demos demos/recording.npz --headless \
  --cvae-epochs 200 --cvae-beta 0.05 --cvae-z-dim 16

# Resume training from checkpoint (--ent-coef-init resets collapsed ent_coef from checkpoint)
./run.sh train_sac.py --demos demos/recording.npz --headless \
  --resume models/checkpoints/crossq_jetbot_50000_steps.zip --timesteps 500000 --ent-coef-init 0.1

# SafeTQC: constrained RL with cost critic + Lagrange multiplier
./run.sh train_sac.py --demos demos/recording.npz --headless --safe

# SafeTQC with custom cost limit and cost type
./run.sh train_sac.py --demos demos/recording.npz --headless --safe \
  --cost-limit 10.0 --cost-type both

# SafeTQC keeping proximity penalty in reward (default: auto-removed)
./run.sh train_sac.py --demos demos/recording.npz --headless --safe --keep-proximity-reward

# PPO with BC warmstart
./run.sh train_rl.py --headless --bc-warmstart demos/recording.npz --timesteps 1000000

# Train BC model only
./run.sh train_bc.py demos/recording.npz --epochs 100
```

#### CrossQ + Chunk CVAE + Q-Chunking Pipeline (Recommended)

The `train_sac.py` script uses **CrossQ** (BatchRenorm critics, no target networks) by default with action chunking and RLPD-style 50/50 demo/online replay buffer sampling. CrossQ achieves equal sample efficiency to TQC@UTD=20 at UTD=1, providing a ~13x speedup. The actor predicts k-step action chunks, a CVAE handles multimodal demonstrations, and a `ChunkedEnvWrapper` enables chunk-level Q-values (Q-chunking).

Use `--legacy-tqc` to switch to TQC (Truncated Quantile Critics) with UTD=20 for backward compatibility.

**Architecture:**
```
ChunkedEnvWrapper: action_space (2,) → (k*2,), executes k sub-steps per wrapper step
  R_chunk = Σ γ^i r_i, effective gamma = γ^k

Actor:  obs(34D) → ChunkCVAEFeatureExtractor → (obs_features || z=0) → latent_pi → mu → tanh → (k*2)
Critic: Q(obs_features || z=0, action_chunk) → scalar

ChunkCVAEFeatureExtractor (dynamic split: state_dim = obs_dim - 24):
  obs → split → [state 0:state_dim]       → symlog → MLP(state_dim→64→32) → 32D ┐
                 [lidar state_dim:obs_dim] → symlog → MLP(24→128→64)       → 64D ├→ concat → 96D + z_pad(8D) = 104D
```

**Pipeline order:**
1. Create env with `ChunkedEnvWrapper(env, chunk_size=k, gamma=γ)`
2. Build chunk-level demo transitions via `make_chunk_transitions()` → replay buffer
3. Create model with `ChunkCVAEFeatureExtractor`, gamma=γ^k, target_entropy=-2.0
4. Inject LayerNorm into critics (skipped for CrossQ — BatchRenorm built-in)
5. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks via CVAE
6. Copy pretrained feature extractor weights → critic (and critic_target if present; CrossQ has none)
7. Train with CrossQ/TQC/SAC (CVAE encoder discarded, z-slot zeroed during RL)

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | 10 | Action chunk size (k) |
| `--legacy-tqc` | off | Use TQC instead of CrossQ (legacy mode) |
| `--utd-ratio` | 1 | Gradient steps per env step (CrossQ=1, TQC=20) |
| `--policy-delay` | 1 | Actor update delay (for CrossQ with high UTD) |
| `--add-prev-action` | off | Include previous action in observations (36D) |
| `--cvae-z-dim` | 8 | CVAE latent dimension |
| `--cvae-epochs` | 100 | CVAE pretraining epochs |
| `--cvae-beta` | 0.1 | CVAE KL weight |
| `--cvae-lr` | 1e-3 | CVAE pretraining learning rate |
| `--demo-ratio` | 0.5 | Fraction of batch from demos |
| `--log-std-init` | -0.5 | Actor log_std after CVAE. CVAE sets -2.0 (std=0.135) collapsing entropy before SAC starts; -0.5 (std=0.61) gives SAC room to explore. Use -2.0 to keep CVAE value. |
| `--ent-coef-init` | 0.1 | ent_coef set after CVAE pretraining and on `--resume`. CVAE/checkpoint may leave ent_coef ≈0.006 (entropy bonus negligible vs Q-values). Use 0 to disable. |

#### SafeTQC: Constrained RL with Lagrange Multiplier

When `--safe` is enabled, `train_sac.py` adds a **cost critic** and **learned Lagrange multiplier** on top of the standard CrossQ/TQC pipeline. This replaces the hand-tuned proximity penalty in the reward function with a principled constrained optimization approach.

**How it works:**
- A separate cost critic estimates cumulative obstacle violation costs (binary: 1.0 if min LiDAR < 0.3m, else 0.0)
- The actor loss becomes: `ent_coef * log_prob - Q_reward + λ * Q_cost`
- The Lagrange multiplier λ is learned to satisfy a per-episode cost budget
- The proximity penalty is automatically removed from the reward (override with `--keep-proximity-reward`)

**Requirements:** Demos must be recorded with cost data (`has_cost` metadata). Any demos recorded after this feature was added will include costs automatically.

| Flag | Default | Description |
|------|---------|-------------|
| `--safe` | off | Enable SafeTQC |
| `--cost-limit` | 25.0 | Per-episode cost budget |
| `--lagrange-lr` | 3e-4 | Lagrange multiplier learning rate |
| `--lagrange-init` | 0.0 | Initial log-lambda |
| `--cost-n-critics` | 2 | Number of cost critic networks |
| `--cost-critic-type` | mean | `mean` (MSE) or `quantile` |
| `--cost-type` | proximity | `proximity`, `collision`, or `both` |
| `--keep-proximity-reward` | off | Keep proximity penalty with `--safe` |

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
# Evaluate trained policy (auto-detects CrossQ/TQC/SAC/PPO and chunk size)
./run.sh eval_policy.py models/crossq_jetbot.zip --episodes 100

# Headless evaluation
./run.sh eval_policy.py models/crossq_jetbot.zip --headless --episodes 100

# Evaluation with cost tracking (for SafeTQC models)
./run.sh eval_policy.py models/crossq_jetbot.zip --episodes 100 --safe --cost-type proximity

# Override chunk size or inflation radius
./run.sh eval_policy.py models/crossq_jetbot.zip --chunk-size 5 --inflation-radius 0.08
```

### Demo File Format

New recordings default to **HDF5** (`.hdf5`). All tools accept both `.npz` and `.hdf5` transparently via `open_demo()`.

```bash
# Convert existing NPZ demos to HDF5
python src/demo_io.py demos/recording.npz demos/recording.hdf5

# Or let the output path default to same name with .hdf5 extension
python src/demo_io.py demos/recording.npz
```

### Demo Inspection

```bash
# Show demo statistics (no simulation needed) — works with .npz or .hdf5
./run.sh replay.py demos/recording.hdf5 --info

# Visual playback
./run.sh replay.py demos/recording.hdf5

# Replay specific episode
./run.sh replay.py demos/recording.hdf5 --episode 0

# Replay successful episodes only
./run.sh replay.py demos/recording.hdf5 --successful
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

## Observation Space (34D ego-centric for RL, 10D base for teleoperation)

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Normalized workspace X | [0, 1] |
| 1 | Normalized workspace Y | [0, 1] |
| 2 | sin(heading) | [-1, 1] |
| 3 | cos(heading) | [-1, 1] |
| 4 | Linear velocity | m/s |
| 5 | Angular velocity | rad/s |
| 6 | Goal body-frame X (dist * cos(angle)) | meters |
| 7 | Goal body-frame Y (dist * sin(angle)) | meters |
| 8 | Distance to goal | meters |
| 9 | Goal reached flag | 0 or 1 |
| 10-33 | LiDAR distances (24 rays, 180 FOV) | 0.0 (touching) to 1.0 (max range) |

With `--add-prev-action` (36D): inserts `[prev_linear_vel, prev_angular_vel]` at indices [10:12], pushing LiDAR to [12:36].

The RL environment (`JetbotNavigationEnv`) always uses 34D ego-centric observations with LiDAR.
The keyboard controller uses 10D by default; pass `--use-lidar` for 34D.

Old demos (pre-OBS_VERSION=2) are auto-converted to the new ego-centric format on load.

## Action Space (2D)

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Linear velocity | [-1, 1] -> [-0.3, 0.3] m/s |
| 1 | Angular velocity | [-1, 1] -> [-1.0, 1.0] rad/s |

## Reward Function

### Dense Mode (default)
- **Goal reached**: +50.0 (terminal)
- **Collision**: -25.0 (terminal, LiDAR distance < 0.08m)
- **Distance shaping**: `(prev_dist - curr_dist) * 1.0` — reward for getting closer
- **Heading bonus**: `((pi - |angle_to_goal|) / pi) * progress * 0.5` — only when making forward progress, prevents exploitation from circling near the goal
- **Approach bonus**: Potential-based shaping within 1m of goal — `10.0 * (max(0, 1 - curr_dist/1.0) - max(0, 1 - prev_dist/1.0))`. Amplifies distance shaping 10x near the goal. Provably unexploitable (Ng et al., 1999): orbiting gives 0, oscillating cancels out
- **Proximity penalty**: `0.05 * (1.0 - min_lidar / 0.3)` when near obstacles, gated by goal distance (auto-removed with `--safe`, handled by cost critic instead)
- **Time penalty**: -0.005 per step

**Reward hierarchy:** collision (-25) >> max proximity accumulation (~-10) >> time penalty (-10 over 400 steps). This ensures collisions are always worse than proximity accumulation or bad truncations.

### Sparse Mode
- **Goal reached**: +50.0
- **Collision**: -25.0
- **Otherwise**: 0.0

## TensorBoard

Monitor training progress:

```bash
tensorboard --logdir runs/
```

## Training Performance

CrossQ (default) achieves equal sample efficiency to TQC@UTD=20 at UTD=1 via BatchRenorm in critics and no target networks, providing a ~13x speedup:

| Algorithm | UTD | n-frames | Steps/s | 1M Steps Wall Time | Speedup |
|-----------|-----|----------|---------|---------------------|---------|
| TQC       | 20  | 1        | ~3      | ~4 days             | 1x      |
| TQC       | 5   | 1        | ~10-12  | ~24 hours           | ~4x     |
| **CrossQ**| **1** | **1** | **~35-39** | **~7 hours**  | **~13x** |
| CrossQ    | 1   | 4 (GRU)  | ~17     | ~16 hours           | ~6x     |

*Measured on RTX 3090 Ti, chunk_size=5, headless, batch size 256.*

**Measured step-time breakdown (chunk_size=5):** `world.step()` = 1.3ms · full `env.step()` = 3.0ms · 5-step chunk = 15ms · CrossQ+GRU gradient = ~43ms · total fps≈17 with n-frames=4.

**Recommendation:** Use CrossQ (default) for fast training. Use `--legacy-tqc --utd-ratio 20` only if you need exact TQC reproduction.

Additional optimizations applied in headless mode:
- Obstacles use `Visual*` primitives (no PhysX collision — LiDAR is analytical)
- Viewport updates disabled, rendering decoupled from physics
- Reduced PhysX solver iterations for single-robot scene

## Architecture

### Main Components

- **JetbotKeyboardController**: Main application with keyboard input and Rich TUI
- **JetbotNavigationEnv**: Gymnasium-compatible RL environment with A* solvability checks on reset
- **ChunkedEnvWrapper**: Gymnasium wrapper converting single-step env to k-step chunked env for Q-chunking
- **ChunkCVAEFeatureExtractor**: Dynamic split state/lidar MLPs + z-pad slot for CVAE latent variable (supports 34D and 36D obs)
- **DifferentialController**: Converts velocity commands to wheel speeds
- **SceneManager**: Manages goal markers, obstacles, and scene objects
- **DemoRecorder/DemoPlayer**: Recording and playback of demonstrations; HDF5 incremental writes by default (with cost data for SafeTQC)
- **demo_io**: Unified demo I/O — `open_demo()` dispatches `.npz`/`.hdf5`, `HDF5DemoWriter` for O(delta) appends
- **SafeTQC**: CrossQ/TQC subclass with cost critic + Lagrange multiplier for constrained RL
- **CostReplayBuffer**: Parallel cost buffer mirroring the main replay buffer
- **MeanCostCritic**: Twin-Q cost critic with independent feature extractor
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
