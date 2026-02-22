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
   - A* solvability check on `reset()`: retries goal+obstacle layouts until A* finds a valid path (up to 20 attempts)
   - `inflation_radius` parameter controls obstacle inflation for A* solvability checks (default 0.08m)

3. **Training Pipeline** (`src/train_rl.py`)
   - PPO training with BC warmstart from demonstrations
   - VecNormalize pre-warming from demo data (critical for BC→RL transfer)
   - Critic pretraining on Monte Carlo returns
   - Pipeline: validate → prewarm VecNormalize → BC warmstart → critic pretrain → PPO

4. **SAC/TQC + Chunk CVAE + Q-Chunking Pipeline** (`src/train_sac.py`)
   - TQC (sb3-contrib) with SAC fallback
   - **Chunk CVAE**: Actor predicts k-step action chunks; CVAE handles multimodal demos
   - **Q-chunking**: `ChunkedEnvWrapper` makes critic evaluate chunk-level Q-values
   - RLPD-style demo/online replay buffer sampling (configurable via `--demo-ratio`)
   - LayerNorm in critics (replaces VecNormalize)
   - UTD ratio configurable via `--utd-ratio` (default 20, recommended 5 for speed)
   - `--resume` to continue training from a checkpoint (step counter, weights preserved)
   - `--chunk-size` to control action chunk length (default 10)
   - **SafeTQC** (`--safe`): Constrained RL with dual critic + Lagrange multiplier
     - Separate cost critic estimates obstacle violation costs
     - Learned Lagrange multiplier auto-balances reward vs. safety
     - Auto-removes proximity penalty from reward (override with `--keep-proximity-reward`)
     - Requires demos recorded with cost data (`has_cost` metadata)

5. **Supporting Scripts**
   - `eval_policy.py` - Policy evaluation and metrics (auto-detects TQC/SAC/PPO and chunk size from model action space; wraps env with `ChunkedEnvWrapper` for chunked models)
   - `train_bc.py` - Behavioral cloning from demonstrations
   - `replay.py` - Demo playback and inspection

6. **Shared Modules**
   - `jetbot_config.py` - Single source of truth for robot physical constants (`WHEEL_RADIUS`, `WHEEL_BASE`, velocity limits, start pose, workspace bounds) and `quaternion_to_yaw()` utility
   - `demo_utils.py` - Shared demo data functions: `validate_demo_data()`, `load_demo_data()`, `load_demo_transitions()`, `extract_action_chunks()`, `make_chunk_transitions()`, `build_frame_stacks()`, and `VerboseEpisodeCallback`
   - `demo_io.py` - Unified demo I/O adapter: `open_demo()` dispatches `.npz`/`.hdf5`, `HDF5DemoWriter` for O(delta) incremental recording, `convert_npz_to_hdf5()` migration utility

### Key Classes

- **TUIRenderer**: Rich-based terminal UI for robot state display
- **SceneManager**: Manages goal markers and scene objects
- **DemoRecorder**: Records (obs, action, reward, done, cost) tuples; uses HDF5 incremental writes by default (O(delta) checkpoints), falls back to NPZ
- **DemoPlayer**: Loads and replays recorded demonstrations (supports both `.npz` and `.hdf5`)
- **open_demo()**: Unified reader dispatching on file extension — returns dict-like `NpzDemoData` or `Hdf5DemoData` (`src/demo_io.py`)
- **HDF5DemoWriter**: Incremental HDF5 writer with resizable chunked datasets, O(delta) `append_steps()`/`flush()` (`src/demo_io.py`)
- **ActionMapper**: Maps keyboard keys to velocity commands
- **ObservationBuilder**: Builds observation vectors from robot state
- **RewardComputer**: Computes navigation rewards; `compute_cost()` static method for SafeTQC cost signal; `safe_mode` suppresses proximity penalty
- **CameraStreamer**: GStreamer H264 RTP UDP camera streaming (`src/camera_streamer.py`)
- **LidarSensor**: Analytical 2D raycasting for obstacle detection (no physics dependency)
- **OccupancyGrid**: 2D boolean grid from obstacle geometry, inflated by robot radius for C-space planning
- **AutoPilot**: A*-based expert controller with privileged scene access for collision-free demo collection
- **FrameStackWrapper**: Gymnasium wrapper stacking last n_frames observations into a single flattened vector; oldest first for natural GRU input order (`src/jetbot_rl_env.py`)
- **ChunkedEnvWrapper**: Gymnasium wrapper converting single-step env to k-step chunked env for Q-chunking (`src/jetbot_rl_env.py`)
- **ChunkCVAEFeatureExtractor**: SB3 feature extractor splitting obs into state MLP (10→32D) + LiDAR MLP (24→64D) + z-pad (8D) = 104D
- **TemporalCVAEFeatureExtractor**: GRU-based SB3 feature extractor for frame-stacked obs; per-frame state/lidar MLPs → GRU → hidden state + z-pad (`src/train_sac.py`)
- **pretrain_chunk_cvae()**: CVAE pretraining — encoder maps (obs, action_chunk) → z, decoder (= actor's latent_pi + mu) maps (obs_features || z) → action_chunk; encoder discarded after pretraining
- **SafeTQC**: TQC subclass with dual cost critic + Lagrange multiplier for constrained RL (`src/train_sac.py`)
- **CostReplayBuffer**: Parallel ring buffer storing per-transition costs alongside the main replay buffer (`src/train_sac.py`)
- **MeanCostCritic**: Twin-Q mean-value cost critic with independent feature extractor (`src/train_sac.py`)
- **SafeTrainingCallback**: SB3 callback tracking per-transition and per-episode costs, logging to TensorBoard (`src/train_sac.py`)

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

# SAC/TQC + Chunk CVAE + Q-Chunking (recommended)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000

# Custom chunk size and UTD ratio
./run.sh train_sac.py --demos demos/recording.npz --headless --chunk-size 5 --utd-ratio 5

# SAC/TQC with custom demo ratio (75% demo, 25% online)
./run.sh train_sac.py --demos demos/recording.npz --headless --demo-ratio 0.75

# Custom CVAE hyperparameters
./run.sh train_sac.py --demos demos/recording.npz --headless \
  --cvae-epochs 200 --cvae-beta 0.05 --cvae-z-dim 16

# SAC/TQC with GRU recurrent policy via frame stacking
./run.sh train_sac.py --demos demos/recording.npz --headless --n-frames 4 --gru-hidden 128

# Resume SAC/TQC training from checkpoint
./run.sh train_sac.py --demos demos/recording.npz --headless --resume models/checkpoints/tqc_jetbot_50000_steps.zip --timesteps 500000

# SafeTQC: constrained RL with cost critic + Lagrange multiplier
./run.sh train_sac.py --demos demos/recording.npz --headless --safe

# SafeTQC with custom cost limit and cost type
./run.sh train_sac.py --demos demos/recording.npz --headless --safe --cost-limit 10.0 --cost-type both

# SafeTQC keeping proximity penalty in reward (default: auto-removed)
./run.sh train_sac.py --demos demos/recording.npz --headless --safe --keep-proximity-reward

# Evaluation (auto-detects TQC/SAC/PPO and chunk size)
./run.sh eval_policy.py models/tqc_jetbot.zip --episodes 100
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Evaluation with cost tracking
./run.sh eval_policy.py models/tqc_jetbot.zip --episodes 100 --safe --cost-type proximity

# Evaluation with explicit chunk size or inflation radius
./run.sh eval_policy.py models/tqc_jetbot.zip --chunk-size 5 --inflation-radius 0.08

# Tests
./run_tests.sh

# Convert existing NPZ demos to HDF5
python src/demo_io.py demos/recording.npz demos/recording.hdf5
```

### Demo File Format

New recordings default to **HDF5** (`.hdf5`) for O(delta) incremental checkpoint saves. All loading functions (`open_demo()`, `validate_demo_data()`, `load_demo_transitions()`, etc.) accept both `.npz` and `.hdf5` transparently. Old `.npz` files continue to work without conversion.

## File Structure

```
isaac-sim-jetbot-keyboard/
├── src/
│   ├── jetbot_keyboard_control.py    # Main teleoperation app
│   ├── jetbot_config.py              # Shared robot constants & quaternion_to_yaw()
│   ├── demo_utils.py                 # Shared demo loading/validation, chunk extraction & VerboseEpisodeCallback
│   ├── demo_io.py                    # Unified demo I/O: open_demo(), HDF5DemoWriter, convert_npz_to_hdf5()
│   ├── camera_streamer.py            # Camera streaming module
│   ├── jetbot_rl_env.py              # Gymnasium RL environment + ChunkedEnvWrapper
│   ├── train_rl.py                   # PPO training script
│   ├── train_sac.py                  # SAC/TQC + Chunk CVAE + Q-Chunking training script
│   ├── eval_policy.py                # Policy evaluation (auto-detects TQC/SAC/PPO + chunk size)
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

Obstacles use `VisualCylinder` primitives instead of `Fixed*`. This removes them from PhysX broadphase collision. The analytical LiDAR (`LidarSensor`) only reads `obstacle_metadata` (2D position + radius), not physics colliders, so Visual obstacles work identically. This provides a small speedup by reducing PhysX overhead.

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

## Chunk CVAE + Q-Chunking

The SAC/TQC pipeline uses action chunking to reduce compounding errors from single-step BC. The actor predicts k-step action chunks, a CVAE handles multimodal demonstrations, and a `ChunkedEnvWrapper` enables chunk-level Q-values (Q-chunking).

### Architecture
```
ChunkedEnvWrapper: action_space (2,) → (k*2,), executes k sub-steps per wrapper step
  R_chunk = Σ γ^i r_i, effective gamma = γ^k

Actor:  obs(34D) → ChunkCVAEFeatureExtractor → (obs_features || z=0) → latent_pi → mu → tanh → (k*2)
Critic: Q(obs_features || z=0, action_chunk) → scalar

ChunkCVAEFeatureExtractor:
  34D obs → split → [state 0:10]  → symlog → MLP(10→64→32)  →  32D ┐
                     [lidar 10:34] → symlog → MLP(24→128→64)  →  64D ├→ concat → 96D + z_pad(8D) = 104D
                                                                      └→ z_pad = zeros(z_dim)

TemporalCVAEFeatureExtractor (when --n-frames > 1):
  Wrapping order: ChunkedEnvWrapper( FrameStackWrapper( JetbotNavigationEnv ) )
  Input: (batch, n_frames * 34) flattened
  Reshape → (batch, n_frames, 34)
  Per-frame:  state_mlp(obs[:10]) → 32D  }
              lidar_mlp(obs[10:34]) → 64D } → 96D per frame
  GRU: (batch, n_frames, 96) → last hidden → (batch, gru_hidden_dim)
  Z-pad: concat(gru_output, zeros(z_dim)) → (gru_hidden_dim + z_dim)D

CVAE pretraining (replaces BC warmstart):
  Encoder (train-only): (obs_features, action_chunk) → z
  Decoder (= actor's latent_pi + mu): (obs_features, z) → action_chunk
  Loss: L1 reconstruction + β·KL
  After pretraining: encoder discarded, z fixed to 0
```

### Pipeline Order
1. Create env with `ChunkedEnvWrapper(env, chunk_size=k, gamma=γ)`
2. Build chunk-level demo transitions via `make_chunk_transitions()` → replay buffer
3. Create model with `ChunkCVAEFeatureExtractor`, gamma=γ^k, target_entropy=-2.0
4. Inject LayerNorm into critics (`inject_layernorm_into_critics()`)
5. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks via CVAE
6. Copy pretrained feature extractor weights → critic/critic_target
7. Train with SAC/TQC (no auxiliary callback needed — CVAE encoder is discarded)

### Key Functions & Classes (`src/train_sac.py`)
- **symlog()**: DreamerV3 symmetric log compression
- **ChunkCVAEFeatureExtractor.get_class()**: Returns SB3 BaseFeaturesExtractor with split state/lidar MLPs + z-pad
- **TemporalCVAEFeatureExtractor.get_class()**: Returns GRU-based SB3 feature extractor for frame-stacked observations
- **pretrain_chunk_cvae()**: CVAE pretraining on demo action chunks; trains feature extractor + actor layers

### Key Functions (`src/demo_utils.py`)
- **extract_action_chunks()**: Sliding-window action chunks within episode boundaries
- **make_chunk_transitions()**: Chunk-level (obs, action, R_chunk, next_obs, done) transitions
- **build_frame_stacks()**: Converts step-level obs (N, 34) → (N, n_frames * 34), respecting episode boundaries

### Key Classes (`src/jetbot_rl_env.py`)
- **FrameStackWrapper**: Gymnasium wrapper stacking last n_frames observations into (n_frames * obs_dim,), oldest first
- **ChunkedEnvWrapper**: Gymnasium wrapper expanding action space to (k*2,), executing k sub-steps

### CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | 10 | Action chunk size (k) |
| `--n-frames` | 1 | Number of observations to stack (1 = no stacking) |
| `--gru-hidden` | 128 | GRU hidden dimension (only used when n-frames > 1) |
| `--gru-lr` | 1e-5 | GRU learning rate (only used when n-frames > 1) |
| `--cvae-z-dim` | 8 | CVAE latent dimension |
| `--cvae-epochs` | 100 | CVAE pretraining epochs |
| `--cvae-beta` | 0.1 | CVAE KL weight |
| `--cvae-lr` | 1e-3 | CVAE pretraining learning rate |
| `--safe` | off | Enable SafeTQC (cost critic + Lagrange) |
| `--cost-limit` | 25.0 | Per-episode cost budget |
| `--lagrange-lr` | 3e-4 | Lagrange multiplier learning rate |
| `--lagrange-init` | 0.0 | Initial log-lambda |
| `--cost-n-critics` | 2 | Number of cost critic networks |
| `--cost-critic-type` | mean | `mean` (MSE) or `quantile` |
| `--cost-type` | proximity | `proximity`, `collision`, or `both` |
| `--keep-proximity-reward` | off | Keep proximity penalty with `--safe` |

### SafeTQC Pipeline Order
1. Create env with `ChunkedEnvWrapper` + `safe_mode=True` (disables proximity penalty in reward)
2. Load demo transitions with costs (`load_costs=True`)
3. Build chunk-level transitions with costs (`demo_costs=...`)
4. Create `CostReplayBuffer` with demo chunk costs
5. Create `SafeTQC` model (cost critic + Lagrange multiplier)
6. Inject LayerNorm into reward critics and cost critics
7. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks
8. Copy pretrained feature extractor weights → reward critic + cost critic
9. Train with `SafeTrainingCallback` tracking per-step costs

### Compatibility
- `inject_layernorm_into_critics`: Unaffected — operates on critic nets after feature extraction
- Replay buffer: Uses chunk-level transitions (obs, k*2 action, R_chunk, next_obs, done)
- `--resume`: Works — SB3 pickles `features_extractor_class`; chunk_size auto-detected from action_dim
- `eval_policy.py`: Auto-detects chunk_size from model action space, wraps eval env with ChunkedEnvWrapper

## Testing

Tests use pytest with mocked Isaac Sim imports:
```bash
./run_tests.bat          # All tests (Windows)
./run_tests.sh           # All tests (Linux)
./run_tests.sh -v        # Verbose
./run_tests.sh -k name   # Specific test
```

### Mocking gymnasium.Env Subclasses

`gymnasium.Wrapper.__init__` performs a hard `isinstance(env, Env)` assertion. Plain `Mock()` objects fail this check. Any test helper that creates a fake env for use with `ChunkedEnvWrapper` (or any other `gymnasium.Wrapper` subclass) **must** return a real `gymnasium.Env` subclass, not a `Mock`.

**Wrong — fails at wrapper construction:**
```python
env = Mock()
env.observation_space = gym.spaces.Box(...)
env.action_space = gym.spaces.Box(...)
```

**Correct — define a minimal concrete subclass:**
```python
class _MinimalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(...)
        self.action_space = gym.spaces.Box(...)
    def reset(self, **kwargs): return np.zeros(...), {}
    def step(self, action): return np.zeros(...), 0.0, False, False, {}
```

Per-test behaviour can still be customised by overriding the instance's `step` attribute after construction (e.g. `inner.step = mock_step`).

## Reward Function

### Dense Mode (default)
- **Goal reached**: +10.0 (terminal)
- **Collision**: -10.0 (terminal, LiDAR distance < 0.08m)
- **Distance shaping**: `(prev_dist - curr_dist) * 1.0`
- **Heading bonus**: `((pi - |angle_to_goal|) / pi) * 0.1`
- **Proximity penalty**: `0.1 * (1.0 - min_lidar / 0.3)` when min_lidar < 0.3m, gated by goal distance (linearly reduced within 0.5m of goal, zero at goal). Auto-removed when `--safe` is active (handled by cost critic); `--keep-proximity-reward` to override
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

Optional for HDF5 demo recording (recommended):
- h5py (incremental O(delta) demo checkpoints; falls back to NPZ without it)

Optional for RL:
- torch (PyTorch — bundled with Isaac Sim)
- stable-baselines3
- sb3-contrib (for TQC; falls back to SAC without it)
- gymnasium
- tensorboard
