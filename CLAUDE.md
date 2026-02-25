# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**isaac-sim-jetbot-keyboard** is a keyboard-controlled Jetbot mobile robot teleoperation system with demonstration recording and reinforcement learning training pipeline for Isaac Sim 5.1.0.

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

4. **CrossQ/TQC + Chunk CVAE + Q-Chunking Pipeline** (`src/train_sac.py`)
   - **CrossQ** (default, sb3-contrib ≥ 2.4.0): BatchRenorm critics, no target networks, UTD=1 (~35-39 steps/s)
   - TQC fallback via `--legacy-tqc` (5 quantile critics, UTD=20, ~2.9 steps/s)
   - SAC fallback if sb3-contrib unavailable
   - **Chunk CVAE**: Actor predicts k-step action chunks; CVAE handles multimodal demos
   - **Q-chunking**: `ChunkedEnvWrapper` makes critic evaluate chunk-level Q-values
   - RLPD-style demo/online replay buffer sampling (configurable via `--demo-ratio`)
   - LayerNorm in critics for TQC/SAC (skipped for CrossQ — BatchRenorm built-in)
   - UTD ratio configurable via `--utd-ratio` (default 1 for CrossQ, recommended 20 for legacy TQC)
   - `--resume` to continue training from a checkpoint (step counter, weights preserved)
   - `--chunk-size` to control action chunk length (default 10)
   - **SafeTQC** (`--safe`): Constrained RL with dual critic + Lagrange multiplier
     - Separate cost critic estimates obstacle violation costs (always has its own target network)
     - Learned Lagrange multiplier auto-balances reward vs. safety
     - Auto-removes proximity penalty from reward (override with `--keep-proximity-reward`)
     - Requires demos recorded with cost data (`has_cost` metadata)
     - Compatible with CrossQ base (no reward critic target, but cost critic target preserved)

5. **Supporting Scripts**
   - `eval_policy.py` - Policy evaluation and metrics (auto-detects CrossQ/TQC/SAC/PPO and chunk size from model action space; wraps env with `ChunkedEnvWrapper` for chunked models)
   - `train_bc.py` - Behavioral cloning from demonstrations
   - `replay.py` - Demo playback and inspection

6. **Shared Modules**
   - `jetbot_config.py` - Single source of truth for robot physical constants (`WHEEL_RADIUS`, `WHEEL_BASE`, velocity limits, start pose, workspace bounds), `quaternion_to_yaw()` utility, and `OBS_VERSION` constant
   - `demo_utils.py` - Shared demo data functions: `validate_demo_data()`, `load_demo_data()`, `load_demo_transitions()`, `extract_action_chunks()`, `make_chunk_transitions()`, `build_frame_stacks()`, `convert_obs_to_egocentric()`, and `VerboseEpisodeCallback` (enhanced training diagnostics)
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
- **ChunkedEnvWrapper**: Gymnasium wrapper converting single-step env to k-step chunked env for Q-chunking (`src/jetbot_rl_env.py`); info dict includes `goal_distance`, `min_lidar_distance`, `collision`, `is_success`, `cost`
- **ChunkCVAEFeatureExtractor**: SB3 feature extractor with dynamic split: state MLP (state_dim→32D) + LiDAR MLP (24→64D) + z-pad (8D) = 104D. Split is `obs[:obs_dim-24]` / `obs[obs_dim-24:]` to support 34D and 36D
- **TemporalCVAEFeatureExtractor**: GRU-based SB3 feature extractor for frame-stacked obs; per-frame state/lidar MLPs → GRU → hidden state + z-pad (`src/train_sac.py`)
- **pretrain_chunk_cvae()**: CVAE pretraining — encoder maps (obs, action_chunk) → z, decoder (= actor's latent_pi + mu) maps (obs_features || z) → action_chunk; encoder discarded after pretraining
- **SafeTQC**: CrossQ/TQC subclass with dual cost critic + Lagrange multiplier for constrained RL (`src/train_sac.py`)
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

### Observation Space (34D ego-centric, OBS_VERSION=2)
```
[0]     - Normalized workspace X: (x - x_min) / (x_max - x_min) ∈ [0,1]
[1]     - Normalized workspace Y: (y - y_min) / (y_max - y_min) ∈ [0,1]
[2]     - sin(heading) — no discontinuity at ±π
[3]     - cos(heading)
[4]     - Linear velocity
[5]     - Angular velocity
[6]     - Goal body-frame X: dist * cos(angle_to_goal)
[7]     - Goal body-frame Y: dist * sin(angle_to_goal)
[8]     - Distance to goal
[9]     - Goal reached flag
[10:34] - LiDAR: 24 normalized distances (0=touching, 1=max range), 180° FOV
```

With `--add-prev-action` (36D): inserts `[prev_linear_vel, prev_angular_vel]` at indices [10:12], pushing LiDAR to [12:36].

The RL environment (`JetbotNavigationEnv`) always uses 34D observations with LiDAR (36D with `--add-prev-action`).
The keyboard controller uses 10D by default; pass `--use-lidar` for 34D.

**Feature extractor split**: `state = obs[:obs_dim-24]`, `lidar = obs[obs_dim-24:]` — dynamic split supports both 34D and 36D observations.

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

# CrossQ + Chunk CVAE + Q-Chunking (recommended, ~35-39 steps/s)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000

# Legacy TQC (slower but proven, ~2.9 steps/s at UTD=20)
./run.sh train_sac.py --demos demos/recording.npz --headless --legacy-tqc --utd-ratio 20

# Custom chunk size and UTD ratio
./run.sh train_sac.py --demos demos/recording.npz --headless --chunk-size 5 --utd-ratio 5

# CrossQ with custom demo ratio (75% demo, 25% online)
./run.sh train_sac.py --demos demos/recording.npz --headless --demo-ratio 0.75

# Custom CVAE hyperparameters
./run.sh train_sac.py --demos demos/recording.npz --headless \
  --cvae-epochs 200 --cvae-beta 0.05 --cvae-z-dim 16

# GRU recurrent policy via frame stacking
./run.sh train_sac.py --demos demos/recording.npz --headless --n-frames 4 --gru-hidden 128

# Resume training from checkpoint
./run.sh train_sac.py --demos demos/recording.npz --headless --resume models/checkpoints/crossq_jetbot_50000_steps.zip --timesteps 500000

# Include previous action in observations (36D instead of 34D)
./run.sh train_sac.py --demos demos/recording.npz --headless --add-prev-action

# SafeTQC: constrained RL with cost critic + Lagrange multiplier
./run.sh train_sac.py --demos demos/recording.npz --headless --safe

# SafeTQC with custom cost limit and cost type
./run.sh train_sac.py --demos demos/recording.npz --headless --safe --cost-limit 10.0 --cost-type both

# SafeTQC keeping proximity penalty in reward (default: auto-removed)
./run.sh train_sac.py --demos demos/recording.npz --headless --safe --keep-proximity-reward

# Evaluation (auto-detects CrossQ/TQC/SAC/PPO and chunk size)
./run.sh eval_policy.py models/crossq_jetbot.zip --episodes 100
./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100

# Evaluation with cost tracking
./run.sh eval_policy.py models/crossq_jetbot.zip --episodes 100 --safe --cost-type proximity

# Evaluation with explicit chunk size or inflation radius
./run.sh eval_policy.py models/crossq_jetbot.zip --chunk-size 5 --inflation-radius 0.08

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
│   ├── train_sac.py                  # CrossQ/TQC/SAC + Chunk CVAE + Q-Chunking training script
│   ├── eval_policy.py                # Policy evaluation (auto-detects CrossQ/TQC/SAC/PPO + chunk size)
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

### CrossQ@UTD=1 vs TQC@UTD=20

CrossQ (default) achieves equal sample efficiency to TQC@UTD=20 at **UTD=1** via BatchRenorm in critics and no target networks. This eliminates the gradient bottleneck:

| Algorithm | UTD | n-frames | Steps/s | 1M Steps Wall Time | Speedup |
|-----------|-----|----------|---------|-------------------|---------|
| TQC       | 20  | 1        | ~2.9    | ~4 days           | 1x      |
| TQC       | 5   | 1        | ~10-12  | ~24 hours         | ~4x     |
| CrossQ    | 1   | 1        | ~35-39  | ~7 hours          | ~13x    |
| CrossQ    | 1   | 4 (GRU)  | ~17     | ~16 hours         | ~6x     |

**Measured step-time breakdown (chunk_size=5, headless, RTX 3090 Ti):**
- `world.step()` alone: **~1.3ms** per physics tick
- Full inner `env.step()` (physics + obs build + reward): **~3.0ms**
- `ChunkedEnvWrapper` total (5 sub-steps, Python overhead): **~15ms** (5 × 3.0ms, overhead≈0)
- CrossQ gradient step (GRU n-frames=4): **~43ms** (inferred from fps=17: 58ms total − 15ms physics)
- CrossQ gradient step (n-frames=1): **~1ms** (physics dominates at ~15–26ms per chunk)

The `rate=` field in `[DEBUG]` lines is a **cumulative average** (total_steps / total_elapsed), not instantaneous throughput. SB3's `fps` metric is the correct instantaneous rate. The cumulative average is inflated by fast startup phases (CVAE pretraining, demo loading).

TQC@UTD=20 adds ~330ms of gradient computation per wrapper step (20 × ~16ms per gradient step).

**Recommendation:** Use CrossQ (default) for ~13x speedup. Use `--legacy-tqc --utd-ratio 20` only if you need exact TQC reproduction.

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

The CrossQ/TQC pipeline uses action chunking to reduce compounding errors from single-step BC. The actor predicts k-step action chunks, a CVAE handles multimodal demonstrations, and a `ChunkedEnvWrapper` enables chunk-level Q-values (Q-chunking).

### Architecture
```
ChunkedEnvWrapper: action_space (2,) → (k*2,), executes k sub-steps per wrapper step
  R_chunk = Σ γ^i r_i, effective gamma = γ^k

Actor:  obs(34D) → ChunkCVAEFeatureExtractor → (obs_features || z=0) → latent_pi → mu → tanh → (k*2)
Critic: Q(obs_features || z=0, action_chunk) → scalar

ChunkCVAEFeatureExtractor (dynamic split: state_dim = obs_dim - 24):
  obs → split → [state 0:state_dim]       → symlog → MLP(state_dim→64→32) → 32D ┐
                 [lidar state_dim:obs_dim] → symlog → MLP(24→128→64)       → 64D ├→ concat → 96D + z_pad(8D) = 104D
                                                                                   └→ z_pad = zeros(z_dim)

TemporalCVAEFeatureExtractor (when --n-frames > 1):
  Wrapping order: ChunkedEnvWrapper( FrameStackWrapper( JetbotNavigationEnv ) )
  Input: (batch, n_frames * per_frame_dim) flattened
  Reshape → (batch, n_frames, per_frame_dim)
  Per-frame:  state_mlp(obs[:state_dim]) → 32D  }
              lidar_mlp(obs[state_dim:]) → 64D   } → 96D per frame
  GRU: (batch, n_frames, 96) → last hidden → (batch, gru_hidden_dim)
  Z-pad: concat(gru_output, zeros(z_dim)) → (gru_hidden_dim + z_dim)D

CVAE pretraining (replaces BC warmstart):
  Encoder (train-only): (obs_features, action_chunk) → z
  Decoder (= actor's latent_pi + mu): (obs_features, z) → action_chunk
  Loss: L1 reconstruction + β·KL (free bits floor at 0.25/dim, KL annealing over 40% of epochs)
  After pretraining: encoder discarded, z fixed to 0
```

### CVAE KL Collapse (Expected Behavior for Deterministic Demos)

The CVAE encoder will collapse to the prior (`active_dims=0/z_dim`) when demos are deterministic (e.g., A* autopilot). This is **correct behavior** — the decoder can reconstruct actions from `obs_features` alone because the A* planner is a deterministic function of the observation. The latent z carries no information by design.

**When collapse is expected:** Deterministic demo sources (A* autopilot, scripted policies). The CVAE effectively becomes a pure BC warmstart with z=0.

**When collapse is a problem:** Multi-modal human demos where different strategies exist for the same state. ACT (Zhao et al., RSS 2023) shows performance drops from 35% to 2% without CVAE on human data.

**Fixes for multi-modal demos (if needed in future):**
- **Observation dropout** (zero 30-50% of obs_features in decoder during training) — forces z usage
- **Increase beta** (0.1 → 1.0-10.0; ACT uses beta=10) — counterintuitive but higher beta prevents collapse
- **Cyclical annealing** (4 cycles instead of monotonic ramp)
- **Increase z_dim** (8 → 16-32; ACT uses 32)

**Key diagnostic:** Check `active_dims` in CVAE pretraining output. If L1 loss is low and decreasing, pretraining is working correctly regardless of z collapse.

### Pipeline Order
1. Create env with `ChunkedEnvWrapper(env, chunk_size=k, gamma=γ)`
2. Load demo transitions; **recompute demo rewards** with current `RewardComputer` to ensure consistency between demo and online data (avoids stale reward shaping from old reward function)
3. Build chunk-level demo transitions via `make_chunk_transitions()` → replay buffer
4. Create model with `ChunkCVAEFeatureExtractor`, gamma=γ^k, target_entropy=-k (one nat per chunk step; matches RLPD's `-dim(A)/2` heuristic for correlated chunked actions; avoids entropy collapse with tanh squashing)
5. Inject LayerNorm into critics (`inject_layernorm_into_critics()`) — skipped for CrossQ (BatchRenorm built-in)
6. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks via CVAE
7. **Reset `log_std` and `ent_coef` after CVAE** (critical — see entropy pitfall below): CVAE sets `log_std.bias=-2.0` (std=0.135) and `ent_coef_init=0.006`. This collapses entropy below `target_entropy` before SAC starts. Reset `log_std` to -0.5 and `ent_coef` to 0.1 via `--log-std-init -0.5 --ent-coef-init 0.1` (now default).
8. Copy pretrained feature extractor weights → critic (and critic_target if present; CrossQ has none)
9. Train with CrossQ/TQC/SAC (no auxiliary callback needed — CVAE encoder is discarded)

### Key Functions & Classes (`src/train_sac.py`)
- **symlog()**: DreamerV3 symmetric log compression
- **_set_bn_mode()**: Toggle BatchRenorm training mode if supported (CrossQ only; no-op for TQC/SAC)
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
| `--legacy-tqc` | off | Use TQC instead of CrossQ (legacy mode) |
| `--utd-ratio` | 1 | Update-to-data ratio (CrossQ=1, legacy TQC=20) |
| `--policy-delay` | 1 | Actor update delay (CrossQ paper uses 20 with UTD=20) |
| `--add-prev-action` | off | Include previous action in observations (36D instead of 34D) |
| `--chunk-size` | 10 | Action chunk size (k) |
| `--n-frames` | 1 | Number of observations to stack (1 = no stacking) |
| `--gru-hidden` | 128 | GRU hidden dimension (only used when n-frames > 1) |
| `--gru-lr` | 1e-5 | GRU learning rate (only used when n-frames > 1) |
| `--cvae-z-dim` | 8 | CVAE latent dimension |
| `--cvae-epochs` | 100 | CVAE pretraining epochs |
| `--cvae-beta` | 0.1 | CVAE KL weight |
| `--cvae-lr` | 1e-3 | CVAE pretraining learning rate |
| `--log-std-init` | -0.5 | Actor log_std after CVAE (fresh start). CVAE sets -2.0 (std=0.135) which collapses entropy before SAC starts; -0.5 (std=0.61) gives SAC room to explore. Use -2.0 to keep CVAE value. |
| `--ent-coef-init` | 0.1 | ent_coef to set after CVAE pretraining or on resume. CVAE/checkpoint may leave ent_coef near 0.006 giving negligible entropy bonus vs Q-values. Applied on both fresh start and `--resume`; pass 0 to disable. |
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
2. Load demo transitions with costs (`load_costs=True`); recompute demo rewards with current RewardComputer
3. Build chunk-level transitions with costs (`demo_costs=...`)
4. Create `CostReplayBuffer` with demo chunk costs
5. Create `SafeTQC` model (cost critic + Lagrange multiplier)
6. Inject LayerNorm into reward critics and cost critics
7. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks
8. Copy pretrained feature extractor weights → reward critic + cost critic
9. Train with `SafeTrainingCallback` tracking per-step costs

### Compatibility
- `inject_layernorm_into_critics`: Skipped for CrossQ (BatchRenorm built-in); operates on critic nets for TQC/SAC
- Replay buffer: Uses chunk-level transitions (obs, k*2 action, R_chunk, next_obs, done)
- `--resume`: Works — SB3 pickles `features_extractor_class`; chunk_size auto-detected from action_dim
- `eval_policy.py`: Auto-detects CrossQ/TQC/SAC/PPO, chunk_size, and n_frames from model spaces
- **CrossQ + SafeTQC**: Joint forward pass concatenates `[obs, next_obs]` (2*B batch) through critic so BatchRenorm sees the mixture distribution; cost critic has its own target network (independent of base algorithm); `_set_bn_mode()` toggles BatchRenorm training mode; actor updates gated by `policy_delay`
- **CrossQ + DualPolicy**: Joint forward pass concatenates `[obs, next_obs, next_obs]` (3*B batch) for IBRL current/RL/IL Q-values; `_set_bn_mode()` + `policy_delay` gating; polyak update guarded
- **Demo format**: Old demos (pre-OBS_VERSION=2) auto-converted to ego-centric layout on load via `convert_obs_to_egocentric()` in `demo_utils.py`; reads `arena_size` from demo metadata for correct workspace bounds normalization

## Training Diagnostics

The `VerboseEpisodeCallback` and SafeTQC train loop provide rich diagnostics to both console and TensorBoard.

### Console Output Format

**Every 100 steps:**
```
[DEBUG] step=   1000 | elapsed=102.9s | rate=9.7 steps/s | episodes=4 | policy_std=[0.1469] | ent=0.00656
```

**Every 500 steps (Q-value and buffer probe):**
```
[DIAG]  step=   500 | Q_pi=[+8.2, +12.3, +15.1] | H=-6.42 | buf=500/1000000 demos=189394
```
- `Q_pi=[min, mean, max]` — critic's value estimate under current policy (watch for overestimation)
- `H` — policy entropy (should be near target_entropy, not collapsing toward -inf)
- `buf` — online buffer fill / capacity + demo count

**Every episode end:**
```
[EP   30 END]    SUCCESS | steps=299 | return=+180.76 | min_lidar=0.090m | gd=0.45m | act=[0.15,0.42] | running SR=3.3% (1S/15C/14T) | t=865s
```
- `gd` — goal distance at episode end (close but timing out? or wandering?)
- `act=[lin,ang]` — mean |linear_vel|, mean |angular_vel| (is agent moving? spinning in circles?)

**Every 20 episodes (rolling summary):**
```
[SUMMARY @EP   20] last20: SR=5% CR=60% | ret=+65.3±40.2 | len=280 | goal_dist=2.10m | min_lid=0.120m
```

### TensorBoard Metrics

| Metric | Source | What it reveals |
|--------|--------|----------------|
| `diag/Q_pi_mean` | Callback | Critic value estimate — overestimation? |
| `diag/Q_pi_min`, `diag/Q_pi_max` | Callback | Q-value spread |
| `diag/policy_entropy` | Callback | Exploration level |
| `diag/buffer_online` | Callback | Online data accumulation |
| `rollout/ep_return` | Callback | Per-episode return |
| `rollout/ep_goal_dist` | Callback | Goal distance at episode end |
| `rollout/ep_min_lidar` | Callback | Closest obstacle approach |
| `rollout/return_20ep` | Callback | Rolling mean return (last 20 eps) |
| `rollout/sr_20ep` | Callback | Rolling success rate (last 20 eps) |
| `train/target_q_mean` | SafeTQC | Mean TD target (what critic learns toward) |
| `train/current_q_mean` | SafeTQC | Mean current Q estimate |
| `train/batch_reward_mean` | SafeTQC | Mean reward in sampled batch (demo/online mix) |
| `train/batch_action_mag` | SafeTQC | Mean |action| in batch |
| `timing/grad_total_ms` | SafeTQC | Wall time per gradient step (ms) |
| `timing/grad_critic_ms` | SafeTQC | Critic forward+backward time (ms) |
| `timing/grad_actor_ms` | SafeTQC | Actor forward+backward time (ms, when policy_delay fires) |

### Step-Timing Console Output (`[TIMING]` lines)

```
[TIMING] world.step(): 1.31ms avg (1000 calls)
[TIMING] ChunkedWrapper.step(): total=15.2ms | inner env.step()=3.0ms avg | overhead=0.0ms (chunk_size=5, n=500)
[TIMING] gradient step: total=43.1ms | critic=38.4ms | actor=4.2ms | other(sample+lagrange+polyak)=0.5ms
```
- `world.step()` prints every 1000 physics calls; `ChunkedWrapper` every 500 wrapper steps; gradient every 1000 gradient steps
- `inner env.step()` = physics + obs build + reward; `overhead` = Python loop cost (typically ≈0)
- gradient `other` = buffer sampling + Lagrange update + polyak copy
- **Caution**: `_time.perf_counter()` in `SafeTQC.train()` uses a local `import time as _t` — module-level `_time` does NOT resolve inside inner classes defined in closures

### What to Look For When Debugging

| Symptom | Likely Cause | Diagnostic to Check |
|---------|-------------|---------------------|
| High collision rate (>50%) | Agent ignores LiDAR | `act=[lin,ang]` — high angular = spinning; `min_lidar` pattern |
| 0% success rate after 50k steps | Goal reward too weak or agent can't navigate | `gd` — if decreasing, agent approaches but fails; if flat, agent wanders |
| `ent_coef` monotonically decreasing | Entropy death spiral, target_entropy too aggressive | Compare policy entropy `H` vs target_entropy; should stabilize |
| `ent_coef` oscillating near 0.002, entropy swings ±10 nats | CVAE collapsed log_std + insufficient ent_coef_init | Add `--log-std-init -0.5 --ent-coef-init 0.1`; reset on resume too |
| Q_pi swings ±20 in 3000 steps without diverging | Entropy collapse triggering sharp actor updates + BatchRenorm distribution shock | Fix entropy first (`--ent-coef-init 0.1`); optionally freeze actor 5k steps after resume |
| `ent_coef` monotonically increasing | Policy too deterministic | `policy_std` — if pinned near initial value, CVAE overtightened |
| Q_pi values growing unboundedly | Critic overestimation | `train/target_q_mean` — should stabilize; if growing > 1000, problem |
| Returns high but SR=0% | Reward exploitation (distance shaping) | Check if truncated episodes dominate; heading bonus is gated by progress |
| Near-miss episodes (gd<0.5m) still negative returns | Approach bonus too weak or radius too small | Check `APPROACH_BONUS_SCALE` and `APPROACH_BONUS_RADIUS`; near-misses at gd=0.25m should get ~+7.5 approach bonus |
| Proximity penalty > collision penalty | `PROXIMITY_SCALE` too high or `COLLISION_PENALTY` too low | Per-episode proximity should stay below collision magnitude; check hierarchy: collision >> proximity >> time |

## Known Pitfalls & Fixes

### target_entropy for Chunked Actions
- **Wrong:** `target_entropy="auto"` → SB3 computes `-dim(A)` = `-chunk_size*2` (e.g., -10 for chunk_size=5). This is too aggressive for correlated chunked actions and causes entropy death spiral.
- **Correct:** `target_entropy=-chunk_size` (e.g., -5 for chunk_size=5). Matches RLPD's `-dim(A)/2` heuristic. One nat per temporal step in the chunk.

### Demo Observation v1→v2 Conversion and Arena Size
- Old demos (OBS_VERSION=1) are auto-converted to ego-centric layout (v2) on load
- `convert_obs_to_egocentric()` normalizes positions using workspace bounds: `(x - x_min) / (x_max - x_min)`
- Must read `arena_size` from demo metadata to get correct bounds. Using wrong bounds (e.g., default 4m when demo was 10m) produces obs[0:2] outside [0,1]
- **Always match `--arena-size` between demo collection and training**

### Demo Reward Recomputation
- Demo rewards are recomputed with current `RewardComputer` at load time to ensure consistency with online data
- Prevents stale reward shaping terms (old heading bonus, old goal reward) from corrupting the critic
- RLPD assumes reward consistency between demo and online transitions; mismatch causes contradictory gradient signals

### CVAE Beta Selection
- `--cvae-beta 0.1` (default) works well for deterministic demos
- `--cvae-beta 10` causes immediate KL collapse (36x KL dominance over reconstruction)
- For multi-modal demos, higher beta (1.0-10.0) with larger encoder is needed (ACT paper uses beta=10)
- **Lower beta does NOT prevent collapse** — it makes collapse worse by reducing pressure on encoder

### Reward Hierarchy Design
The reward constants are tuned to maintain a clear penalty hierarchy:
- **Collision (-25)** >> **max proximity accumulation (~-10/episode)** >> **time penalty (-10 over 2000 sub-steps)**
- Without this hierarchy, proximity penalty can accumulate to -20+/episode — worse than collision (-10), making the agent prefer crashing over cautious navigation
- The **approach bonus** (potential-based, SCALE=10.0, RADIUS=1.0m) smooths the "success cliff": without it, an agent at gd=0.26m gets -6.65 while gd=0.15m gets +49 — a 55-point cliff that's hard for the critic to model. With it, gd=0.26m gets ~+1, gd=0.15m gets ~+57 — much smoother gradient
- Potential-based shaping (Ng et al., 1999) is provably unexploitable: orbiting at fixed distance gives 0, oscillating in/out cancels. Only net approach toward goal yields reward
- **Heading bonus (0.5)** is gated by forward progress (`prev_dist > curr_dist`), so it's proportional to `progress * alignment` — can't be exploited by slow creeping or circling

### CVAE Entropy Death Spiral + Q Instability (Critical)

**Root cause (confirmed via instrumentation, 2026-02-25):**

CVAE pretraining sets `log_std.bias = -2.0` (std=0.135). For a 10D tanh-squashed policy, this puts entropy at ≈ -6 to -10 nats — already **below** `target_entropy=-chunk_size` before SAC starts. With `ent_coef_init=0.006`, the actor entropy bonus is `0.006 × (-10) = -0.06`, which is ~200× weaker than Q-values (±10–20). The actor ignores entropy entirely, making sharp greedy updates. On `--resume`, the checkpoint's collapsed ent_coef (~0.002) is restored, making the problem worse.

**Without the fix, a death spiral occurs:**
1. CVAE policy already below target entropy → SAC auto-tuner tries to increase ent_coef
2. But ent_coef too small → entropy bonus negligible → actor stays deterministic
3. Near-deterministic policy makes sharp Q updates → critic diverges → Q oscillates
4. Q oscillation with no entropy regularization → ±20 Q swings in 3000 steps (observed)
5. `ent_coef` oscillates around zero, never stabilizing

**Fix (now default via `--log-std-init -0.5 --ent-coef-init 0.1`):**
- Reset `log_std.bias` from -2.0 → -0.5 (std 0.135 → 0.61) after CVAE: preserves learned action means, gives SAC room to tune variance
- Reset `ent_coef` from 0.006 → 0.1 after CVAE or on resume: entropy bonus `0.1 × (-10) = -1.0` is now non-trivial vs Q-values
- On resume: `log_ent_coef` tensor is explicitly overwritten (SB3 restores it from checkpoint otherwise)

**Root cause for Q oscillation (separately):**
- Post-resume distribution shock: BatchRenorm running stats calibrated at checkpoint time see shifted distribution from new online data
- Action chunking (k=5 → 10D action space) amplifies any critic overestimation episode
- CrossQ three-distribution mixing (demo / old-online / new-online) breaks the two-distribution BatchRenorm assumption (CrossQ paper only analyzes two distributions)
- Fix: `--ent-coef-init 0.1` breaks the death spiral; a 5k-step frozen-actor warmup after resume would further stabilize BatchRenorm stats

**Healthy post-fix signs:**
- `policy_std` slowly rising 0.61 → 0.65+ over first 5k steps
- `ent_coef` starting ~0.1, gently adjusting toward equilibrium
- `Q_pi` varying ≤ ±5 units over 1000 steps (not ±20)

**References:** RLPD (Ball et al., ICML 2023), CrossQ (Bhatt et al., ICLR 2024), "Understanding Q-Value Divergence in Offline-RL" (Zheng et al., NeurIPS 2023), "Scaling CrossQ with Weight Normalization" (Palo et al., arXiv 2506.03758)

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
- **Goal reached**: +50.0 (terminal)
- **Collision**: -25.0 (terminal, LiDAR distance < 0.08m)
- **Distance shaping**: `(prev_dist - curr_dist) * 1.0`
- **Heading bonus**: `((pi - |angle_to_goal|) / pi) * 0.5` — **only when making forward progress** (`prev_dist > curr_dist`), prevents reward exploitation from circling near the goal
- **Approach bonus**: Potential-based shaping within 1m of goal — `10.0 * (max(0, 1 - curr_dist/1.0) - max(0, 1 - prev_dist/1.0))`. Amplifies distance shaping 10x near the goal. Provably unexploitable: orbiting gives 0, oscillating cancels out (Ng et al., 1999)
- **Proximity penalty**: `0.05 * (1.0 - min_lidar / 0.3)` when min_lidar < 0.3m, gated by goal distance (linearly reduced within 0.5m of goal, zero at goal). Auto-removed when `--safe` is active (handled by cost critic); `--keep-proximity-reward` to override
- **Time penalty**: -0.005 per step

### Sparse Mode
- **Goal reached**: +50.0
- **Collision**: -25.0
- **Otherwise**: 0.0

## Dependencies

- Isaac Sim 5.1.0
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
- sb3-contrib ≥ 2.4.0 (for CrossQ and TQC; falls back to SAC without it)
- gymnasium
- tensorboard
