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
   - `jetbot_config.py` - Single source of truth for robot physical constants (`WHEEL_RADIUS`, `WHEEL_BASE`, velocity limits, start pose, workspace bounds), `quaternion_to_yaw()` utility, `OBS_VERSION` constant, and camera constants (`CAMERA_PRIM_SUFFIX`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `IMAGE_FEATURE_DIM`)
   - `demo_utils.py` - Shared demo data functions: `validate_demo_data()`, `load_demo_data()`, `load_demo_transitions()`, `extract_action_chunks()`, `make_chunk_transitions()`, `build_frame_stacks()`, `convert_obs_to_egocentric()`, `VerboseEpisodeCallback` (enhanced training diagnostics), and DINOv2 utilities (`encode_images_dinov2()`, `build_camera_obs()`, `load_demo_images()`)
   - `demo_io.py` - Unified demo I/O adapter: `open_demo()` dispatches `.npz`/`.hdf5`, `HDF5DemoWriter` for O(delta) incremental recording (with optional `/images` dataset for camera demos), `convert_npz_to_hdf5()` migration utility

### Key Classes

- **DemoRecorder**: Records (obs, action, reward, done, cost) tuples; HDF5 incremental writes by default, falls back to NPZ
- **open_demo()**: Unified reader dispatching `.npz`/`.hdf5` — returns dict-like `NpzDemoData` or `Hdf5DemoData` (`src/demo_io.py`)
- **HDF5DemoWriter**: Incremental HDF5 writer with O(delta) `append_steps()`/`flush()` (`src/demo_io.py`)
- **RewardComputer**: Computes navigation rewards; `compute_cost()` for SafeTQC; `safe_mode` suppresses proximity penalty
- **LidarSensor**: Analytical 2D raycasting for obstacle detection (no physics dependency)
- **AutoPilot**: A*-based expert controller with privileged scene access for collision-free demo collection
- **FrameStackWrapper**: Stacks last n_frames observations into (n_frames * obs_dim,), oldest first (`src/jetbot_rl_env.py`)
- **ChunkedEnvWrapper**: Converts single-step env to k-step chunked env for Q-chunking (`src/jetbot_rl_env.py`); info dict includes `goal_distance`, `min_lidar_distance`, `collision`, `is_success`, `cost`
- **ChunkCVAEFeatureExtractor**: Dynamic split: state MLP (state_dim->32D) + LiDAR MLP (24->64D) + z-pad (8D) = 104D. Split is `obs[:obs_dim-24]` / `obs[obs_dim-24:]` for 34D/36D
- **VisionCVAEFeatureExtractor**: Three-way split for `--use-camera`: state MLP (state_dim->32D) + image MLP (384->64D, no symlog) + LiDAR MLP (24->64D) + z-pad (8D) = 168D (`src/train_sac.py`)
- **TemporalCVAEFeatureExtractor**: GRU-based feature extractor for frame-stacked obs (`src/train_sac.py`)
- **SafeTQC**: CrossQ/TQC subclass with dual cost critic + Lagrange multiplier (`src/train_sac.py`)
- **CostReplayBuffer**: Parallel ring buffer storing per-transition costs (`src/train_sac.py`)

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

With `--use-camera` (418D/420D): inserts 384D DINOv2 ViT-S/14 CLS features between state and LiDAR:
```
Without camera: [state(10D), lidar(24D)] = 34D
With camera:    [state(10D), image_features(384D), lidar(24D)] = 418D
With camera+prev_action: [state(10D), prev_action(2D), image_features(384D), lidar(24D)] = 420D
```
LiDAR always occupies the last 24 dimensions. DINOv2 features are extracted inside `env.step()` from an 84x84 RGB camera. Raw images are stored in HDF5 demos; DINOv2 encoding happens at training load time via `encode_images_dinov2()`.

The RL environment (`JetbotNavigationEnv`) always uses 34D observations with LiDAR (36D with `--add-prev-action`, 418D/420D with `--use-camera`).
The keyboard controller uses 10D by default; pass `--use-lidar` for 34D.

**Feature extractor split**: `state = obs[:obs_dim-24]`, `lidar = obs[obs_dim-24:]` — dynamic split supports both 34D and 36D observations. With `--use-camera`, `VisionCVAEFeatureExtractor` uses a three-way split: `state = obs[:state_dim]`, `image = obs[state_dim:-24]`, `lidar = obs[-24:]`.

### Action Space (2D)
```
[0] - Linear velocity command (m/s)
[1] - Angular velocity command (rad/s)
```

## Running the Project

```bash
# Teleoperation
./run.sh                                      # camera streaming on by default
./run.sh --no-camera                          # disable camera
./run.sh --enable-recording                   # manual recording

# Automatic demo collection (--automatic forces --use-lidar)
./run.sh --enable-recording --automatic --num-episodes 200
./run.sh --enable-recording --automatic --continuous --headless-tui

# PPO Training
./run.sh train_rl.py --headless --timesteps 500000
./run.sh train_rl.py --headless --bc-warmstart demos/recording.npz --timesteps 1000000

# CrossQ + Chunk CVAE + Q-Chunking (recommended, ~35-39 steps/s)
./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000

# Legacy TQC (~2.9 steps/s at UTD=20)
./run.sh train_sac.py --demos demos/recording.npz --headless --legacy-tqc --utd-ratio 20

# Resume training from checkpoint
./run.sh train_sac.py --demos demos/recording.npz --headless --resume models/checkpoints/crossq_jetbot_50000_steps.zip

# SafeTQC: constrained RL with cost critic + Lagrange multiplier
./run.sh train_sac.py --demos demos/recording.npz --headless --safe

# DINOv2 camera training (requires camera demos)
./run.sh train_sac.py --demos demos/camera_demo.hdf5 --headless --use-camera

# Record camera demos
./run.sh --enable-recording --automatic --use-camera --num-episodes 50

# Evaluation (auto-detects algorithm, chunk size, and camera from obs dim)
./run.sh eval_policy.py models/crossq_jetbot.zip --episodes 100

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

CrossQ (default) achieves equal sample efficiency to TQC@UTD=20 at **UTD=1** via BatchRenorm in critics and no target networks:

| Algorithm | UTD | n-frames | Steps/s | 1M Steps Wall Time | Speedup |
|-----------|-----|----------|---------|-------------------|---------|
| TQC       | 20  | 1        | ~2.9    | ~4 days           | 1x      |
| TQC       | 5   | 1        | ~10-12  | ~24 hours         | ~4x     |
| CrossQ    | 1   | 1        | ~35-39  | ~7 hours          | ~13x    |
| CrossQ    | 1   | 4 (GRU)  | ~17     | ~16 hours         | ~6x     |

Step-time breakdown (chunk_size=5, headless, RTX 3090 Ti): `world.step()` ~1.3ms, `env.step()` ~3.0ms, `ChunkedEnvWrapper` ~15ms (5 sub-steps), CrossQ gradient ~1ms (n-frames=1) or ~23ms (GRU). SB3 framework overhead ~13ms/step. Note: `rate=` in `[DEBUG]` lines is cumulative average, not instantaneous — use SB3's `fps` metric.

**Recommendation:** Use CrossQ (default). Use `--legacy-tqc --utd-ratio 20` only for exact TQC reproduction.

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

**Camera mode exception**: When `--use-camera` is active, headless optimizations that disable rendering are skipped (`enable_cameras=True`, no `rendering_dt=1.0`, no `disable_viewport_updates`). The camera requires active rendering even in headless mode, reducing throughput to ~10-20 steps/s.

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

VisionCVAEFeatureExtractor (when --use-camera):
  obs → split → [state 0:state_dim]                → symlog → MLP(state_dim→64→32) → 32D ┐
                 [image state_dim:state_dim+384]    →          MLP(384→256→64)      → 64D ├→ concat → 160D + z_pad(8D) = 168D
                 [lidar obs_dim-24:]                → symlog → MLP(24→128→64)       → 64D ┘
  Note: No symlog on DINOv2 features (already normalized by ImageNet stats)

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

### CVAE KL Collapse

KL collapse (`active_dims=0`) is **expected** for deterministic demos (A* autopilot) — the decoder reconstructs from `obs_features` alone. Check `active_dims` in CVAE output; if L1 loss is low and decreasing, pretraining is correct regardless of z collapse. `--cvae-beta 0.1` works for deterministic demos; `--cvae-beta 10` causes immediate collapse.

### Pipeline Order
1. Create env with `ChunkedEnvWrapper(env, chunk_size=k, gamma=γ)`
2. Load demo transitions; **recompute demo rewards** with current `RewardComputer` to ensure consistency between demo and online data (avoids stale reward shaping from old reward function)
3. Build chunk-level demo transitions via `make_chunk_transitions()` → replay buffer
4. Create model with `ChunkCVAEFeatureExtractor`, gamma=γ^k, target_entropy=`--target-entropy` (default: -chunk_size; override with 0 or +5 for tanh-squashed chunked actions where actual entropy is ~+10 nats)
5. Inject LayerNorm into critics (`inject_layernorm_into_critics()`) — skipped for CrossQ (BatchRenorm built-in)
6. `pretrain_chunk_cvae()` — trains feature extractor + actor on demo chunks via CVAE
7. **Reset `log_std` and `ent_coef` after CVAE** (critical — see entropy pitfall below): CVAE sets `log_std.bias=-2.0` (std=0.135) and `ent_coef_init=0.006`. The new stability system (`--mean-clamp`, `--log-std-min`, `--ent-coef-min`) prevents entropy collapse without requiring log_std inflation. Pass `--log-std-init -0.5` for the old behavior of boosting exploration noise.
8. Copy pretrained feature extractor weights → critic (and critic_target if present; CrossQ has none)
9. Train with CrossQ/TQC/SAC (no auxiliary callback needed — CVAE encoder is discarded)

### CLI Flags
| Flag | Default | Description |
|------|---------|-------------|
| `--legacy-tqc` | off | Use TQC instead of CrossQ |
| `--utd-ratio` | 1 | Update-to-data ratio (CrossQ=1, TQC=20) |
| `--policy-delay` | 1 | Actor update delay |
| `--add-prev-action` | off | 36D obs (adds prev action) |
| `--chunk-size` | 10 | Action chunk size (k) |
| `--n-frames` | 1 | Obs stack depth (1=no stacking) |
| `--gru-hidden` | 128 | GRU hidden dim (n-frames>1 only) |
| `--gru-lr` | 1e-5 | GRU learning rate (n-frames>1 only) |
| `--cvae-z-dim` | 8 | CVAE latent dimension |
| `--cvae-epochs` | 100 | CVAE pretraining epochs |
| `--cvae-beta` | 0.1 | CVAE KL weight |
| `--cvae-lr` | 1e-3 | CVAE learning rate |
| `--log-std-init` | -2.0 | Actor log_std after CVAE/resume (-2.0=keep, -0.5=boost exploration) |
| `--ent-coef-init` | 0.1 | ent_coef after CVAE/resume (0=disable reset) |
| `--mean-clamp` | 3.0 | Clamp \|pre-tanh mean\| (0=disable) |
| `--mean-reg` | 0.001 | L2 reg on pre-tanh means (0=disable) |
| `--log-std-min` | -5.0 | log_std floor (SB3 default: -20) |
| `--ent-coef-min` | 0.005 | ent_coef floor (0=disable) |
| `--target-entropy` | -chunk_size | SAC target entropy; use 0 or +5 for reachable equilibrium |
| `--no-backup-entropy` | off | Remove entropy from TD target |
| `--use-camera` | off | DINOv2 camera features (418D obs) |
| `--safe` | off | SafeTQC (cost critic + Lagrange) |
| `--cost-limit` | 25.0 | Per-episode cost budget |
| `--lagrange-lr` | 3e-4 | Lagrange multiplier LR |
| `--lagrange-init` | 0.0 | Initial log-lambda |
| `--cost-n-critics` | 2 | Cost critic count |
| `--cost-critic-type` | mean | `mean` or `quantile` |
| `--cost-type` | proximity | `proximity`, `collision`, or `both` |
| `--keep-proximity-reward` | off | Keep proximity penalty with `--safe` |

### SafeTQC Pipeline

Same as above but: env uses `safe_mode=True` (removes proximity penalty); load demos with `load_costs=True`; build chunk transitions with costs; create `CostReplayBuffer` + `SafeTQC` model; inject LayerNorm into both reward and cost critics; copy feature extractor to both critics; train with `SafeTrainingCallback`.

### Compatibility
- `--resume`: SB3 pickles `features_extractor_class`; chunk_size auto-detected from action_dim
- `eval_policy.py`: Auto-detects CrossQ/TQC/SAC/PPO, chunk_size, n_frames, and camera from model obs dimension (`obs_dim - 384` in `{34, 36}` implies camera)
- **Demo format**: Old demos (pre-OBS_VERSION=2) auto-converted to ego-centric via `convert_obs_to_egocentric()` in `demo_utils.py`; reads `arena_size` from demo metadata
- **Camera demos**: HDF5 demos with `/images` dataset (84x84x3 uint8, gzip); old demos without images work normally (`load_images=False` by default)

## Training Diagnostics

The `VerboseEpisodeCallback` and SafeTQC train loop provide rich diagnostics to both console and TensorBoard.

### Console Output Format

```
[DEBUG] step=   1000 | elapsed=102.9s | rate=9.7 steps/s | episodes=4 | policy_std=[0.1469] | ent=0.00656
[DIAG]  step=   500 | Q_pi=[+8.2, +12.3, +15.1] | H=-6.42 | buf=500/1000000 demos=189394
[EP   30 END]    SUCCESS | steps=299 | return=+180.76 | min_lidar=0.090m | gd=0.45m | act=[0.15,0.42] | running SR=3.3% (1S/15C/14T) | t=865s
[SUMMARY @EP   20] last20: SR=5% CR=60% | ret=+65.3±40.2 | len=280 | goal_dist=2.10m | min_lid=0.120m
```
Key fields: `Q_pi=[min,mean,max]` (critic estimate), `H` (entropy, should be near target), `gd` (goal distance at end), `act=[lin,ang]` (mean velocities).

### TensorBoard Metrics

| Metric | Description |
|--------|-------------|
| `diag/Q_pi_mean`, `_min`, `_max` | Critic value estimate and spread |
| `diag/policy_entropy` | Exploration level |
| `diag/mean_mu_abs` | Pre-tanh mean magnitude (>2.0 = saturation risk) |
| `diag/ent_coef` | Current entropy coefficient |
| `rollout/ep_return`, `ep_goal_dist`, `ep_min_lidar` | Per-episode metrics |
| `rollout/return_20ep`, `sr_20ep` | Rolling 20-episode averages |
| `train/target_q_mean`, `current_q_mean` | TD target vs current Q (SafeTQC) |
| `train/mean_mu_abs`, `mean_reg_loss` | Stability diagnostics (SafeTQC) |
| `timing/grad_total_ms`, `grad_critic_ms`, `grad_actor_ms` | Gradient step wall time |

### Step-Timing Console Output (`[TIMING]` lines)

```
[TIMING] world.step(): 1.13ms avg (1000 calls)
[TIMING] ChunkedWrapper.step(): total=13.8ms | inner env.step()=2.8ms avg | overhead=0.0ms (chunk_size=5, n=500)
[TIMING] gradient step: total=23.3ms avg (1000 grad steps)          # CrossQ
[TIMING] gradient step: total=43.1ms | critic=38.4ms | actor=4.2ms  # SafeTQC
```

### What to Look For When Debugging

| Symptom | Likely Cause | Diagnostic to Check |
|---------|-------------|---------------------|
| High collision rate (>50%) | Agent ignores LiDAR | `act=[lin,ang]` — high angular = spinning; `min_lidar` pattern |
| 0% success rate after 50k steps | Goal reward too weak or agent can't navigate | `gd` — if decreasing, agent approaches but fails; if flat, agent wanders |
| `ent_coef` monotonically decreasing | Entropy death spiral, target_entropy too aggressive | Compare policy entropy `H` vs target_entropy; should stabilize |
| `ent_coef` oscillating near 0.002, entropy swings ±10 nats | CVAE collapsed log_std + insufficient ent_coef_init | New defaults fix this: `--ent-coef-min 0.005` floors ent_coef, `--mean-clamp 3.0` prevents tanh saturation. Optionally pass `--log-std-init -0.5 --ent-coef-init 0.1` for the old fix. |
| Q_pi swings ±20 in 3000 steps without diverging | Entropy collapse triggering sharp actor updates + BatchRenorm distribution shock | Fix entropy first (`--ent-coef-init 0.1`); optionally freeze actor 5k steps after resume |
| `ent_coef` monotonically increasing | Policy too deterministic | `policy_std` — if pinned near initial value, CVAE overtightened |
| Q_pi values growing unboundedly | Critic overestimation | `train/target_q_mean` — should stabilize; if growing > 1000, problem |
| Returns high but SR=0% | Reward exploitation (distance shaping) | Check if truncated episodes dominate; heading bonus is gated by progress |
| Near-miss episodes (gd<0.5m) still negative returns | Approach bonus too weak or radius too small | Check `APPROACH_BONUS_SCALE` and `APPROACH_BONUS_RADIUS`; near-misses at gd=0.25m should get ~+7.5 approach bonus |
| Proximity penalty > collision penalty | `PROXIMITY_SCALE` too high or `COLLISION_PENALTY` too low | Per-episode proximity should stay below collision magnitude; check hierarchy: collision >> proximity >> time |
| `\|mu\|` > 2.0 in `[DEBUG]` lines | Pre-tanh mean explosion / tanh saturation | `--mean-clamp 3.0` (default) prevents runaway; `--mean-reg 0.001` adds L2 penalty; check `train/mean_mu_abs` in TensorBoard |
| `ent_coef` pinned at floor (0.005) | target_entropy unreachable OR policy too deterministic | If `ent_coef_loss` is strongly negative, raise `--target-entropy` (e.g., 0 or +5). If `ent_coef_loss` oscillates near 0, the floor is correct and the policy is near equilibrium. |

## Known Pitfalls & Fixes

### target_entropy for Chunked Actions
- **Never** use `target_entropy="auto"` (SB3 gives `-chunk_size*2`, causes entropy death spiral)
- Default `-chunk_size` is still unreachable for tanh-squashed chunked actions (actual H~+10 nats)
- **Recommended:** `--target-entropy 0` (conservative) or `--target-entropy 5` (moderate exploration)

### Demo Observation v1→v2 Conversion and Arena Size
- Old demos (OBS_VERSION=1) are auto-converted to ego-centric layout (v2) on load
- `convert_obs_to_egocentric()` normalizes positions using workspace bounds: `(x - x_min) / (x_max - x_min)`
- Must read `arena_size` from demo metadata to get correct bounds. Using wrong bounds (e.g., default 4m when demo was 10m) produces obs[0:2] outside [0,1]
- **Always match `--arena-size` between demo collection and training**

### Demo Reward Recomputation
- Demo rewards are recomputed with current `RewardComputer` at load time to ensure consistency with online data
- Prevents stale reward shaping terms (old heading bonus, old goal reward) from corrupting the critic
- RLPD assumes reward consistency between demo and online transitions; mismatch causes contradictory gradient signals

### Isaac Sim Console Encoding (cp1252)
Isaac Sim's Python process uses **Windows cp1252** encoding for stdout/stderr. Unicode characters like `→` (U+2192), `±`, `×` in `print()` cause a fatal `UnicodeEncodeError: 'charmap' codec can't encode character` that crashes training immediately. Use ASCII equivalents (`->`, `+/-`, `x`) in all print statements inside scripts run via `run.bat`.

### Corrupted Checkpoint + Large ent_coef Jump -> Actor NaN
- Resuming from unstable checkpoint with large `--ent-coef-init` jump can cause NaN in actor within ~200 steps
- **Symptom**: `ValueError: Expected parameter loc ... found invalid values: tensor([[nan, ...]])`
- **Fix**: Use a cleaner checkpoint or reduce `--ent-coef-init` (e.g., 0.01 instead of 0.1)

### Tanh Saturation and Pre-Tanh Mean Explosion
Large pre-tanh means (|mu| > 1.7) saturate tanh, making the policy near-deterministic. The **stability system** (all on by default) prevents this: `--mean-clamp 3.0`, `--mean-reg 0.001`, `--log-std-min -5.0`, `--ent-coef-min 0.005`. Watch `diag/mean_mu_abs` in TensorBoard — should stay < 2.0.

### Reward Hierarchy Design
- **Collision (-25)** >> **max proximity (~-10/ep)** >> **time penalty (-10 over 2000 sub-steps)**
- **Approach bonus** (potential-based, SCALE=10.0, RADIUS=1.0m) smooths the success cliff near goal. Provably unexploitable (Ng et al., 1999)
- **Heading bonus** (0.5) gated by forward progress — can't be exploited by circling

### CVAE Entropy Death Spiral + Q Instability
CVAE pretraining sets low `log_std` (std=0.135) and `ent_coef` (~0.006), making the entropy bonus negligible vs Q-values. Without the stability system, the actor ignores entropy, makes sharp greedy updates, and Q oscillates +-20 in 3000 steps.

**Fix (now default):** Stability system (`--mean-clamp 3.0 --log-std-min -5.0 --ent-coef-min 0.005`) plus `--ent-coef-init 0.1` and `--log-std-init` reset after CVAE/resume. Pass `--log-std-init -2.0` or `--ent-coef-init 0` to keep checkpoint values.

**Healthy signs:** `policy_std` slowly rising over first 5k steps, `ent_coef` ~0.1 gently adjusting, `Q_pi` varying <= +-5 per 1000 steps.

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
