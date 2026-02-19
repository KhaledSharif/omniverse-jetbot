# Plan 01: Domain Randomization for Sim-to-Real Transfer

## Motivation

Domain randomization is the single highest-impact change for eventual real-world deployment. The current environment uses fixed physics parameters, fixed sensor models, and deterministic kinematics. A policy trained this way will almost certainly fail on a real JetBot due to the sim-to-real gap in friction, motor response, sensor noise, and latency.

Research confirms that for LiDAR-based mobile robot navigation, **the LiDAR noise model is more important than dynamics randomization** for sim-to-real transfer (NeuronsGym, 2023). A 2025 study demonstrated zero-shot transfer of navigation policies from Isaac Sim to real ROS 2 robots using domain randomization with 2D LiDAR.

## What to Randomize

### Tier 1 — High Priority

| Parameter | Range | Distribution | Frequency | Rationale |
|-----------|-------|-------------|-----------|-----------|
| **LiDAR Gaussian noise** (sigma) | 0.005–0.03 m | Uniform per-episode sigma, Gaussian per-step noise | Per-step | Real LiDARs have measurement noise; most impactful for sim-to-real |
| **LiDAR ray dropout** | 0–5% of rays | Uniform per-episode rate, Bernoulli per-step | Per-step | Real sensors miss returns on specular/dark surfaces |
| **Ground static friction** | 0.4–1.2 | Uniform | Per-episode | Varies with floor surface (tile, carpet, wood) |
| **Ground dynamic friction** | 0.3–0.8 | Uniform | Per-episode | Affects braking and turning dynamics |
| **Motor asymmetry** (L/R scale) | 0.90–1.10 each | Uniform, independent per wheel | Per-episode | Real motors are never perfectly matched; causes drift |
| **Wheel radius perturbation** | +/-5% of 0.03 m | Uniform scaling | Per-episode | Tire wear, manufacturing tolerance |

### Tier 2 — Medium Priority

| Parameter | Range | Distribution | Frequency | Rationale |
|-----------|-------|-------------|-----------|-----------|
| **Wheel base perturbation** | +/-3% of 0.1125 m | Uniform scaling | Per-episode | Mechanical tolerance |
| **Action delay** | 0–3 timesteps (0–50 ms at 60 Hz) | Uniform integer | Per-episode | Communication + motor inertia latency |
| **Action noise** (lin/ang) | sigma 0.006 m/s / 0.05 rad/s | Gaussian | Per-step | Actuator imprecision |
| **Robot mass** | +/-15% of 1 kg | Uniform scaling | Per-episode | Battery weight variation, payload |
| **Observation noise** (position) | sigma 0.01–0.03 m | Gaussian | Per-step | Odometry drift |
| **Observation noise** (heading) | sigma 0.01–0.03 rad | Gaussian | Per-step | Gyro drift |

### Tier 3 — Low Priority

| Parameter | Range | Distribution | Frequency | Rationale |
|-----------|-------|-------------|-----------|-----------|
| **Observation noise** (velocity) | sigma 0.01 / 0.03 | Gaussian | Per-step | Encoder noise |
| **LiDAR angular offset** | sigma 0.01–0.02 rad | Gaussian | Per-episode | Mounting misalignment |
| **LiDAR bias** | -0.02 to +0.02 m | Uniform | Per-episode | Calibration error |
| **Start position offset** | +/-0.1 m in x,y | Uniform | Per-episode | Localization error |

## Implementation Plan

### Step 1: Add a `DomainRandomizer` class to `jetbot_rl_env.py`

```python
class DomainRandomizer:
    """Samples and applies domain randomization parameters."""

    def __init__(self, np_random, enabled=True):
        self.np_random = np_random
        self.enabled = enabled
        # Per-episode parameters (sampled in .reset())
        self.lidar_noise_sigma = 0.0
        self.lidar_dropout_rate = 0.0
        self.left_motor_scale = 1.0
        self.right_motor_scale = 1.0
        self.wheel_radius = WHEEL_RADIUS
        self.wheel_base = WHEEL_BASE
        self.action_delay = 0
        self.obs_noise_scales = np.zeros(10)

    def sample_episode_params(self):
        """Called at each env reset. Samples per-episode DR params."""
        if not self.enabled:
            return
        self.lidar_noise_sigma = self.np_random.uniform(0.005, 0.03)
        self.lidar_dropout_rate = self.np_random.uniform(0.0, 0.05)
        self.left_motor_scale = self.np_random.uniform(0.90, 1.10)
        self.right_motor_scale = self.np_random.uniform(0.90, 1.10)
        self.wheel_radius = WHEEL_RADIUS * self.np_random.uniform(0.95, 1.05)
        self.wheel_base = WHEEL_BASE * self.np_random.uniform(0.97, 1.03)
        self.action_delay = self.np_random.integers(0, 4)
        # ...

    def apply_lidar_noise(self, distances, max_range):
        """Called per-step on raw LiDAR readings."""
        if not self.enabled:
            return distances
        noise = self.np_random.normal(0, self.lidar_noise_sigma, distances.shape)
        distances = distances + noise
        dropout = self.np_random.random(distances.shape) < self.lidar_dropout_rate
        distances[dropout] = max_range
        return np.clip(distances, 0, max_range)

    def apply_motor_asymmetry(self, wheel_velocities):
        """Called per-step on wheel velocity commands."""
        if not self.enabled:
            return wheel_velocities
        wheel_velocities[0] *= self.left_motor_scale
        wheel_velocities[1] *= self.right_motor_scale
        return wheel_velocities
```

### Step 2: Integrate into `JetbotNavigationEnv`

- In `__init__`: create `DomainRandomizer`, add `--domain-rand` CLI flag
- In `reset()`: call `self.dr.sample_episode_params()`, recreate `DifferentialController` with perturbed `wheel_radius`/`wheel_base`
- In `step()`: apply motor asymmetry to wheel actions, apply action noise, handle action delay buffer
- In `_build_observation()`: apply LiDAR noise after `lidar_sensor.scan()`, apply observation noise to state channels

### Step 3: Action delay buffer

```python
# In __init__:
self._action_buffer = collections.deque(maxlen=4)

# In reset():
self._action_buffer.clear()
for _ in range(self.dr.action_delay + 1):
    self._action_buffer.append(np.zeros(2))

# In step():
self._action_buffer.append(action.copy())
delayed_action = self._action_buffer[0]  # FIFO
```

### Step 4: Ground friction randomization (Isaac Sim specific)

```python
from pxr import UsdPhysics, UsdShade

# In __init__ (one-time material creation):
material_path = "/World/GroundMaterial"
UsdShade.Material.Define(stage, material_path)
material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
# Assign to ground collider

# In reset():
material.GetStaticFrictionAttr().Set(self.dr.static_friction)
material.GetDynamicFrictionAttr().Set(self.dr.dynamic_friction)
```

### Step 5: CLI flags

```
--domain-rand          Enable domain randomization (default: off)
--dr-lidar-noise       LiDAR noise sigma range [0.005, 0.03] (default)
--dr-motor-asym        Motor asymmetry range [0.90, 1.10] (default)
--dr-action-delay      Action delay range [0, 3] steps (default)
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/jetbot_rl_env.py` | Add `DomainRandomizer` class, integrate into env `reset()`/`step()`/`_build_observation()` |
| `src/train_sac.py` | Pass `--domain-rand` flag through to env construction |
| `src/train_rl.py` | Same |

## Best Practices

1. **Start minimal, widen gradually.** Train first without DR until convergence, then enable DR with narrow ranges, then widen. Alternatively, combine with curriculum learning (Plan 02) to progressively increase DR intensity.

2. **Log-uniform for actuator gains, uniform for physical properties.** Stiffness/damping span orders of magnitude; friction and mass are linear-scale.

3. **LiDAR noise > dynamics randomization** for navigation sim-to-real. Prioritize sensor noise modeling.

4. **Motor asymmetry is the #1 sim-to-real gap** for differential drive. Real left/right motors are never matched. This causes the robot to curve even under equal commands.

5. **If action delay > 1 step**, consider including the last 2–3 actions in the observation space (action history) to partially restore the Markov property.

6. **Add a control smoothness penalty** to the reward function (penalize `|a_t - a_{t-1}|`) when using action delay and noise. This discourages bang-bang behavior that amplifies actuator nonlinearities.

7. **Sim2Real2Sim calibration** (from Wheeled Lab): measure real-world parameters cheaply (spring scale for friction, stopwatch for motor response) and center randomization ranges around measured values.

## Pitfalls to Avoid

- **Do NOT use very large LiDAR noise** (>0.1 m sigma) — the policy will learn to ignore LiDAR entirely
- **Do NOT set friction below ~0.1** — PhysX becomes unstable (wheels slip with no traction)
- **Do NOT apply per-episode noise where per-step is needed** — LiDAR noise must be fresh each step
- **Always clip after noise** — LiDAR distances must stay in [0, max_range]
- **Adding DR increases task difficulty** — may need to increase total timesteps or learning rate

## References

- [Wheeled Lab: Sim2Real for Wheeled Robotics (2025)](https://arxiv.org/abs/2502.07380)
- [Sim-to-Real Transfer from Isaac Sim to Real ROS 2 Robots (2025)](https://arxiv.org/abs/2501.02902)
- [NeuronsGym: LiDAR Noise Modeling (2023)](https://arxiv.org/abs/2302.03385)
- [OpenAI Learning Dexterity — DR + LSTM](https://openai.com/index/learning-dexterity/)
- [Isaac Lab Domain Randomization Discussion #2813](https://github.com/isaac-sim/IsaacLab/discussions/2813)
- [IsaacGymEnvs DR Documentation](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/domain_randomization.md)
- [Gazebo Sensor Noise Model](https://classic.gazebosim.org/tutorials?tut=sensor_noise)
