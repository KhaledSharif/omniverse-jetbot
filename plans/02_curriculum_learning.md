# Plan 02: Curriculum Learning

## Motivation

The current environment resets with a fixed number of obstacles (default 5, up to 50) at uniform random positions. Training from scratch with high obstacle density is difficult — the agent must simultaneously learn basic goal-seeking, obstacle avoidance, and long-horizon planning. Curriculum learning starts with easy tasks and progressively increases difficulty, matching proven approaches in DreamerNav (2025), CTSAC (2025), and A*-curriculum papers.

Research shows curriculum learning can improve success rate from 61.5% to 83.6% on navigation tasks by adding intermediate checkpoint rewards and progressive difficulty (HMP-DRL, 2025).

## Curriculum Dimensions

### Primary Axis: Obstacle Count

The dominant difficulty factor in a 4m x 4m workspace. More obstacles = narrower passages = harder planning.

### Secondary Axes (optional)

| Dimension | Range | Effect |
|-----------|-------|--------|
| Goal distance | 0.3–full workspace | Longer paths require more planning |
| Obstacle radii | [0.05, 0.15] m | Larger obstacles create tighter gaps |
| Episode length | 100–500 steps | Shorter horizons force efficient paths |

## Recommended Schedule: 5-Stage Performance-Gated

| Stage | Obstacles | Obstacle Radii | Max Steps | Advance When |
|-------|-----------|---------------|-----------|-------------|
| 1 | 0 | — | 200 | Success >= 90% over 100 episodes |
| 2 | 5 | [0.08, 0.12] | 300 | Success >= 85% over 100 episodes |
| 3 | 15 | [0.08, 0.15] | 400 | Success >= 80% over 100 episodes |
| 4 | 30 | [0.08, 0.15] | 500 | Success >= 75% over 100 episodes |
| 5 | 50 | [0.08, 0.15] | 500 | Terminal (full difficulty) |

**Minimum steps per stage:** 20,000 (prevents premature advancement on lucky streaks).

**Alternative — Continuous Interpolation:**
```
num_obstacles(step) = min(50, int(50 * step / ramp_steps))
```
where `ramp_steps` = 300,000. Avoids sharp transitions.

## Implementation Plan

### Step 1: Add `CurriculumManager` class

```python
class CurriculumManager:
    """Manages progressive difficulty for navigation training."""

    STAGES = [
        {"obstacles": 0,  "max_steps": 200, "radius_range": (0.08, 0.08), "advance_threshold": 0.90},
        {"obstacles": 5,  "max_steps": 300, "radius_range": (0.08, 0.12), "advance_threshold": 0.85},
        {"obstacles": 15, "max_steps": 400, "radius_range": (0.08, 0.15), "advance_threshold": 0.80},
        {"obstacles": 30, "max_steps": 500, "radius_range": (0.08, 0.15), "advance_threshold": 0.75},
        {"obstacles": 50, "max_steps": 500, "radius_range": (0.08, 0.15), "advance_threshold": None},
    ]

    def __init__(self, window_size=100, min_steps_per_stage=20000):
        self.current_stage = 0
        self.window_size = window_size
        self.min_steps_per_stage = min_steps_per_stage
        self.episode_results = collections.deque(maxlen=window_size)
        self.steps_in_stage = 0

    def record_episode(self, success: bool, steps: int):
        self.episode_results.append(success)
        self.steps_in_stage += steps

    def should_advance(self) -> bool:
        stage = self.STAGES[self.current_stage]
        if stage["advance_threshold"] is None:
            return False  # terminal stage
        if self.steps_in_stage < self.min_steps_per_stage:
            return False
        if len(self.episode_results) < self.window_size:
            return False
        success_rate = sum(self.episode_results) / len(self.episode_results)
        return success_rate >= stage["advance_threshold"]

    def advance(self):
        if self.current_stage < len(self.STAGES) - 1:
            self.current_stage += 1
            self.episode_results.clear()
            self.steps_in_stage = 0

    @property
    def config(self):
        return self.STAGES[self.current_stage]
```

### Step 2: Integrate into `JetbotNavigationEnv.reset()`

```python
def reset(self, **kwargs):
    if self.curriculum is not None:
        cfg = self.curriculum.config
        self._num_obstacles = cfg["obstacles"]
        self._max_episode_steps = cfg["max_steps"]
        self._obstacle_radius_range = cfg["radius_range"]
    # ... existing reset logic (spawn obstacles, place goal, A* check) ...
```

### Step 3: Integrate into training callback

```python
class CurriculumCallback(BaseCallback):
    """SB3 callback that advances curriculum stages."""

    def __init__(self, curriculum_manager, env, verbose=1):
        super().__init__(verbose)
        self.cm = curriculum_manager
        self.env = env

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info:
                self.cm.record_episode(info["is_success"], info.get("episode_steps", 1))
                if self.cm.should_advance():
                    self.cm.advance()
                    if self.verbose:
                        print(f"[Curriculum] Advanced to stage {self.cm.current_stage}: "
                              f"{self.cm.config}")
        return True
```

### Step 4: CLI flags for `train_sac.py`

```
--curriculum           Enable curriculum learning (default: off)
--curriculum-stages    Number of stages (default: 5)
--curriculum-window    Rolling window size for success rate (default: 100)
--curriculum-min-steps Minimum steps per stage (default: 20000)
```

### Step 5: TensorBoard logging

Log per-stage metrics:
- `curriculum/stage` — current stage index
- `curriculum/success_rate` — rolling success rate
- `curriculum/steps_in_stage` — steps spent in current stage
- `curriculum/obstacles` — current obstacle count

## Interaction with Replay Buffer (Off-Policy RL)

This is the critical design consideration for SAC/TQC:

### Recommended approach: Keep all transitions

- **Do NOT flush** the replay buffer on stage transitions — SAC handles off-policy data well
- Transitions from Stage 1 (0 obstacles) still teach goal-seeking behavior
- The RLPD demo transitions remain at full difficulty (50 obstacles) — they act as an anchor for the target distribution

### Optional: Recency weighting

Sample recent transitions with 2x probability vs old transitions. This can be achieved by modifying `ReplayBuffer.sample()` to use a linear priority based on insertion order:

```python
# Priority proportional to recency
priorities = np.linspace(0.5, 1.0, len(buffer))
probs = priorities / priorities.sum()
indices = np.random.choice(len(buffer), size=batch_size, p=probs)
```

### Demo ratio interaction

- Demo transitions were collected at full difficulty — they remain relevant across all stages
- As the curriculum advances and online data catches up to demo difficulty, the demo ratio becomes less critical
- Optionally anneal `--demo-ratio` from 0.75 (early, when online data is easy-stage) to 0.25 (late, when online data matches demo difficulty)

## Combining with Domain Randomization (Plan 01)

Two strategies:

1. **Parallel ramp:** Enable DR from the start with narrow ranges, widen DR ranges as curriculum stages advance. Stage 1 gets minimal DR; Stage 5 gets full DR.

2. **Sequential:** Train through all curriculum stages without DR first, then restart with DR enabled at full difficulty. Simpler but wastes the curriculum benefit.

Strategy 1 is recommended — it produces a policy that is simultaneously robust to difficulty variation and domain variation.

## Alternative: Prioritized Level Replay (PLR)

If the manual schedule proves fragile, upgrade to automatic curriculum via PLR:

- Maintain a buffer of ~200 environment configurations (obstacle layouts)
- Score each by the agent's TD error when last visited
- Sample proportional to `score^(1/temperature)` with `temperature=0.1`
- Use `staleness_coef=0.1` to ensure periodic revisitation

The [Syllabus library](https://github.com/RyanNavillus/Syllabus) provides a portable PLR implementation tested with SB3.

**Caveat:** PLR was designed for on-policy PPO. Adapting to off-policy SAC requires care — the scoring metric should be TD error (not GAE), and stale scores must be refreshed more frequently.

## Files to Modify

| File | Changes |
|------|---------|
| `src/jetbot_rl_env.py` | Add `CurriculumManager`, integrate into `reset()` to vary obstacles/steps/radii |
| `src/train_sac.py` | Add `CurriculumCallback`, CLI flags, TensorBoard logging |
| `src/train_rl.py` | Same (for PPO pipeline) |
| `src/demo_utils.py` | Add `CurriculumCallback` if shared across pipelines |

## Expected Impact

- **Faster convergence:** Agent learns basic navigation in Stage 1 (0 obstacles) within minutes, then transfers skills to progressively harder stages
- **Higher final success rate:** Literature shows +20 percentage points from curriculum vs uniform difficulty
- **More stable training:** Avoids the common failure mode where the agent learns to sit still because any movement in a dense obstacle field leads to collision

## References

- [CTSAC: Curriculum Transformer SAC for Robot Exploration (2025)](https://arxiv.org/abs/2503.14254)
- [A* Curriculum for Robot Navigation (2021)](https://arxiv.org/abs/2101.01774)
- [DreamerNav: Curriculum Training with World Models (2025)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full)
- [HMP-DRL: Hybrid Motion Planning with Deep RL (2025)](https://arxiv.org/abs/2512.24651)
- [CurricuLLM: LLM-Designed Curricula (ICRA 2025)](https://arxiv.org/abs/2409.18382)
- [PLR: Prioritized Level Replay (2020)](https://arxiv.org/abs/2010.03934)
- [Syllabus: Portable Curricula Library](https://github.com/RyanNavillus/Syllabus)
- [HiER: Highlight Experience Replay + Easy2Hard](https://arxiv.org/abs/2312.09394)
- [Reward Curriculum for SAC (2024)](https://arxiv.org/abs/2410.16790)
- [CURATE: Automatic Curriculum via Exploration by Exploitation](https://openreview.net/forum?id=7wdCgG6K7i)
