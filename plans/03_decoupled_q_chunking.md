# Plan 03: Decoupled Q-Chunking (DQC)

## Motivation

The current `ChunkedEnvWrapper` ties the policy and critic to the same chunk size `k` (default 10). This means the critic evaluates 10-step action chunks and the policy outputs 10-step chunks. Decoupled Q-Chunking (Li, Park, Levine — ICLR 2026) shows that using a **longer chunk for the critic** (better multi-step value propagation) with a **shorter chunk for the policy** (more reactive, easier to learn) yields up to **+60 percentage points** on hard long-horizon tasks.

The key insight: a long critic chunk gives you unbiased h-step TD backups without importance sampling, while the policy only needs to predict a short chunk to remain tractable. A "distilled partial critic" bridges the gap by estimating "what is the best value achievable if I extend this short chunk optimally?"

**Paper:** [arXiv:2512.10926](https://arxiv.org/abs/2512.10926) | [Code](https://github.com/ColinQiyangLi/dqc)

## Architecture Overview

```
Current (Q-Chunking):
  Actor:  obs -> (k*2) action chunk     [k=10]
  Critic: Q(obs, k*2 action chunk)      [k=10]
  Env:    ChunkedEnvWrapper(chunk_size=10)

Proposed (DQC):
  Actor:  obs -> (h_a*2) action chunk   [h_a=5, short]
  Full Critic:     Q_phi(obs, h*2)      [h=25, long]
  Partial Critic:  Q^P_psi(obs, h_a*2)  [h_a=5, short]
  Env:    ChunkedEnvWrapper(chunk_size=h_a=5)  [policy executes short chunks]
  Data:   Replay buffer stores h-step chunks    [for full critic training]
```

## Key Equations

### Full Critic (h-step Bellman)

```
target = R_{0:h} + gamma^h * V(s_{t+h})
L_critic = MSE(Q_phi(s_t, a_{t:t+h}), target)

where R_{0:h} = sum_{i=0}^{h-1} gamma^i * r_i
```

### Distilled Partial Critic (Optimistic Regression)

The partial critic approximates `max_{a_{h_a:h}} Q_phi(s, [a_{0:h_a}, a_{h_a:h}])` — the best value achievable by extending a short chunk optimally:

```
target_q = Q_phi_bar(s_t, a_{0:h})        # full chunk Q (from target network)
pred_q   = Q^P_psi(s_t, a_{0:h_a})        # partial chunk Q

# Asymmetric (expectile) regression — penalizes underestimates more
weight = kappa_d   if target_q >= pred_q   # push UP toward optimistic completions
         (1-kappa_d) otherwise             # allow some overestimates
loss = weight * (target_q - pred_q)^2
```

With `kappa_d = 0.8`, the partial critic learns to approximate the **upper tail** of the full-chunk Q distribution conditioned on the short prefix.

### Policy Update

Standard SAC-style gradient ascent, but against the **partial** critic:

```
loss_actor = -Q^P_psi(s, pi(s)) + alpha * log_prob(pi(s))
```

The policy only needs to output `h_a`-dimensional chunks, scored by the partial critic.

## Recommended Chunk Sizes

From the paper's experiments:

| Config | Critic h | Policy h_a | Ratio | Performance |
|--------|----------|-----------|-------|-------------|
| Standard QC | 10 | 10 | 1:1 | Baseline |
| **DQC (recommended)** | **25** | **5** | **5:1** | **Best** |
| DQC (conservative) | 15 | 5 | 3:1 | Good |

For the JetBot (60 Hz physics, ~25 ms/step):
- `h_a = 5` = 83 ms of committed action = responsive enough for obstacle avoidance
- `h = 25` = 417 ms of value horizon = captures medium-term planning

## Implementation Plan

### Step 1: Modify replay buffer to store h-step segments

The current `ChunkedEnvWrapper` stores chunk-level transitions `(obs_t, a_{0:k}, R_chunk, obs_{t+k}, done)`. For DQC, we need to store **h-step** segments even though the policy only executes h_a steps at a time.

**Approach: Rolling trajectory buffer**

```python
class DQCTrajectoryBuffer:
    """Stores raw (obs, action, reward, done) transitions and reconstructs
    h-step segments on demand for the full critic."""

    def __init__(self, capacity, h, h_a, gamma):
        self.h = h       # full critic chunk size (25)
        self.h_a = h_a   # policy chunk size (5)
        self.gamma = gamma
        # Store raw single-step transitions
        self.obs = np.zeros((capacity, obs_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros(capacity)
        self.dones = np.zeros(capacity, dtype=bool)

    def sample_full_chunks(self, batch_size):
        """Sample h-step chunks for full critic training."""
        # Find valid starting indices (h consecutive non-terminal steps)
        # Return (obs_t, a_{t:t+h}, R_{0:h}, obs_{t+h}, done_{t+h})

    def sample_partial_chunks(self, batch_size):
        """Sample h_a-step chunks for partial critic / actor training."""
        # Return (obs_t, a_{t:t+h_a})
        # Also return full h-step chunk for distillation target
```

### Step 2: Add partial critic network

```python
class PartialCritic(nn.Module):
    """Q^P(obs_features, action_chunk_short) -> scalar.
    Input action dim = h_a * 2, not h * 2."""

    def __init__(self, obs_dim, action_dim_short, hidden=256, num_critics=5):
        # Same architecture as TQC critic but with h_a*2 action input
        ...
```

### Step 3: Distillation training step

```python
def update_partial_critic(self, batch):
    """Train Q^P via asymmetric regression against Q_phi."""
    obs, short_actions, full_actions = batch

    # Full critic evaluates the h-step chunk (target, detached)
    with torch.no_grad():
        target_q = self.full_critic_target(obs, full_actions).min(dim=0)

    # Partial critic evaluates only the h_a-step prefix
    pred_q = self.partial_critic(obs, short_actions)

    # Asymmetric (expectile) loss
    diff = target_q - pred_q
    weight = torch.where(diff >= 0, self.kappa_d, 1 - self.kappa_d)
    loss = (weight * diff.pow(2)).mean()

    self.partial_critic_optimizer.zero_grad()
    loss.backward()
    self.partial_critic_optimizer.step()
```

### Step 4: Modified training loop

Each training iteration now has **three** loss computations:

```
For each gradient step:
  1. Full Critic Update:
     - Sample h-step chunks from trajectory buffer
     - TD target = R_{0:h} + gamma^h * V(s_{t+h})   [or min-Q target]
     - Update Q_phi via MSE

  2. Partial Critic Distillation:
     - Same batch, extract h_a prefix of actions
     - target = Q_phi_bar(obs, full_chunk)
     - pred = Q^P_psi(obs, short_chunk)
     - Update Q^P via asymmetric regression (kappa_d=0.8)

  3. Actor Update:
     - Sample obs from buffer
     - a_short = pi(obs)         [h_a*2 dimensional]
     - loss = -Q^P_psi(obs, a_short) + alpha * log_prob
     - Update pi
```

### Step 5: ChunkedEnvWrapper modification

The wrapper still uses `chunk_size = h_a` (5) for execution. The policy predicts 5-step chunks and the env executes them. The difference is only in how the **replay data is stored and used for critic training**.

```python
# Env wrapper: chunk_size = h_a = 5 (unchanged from current behavior)
env = ChunkedEnvWrapper(inner_env, chunk_size=5, gamma=0.99)

# But the replay buffer tracks raw transitions to reconstruct h=25 segments
```

### Step 6: CLI flags

```
--dqc                  Enable Decoupled Q-Chunking (default: off)
--dqc-critic-horizon   Full critic chunk size h (default: 25)
--dqc-policy-chunk     Policy chunk size h_a (default: 5)
--dqc-kappa            Asymmetric regression weight (default: 0.8)
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `h` (critic horizon) | 25 | Full critic chunk size |
| `h_a` (policy chunk) | 5 | Policy output chunk size |
| `kappa_d` (distill optimism) | 0.8 | Higher = more optimistic partial critic |
| `kappa_b` (value optimism) | 0.9 | For value function bootstrap (if used) |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 256–4096 | Larger batches help on hard tasks |
| `num_critics` | 5 | TQC ensemble size (unchanged) |

## Challenges and Considerations

### 1. SB3 Compatibility

SB3's SAC/TQC assumes a single critic architecture with action_dim matching the policy. DQC requires a second critic with a different action input dimension. Options:

- **Custom SAC implementation** — fork SB3's SAC and add the partial critic + distillation step. Most clean but most work.
- **Callback-based** — train the full critic and partial critic in a custom callback that runs alongside the standard TQC update. Hacky but minimal changes to SB3.
- **Standalone training loop** — bypass SB3's `model.learn()` and write a custom training loop that orchestrates all three updates. Recommended for DQC's complexity.

### 2. Replay Buffer Mismatch

The current RLPD demo buffer stores chunk-level transitions at the policy's chunk size. For DQC, demo data must be reconstructable into h-step segments. This means either:
- Store raw single-step demo transitions and reconstruct chunks on-the-fly
- Pre-compute h-step chunks from demo episodes via sliding window (similar to existing `make_chunk_transitions()`)

### 3. CVAE Pretraining Adaptation

The CVAE pretrained actor currently outputs `k*2` actions. With DQC, it would output `h_a*2`. The CVAE encoder/decoder dimensions change accordingly. The full critic's h-step chunks can be constructed by concatenating `h/h_a` consecutive predicted short chunks during pretraining.

### 4. Effective Discount

With h_a=5 at 60 Hz: effective gamma = 0.99^5 = 0.951 per chunk step. With h=25: full critic gamma = 0.99^25 = 0.778 per full chunk. This aggressive discounting for the full critic is fine — it enables long-range value propagation while maintaining training stability.

## Expected Impact

- **Better value propagation:** 25-step critic captures planning horizons of ~417 ms, vs 167 ms for the current 10-step critic
- **More reactive policy:** 5-step chunks (83 ms) allow faster course corrections near obstacles
- **Potentially +10-30% success rate** on cluttered environments where the current 10-step open-loop policy commits too early to collision trajectories

## Files to Modify

| File | Changes |
|------|---------|
| `src/train_sac.py` | Add full critic, partial critic, distillation step, new training loop |
| `src/jetbot_rl_env.py` | Modify `ChunkedEnvWrapper` to optionally expose raw transitions |
| `src/demo_utils.py` | Add `make_dqc_transitions()` for h-step chunk extraction |
| `src/eval_policy.py` | Update auto-detection for DQC models (action_dim = h_a*2) |

## References

- [Decoupled Q-Chunking (Li, Park, Levine — ICLR 2026)](https://arxiv.org/abs/2512.10926)
- [DQC GitHub Repository](https://github.com/ColinQiyangLi/dqc)
- [Q-Chunking (Park et al. — NeurIPS 2025)](https://arxiv.org/abs/2507.07969)
- [SHARSA: Horizon Reduction (Park et al. — NeurIPS 2025)](https://arxiv.org/abs/2506.04168)
- [Implicit Q-Learning / Expectile Regression (Kostrikov et al.)](https://arxiv.org/abs/2110.06169)
