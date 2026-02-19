# Plan 05: DreamerV3 World Model Integration

## Motivation

DreamerV3 (Hafner et al. — Nature 2025) is the most sample-efficient general-purpose RL algorithm available. A 2025 study applying DreamerV3 to TurtleBot3 LiDAR navigation achieved **100% success rate** across all test environments, where model-free SAC/DDPG/TD3 plateaued below 85%. DreamerV3 learns a world model from real transitions, then trains the policy entirely on **imagined** trajectories — amplifying data efficiency by orders of magnitude.

For the JetBot's 34D observation space (10D state + 24D LiDAR) with 2D continuous actions, DreamerV3 is a natural fit. The architecture uses MLPs for vector observations (no CNNs needed), and the symlog normalization already used in `ChunkCVAEFeatureExtractor` comes directly from DreamerV3.

**This is the largest potential improvement in sample efficiency but also the largest implementation effort.**

## Architecture for JetBot

### Model Size: `size12m` (~12M parameters)

The 200M default is for image-based Atari/Minecraft. For 34D vector observations:

| Component | Configuration |
|-----------|--------------|
| **RSSM GRU** (deterministic) | 2048 hidden units, 8 block-diagonal blocks |
| **Stochastic latent** | 32 categoricals x 16 classes = 512D |
| **MLP hidden** | 256 units per layer |
| **Encoder** | MLP: 34D -> 256 -> 256 -> latent |
| **Decoder** | MLP: latent -> 256 -> 256 -> 34D |
| **Reward head** | MLP: latent -> 256 -> twohot bins |
| **Continue head** | MLP: latent -> 256 -> Bernoulli |
| **Actor** | MLP: latent -> 256 -> 256 -> 2D (continuous) |
| **Critic** | MLP: latent -> 256 -> 256 -> twohot bins |

### LiDAR Handling

For 24 LiDAR rays, the standard MLP encoder with symlog preprocessing is sufficient. No MLP-VAE or CNN needed:

```
obs(34D) -> symlog -> MLP_encoder(34 -> 256 -> 256) -> posterior(h_t, e_t) -> z_t
```

The decoder reconstructs the full 34D observation from the latent state, also through symlog for the target.

## Training Loop

```
Initialize: replay buffer B, world model M, actor pi, critic V
Prefill B with 2500 random-policy steps (+ optionally seed with demo data)

For each environment step:
  1. ENVIRONMENT INTERACTION
     a_t ~ pi(h_t, z_t)                    # actor samples from latent state
     o_{t+1}, r_t, done = env.step(a_t)
     Store (o_t, a_t, r_t, done) in B

  2. WORLD MODEL TRAINING (train_ratio gradient steps)
     For i in 1..train_ratio:
       Sample batch of 16 sequences of length 64 from B
       Run RSSM forward on real sequences:
         e_t = encoder(o_t)                 # encode observations
         h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})  # deterministic transition
         z_t ~ posterior(h_t, e_t)          # stochastic state (uses real obs)
         z_hat_t ~ prior(h_t)              # stochastic state (model prediction)
       Losses:
         L_recon  = -log p(o_t | h_t, z_t)           # observation reconstruction
         L_reward = -log p(r_t | h_t, z_t)            # reward prediction
         L_cont   = -log p(c_t | h_t, z_t)            # continue prediction
         L_dyn    = max(1, KL(posterior || prior))      # dynamics (free nats=1)
         L_rep    = max(1, KL(prior || posterior)) * 0.5 # representation
       Update M with sum of losses

  3. IMAGINATION (within same gradient steps)
     Sample 16 starting states from the replay batch
     Imagine H=15 steps forward using only the prior (no real observations):
       For t in 1..H:
         a_t ~ pi(h_t, z_t)
         h_{t+1} = GRU(h_t, z_t, a_t)
         z_{t+1} ~ prior(h_{t+1})
         r_hat_t = reward_head(h_{t+1}, z_{t+1})
         c_hat_t = continue_head(h_{t+1}, z_{t+1})

  4. POLICY LEARNING (on imagined trajectories)
     Compute lambda-returns from imagined rewards and critic values:
       V_lambda_t = r_t + gamma * ((1-lambda) * V(s_{t+1}) + lambda * V_lambda_{t+1})
     Update actor to maximize: E[V_lambda_t]
     Update critic to predict: V_lambda_t (using symexp twohot distribution)
```

### Key Ratio: `train_ratio`

| Setting | Gradient Steps/Env Step | Sample Efficiency | Wall-Clock |
|---------|------------------------|-------------------|------------|
| train_ratio=32 | 32 | High | Moderate |
| train_ratio=512 | 512 | Very High | Slow |

**Recommendation:** Start with `train_ratio=32` for the JetBot. The env step is fast (~25ms), so extreme train ratios waste wall-clock time relative to the cheap data collection.

## Integrating Demo Data

### Approach A: Seed the Replay Buffer (Recommended)

Load demo episodes into DreamerV3's replay buffer before online training. The world model immediately learns environment dynamics from expert data:

```python
# Load demos as episode sequences
demos = load_demo_data("demos/recording.npz")
for episode in demos:
    # episode = dict(obs=[T, 34], action=[T, 2], reward=[T], done=[T])
    replay_buffer.add_episode(episode)

# Then run DreamerV3 normally
dreamer.learn(total_steps=500_000)
```

### Approach B: World Model Pretraining (Optional)

Pretrain only the world model on demo data for ~50K gradient steps before starting online training. This gives the imagination phase a head start:

```python
# Phase 1: Pretrain world model on demos
for step in range(50_000):
    batch = replay_buffer.sample()  # only demo data
    world_model.update(batch)       # encoder, RSSM, decoder, reward, continue

# Phase 2: Online training with pretrained world model
dreamer.learn(total_steps=500_000)  # actor/critic + continued world model updates
```

### Demo Format Compatibility

DreamerV3 needs **episode sequences**, not individual transitions. Your `load_demo_data()` already returns episode-structured data. The conversion is straightforward:

```python
# Your format: dict with 'observations', 'actions', 'rewards', 'dones'
# DreamerV3 format: list of episodes, each a dict of arrays
episodes = []
start_idx = 0
for i, done in enumerate(demo_data['dones']):
    if done:
        episodes.append({
            'obs': demo_data['observations'][start_idx:i+1],
            'action': demo_data['actions'][start_idx:i+1],
            'reward': demo_data['rewards'][start_idx:i+1],
            'done': demo_data['dones'][start_idx:i+1],
        })
        start_idx = i + 1
```

## Implementation Options

### Option 1: NM512/dreamerv3-torch (Recommended)

The most practical PyTorch implementation for integration with your existing codebase.

**Repository:** [github.com/NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)

Integration steps:
1. Install as a submodule or vendored dependency
2. Write a thin Gymnasium-to-DreamerV3 env wrapper
3. Configure for 34D vector obs (no image encoder needed)
4. Seed replay buffer with demo episodes

```python
# Env wrapper for DreamerV3's expected interface
class DreamerEnvWrapper:
    def __init__(self, gym_env):
        self.env = gym_env
        self.obs_space = {'vector': gym_env.observation_space}
        self.act_space = gym_env.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return {'vector': obs}, reward, done, info

    def reset(self):
        obs, info = self.env.reset()
        return {'vector': obs}
```

### Option 2: SheepRL

Production-grade PyTorch + Lightning Fabric implementation. More batteries-included but heavier dependency:

```bash
pip install sheeprl
sheeprl dreamerv3 --env your_env --config custom_config.yaml
```

### Option 3: Custom Minimal Implementation

For maximum control, implement a stripped-down DreamerV3 with only the components needed for vector observations. This avoids the CNN encoder/decoder code paths entirely. Estimated: ~800 lines of new code for RSSM + imagination + actor-critic.

## Expected Performance

### Sample Efficiency

| Method | Env Steps to Converge | Wall-Clock (RTX 3090 Ti) |
|--------|----------------------|--------------------------|
| SAC/TQC (UTD=5) | 200–500K | 6–14 hours |
| SAC/TQC (UTD=20) | 100–200K | 10–20 hours |
| **DreamerV3 (tr=32)** | **50–200K** | **7–28 hours (est.)** |
| DreamerV3 (tr=512) | 30–100K | 20–60 hours (est.) |

DreamerV3 uses **2–10x fewer env steps** but each step includes more computation (world model + imagination + policy update). Net wall-clock may be similar to SAC/TQC for your fast simulator.

### Memory

| Model | VRAM |
|-------|------|
| TQC (5 critics, UTD=20) | ~4–6 GB |
| DreamerV3 (size12m) | ~2–4 GB |
| DreamerV3 (size1m) | ~1–2 GB |

### Where DreamerV3 Shines

- **Sparse rewards:** Imagination allows the agent to "practice" in its head, discovering reward-yielding trajectories without costly real interaction
- **Complex dynamics:** The world model captures dynamics once; the policy exploits them via imagination
- **Limited env budget:** If you can only afford 50K env steps (e.g., real robot), DreamerV3 is the clear winner

### Where SAC/TQC May Win

- **Simple dynamics + fast simulator:** When env steps are cheap, model-free methods avoid world model overhead
- **High UTD already works:** If SAC/TQC at UTD=5 converges fast enough, the added complexity of DreamerV3 may not be justified
- **Action chunking + RLPD:** Your current pipeline already has strong sample efficiency; DreamerV3 replaces rather than complements this

## DreamerV3 vs TD-MPC2

| Factor | DreamerV3 | TD-MPC2 |
|--------|-----------|---------|
| World model | Explicit (encoder + RSSM + decoder) | Implicit (no decoder) |
| Planning | Imagined rollouts + learned actor | MPPI in latent space (iterative) |
| Inference speed | **Fast** (single actor forward pass) | Slow (6 iters x 512 samples) |
| Exploration | Good (imagination + entropy) | Good (MPPI sampling) |
| Demo integration | Seed replay buffer | Seed replay buffer |
| Deployment | Actor network only | Requires MPPI planner |
| Best for | Exploration-heavy, deployment | Fine-grained continuous control |

**For JetBot navigation:** DreamerV3 is preferred because (a) fast inference for real-time deployment, (b) better exploration for navigation with sparse goals, and (c) the TurtleBot-DreamerV3 paper provides a direct reference.

## CLI Flags

```
# New training script: train_dreamer.py
./run.sh train_dreamer.py --headless --timesteps 200000

# With demo seeding
./run.sh train_dreamer.py --demos demos/recording.npz --headless

# Custom model size
./run.sh train_dreamer.py --headless --model-size size1m --train-ratio 32

# World model pretraining
./run.sh train_dreamer.py --demos demos/recording.npz --pretrain-wm 50000 --headless
```

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/train_dreamer.py` | **New file** — DreamerV3 training script with Isaac Sim integration |
| `src/jetbot_rl_env.py` | Add DreamerV3 env wrapper (thin adapter) |
| `src/eval_policy.py` | Add DreamerV3 model loading and evaluation |
| `src/demo_utils.py` | Add episode sequence extraction for DreamerV3 replay buffer format |

## Implementation Complexity

This is the most complex plan of the five. Estimated effort:

| Component | Effort | Notes |
|-----------|--------|-------|
| Env wrapper | Low | Thin adapter around `JetbotNavigationEnv` |
| DreamerV3 integration (NM512) | Medium | Configure for vector obs, wire up replay buffer |
| Demo seeding | Low | Convert NPZ episodes to DreamerV3 format |
| Evaluation | Low | Load actor weights, run in env |
| World model pretraining | Medium | Add pretraining phase before online loop |
| Custom RSSM tuning | Medium | May need to tune model size for 34D obs |
| **Total** | **Medium-High** | ~1-2 days of focused work |

## Risks

1. **Wall-clock regression:** DreamerV3 may not be faster in wall-clock than SAC/TQC with your fast Isaac Sim setup. The benefit is primarily sample efficiency.

2. **Replaces current pipeline:** DreamerV3 is a fundamentally different training paradigm. It does not compose with Chunk CVAE, Q-Chunking, or RLPD. Adopting DreamerV3 means maintaining two separate training pipelines.

3. **Tuning:** DreamerV3 is designed to be hyperparameter-free across domains, but the `train_ratio` and model size still need tuning for your specific env step cost.

4. **No action chunking:** DreamerV3's actor predicts single-step actions. To combine with action chunking, you would need to modify the imagination phase to roll out chunks, which is non-trivial.

## References

- [DreamerV3: Mastering Diverse Domains (Nature 2025)](https://www.nature.com/articles/s41586-025-08744-2)
- [DreamerV3 Paper (arXiv)](https://arxiv.org/abs/2301.04104)
- [DreamerV3 Official Repository (JAX)](https://github.com/danijar/dreamerv3)
- [NM512/dreamerv3-torch (PyTorch)](https://github.com/NM512/dreamerv3-torch)
- [World Models for LiDAR Navigation (Steinmetz et al., 2025)](https://arxiv.org/abs/2512.03429)
- [DreamerNav: Navigation with World Models (2025)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full)
- [TD-MPC2 (Hansen et al., ICLR 2024)](https://arxiv.org/abs/2310.16828)
- [SheepRL DreamerV3](https://github.com/Eclectic-Sheep/sheeprl)
- [DreamerV3 Project Page](https://danijar.com/project/dreamerv3/)
