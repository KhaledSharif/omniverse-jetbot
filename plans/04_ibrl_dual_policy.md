# Plan 04: IBRL — Imitation Bootstrapped Reinforcement Learning

## Motivation

The current pipeline pretrains a Chunk CVAE actor on demonstrations, then discards the CVAE encoder and fine-tunes the decoder with SAC/TQC. The pretrained knowledge is at risk of being "washed out" by randomly-initialized critics that produce noisy gradients early in RL training.

IBRL (Hu, Mirchandani, Sadigh — RSS 2024) solves this by keeping a **frozen copy** of the pretrained actor as an immutable anchor. Both the frozen IL policy and the RL policy propose actions at every step; the one with higher Q-value wins. This provides a safety net during early RL and accelerates critic learning via better TD targets.

**Result:** 6.4x higher success rate than RLPD on PickPlaceCan with 10 demos and 100K interactions. On real-world tasks, IBRL achieves 85% success where RLPD gets 15%.

**Paper:** [arXiv:2311.02198](https://arxiv.org/abs/2311.02198) | [Code](https://github.com/hengyuan-hu/ibrl)

## Algorithm

At every environment step:

```
1. ACTOR PROPOSAL (exploration):
   a_IL  = frozen_cvae_decoder(obs, z=0)           # frozen IL policy
   a_RL  = rl_actor(obs) + exploration_noise        # or SAC stochastic sample
   Q_IL  = min over K critics of Q(obs, a_IL)
   Q_RL  = min over K critics of Q(obs, a_RL)
   Execute argmax(Q_IL, Q_RL) in environment

2. BOOTSTRAP PROPOSAL (TD targets):
   a_IL' = frozen_cvae_decoder(next_obs, z=0)
   a_RL' = rl_actor_target(next_obs)
   Q_IL' = min over K target critics of Q'(next_obs, a_IL')
   Q_RL' = min over K target critics of Q'(next_obs, a_RL')
   target_Q = r + gamma * max(Q_IL', Q_RL' - alpha * log_prob_RL')

3. CRITIC UPDATE:
   Standard SAC/TQC critic loss with target_Q from step 2

4. ACTOR UPDATE:
   Standard SAC policy gradient: loss = -Q(obs, pi(obs)) + alpha * H(pi)
   Only the RL actor is updated; IL policy stays frozen
```

### Why Both Sites Matter (Ablation Results)

| Variant | Success Rate |
|---------|-------------|
| Full IBRL (both proposals) | ~95% |
| Only Actor Proposal (no bootstrap) | ~50% |
| Only Bootstrap Proposal (no actor) | ~70% |
| Neither (standard RLPD) | fails |

The **bootstrap proposal is more important** — using the IL policy to compute better TD targets accelerates critic learning significantly.

## Adaptation to Our CVAE Setup

### Current Architecture

```
Pretrain:
  CVAE Encoder: (obs_features, action_chunk) -> z
  CVAE Decoder: (obs_features || z) -> latent_pi -> mu -> tanh -> (k*2)

After pretraining:
  Encoder discarded, z fixed to 0
  RL actor = decoder path, fine-tuned by SAC
```

### IBRL Adaptation

```
After CVAE pretraining:
  1. FREEZE a complete copy of:
     - Feature extractor (state MLP + LiDAR MLP + z_pad)
     - latent_pi layer
     - mu layer
     This frozen copy = IL policy (mu_psi)

  2. INITIALIZE RL actor from same pretrained weights
     This copy = RL policy (pi_theta), updated by SAC/TQC

  3. Both use z=0 at inference (CVAE encoder already discarded)
```

**Important:** The frozen IL policy must have its **own frozen copy** of the feature extractor. If it shares the feature extractor with the RL actor, then RL fine-tuning of features would change the IL policy's effective behavior, breaking the "immutable anchor" property.

### Memory Cost

Doubling the actor is cheap. The CVAE decoder is small (~50K parameters) compared to the TQC critic ensemble (~500K+ parameters). The frozen copy adds negligible memory.

## Implementation Plan

### Step 1: Create frozen IL policy after CVAE pretraining

In `train_sac.py`, after `pretrain_chunk_cvae()`:

```python
import copy

# Freeze a complete copy of the pretrained actor path
il_policy = copy.deepcopy(model.actor)
for param in il_policy.parameters():
    param.requires_grad = False
il_policy.eval()

# Also freeze a copy of the feature extractor for IL
il_features = copy.deepcopy(model.policy.features_extractor)
for param in il_features.parameters():
    param.requires_grad = False
il_features.eval()
```

### Step 2: Modify action selection

Create a wrapper around the TQC model's `predict()`:

```python
class IBRLActionSelector:
    def __init__(self, rl_model, il_actor, il_features, exploration_noise=0.1):
        self.rl_model = rl_model
        self.il_actor = il_actor
        self.il_features = il_features
        self.noise_std = exploration_noise

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            # IL action (frozen)
            il_features = self.il_features(obs_tensor)
            a_il = self.il_actor(il_features)

            # RL action
            a_rl = self.rl_model.actor(obs_tensor)
            if not deterministic:
                a_rl = a_rl + torch.randn_like(a_rl) * self.noise_std

            # Q-value comparison (min over K=2 sampled critics)
            q_il = self.rl_model.critic(obs_tensor, a_il).min()
            q_rl = self.rl_model.critic(obs_tensor, a_rl).min()

            return a_il if q_il > q_rl else a_rl
```

### Step 3: Modify TD target computation

This requires overriding TQC's `train()` method or using a custom training loop:

```python
def compute_ibrl_target(self, next_obs, rewards, dones):
    with torch.no_grad():
        # IL next action
        il_features = self.il_features(next_obs)
        a_il_next = self.il_actor(il_features)

        # RL next action (from target actor)
        a_rl_next = self.rl_model.actor_target(next_obs)

        # Q-values for both
        q_il = self.rl_model.critic_target(next_obs, a_il_next).min()
        q_rl = self.rl_model.critic_target(next_obs, a_rl_next).min()

        # SAC entropy only for RL action
        log_prob = self.rl_model.actor.get_log_prob(next_obs, a_rl_next)
        q_rl_adjusted = q_rl - self.alpha * log_prob

        # Take the better one
        next_q = torch.max(q_il, q_rl_adjusted)
        target_q = rewards + (1 - dones) * self.gamma * next_q

    return target_q
```

### Step 4: Actor dropout (IBRL recommendation)

IBRL uses dropout=0.5 in the RL actor during training AND action selection. This is independently valuable:

```python
# Add dropout to the RL actor MLP layers
# In the actor network definition, add nn.Dropout(0.5) after each hidden layer
```

### Step 5: Custom training loop

SB3's built-in `model.learn()` does not support dual-policy action selection or modified TD targets. Options:

**Option A: Custom callback + monkey-patch (minimal changes)**
- Override `model.train()` in a callback to inject IBRL TD targets
- Override `model.predict()` to use IBRL action selection
- Fragile but quick to prototype

**Option B: Subclass TQC (cleaner)**
- Create `IBRLTQCPolicy` that overrides `_get_td_target()` and `forward()`
- Registers the frozen IL actor as a non-trainable submodule
- More work but robust

**Option C: Standalone training loop (recommended)**
- Write a custom `ibrl_train()` function that:
  1. Collects data using `IBRLActionSelector`
  2. Samples from RLPD replay buffers (demo + online)
  3. Computes IBRL TD targets
  4. Updates critics, then actor
- Most flexible, easiest to debug

### Step 6: CLI flags

```
--ibrl                 Enable IBRL dual-policy (default: off)
--ibrl-noise           RL exploration noise std (default: 0.1)
--ibrl-dropout         Actor dropout rate (default: 0.5)
--ibrl-soft            Use soft (Boltzmann) selection instead of greedy (default: off)
--ibrl-beta            Boltzmann temperature for soft IBRL (default: 10.0)
```

## Interaction with RLPD Demo Ratio

IBRL's original paper pre-fills the replay buffer with demos but does **not** explicitly oversample them (unlike RLPD's 50% demo ratio). With IBRL, demo oversampling becomes less critical because:

1. The frozen IL policy already provides good actions — no need for demos to keep the policy anchored
2. The bootstrap proposal ensures TD targets are good even when online data is limited

**Recommendation:** When using IBRL, reduce `--demo-ratio` from 0.5 to 0.1–0.25 or even 0. The IL policy replaces the function that demo oversampling serves.

## Interaction with Chunk CVAE

The IBRL dual-policy mechanism works at the **chunk level**:
- IL proposes a `(k*2,)` action chunk
- RL proposes a `(k*2,)` action chunk
- Q-value comparison selects the better chunk
- The entire selected chunk is executed in `ChunkedEnvWrapper`

This is correct — you want to select the better **trajectory segment**, not mix individual actions from different policies.

## Soft IBRL Variant

The greedy argmax has a theoretical failure mode: if Q(s, a_optimal) is initially lower than Q(s, a_IL), the optimal action is never selected and never updated. Soft IBRL uses Boltzmann sampling:

```python
# Instead of argmax:
q_values = torch.stack([q_rl, q_il], dim=-1)
probs = F.softmax(q_values * beta, dim=-1)  # beta=10
idx = torch.multinomial(probs, 1)
action = a_rl * (1 - idx) + a_il * idx
```

Use soft IBRL (`beta=10`) if you observe the RL policy never improving beyond the IL policy.

## Expected Impact

| Metric | Without IBRL | With IBRL |
|--------|-------------|-----------|
| Early training stability | Policy degrades as critics are random | IL anchor prevents regression |
| Exploration quality | Random noise on bad RL policy | IL provides guided exploration |
| TD target quality | Bootstraps from poor RL policy | Bootstraps from better of IL/RL |
| Sample efficiency | RLPD baseline | 2-6x improvement (paper claims) |
| Final performance | Limited by critic initialization | IL provides floor, RL raises ceiling |

## Graceful Degradation

If the IL policy is poor (too few demos, distribution mismatch), IBRL degrades gracefully to standard RL — the argmax simply always picks the RL action once Q-values improve. There is no penalty for having IBRL enabled with a weak IL policy.

## Files to Modify

| File | Changes |
|------|---------|
| `src/train_sac.py` | Add `IBRLActionSelector`, modify TD target computation, freeze IL copy after CVAE pretraining |
| `src/eval_policy.py` | At eval time, use only the RL actor (or optionally keep IBRL selection) |

## Known Limitations

1. **Two forward passes per step** — both IL and RL must produce actions, and both must be evaluated by the critic. Roughly 2x inference cost per env step. For a 25ms env step with 330ms gradient compute (UTD=20), this is negligible.

2. **Q-value overestimation risk** — if the RL policy diverges from the replay buffer distribution, critic networks may assign unreliable Q-values to OOD actions. SAC's entropy regularization mitigates this better than TD3's deterministic policy (confirmed by DRLR follow-up paper).

3. **IL policy must be reasonable** in states the RL agent visits. If the IL policy was trained on demonstrations from a very different state distribution, its proposals may be unhelpful.

## References

- [IBRL: Imitation Bootstrapped RL (RSS 2024)](https://arxiv.org/abs/2311.02198)
- [IBRL Project Page](https://ibrl.hengyuanhu.com/)
- [IBRL GitHub Repository](https://github.com/hengyuan-hu/ibrl)
- [DRLR: Improvements on IBRL (2025)](https://arxiv.org/abs/2509.04069)
- [SERL: Sample-Efficient Robotic RL (ICRA 2024)](https://arxiv.org/abs/2401.16013)
- [RLPD: Efficient Online RL with Offline Data (ICML 2023)](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)
