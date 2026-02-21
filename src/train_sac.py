#!/usr/bin/env python3
"""Train SAC/TQC agent with RLPD-style demo replay for Jetbot navigation task.

Uses demonstrations in a 50/50 replay buffer (RLPD) instead of the fragile
BC warmstart → VecNormalize pre-warming pipeline. LayerNorm in critics replaces
VecNormalize entirely.

Usage:
    ./run.sh train_sac.py --demos demos/recording.npz --headless
    ./run.sh train_sac.py --demos demos/recording.npz --headless --timesteps 500000
    ./run.sh train_sac.py --demos demos/recording.npz --headless --cpu --timesteps 1000
"""

import argparse
import numpy as np
from pathlib import Path

from demo_utils import validate_demo_data, load_demo_transitions, VerboseEpisodeCallback


def symlog(x):
    """DreamerV3 symmetric log compression: sign(x) * log(|x| + 1)."""
    import torch
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _get_tqc_quantile_critics(critic_module):
    """Find quantile critic networks from a TQC critic, supporting various sb3_contrib versions.

    Checks known attribute names: 'quantile_critics' (v2.3+), 'critics' (older).
    Returns the nn.ModuleList of quantile critic networks, or None for SAC.
    """
    import torch.nn as nn
    for attr in ('quantile_critics', 'critics'):
        val = getattr(critic_module, attr, None)
        if isinstance(val, nn.ModuleList):
            return val
    return None


class ChunkCVAEFeatureExtractor:
    """SB3 feature extractor for Chunk CVAE with split state/lidar MLPs.

    Splits 34D obs into state (0:10) and LiDAR (10:34), applies symlog,
    processes through separate MLPs, and concatenates with a zero-padded
    z-slot for the CVAE latent variable.

    Output: concat(state_features, lidar_features, z_pad) = 96 + z_dim
    """

    _cls = None

    @staticmethod
    def create(base_extractor_cls, z_dim=8):
        import torch
        import torch.nn as nn

        class _ChunkCVAEFeatureExtractor(base_extractor_cls):
            def __init__(self, observation_space, features_dim=104):
                super().__init__(observation_space, features_dim=features_dim)
                self._z_dim = z_dim
                self._obs_feature_dim = features_dim - z_dim

                self.state_mlp = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                )

                self.lidar_mlp = nn.Sequential(
                    nn.Linear(24, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                )

            def encode_obs(self, observations):
                """Encode observations into obs_features (without z padding).

                Returns:
                    obs_features tensor of shape (batch, 96)
                """
                state = symlog(observations[:, :10])
                lidar = symlog(observations[:, 10:34])
                state_features = self.state_mlp(state)
                lidar_features = self.lidar_mlp(lidar)
                return torch.cat([state_features, lidar_features], dim=-1)

            def forward(self, observations):
                obs_features = self.encode_obs(observations)
                z_pad = torch.zeros(
                    obs_features.shape[0], self._z_dim,
                    device=obs_features.device, dtype=obs_features.dtype,
                )
                return torch.cat([obs_features, z_pad], dim=-1)

        return _ChunkCVAEFeatureExtractor

    @classmethod
    def get_class(cls, base_extractor_cls, z_dim=8):
        # Always recreate to capture z_dim closure
        cls._cls = cls.create(base_extractor_cls, z_dim=z_dim)
        return cls._cls


def pretrain_chunk_cvae(model, demo_obs, demo_actions, episode_lengths,
                        chunk_size, z_dim=8, epochs=100, batch_size=256,
                        lr=1e-3, beta=0.1, gamma=0.99):
    """Pretrain actor via Chunk CVAE: encoder maps (obs, action_chunk) → z,
    decoder (= actor's latent_pi + mu) maps (obs_features || z) → action_chunk.

    After pretraining the encoder is discarded. The z-slot is zeroed during RL.

    Args:
        model: SB3 SAC/TQC model (actor must use ChunkCVAEFeatureExtractor)
        demo_obs: numpy (N, 34) step-level observations
        demo_actions: numpy (N, 2) step-level actions
        episode_lengths: numpy array of per-episode step counts
        chunk_size: action chunk size (k)
        z_dim: CVAE latent dimension
        epochs: pretraining epochs
        batch_size: mini-batch size
        lr: learning rate
        beta: KL weight
        gamma: discount factor (unused here, kept for API consistency)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from demo_utils import extract_action_chunks

    print("\n" + "=" * 60)
    print("Chunk CVAE Pretraining")
    print("=" * 60)

    # Extract action chunks from demo data
    chunk_obs, chunk_actions_flat = extract_action_chunks(
        demo_obs, demo_actions, episode_lengths, chunk_size)
    action_chunk_dim = chunk_size * demo_actions.shape[1]

    print(f"  Chunk size: {chunk_size}, z_dim: {z_dim}")
    print(f"  {len(chunk_obs)} chunks from {len(episode_lengths)} episodes")

    device = model.device
    dataset = TensorDataset(
        torch.tensor(chunk_obs, dtype=torch.float32),
        torch.tensor(chunk_actions_flat, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get obs feature dim from the feature extractor
    obs_feature_dim = model.actor.features_extractor._obs_feature_dim

    # Temporary CVAE encoder (discarded after pretraining)
    cvae_encoder = nn.Sequential(
        nn.Linear(obs_feature_dim + action_chunk_dim, 128),
        nn.SiLU(),
        nn.Linear(128, 64),
        nn.SiLU(),
    ).to(device)
    cvae_mu = nn.Linear(64, z_dim).to(device)
    cvae_logvar = nn.Linear(64, z_dim).to(device)

    # Parameters to optimize: feature_extractor + latent_pi + mu + cvae_encoder
    params = (
        list(model.actor.features_extractor.parameters())
        + list(model.actor.latent_pi.parameters())
        + list(model.actor.mu.parameters())
        + list(cvae_encoder.parameters())
        + list(cvae_mu.parameters())
        + list(cvae_logvar.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"  Training CVAE for {epochs} epochs...")

    for epoch in range(epochs):
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Encode observations → obs_features (96D)
            obs_features = model.actor.features_extractor.encode_obs(obs_batch)

            # CVAE encoder: (obs_features, action_chunk) → z
            enc_input = torch.cat([obs_features, act_batch], dim=-1)
            h = cvae_encoder(enc_input)
            mu_z = cvae_mu(h)
            logvar_z = cvae_logvar(h)

            # Reparameterize
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            z = mu_z + eps * std_z

            # Replace z-slot in features: concat(obs_features, z)
            features = torch.cat([obs_features, z], dim=-1)

            # Decode through actor's latent_pi → mu → tanh
            latent = model.actor.latent_pi(features)
            mean_actions = model.actor.mu(latent)
            pred_actions = torch.tanh(mean_actions)

            # Loss: L1 reconstruction + β·KL
            recon_loss = torch.nn.functional.l1_loss(pred_actions, act_batch)
            kl_loss = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}, "
                  f"L1: {total_recon / n_batches:.6f}, "
                  f"KL: {total_kl / n_batches:.6f}")

    # Copy pretrained features_extractor weights → critic and critic_target
    fe_state = model.actor.features_extractor.state_dict()
    model.critic.features_extractor.load_state_dict(fe_state)
    model.critic_target.features_extractor.load_state_dict(fe_state)
    print("  Feature extractor weights copied to critic/critic_target")

    # Tighten log_std to preserve CVAE-learned behavior
    model.actor.log_std.weight.data.zero_()
    model.actor.log_std.bias.data.fill_(-2.0)
    print("  Exploration noise tightened (log_std bias = -2.0, std ~ 0.135)")

    print("Chunk CVAE pretraining complete!")
    print("=" * 60 + "\n")


def make_demo_replay_buffer(buffer_cls, buffer_size, observation_space, action_space,
                            device, demo_obs, demo_actions, demo_rewards,
                            demo_next_obs, demo_dones, demo_ratio=0.5):
    """Create a replay buffer that mixes demos and online data at a given ratio.

    Returns a subclass instance of the given buffer_cls that overrides sample()
    to mix demo and online transitions.
    """
    import torch as th
    from stable_baselines3.common.buffers import ReplayBufferSamples

    class DemoReplayBuffer(buffer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Store demo data as tensors on device
            self.demo_obs = th.tensor(demo_obs, device=device)
            self.demo_actions = th.tensor(demo_actions, device=device)
            self.demo_rewards = th.tensor(demo_rewards, device=device).unsqueeze(1)
            self.demo_next_obs = th.tensor(demo_next_obs, device=device)
            self.demo_dones = th.tensor(demo_dones, device=device).unsqueeze(1)
            self.n_demos = len(demo_obs)

        def sample(self, batch_size, env=None):
            # When online buffer is empty, use 100% demo
            if self.size() == 0:
                demo_batch_size = batch_size
                online_batch_size = 0
            else:
                demo_batch_size = int(batch_size * demo_ratio)
                online_batch_size = batch_size - demo_batch_size

            # Sample demo indices
            demo_idx = np.random.randint(0, self.n_demos, size=demo_batch_size)
            demo_samples = ReplayBufferSamples(
                observations=self.demo_obs[demo_idx],
                actions=self.demo_actions[demo_idx],
                next_observations=self.demo_next_obs[demo_idx],
                dones=self.demo_dones[demo_idx],
                rewards=self.demo_rewards[demo_idx],
            )

            # Expose sampled indices for CostReplayBuffer
            self._last_demo_indices = demo_idx

            if online_batch_size == 0:
                self._last_online_indices = None
                return demo_samples

            # Sample from online buffer using explicit indices
            online_idx = np.random.randint(0, self.size(), size=online_batch_size)
            self._last_online_indices = online_idx
            online_samples = self._get_samples(
                th.tensor(online_idx, dtype=th.long), env=env
            )

            # Concatenate
            return ReplayBufferSamples(
                observations=th.cat([demo_samples.observations, online_samples.observations]),
                actions=th.cat([demo_samples.actions, online_samples.actions]),
                next_observations=th.cat([demo_samples.next_observations, online_samples.next_observations]),
                dones=th.cat([demo_samples.dones, online_samples.dones]),
                rewards=th.cat([demo_samples.rewards, online_samples.rewards]),
            )

    # Instantiate
    buf = DemoReplayBuffer(
        buffer_size,
        observation_space,
        action_space,
        device=device,
    )
    return buf


def inject_layernorm_into_critics(model):
    """Post-hoc inject LayerNorm + OFN into critic networks.

    Injects LayerNorm after each hidden Linear layer and Output Feature
    Normalization (OFN) before the final output Linear layer.

    Handles both TQC (quantile_critics) and SAC (critic.qf*) structures.
    After injection, re-syncs critic_target and recreates the critic optimizer.
    """
    import torch
    import torch.nn as nn

    class OutputFeatureNorm(nn.Module):
        """L2-normalize features: x / ||x||_2 (RLC 2024 OFN)."""
        def forward(self, x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def _inject_norms(sequential):
        """Insert LayerNorm after hidden Linears and OFN before output Linear."""
        new_modules = []
        modules = list(sequential)
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                remaining = modules[i + 1:]
                has_more_linear = any(isinstance(m, nn.Linear) for m in remaining)
                if has_more_linear:
                    # Hidden Linear: append layer, then LayerNorm
                    new_modules.append(module)
                    new_modules.append(nn.LayerNorm(module.out_features))
                else:
                    # Output Linear: insert OFN before it
                    new_modules.append(OutputFeatureNorm())
                    new_modules.append(module)
            else:
                new_modules.append(module)
        return nn.Sequential(*new_modules)

    qc = _get_tqc_quantile_critics(model.critic)

    if qc is not None:
        for i, critic_net in enumerate(qc):
            qc[i] = _inject_norms(critic_net)
        qc_target = _get_tqc_quantile_critics(model.critic_target)
        for i, critic_net in enumerate(qc_target):
            qc_target[i] = _inject_norms(critic_net)
    else:
        qf_attrs = sorted(a for a in dir(model.critic) if a.startswith('qf') and a[2:].isdigit())
        for attr in qf_attrs:
            setattr(model.critic, attr, _inject_norms(getattr(model.critic, attr)))
            setattr(model.critic_target, attr, _inject_norms(getattr(model.critic_target, attr)))

    # Move new parameters to device
    model.critic = model.critic.to(model.device)
    model.critic_target = model.critic_target.to(model.device)

    # Sync target from critic weights
    model.critic_target.load_state_dict(model.critic.state_dict())

    # Recreate critic optimizer to include LayerNorm + OFN parameters
    model.critic.optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=model.lr_schedule(1)
    )

    print("LayerNorm + OFN injected into critic networks")


def _inject_layernorm_cost_critic(model):
    """Inject LayerNorm + OFN into cost critic networks and re-sync target."""
    import torch
    import torch.nn as nn

    class OutputFeatureNorm(nn.Module):
        def forward(self, x):
            return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

    def _inject_norms(sequential):
        new_modules = []
        modules = list(sequential)
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                remaining = modules[i + 1:]
                has_more_linear = any(isinstance(m, nn.Linear) for m in remaining)
                if has_more_linear:
                    new_modules.append(module)
                    new_modules.append(nn.LayerNorm(module.out_features))
                else:
                    new_modules.append(OutputFeatureNorm())
                    new_modules.append(module)
            else:
                new_modules.append(module)
        return nn.Sequential(*new_modules)

    for i, qf in enumerate(model.cost_critic.q_networks):
        model.cost_critic.q_networks[i] = _inject_norms(qf)
    for i, qf in enumerate(model.cost_critic_target.q_networks):
        model.cost_critic_target.q_networks[i] = _inject_norms(qf)

    model.cost_critic = model.cost_critic.to(model.device)
    model.cost_critic_target = model.cost_critic_target.to(model.device)
    model.cost_critic_target.load_state_dict(model.cost_critic.state_dict())
    model.cost_critic_optimizer = torch.optim.Adam(
        model.cost_critic.parameters(), lr=model.lr_schedule(1)
    )
    print("LayerNorm + OFN injected into cost critic networks")


class CostReplayBuffer:
    """Parallel ring buffer storing per-transition costs alongside the main replay buffer.

    Mirrors the main replay buffer's position so that costs can be retrieved
    for the same transitions sampled by DemoReplayBuffer.
    """

    def __init__(self, buffer_size, device, demo_costs=None, demo_ratio=0.5):
        """Initialize the CostReplayBuffer.

        Args:
            buffer_size: Max capacity for online costs
            device: torch device
            demo_costs: numpy array of demo chunk costs (N_demo,)
            demo_ratio: fraction of batch from demos (for reference)
        """
        import torch as th
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False
        self.online_costs = np.zeros(buffer_size, dtype=np.float32)
        if demo_costs is not None:
            self.demo_costs = th.tensor(demo_costs, device=device, dtype=th.float32)
        else:
            self.demo_costs = None

    def add(self, cost):
        """Store a single transition cost at the current position."""
        self.online_costs[self.pos] = cost
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, demo_indices, online_indices):
        """Retrieve costs for the same indices sampled by DemoReplayBuffer.

        Args:
            demo_indices: numpy array of demo indices (or None)
            online_indices: numpy array of online buffer indices (or None)

        Returns:
            torch tensor of costs (batch, 1)
        """
        import torch as th
        parts = []
        if demo_indices is not None and self.demo_costs is not None:
            parts.append(self.demo_costs[demo_indices].unsqueeze(1))
        if online_indices is not None:
            online_c = th.tensor(
                self.online_costs[online_indices],
                device=self.device, dtype=th.float32
            ).unsqueeze(1)
            parts.append(online_c)
        return th.cat(parts, dim=0)


class MeanCostCritic:
    """Factory for creating a mean-value cost critic (twin-Q with MSE loss).

    Creates n_critics MLP networks: (features_dim + action_dim) -> [256, 256] -> 1.
    """

    @staticmethod
    def create(n_critics, features_dim, action_dim, fe_class, fe_kwargs, obs_space, device):
        """Build cost critic networks with their own feature extractor.

        Args:
            n_critics: Number of critic networks
            features_dim: Output dimension of feature extractor
            action_dim: Action dimension
            fe_class: Feature extractor class
            fe_kwargs: Feature extractor kwargs
            obs_space: Observation space
            device: torch device

        Returns:
            (cost_critic, cost_critic_target) as nn.Module instances
        """
        import torch
        import torch.nn as nn

        class _CostCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.features_extractor = fe_class(obs_space, **fe_kwargs)
                self.n_critics = n_critics
                self.q_networks = nn.ModuleList()
                for _ in range(n_critics):
                    self.q_networks.append(nn.Sequential(
                        nn.Linear(features_dim + action_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1),
                    ))

            def forward(self, obs, actions):
                features = self.features_extractor(obs)
                x = torch.cat([features, actions], dim=-1)
                return [qf(x) for qf in self.q_networks]

        critic = _CostCritic().to(device)
        critic_target = _CostCritic().to(device)
        critic_target.load_state_dict(critic.state_dict())
        critic_target.requires_grad_(False)
        return critic, critic_target


def _create_safe_tqc_class(tqc_base_cls):
    """Create SafeTQC class dynamically to handle TQC import availability.

    Args:
        tqc_base_cls: The TQC or SAC class to subclass

    Returns:
        SafeTQC class
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from stable_baselines3.common.utils import polyak_update

    class SafeTQC(tqc_base_cls):
        """TQC/SAC with learned Lagrange multiplier for constraint satisfaction.

        Adds a cost critic that estimates cumulative constraint violation, and a
        Lagrange multiplier that auto-balances reward maximization vs. safety.
        """

        def __init__(self, *args, cost_limit=25.0, lagrange_lr=3e-4,
                     lagrange_init=0.0, cost_n_critics=2,
                     cost_critic_type='mean', cost_buffer=None,
                     max_episode_steps=500, **kwargs):
            self._cost_limit = cost_limit
            self._lagrange_lr = lagrange_lr
            self._lagrange_init = lagrange_init
            self._cost_n_critics = cost_n_critics
            self._cost_critic_type = cost_critic_type
            self.cost_buffer = cost_buffer
            self._max_episode_steps = max_episode_steps
            super().__init__(*args, **kwargs)

        def _setup_model(self):
            super()._setup_model()

            # Lagrange multiplier (log-space for unconstrained optimization)
            self.log_lagrange = nn.Parameter(
                torch.tensor(self._lagrange_init, dtype=torch.float32,
                             device=self.device)
            )
            self.lagrange_optimizer = torch.optim.Adam(
                [self.log_lagrange], lr=self._lagrange_lr
            )

            # Cost critic
            features_dim = self.actor.features_extractor._features_dim
            action_dim = self.action_space.shape[0]
            fe_class = type(self.actor.features_extractor)
            fe_kwargs = dict(features_dim=features_dim)
            obs_space = self.observation_space

            self.cost_critic, self.cost_critic_target = MeanCostCritic.create(
                self._cost_n_critics, features_dim, action_dim,
                fe_class, fe_kwargs, obs_space, self.device
            )

            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic.parameters(), lr=self.lr_schedule(1)
            )

            # Per-step cost limit for Lagrange update
            self._cost_limit_per_step = self._cost_limit / self._max_episode_steps

        def train(self, gradient_steps, batch_size=64):
            # Switch to train mode
            self.policy.set_training_mode(True)

            # Update learning rate
            self._update_learning_rate(
                [self.actor.optimizer, self.critic.optimizer,
                 self.cost_critic_optimizer]
            )

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []
            cost_critic_losses, lagrange_values = [], []

            for _ in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )

                # Get cost data from parallel buffer
                cost_data = None
                if self.cost_buffer is not None:
                    demo_idx = getattr(self.replay_buffer, '_last_demo_indices', None)
                    online_idx = getattr(self.replay_buffer, '_last_online_indices', None)
                    cost_data = self.cost_buffer.sample(demo_idx, online_idx)

                # --- Entropy coefficient update ---
                # Action by the current actor for the sampled observations
                actions_pi, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                log_prob = log_prob.reshape(-1, 1)

                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

                ent_coefs.append(ent_coef.item())

                # --- Reward critic update ---
                with torch.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    next_log_prob = next_log_prob.reshape(-1, 1)

                    # Compute target Q-values
                    _qc_target = _get_tqc_quantile_critics(self.critic_target)
                    if _qc_target is not None:
                        # TQC: iterate quantile critics, sort & drop top quantiles
                        next_quantiles = []
                        for critic_net in _qc_target:
                            features = self.critic_target.features_extractor(
                                replay_data.next_observations
                            )
                            x = torch.cat([features, next_actions], dim=-1)
                            next_quantiles.append(critic_net(x))
                        next_quantiles = torch.stack(next_quantiles, dim=1)
                        # Sort and drop top quantiles
                        sorted_q, _ = torch.sort(
                            next_quantiles.reshape(batch_size, -1), dim=1
                        )
                        n_target_quantiles = sorted_q.shape[1] - self.top_quantiles_to_drop_per_net * len(_qc_target)
                        if n_target_quantiles > 0:
                            sorted_q = sorted_q[:, :n_target_quantiles]
                        next_q = sorted_q.mean(dim=1, keepdim=True)
                    else:
                        # SAC: use critic forward, min of Q-values
                        next_q_values = self.critic_target(
                            replay_data.next_observations, next_actions
                        )
                        if isinstance(next_q_values, (list, tuple)):
                            next_q = torch.min(
                                torch.cat(next_q_values, dim=1), dim=1, keepdim=True
                            )[0]
                        elif next_q_values.dim() == 3:
                            # Stacked TQC output (batch, n_critics, n_quantiles)
                            all_q = next_q_values.reshape(batch_size, -1)
                            sorted_q, _ = torch.sort(all_q, dim=1)
                            n_critics = next_q_values.shape[1]
                            n_drop = getattr(self, 'top_quantiles_to_drop_per_net', 0) * n_critics
                            n_target = sorted_q.shape[1] - n_drop
                            if n_target > 0:
                                sorted_q = sorted_q[:, :n_target]
                            next_q = sorted_q.mean(dim=1, keepdim=True)
                        else:
                            next_q = next_q_values

                    target_q = replay_data.rewards + (
                        1 - replay_data.dones
                    ) * self.gamma * (next_q - ent_coef * next_log_prob)

                # Current Q-values
                _qc = _get_tqc_quantile_critics(self.critic)
                if _qc is not None:
                    current_quantiles = []
                    features = self.critic.features_extractor(
                        replay_data.observations
                    )
                    x = torch.cat([features, replay_data.actions], dim=-1)
                    for critic_net in _qc:
                        current_quantiles.append(critic_net(x))
                    # Quantile Huber loss
                    critic_loss = 0.0
                    n_quantiles = current_quantiles[0].shape[1]
                    tau = (torch.arange(n_quantiles, device=self.device, dtype=torch.float32) + 0.5) / n_quantiles
                    for current_q in current_quantiles:
                        # (batch, n_quantiles, 1) - (batch, 1, 1) -> (batch, n_quantiles, 1)
                        td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)
                        huber = F.smooth_l1_loss(current_q.unsqueeze(2),
                                                 target_q.unsqueeze(1).expand_as(current_q.unsqueeze(2)),
                                                 reduction='none')
                        quantile_loss = torch.abs(tau.view(1, -1, 1) - (td_error < 0).float()) * huber
                        critic_loss = critic_loss + quantile_loss.sum(dim=1).mean()
                else:
                    # SAC or TQC with forward()-based access
                    current_q_raw = self.critic(
                        replay_data.observations, replay_data.actions
                    )
                    if isinstance(current_q_raw, (list, tuple)):
                        current_q_list = list(current_q_raw)
                    elif current_q_raw.dim() == 3:
                        current_q_list = [current_q_raw[:, i, :] for i in range(current_q_raw.shape[1])]
                    else:
                        current_q_list = None

                    if current_q_list is not None and current_q_list[0].shape[-1] > 1:
                        # TQC quantile Huber loss via forward()
                        critic_loss = 0.0
                        n_quantiles = current_q_list[0].shape[-1]
                        tau = (torch.arange(n_quantiles, device=self.device, dtype=torch.float32) + 0.5) / n_quantiles
                        for current_q in current_q_list:
                            td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)
                            huber = F.smooth_l1_loss(current_q.unsqueeze(2),
                                                     target_q.unsqueeze(1).expand_as(current_q.unsqueeze(2)),
                                                     reduction='none')
                            quantile_loss = torch.abs(tau.view(1, -1, 1) - (td_error < 0).float()) * huber
                            critic_loss = critic_loss + quantile_loss.sum(dim=1).mean()
                    else:
                        # SAC MSE loss
                        if current_q_list is not None:
                            critic_loss = sum(F.mse_loss(q, target_q) for q in current_q_list)
                        else:
                            critic_loss = F.mse_loss(current_q_raw, target_q)

                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # --- Cost critic update ---
                if cost_data is not None:
                    with torch.no_grad():
                        next_cost_values = self.cost_critic_target(
                            replay_data.next_observations, next_actions
                        )
                        next_cost_q = torch.min(
                            torch.cat(next_cost_values, dim=1), dim=1, keepdim=True
                        )[0]
                        cost_target = cost_data + (
                            1 - replay_data.dones
                        ) * self.gamma * next_cost_q

                    current_cost_values = self.cost_critic(
                        replay_data.observations, replay_data.actions
                    )
                    cost_critic_loss = sum(
                        F.mse_loss(qc, cost_target) for qc in current_cost_values
                    )
                    cost_critic_losses.append(cost_critic_loss.item())

                    self.cost_critic_optimizer.zero_grad()
                    cost_critic_loss.backward()
                    self.cost_critic_optimizer.step()

                # --- Actor update ---
                # Q-values for current actions
                _qc_actor = _get_tqc_quantile_critics(self.critic)
                if _qc_actor is not None:
                    q_values = []
                    features_pi = self.critic.features_extractor(
                        replay_data.observations
                    )
                    x_pi = torch.cat([features_pi, actions_pi], dim=-1)
                    for critic_net in _qc_actor:
                        q_values.append(critic_net(x_pi).mean(dim=1, keepdim=True))
                    qf_pi = torch.cat(q_values, dim=1).mean(dim=1, keepdim=True)
                else:
                    # Fallback: use critic forward()
                    q_pi_raw = self.critic(replay_data.observations, actions_pi)
                    if isinstance(q_pi_raw, (list, tuple)):
                        qf_pi = torch.min(
                            torch.cat(q_pi_raw, dim=1), dim=1, keepdim=True
                        )[0]
                    elif q_pi_raw.dim() == 3:
                        # TQC stacked: average over quantiles, then mean over critics
                        qf_pi = q_pi_raw.mean(dim=2).mean(dim=1, keepdim=True)
                    else:
                        qf_pi = q_pi_raw

                # Lagrange-penalized actor loss
                lagrange = F.softplus(self.log_lagrange).detach()
                if cost_data is not None:
                    cost_q_values = self.cost_critic(
                        replay_data.observations, actions_pi
                    )
                    qc_pi = torch.min(
                        torch.cat(cost_q_values, dim=1), dim=1, keepdim=True
                    )[0]
                    actor_loss = (ent_coef * log_prob - qf_pi + lagrange * qc_pi).mean()
                else:
                    actor_loss = (ent_coef * log_prob - qf_pi).mean()

                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # --- Lagrange multiplier update ---
                if cost_data is not None:
                    batch_cost_mean = cost_data.mean()
                    lam_loss = -F.softplus(self.log_lagrange) * (
                        batch_cost_mean - self._cost_limit_per_step
                    )
                    self.lagrange_optimizer.zero_grad()
                    lam_loss.backward()
                    self.lagrange_optimizer.step()
                    lagrange_values.append(F.softplus(self.log_lagrange).item())

                # --- Target network update ---
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau
                )
                if cost_data is not None:
                    polyak_update(
                        self.cost_critic.parameters(),
                        self.cost_critic_target.parameters(),
                        self.tau
                    )

            # Logging
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if ent_coef_losses:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if cost_critic_losses:
                self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))
            if lagrange_values:
                self.logger.record("train/lagrange", np.mean(lagrange_values))

        def _get_torch_save_params(self):
            state_dicts, saved_vars = super()._get_torch_save_params()
            state_dicts += [
                "cost_critic.state_dict",
                "cost_critic_target.state_dict",
                "cost_critic_optimizer.state_dict",
            ]
            saved_vars += ["log_lagrange"]
            return state_dicts, saved_vars

        def _excluded_save_params(self):
            excluded = super()._excluded_save_params()
            excluded.add("cost_buffer")
            return excluded

    return SafeTQC


def _create_dual_policy_class(base_cls):
    """Create DualPolicyTQC class dynamically (IBRL: frozen IL anchor + RL actor).

    After CVAE pretraining, a frozen copy of the actor is kept as an immutable
    IL anchor. At every env step, both the frozen IL policy and the RL policy
    propose actions; the one with the higher Q-value is executed. TD targets
    also use max(Q_IL, Q_RL_entropy_adjusted) to accelerate critic learning.

    Args:
        base_cls: The TQC or SAC class to subclass

    Returns:
        DualPolicyTQC class
    """
    import copy
    import torch
    import torch.nn.functional as F
    import numpy as np
    from stable_baselines3.common.utils import obs_as_tensor, polyak_update

    class DualPolicyTQC(base_cls):
        """TQC/SAC with IBRL dual-policy: frozen IL anchor + RL actor.

        The IL actor is frozen after CVAE pretraining. At each env step, both
        actors propose actions; the one with higher Q wins. TD targets use
        max(Q_IL, Q_RL_entropy_adjusted) to accelerate critic bootstrap.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.il_actor = None
            self.il_noise_std = 0.1
            self.il_soft = False
            self.il_beta = 10.0

        def _ibrl_min_q(self, obs_tensor, action_tensor):
            """Compute min-over-critics Q-value estimate for (obs, action) pair.

            Args:
                obs_tensor: (batch, obs_dim) tensor on device
                action_tensor: (batch, action_dim) tensor on device

            Returns:
                (batch,) Q-value tensor
            """
            _qc = _get_tqc_quantile_critics(self.critic)
            if _qc is not None:
                # TQC path: extract features once, run each quantile critic
                features = self.critic.features_extractor(obs_tensor)
                x = torch.cat([features, action_tensor], dim=-1)
                q_per_net = []
                for critic_net in _qc:
                    q_quantiles = critic_net(x)  # (batch, n_quantiles)
                    q_per_net.append(q_quantiles.mean(dim=1))  # (batch,)
                q_stack = torch.stack(q_per_net, dim=1)  # (batch, n_critics)
                return q_stack.min(dim=1).values  # (batch,)
            else:
                # SAC path: use critic forward()
                q_raw = self.critic(obs_tensor, action_tensor)
                if isinstance(q_raw, (list, tuple)):
                    q_cat = torch.cat(q_raw, dim=1)  # (batch, n_critics)
                    return q_cat.min(dim=1).values
                elif q_raw.dim() == 3:
                    # TQC stacked (batch, n_critics, n_quantiles)
                    return q_raw.mean(dim=2).min(dim=1).values
                else:
                    return q_raw.squeeze(1)

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            if self.il_actor is None or self.num_timesteps < learning_starts:
                return super()._sample_action(
                    learning_starts, action_noise=action_noise, n_envs=n_envs
                )

            self.policy.set_training_mode(False)
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            with torch.no_grad():
                a_rl = self.actor._predict(obs_tensor, deterministic=False)
                if self.il_noise_std > 0:
                    a_rl = torch.clamp(
                        a_rl + torch.randn_like(a_rl) * self.il_noise_std,
                        -1.0, 1.0,
                    )

                a_il = self.il_actor._predict(obs_tensor, deterministic=True)

                q_rl = self._ibrl_min_q(obs_tensor, a_rl)   # (batch,)
                q_il = self._ibrl_min_q(obs_tensor, a_il)   # (batch,)

                if self.il_soft:
                    q_stack = torch.stack([q_rl, q_il], dim=1) * self.il_beta
                    probs = torch.softmax(q_stack, dim=1)
                    choices = torch.multinomial(probs, 1).squeeze(1)  # (batch,)
                    chosen = torch.where(
                        choices.bool().unsqueeze(-1), a_il, a_rl
                    )
                else:
                    chosen = torch.where(
                        (q_il > q_rl).unsqueeze(-1), a_il, a_rl
                    )

            buffer_action_np = chosen.cpu().numpy()
            action_np = self.policy.unscale_action(buffer_action_np)
            # action_noise arg intentionally ignored; IBRL noise replaces it
            return action_np, buffer_action_np

        def train(self, gradient_steps, batch_size=64):
            if self.il_actor is None:
                super().train(gradient_steps, batch_size)
                return

            # Switch to train mode
            self.policy.set_training_mode(True)

            # Update learning rate
            self._update_learning_rate(
                [self.actor.optimizer, self.critic.optimizer]
            )

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []
            il_selection_rates = []

            for _ in range(gradient_steps):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )

                # --- Entropy coefficient update ---
                actions_pi, log_prob = self.actor.action_log_prob(
                    replay_data.observations
                )
                log_prob = log_prob.reshape(-1, 1)

                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

                ent_coefs.append(ent_coef.item())

                # --- IBRL TD target ---
                with torch.no_grad():
                    next_obs = replay_data.next_observations
                    B = next_obs.shape[0]

                    # RL next action + log prob
                    next_actions_rl, next_log_prob_rl = self.actor.action_log_prob(
                        next_obs
                    )
                    next_log_prob_rl = next_log_prob_rl.reshape(-1, 1)

                    # IL next action (frozen, deterministic)
                    next_actions_il = self.il_actor._predict(
                        next_obs, deterministic=True
                    )

                    _qc_target = _get_tqc_quantile_critics(self.critic_target)
                    if _qc_target is not None:
                        # TQC quantile path: compute features once, reuse
                        feats = self.critic_target.features_extractor(next_obs)
                        x_rl = torch.cat([feats, next_actions_rl], dim=-1)
                        x_il = torch.cat([feats, next_actions_il], dim=-1)

                        q_rl_all = torch.stack(
                            [cn(x_rl) for cn in _qc_target], dim=1
                        )  # (B, n_critics, n_q)
                        q_il_all = torch.stack(
                            [cn(x_il) for cn in _qc_target], dim=1
                        )

                        # Scalar comparison (mean over quantiles, min over critics)
                        q_rl_scalar = q_rl_all.mean(dim=2).min(dim=1).values  # (B,)
                        q_il_scalar = q_il_all.mean(dim=2).min(dim=1).values

                        q_rl_adj_scalar = (
                            q_rl_scalar - (ent_coef * next_log_prob_rl).squeeze(1)
                        )
                        use_il = q_il_scalar > q_rl_adj_scalar  # (B,) bool
                        il_selection_rates.append(use_il.float().mean().item())

                        # Per-sample selection over full quantile distributions
                        q_rl_flat = q_rl_all.reshape(B, -1)   # (B, n_critics*n_q)
                        q_il_flat = q_il_all.reshape(B, -1)
                        # entropy adjustment broadcasts over quantile dim
                        q_rl_adj_flat = q_rl_flat - ent_coef * next_log_prob_rl

                        q_next_flat = torch.where(
                            use_il.unsqueeze(1), q_il_flat, q_rl_adj_flat
                        )

                        # Sort + drop top quantiles (same as SafeTQC)
                        sorted_q, _ = torch.sort(q_next_flat, dim=1)
                        n_target = (
                            sorted_q.shape[1]
                            - self.top_quantiles_to_drop_per_net * len(_qc_target)
                        )
                        if n_target > 0:
                            sorted_q = sorted_q[:, :n_target]
                        next_q = sorted_q.mean(dim=1, keepdim=True)

                    else:
                        # SAC scalar path
                        next_q_rl = self.critic_target(next_obs, next_actions_rl)
                        next_q_il = self.critic_target(next_obs, next_actions_il)

                        def _min_q(q_raw):
                            if isinstance(q_raw, (list, tuple)):
                                return torch.min(
                                    torch.cat(q_raw, dim=1), dim=1, keepdim=True
                                )[0]
                            return q_raw

                        q_rl_adj = _min_q(next_q_rl) - ent_coef * next_log_prob_rl
                        q_il_v = _min_q(next_q_il)
                        next_q = torch.max(q_rl_adj, q_il_v)

                        use_il = (q_il_v > q_rl_adj).squeeze(1)
                        il_selection_rates.append(use_il.float().mean().item())

                    target_q = replay_data.rewards + (
                        1 - replay_data.dones
                    ) * self.gamma * next_q

                # --- Current Q-values + critic loss ---
                _qc = _get_tqc_quantile_critics(self.critic)
                if _qc is not None:
                    current_quantiles = []
                    features = self.critic.features_extractor(
                        replay_data.observations
                    )
                    x = torch.cat([features, replay_data.actions], dim=-1)
                    for critic_net in _qc:
                        current_quantiles.append(critic_net(x))
                    critic_loss = 0.0
                    n_quantiles = current_quantiles[0].shape[1]
                    tau = (
                        torch.arange(
                            n_quantiles, device=self.device, dtype=torch.float32
                        ) + 0.5
                    ) / n_quantiles
                    for current_q in current_quantiles:
                        td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)
                        huber = F.smooth_l1_loss(
                            current_q.unsqueeze(2),
                            target_q.unsqueeze(1).expand_as(current_q.unsqueeze(2)),
                            reduction='none',
                        )
                        quantile_loss = (
                            torch.abs(tau.view(1, -1, 1) - (td_error < 0).float())
                            * huber
                        )
                        critic_loss = critic_loss + quantile_loss.sum(dim=1).mean()
                else:
                    current_q_raw = self.critic(
                        replay_data.observations, replay_data.actions
                    )
                    if isinstance(current_q_raw, (list, tuple)):
                        current_q_list = list(current_q_raw)
                    elif current_q_raw.dim() == 3:
                        current_q_list = [
                            current_q_raw[:, i, :]
                            for i in range(current_q_raw.shape[1])
                        ]
                    else:
                        current_q_list = None

                    if current_q_list is not None and current_q_list[0].shape[-1] > 1:
                        critic_loss = 0.0
                        n_quantiles = current_q_list[0].shape[-1]
                        tau = (
                            torch.arange(
                                n_quantiles, device=self.device, dtype=torch.float32
                            ) + 0.5
                        ) / n_quantiles
                        for current_q in current_q_list:
                            td_error = target_q.unsqueeze(1) - current_q.unsqueeze(2)
                            huber = F.smooth_l1_loss(
                                current_q.unsqueeze(2),
                                target_q.unsqueeze(1).expand_as(
                                    current_q.unsqueeze(2)
                                ),
                                reduction='none',
                            )
                            quantile_loss = (
                                torch.abs(
                                    tau.view(1, -1, 1) - (td_error < 0).float()
                                ) * huber
                            )
                            critic_loss = (
                                critic_loss + quantile_loss.sum(dim=1).mean()
                            )
                    else:
                        if current_q_list is not None:
                            critic_loss = sum(
                                F.mse_loss(q, target_q) for q in current_q_list
                            )
                        else:
                            critic_loss = F.mse_loss(current_q_raw, target_q)

                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                # --- Actor update ---
                _qc_actor = _get_tqc_quantile_critics(self.critic)
                if _qc_actor is not None:
                    q_values = []
                    features_pi = self.critic.features_extractor(
                        replay_data.observations
                    )
                    x_pi = torch.cat([features_pi, actions_pi], dim=-1)
                    for critic_net in _qc_actor:
                        q_values.append(critic_net(x_pi).mean(dim=1, keepdim=True))
                    qf_pi = torch.cat(q_values, dim=1).mean(dim=1, keepdim=True)
                else:
                    q_pi_raw = self.critic(replay_data.observations, actions_pi)
                    if isinstance(q_pi_raw, (list, tuple)):
                        qf_pi = torch.min(
                            torch.cat(q_pi_raw, dim=1), dim=1, keepdim=True
                        )[0]
                    elif q_pi_raw.dim() == 3:
                        qf_pi = q_pi_raw.mean(dim=2).mean(dim=1, keepdim=True)
                    else:
                        qf_pi = q_pi_raw

                actor_loss = (ent_coef * log_prob - qf_pi).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # --- Target network update ---
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )

            # Logging
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if ent_coef_losses:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if il_selection_rates:
                self.logger.record(
                    "train/ibrl_il_selection_rate", np.mean(il_selection_rates)
                )

        def _get_torch_save_params(self):
            state_dicts, saved_vars = super()._get_torch_save_params()
            if self.il_actor is not None:
                state_dicts.append("il_actor.state_dict")
            saved_vars += ["il_noise_std", "il_soft", "il_beta"]
            return state_dicts, saved_vars

    return DualPolicyTQC


class SafeTrainingCallback:
    """Factory for creating the SafeTQC training callback.

    Tracks per-transition costs and per-episode cumulative costs.
    """

    @staticmethod
    def create(base_callback_cls):
        """Create callback class using the imported BaseCallback."""

        class _SafeTrainingCallback(base_callback_cls):
            def __init__(self, cost_buffer):
                super().__init__(verbose=0)
                self.cost_buffer = cost_buffer
                self._ep_cost = 0.0
                self._ep_costs = []

            def _on_step(self):
                infos = self.locals.get('infos', [{}])
                dones = self.locals.get('dones', [False])

                for i in range(len(dones)):
                    info = infos[i] if i < len(infos) else {}
                    cost = info.get('cost_chunk', info.get('cost', 0.0))
                    self.cost_buffer.add(cost)
                    self._ep_cost += cost

                    if dones[i]:
                        self._ep_costs.append(self._ep_cost)
                        self._ep_cost = 0.0

                # Log periodically
                if self._ep_costs and self.num_timesteps % 100 == 0:
                    self.logger.record("safe/mean_episode_cost", np.mean(self._ep_costs[-100:]))
                    lagrange = getattr(self.model, 'log_lagrange', None)
                    if lagrange is not None:
                        import torch
                        self.logger.record("safe/lagrange", torch.nn.functional.softplus(lagrange).item())

                return True

        return _SafeTrainingCallback


def main():
    parser = argparse.ArgumentParser(
        description='Train SAC/TQC agent with Chunk CVAE + Q-chunking for Jetbot navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demos demos/recording.npz --headless
      Train headless with demo replay

  %(prog)s --demos demos/recording.npz --headless --timesteps 500000
      Train for 500k timesteps

  %(prog)s --demos demos/recording.npz --headless --chunk-size 5
      Use chunk size 5 instead of default 10

  %(prog)s --demos demos/recording.npz --headless --cpu --timesteps 1000
      Quick smoke test on CPU
        """
    )

    # Training arguments
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--demos', type=str, required=True,
                        help='Path to demo .npz file (required)')
    parser.add_argument('--utd-ratio', type=int, default=20,
                        help='Update-to-data ratio / gradient steps per env step (default: 20)')
    parser.add_argument('--buffer-size', type=int, default=300000,
                        help='Replay buffer size (default: 300000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005)')
    parser.add_argument('--ent-coef', type=str, default='auto_0.006',
                        help='Entropy coefficient, "auto" or "auto_<init>" for learned (default: auto_0.006)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--learning-starts', type=int, default=0,
                        help='Steps before training starts (default: 0, demos available immediately)')
    parser.add_argument('--demo-ratio', type=float, default=0.5,
                        help='Fraction of each batch sampled from demos, 0.0-1.0 (default: 0.5)')

    # Chunk CVAE arguments
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Action chunk size for Q-chunking (default: 10)')
    parser.add_argument('--cvae-z-dim', type=int, default=8,
                        help='CVAE latent dimension (default: 8)')
    parser.add_argument('--cvae-epochs', type=int, default=100,
                        help='CVAE pretraining epochs (default: 100)')
    parser.add_argument('--cvae-beta', type=float, default=0.1,
                        help='CVAE KL weight (default: 0.1)')
    parser.add_argument('--cvae-lr', type=float, default=1e-3,
                        help='CVAE pretraining learning rate (default: 1e-3)')

    # Environment arguments
    parser.add_argument('--reward-mode', choices=['dense', 'sparse'], default='dense',
                        help='Reward mode (default: dense)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (faster training)')
    parser.add_argument('--num-obstacles', type=int, default=5,
                        help='Number of obstacles to spawn (default: 5)')
    parser.add_argument('--arena-size', type=float, default=4.0,
                        help='Side length of square arena in meters (default: 4.0)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force training on CPU instead of GPU')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    parser.add_argument('--min-goal', type=float, default=0.5,
                        help='Minimum distance from robot start to goal in meters (default: 0.5)')
    parser.add_argument('--inflation-radius', type=float, default=0.08,
                        help='Obstacle inflation radius for A* planner in meters (default: 0.08)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Save checkpoint every N steps (default: 50000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .zip to resume training from')

    # Logging arguments
    parser.add_argument('--more-debug', action='store_true',
                        help='Print per-episode stats')

    # Output arguments
    parser.add_argument('--output', type=str, default='models/tqc_jetbot.zip',
                        help='Output model path (default: models/tqc_jetbot.zip)')
    parser.add_argument('--tensorboard-log', type=str, default='./runs/',
                        help='TensorBoard log directory (default: ./runs/)')

    # Safe RL arguments
    safe_group = parser.add_argument_group('Safe RL (SafeTQC)')
    safe_group.add_argument('--safe', action='store_true',
                            help='Enable SafeTQC with cost critic + Lagrange multiplier')
    safe_group.add_argument('--cost-limit', type=float, default=25.0,
                            help='Per-episode cost budget (default: 25.0)')
    safe_group.add_argument('--lagrange-lr', type=float, default=3e-4,
                            help='Lagrange multiplier learning rate (default: 3e-4)')
    safe_group.add_argument('--lagrange-init', type=float, default=0.0,
                            help='Initial log-lambda value (default: 0.0)')
    safe_group.add_argument('--cost-n-critics', type=int, default=2,
                            help='Number of cost critic networks (default: 2)')
    safe_group.add_argument('--cost-critic-type', choices=['mean', 'quantile'],
                            default='mean',
                            help='Cost critic type: mean (MSE) or quantile (default: mean)')
    safe_group.add_argument('--cost-type', choices=['proximity', 'collision', 'both'],
                            default='proximity',
                            help='Cost signal type (default: proximity)')
    safe_group.add_argument('--keep-proximity-reward', action='store_true',
                            help='Keep proximity penalty in reward even with --safe')

    # Dual-policy (IBRL) arguments
    dual_group = parser.add_argument_group('Dual-policy (IBRL)')
    dual_group.add_argument('--dual-policy', action='store_true',
                            help='Enable IBRL dual-policy (frozen IL anchor + RL actor)')
    dual_group.add_argument('--dual-policy-noise', type=float, default=0.1,
                            help='Gaussian noise std added to RL action proposals (default: 0.1)')
    dual_group.add_argument('--dual-policy-soft', action='store_true',
                            help='Boltzmann action selection instead of greedy argmax')
    dual_group.add_argument('--dual-policy-beta', type=float, default=10.0,
                            help='Boltzmann temperature for soft dual-policy (default: 10.0)')

    args = parser.parse_args()

    # Validate demo_ratio
    if not 0.0 <= args.demo_ratio <= 1.0:
        parser.error(f"--demo-ratio must be between 0.0 and 1.0, got {args.demo_ratio}")

    # Parse ent_coef
    ent_coef = args.ent_coef
    if not ent_coef.startswith('auto'):
        ent_coef = float(ent_coef)

    print("=" * 60)
    print("SAC/TQC + Chunk CVAE + Q-Chunking for Jetbot Navigation")
    print("=" * 60)
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Demos: {args.demos}")
    print(f"  UTD ratio: {args.utd_ratio}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma} (effective: {args.gamma ** args.chunk_size:.6f})")
    print(f"  Tau: {args.tau}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Demo ratio: {args.demo_ratio}")
    print(f"  Learning starts: {args.learning_starts}")
    print(f"  CVAE: z_dim={args.cvae_z_dim}, epochs={args.cvae_epochs}, "
          f"beta={args.cvae_beta}, lr={args.cvae_lr}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Headless: {args.headless}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"  Output: {args.output}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    if args.safe:
        print(f"  SafeTQC: ENABLED")
        print(f"    Cost limit: {args.cost_limit}")
        print(f"    Cost type: {args.cost_type}")
        print(f"    Cost critics: {args.cost_n_critics} ({args.cost_critic_type})")
        print(f"    Lagrange LR: {args.lagrange_lr}, init: {args.lagrange_init}")
        print(f"    Keep proximity reward: {args.keep_proximity_reward}")
    if args.dual_policy:
        print(f"  Dual-policy (IBRL): ENABLED")
        print(f"    IL noise std: {args.dual_policy_noise}")
        print(f"    Soft selection: {args.dual_policy_soft}")
        print(f"    Boltzmann beta: {args.dual_policy_beta}")
    print("=" * 60 + "\n")

    # Validate demo data first (fail fast)
    validate_demo_data(args.demos, require_cost=args.safe)

    # Import here to allow --help without Isaac Sim
    import torch
    from jetbot_rl_env import JetbotNavigationEnv, ChunkedEnvWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from demo_utils import make_chunk_transitions

    # Try TQC first, fall back to SAC
    try:
        from sb3_contrib import TQC
        algo_cls = TQC
        algo_name = "TQC"
    except ImportError:
        from stable_baselines3 import SAC
        algo_cls = SAC
        algo_name = "SAC"
        print("sb3-contrib not found, falling back to SAC")

    # Mutual exclusion
    if args.dual_policy and args.safe:
        parser.error("--dual-policy and --safe are mutually exclusive")

    # Wrap with SafeTQC if --safe
    if args.safe:
        SafeTQC = _create_safe_tqc_class(algo_cls)
        algo_cls = SafeTQC
        algo_name = f"Safe{algo_name}"

    # Wrap with DualPolicy if --dual-policy
    if args.dual_policy:
        DualPolicyClass = _create_dual_policy_class(algo_cls)
        algo_cls = DualPolicyClass
        algo_name = f"DualPolicy{algo_name}"

    print(f"Using algorithm: {algo_name}")

    # Create environment with ChunkedEnvWrapper
    import time as _time
    print("\nCreating environment...")
    _t0 = _time.time()
    half = args.arena_size / 2.0
    workspace_bounds = {'x': [-half, half], 'y': [-half, half]}
    safe_mode = args.safe and not args.keep_proximity_reward
    raw_env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
        num_obstacles=args.num_obstacles,
        workspace_bounds=workspace_bounds,
        max_episode_steps=args.max_steps,
        min_goal_dist=args.min_goal,
        inflation_radius=args.inflation_radius,
        cost_type=getattr(args, 'cost_type', 'proximity'),
        safe_mode=safe_mode,
    )
    raw_env = ChunkedEnvWrapper(raw_env, chunk_size=args.chunk_size, gamma=args.gamma)
    print(f"  Environment created in {_time.time() - _t0:.1f}s")
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space: {raw_env.action_space.shape} (chunk_size={args.chunk_size})")

    env = DummyVecEnv([lambda: raw_env])
    print("  No VecNormalize (using LayerNorm in critics instead)")
    print()

    # Load step-level demo transitions (for CVAE pretraining)
    print("Loading demo transitions...")
    demo_costs_step = None
    if args.safe:
        demo_obs_step, demo_actions_step, demo_rewards_step, _, demo_dones_step, demo_costs_step = \
            load_demo_transitions(args.demos, load_costs=True)
    else:
        demo_obs_step, demo_actions_step, demo_rewards_step, _, demo_dones_step = \
            load_demo_transitions(args.demos)
    from demo_io import open_demo
    demo_data = open_demo(args.demos)
    episode_lengths = demo_data['episode_lengths']

    # Build chunk-level transitions (for replay buffer)
    print("Building chunk-level transitions...")
    chunk_costs = None
    if args.safe and demo_costs_step is not None:
        chunk_obs, chunk_acts, chunk_rews, chunk_next, chunk_dones, chunk_costs = make_chunk_transitions(
            demo_obs_step, demo_actions_step, demo_rewards_step, demo_dones_step,
            episode_lengths, args.chunk_size, args.gamma, demo_costs=demo_costs_step)
    else:
        chunk_obs, chunk_acts, chunk_rews, chunk_next, chunk_dones = make_chunk_transitions(
            demo_obs_step, demo_actions_step, demo_rewards_step, demo_dones_step,
            episode_lengths, args.chunk_size, args.gamma)
    print(f"  {len(chunk_obs)} chunk transitions from {len(episode_lengths)} episodes")
    print()

    # Create replay buffer with chunk-level demo data
    device_str = "cpu" if args.cpu else "auto"
    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    replay_buffer = make_demo_replay_buffer(
        ReplayBuffer,
        args.buffer_size,
        raw_env.observation_space,
        raw_env.action_space,
        device=device,
        demo_obs=chunk_obs,
        demo_actions=chunk_acts,
        demo_rewards=chunk_rews,
        demo_next_obs=chunk_next,
        demo_dones=chunk_dones,
        demo_ratio=args.demo_ratio,
    )
    print(f"Demo replay buffer created: {len(chunk_obs)} chunk-level demo transitions")

    # Create cost replay buffer if safe mode
    cost_buffer = None
    if args.safe:
        cost_buffer = CostReplayBuffer(
            args.buffer_size, device,
            demo_costs=chunk_costs,
            demo_ratio=args.demo_ratio,
        )
        print(f"Cost replay buffer created: {len(chunk_costs) if chunk_costs is not None else 0} demo costs")
    print()

    # Effective gamma for chunk-level Bellman updates
    effective_gamma = args.gamma ** args.chunk_size

    # Feature extractor dimensions
    obs_feature_dim = 96
    features_dim = obs_feature_dim + args.cvae_z_dim

    # Create or resume model
    _t0 = _time.time()
    if args.resume:
        print(f"Resuming {algo_name} model from {args.resume}...")
        model = algo_cls.load(
            args.resume,
            env=env,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
        )
        # Verify chunk size matches
        loaded_chunk = model.action_space.shape[0] // 2
        if loaded_chunk != args.chunk_size:
            print(f"  Warning: loaded model chunk_size={loaded_chunk} != --chunk-size={args.chunk_size}")
            print(f"  Using loaded chunk_size={loaded_chunk}")
        # Override mutable hyperparams the user may have changed
        model.learning_rate = args.lr
        model.batch_size = args.batch_size
        model.gamma = effective_gamma
        model.tau = args.tau
        model.gradient_steps = args.utd_ratio
        model.ent_coef = ent_coef
        # Replace replay buffer with fresh demo buffer (online data is lost)
        model.replay_buffer = replay_buffer
        # LayerNorm + CVAE weights are already baked into the loaded checkpoint
        print(f"  Model resumed in {_time.time() - _t0:.1f}s")
    else:
        print(f"Creating {algo_name} model...")
        fe_cls = ChunkCVAEFeatureExtractor.get_class(
            BaseFeaturesExtractor, z_dim=args.cvae_z_dim)
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            features_extractor_class=fe_cls,
            features_extractor_kwargs=dict(features_dim=features_dim),
        )
        if "TQC" in algo_name:
            policy_kwargs['n_critics'] = 5

        model_kwargs = dict(
            verbose=1,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=effective_gamma,
            tau=args.tau,
            ent_coef=ent_coef,
            target_entropy=-2.0,
            gradient_steps=args.utd_ratio,
            learning_starts=args.learning_starts,
            train_freq=1,
            policy_kwargs=policy_kwargs,
        )
        if args.safe:
            model_kwargs.update(dict(
                cost_limit=args.cost_limit,
                lagrange_lr=args.lagrange_lr,
                lagrange_init=args.lagrange_init,
                cost_n_critics=args.cost_n_critics,
                cost_critic_type=args.cost_critic_type,
                cost_buffer=cost_buffer,
                max_episode_steps=args.max_steps,
            ))
        model = algo_cls("MlpPolicy", env, **model_kwargs)
        # Replace the default replay buffer with our demo buffer
        model.replay_buffer = replay_buffer
        # Inject LayerNorm into critics
        inject_layernorm_into_critics(model)
        # Also inject LayerNorm into cost critic if safe mode
        if args.safe and hasattr(model, 'cost_critic'):
            _inject_layernorm_cost_critic(model)
        print(f"  Model created in {_time.time() - _t0:.1f}s")
    print()

    # CVAE pretraining (replaces BC warmstart)
    if not args.resume:
        pretrain_chunk_cvae(
            model, demo_obs_step, demo_actions_step, episode_lengths,
            chunk_size=args.chunk_size, z_dim=args.cvae_z_dim,
            epochs=args.cvae_epochs, batch_size=args.batch_size,
            lr=args.cvae_lr, beta=args.cvae_beta, gamma=args.gamma,
        )
        # Copy pretrained FE weights to cost critic too
        if args.safe and hasattr(model, 'cost_critic'):
            fe_state = model.actor.features_extractor.state_dict()
            model.cost_critic.features_extractor.load_state_dict(fe_state)
            model.cost_critic_target.features_extractor.load_state_dict(fe_state)
            print("  Feature extractor weights copied to cost critic/cost_critic_target")

    # Freeze IL actor for dual-policy (IBRL)
    if args.dual_policy:
        import copy as _copy
        il_actor = _copy.deepcopy(model.policy.actor)
        for p in il_actor.parameters():
            p.requires_grad = False
        il_actor.eval()
        il_actor.to(device)
        model.il_actor = il_actor
        model.il_noise_std = args.dual_policy_noise
        model.il_soft = args.dual_policy_soft
        model.il_beta = args.dual_policy_beta
        n_params = sum(p.numel() for p in il_actor.parameters())
        if not args.resume:
            print(f"  IL policy frozen from CVAE-pretrained actor ({n_params:,} params)")
        else:
            print(f"  IL policy frozen from resumed checkpoint ({n_params:,} params)")

    # Create callbacks
    checkpoint_dir = Path(args.output).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if "Safe" in algo_name:
        prefix = "safe_tqc_jetbot" if "TQC" in algo_name else "safe_sac_jetbot"
    elif "DualPolicy" in algo_name:
        prefix = "dual_tqc_jetbot" if "TQC" in algo_name else "dual_sac_jetbot"
    else:
        prefix = "tqc_jetbot" if algo_name == "TQC" else "sac_jetbot"
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=prefix,
        verbose=1,
    )
    callbacks = [checkpoint_callback]
    if args.more_debug:
        callbacks.append(VerboseEpisodeCallback.create(BaseCallback))
    if args.safe and cost_buffer is not None:
        safe_cb_cls = SafeTrainingCallback.create(BaseCallback)
        callbacks.append(safe_cb_cls(cost_buffer))
    callback = CallbackList(callbacks)

    # Train
    print("\n" + "=" * 60)
    print(f"Starting {algo_name} + Chunk CVAE + Q-Chunking Training")
    print("=" * 60)
    print(f"View TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")

    # Save final model (no VecNormalize stats to save)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {output_path}")
    print(f"  Checkpoints in: {checkpoint_dir}")
    print(f"  TensorBoard logs: {args.tensorboard_log}")
    print("\nTo evaluate the trained policy:")
    print(f"  ./run.sh eval_policy.py {output_path}")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    main()
