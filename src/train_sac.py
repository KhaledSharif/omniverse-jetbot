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
import time as _time
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


def _is_crossq_model(model):
    """Check if model is a CrossQ instance (uses BatchRenorm, no target networks)."""
    try:
        from sb3_contrib import CrossQ
        return isinstance(model, CrossQ)
    except ImportError:
        return False


def _set_bn_mode(module, mode):
    """Toggle BatchRenorm training mode if supported (CrossQ only; no-op for TQC/SAC)."""
    fn = getattr(module, 'set_bn_training_mode', None)
    if fn is not None:
        fn(mode)


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

                # Dynamic split: state = obs[:obs_dim-24], lidar = obs[obs_dim-24:]
                obs_dim = observation_space.shape[0]
                self._state_dim = obs_dim - 24
                self._lidar_dim = 24

                self.state_mlp = nn.Sequential(
                    nn.Linear(self._state_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                )

                self.lidar_mlp = nn.Sequential(
                    nn.Linear(self._lidar_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                )

            def encode_obs(self, observations):
                """Encode observations into obs_features (without z padding).

                Returns:
                    obs_features tensor of shape (batch, 96)
                """
                state = symlog(observations[:, :self._state_dim])
                lidar = symlog(observations[:, self._state_dim:])
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


class TemporalCVAEFeatureExtractor:
    """SB3 feature extractor with GRU over stacked observation frames.

    Input: (batch, n_frames * 34) flattened frame-stacked observations.
    Reshapes to (batch, n_frames, 34), applies per-frame state/lidar MLPs
    to get 96D per frame, feeds through a single-layer GRU, takes the last
    hidden state, and concatenates a zero-padded z-slot.

    Output: concat(gru_hidden, z_pad) = gru_hidden_dim + z_dim
    """

    _cls = None

    @staticmethod
    def create(base_extractor_cls, n_frames, gru_hidden_dim=128, z_dim=8):
        import torch
        import torch.nn as nn

        class _TemporalCVAEFeatureExtractor(base_extractor_cls):
            def __init__(self, observation_space, features_dim=None):
                if features_dim is None:
                    features_dim = gru_hidden_dim + z_dim
                super().__init__(observation_space, features_dim=features_dim)
                self._z_dim = z_dim
                self._obs_feature_dim = gru_hidden_dim
                self._n_frames = n_frames
                # Dynamic per-frame dim: total obs / n_frames
                self._per_frame_dim = observation_space.shape[0] // n_frames
                # Dynamic split: state = frame[:frame_dim-24], lidar = frame[frame_dim-24:]
                self._state_dim = self._per_frame_dim - 24
                self._lidar_dim = 24

                self.state_mlp = nn.Sequential(
                    nn.Linear(self._state_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                )

                self.lidar_mlp = nn.Sequential(
                    nn.Linear(self._lidar_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                )

                self.gru = nn.GRU(
                    input_size=96,
                    hidden_size=gru_hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )

            def encode_obs(self, observations):
                """Encode frame-stacked observations into obs_features via GRU.

                Args:
                    observations: (batch, n_frames * per_frame_dim) tensor

                Returns:
                    obs_features tensor of shape (batch, gru_hidden_dim)
                """
                batch = observations.shape[0]
                # Reshape to (batch, n_frames, per_frame_dim)
                x = observations.view(batch, self._n_frames, self._per_frame_dim)

                # Per-frame MLP: process all frames at once
                # Flatten to (batch * n_frames, per_frame_dim) for MLP
                x_flat = x.reshape(batch * self._n_frames, self._per_frame_dim)
                state = symlog(x_flat[:, :self._state_dim])
                lidar = symlog(x_flat[:, self._state_dim:])
                state_features = self.state_mlp(state)   # (B*T, 32)
                lidar_features = self.lidar_mlp(lidar)    # (B*T, 64)
                frame_features = torch.cat([state_features, lidar_features], dim=-1)  # (B*T, 96)

                # Reshape back to sequence: (batch, n_frames, 96)
                frame_features = frame_features.view(batch, self._n_frames, 96)

                # GRU: take last hidden state
                _, h_n = self.gru(frame_features)  # h_n: (1, batch, gru_hidden_dim)
                return h_n.squeeze(0)  # (batch, gru_hidden_dim)

            def forward(self, observations):
                obs_features = self.encode_obs(observations)
                z_pad = torch.zeros(
                    obs_features.shape[0], self._z_dim,
                    device=obs_features.device, dtype=obs_features.dtype,
                )
                return torch.cat([obs_features, z_pad], dim=-1)

        return _TemporalCVAEFeatureExtractor

    @classmethod
    def get_class(cls, base_extractor_cls, n_frames, gru_hidden_dim=128, z_dim=8):
        cls._cls = cls.create(base_extractor_cls, n_frames,
                              gru_hidden_dim=gru_hidden_dim, z_dim=z_dim)
        return cls._cls


class VisionCVAEFeatureExtractor:
    """SB3 feature extractor for camera observations with three-way split.

    Splits obs into state (0:state_dim), image features (state_dim:-24),
    and LiDAR (-24:). State and LiDAR use symlog; DINOv2 features are raw.

    Output: concat(state_32D, image_64D, lidar_64D, z_pad) = 160 + z_dim = 168D
    """

    _cls = None

    @staticmethod
    def create(base_extractor_cls, z_dim=8):
        import torch
        import torch.nn as nn

        class _VisionCVAEFeatureExtractor(base_extractor_cls):
            def __init__(self, observation_space, features_dim=168):
                super().__init__(observation_space, features_dim=features_dim)
                self._z_dim = z_dim
                self._obs_feature_dim = features_dim - z_dim  # 160

                obs_dim = observation_space.shape[0]
                self._lidar_dim = 24
                # image_dim = IMAGE_FEATURE_DIM (384)
                # state_dim = obs_dim - 384 - 24 = 10 or 12
                from jetbot_config import IMAGE_FEATURE_DIM
                self._image_dim = IMAGE_FEATURE_DIM
                self._state_dim = obs_dim - self._image_dim - self._lidar_dim

                self.state_mlp = nn.Sequential(
                    nn.Linear(self._state_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                )

                self.image_mlp = nn.Sequential(
                    nn.Linear(self._image_dim, 256),
                    nn.SiLU(),
                    nn.Linear(256, 64),
                    nn.SiLU(),
                )

                self.lidar_mlp = nn.Sequential(
                    nn.Linear(self._lidar_dim, 128),
                    nn.SiLU(),
                    nn.Linear(128, 64),
                    nn.SiLU(),
                )

            def encode_obs(self, observations):
                """Encode observations into obs_features (without z padding).

                Returns:
                    obs_features tensor of shape (batch, 160)
                """
                state = symlog(observations[:, :self._state_dim])
                image = observations[:, self._state_dim:self._state_dim + self._image_dim]
                lidar = symlog(observations[:, -self._lidar_dim:])
                state_features = self.state_mlp(state)    # 32D
                image_features = self.image_mlp(image)    # 64D
                lidar_features = self.lidar_mlp(lidar)    # 64D
                return torch.cat([state_features, image_features, lidar_features], dim=-1)

            def forward(self, observations):
                obs_features = self.encode_obs(observations)
                z_pad = torch.zeros(
                    obs_features.shape[0], self._z_dim,
                    device=obs_features.device, dtype=obs_features.dtype,
                )
                return torch.cat([obs_features, z_pad], dim=-1)

        return _VisionCVAEFeatureExtractor

    @classmethod
    def get_class(cls, base_extractor_cls, z_dim=8):
        cls._cls = cls.create(base_extractor_cls, z_dim=z_dim)
        return cls._cls


def pretrain_chunk_cvae(model, demo_obs, demo_actions, episode_lengths,
                        chunk_size, z_dim=8, epochs=100, batch_size=256,
                        lr=1e-3, beta=0.1, gamma=0.99, gru_lr=None):
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
        gru_lr: separate learning rate for GRU parameters (None = use lr for all)
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
    if gru_lr is not None and hasattr(model.actor.features_extractor, 'gru'):
        # Split into GRU params (low LR) and everything else (CVAE LR)
        gru_params, other_fe_params = [], []
        for name, param in model.actor.features_extractor.named_parameters():
            if 'gru.' in name:
                gru_params.append(param)
            else:
                other_fe_params.append(param)
        other_params = (
            other_fe_params
            + list(model.actor.latent_pi.parameters())
            + list(model.actor.mu.parameters())
            + list(cvae_encoder.parameters())
            + list(cvae_mu.parameters())
            + list(cvae_logvar.parameters())
        )
        param_groups = [
            {'params': other_params, 'lr': lr},
            {'params': gru_params, 'lr': gru_lr},
        ]
        optimizer = torch.optim.Adam(param_groups)
        print(f"  CVAE optimizer: base_lr={lr}, gru_lr={gru_lr}")
    else:
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
        _diag_raw_kl_sum = 0.0
        _diag_mu_abs_sum = 0.0
        _diag_std_mean_sum = 0.0
        _diag_active_dims_sum = 0.0
        _diag_z_abs_sum = 0.0
        n_batches = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Encode observations → obs_features (96D)
            obs_features = model.actor.features_extractor.encode_obs(obs_batch)

            # CVAE encoder: (obs_features, action_chunk) → z
            # Detach obs_features so encoder gradient flows only through z,
            # preventing the decoder from bypassing z via obs_features.
            enc_input = torch.cat([obs_features.detach(), act_batch], dim=-1)
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

            # Loss: L1 reconstruction + β·KL (with annealing + free bits)
            recon_loss = torch.nn.functional.l1_loss(pred_actions, act_batch)
            # KL annealing: β ramps from 0 → beta over first 40% of epochs
            anneal_frac = min(1.0, epoch / max(1, int(0.4 * epochs)))
            beta_t = beta * anneal_frac
            # Free bits: clamp per-dim KL to ≥ 0.25 nats to prevent collapse
            kl_per_dim = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            kl_loss = torch.mean(torch.clamp(kl_per_dim, min=0.25))
            loss = recon_loss + beta_t * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

            # Accumulate diagnostics (detached, no grad overhead)
            with torch.no_grad():
                _diag_raw_kl_sum += kl_per_dim.mean().item()
                _diag_mu_abs_sum += mu_z.abs().mean().item()
                _diag_std_mean_sum += std_z.mean().item()
                _diag_active_dims_sum += (kl_per_dim.mean(dim=0) > 0.25).float().sum().item()
                _diag_z_abs_sum += z.abs().mean().item()

        if True:  # log every epoch for diagnostics
            avg_recon = total_recon / n_batches
            avg_kl = total_kl / n_batches
            avg_raw_kl = _diag_raw_kl_sum / n_batches
            avg_mu_abs = _diag_mu_abs_sum / n_batches
            avg_std = _diag_std_mean_sum / n_batches
            avg_active = _diag_active_dims_sum / n_batches
            avg_z_abs = _diag_z_abs_sum / n_batches
            print(f"  Epoch {epoch+1:4d}/{epochs}, "
                  f"L1: {avg_recon:.6f}, "
                  f"KL: {avg_kl:.6f} (raw: {avg_raw_kl:.4f}), "
                  f"beta_t: {beta_t:.4f}")
            print(f"         "
                  f"enc |mu|: {avg_mu_abs:.4f}, "
                  f"enc std: {avg_std:.4f}, "
                  f"|z|: {avg_z_abs:.4f}, "
                  f"active_dims: {avg_active:.1f}/{z_dim}")

    # Copy pretrained features_extractor weights → critic and critic_target
    fe_state = model.actor.features_extractor.state_dict()
    model.critic.features_extractor.load_state_dict(fe_state)
    if hasattr(model, 'critic_target') and model.critic_target is not None:
        model.critic_target.features_extractor.load_state_dict(fe_state)
        print("  Feature extractor weights copied to critic/critic_target")
    else:
        print("  Feature extractor weights copied to critic (no target network)")

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

    Skipped for CrossQ models (BatchRenorm is built into the critic architecture).
    """
    if _is_crossq_model(model):
        print("CrossQ detected — skipping LayerNorm injection (BatchRenorm built-in)")
        return

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


def _apply_gru_lr(model, gru_lr):
    """Recreate actor & critic optimizers with a separate lower LR for GRU parameters.

    Splits parameters into GRU group (gru_lr) and non-GRU group (model.lr_schedule(1)).
    Safe no-op if the feature extractor has no GRU (e.g. ChunkCVAEFeatureExtractor).
    """
    import torch

    if not hasattr(model.actor.features_extractor, 'gru'):
        return

    base_lr = model.lr_schedule(1)

    def _split_params(module):
        """Split module parameters into (gru_params, other_params)."""
        gru_params, other_params = [], []
        for name, param in module.named_parameters():
            if '.gru.' in name or name.startswith('gru.'):
                gru_params.append(param)
            else:
                other_params.append(param)
        return gru_params, other_params

    # Actor optimizer
    gru_p, other_p = _split_params(model.actor)
    groups = []
    if other_p:
        groups.append({'params': other_p, 'lr': base_lr})
    if gru_p:
        groups.append({'params': gru_p, 'lr': gru_lr})
    model.actor.optimizer = torch.optim.Adam(groups)

    # Critic optimizer
    gru_p, other_p = _split_params(model.critic)
    groups = []
    if other_p:
        groups.append({'params': other_p, 'lr': base_lr})
    if gru_p:
        groups.append({'params': gru_p, 'lr': gru_lr})
    model.critic.optimizer = torch.optim.Adam(groups)

    # Cost critic optimizer (SafeTQC)
    if hasattr(model, 'cost_critic') and hasattr(model, 'cost_critic_optimizer'):
        gru_p, other_p = _split_params(model.cost_critic)
        groups = []
        if other_p:
            groups.append({'params': other_p, 'lr': base_lr})
        if gru_p:
            groups.append({'params': gru_p, 'lr': gru_lr})
        model.cost_critic_optimizer = torch.optim.Adam(groups)

    print(f"GRU learning rate applied: gru_lr={gru_lr}, base_lr={base_lr}")


def _patch_actor_for_stability(actor, mean_clamp=3.0, log_std_min=-5.0):
    """Monkey-patch actor.get_action_dist_params() for training stability.

    Addresses two failure modes in SAC with tanh-squashed policies:
    1. Pre-tanh mean explosion: |mu| > 2 saturates tanh, making policy
       near-deterministic regardless of log_std. SB3 has no mean clamping
       (original Haarnoja SAC had L2 reg, SB3 dropped it).
    2. log_std collapse: SB3's LOG_STD_MIN=-20 provides no protection
       (exp(-20) ~ 0). CleanRL uses -5.

    Patches:
    - Clamps pre-tanh means to [-mean_clamp, mean_clamp]
    - Re-clamps log_std to [log_std_min, 2.0]
    - Stores actor._last_mean_actions for diagnostics and mean regularization

    Args:
        actor: SB3 actor module
        mean_clamp: Max |pre-tanh mean| (0 = disable clamping)
        log_std_min: Minimum log_std floor (SB3 default: -20)
    """
    import types
    import torch

    if getattr(actor, '_stability_patched', False):
        return

    _orig_get_action_dist_params = actor.get_action_dist_params

    def _patched_get_action_dist_params(self, obs):
        mean_actions, log_std, kwargs = _orig_get_action_dist_params(obs)

        # Clamp pre-tanh means (keep grad via clamp, not detach)
        if mean_clamp > 0:
            mean_actions = torch.clamp(mean_actions, -mean_clamp, mean_clamp)

        # Re-clamp log_std with tighter floor
        log_std = torch.clamp(log_std, log_std_min, 2.0)

        # Store for diagnostics + mean regularization (keeps grad)
        self._last_mean_actions = mean_actions

        return mean_actions, log_std, kwargs

    actor.get_action_dist_params = types.MethodType(
        _patched_get_action_dist_params, actor)
    actor._stability_patched = True
    actor._last_mean_actions = None
    print(f"  Actor patched: mean_clamp={mean_clamp}, log_std_min={log_std_min}")


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


def _create_timed_cls(base_cls):
    """Wrap any SB3 algo class with gradient step timing instrumentation.

    Overrides train() to measure total wall time per gradient step, printing
    and logging to TensorBoard every 1000 gradient steps.  SafeTQC / DualPolicy
    have their own detailed timing; wrapping them here has no effect because
    their train() overrides this one.
    """
    class TimedAlgo(base_cls):
        def train(self, gradient_steps, batch_size=64):
            import time as _t
            _t0 = _t.perf_counter()
            result = super().train(gradient_steps, batch_size)
            elapsed_ms = (_t.perf_counter() - _t0) * 1000
            self._timing_n = getattr(self, '_timing_n', 0) + gradient_steps
            if self._timing_n % 1000 < gradient_steps:
                _n = max(gradient_steps, 1)
                print(
                    f"[TIMING] gradient step: total={elapsed_ms/_n:.1f}ms avg "
                    f"({self._timing_n} grad steps)",
                    flush=True,
                )
                self.logger.record("timing/grad_total_ms", elapsed_ms / _n)
            # Enforce ent_coef floor for plain CrossQ/SAC (SafeTQC/DualPolicy
            # have their own floor in their train() overrides).
            _ent_min = getattr(self, '_ent_coef_min', 0.0)
            if _ent_min > 0 and hasattr(self, 'log_ent_coef') and self.log_ent_coef is not None:
                import math as _math
                self.log_ent_coef.data.clamp_(min=_math.log(_ent_min))
            return result
    TimedAlgo.__name__ = f"Timed{base_cls.__name__}"
    TimedAlgo.__qualname__ = f"Timed{base_cls.__qualname__}"
    return TimedAlgo


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
            import time as _t  # local import — avoids LEGB closure issues with inner classes
            # Switch to train mode
            self.policy.set_training_mode(True)
            is_crossq = _is_crossq_model(self)
            policy_delay = getattr(self, 'policy_delay', 1)

            # Update learning rate
            self._update_learning_rate(
                [self.actor.optimizer, self.critic.optimizer,
                 self.cost_critic_optimizer]
            )

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []
            cost_critic_losses, lagrange_values = [], []
            # Enhanced diagnostics
            _diag_target_q, _diag_current_q = [], []
            _diag_batch_reward, _diag_batch_action_mag = [], []

            # Per-call timing accumulators
            _ta_total = 0.0
            _ta_critic = 0.0
            _ta_actor = 0.0

            for _ in range(gradient_steps):
                _t_iter0 = _t.perf_counter()
                if is_crossq:
                    self._n_updates += 1

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

                # Read ent_coef before critic update (no actor forward pass yet)
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coefs.append(ent_coef.item())

                # --- Reward critic update ---
                _t_critic0 = _t.perf_counter()
                with torch.no_grad():
                    _set_bn_mode(self.actor, False)
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    next_log_prob = next_log_prob.reshape(-1, 1)

                if is_crossq:
                    # CrossQ joint forward pass: BatchRenorm sees obs+next_obs mixture
                    _backup_ent = getattr(self, '_backup_entropy', True)
                    with torch.no_grad():
                        next_q_target = torch.min(
                            torch.cat(self.critic(
                                replay_data.next_observations, next_actions
                            ), dim=1), dim=1, keepdim=True
                        )[0]
                        _ent_backup = ent_coef * next_log_prob if _backup_ent else 0.0
                        target_q = replay_data.rewards + (
                            1 - replay_data.dones
                        ) * self.gamma * (next_q_target.detach() - _ent_backup)

                    all_obs = torch.cat([replay_data.observations, replay_data.next_observations], dim=0)
                    all_actions = torch.cat([replay_data.actions, next_actions], dim=0)
                    _set_bn_mode(self.critic, True)
                    all_q_values = torch.cat(
                        self.critic(all_obs, all_actions), dim=1
                    )  # (2*B, n_critics)
                    _set_bn_mode(self.critic, False)
                    current_q_values, _ = torch.split(all_q_values, batch_size, dim=0)
                    critic_loss = 0.5 * sum(
                        F.mse_loss(current_q_values[:, i:i+1], target_q)
                        for i in range(current_q_values.shape[1])
                    )
                else:
                    # TQC/SAC: separate forward passes (existing code)
                    with torch.no_grad():
                        # Compute target Q-values
                        _critic_for_target = getattr(self, 'critic_target', None) or self.critic
                        _qc_target = _get_tqc_quantile_critics(_critic_for_target)
                        if _qc_target is not None:
                            # TQC: iterate quantile critics, sort & drop top quantiles
                            next_quantiles = []
                            for critic_net in _qc_target:
                                features = _critic_for_target.features_extractor(
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
                            next_q_values = _critic_for_target(
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

                        _backup_ent = getattr(self, '_backup_entropy', True)
                        _ent_backup = ent_coef * next_log_prob if _backup_ent else 0.0
                        target_q = replay_data.rewards + (
                            1 - replay_data.dones
                        ) * self.gamma * (next_q - _ent_backup)

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

                # Accumulate Q-value and batch diagnostics
                with torch.no_grad():
                    _diag_target_q.append(target_q.mean().item())
                    _diag_batch_reward.append(
                        replay_data.rewards.mean().item())
                    _diag_batch_action_mag.append(
                        replay_data.actions.abs().mean().item())
                    if is_crossq:
                        _diag_current_q.append(
                            current_q_values.mean().item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                _ta_critic += (_t.perf_counter() - _t_critic0) * 1000

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

                # --- Actor + entropy update (gated by policy_delay for CrossQ) ---
                if self._n_updates % policy_delay == 0 or not is_crossq:
                    _t_actor0 = _t.perf_counter()
                    _set_bn_mode(self.actor, True)
                    actions_pi, log_prob = self.actor.action_log_prob(
                        replay_data.observations
                    )
                    log_prob = log_prob.reshape(-1, 1)
                    _set_bn_mode(self.actor, False)

                    # Entropy coefficient update
                    ent_coef_loss = -(
                        self.log_ent_coef * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.item())

                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                    # ent_coef floor: prevent entropy death spiral
                    _ent_min = getattr(self, '_ent_coef_min', 0.0)
                    if _ent_min > 0:
                        import math as _math
                        self.log_ent_coef.data.clamp_(min=_math.log(_ent_min))

                    # Q-values for current actions
                    _qc_actor = _get_tqc_quantile_critics(self.critic)
                    if is_crossq:
                        _set_bn_mode(self.critic, False)
                        q_pi_values = self.critic(replay_data.observations, actions_pi)
                        qf_pi = torch.min(
                            torch.cat(q_pi_values, dim=1), dim=1, keepdim=True
                        )[0]
                    elif _qc_actor is not None:
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

                    # Mean regularization: penalize large pre-tanh means
                    _mr = getattr(self, '_mean_reg', 0.0)
                    if _mr > 0 and getattr(self.actor, '_last_mean_actions', None) is not None:
                        _mr_loss = _mr * self.actor._last_mean_actions.pow(2).mean()
                        actor_loss = actor_loss + _mr_loss

                    actor_losses.append(actor_loss.item())

                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()
                    _ta_actor += (_t.perf_counter() - _t_actor0) * 1000

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
                # CrossQ has no reward critic target; TQC/SAC need polyak
                if hasattr(self, 'critic_target') and self.critic_target is not None:
                    _cost_tau = getattr(self, '_cost_tau', self.tau)
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.tau
                    )
                else:
                    _cost_tau = 0.005  # default for cost critic when base has no tau
                if cost_data is not None:
                    polyak_update(
                        self.cost_critic.parameters(),
                        self.cost_critic_target.parameters(),
                        _cost_tau
                    )
                _ta_total += (_t.perf_counter() - _t_iter0) * 1000

            # Timing: print + log to TensorBoard every 1000 gradient steps
            self._timing_n = getattr(self, '_timing_n', 0) + gradient_steps
            if self._timing_n % 1000 < gradient_steps:
                _n = max(gradient_steps, 1)
                _other = max(0.0, _ta_total / _n - _ta_critic / _n - _ta_actor / _n)
                print(
                    f"[TIMING] gradient step: total={_ta_total/_n:.1f}ms | "
                    f"critic={_ta_critic/_n:.1f}ms | "
                    f"actor={_ta_actor/_n:.1f}ms | "
                    f"other(sample+lagrange+polyak)={_other:.1f}ms",
                    flush=True,
                )
                self.logger.record("timing/grad_total_ms", _ta_total / _n)
                self.logger.record("timing/grad_critic_ms", _ta_critic / _n)
                self.logger.record("timing/grad_actor_ms", _ta_actor / _n)

            # Logging
            if not is_crossq:
                self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            if actor_losses:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            if ent_coef_losses:
                self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            if cost_critic_losses:
                self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))
            if lagrange_values:
                self.logger.record("train/lagrange", np.mean(lagrange_values))
            # Enhanced Q-value and batch diagnostics
            if _diag_target_q:
                self.logger.record("train/target_q_mean",
                                   np.mean(_diag_target_q))
            if _diag_current_q:
                self.logger.record("train/current_q_mean",
                                   np.mean(_diag_current_q))
            if _diag_batch_reward:
                self.logger.record("train/batch_reward_mean",
                                   np.mean(_diag_batch_reward))
            if _diag_batch_action_mag:
                self.logger.record("train/batch_action_mag",
                                   np.mean(_diag_batch_action_mag))
            # Pre-tanh mean magnitude and mean regularization loss
            _last_mu = getattr(self.actor, '_last_mean_actions', None)
            if _last_mu is not None:
                _mu_abs = _last_mu.detach().abs().mean().item()
                self.logger.record("train/mean_mu_abs", _mu_abs)
                _mr = getattr(self, '_mean_reg', 0.0)
                if _mr > 0:
                    self.logger.record("train/mean_reg_loss",
                                       _mr * _last_mu.detach().pow(2).mean().item())

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
            excluded = set(super()._excluded_save_params())
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
            is_crossq = _is_crossq_model(self)
            policy_delay = getattr(self, 'policy_delay', 1)

            # Update learning rate
            self._update_learning_rate(
                [self.actor.optimizer, self.critic.optimizer]
            )

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses = [], []
            il_selection_rates = []

            for _ in range(gradient_steps):
                if is_crossq:
                    self._n_updates += 1

                # Sample replay buffer
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )

                # Read ent_coef before critic update (no actor forward pass yet)
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coefs.append(ent_coef.item())

                # --- IBRL TD target ---
                with torch.no_grad():
                    next_obs = replay_data.next_observations
                    B = next_obs.shape[0]

                    # RL next action + log prob
                    _set_bn_mode(self.actor, False)
                    next_actions_rl, next_log_prob_rl = self.actor.action_log_prob(
                        next_obs
                    )
                    next_log_prob_rl = next_log_prob_rl.reshape(-1, 1)

                    # IL next action (frozen, deterministic)
                    next_actions_il = self.il_actor._predict(
                        next_obs, deterministic=True
                    )

                if is_crossq:
                    # CrossQ 3x-batch joint forward pass: BatchRenorm sees
                    # obs + next_obs(RL) + next_obs(IL) mixture distribution
                    with torch.no_grad():
                        # Compute IBRL max(Q_RL_adj, Q_IL) for TD targets
                        # Use separate forward for target computation (detached)
                        next_q_rl_raw = self.critic(next_obs, next_actions_rl)
                        next_q_il_raw = self.critic(next_obs, next_actions_il)

                        def _min_q_crossq(q_raw):
                            if isinstance(q_raw, (list, tuple)):
                                return torch.min(
                                    torch.cat(q_raw, dim=1), dim=1, keepdim=True
                                )[0]
                            return q_raw

                        q_rl_adj = _min_q_crossq(next_q_rl_raw) - ent_coef * next_log_prob_rl
                        q_il_v = _min_q_crossq(next_q_il_raw)
                        next_q = torch.max(q_rl_adj, q_il_v)

                        use_il = (q_il_v > q_rl_adj).squeeze(1)
                        il_selection_rates.append(use_il.float().mean().item())

                        target_q = replay_data.rewards + (
                            1 - replay_data.dones
                        ) * self.gamma * next_q

                    # Joint forward pass through critic with BN training mode
                    all_obs = torch.cat([replay_data.observations, next_obs, next_obs], dim=0)
                    all_actions = torch.cat([replay_data.actions, next_actions_rl, next_actions_il], dim=0)
                    _set_bn_mode(self.critic, True)
                    all_q_values = torch.cat(
                        self.critic(all_obs, all_actions), dim=1
                    )  # (3*B, n_critics)
                    _set_bn_mode(self.critic, False)
                    current_q_values = all_q_values[:B]  # first B rows = current obs
                    critic_loss = 0.5 * sum(
                        F.mse_loss(current_q_values[:, i:i+1], target_q)
                        for i in range(current_q_values.shape[1])
                    )
                else:
                    # TQC/SAC: separate forward passes (existing code)
                    with torch.no_grad():
                        _critic_for_target = getattr(self, 'critic_target', None) or self.critic
                        _qc_target = _get_tqc_quantile_critics(_critic_for_target)
                        if _qc_target is not None:
                            # TQC quantile path: compute features once, reuse
                            feats = _critic_for_target.features_extractor(next_obs)
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
                            next_q_rl = _critic_for_target(next_obs, next_actions_rl)
                            next_q_il = _critic_for_target(next_obs, next_actions_il)

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

                # --- Actor + entropy update (gated by policy_delay for CrossQ) ---
                if self._n_updates % policy_delay == 0 or not is_crossq:
                    _set_bn_mode(self.actor, True)
                    actions_pi, log_prob = self.actor.action_log_prob(
                        replay_data.observations
                    )
                    log_prob = log_prob.reshape(-1, 1)
                    _set_bn_mode(self.actor, False)

                    # Entropy coefficient update
                    ent_coef_loss = -(
                        self.log_ent_coef * (log_prob + self.target_entropy).detach()
                    ).mean()
                    ent_coef_losses.append(ent_coef_loss.item())

                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                    # ent_coef floor: prevent entropy death spiral
                    _ent_min = getattr(self, '_ent_coef_min', 0.0)
                    if _ent_min > 0:
                        import math as _math
                        self.log_ent_coef.data.clamp_(min=_math.log(_ent_min))

                    # Q-values for current actions
                    _qc_actor = _get_tqc_quantile_critics(self.critic)
                    if is_crossq:
                        _set_bn_mode(self.critic, False)
                        q_pi_values = self.critic(replay_data.observations, actions_pi)
                        qf_pi = torch.min(
                            torch.cat(q_pi_values, dim=1), dim=1, keepdim=True
                        )[0]
                    elif _qc_actor is not None:
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

                    # Mean regularization: penalize large pre-tanh means
                    _mr = getattr(self, '_mean_reg', 0.0)
                    if _mr > 0 and getattr(self.actor, '_last_mean_actions', None) is not None:
                        _mr_loss = _mr * self.actor._last_mean_actions.pow(2).mean()
                        actor_loss = actor_loss + _mr_loss

                    actor_losses.append(actor_loss.item())

                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer.step()

                # --- Target network update ---
                if hasattr(self, 'critic_target') and self.critic_target is not None:
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.tau,
                    )

            # Logging
            if not is_crossq:
                self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/ent_coef", np.mean(ent_coefs))
            if actor_losses:
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
    parser.add_argument('--utd-ratio', type=int, default=1,
                        help='Update-to-data ratio / gradient steps per env step (default: 1)')
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
    parser.add_argument('--legacy-tqc', action='store_true',
                        help='Use TQC instead of CrossQ (legacy mode)')
    parser.add_argument('--policy-delay', type=int, default=1,
                        help='Actor update delay (default: 1; CrossQ paper uses 20 with UTD=20)')
    parser.add_argument('--ent-coef', type=str, default='auto_0.006',
                        help='Entropy coefficient, "auto" or "auto_<init>" for learned (default: auto_0.006)')
    parser.add_argument('--log-std-init', type=float, default=-2.0,
                        help='Actor log_std to set after CVAE pretraining or on --resume. '
                             '-2.0 (default) keeps the CVAE/checkpoint value; the new stability '
                             'fixes (--mean-clamp, --log-std-min, --ent-coef-min) prevent entropy '
                             'collapse without inflating log_std. Pass -0.5 (std=0.61) for the '
                             'old behavior of boosting exploration noise. '
                             '(default: -2.0 = keep CVAE/checkpoint value)')
    parser.add_argument('--ent-coef-init', type=float, default=0.1,
                        help='ent_coef value to set after CVAE pretraining or on resume. '
                             'CVAE leaves ent_coef too small (~0.006) giving negligible entropy '
                             'bonus vs Q-values. 0.1 gives meaningful regularization. '
                             '(default: 0.1, set 0 to disable)')
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
    parser.add_argument('--n-frames', type=int, default=1,
                        help='Number of observations to stack (1 = no stacking, default: 1)')
    parser.add_argument('--gru-hidden', type=int, default=128,
                        help='GRU hidden dimension (only used when --n-frames > 1, default: 128)')
    parser.add_argument('--gru-lr', type=float, default=1e-5,
                        help='GRU learning rate (only used when --n-frames > 1, default: 1e-5)')

    # Actor stability arguments
    stab_group = parser.add_argument_group('Actor stability')
    stab_group.add_argument('--mean-clamp', type=float, default=3.0,
                            help='Clamp |pre-tanh mean| to this value (0=disable). '
                                 'Prevents tanh saturation (tanh(3)=0.995). (default: 3.0)')
    stab_group.add_argument('--mean-reg', type=float, default=0.001,
                            help='L2 regularization weight on pre-tanh means (0=disable). '
                                 'Penalizes large means to keep actions away from tanh saturation. '
                                 '(default: 0.001)')
    stab_group.add_argument('--log-std-min', type=float, default=-5.0,
                            help='Minimum log_std floor (SB3 default: -20, CleanRL: -5). '
                                 'exp(-5)=0.007 prevents near-zero std. (default: -5.0)')
    stab_group.add_argument('--ent-coef-min', type=float, default=0.005,
                            help='Floor for ent_coef (0=disable). Prevents entropy death spiral '
                                 'by clamping log_ent_coef >= log(ent_coef_min). (default: 0.005)')
    stab_group.add_argument('--target-entropy', type=float, default=None,
                            help='Target entropy for SAC auto-tuner. Default: -chunk_size '
                                 '(RLPD heuristic). For tanh-squashed chunked actions, the '
                                 'actual entropy at std~0.6 is ~+10 nats, so -chunk_size '
                                 'is unreachable and permanently pins ent_coef at the floor. '
                                 'Try 0 or +5 for a reachable equilibrium.')
    stab_group.add_argument('--no-backup-entropy', action='store_true',
                            help='Remove entropy term from TD target (RLPD-style). '
                                 'Reduces entropy-driven Q instability at the cost of slightly '
                                 'less exploration incentive.')

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
    parser.add_argument('--add-prev-action', action='store_true',
                        help='Add previous action to observations (34D -> 36D)')
    parser.add_argument('--use-camera', action='store_true',
                        help='Enable DINOv2 camera features in observations (34D -> 418D)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Save checkpoint every N steps (default: 50000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .zip to resume training from')

    # Logging arguments
    parser.add_argument('--more-debug', action='store_true',
                        help='Print per-episode stats')

    # Output arguments
    parser.add_argument('--output', type=str, default='models/crossq_jetbot.zip',
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
    _te = args.target_entropy if args.target_entropy is not None else -float(args.chunk_size)
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Target entropy: {_te} ({'--target-entropy' if args.target_entropy is not None else f'-chunk_size={args.chunk_size}'})")
    print(f"  Demo ratio: {args.demo_ratio}")
    print(f"  Learning starts: {args.learning_starts}")
    print(f"  CVAE: z_dim={args.cvae_z_dim}, epochs={args.cvae_epochs}, "
          f"beta={args.cvae_beta}, lr={args.cvae_lr}")
    if args.n_frames > 1:
        print(f"  Frame stacking: n_frames={args.n_frames}, gru_hidden={args.gru_hidden}, gru_lr={args.gru_lr}")
    if args.use_camera:
        print(f"  Camera: DINOv2 ViT-S/14 (384D features)")
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
    from jetbot_rl_env import JetbotNavigationEnv, ChunkedEnvWrapper, FrameStackWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from demo_utils import make_chunk_transitions

    # If resuming, peek at checkpoint to resolve chunk_size BEFORE creating the env.
    # The env is built with args.chunk_size, so we must reconcile any mismatch early.
    if args.resume:
        from stable_baselines3.common.save_util import load_from_zip_file as _peek_zip
        _peek_data, _, _ = _peek_zip(args.resume, device="cpu")
        _ckpt_action_space = _peek_data.get("action_space", None)
        if _ckpt_action_space is not None:
            _ckpt_chunk = _ckpt_action_space.shape[0] // 2
            if _ckpt_chunk != args.chunk_size:
                print(f"  [Resume] Checkpoint chunk_size={_ckpt_chunk} overrides --chunk-size={args.chunk_size}")
                args.chunk_size = _ckpt_chunk
        del _peek_zip, _peek_data, _ckpt_action_space

    # Algorithm selection: CrossQ (default) > TQC (--legacy-tqc) > SAC (fallback)
    if args.legacy_tqc:
        try:
            from sb3_contrib import TQC
            algo_cls = TQC
            algo_name = "TQC"
        except ImportError:
            from stable_baselines3 import SAC
            algo_cls = SAC
            algo_name = "SAC"
            print("sb3-contrib not found, falling back to SAC")
    else:
        try:
            from sb3_contrib import CrossQ
            algo_cls = CrossQ
            algo_name = "CrossQ"
        except ImportError:
            try:
                from sb3_contrib import TQC
                algo_cls = TQC
                algo_name = "TQC"
                print("CrossQ not available, falling back to TQC")
            except ImportError:
                from stable_baselines3 import SAC
                algo_cls = SAC
                algo_name = "SAC"
                print("sb3-contrib not found, falling back to SAC")

    # Mutual exclusion
    if args.dual_policy and args.safe:
        parser.error("--dual-policy and --safe are mutually exclusive")

    # Always wrap with timing instrumentation (SafeTQC/DualPolicy override train()
    # with their own detailed timing, so this only fires for plain CrossQ/TQC/SAC)
    algo_cls = _create_timed_cls(algo_cls)

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
        add_prev_action=getattr(args, 'add_prev_action', False),
        use_camera=getattr(args, 'use_camera', False),
    )
    if args.n_frames > 1:
        base_obs_dim = raw_env.observation_space.shape[0]
        raw_env = FrameStackWrapper(raw_env, n_frames=args.n_frames)
        print(f"  FrameStackWrapper: n_frames={args.n_frames}, obs {base_obs_dim} -> {raw_env.observation_space.shape[0]}")
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
    _load_images = getattr(args, 'use_camera', False)
    if args.safe:
        demo_obs_step, demo_actions_step, demo_rewards_step, _, demo_dones_step, demo_costs_step = \
            load_demo_transitions(args.demos, load_costs=True, load_images=_load_images)
    else:
        demo_obs_step, demo_actions_step, demo_rewards_step, _, demo_dones_step = \
            load_demo_transitions(args.demos, load_images=_load_images)
    from demo_io import open_demo
    demo_data = open_demo(args.demos)
    episode_lengths = demo_data['episode_lengths']

    # Recompute demo rewards with current RewardComputer to match RL env
    # (demos may have been recorded with a different reward function)
    from jetbot_keyboard_control import RewardComputer
    _rc = RewardComputer(mode=args.reward_mode,
                         safe_mode=args.safe and not args.keep_proximity_reward)
    old_mean = float(demo_rewards_step.mean())
    _N = len(demo_rewards_step)
    _lidar_start = demo_obs_step.shape[1] - 24  # dynamic split: last 24 dims are LiDAR
    # Build next_obs for reward computation (shift within episodes)
    _next_obs = np.zeros_like(demo_obs_step)
    _off = 0
    for _el in episode_lengths:
        _el = int(_el)
        _next_obs[_off:_off + _el - 1] = demo_obs_step[_off + 1:_off + _el]
        _next_obs[_off + _el - 1] = demo_obs_step[_off + _el - 1]
        _off += _el
    for _i in range(_N):
        _obs_i = demo_obs_step[_i]
        _nobs_i = _next_obs[_i]
        _min_lidar = float(_nobs_i[_lidar_start:].min()) * 3.0  # denormalize
        _collision = bool(demo_dones_step[_i]) and _min_lidar < 0.08
        _goal_reached = bool(_nobs_i[9] > 0.5) if _nobs_i.shape[0] > 9 else False
        _info = {
            'collision': _collision,
            'goal_reached': _goal_reached,
            'min_lidar_distance': _min_lidar,
        }
        demo_rewards_step[_i] = _rc.compute(_obs_i, demo_actions_step[_i], _nobs_i, _info)
    new_mean = float(demo_rewards_step.mean())
    print(f"  Recomputed demo rewards: mean {old_mean:.4f} -> {new_mean:.4f}")

    # Frame-stack demo observations if using temporal processing
    if args.n_frames > 1:
        from demo_utils import build_frame_stacks
        print(f"Frame-stacking demo observations: {demo_obs_step.shape[1]}D -> "
              f"{args.n_frames * demo_obs_step.shape[1]}D")
        demo_obs_step = build_frame_stacks(demo_obs_step, episode_lengths, args.n_frames)
        print(f"  Stacked obs shape: {demo_obs_step.shape}")

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
    if args.n_frames > 1:
        obs_feature_dim = args.gru_hidden
    elif getattr(args, 'use_camera', False):
        obs_feature_dim = 160  # 32 + 64 + 64 (state + image + lidar)
    else:
        obs_feature_dim = 96
    features_dim = obs_feature_dim + args.cvae_z_dim

    # Create or resume model
    _t0 = _time.time()
    if args.resume:
        print(f"Resuming {algo_name} model from {args.resume}...")
        from stable_baselines3.common.save_util import (
            load_from_zip_file, recursive_getattr,
        )
        data, params, pytorch_vars = load_from_zip_file(
            args.resume, device=device_str,
        )
        # Reconstruct model from saved hyperparams (mirrors SB3 load())
        model = algo_cls(
            policy=data["policy_class"],
            env=env,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
            _init_setup_model=False,
        )
        model.__dict__.update(data)
        model._setup_model()  # creates 1-group optimizers
        # Re-apply GRU split-LR BEFORE loading optimizer state dicts so the
        # optimizer structure (2 param groups) matches the checkpoint layout.
        if args.n_frames > 1:
            _apply_gru_lr(model, gru_lr=args.gru_lr)
        # Load all state dicts including optimizers (per-key try/except)
        state_names, var_names = model._get_torch_save_params()
        loaded_keys, failed_keys = [], []
        for name in state_names:
            if name in params:
                attr = recursive_getattr(model, name)
                try:
                    attr.load_state_dict(params[name])
                    loaded_keys.append(name)
                except (ValueError, RuntimeError) as e:
                    failed_keys.append(name)
                    print(f"  Warning: could not load {name}: {e}")
            else:
                failed_keys.append(name + " (missing)")
        print(f"  Loaded state dicts: {loaded_keys}")
        if failed_keys:
            print(f"  Failed state dicts: {failed_keys}")
        # Load pytorch variables (log_ent_coef, log_lagrange, etc.)
        # pytorch_vars is a dict {name: tensor}, not a list
        if pytorch_vars is not None and isinstance(pytorch_vars, dict):
            for name in var_names:
                if name in pytorch_vars:
                    attr = recursive_getattr(model, name)
                    attr.data = pytorch_vars[name]
                    print(f"  Restored variable: {name} = {pytorch_vars[name].item():.6f}")
                else:
                    print(f"  Variable {name} not in checkpoint, using default")
        elif pytorch_vars is not None:
            # Fallback for older SB3 list format
            for name, val in zip(var_names, pytorch_vars):
                attr = recursive_getattr(model, name)
                attr.data = val
                print(f"  Restored variable: {name}")
        # Verify chunk size matches (should already match after early reconciliation above)
        loaded_chunk = model.action_space.shape[0] // 2
        if loaded_chunk != args.chunk_size:
            raise RuntimeError(
                f"Chunk size mismatch after resume: model={loaded_chunk}, env={args.chunk_size}. "
                f"This should not happen — please report a bug."
            )
        # Auto-detect n_frames from loaded model obs space
        loaded_obs_dim = model.observation_space.shape[0]
        if loaded_obs_dim > 34 and args.n_frames == 1:
            detected_n_frames = loaded_obs_dim // 34
            print(f"  Auto-detected n_frames={detected_n_frames} from model obs_dim={loaded_obs_dim}")
            args.n_frames = detected_n_frames
        # Override mutable hyperparams the user may have changed
        model.learning_rate = args.lr
        model.batch_size = args.batch_size
        model.gamma = effective_gamma
        if not _is_crossq_model(model):
            model.tau = args.tau
        model.gradient_steps = args.utd_ratio
        model.ent_coef = ent_coef
        # On resume: the checkpoint may have collapsed ent_coef (~0.002) and/or
        # collapsed log_std (~-1.76) from a previous entropy death spiral.
        # Reset both if the corresponding flags are set.
        import math as _math
        import torch as _torch
        if args.log_std_init != -2.0:  # -2.0 = keep checkpoint value
            ls = model.actor.log_std
            old_std = float(ls.bias.data.mean().exp()) if hasattr(ls, 'bias') else float(ls.data.mean().exp())
            with _torch.no_grad():
                if hasattr(ls, 'bias'):
                    ls.bias.data.fill_(args.log_std_init)
                    ls.weight.data.zero_()
                else:
                    ls.data.fill_(args.log_std_init)
            print(f"  log_std reset on resume: {_math.log(old_std):.3f} -> {args.log_std_init:.2f} "
                  f"(std {old_std:.3f} -> {_math.exp(args.log_std_init):.3f})  "
                  f"[pass --log-std-init -2.0 to keep checkpoint value]")
        if args.ent_coef_init > 0 and hasattr(model, 'log_ent_coef'):
            old_val = float(model.log_ent_coef.exp())
            with _torch.no_grad():
                model.log_ent_coef.data.fill_(_math.log(args.ent_coef_init))
            print(f"  ent_coef reset on resume: {old_val:.5f} -> {args.ent_coef_init:.4f}  "
                  f"[pass --ent-coef-init 0 to keep checkpoint value]")
        # Actor stability fixes
        model._mean_reg = args.mean_reg
        model._ent_coef_min = args.ent_coef_min
        model._backup_entropy = not args.no_backup_entropy
        _patch_actor_for_stability(model.actor, args.mean_clamp, args.log_std_min)
        print(f"  Stability: mean_clamp={args.mean_clamp}, mean_reg={args.mean_reg}, "
              f"log_std_min={args.log_std_min}, ent_coef_min={args.ent_coef_min}, "
              f"backup_entropy={not args.no_backup_entropy}")
        # Replace replay buffer with fresh demo buffer (online data is lost)
        model.replay_buffer = replay_buffer
        # Force initial env.reset() — __dict__.update(data) restores a stale
        # _last_obs from the checkpoint which makes SB3 skip reset(), leaving
        # the new env's _prev_obs as None.
        model._last_obs = None
        # LayerNorm + CVAE weights are already baked into the loaded checkpoint
        print(f"  Model resumed in {_time.time() - _t0:.1f}s")
    else:
        print(f"Creating {algo_name} model...")
        if args.n_frames > 1:
            fe_cls = TemporalCVAEFeatureExtractor.get_class(
                BaseFeaturesExtractor, n_frames=args.n_frames,
                gru_hidden_dim=args.gru_hidden, z_dim=args.cvae_z_dim)
        elif getattr(args, 'use_camera', False):
            fe_cls = VisionCVAEFeatureExtractor.get_class(
                BaseFeaturesExtractor, z_dim=args.cvae_z_dim)
        else:
            fe_cls = ChunkCVAEFeatureExtractor.get_class(
                BaseFeaturesExtractor, z_dim=args.cvae_z_dim)
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=torch.nn.ReLU,
            features_extractor_class=fe_cls,
            features_extractor_kwargs=dict(features_dim=features_dim),
        )
        is_crossq = (algo_name == "CrossQ" or algo_name.endswith("CrossQ"))
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
            ent_coef=ent_coef,
            target_entropy=args.target_entropy if args.target_entropy is not None else -float(args.chunk_size),
            gradient_steps=args.utd_ratio,
            learning_starts=args.learning_starts,
            train_freq=1,
            policy_kwargs=policy_kwargs,
        )
        # CrossQ has no target networks (tau is internal); TQC/SAC need tau for polyak
        if not is_crossq:
            model_kwargs['tau'] = args.tau
        # CrossQ supports policy_delay natively
        if is_crossq and args.policy_delay > 1:
            model_kwargs['policy_delay'] = args.policy_delay
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
        # Apply separate GRU learning rate
        if args.n_frames > 1:
            _apply_gru_lr(model, gru_lr=args.gru_lr)
        print(f"  Model created in {_time.time() - _t0:.1f}s")
    print()

    # CVAE pretraining (replaces BC warmstart)
    if not args.resume:
        pretrain_chunk_cvae(
            model, demo_obs_step, demo_actions_step, episode_lengths,
            chunk_size=args.chunk_size, z_dim=args.cvae_z_dim,
            epochs=args.cvae_epochs, batch_size=args.batch_size,
            lr=args.cvae_lr, beta=args.cvae_beta, gamma=args.gamma,
            gru_lr=args.gru_lr if args.n_frames > 1 else None,
        )
        # Copy pretrained FE weights to cost critic too
        if args.safe and hasattr(model, 'cost_critic'):
            fe_state = model.actor.features_extractor.state_dict()
            model.cost_critic.features_extractor.load_state_dict(fe_state)
            model.cost_critic_target.features_extractor.load_state_dict(fe_state)
            print("  Feature extractor weights copied to cost critic/cost_critic_target")

        # --- Post-CVAE entropy fixes ---
        # CVAE pretraining sets log_std.bias=-2.0 (std=0.135) which collapses entropy below
        # target_entropy before SAC even starts. With ent_coef~0.006 the entropy bonus is
        # ~0.006*(-10)=-0.06, negligible vs Q-values of ±10-20. Both must be reset.
        # (RLPD, Ball et al. ICML 2023: "standard ent_coef init fails when policy is near-deterministic")
        import math as _math
        import torch as _torch

        if args.log_std_init != -2.0:  # -2.0 = keep CVAE value
            with _torch.no_grad():
                ls = model.actor.log_std
                if hasattr(ls, 'bias'):
                    ls.bias.data.fill_(args.log_std_init)
                    ls.weight.data.zero_()
                else:
                    ls.data.fill_(args.log_std_init)
            print(f"  log_std reset: -2.00 -> {args.log_std_init:.2f} "
                  f"(std 0.135 -> {_math.exp(args.log_std_init):.3f})  "
                  f"[pass --log-std-init -2.0 to keep CVAE value]")

        if args.ent_coef_init > 0 and hasattr(model, 'log_ent_coef'):
            with _torch.no_grad():
                model.log_ent_coef.data.fill_(_math.log(args.ent_coef_init))
            print(f"  ent_coef reset: 0.006 -> {args.ent_coef_init:.4f} "
                  f"(log_ent_coef={_math.log(args.ent_coef_init):.2f})  "
                  f"[pass --ent-coef-init 0 to disable]")

        # Actor stability fixes
        model._mean_reg = args.mean_reg
        model._ent_coef_min = args.ent_coef_min
        model._backup_entropy = not args.no_backup_entropy
        _patch_actor_for_stability(model.actor, args.mean_clamp, args.log_std_min)
        print(f"  Stability: mean_clamp={args.mean_clamp}, mean_reg={args.mean_reg}, "
              f"log_std_min={args.log_std_min}, ent_coef_min={args.ent_coef_min}, "
              f"backup_entropy={not args.no_backup_entropy}")

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
        base = "tqc" if "TQC" in algo_name else ("crossq" if "CrossQ" in algo_name else "sac")
        prefix = f"safe_{base}_jetbot"
    elif "DualPolicy" in algo_name:
        base = "tqc" if "TQC" in algo_name else ("crossq" if "CrossQ" in algo_name else "sac")
        prefix = f"dual_{base}_jetbot"
    else:
        prefix = {"TQC": "tqc_jetbot", "CrossQ": "crossq_jetbot", "SAC": "sac_jetbot"}.get(
            algo_name, "sac_jetbot")
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
