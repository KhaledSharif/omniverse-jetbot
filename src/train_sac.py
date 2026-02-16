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


def validate_demo_data(filepath: str) -> dict:
    """Validate demonstration data meets minimum requirements for training.

    Args:
        filepath: Path to .npz demo file

    Returns:
        dict with keys: episodes, transitions, successful, avg_return

    Raises:
        ValueError: If data doesn't meet minimum requirements
    """
    data = np.load(filepath, allow_pickle=True)

    episode_lengths = data['episode_lengths']
    num_episodes = len(episode_lengths)
    total_transitions = int(np.sum(episode_lengths))

    num_successful = 0
    if 'episode_success' in data:
        num_successful = int(np.sum(data['episode_success']))

    rewards = data['rewards']
    avg_return = 0.0
    if total_transitions > 0:
        offset = 0
        returns = []
        for length in episode_lengths:
            ep_return = float(np.sum(rewards[offset:offset + length]))
            returns.append(ep_return)
            offset += length
        avg_return = float(np.mean(returns))

    summary = dict(
        episodes=num_episodes,
        transitions=total_transitions,
        successful=num_successful,
        avg_return=avg_return,
    )

    print(f"\n{'=' * 60}")
    print("Demo Data Validation")
    print(f"{'=' * 60}")
    print(f"  File: {filepath}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Transitions: {total_transitions}")
    print(f"  Successful: {num_successful}")
    print(f"  Avg return: {avg_return:.2f}")
    print(f"{'=' * 60}")

    if num_episodes < 10:
        raise ValueError(
            f"Need >= 10 episodes, got {num_episodes}. Record more demos."
        )
    if total_transitions < 500:
        raise ValueError(
            f"Need >= 500 transitions, got {total_transitions}. Record more demos."
        )
    if 'episode_success' in data and num_successful < 3:
        raise ValueError(
            f"Need >= 3 successful episodes, got {num_successful}. Record more successful demos."
        )

    print("  Validation: PASSED\n")
    return summary


def load_demo_transitions(npz_path: str):
    """Load demo data and reconstruct (obs, action, reward, next_obs, done) tuples.

    Args:
        npz_path: Path to .npz demo file

    Returns:
        Tuple of (obs, actions, rewards, next_obs, dones) numpy arrays
    """
    data = np.load(npz_path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    episode_lengths = data['episode_lengths']

    total = len(observations)
    next_obs = np.zeros_like(observations)
    dones = np.zeros(total, dtype=np.float32)

    offset = 0
    for length in episode_lengths:
        length = int(length)
        for t in range(length - 1):
            next_obs[offset + t] = observations[offset + t + 1]
        # Terminal step: next_obs doesn't matter (masked by done=True)
        next_obs[offset + length - 1] = observations[offset + length - 1]
        dones[offset + length - 1] = 1.0
        offset += length

    print(f"Loaded {len(episode_lengths)} demo episodes, {total} transitions")
    return observations, actions, rewards, next_obs, dones


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

            if online_batch_size == 0:
                return demo_samples

            # Sample from online buffer
            online_samples = super().sample(online_batch_size, env=env)

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
    """Post-hoc inject LayerNorm after hidden Linear layers in critic networks.

    Handles both TQC (quantile_critics) and SAC (critic.qf0/qf1) structures.
    After injection, re-syncs critic_target and recreates the critic optimizer.
    """
    import torch.nn as nn

    def _inject_layernorm(sequential):
        """Insert LayerNorm after each hidden Linear layer in an nn.Sequential."""
        new_modules = []
        modules = list(sequential)
        for i, module in enumerate(modules):
            new_modules.append(module)
            if isinstance(module, nn.Linear):
                # Add LayerNorm after hidden Linear layers (not the output layer)
                # Output layer is typically the last Linear
                remaining = modules[i + 1:]
                has_more_linear = any(isinstance(m, nn.Linear) for m in remaining)
                if has_more_linear:
                    new_modules.append(nn.LayerNorm(module.out_features))
        return nn.Sequential(*new_modules)

    is_tqc = hasattr(model.critic, 'quantile_critics')

    if is_tqc:
        for i, critic_net in enumerate(model.critic.quantile_critics):
            model.critic.quantile_critics[i] = _inject_layernorm(critic_net)
        for i, critic_net in enumerate(model.critic_target.quantile_critics):
            model.critic_target.quantile_critics[i] = _inject_layernorm(critic_net)
    else:
        for attr in ('qf0', 'qf1'):
            setattr(model.critic, attr, _inject_layernorm(getattr(model.critic, attr)))
            setattr(model.critic_target, attr, _inject_layernorm(getattr(model.critic_target, attr)))

    # Move new parameters to device
    model.critic = model.critic.to(model.device)
    model.critic_target = model.critic_target.to(model.device)

    # Sync target from critic weights
    model.critic_target.load_state_dict(model.critic.state_dict())

    # Recreate critic optimizer to include LayerNorm parameters
    import torch
    model.critic.optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=model.lr_schedule(1)
    )

    print("LayerNorm injected into critic networks")


def bc_warmstart_sac(model, demo_obs, demo_actions, epochs=50, batch_size=256, lr=1e-3):
    """Pretrain SAC/TQC actor on demo actions via behavioral cloning.

    Only trains latent_pi (hidden layers) and mu (mean output) — leaves log_std
    untouched so SAC's entropy tuning can adjust exploration post-BC.

    Args:
        model: SB3 SAC or TQC model
        demo_obs: numpy array of demo observations
        demo_actions: numpy array of demo actions (normalized to [-1, 1])
        epochs: Number of BC epochs
        batch_size: BC mini-batch size
        lr: Learning rate for BC optimizer
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 60)
    print("BC Warmstart (SAC/TQC Actor)")
    print("=" * 60)

    dataset = TensorDataset(
        torch.tensor(demo_obs, dtype=torch.float32),
        torch.tensor(demo_actions, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Only optimize actor mean path (not log_std)
    actor_params = (
        list(model.actor.latent_pi.parameters())
        + list(model.actor.mu.parameters())
    )
    optimizer = torch.optim.Adam(actor_params, lr=lr)
    loss_fn = torch.nn.MSELoss()
    device = model.device

    print(f"  Pretraining actor for {epochs} epochs on {len(dataset)} transitions...")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            mean_actions, _, _ = model.actor.get_action_dist_params(obs_batch)
            pred_actions = torch.tanh(mean_actions)
            loss = loss_fn(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}, Loss: {total_loss / n_batches:.6f}")

    print("BC warmstart complete!")
    print("=" * 60 + "\n")


class VerboseEpisodeCallback:
    """Prints episode stats and periodic step-rate info during training."""

    @staticmethod
    def create(base_callback_cls):
        class _VerboseEpisodeCallback(base_callback_cls):
            def __init__(self):
                super().__init__(verbose=0)
                self._ep_count = 0
                self._ep_reward = 0.0
                self._ep_steps = 0
                self._ep_min_lidar = float('inf')
                self._total_successes = 0
                self._total_collisions = 0
                self._total_truncations = 0
                self._start_time = None
                self._last_report_step = 0
                self._report_interval = 100  # print step rate every N steps

            def _on_training_start(self):
                import time, sys
                self._start_time = time.time()
                print(f"[DEBUG] Training loop started at step {self.num_timesteps}", flush=True)

            def _on_step(self):
                import time

                total_steps = self.num_timesteps
                infos = self.locals.get('infos', [{}])
                rewards = self.locals.get('rewards', [0.0])
                dones = self.locals.get('dones', [False])

                # First step announcement
                if total_steps == 1:
                    print(f"[DEBUG] First env step completed", flush=True)

                # Periodic step-rate report
                if total_steps - self._last_report_step >= self._report_interval:
                    elapsed = time.time() - self._start_time if self._start_time else 0
                    rate = total_steps / elapsed if elapsed > 0 else 0
                    print(
                        f"[DEBUG] step={total_steps:>7d} | "
                        f"elapsed={elapsed:.1f}s | "
                        f"rate={rate:.1f} steps/s | "
                        f"episodes={self._ep_count}",
                        flush=True
                    )
                    self._last_report_step = total_steps

                for i in range(len(dones)):
                    info = infos[i] if i < len(infos) else {}
                    reward = float(rewards[i]) if i < len(rewards) else 0.0

                    self._ep_reward += reward
                    self._ep_steps += 1

                    min_lid = info.get('min_lidar_distance', float('inf'))
                    if min_lid < self._ep_min_lidar:
                        self._ep_min_lidar = min_lid

                    if dones[i]:
                        self._ep_count += 1
                        success = info.get('is_success', False)
                        collision = info.get('collision', False)

                        if success:
                            self._total_successes += 1
                            outcome = "SUCCESS"
                        elif collision:
                            self._total_collisions += 1
                            outcome = "COLLISION"
                        else:
                            self._total_truncations += 1
                            outcome = "TRUNCATED"

                        sr = self._total_successes / self._ep_count * 100
                        elapsed = time.time() - self._start_time if self._start_time else 0

                        print(
                            f"[EP {self._ep_count:4d} END]  "
                            f"{outcome:>9s} | "
                            f"steps={self._ep_steps:3d} | "
                            f"return={self._ep_reward:+7.2f} | "
                            f"min_lidar={self._ep_min_lidar:.3f}m | "
                            f"running SR={sr:.1f}% "
                            f"({self._total_successes}S/{self._total_collisions}C/{self._total_truncations}T) | "
                            f"t={elapsed:.0f}s",
                            flush=True
                        )

                        new_obs = self.locals.get('new_obs')
                        if new_obs is not None:
                            obs = new_obs[i] if len(new_obs) > i else new_obs[0]
                            goal_x, goal_y = obs[5], obs[6]
                            dist = obs[7]
                            heading_deg = float(np.degrees(obs[2]))
                            print(
                                f"[EP {self._ep_count + 1:4d} START] "
                                f"goal=({goal_x:+.2f}, {goal_y:+.2f}) | "
                                f"dist={dist:.2f}m | "
                                f"heading={heading_deg:+.1f}deg",
                                flush=True
                            )

                        self._ep_reward = 0.0
                        self._ep_steps = 0
                        self._ep_min_lidar = float('inf')

                return True

        return _VerboseEpisodeCallback()


def main():
    parser = argparse.ArgumentParser(
        description='Train SAC/TQC agent with RLPD-style demo replay for Jetbot navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demos demos/recording.npz --headless
      Train headless with demo replay

  %(prog)s --demos demos/recording.npz --headless --timesteps 500000
      Train for 500k timesteps

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
    parser.add_argument('--ent-coef', type=str, default='auto',
                        help='Entropy coefficient, "auto" for learned (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--learning-starts', type=int, default=0,
                        help='Steps before training starts (default: 0, demos available immediately)')
    parser.add_argument('--demo-ratio', type=float, default=0.5,
                        help='Fraction of each batch sampled from demos, 0.0-1.0 (default: 0.5)')

    # BC warmstart arguments
    parser.add_argument('--bc-epochs', type=int, default=50,
                        help='Number of BC pretraining epochs (default: 50)')
    parser.add_argument('--bc-batch-size', type=int, default=256,
                        help='BC pretraining batch size (default: 256)')
    parser.add_argument('--bc-lr', type=float, default=1e-3,
                        help='BC pretraining learning rate (default: 1e-3)')

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

    args = parser.parse_args()

    # Validate demo_ratio
    if not 0.0 <= args.demo_ratio <= 1.0:
        parser.error(f"--demo-ratio must be between 0.0 and 1.0, got {args.demo_ratio}")

    # Parse ent_coef
    ent_coef = args.ent_coef
    if ent_coef != 'auto':
        ent_coef = float(ent_coef)

    print("=" * 60)
    print("SAC/TQC + RLPD Training for Jetbot Navigation")
    print("=" * 60)
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Demos: {args.demos}")
    print(f"  UTD ratio: {args.utd_ratio}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Tau: {args.tau}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Demo ratio: {args.demo_ratio}")
    print(f"  Learning starts: {args.learning_starts}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Headless: {args.headless}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps/episode: {args.max_steps}")
    print(f"  Output: {args.output}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    print("=" * 60 + "\n")

    # Validate demo data first (fail fast)
    validate_demo_data(args.demos)

    # Import here to allow --help without Isaac Sim
    import torch
    from jetbot_rl_env import JetbotNavigationEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

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

    print(f"Using algorithm: {algo_name}")

    # Create environment (NO VecNormalize)
    import time as _time
    print("\nCreating environment...")
    _t0 = _time.time()
    half = args.arena_size / 2.0
    workspace_bounds = {'x': [-half, half], 'y': [-half, half]}
    raw_env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
        num_obstacles=args.num_obstacles,
        workspace_bounds=workspace_bounds,
        max_episode_steps=args.max_steps,
        min_goal_dist=args.min_goal,
    )
    print(f"  Environment created in {_time.time() - _t0:.1f}s")
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space: {raw_env.action_space.shape}")

    env = DummyVecEnv([lambda: raw_env])
    print("  No VecNormalize (using LayerNorm in critics instead)")
    print()

    # Load demo transitions
    print("Loading demo transitions...")
    demo_obs, demo_actions, demo_rewards, demo_next_obs, demo_dones = \
        load_demo_transitions(args.demos)

    # Create replay buffer with demo data
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
        demo_obs=demo_obs,
        demo_actions=demo_actions,
        demo_rewards=demo_rewards,
        demo_next_obs=demo_next_obs,
        demo_dones=demo_dones,
        demo_ratio=args.demo_ratio,
    )
    print(f"Demo replay buffer created: {len(demo_obs)} demo transitions")
    print()

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
        # Override mutable hyperparams the user may have changed
        model.learning_rate = args.lr
        model.batch_size = args.batch_size
        model.gamma = args.gamma
        model.tau = args.tau
        model.gradient_steps = args.utd_ratio
        model.ent_coef = ent_coef
        # Replace replay buffer with fresh demo buffer (online data is lost)
        model.replay_buffer = replay_buffer
        # LayerNorm is already baked into the loaded checkpoint weights
        print(f"  Model resumed in {_time.time() - _t0:.1f}s")
    else:
        print(f"Creating {algo_name} model...")
        model = algo_cls(
            "MlpPolicy",
            env,
            verbose=1,
            device=device_str,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            ent_coef=ent_coef,
            gradient_steps=args.utd_ratio,
            learning_starts=args.learning_starts,
            train_freq=1,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            ),
        )
        # Replace the default replay buffer with our demo buffer
        model.replay_buffer = replay_buffer
        # Inject LayerNorm into critics
        inject_layernorm_into_critics(model)
        print(f"  Model created in {_time.time() - _t0:.1f}s")
    print()

    # BC warmstart on actor
    bc_warmstart_sac(model, demo_obs, demo_actions,
                     epochs=args.bc_epochs, batch_size=args.bc_batch_size, lr=args.bc_lr)

    # Tighten exploration noise to preserve BC-learned behavior
    # log_std is a Linear layer; zero weights + low bias → constant std ≈ 0.135
    model.actor.log_std.weight.data.zero_()
    model.actor.log_std.bias.data.fill_(-2.0)
    print("Exploration noise tightened (log_std bias = -2.0, std ≈ 0.135)")

    # Create callbacks
    checkpoint_dir = Path(args.output).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    callback = CallbackList(callbacks)

    # Train
    print("\n" + "=" * 60)
    print(f"Starting {algo_name} + RLPD Training")
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
