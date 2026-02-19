#!/usr/bin/env python3
"""Train PPO agent for Jetbot navigation task.

This script trains a PPO agent using Stable-Baselines3 on the Jetbot navigation
environment in Isaac Sim. Supports behavioral cloning warmstart from demonstrations.

Usage:
    ./run.sh train_rl.py                                    # Train with GUI
    ./run.sh train_rl.py --headless                         # Train headless (faster)
    ./run.sh train_rl.py --bc-warmstart demos/demo.npz      # Pretrain from demos
    ./run.sh train_rl.py --timesteps 500000                 # Custom timesteps
    ./run.sh train_rl.py --headless --timesteps 1000000     # Full training run
"""

import argparse
import numpy as np
from pathlib import Path

from demo_utils import load_demo_data, validate_demo_data, VerboseEpisodeCallback


def linear_schedule(initial_value: float):
    """Linear learning rate schedule: decays from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func



def bc_warmstart(model, env, demo_path: str, epochs: int = 50, batch_size: int = 64):
    """Pretrain PPO policy using behavioral cloning from demonstrations.

    Uses pure PyTorch to train the PPO actor network via MSE loss against
    demonstration actions. Only actor parameters are updated (value network
    is left untouched so PPO can learn its own value estimates).

    Args:
        model: SB3 PPO model to pretrain
        env: Gymnasium environment (for spaces)
        demo_path: Path to demo .npz file
        epochs: Number of BC training epochs
        batch_size: BC training batch size
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 60)
    print("Behavioral Cloning Warmstart (PyTorch)")
    print("=" * 60)

    # Load demo data and normalize observations to match VecNormalize
    observations, actions = load_demo_data(demo_path)
    observations = normalize_obs(observations, env)

    # Build dataloader
    dataset = TensorDataset(
        torch.tensor(observations),
        torch.tensor(actions),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Collect actor-only parameters (exclude value network)
    policy = model.policy
    actor_params = (
        list(policy.features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(actor_params, lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    device = policy.device

    print(f"Pretraining policy for {epochs} epochs on {len(dataset)} transitions...")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred_actions = policy.get_distribution(obs_batch).distribution.mean
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




def compute_mc_returns(filepath: str, gamma: float = 0.99) -> np.ndarray:
    """Compute Monte Carlo returns from demonstration data.

    Loads rewards and episode boundaries directly from NPZ and computes
    per-timestep discounted returns via backward pass within each episode.

    Args:
        filepath: Path to .npz demo file
        gamma: Discount factor

    Returns:
        (N,) float32 array of MC returns for each transition
    """
    data = np.load(filepath, allow_pickle=True)
    rewards = data['rewards']
    episode_lengths = data['episode_lengths']

    returns = np.zeros(len(rewards), dtype=np.float32)

    offset = 0
    for length in episode_lengths:
        # Backward pass within episode
        g = 0.0
        for t in range(int(length) - 1, -1, -1):
            g = rewards[offset + t] + gamma * g
            returns[offset + t] = g
        offset += int(length)

    return returns


def pretrain_critic(model, env, filepath: str, gamma: float = 0.99,
                    epochs: int = 50, batch_size: int = 64):
    """Pretrain PPO critic (value network) using Monte Carlo returns from demos.

    Trains only the critic parameters (value_net and mlp_extractor.value_net)
    to predict MC returns, giving the critic a meaningful initialization before RL.

    Observations are normalized using VecNormalize's obs_rms and returns are
    scaled by VecNormalize's ret_rms so the critic trains on the same scale
    it will see during PPO training.

    Args:
        model: SB3 PPO model
        env: VecNormalize-wrapped environment (for normalization stats)
        filepath: Path to demo .npz file
        gamma: Discount factor for MC returns
        epochs: Training epochs
        batch_size: Training batch size
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    print("\n" + "=" * 60)
    print("Critic Pretraining (Monte Carlo Returns)")
    print("=" * 60)

    # Compute MC returns and normalize to VecNormalize reward scale
    mc_returns = compute_mc_returns(filepath, gamma)
    mc_returns = normalize_returns(mc_returns, env)

    # Load observations and normalize to VecNormalize obs scale
    observations, _ = load_demo_data(filepath)
    observations = normalize_obs(observations, env)

    # Build dataset
    dataset = TensorDataset(
        torch.tensor(observations),
        torch.tensor(mc_returns).unsqueeze(1),  # (N, 1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Collect critic-only parameters
    policy = model.policy
    critic_params = (
        list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters())
    )
    optimizer = torch.optim.Adam(critic_params, lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    device = policy.device

    print(f"Pretraining critic for {epochs} epochs on {len(dataset)} transitions...")

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, returns_batch in loader:
            obs_batch = obs_batch.to(device)
            returns_batch = returns_batch.to(device)

            # Forward pass through critic
            features = policy.extract_features(obs_batch, policy.vf_features_extractor)
            value_latent = policy.mlp_extractor.forward_critic(features)
            predicted_values = policy.value_net(value_latent)

            loss = loss_fn(predicted_values, returns_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}, Loss: {total_loss / n_batches:.6f}")

    print("Critic pretraining complete!")
    print("=" * 60 + "\n")


def prewarm_vecnormalize(env, demo_path: str, gamma: float = 0.99):
    """Pre-warm VecNormalize running statistics from demonstration data.

    Feeds demo observations and simulated discounted returns through
    VecNormalize's RunningMeanStd so that normalization during RL matches
    the distribution used for BC and critic pretraining.

    Args:
        env: VecNormalize-wrapped environment
        demo_path: Path to demo .npz file
        gamma: Discount factor (must match PPO gamma)
    """
    data = np.load(demo_path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    episode_lengths = data['episode_lengths']

    # Pre-warm observation running stats
    env.obs_rms.update(observations)

    # Simulate VecNormalize's return accumulation (self.returns * gamma + reward)
    # to pre-warm ret_rms with the correct variance estimate
    running_returns = []
    offset = 0
    for length in episode_lengths:
        ret = 0.0
        for t in range(int(length)):
            ret = ret * gamma + rewards[offset + t]
            running_returns.append(ret)
        offset += int(length)

    env.ret_rms.update(np.array(running_returns))

    print("\n" + "=" * 60)
    print("VecNormalize Pre-warmed from Demo Data")
    print("=" * 60)
    obs_std = np.sqrt(env.obs_rms.var)
    print(f"  Observations: {len(observations)} samples")
    print(f"  obs_rms mean range: [{env.obs_rms.mean.min():.3f}, {env.obs_rms.mean.max():.3f}]")
    print(f"  obs_rms std range:  [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    print(f"  ret_rms std: {np.sqrt(env.ret_rms.var):.3f}")
    print("=" * 60 + "\n")


def normalize_obs(observations: np.ndarray, env) -> np.ndarray:
    """Normalize observations using VecNormalize's running stats.

    Applies the same (obs - mean) / std transformation and clipping
    that VecNormalize applies during RL rollouts.

    Args:
        observations: Raw observations array (N, obs_dim)
        env: VecNormalize-wrapped environment

    Returns:
        Normalized observations as float32 array
    """
    return np.clip(
        (observations - env.obs_rms.mean) / np.sqrt(env.obs_rms.var + env.epsilon),
        -env.clip_obs, env.clip_obs,
    ).astype(np.float32)


def normalize_returns(returns: np.ndarray, env) -> np.ndarray:
    """Normalize returns using VecNormalize's reward normalization.

    VecNormalize divides rewards by sqrt(ret_rms.var) (no mean subtraction).
    Apply the same scaling to MC returns for critic pretraining.

    Args:
        returns: Raw MC returns array (N,)
        env: VecNormalize-wrapped environment

    Returns:
        Normalized returns as float32 array
    """
    return np.clip(
        returns / np.sqrt(env.ret_rms.var + env.epsilon),
        -env.clip_reward, env.clip_reward,
    ).astype(np.float32)


class SaveVecNormalizeCallback:
    """Saves VecNormalize statistics alongside model checkpoints.

    Instantiated after deferred imports provide BaseCallback.
    Use create() classmethod inside main() after imports are available.
    """

    @staticmethod
    def create(base_callback_cls, save_freq, save_path, name_prefix="vecnormalize", verbose=0):
        """Create callback class using the imported BaseCallback."""

        class _SaveVecNormalizeCallback(base_callback_cls):
            def __init__(self):
                super().__init__(verbose)
                self.save_freq = save_freq
                self.save_path = Path(save_path)
                self.name_prefix = name_prefix

            def _on_step(self):
                if self.n_calls % self.save_freq == 0:
                    path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.pkl"
                    self.training_env.save(str(path))
                    if self.verbose > 0:
                        print(f"Saving VecNormalize stats to {path}")
                return True

        return _SaveVecNormalizeCallback()


def main():
    parser = argparse.ArgumentParser(
        description='Train PPO agent for Jetbot navigation task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
      Train with GUI and default settings

  %(prog)s --headless --timesteps 500000
      Train headless for 500k timesteps

  %(prog)s --bc-warmstart demos/recording.npz
      Pretrain from demonstrations before RL

  %(prog)s --headless --bc-warmstart demos/recording.npz --timesteps 1000000
      Full training: BC warmstart + 1M timesteps of RL
        """
    )

    # Training arguments
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--reward-mode', choices=['dense', 'sparse'], default='dense',
                        help='Reward mode (default: dense)')

    # Environment arguments
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

    # BC warmstart arguments
    parser.add_argument('--bc-warmstart', type=str, metavar='DEMO_FILE',
                        help='Path to demo .npz file for BC warmstart')
    parser.add_argument('--bc-epochs', type=int, default=50,
                        help='BC pretraining epochs (default: 50)')
    parser.add_argument('--bc-batch-size', type=int, default=64,
                        help='BC training batch size (default: 64)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                        help='Save checkpoint every N steps (default: 10000)')

    # Logging arguments
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-episode stats (outcome, return, lidar, goal)')

    # Output arguments
    parser.add_argument('--output', type=str, default='models/ppo_jetbot.zip',
                        help='Output model path (default: models/ppo_jetbot.zip)')
    parser.add_argument('--tensorboard-log', type=str, default='./runs/',
                        help='TensorBoard log directory (default: ./runs/)')

    args = parser.parse_args()

    print("=" * 60)
    print("PPO Training for Jetbot Navigation")
    print("=" * 60)
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Headless: {args.headless}")
    print(f"  Seed: {args.seed}")
    print(f"  Max steps/episode: {args.max_steps}")
    if args.bc_warmstart:
        print(f"  BC warmstart: {args.bc_warmstart}")
        print(f"  BC epochs: {args.bc_epochs}")
    print(f"  Output: {args.output}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    print("=" * 60 + "\n")

    # Import here to allow --help without Isaac Sim
    import torch
    from jetbot_rl_env import JetbotNavigationEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

    # Create environment
    print("Creating environment...")
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
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space: {raw_env.action_space.shape}")

    # Wrap with VecNormalize for observation/reward normalization
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    print("  VecNormalize: obs=True, reward=True")
    print()

    # Create PPO model
    print("Creating PPO model...")
    device = "cpu" if args.cpu else "auto"
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        # PPO hyperparameters tuned for continuous control
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        target_kl=0.1,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.Tanh,
            log_std_init=-1.0,
        ),
    )
    print()

    # BC warmstart if requested
    if args.bc_warmstart:
        # Validate demo data first (fail fast)
        validate_demo_data(args.bc_warmstart)

        # Pre-warm VecNormalize stats from demo data so BC and critic
        # pretraining use the same observation/reward scale as PPO
        prewarm_vecnormalize(env, args.bc_warmstart, gamma=0.99)

        # Pretrain actor via behavioral cloning
        bc_warmstart(
            model, env,
            demo_path=args.bc_warmstart,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size
        )

        # Pretrain critic via Monte Carlo returns
        pretrain_critic(
            model, env,
            filepath=args.bc_warmstart,
            gamma=0.99,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
        )

        # Tighten exploration noise after BC warmstart (std ~ 0.368)
        model.policy.log_std.data.fill_(-1.0)
        print("log_std tightened to -1.0 (std ~ 0.368) after BC warmstart")

    # Create checkpoint callbacks
    checkpoint_dir = Path(args.output).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_jetbot",
        verbose=1,
    )
    vecnorm_callback = SaveVecNormalizeCallback.create(
        BaseCallback,
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        verbose=1,
    )
    callbacks = [checkpoint_callback, vecnorm_callback]
    if args.verbose:
        callbacks.append(VerboseEpisodeCallback.create(BaseCallback))
    callback = CallbackList(callbacks)

    # Train
    print("\n" + "=" * 60)
    print("Starting PPO Training")
    print("=" * 60)
    print(f"View TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")

    # Save final model and VecNormalize stats
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    vecnorm_path = output_path.with_suffix('.pkl')
    env.save(str(vecnorm_path))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Model saved to: {output_path}")
    print(f"  VecNormalize stats: {vecnorm_path}")
    print(f"  Checkpoints in: {checkpoint_dir}")
    print(f"  TensorBoard logs: {args.tensorboard_log}")
    print("\nTo evaluate the trained policy:")
    print(f"  ./run.sh eval_policy.py {output_path}")
    print("=" * 60)

    # Cleanup
    env.close()


if __name__ == '__main__':
    main()
