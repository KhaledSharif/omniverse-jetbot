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


def load_demo_data(filepath: str, successful_only: bool = True):
    """Load demonstration data from NPZ file.

    Args:
        filepath: Path to .npz demo file
        successful_only: If True, only load successful episodes

    Returns:
        Tuple of (observations, actions) numpy arrays
    """
    from jetbot_keyboard_control import DemoPlayer
    player = DemoPlayer(filepath)

    print(f"Loaded {player.num_episodes} episodes, {player.total_frames} frames")

    # Filter to successful only if requested
    if successful_only:
        episodes = player.get_successful_episodes()
        print(f"Using {len(episodes)} successful episodes for BC warmstart")
    else:
        episodes = list(range(player.num_episodes))

    if len(episodes) == 0:
        raise ValueError("No episodes to train on! Record some successful demos first.")

    # Collect data for training
    observations = []
    actions = []
    for ep_idx in episodes:
        obs, acts = player.get_episode(ep_idx)
        observations.append(obs)
        actions.append(acts)

    observations = np.vstack(observations)
    actions = np.vstack(actions)

    print(f"BC warmstart data: {len(observations)} transitions")
    return observations.astype(np.float32), actions.astype(np.float32)


def bc_warmstart(model, env, demo_path: str, epochs: int = 50, batch_size: int = 64):
    """Pretrain PPO policy using behavioral cloning from demonstrations.

    Args:
        model: SB3 PPO model to pretrain
        env: Gymnasium environment (for spaces)
        demo_path: Path to demo .npz file
        epochs: Number of BC training epochs
        batch_size: BC training batch size
    """
    try:
        from imitation.algorithms import bc
        from imitation.data import types

        print("\n" + "=" * 60)
        print("Behavioral Cloning Warmstart")
        print("=" * 60)

        # Load demo data
        observations, actions = load_demo_data(demo_path)

        # Create transitions dataset
        transitions = types.TransitionsMinimal(
            obs=observations,
            acts=actions,
            infos=np.array([{}] * len(observations))
        )

        # Create BC trainer using the PPO's policy
        rng = np.random.default_rng(42)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            policy=model.policy,
            batch_size=batch_size,
            rng=rng,
        )

        # Train
        print(f"Pretraining policy for {epochs} epochs...")
        bc_trainer.train(n_epochs=epochs)
        print("BC warmstart complete!")
        print("=" * 60 + "\n")

    except ImportError as e:
        print(f"\nWarning: imitation library not available: {e}")
        print("Skipping BC warmstart. Install with: pip install imitation")
        print("Continuing with random policy initialization...\n")


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
    if args.bc_warmstart:
        print(f"  BC warmstart: {args.bc_warmstart}")
        print(f"  BC epochs: {args.bc_epochs}")
    print(f"  Output: {args.output}")
    print(f"  TensorBoard: {args.tensorboard_log}")
    print("=" * 60 + "\n")

    # Import here to allow --help without Isaac Sim
    from jetbot_rl_env import JetbotNavigationEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    # Create environment
    print("Creating environment...")
    env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
    )
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print()

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        # PPO hyperparameters (can be tuned)
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    print()

    # BC warmstart if requested
    if args.bc_warmstart:
        bc_warmstart(
            model, env,
            demo_path=args.bc_warmstart,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size
        )

    # Create checkpoint callback
    checkpoint_dir = Path(args.output).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_jetbot",
        verbose=1,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting PPO Training")
    print("=" * 60)
    print(f"View TensorBoard: tensorboard --logdir {args.tensorboard_log}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")

    # Save final model
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

    # Cleanup
    env.close()


if __name__ == '__main__':
    main()
