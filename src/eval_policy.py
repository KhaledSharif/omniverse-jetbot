#!/usr/bin/env python3
"""Evaluate trained PPO policy on Jetbot navigation task.

This script loads a trained PPO policy and evaluates it on the Jetbot navigation
environment, computing success rate and other metrics.

Usage:
    ./run.sh eval_policy.py models/ppo_jetbot.zip              # Evaluate with GUI
    ./run.sh eval_policy.py models/ppo_jetbot.zip --headless   # Headless evaluation
    ./run.sh eval_policy.py models/ppo_jetbot.zip --episodes 100
"""

import argparse
import numpy as np
from pathlib import Path


def compute_eval_metrics(successes, total_rewards, episode_lengths, total_episodes):
    """Compute evaluation metrics from episode results.

    Args:
        successes: Number of successful episodes
        total_rewards: List of per-episode total rewards
        episode_lengths: List of per-episode step counts
        total_episodes: Total number of episodes evaluated

    Returns:
        Dict with keys: success_rate, avg_reward, std_reward, avg_length,
        std_length, min_reward, max_reward
    """
    return {
        'success_rate': 100 * successes / total_episodes,
        'avg_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_reward': float(np.min(total_rewards)),
        'max_reward': float(np.max(total_rewards)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO policy on Jetbot navigation task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s models/ppo_jetbot.zip
      Evaluate with GUI for 10 episodes

  %(prog)s models/ppo_jetbot.zip --episodes 100 --headless
      Evaluate headless for 100 episodes

  %(prog)s models/ppo_jetbot.zip --stochastic
      Use stochastic (non-deterministic) actions
        """
    )

    parser.add_argument('policy_path', type=str,
                        help='Path to trained policy (.zip file)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--reward-mode', choices=['dense', 'sparse'], default='dense',
                        help='Reward mode (default: dense)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--vecnormalize', type=str, default=None, metavar='PKL_FILE',
                        help='Path to VecNormalize stats .pkl file (auto-detected if not given)')

    args = parser.parse_args()

    # Verify policy file exists
    policy_path = Path(args.policy_path)
    if not policy_path.exists():
        print(f"Error: Policy file not found: {policy_path}")
        print("\nAvailable models in models/ directory:")
        models_dir = Path("models")
        if models_dir.exists():
            for f in models_dir.glob("*.zip"):
                print(f"  {f}")
        return 1

    print("=" * 60)
    print("Policy Evaluation for Jetbot Navigation")
    print("=" * 60)
    print(f"  Policy: {args.policy_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Headless: {args.headless}")
    print(f"  Deterministic: {not args.stochastic}")
    print(f"  Reward mode: {args.reward_mode}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print("=" * 60 + "\n")

    # Import here to allow --help without Isaac Sim
    from jetbot_rl_env import JetbotNavigationEnv
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Try importing TQC from sb3-contrib
    try:
        from sb3_contrib import TQC
    except ImportError:
        TQC = None

    # Create environment
    print("Creating environment...")
    raw_env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
    )
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space: {raw_env.action_space.shape}")

    # Wrap with VecNormalize if stats file exists
    vec_env = DummyVecEnv([lambda: raw_env])

    # Auto-detect VecNormalize stats
    vecnorm_path = None
    if args.vecnormalize:
        vecnorm_path = Path(args.vecnormalize)
    else:
        auto_path = policy_path.with_suffix('.pkl')
        if auto_path.exists():
            vecnorm_path = auto_path

    if vecnorm_path and vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False      # don't update running stats
        vec_env.norm_reward = False   # raw rewards for metrics
        print(f"  VecNormalize stats loaded from: {vecnorm_path}")
    else:
        print("  VecNormalize stats: not found (using raw observations)")
    print()

    # Load trained policy (auto-detect algorithm)
    print(f"Loading policy from {args.policy_path}...")
    model = None
    for algo_cls, name in [(TQC, "TQC"), (SAC, "SAC"), (PPO, "PPO")]:
        if algo_cls is None:
            continue
        try:
            model = algo_cls.load(args.policy_path, env=vec_env)
            print(f"Loaded as {name} policy successfully!\n")
            break
        except Exception:
            continue
    if model is None:
        print("Error: Could not load policy with any supported algorithm (TQC/SAC/PPO)")
        vec_env.close()
        return 1

    # Evaluation metrics
    successes = 0
    total_rewards = []
    episode_lengths = []
    deterministic = not args.stochastic

    print("=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    for episode in range(args.episodes):
        # Reset environment (VecEnv API: reset returns obs only)
        if args.seed is not None:
            vec_env.seed(args.seed + episode)
        obs = vec_env.reset()

        done = False
        episode_reward = 0.0
        steps = 0
        goal_reached = False

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment (VecEnv API: returns arrays, infos is list of dicts)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += rewards[0]
            steps += 1
            done = dones[0]

            # Check for goal reached
            if infos[0].get('goal_reached', False):
                goal_reached = True

        # Record episode results
        if goal_reached:
            successes += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Print episode summary
        status = "SUCCESS" if goal_reached else "FAILED"
        print(f"Episode {episode + 1:3d}/{args.episodes}: "
              f"reward={episode_reward:8.2f}, steps={steps:4d}, {status}")

    # Compute statistics
    metrics = compute_eval_metrics(successes, total_rewards, episode_lengths, args.episodes)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Success rate:      {successes}/{args.episodes} ({metrics['success_rate']:.1f}%)")
    print(f"  Average reward:    {metrics['avg_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Average length:    {metrics['avg_length']:.1f} +/- {metrics['std_length']:.1f} steps")
    print(f"  Min reward:        {metrics['min_reward']:.2f}")
    print(f"  Max reward:        {metrics['max_reward']:.2f}")
    print("=" * 60)

    # Return success rate as exit code (0-100)
    vec_env.close()
    return 0


if __name__ == '__main__':
    exit(main())
