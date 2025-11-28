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
    from stable_baselines3 import PPO

    # Create environment
    print("Creating environment...")
    env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
    )
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print()

    # Load trained policy
    print(f"Loading policy from {args.policy_path}...")
    model = PPO.load(args.policy_path)
    print("Policy loaded successfully!\n")

    # Evaluation metrics
    successes = 0
    total_rewards = []
    episode_lengths = []
    deterministic = not args.stochastic

    print("=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    for episode in range(args.episodes):
        # Reset environment
        if args.seed is not None:
            obs, info = env.reset(seed=args.seed + episode)
        else:
            obs, info = env.reset()

        done = False
        episode_reward = 0.0
        steps = 0
        goal_reached = False

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

            # Check for goal reached
            if info.get('goal_reached', False):
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
    success_rate = 100 * successes / args.episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Success rate:      {successes}/{args.episodes} ({success_rate:.1f}%)")
    print(f"  Average reward:    {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Average length:    {avg_length:.1f} +/- {std_length:.1f} steps")
    print(f"  Min reward:        {np.min(total_rewards):.2f}")
    print(f"  Max reward:        {np.max(total_rewards):.2f}")
    print("=" * 60)

    # Return success rate as exit code (0-100)
    env.close()
    return 0


if __name__ == '__main__':
    exit(main())
