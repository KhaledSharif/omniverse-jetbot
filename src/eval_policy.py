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
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Action chunk size (default: auto-detect from model)')
    parser.add_argument('--n-frames', type=int, default=None,
                        help='Number of stacked frames (default: auto-detect from model)')
    parser.add_argument('--inflation-radius', type=float, default=0.08,
                        help='Obstacle inflation radius for A* planner in meters (default: 0.08)')
    parser.add_argument('--safe', action='store_true',
                        help='Track constraint costs during evaluation')
    parser.add_argument('--cost-type', choices=['proximity', 'collision', 'both'],
                        default='proximity',
                        help='Cost signal type for evaluation (default: proximity)')
    parser.add_argument('--add-prev-action', action='store_true',
                        help='Include previous action in observations (36D instead of 34D)')
    parser.add_argument('--use-camera', action='store_true',
                        help='Enable DINOv2 camera features (auto-detected from model obs dim)')

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
    import train_sac  # noqa: F401 — registers ChunkCVAEFeatureExtractor for model deserialization
    from jetbot_rl_env import JetbotNavigationEnv, ChunkedEnvWrapper, FrameStackWrapper
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Try importing TQC and CrossQ from sb3-contrib
    try:
        from sb3_contrib import CrossQ
    except ImportError:
        CrossQ = None
    try:
        from sb3_contrib import TQC
    except ImportError:
        TQC = None

    # Create environment (use_camera may be overridden by auto-detection after model load)
    print("Creating environment...")
    # Auto-detect add_prev_action from model obs dim if not explicitly set
    add_prev_action = getattr(args, 'add_prev_action', False)
    use_camera = getattr(args, 'use_camera', False)

    raw_env = JetbotNavigationEnv(
        reward_mode=args.reward_mode,
        headless=args.headless,
        inflation_radius=args.inflation_radius,
        cost_type=args.cost_type,
        add_prev_action=add_prev_action,
        use_camera=use_camera,
    )
    print(f"  Observation space: {raw_env.observation_space.shape}")
    print(f"  Action space (inner): {raw_env.action_space.shape}")

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

    # Load trained policy (auto-detect algorithm) — load without env first to detect chunk size
    print(f"Loading policy from {args.policy_path}...")
    model = None
    algo_used = None
    for algo_cls, name in [(CrossQ, "CrossQ"), (TQC, "TQC"), (SAC, "SAC"), (PPO, "PPO")]:
        if algo_cls is None:
            continue
        try:
            model = algo_cls.load(args.policy_path)
            algo_used = name
            break
        except Exception:
            continue
    if model is None:
        print("Error: Could not load policy with any supported algorithm (TQC/SAC/PPO)")
        vec_env.close()
        return 1

    # Auto-detect chunk size from model action space
    model_action_dim = model.action_space.shape[0]
    chunk_size = args.chunk_size
    if chunk_size is None and model_action_dim > 2:
        chunk_size = model_action_dim // 2
        print(f"  Auto-detected chunk_size={chunk_size} from model action_dim={model_action_dim}")

    # Auto-detect camera from model observation space
    model_obs_dim = model.observation_space.shape[0]
    if not use_camera and (model_obs_dim - 384) in (34, 36):
        use_camera = True
        print(f"  Auto-detected use_camera=True from model obs_dim={model_obs_dim} "
              f"(base={model_obs_dim - 384}D + 384D DINOv2)")
        # Recreate env with camera enabled
        raw_env.close()
        raw_env = JetbotNavigationEnv(
            reward_mode=args.reward_mode,
            headless=args.headless,
            inflation_radius=args.inflation_radius,
            cost_type=args.cost_type,
            add_prev_action=add_prev_action,
            use_camera=True,
        )
        vec_env = DummyVecEnv([lambda: raw_env])
        print(f"  Recreated env with camera: obs_space={raw_env.observation_space.shape}")

    # Auto-detect n_frames from model observation space
    n_frames = args.n_frames
    if n_frames is None and model_obs_dim > 34:
        # base_obs_dim depends on camera
        if use_camera:
            base_obs_dim = model_obs_dim  # camera models don't use frame stacking yet
            n_frames = 1
        else:
            base_obs_dim = 36 if (model_obs_dim % 36 == 0 and model_obs_dim % 34 != 0) else 34
            n_frames = model_obs_dim // base_obs_dim
            print(f"  Auto-detected n_frames={n_frames} from model obs_dim={model_obs_dim} (base={base_obs_dim})")

    # Wrap env with FrameStackWrapper if frame-stacked model
    if n_frames is not None and n_frames > 1:
        raw_env = FrameStackWrapper(raw_env, n_frames=n_frames)
        print(f"  Wrapped env with FrameStackWrapper(n_frames={n_frames})")

    # Wrap env with ChunkedEnvWrapper if chunked model
    if chunk_size is not None and chunk_size > 1:
        raw_env = ChunkedEnvWrapper(raw_env, chunk_size=chunk_size, gamma=0.99)
        print(f"  Wrapped env with ChunkedEnvWrapper(k={chunk_size})")
        print(f"  Action space (chunked): {raw_env.action_space.shape}")

    # Recreate vec_env if any wrappers were applied
    if (n_frames is not None and n_frames > 1) or (chunk_size is not None and chunk_size > 1):
        # Recreate vec_env with the wrapped env
        vec_env = DummyVecEnv([lambda: raw_env])
        if vecnorm_path and vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

    # Re-load model with the (possibly wrapped) env
    model.set_env(vec_env)
    print(f"Loaded as {algo_used} policy successfully!\n")

    # Evaluation metrics
    successes = 0
    total_rewards = []
    episode_lengths = []
    episode_costs = []
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
        episode_cost = 0.0
        steps = 0
        goal_reached = False

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment (VecEnv API: returns arrays, infos is list of dicts)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += rewards[0]
            episode_cost += infos[0].get('cost_chunk', infos[0].get('cost', 0.0))
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
        episode_costs.append(episode_cost)

        # Print episode summary
        status = "SUCCESS" if goal_reached else "FAILED"
        cost_str = f", cost={episode_cost:6.2f}" if args.safe else ""
        print(f"Episode {episode + 1:3d}/{args.episodes}: "
              f"reward={episode_reward:8.2f}, steps={steps:4d}{cost_str}, {status}")

    # Compute statistics
    metrics = compute_eval_metrics(successes, total_rewards, episode_lengths, args.episodes)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Success rate:      {successes}/{args.episodes} ({metrics['success_rate']:.1f}%)")
    print(f"  Average reward:    {metrics['avg_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    step_label = f"chunk-steps (k={chunk_size})" if chunk_size and chunk_size > 1 else "steps"
    print(f"  Average length:    {metrics['avg_length']:.1f} +/- {metrics['std_length']:.1f} {step_label}")
    print(f"  Min reward:        {metrics['min_reward']:.2f}")
    print(f"  Max reward:        {metrics['max_reward']:.2f}")
    if args.safe and episode_costs:
        mean_cost = float(np.mean(episode_costs))
        std_cost = float(np.std(episode_costs))
        exceedances = sum(1 for c in episode_costs if c > 25.0)
        print(f"  Mean episode cost: {mean_cost:.2f} +/- {std_cost:.2f}")
        print(f"  Budget exceedances: {exceedances}/{args.episodes} (budget=25.0)")
    print("=" * 60)

    # Return success rate as exit code (0-100)
    vec_env.close()
    return 0


if __name__ == '__main__':
    exit(main())
