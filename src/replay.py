#!/usr/bin/env python3
"""Replay and inspect recorded demonstrations.

Usage:
    python replay.py demos/recording.npz --info          # Show statistics only
    python replay.py demos/recording.npz --episode 0     # Replay specific episode
    python replay.py demos/recording.npz --speed 0.5     # Slow motion playback
    python replay.py demos/recording.npz --successful    # Only replay successful episodes
"""

import argparse
import numpy as np
import time

from jetbot_config import (
    WHEEL_RADIUS, WHEEL_BASE,
    MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    START_POSITION,
)


def show_info(filepath: str):
    """Display demonstration statistics without loading Isaac Sim."""
    from demo_io import open_demo
    data = open_demo(filepath)

    print(f"\n{'='*60}")
    print(f"Demo File: {filepath}")
    print(f"{'='*60}")

    # Basic stats
    total_frames = len(data['observations'])
    num_episodes = len(data['episode_starts'])
    episode_success = data['episode_success']
    num_success = np.sum(episode_success)
    num_failed = num_episodes - num_success

    print(f"\nSummary:")
    print(f"  Total Frames:     {total_frames}")
    print(f"  Total Episodes:   {num_episodes}")
    print(f"  Successful:       {num_success} ({100*num_success/num_episodes:.1f}%)" if num_episodes > 0 else "  Successful:       0")
    print(f"  Failed:           {num_failed} ({100*num_failed/num_episodes:.1f}%)" if num_episodes > 0 else "  Failed:           0")

    # Data shapes
    print(f"\nData Shapes:")
    print(f"  Observations:     {data['observations'].shape}")
    print(f"  Actions:          {data['actions'].shape}")

    # Metadata if available
    if 'metadata' in data:
        metadata = data['metadata'].item()
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    # Episode details
    print(f"\nEpisode Details:")
    print(f"  {'Ep':<4} {'Start':<8} {'Length':<8} {'Return':<10} {'Status':<8}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for i in range(num_episodes):
        start = data['episode_starts'][i]
        length = data['episode_lengths'][i]
        ret = data['episode_returns'][i] if 'episode_returns' in data else 0.0
        success = "SUCCESS" if episode_success[i] else "FAILED"
        status_color = success

        print(f"  {i:<4} {start:<8} {length:<8} {ret:<10.2f} {status_color:<8}")

    print(f"{'='*60}\n")


def visual_playback(filepath: str, episode_idx: int = None, speed: float = 1.0,
                   successful_only: bool = False):
    """Run visual playback in the Isaac Sim simulator."""
    # Import Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from isaacsim.robot.wheeled_robots.controllers import DifferentialController
    from isaacsim.core.utils.nucleus import get_assets_root_path

    # Load demo data
    from jetbot_keyboard_control import DemoPlayer
    player = DemoPlayer(filepath)

    print(f"\nLoaded {player.num_episodes} episodes, {player.total_frames} frames")

    # Determine which episodes to play
    if successful_only:
        episodes = player.get_successful_episodes()
        print(f"Playing {len(episodes)} successful episodes")
    elif episode_idx is not None:
        episodes = [episode_idx]
        print(f"Playing episode {episode_idx}")
    else:
        episodes = list(range(player.num_episodes))
        print(f"Playing all {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No episodes to play!")
        simulation_app.close()
        return

    # Create world and robot
    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()

    jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="replay_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            position=START_POSITION,
        )
    )
    world.scene.add_default_ground_plane()

    # Create controller
    controller = DifferentialController(
        name="replay_controller",
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
    )

    world.reset()

    # Play each episode
    for ep_idx in episodes:
        obs, actions = player.get_episode(ep_idx)
        success_str = "SUCCESS" if player.episode_success[ep_idx] else "FAILED"
        print(f"\nPlaying Episode {ep_idx} ({len(actions)} frames) - {success_str}")

        # Reset robot to start position from first observation
        start_obs = obs[0]
        start_pos = np.array([start_obs[0], start_obs[1], 0.05])
        jetbot.set_world_pose(position=start_pos)

        for frame_idx, action in enumerate(actions):
            if not simulation_app.is_running():
                break

            # Scale action back to physical units
            linear_vel = action[0] * MAX_LINEAR_VELOCITY
            angular_vel = action[1] * MAX_ANGULAR_VELOCITY

            # Apply control
            wheel_actions = controller.forward(command=[linear_vel, angular_vel])
            jetbot.apply_wheel_actions(wheel_actions)

            # Step simulation
            world.step(render=True)

            # Sleep to control playback speed
            time.sleep(0.01 / speed)

            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{len(actions)}")

        print(f"Episode {ep_idx} complete")

    print("\nPlayback complete!")
    simulation_app.close()


def main():
    parser = argparse.ArgumentParser(
        description='Replay and inspect recorded demonstrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demos/recording.npz --info
      Show demo statistics without simulation

  %(prog)s demos/recording.npz --episode 0
      Replay episode 0 in simulator

  %(prog)s demos/recording.npz --successful --speed 0.5
      Replay only successful episodes at half speed
        """
    )
    parser.add_argument('demo_file', help='Path to .npz demo file')
    parser.add_argument('--info', action='store_true',
                       help='Show demo statistics only (no visual playback)')
    parser.add_argument('--episode', type=int, default=None,
                       help='Replay specific episode index')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--successful', action='store_true',
                       help='Only replay successful episodes')

    args = parser.parse_args()

    if args.info:
        show_info(args.demo_file)
    else:
        visual_playback(
            args.demo_file,
            episode_idx=args.episode,
            speed=args.speed,
            successful_only=args.successful
        )


if __name__ == '__main__':
    main()
