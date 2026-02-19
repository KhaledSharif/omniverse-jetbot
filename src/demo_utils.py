"""Shared utilities for loading and validating demonstration data.

Used by train_rl.py, train_sac.py, and train_bc.py to avoid duplication.
"""

import numpy as np


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


def load_demo_data(filepath: str, successful_only: bool = False):
    """Load demonstration data from NPZ file.

    Args:
        filepath: Path to .npz demo file
        successful_only: If True, only load successful episodes

    Returns:
        Tuple of (observations, actions) as float32 numpy arrays
    """
    from jetbot_keyboard_control import DemoPlayer
    player = DemoPlayer(filepath)

    print(f"Loaded {player.num_episodes} episodes, {player.total_frames} frames")

    if successful_only:
        episodes = player.get_successful_episodes()
        print(f"Using {len(episodes)} successful episodes")
    else:
        episodes = list(range(player.num_episodes))

    if len(episodes) == 0:
        raise ValueError("No episodes to train on! Record some successful demos first.")

    observations = []
    actions = []
    for ep_idx in episodes:
        obs, acts = player.get_episode(ep_idx)
        observations.append(obs)
        actions.append(acts)

    observations = np.vstack(observations)
    actions = np.vstack(actions)

    print(f"Demo data: {len(observations)} transitions")
    return observations.astype(np.float32), actions.astype(np.float32)


def load_demo_transitions(npz_path: str):
    """Load demo data and reconstruct (obs, action, reward, next_obs, done) tuples.

    Uses the recorded ``dones`` from the NPZ file, which mark true MDP terminals
    (goal reached, collision, out-of-bounds) but NOT timeouts (truncations).
    This ensures correct Q-learning bootstrapping: ``Q_target = r + gamma * (1 - done) * Q(s', a')``.

    For backward compatibility with legacy NPZ files where ``dones`` only reflected
    ``goal_reached``, the last step of each episode falls back to ``done=1.0`` if
    no terminal was recorded.

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
    raw_dones = data['dones'].astype(np.float32) if 'dones' in data else np.zeros(len(observations), dtype=np.float32)

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

        # Use recorded dones (true MDP terminals only)
        dones[offset:offset + length] = raw_dones[offset:offset + length]

        # Fallback for legacy files: ensure last step has done=True
        if dones[offset + length - 1] == 0.0:
            dones[offset + length - 1] = 1.0

        offset += length

    print(f"Loaded {len(episode_lengths)} demo episodes, {total} transitions")
    return observations, actions, rewards, next_obs, dones


class VerboseEpisodeCallback:
    """Prints episode stats and periodic step-rate info during training.

    Instantiated after deferred imports provide BaseCallback.
    Use create() classmethod inside main() after imports are available.
    """

    @staticmethod
    def create(base_callback_cls):
        """Create callback class using the imported BaseCallback."""

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
                self._report_interval = 100

            def _on_training_start(self):
                import time
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
