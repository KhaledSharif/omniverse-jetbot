"""Shared utilities for loading and validating demonstration data.

Used by train_rl.py, train_sac.py, and train_bc.py to avoid duplication.
"""

import numpy as np
from demo_io import open_demo


def convert_obs_to_egocentric(obs_array, workspace_bounds):
    """Vectorized conversion from old (v1) 34D obs to new ego-centric (v2) 34D obs.

    Old layout: [x, y, heading, lin_vel, ang_vel, goal_x, goal_y, dist, angle_to_goal, reached, lidar...]
    New layout: [norm_ws_x, norm_ws_y, sin(h), cos(h), lin_vel, ang_vel, goal_body_x, goal_body_y, dist, reached, lidar...]

    Args:
        obs_array: numpy (N, 34) old-format observations
        workspace_bounds: dict with 'x' and 'y' keys, each [min, max]

    Returns:
        numpy (N, 34) ego-centric observations
    """
    out = obs_array.copy()
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Old indices
    robot_x = obs_array[:, 0]
    robot_y = obs_array[:, 1]
    heading = obs_array[:, 2]
    # lin_vel (3), ang_vel (4) stay at same relative position in the base obs
    dist = obs_array[:, 7]
    angle_to_goal = obs_array[:, 8]

    # New features
    out[:, 0] = (robot_x - x_min) / x_range if x_range > 0 else 0.5
    out[:, 1] = (robot_y - y_min) / y_range if y_range > 0 else 0.5
    out[:, 2] = np.sin(heading)
    out[:, 3] = np.cos(heading)
    out[:, 4] = obs_array[:, 3]  # lin_vel
    out[:, 5] = obs_array[:, 4]  # ang_vel
    out[:, 6] = dist * np.cos(angle_to_goal)  # goal_body_x
    out[:, 7] = dist * np.sin(angle_to_goal)  # goal_body_y
    out[:, 8] = dist
    out[:, 9] = obs_array[:, 9]  # reached
    # LiDAR [10:34] stays the same

    return out.astype(np.float32)


def validate_demo_data(filepath: str, require_cost: bool = False) -> dict:
    """Validate demonstration data meets minimum requirements for training.

    Args:
        filepath: Path to demo file (.npz or .hdf5)
        require_cost: If True, require cost data (has_cost metadata + costs array)

    Returns:
        dict with keys: episodes, transitions, successful, avg_return

    Raises:
        ValueError: If data doesn't meet minimum requirements
    """
    data = open_demo(filepath)

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

    if require_cost:
        metadata = data['metadata'].item() if 'metadata' in data else {}
        has_cost = metadata.get('has_cost', False) if isinstance(metadata, dict) else False
        if not has_cost or 'costs' not in data:
            raise ValueError(
                "Demo file missing cost data (required for --safe). "
                "Re-record demos with a version that records costs."
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


def load_demo_transitions(npz_path: str, load_costs: bool = False):
    """Load demo data and reconstruct (obs, action, reward, next_obs, done) tuples.

    Uses the recorded ``dones`` from the NPZ file, which mark true MDP terminals
    (goal reached, collision, out-of-bounds) but NOT timeouts (truncations).
    This ensures correct Q-learning bootstrapping: ``Q_target = r + gamma * (1 - done) * Q(s', a')``.

    For backward compatibility with legacy NPZ files where ``dones`` only reflected
    ``goal_reached``, the last step of each episode falls back to ``done=1.0`` if
    no terminal was recorded.

    Args:
        npz_path: Path to .npz demo file
        load_costs: If True, also return costs array (6-tuple)

    Returns:
        5-tuple of (obs, actions, rewards, next_obs, dones) numpy arrays, or
        6-tuple adding costs when load_costs=True
    """
    from jetbot_config import DEFAULT_WORKSPACE_BOUNDS, OBS_VERSION

    data = open_demo(npz_path)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    episode_lengths = data['episode_lengths']
    raw_dones = data['dones'].astype(np.float32) if 'dones' in data else np.zeros(len(observations), dtype=np.float32)

    # Auto-convert old v1 observations to ego-centric v2 format
    metadata = data['metadata'].item() if 'metadata' in data else {}
    if isinstance(metadata, dict):
        demo_obs_version = metadata.get('obs_version', 1)
    else:
        demo_obs_version = 1
    if demo_obs_version < OBS_VERSION and observations.shape[1] >= 10:
        print(f"  Auto-converting demo obs from v{demo_obs_version} to v{OBS_VERSION} (ego-centric)")
        observations = convert_obs_to_egocentric(observations, DEFAULT_WORKSPACE_BOUNDS)

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
    if load_costs:
        costs = data['costs'].astype(np.float32) if 'costs' in data else np.zeros(total, dtype=np.float32)
        return observations, actions, rewards, next_obs, dones, costs
    return observations, actions, rewards, next_obs, dones


def extract_action_chunks(demo_obs, demo_actions, episode_lengths, chunk_size):
    """Extract action chunks using a sliding window within each episode.

    Args:
        demo_obs: numpy array of observations (N, obs_dim)
        demo_actions: numpy array of actions (N, act_dim)
        episode_lengths: numpy array of per-episode step counts
        chunk_size: number of actions per chunk (k)

    Returns:
        Tuple of (chunk_obs, chunk_actions_flat) — shapes (M, obs_dim) and (M, k*act_dim)
    """
    act_dim = demo_actions.shape[1]
    chunk_obs_list = []
    chunk_actions_list = []

    offset = 0
    for ep_idx, length in enumerate(episode_lengths):
        length = int(length)
        if length < chunk_size:
            print(f"  Warning: episode {ep_idx} has {length} steps < chunk_size {chunk_size}, skipping")
            offset += length
            continue
        for t in range(length - chunk_size + 1):
            chunk_obs_list.append(demo_obs[offset + t])
            chunk_actions_list.append(
                demo_actions[offset + t:offset + t + chunk_size].reshape(-1)
            )
        offset += length

    chunk_obs = np.array(chunk_obs_list, dtype=np.float32)
    chunk_actions_flat = np.array(chunk_actions_list, dtype=np.float32)
    print(f"Extracted {len(chunk_obs)} action chunks (k={chunk_size}) from {len(episode_lengths)} episodes")
    return chunk_obs, chunk_actions_flat


def make_chunk_transitions(demo_obs, demo_actions, demo_rewards, demo_dones,
                           episode_lengths, chunk_size, gamma, demo_costs=None):
    """Build chunk-level (obs, action, reward, next_obs, done) transitions.

    Uses a sliding window within each episode. For terminal chunks where the
    episode ends at step j < k, the partial chunk is included with truncated
    reward sum and done=True.

    Args:
        demo_obs: numpy array of observations (N, obs_dim)
        demo_actions: numpy array of actions (N, act_dim)
        demo_rewards: numpy array of rewards (N,)
        demo_dones: numpy array of dones (N,)
        episode_lengths: numpy array of per-episode step counts
        chunk_size: number of actions per chunk (k)
        gamma: discount factor for computing R_chunk
        demo_costs: optional numpy array of costs (N,). When provided, also
            returns chunk-level discounted costs.

    Returns:
        5-tuple of (chunk_obs, chunk_actions, chunk_rewards, chunk_next_obs, chunk_dones),
        or 6-tuple adding chunk_costs when demo_costs is provided
    """
    act_dim = demo_actions.shape[1]
    obs_dim = demo_obs.shape[1]

    c_obs, c_acts, c_rews, c_next, c_dones = [], [], [], [], []
    c_costs = [] if demo_costs is not None else None

    offset = 0
    for ep_idx, length in enumerate(episode_lengths):
        length = int(length)
        if length < chunk_size:
            print(f"  Warning: episode {ep_idx} has {length} steps < chunk_size {chunk_size}, skipping")
            offset += length
            continue

        for t in range(length - chunk_size + 1):
            abs_t = offset + t
            # Observation at chunk start
            c_obs.append(demo_obs[abs_t])

            # Action chunk (full k steps)
            c_acts.append(demo_actions[abs_t:abs_t + chunk_size].reshape(-1))

            # Discounted reward sum, cost sum, and terminal check
            r_chunk = 0.0
            cost_chunk = 0.0
            done_any = False
            for i in range(chunk_size):
                r_chunk += (gamma ** i) * demo_rewards[abs_t + i]
                if demo_costs is not None:
                    cost_chunk += (gamma ** i) * demo_costs[abs_t + i]
                if demo_dones[abs_t + i]:
                    done_any = True

            c_rews.append(r_chunk)
            c_dones.append(float(done_any))
            if c_costs is not None:
                c_costs.append(cost_chunk)

            # Next obs = observation after the full chunk
            next_idx = abs_t + chunk_size
            if next_idx < offset + length:
                c_next.append(demo_obs[next_idx])
            else:
                # End of episode: next_obs is the last obs (masked by done=True)
                c_next.append(demo_obs[offset + length - 1])

        offset += length

    result = (
        np.array(c_obs, dtype=np.float32),
        np.array(c_acts, dtype=np.float32),
        np.array(c_rews, dtype=np.float32),
        np.array(c_next, dtype=np.float32),
        np.array(c_dones, dtype=np.float32),
    )
    if c_costs is not None:
        return result + (np.array(c_costs, dtype=np.float32),)
    return result


def build_frame_stacks(demo_obs, episode_lengths, n_frames):
    """Convert step-level demo observations to frame-stacked observations.

    For each step t in each episode, stacks the last n_frames observations.
    Early steps repeat the episode's first observation for missing slots,
    matching FrameStackWrapper.reset() behavior.

    Args:
        demo_obs: numpy array of observations (N, obs_dim)
        episode_lengths: numpy array of per-episode step counts
        n_frames: number of frames to stack

    Returns:
        numpy array of shape (N, n_frames * obs_dim)
    """
    obs_dim = demo_obs.shape[1]
    total = len(demo_obs)
    stacked = np.zeros((total, n_frames * obs_dim), dtype=np.float32)

    offset = 0
    for length in episode_lengths:
        length = int(length)
        for t in range(length):
            abs_t = offset + t
            frames = []
            for f in range(n_frames):
                # Index into the past: t - (n_frames - 1 - f)
                src_t = t - (n_frames - 1 - f)
                if src_t < 0:
                    frames.append(demo_obs[offset])  # Repeat first obs
                else:
                    frames.append(demo_obs[offset + src_t])
            stacked[abs_t] = np.concatenate(frames)
        offset += length

    return stacked


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

                    # Action stats from last batch
                    action_str = ""
                    actions = self.locals.get('actions')
                    if actions is not None:
                        act = np.array(actions)
                        if act.ndim >= 2 and act.shape[-1] >= 2:
                            action_str = (
                                f" | act_mean=[{act[...,0].mean():+.3f},{act[...,1].mean():+.3f}]"
                                f" act_std=[{act[...,0].std():.3f},{act[...,1].std():.3f}]"
                            )

                    # Entropy coef
                    ent_str = ""
                    if hasattr(self.model, 'ent_coef_tensor'):
                        ent_str = f" | ent={self.model.ent_coef_tensor.item():.5f}"

                    print(
                        f"[DEBUG] step={total_steps:>7d} | "
                        f"elapsed={elapsed:.1f}s | "
                        f"rate={rate:.1f} steps/s | "
                        f"episodes={self._ep_count}"
                        f"{action_str}{ent_str}",
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
                            # Ego-centric: goal_body_x=[6], goal_body_y=[7], dist=[8]
                            goal_body_x, goal_body_y = obs[6], obs[7]
                            dist = obs[8]
                            # Derive heading from sin/cos: obs[2]=sin(h), obs[3]=cos(h)
                            heading_deg = float(np.degrees(np.arctan2(obs[2], obs[3])))
                            print(
                                f"[EP {self._ep_count + 1:4d} START] "
                                f"goal_body=({goal_body_x:+.2f}, {goal_body_y:+.2f}) | "
                                f"dist={dist:.2f}m | "
                                f"heading={heading_deg:+.1f}deg",
                                flush=True
                            )

                        self._ep_reward = 0.0
                        self._ep_steps = 0
                        self._ep_min_lidar = float('inf')

                return True

        return _VerboseEpisodeCallback()
