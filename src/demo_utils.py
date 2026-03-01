"""Shared utilities for loading and validating demonstration data.

Used by train_rl.py, train_sac.py, and train_bc.py to avoid duplication.
"""

import numpy as np
from demo_io import open_demo


# Module-level DINOv2 cache (avoid reloading per call)
_dinov2_cache = {}


def _get_dinov2_model(device=None):
    """Load and cache a frozen DINOv2 ViT-S/14 model.

    Args:
        device: torch device (auto-selects CUDA if available)

    Returns:
        Tuple of (model, device, mean_tensor, std_tensor)
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = str(device)
    if key not in _dinov2_cache:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        _dinov2_cache[key] = (model, device, mean, std)
    return _dinov2_cache[key]


def encode_images_dinov2(images, batch_size=64, device=None):
    """Batch-encode uint8 RGB images to DINOv2 feature vectors.

    Args:
        images: numpy array (N, H, W, 3) uint8
        batch_size: mini-batch size for GPU inference
        device: torch device (auto-selects CUDA if available)

    Returns:
        numpy array (N, 384) float32
    """
    import torch
    from jetbot_config import IMAGE_FEATURE_DIM

    model, dev, mean, std = _get_dinov2_model(device)
    n = len(images)
    features = np.zeros((n, IMAGE_FEATURE_DIM), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # (B, H, W, 3) uint8 -> (B, 3, H, W) float32 [0, 1]
        batch = torch.from_numpy(images[start:end]).float().permute(0, 3, 1, 2).to(dev) / 255.0
        batch = (batch - mean) / std
        with torch.no_grad():
            out = model(batch)  # (B, 384)
        features[start:end] = out.cpu().numpy()
        if n > batch_size and start % (batch_size * 10) == 0:
            print(f"  DINOv2 encoding: {end}/{n} frames", flush=True)

    print(f"  DINOv2 encoded {n} frames -> ({n}, {IMAGE_FEATURE_DIM})")
    return features


def build_camera_obs(base_obs, image_features):
    """Insert image features into base observations (before LiDAR).

    Args:
        base_obs: numpy array (N, 34) or (N, 36) base observations
        image_features: numpy array (N, 384) DINOv2 features

    Returns:
        numpy array (N, 418) or (N, 420) with features inserted before last 24D
    """
    # Split: state = obs[:, :-24], lidar = obs[:, -24:]
    state = base_obs[:, :-24]
    lidar = base_obs[:, -24:]
    return np.concatenate([state, image_features, lidar], axis=1).astype(np.float32)


def load_demo_images(filepath):
    """Load raw camera images from a demo file.

    Args:
        filepath: Path to demo file (.hdf5 or .npz)

    Returns:
        numpy array (N, H, W, 3) uint8, or None if no images stored
    """
    data = open_demo(filepath)
    if 'images' in data:
        images = data['images']
        data.close()
        return images
    data.close()
    return None


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


def load_demo_transitions(npz_path: str, load_costs: bool = False,
                           load_images: bool = False):
    """Load demo data and reconstruct (obs, action, reward, next_obs, done) tuples.

    Uses the recorded ``dones`` from the NPZ file, which mark true MDP terminals
    (goal reached, collision, out-of-bounds) but NOT timeouts (truncations).
    This ensures correct Q-learning bootstrapping: ``Q_target = r + gamma * (1 - done) * Q(s', a')``.

    For backward compatibility with legacy NPZ files where ``dones`` only reflected
    ``goal_reached``, the last step of each episode falls back to ``done=1.0`` if
    no terminal was recorded.

    When ``load_images=True``, loads raw images from the demo file, encodes them
    via DINOv2, and inserts the 384D features into observations (before LiDAR),
    producing 418D or 420D observations.

    Args:
        npz_path: Path to .npz demo file
        load_costs: If True, also return costs array (6-tuple)
        load_images: If True, load camera images, encode with DINOv2, and expand obs

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
        # Use demo's actual arena bounds for conversion, not hardcoded defaults
        demo_arena = metadata.get('arena_size', None) if isinstance(metadata, dict) else None
        if demo_arena is not None:
            half = demo_arena / 2.0
            ws_bounds = {'x': [-half, half], 'y': [-half, half]}
            print(f"  Auto-converting demo obs from v{demo_obs_version} to v{OBS_VERSION} (ego-centric, arena={demo_arena}m)")
        else:
            ws_bounds = DEFAULT_WORKSPACE_BOUNDS
            print(f"  Auto-converting demo obs from v{demo_obs_version} to v{OBS_VERSION} (ego-centric, default bounds)")
        observations = convert_obs_to_egocentric(observations, ws_bounds)

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

    # Expand observations with DINOv2 features if camera images available
    if load_images:
        images = load_demo_images(npz_path)
        if images is not None:
            print(f"  Encoding {len(images)} camera images with DINOv2...")
            features = encode_images_dinov2(images)
            observations = build_camera_obs(observations, features)
            next_obs = build_camera_obs(next_obs, features)
            print(f"  Observations expanded: {observations.shape[1]}D (with camera features)")
        else:
            print("  Warning: --use-camera but demo has no images, using base obs")

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
                # Enhanced diagnostics
                self._ep_actions_lin = []   # |linear_vel| per step
                self._ep_actions_ang = []   # |angular_vel| per step
                self._ep_goal_dist = None   # goal distance at episode end
                self._ep_min_goal_dist = float('inf')  # closest to goal during episode
                self._ep_near_goal_steps = 0           # steps with gd < 0.5m
                self._diag_interval = 500   # Q-value probe interval
                # Rolling stats (last 20 episodes)
                from collections import deque
                self._recent_returns = deque(maxlen=20)
                self._recent_lengths = deque(maxlen=20)
                self._recent_outcomes = deque(maxlen=20)  # 'S','C','T'
                self._recent_goal_dists = deque(maxlen=20)
                self._recent_min_lidars = deque(maxlen=20)
                self._summary_ep_interval = 20

            def _on_training_start(self):
                import time
                self._start_time = time.time()
                self._start_step = self.num_timesteps
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
                    session_steps = total_steps - (self._start_step if hasattr(self, '_start_step') else 0)
                    rate = session_steps / elapsed if elapsed > 0 else 0

                    # Policy log_std (actual exploration noise)
                    policy_str = ""
                    try:
                        import torch as _th
                        with _th.no_grad():
                            log_std = self.model.actor.log_std
                            # log_std may be a Linear layer; get effective bias
                            if hasattr(log_std, 'bias'):
                                ls = log_std.bias.data.cpu().numpy()
                            else:
                                ls = log_std.data.cpu().numpy()
                            std_vals = np.exp(ls)
                            policy_str = f" | policy_std=[{std_vals.mean():.4f}]"
                    except Exception:
                        pass

                    # Entropy coef (try log_ent_coef first — CrossQ stores it there)
                    ent_str = ""
                    try:
                        import torch as _th2
                        with _th2.no_grad():
                            if hasattr(self.model, 'log_ent_coef'):
                                _ec = float(self.model.log_ent_coef.exp())
                            elif hasattr(self.model, 'ent_coef_tensor'):
                                _ec = float(self.model.ent_coef_tensor.item())
                            else:
                                _ec = None
                        if _ec is not None:
                            ent_str = f" | ent_coef={_ec:.5f}"
                    except Exception:
                        pass

                    # Pre-tanh mean magnitude (from stability patch)
                    mu_str = ""
                    try:
                        _lma = getattr(self.model.actor, '_last_mean_actions', None)
                        if _lma is not None:
                            mu_str = f" | |mu|={_lma.detach().abs().mean().item():.2f}"
                    except Exception:
                        pass

                    print(
                        f"[DEBUG] step={total_steps:>7d} | "
                        f"elapsed={elapsed:.1f}s | "
                        f"rate={rate:.1f} steps/s | "
                        f"episodes={self._ep_count}"
                        f"{policy_str}{ent_str}{mu_str}",
                        flush=True
                    )
                    self._last_report_step = total_steps

                # Q-value and buffer diagnostics (every 500 steps)
                if total_steps % self._diag_interval == 0 and total_steps > 0:
                    try:
                        import torch as _th
                        with _th.no_grad():
                            # Sample from replay buffer for diverse Q probe
                            # (single new_obs gives min=mean=max; buffer gives spread)
                            buf = self.model.replay_buffer
                            buf_online = buf.size() if hasattr(buf, 'size') else 0
                            n_demos = getattr(buf, 'n_demos', 0)
                            buf_cap = getattr(buf, 'buffer_size', 0)
                            obs_probe = None
                            try:
                                if buf_online + n_demos >= 32:
                                    rb_sample = buf.sample(32)
                                    obs_probe = rb_sample.observations
                            except Exception:
                                obs_probe = None
                            if obs_probe is None:
                                obs_probe = _th.as_tensor(
                                    self.locals['new_obs'],
                                    dtype=_th.float32,
                                    device=self.model.device,
                                )

                            act_pi, log_prob = self.model.actor.action_log_prob(obs_probe)
                            q_vals = self.model.critic(obs_probe, act_pi)
                            if isinstance(q_vals, (list, tuple)):
                                q_cat = _th.cat(q_vals, dim=1)
                                q_mean = q_cat.mean().item()
                                q_min_v = q_cat.min().item()
                                q_max_v = q_cat.max().item()
                                q_std = q_cat.std().item()
                            else:
                                q_mean = q_vals.mean().item()
                                q_min_v = q_vals.min().item()
                                q_max_v = q_vals.max().item()
                                q_std = q_vals.std().item()
                            entropy = -log_prob.mean().item()

                            # Pre-tanh mean magnitude (from stability patch)
                            _diag_mu_abs = None
                            _lma = getattr(self.model.actor, '_last_mean_actions', None)
                            if _lma is not None:
                                _diag_mu_abs = _lma.detach().abs().mean().item()

                            # ent_coef value
                            _diag_ent_coef = None
                            if hasattr(self.model, 'log_ent_coef'):
                                _diag_ent_coef = float(self.model.log_ent_coef.exp().item())

                            # Log to TensorBoard
                            self.model.logger.record(
                                "diag/Q_pi_mean", q_mean)
                            self.model.logger.record(
                                "diag/Q_pi_min", q_min_v)
                            self.model.logger.record(
                                "diag/Q_pi_max", q_max_v)
                            self.model.logger.record(
                                "diag/Q_pi_std", q_std)
                            self.model.logger.record(
                                "diag/policy_entropy", entropy)
                            self.model.logger.record(
                                "diag/buffer_online", buf_online)
                            if _diag_mu_abs is not None:
                                self.model.logger.record(
                                    "diag/mean_mu_abs", _diag_mu_abs)
                            if _diag_ent_coef is not None:
                                self.model.logger.record(
                                    "diag/ent_coef", _diag_ent_coef)

                            # Build extra fields for console
                            _mu_str = f" | |mu|={_diag_mu_abs:.2f}" if _diag_mu_abs is not None else ""
                            _ec_str = f" | ent_coef={_diag_ent_coef:.5f}" if _diag_ent_coef is not None else ""

                            src = "buf" if obs_probe.shape[0] == 32 else "obs"
                            print(
                                f"[DIAG]  step={total_steps:>7d} | "
                                f"Q_pi([{src}32])=[{q_min_v:+.1f}, {q_mean:+.1f}, "
                                f"{q_max_v:+.1f}] std={q_std:.2f} | "
                                f"H={entropy:+.2f}{_mu_str}{_ec_str} | "
                                f"buf={buf_online}/{buf_cap} "
                                f"demos={n_demos}",
                                flush=True,
                            )
                    except Exception:
                        pass

                for i in range(len(dones)):
                    info = infos[i] if i < len(infos) else {}
                    reward = float(rewards[i]) if i < len(rewards) else 0.0

                    self._ep_reward += reward
                    self._ep_steps += 1

                    min_lid = info.get('min_lidar_distance', float('inf'))
                    if min_lid < self._ep_min_lidar:
                        self._ep_min_lidar = min_lid

                    # Track goal distance and actions for diagnostics
                    self._ep_goal_dist = info.get(
                        'goal_distance', self._ep_goal_dist)
                    _gd_now = info.get('goal_distance')
                    if _gd_now is not None:
                        if _gd_now < self._ep_min_goal_dist:
                            self._ep_min_goal_dist = _gd_now
                        if _gd_now < 0.5:
                            self._ep_near_goal_steps += 1
                    actions = self.locals.get('actions')
                    if actions is not None:
                        act = np.asarray(
                            actions[i] if len(actions) > i else actions[0]
                        ).ravel()
                        if len(act) >= 2:
                            self._ep_actions_lin.append(
                                float(np.mean(np.abs(act[0::2]))))
                            self._ep_actions_ang.append(
                                float(np.mean(np.abs(act[1::2]))))


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

                        # Goal distance at episode end + min during episode
                        gd = self._ep_goal_dist
                        gd_str = f" | gd={gd:.2f}m" if gd is not None else ""
                        min_gd = self._ep_min_goal_dist
                        min_gd_str = (f" | min_gd={min_gd:.2f}m"
                                      if min_gd < float('inf') else "")
                        near_str = (f" | near={self._ep_near_goal_steps}"
                                    if self._ep_near_goal_steps > 0 else "")

                        # Action stats for this episode
                        act_str = ""
                        if self._ep_actions_lin:
                            ml = np.mean(self._ep_actions_lin)
                            ma = np.mean(self._ep_actions_ang)
                            act_str = f" | act=[{ml:.2f},{ma:.2f}]"

                        print(
                            f"[EP {self._ep_count:4d} END]  "
                            f"{outcome:>9s} | "
                            f"steps={self._ep_steps:3d} | "
                            f"return={self._ep_reward:+7.2f} | "
                            f"min_lidar={self._ep_min_lidar:.3f}m"
                            f"{gd_str}{min_gd_str}{near_str}{act_str} | "
                            f"running SR={sr:.1f}% "
                            f"({self._total_successes}S/"
                            f"{self._total_collisions}C/"
                            f"{self._total_truncations}T) | "
                            f"t={elapsed:.0f}s",
                            flush=True
                        )

                        # Update rolling stats
                        self._recent_returns.append(self._ep_reward)
                        self._recent_lengths.append(self._ep_steps)
                        self._recent_outcomes.append(outcome[0])
                        if gd is not None:
                            self._recent_goal_dists.append(gd)
                        self._recent_min_lidars.append(self._ep_min_lidar)

                        # TensorBoard per-episode metrics
                        try:
                            lg = self.model.logger
                            lg.record("rollout/ep_return", self._ep_reward)
                            lg.record("rollout/ep_length", self._ep_steps)
                            lg.record("rollout/ep_min_lidar",
                                      self._ep_min_lidar)
                            if gd is not None:
                                lg.record("rollout/ep_goal_dist", gd)
                            if min_gd < float('inf'):
                                lg.record("rollout/ep_min_goal_dist", min_gd)
                            lg.record("rollout/ep_near_goal_steps",
                                      self._ep_near_goal_steps)
                            n_r = len(self._recent_returns)
                            if n_r > 0:
                                lg.record("rollout/return_20ep",
                                          np.mean(self._recent_returns))
                                sr_20 = (sum(
                                    1 for o in self._recent_outcomes
                                    if o == 'S'
                                ) / n_r * 100)
                                lg.record("rollout/sr_20ep", sr_20)
                        except Exception:
                            pass

                        # Rolling summary every N episodes
                        if (self._ep_count % self._summary_ep_interval
                                == 0):
                            n = len(self._recent_returns)
                            if n > 0:
                                r_m = np.mean(self._recent_returns)
                                r_s = np.std(self._recent_returns)
                                l_m = np.mean(self._recent_lengths)
                                sr20 = (sum(
                                    1 for o in self._recent_outcomes
                                    if o == 'S'
                                ) / n * 100)
                                cr20 = (sum(
                                    1 for o in self._recent_outcomes
                                    if o == 'C'
                                ) / n * 100)
                                gd_m = (np.mean(self._recent_goal_dists)
                                        if self._recent_goal_dists
                                        else float('nan'))
                                ml_m = np.mean(self._recent_min_lidars)
                                print(
                                    f"[SUMMARY @EP {self._ep_count:4d}] "
                                    f"last{n}: "
                                    f"SR={sr20:.0f}% CR={cr20:.0f}% | "
                                    f"ret={r_m:+.1f}\u00b1{r_s:.1f} | "
                                    f"len={l_m:.0f} | "
                                    f"goal_dist={gd_m:.2f}m | "
                                    f"min_lid={ml_m:.3f}m",
                                    flush=True,
                                )

                        # Next episode start info
                        new_obs = self.locals.get('new_obs')
                        if new_obs is not None:
                            obs = new_obs[i] if len(new_obs) > i else new_obs[0]
                            # For frame-stacked obs, use last frame
                            if len(obs) > 34 and len(obs) % 34 == 0:
                                obs = obs[-34:]
                            elif len(obs) > 36 and len(obs) % 36 == 0:
                                obs = obs[-36:]
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

                        # Reset episode accumulators
                        self._ep_reward = 0.0
                        self._ep_steps = 0
                        self._ep_min_lidar = float('inf')
                        self._ep_goal_dist = None
                        self._ep_min_goal_dist = float('inf')
                        self._ep_near_goal_steps = 0
                        self._ep_actions_lin = []
                        self._ep_actions_ang = []

                return True

        return _VerboseEpisodeCallback()
