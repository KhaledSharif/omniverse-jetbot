"""
Gymnasium environment wrapper for Jetbot navigation task in Isaac Sim.

This module provides a Gymnasium-compatible environment for training RL agents
on a navigation task using the Jetbot robot in NVIDIA Isaac Sim.

Usage:
    from jetbot_rl_env import JetbotNavigationEnv
    env = JetbotNavigationEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

Compatible with Stable-Baselines3 and other Gymnasium-compatible RL libraries.
"""

from isaacsim import SimulationApp

# Module-level globals (set after SimulationApp is created)
simulation_app = None
World = None
ArticulationAction = None
WheeledRobot = None
DifferentialController = None
get_assets_root_path = None

import time
import numpy as np

# Import gymnasium
import gymnasium
from gymnasium import spaces

# Import reusable components from jetbot_keyboard_control
from jetbot_keyboard_control import (
    LidarSensor,
    ObservationBuilder,
    OccupancyGrid,
    RewardComputer,
    SceneManager,
    astar_search,
)
from jetbot_config import (
    WHEEL_RADIUS, WHEEL_BASE,
    MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    START_POSITION, START_ORIENTATION,
    DEFAULT_WORKSPACE_BOUNDS,
    CAMERA_PRIM_SUFFIX, CAMERA_WIDTH, CAMERA_HEIGHT, IMAGE_FEATURE_DIM,
    quaternion_to_yaw,
)


class JetbotNavigationEnv(gymnasium.Env):
    """Gymnasium environment for Jetbot navigation task in Isaac Sim.

    This environment wraps the Isaac Sim simulation of a Jetbot robot
    performing a point-to-point navigation task. The robot must navigate
    to a goal location while avoiding obstacles using LiDAR sensing.

    Observation Space (34D Box, ego-centric):
        [0]     - Normalized workspace x: (x - x_min) / (x_max - x_min)
        [1]     - Normalized workspace y: (y - y_min) / (y_max - y_min)
        [2]     - sin(heading)
        [3]     - cos(heading)
        [4]     - Linear velocity (m/s)
        [5]     - Angular velocity (rad/s)
        [6]     - Goal body-frame x: dist * cos(angle_to_goal)
        [7]     - Goal body-frame y: dist * sin(angle_to_goal)
        [8]     - Distance to goal (meters)
        [9]     - Goal reached flag (0.0 or 1.0)
        [10:34] - LiDAR: 24 normalized distances (0=touching, 1=max range)

    With --add-prev-action (36D Box):
        [0:10]  - Same as above
        [10]    - Previous linear velocity command
        [11]    - Previous angular velocity command
        [12:36] - LiDAR: 24 normalized distances

    Action Space (2D Box, continuous [-1, 1]):
        [0]    - Linear velocity command (normalized)
        [1]    - Angular velocity command (normalized)

    Attributes:
        reward_mode: 'dense' for shaped rewards, 'sparse' for goal completion only
        max_episode_steps: Maximum steps before episode truncation
    """

    metadata = {'render_modes': ['human']}

    # Re-export from jetbot_config for backwards compatibility
    MAX_LINEAR_VELOCITY = MAX_LINEAR_VELOCITY
    MAX_ANGULAR_VELOCITY = MAX_ANGULAR_VELOCITY
    WHEEL_RADIUS = WHEEL_RADIUS
    WHEEL_BASE = WHEEL_BASE
    START_POSITION = START_POSITION
    START_ORIENTATION = START_ORIENTATION
    DEFAULT_WORKSPACE_BOUNDS = DEFAULT_WORKSPACE_BOUNDS

    # LiDAR configuration
    NUM_LIDAR_RAYS = 24
    LIDAR_FOV_DEG = 180.0
    LIDAR_MAX_RANGE = 3.0
    COLLISION_THRESHOLD = 0.08  # Robot effective radius

    def __init__(
        self,
        reward_mode: str = 'dense',
        max_episode_steps: int = 500,
        workspace_bounds: dict = None,
        render_mode: str = 'human',
        headless: bool = False,
        goal_threshold: float = 0.15,
        num_obstacles: int = 5,
        min_goal_dist: float = 0.5,
        inflation_radius: float = 0.08,
        cost_type: str = 'proximity',
        safe_mode: bool = False,
        add_prev_action: bool = False,
        use_camera: bool = False,
    ):
        """Initialize the Jetbot navigation environment.

        Args:
            reward_mode: 'dense' for shaped rewards, 'sparse' for goal completion only
            max_episode_steps: Maximum steps before episode truncation
            workspace_bounds: Dict with 'x', 'y' bounds for workspace
            render_mode: Render mode ('human' for always rendering)
            headless: If True, run simulation without GUI (faster for training)
            goal_threshold: Distance threshold for considering goal reached
            num_obstacles: Number of obstacles to spawn (default: 5)
            min_goal_dist: Minimum distance from robot start to goal (meters)
            inflation_radius: Obstacle inflation radius for A* solvability checks (meters)
            use_camera: If True, add 384D DINOv2 features to observations
        """
        super().__init__()

        # Store configuration
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS.copy()
        self.render_mode = render_mode
        self.headless = headless
        self.goal_threshold = goal_threshold
        self.num_obstacles = num_obstacles
        self.min_goal_dist = min_goal_dist
        self.inflation_radius = inflation_radius
        self.cost_type = cost_type
        self.safe_mode = safe_mode
        self.add_prev_action = add_prev_action
        self.use_camera = use_camera

        # Create LiDAR sensor
        self.lidar_sensor = LidarSensor(
            num_rays=self.NUM_LIDAR_RAYS,
            fov_deg=self.LIDAR_FOV_DEG,
            max_range=self.LIDAR_MAX_RANGE
        )

        # Define observation and action spaces
        # 10 base + optional 2 prev_action + optional 384 image features + 24 LiDAR
        obs_dim = (10
                   + (2 if self.add_prev_action else 0)
                   + (IMAGE_FEATURE_DIM if self.use_camera else 0)
                   + self.NUM_LIDAR_RAYS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Initialize Isaac Sim
        self._init_isaac_sim()

        # Initialize helper components
        self.obs_builder = ObservationBuilder(lidar_sensor=self.lidar_sensor)
        self.reward_computer = RewardComputer(mode=reward_mode, safe_mode=safe_mode)
        self.scene_manager = SceneManager(
            self.world,
            workspace_bounds=self.workspace_bounds,
            num_obstacles=self.num_obstacles,
            min_goal_dist=self.min_goal_dist,
            robot_radius=self.inflation_radius,
        )

        # Spawn initial goal marker
        self.scene_manager.spawn_goal_marker()

        # Initialize tracking state
        self._step_count = 0
        self._prev_obs = None
        self._current_linear_vel = 0.0
        self._current_angular_vel = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Camera state
        self._last_camera_frame = None
        self._camera_features = None

        # Timing accumulators for physics step profiling
        self._wstep_ms_acc = 0.0
        self._wstep_n = 0

    def _init_isaac_sim(self):
        """Initialize Isaac Sim simulation environment."""
        global simulation_app, World, ArticulationAction, WheeledRobot
        global DifferentialController, get_assets_root_path

        # Create SimulationApp if not already created
        if simulation_app is None:
            config = {"headless": self.headless}
            if self.headless and not self.use_camera:
                config["disable_viewport_updates"] = True
                config["anti_aliasing"] = 0
                config["samples_per_pixel_per_frame"] = 1
            elif self.headless and self.use_camera:
                # Camera needs rendering even in headless; enable_cameras is required
                config["enable_cameras"] = True
                config["anti_aliasing"] = 0
            simulation_app = SimulationApp(config)

            # Disable rendering subsystems in headless mode
            if self.headless:
                import carb
                settings = carb.settings.get_settings()
                settings.set_bool("/rtx-transient/resourcemanager/texturestreaming/enabled", False)
                settings.set_int("/persistent/physics/numThreads", 0)  # Single-threaded (faster for 1 robot)
        self.simulation_app = simulation_app

        # Import Isaac Sim modules after SimulationApp is created
        if World is None:
            from isaacsim.core.api import World as _World
            from isaacsim.core.utils.types import ArticulationAction as _ArticulationAction
            from isaacsim.robot.wheeled_robots.robots import WheeledRobot as _WheeledRobot
            from isaacsim.robot.wheeled_robots.controllers import DifferentialController as _DifferentialController
            from isaacsim.core.utils.nucleus import get_assets_root_path as _get_assets_root_path

            World = _World
            ArticulationAction = _ArticulationAction
            WheeledRobot = _WheeledRobot
            DifferentialController = _DifferentialController
            get_assets_root_path = _get_assets_root_path

        # Create world — decouple rendering from physics in headless mode
        if self.headless and not self.use_camera:
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0 / 60.0,
                rendering_dt=1.0,  # Effectively never render
            )
        else:
            self.world = World(stage_units_in_meters=1.0)

        # Get assets root path
        assets_root = get_assets_root_path()

        # Create Jetbot
        self.jetbot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="jetbot_rl",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
                position=self.START_POSITION,
                orientation=self.START_ORIENTATION
            )
        )

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Create differential controller
        self.controller = DifferentialController(
            name="jetbot_controller",
            wheel_radius=self.WHEEL_RADIUS,
            wheel_base=self.WHEEL_BASE
        )

        # Reset world to initialize physics
        self.world.reset()

        # Optimize physics scene for RL training (single robot, flat ground)
        if self.headless and not self.use_camera:
            self._optimize_physics_scene()

        # Initialize camera + DINOv2 if enabled
        if self.use_camera:
            self._init_camera()
            self._init_dinov2()

    def _optimize_physics_scene(self):
        """Tune PhysX scene for single-robot RL: CPU broadphase, fewer solver iters."""
        try:
            from pxr import UsdPhysics, PhysxSchema
            stage = self.world.stage
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.Scene):
                    physx_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    # CPU dynamics is faster for single-robot scenes (no GPU transfer)
                    physx_api.GetEnableGPUDynamicsAttr().Set(False)
                    # MBP broadphase is better for small, static scenes
                    physx_api.GetBroadphaseTypeAttr().Set("MBP")
                    # Minimal solver iterations (flat ground + differential drive)
                    physx_api.GetMinPositionIterationCountAttr().Set(1)
                    physx_api.GetMaxPositionIterationCountAttr().Set(2)
                    physx_api.GetMinVelocityIterationCountAttr().Set(0)
                    physx_api.GetMaxVelocityIterationCountAttr().Set(1)
                    print("  Physics scene optimized: CPU dynamics, MBP broadphase, reduced solver iters")
                    break
        except Exception as e:
            print(f"  Warning: Could not optimize physics scene: {e}")

    def _init_camera(self):
        """Initialize the Isaac Sim camera sensor on the Jetbot's camera prim."""
        from isaacsim.sensors.camera import Camera

        camera_prim_path = "/World/Jetbot" + CAMERA_PRIM_SUFFIX
        self._camera = Camera(
            prim_path=camera_prim_path,
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            name="jetbot_rl_camera",
        )
        self._camera.initialize()
        # Warm-up: render a few frames so camera has valid output
        for _ in range(5):
            self.world.step(render=True)
        self._last_camera_frame = None
        print(f"  Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} on {camera_prim_path}")

    def _init_dinov2(self):
        """Load frozen DINOv2 ViT-S/14 and pre-compute normalization tensors."""
        import torch

        self._dinov2_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dinov2_model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14',
        )
        self._dinov2_model = self._dinov2_model.to(self._dinov2_device)
        self._dinov2_model.eval()
        for p in self._dinov2_model.parameters():
            p.requires_grad = False

        # ImageNet normalization tensors (C, 1, 1) for broadcasting
        self._dinov2_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self._dinov2_device
        ).view(3, 1, 1)
        self._dinov2_std = torch.tensor(
            [0.229, 0.224, 0.225], device=self._dinov2_device
        ).view(3, 1, 1)
        print(f"  DINOv2 ViT-S/14 loaded on {self._dinov2_device} "
              f"(output: {IMAGE_FEATURE_DIM}D)")

    def _capture_camera_features(self):
        """Capture camera frame, run through DINOv2, return features + raw frame.

        Returns:
            Tuple of (features_384D_np, raw_rgb_uint8)
        """
        import torch

        # Get RGBA from camera, convert to RGB uint8
        rgba = self._camera.get_rgba()
        if rgba is None or rgba.size == 0:
            # Fallback: return zeros if camera not ready
            self._last_camera_frame = np.zeros(
                (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32), self._last_camera_frame
        rgb = rgba[:, :, :3].astype(np.uint8)
        self._last_camera_frame = rgb.copy()

        # Convert to tensor: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
        img_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self._dinov2_device) / 255.0
        # ImageNet normalization
        img_tensor = (img_tensor - self._dinov2_mean) / self._dinov2_std

        # DINOv2 forward: CLS token -> (1, 384)
        with torch.no_grad():
            features = self._dinov2_model(img_tensor)  # (1, 384)
        features_np = features.squeeze(0).cpu().numpy().astype(np.float32)
        return features_np, self._last_camera_frame

    @property
    def last_camera_frame(self):
        """Last raw RGB uint8 frame captured by the camera (for demo recording)."""
        return self._last_camera_frame

    def _get_robot_pose(self) -> tuple:
        """Get current robot position and heading.

        Returns:
            Tuple of (position, heading) where position is [x, y, z] and heading is radians
        """
        position, orientation = self.jetbot.get_world_pose()
        heading = quaternion_to_yaw(orientation)
        return position, heading

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation (10D numpy array)
            info: Information dictionary
        """
        super().reset(seed=seed)

        # Randomize start heading for domain randomization
        yaw = self.np_random.uniform(0, 2 * np.pi)
        orientation = np.array([
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        ])  # quaternion (w, x, y, z)

        # Reset robot to start position with random heading
        self.jetbot.set_world_pose(
            position=self.START_POSITION,
            orientation=orientation
        )

        # Reset velocities
        self._current_linear_vel = 0.0
        self._current_angular_vel = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Solvability loop: retry goal + obstacles until A* finds a path
        max_solvability_attempts = 20
        robot_xy = tuple(self.START_POSITION[:2])
        for attempt in range(max_solvability_attempts):
            self.scene_manager.reset_goal()
            grid = OccupancyGrid.from_scene(
                self.scene_manager.get_obstacle_metadata(),
                self.workspace_bounds,
                robot_radius=self.inflation_radius,
            )
            goal_pos = self.scene_manager.get_goal_position()
            path = astar_search(grid, robot_xy, tuple(goal_pos[:2]))
            if path:
                break
        else:
            print(f"Warning: no solvable layout found after {max_solvability_attempts} attempts, using last layout")

        # Step simulation to settle physics (render required for camera)
        _render = not self.headless or self.use_camera
        _settle_steps = 5 if self.use_camera else 2
        for _ in range(_settle_steps):
            self.world.step(render=_render)

        # Reset tracking state
        self._step_count = 0

        # Build initial observation
        observation = self._build_observation()
        self._prev_obs = observation.copy()

        # Build info dictionary
        goal_pos = self.scene_manager.get_goal_position()
        position, heading = self._get_robot_pose()

        info = {
            'robot_position': np.array(position),
            'goal_position': np.array(goal_pos) if goal_pos is not None else np.zeros(3),
            'is_success': False,
        }

        return observation, info

    def step(self, action):
        """Execute one environment step.

        Args:
            action: 2D numpy array with values in [-1, 1]
                    [0] = linear velocity (normalized)
                    [1] = angular velocity (normalized)

        Returns:
            observation: Next observation (10D numpy array)
            reward: Scalar reward
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was cut short (time limit)
            info: Information dictionary
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Scale action from [-1, 1] to physical units
        self._current_linear_vel = action[0] * self.MAX_LINEAR_VELOCITY
        self._current_angular_vel = action[1] * self.MAX_ANGULAR_VELOCITY

        # Apply control
        wheel_actions = self.controller.forward(
            command=[self._current_linear_vel, self._current_angular_vel]
        )
        self.jetbot.apply_wheel_actions(wheel_actions)

        # Step simulation (render required for camera even in headless mode)
        _render = not self.headless or self.use_camera
        _wt0 = time.perf_counter()
        self.world.step(render=_render)
        self._wstep_ms_acc += (time.perf_counter() - _wt0) * 1000
        self._wstep_n += 1
        if self._wstep_n % 1000 == 0:
            print(
                f"[TIMING] world.step(): {self._wstep_ms_acc / self._wstep_n:.2f}ms avg "
                f"({self._wstep_n} calls)", flush=True
            )
            self._wstep_ms_acc = 0.0
            self._wstep_n = 0

        # Increment step counter
        self._step_count += 1

        # Build next observation
        next_obs = self._build_observation()

        # Check goal reached
        position, _ = self._get_robot_pose()
        goal_reached = self.scene_manager.check_goal_reached(position, threshold=self.goal_threshold)

        # Extract min LiDAR distance from the last 24D of the observation
        lidar_readings = next_obs[-self.NUM_LIDAR_RAYS:]  # Normalized LiDAR values
        min_lidar = float(lidar_readings.min()) * self.lidar_sensor.max_range
        collision = min_lidar < self.COLLISION_THRESHOLD

        # Build info dict
        goal_distance = float(next_obs[8])
        info = {
            'goal_reached': goal_reached,
            'is_success': goal_reached,
            'collision': collision,
            'min_lidar_distance': min_lidar,
            'goal_distance': goal_distance,
        }

        # Compute constraint cost for SafeTQC
        info['cost'] = RewardComputer.compute_cost(info, cost_type=self.cost_type)

        # Compute reward
        reward = self.reward_computer.compute(self._prev_obs, action, next_obs, info)

        # Check termination and truncation
        terminated = self._check_termination(info, position)
        truncated = self._check_truncation()

        # Update previous observation and action
        self._prev_obs = next_obs.copy()
        self._prev_action = np.clip(action, -1.0, 1.0).copy()

        return next_obs, reward, terminated, truncated, info

    def _build_observation(self):
        """Build observation vector from current state.

        Returns:
            34D/36D (no camera) or 418D/420D (with camera) observation as float32 numpy array.
            Layout: [state, (prev_action), (image_features_384D), lidar_24D]
        """
        # Get robot state
        position, heading = self._get_robot_pose()

        # Get goal state
        goal_position = self.scene_manager.get_goal_position()
        if goal_position is None:
            goal_position = np.zeros(3)

        # Check goal reached
        goal_reached = self.scene_manager.check_goal_reached(position, threshold=self.goal_threshold)

        obs = self.obs_builder.build(
            robot_position=position,
            robot_heading=heading,
            linear_velocity=self._current_linear_vel,
            angular_velocity=self._current_angular_vel,
            goal_position=goal_position,
            goal_reached=goal_reached,
            obstacle_metadata=self.scene_manager.get_obstacle_metadata(),
            workspace_bounds=self.workspace_bounds
        )

        # Split into base state and LiDAR
        base = obs[:10]
        lidar = obs[10:]

        parts = [base]

        # Insert prev_action after base obs
        if self.add_prev_action:
            parts.append(self._prev_action)

        # Insert camera features before LiDAR
        if self.use_camera:
            features, _ = self._capture_camera_features()
            self._camera_features = features
            parts.append(features)

        parts.append(lidar)
        return np.concatenate(parts).astype(np.float32)

    def _check_termination(self, info, position):
        """Check if episode should terminate (success or failure).

        Args:
            info: Info dictionary from step
            position: Current robot position

        Returns:
            True if episode should terminate, False otherwise
        """
        # SUCCESS: Goal reached
        if info.get('goal_reached', False):
            return True

        # FAILURE: Collision with obstacle
        if info.get('collision', False):
            return True

        # FAILURE: Robot out of workspace bounds
        if self._is_out_of_bounds(position):
            return True

        return False

    def _check_truncation(self):
        """Check if episode should be truncated (time limit).

        Returns:
            True if episode should be truncated, False otherwise
        """
        return self._step_count >= self.max_episode_steps

    def _is_out_of_bounds(self, position):
        """Check if position is outside workspace bounds with margin.

        Args:
            position: [x, y, z] position

        Returns:
            True if position is out of bounds, False otherwise
        """
        margin = 0.5  # Allow some margin beyond workspace
        bounds = self.workspace_bounds

        return (
            position[0] < bounds['x'][0] - margin or
            position[0] > bounds['x'][1] + margin or
            position[1] < bounds['y'][0] - margin or
            position[1] > bounds['y'][1] + margin
        )

    def close(self):
        """Clean up resources."""
        if self.simulation_app is not None:
            self.simulation_app.close()


class FrameStackWrapper(gymnasium.Wrapper):
    """Wrapper that stacks the last n_frames observations into a single vector.

    Maintains a ring buffer of recent observations. On reset(), all slots are
    filled with the initial observation. On step(), the newest observation
    replaces the oldest. Output is flattened: (n_frames * obs_dim,).

    Frame order: oldest first (index 0), newest last — natural for GRU input.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self._obs_dim = env.observation_space.shape[0]

        # Tile observation space bounds
        low = np.tile(env.observation_space.low, n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

        # Ring buffer
        self._frames = np.zeros((n_frames, self._obs_dim), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.n_frames):
            self._frames[i] = obs
        return self._frames.flatten().copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Shift left and append new obs
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = obs
        return self._frames.flatten().copy(), reward, terminated, truncated, info


class ChunkedEnvWrapper(gymnasium.Wrapper):
    """Wrapper that converts single-step actions to k-step action chunks.

    The wrapper expands the action space from (2,) to (k*2,) and executes
    k inner env steps per wrapper step. Rewards are accumulated as a
    discounted sum: R_chunk = sum(gamma^i * r_i for i in range(k)).

    This enables Q-chunking: the critic evaluates chunk-level Q-values,
    and the Bellman target uses gamma^k as the effective discount.
    """

    def __init__(self, env, chunk_size, gamma=0.99):
        super().__init__(env)
        self.chunk_size = chunk_size
        self.gamma = gamma

        # Expand action space: (2,) → (k*2,)
        low = np.tile(env.action_space.low, chunk_size)
        high = np.tile(env.action_space.high, chunk_size)
        self.action_space = spaces.Box(
            low=low, high=high, dtype=env.action_space.dtype
        )

        # Timing accumulators for chunk step profiling
        self._chunk_ms_acc = 0.0
        self._substep_ms_acc = 0.0
        self._chunk_n = 0

    def step(self, action_flat):
        """Execute k inner steps with the action chunk.

        Args:
            action_flat: flat action array of shape (k*2,)

        Returns:
            (obs, R_chunk, terminated, truncated, info) after k inner steps
        """
        actions = action_flat.reshape(self.chunk_size, -1)

        r_chunk = 0.0
        c_chunk = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}

        _chunk_t0 = time.perf_counter()
        for i, act in enumerate(actions):
            _sub_t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = self.env.step(act)
            self._substep_ms_acc += (time.perf_counter() - _sub_t0) * 1000
            r_chunk += (self.gamma ** i) * reward
            c_chunk += (self.gamma ** i) * info.get('cost', 0.0)
            if terminated or truncated:
                break

        self._chunk_ms_acc += (time.perf_counter() - _chunk_t0) * 1000
        self._chunk_n += 1
        if self._chunk_n % 500 == 0:
            _chunk_avg = self._chunk_ms_acc / self._chunk_n
            _sub_avg = self._substep_ms_acc / (self._chunk_n * self.chunk_size)
            print(
                f"[TIMING] ChunkedWrapper.step(): total={_chunk_avg:.1f}ms | "
                f"inner env.step()={_sub_avg:.1f}ms avg | "
                f"overhead={_chunk_avg - _sub_avg * self.chunk_size:.1f}ms "
                f"(chunk_size={self.chunk_size}, n={self._chunk_n})",
                flush=True,
            )
            self._chunk_ms_acc = 0.0
            self._substep_ms_acc = 0.0
            self._chunk_n = 0

        info['cost_chunk'] = c_chunk
        return obs, r_chunk, terminated, truncated, info
