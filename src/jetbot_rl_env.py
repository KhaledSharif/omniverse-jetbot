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

import numpy as np

# Import gymnasium
import gymnasium
from gymnasium import spaces

# Import reusable components from jetbot_keyboard_control
from jetbot_keyboard_control import (
    ObservationBuilder,
    RewardComputer,
    SceneManager,
)


class JetbotNavigationEnv(gymnasium.Env):
    """Gymnasium environment for Jetbot navigation task in Isaac Sim.

    This environment wraps the Isaac Sim simulation of a Jetbot robot
    performing a point-to-point navigation task. The robot must navigate
    to a goal location.

    Observation Space (10D Box):
        [0:2]  - Robot position (x, y in meters)
        [2]    - Robot heading (theta in radians)
        [3]    - Linear velocity (m/s)
        [4]    - Angular velocity (rad/s)
        [5:7]  - Goal position (x, y in meters)
        [7]    - Distance to goal (meters)
        [8]    - Angle to goal (radians, relative to robot heading)
        [9]    - Goal reached flag (0.0 or 1.0)

    Action Space (2D Box, continuous [-1, 1]):
        [0]    - Linear velocity command (normalized)
        [1]    - Angular velocity command (normalized)

    Attributes:
        reward_mode: 'dense' for shaped rewards, 'sparse' for goal completion only
        max_episode_steps: Maximum steps before episode truncation
    """

    metadata = {'render_modes': ['human']}

    # Action scaling factors (from continuous [-1, 1] to physical units)
    MAX_LINEAR_VELOCITY = 0.3   # m/s
    MAX_ANGULAR_VELOCITY = 1.0  # rad/s

    # Jetbot physical parameters
    WHEEL_RADIUS = 0.03    # meters
    WHEEL_BASE = 0.1125    # meters

    # Start position
    START_POSITION = np.array([0.0, 0.0, 0.05])
    START_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion (w, x, y, z)

    # Workspace bounds
    DEFAULT_WORKSPACE_BOUNDS = {
        'x': [-2.0, 2.0],
        'y': [-2.0, 2.0],
    }

    def __init__(
        self,
        reward_mode: str = 'dense',
        max_episode_steps: int = 500,
        workspace_bounds: dict = None,
        render_mode: str = 'human',
        headless: bool = False,
        goal_threshold: float = 0.15,
    ):
        """Initialize the Jetbot navigation environment.

        Args:
            reward_mode: 'dense' for shaped rewards, 'sparse' for goal completion only
            max_episode_steps: Maximum steps before episode truncation
            workspace_bounds: Dict with 'x', 'y' bounds for workspace
            render_mode: Render mode ('human' for always rendering)
            headless: If True, run simulation without GUI (faster for training)
            goal_threshold: Distance threshold for considering goal reached
        """
        super().__init__()

        # Store configuration
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        self.workspace_bounds = workspace_bounds or self.DEFAULT_WORKSPACE_BOUNDS.copy()
        self.render_mode = render_mode
        self.headless = headless
        self.goal_threshold = goal_threshold

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
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
        self.obs_builder = ObservationBuilder()
        self.reward_computer = RewardComputer(mode=reward_mode)
        self.scene_manager = SceneManager(self.world, workspace_bounds=self.workspace_bounds)

        # Spawn initial goal marker
        self.scene_manager.spawn_goal_marker()

        # Initialize tracking state
        self._step_count = 0
        self._prev_obs = None
        self._current_linear_vel = 0.0
        self._current_angular_vel = 0.0

    def _init_isaac_sim(self):
        """Initialize Isaac Sim simulation environment."""
        global simulation_app, World, ArticulationAction, WheeledRobot
        global DifferentialController, get_assets_root_path

        # Create SimulationApp if not already created
        if simulation_app is None:
            simulation_app = SimulationApp({"headless": self.headless})
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

        # Create world
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

    def _get_robot_pose(self) -> tuple:
        """Get current robot position and heading.

        Returns:
            Tuple of (position, heading) where position is [x, y, z] and heading is radians
        """
        position, orientation = self.jetbot.get_world_pose()

        # Convert quaternion to heading angle (yaw)
        w, x, y, z = orientation
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        heading = np.arctan2(siny_cosp, cosy_cosp)

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

        # Reset robot to start position
        self.jetbot.set_world_pose(
            position=self.START_POSITION,
            orientation=self.START_ORIENTATION
        )

        # Reset velocities
        self._current_linear_vel = 0.0
        self._current_angular_vel = 0.0

        # Reset goal to new random position
        self.scene_manager.reset_goal()

        # Step simulation to settle physics
        for _ in range(10):
            self.world.step(render=not self.headless)

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

        # Step simulation
        self.world.step(render=not self.headless)

        # Increment step counter
        self._step_count += 1

        # Build next observation
        next_obs = self._build_observation()

        # Check goal reached
        position, _ = self._get_robot_pose()
        goal_reached = self.scene_manager.check_goal_reached(position, threshold=self.goal_threshold)

        # Build info dict
        info = {
            'goal_reached': goal_reached,
            'is_success': goal_reached,
        }

        # Compute reward
        reward = self.reward_computer.compute(self._prev_obs, action, next_obs, info)

        # Check termination and truncation
        terminated = self._check_termination(info, position)
        truncated = self._check_truncation()

        # Update previous observation
        self._prev_obs = next_obs.copy()

        return next_obs, reward, terminated, truncated, info

    def _build_observation(self):
        """Build observation vector from current state.

        Returns:
            10D observation vector as float32 numpy array
        """
        # Get robot state
        position, heading = self._get_robot_pose()

        # Get goal state
        goal_position = self.scene_manager.get_goal_position()
        if goal_position is None:
            goal_position = np.zeros(3)

        # Check goal reached
        goal_reached = self.scene_manager.check_goal_reached(position, threshold=self.goal_threshold)

        return self.obs_builder.build(
            robot_position=position,
            robot_heading=heading,
            linear_velocity=self._current_linear_vel,
            angular_velocity=self._current_angular_vel,
            goal_position=goal_position,
            goal_reached=goal_reached
        )

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
