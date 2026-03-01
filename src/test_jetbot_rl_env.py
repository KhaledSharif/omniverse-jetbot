"""
Test Suite for Jetbot RL Environment

This test suite covers:
- Environment initialization
- Observation and action space verification
- Step function behavior
- Reset function behavior
- Reward computation

Run with: pytest test_jetbot_rl_env.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing jetbot_rl_env)
# ============================================================================

# Create mock modules for isaacsim
mock_isaacsim = MagicMock()
mock_isaacsim_core = MagicMock()
mock_isaacsim_core_api = MagicMock()
mock_isaacsim_core_utils = MagicMock()
mock_isaacsim_core_utils_types = MagicMock()
mock_isaacsim_core_utils_nucleus = MagicMock()
mock_isaacsim_robot = MagicMock()
mock_isaacsim_robot_wheeled_robots = MagicMock()
mock_isaacsim_robot_wheeled_robots_robots = MagicMock()
mock_isaacsim_robot_wheeled_robots_controllers = MagicMock()

# Mock SimulationApp class
mock_simulation_app_class = MagicMock()
mock_isaacsim.SimulationApp = mock_simulation_app_class

# Mock ArticulationAction class
class MockArticulationAction:
    """Mock ArticulationAction that stores joint_velocities."""
    def __init__(self, joint_velocities=None, **kwargs):
        self.joint_velocities = joint_velocities if joint_velocities is not None else np.array([])

mock_isaacsim_core_utils_types.ArticulationAction = MockArticulationAction

# Mock get_assets_root_path
mock_isaacsim_core_utils_nucleus.get_assets_root_path = Mock(return_value="/Isaac")

# Register all mocks in sys.modules before any imports
sys.modules['isaacsim'] = mock_isaacsim
sys.modules['isaacsim.core'] = mock_isaacsim_core
sys.modules['isaacsim.core.api'] = mock_isaacsim_core_api
sys.modules['isaacsim.core.utils'] = mock_isaacsim_core_utils
sys.modules['isaacsim.core.utils.types'] = mock_isaacsim_core_utils_types
sys.modules['isaacsim.core.utils.nucleus'] = mock_isaacsim_core_utils_nucleus
sys.modules['isaacsim.robot'] = mock_isaacsim_robot
sys.modules['isaacsim.robot.wheeled_robots'] = mock_isaacsim_robot_wheeled_robots
sys.modules['isaacsim.robot.wheeled_robots.robots'] = mock_isaacsim_robot_wheeled_robots_robots
sys.modules['isaacsim.robot.wheeled_robots.controllers'] = mock_isaacsim_robot_wheeled_robots_controllers


# ============================================================================
# TEST SUITE: Observation and Action Spaces
# ============================================================================

class TestSpaces:
    """Tests for observation and action space definitions."""

    def test_observation_space_shape_without_lidar(self):
        """Test observation space has 10D without lidar."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()
        assert builder.obs_dim == 10

    def test_observation_space_shape_with_lidar(self):
        """Test observation space has 34D with lidar."""
        from jetbot_keyboard_control import ObservationBuilder, LidarSensor
        lidar = LidarSensor(num_rays=24)
        builder = ObservationBuilder(lidar_sensor=lidar)
        assert builder.obs_dim == 34

    def test_action_space_shape(self):
        """Test action space has correct shape."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        assert mapper.action_dim == 2

    def test_observation_building_without_lidar(self):
        """Test observation can be built correctly without lidar."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()

        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.05]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([1.0, 1.0, 0.0]),
            goal_reached=False
        )

        assert obs.shape == (10,)
        assert obs.dtype == np.float32

    def test_observation_building_with_lidar(self):
        """Test observation building with lidar produces 34D."""
        from jetbot_keyboard_control import ObservationBuilder, LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
        builder = ObservationBuilder(lidar_sensor=lidar)

        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.05]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([1.0, 1.0, 0.0]),
            goal_reached=False,
            obstacle_metadata=[],
            workspace_bounds={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
        )

        assert obs.shape == (34,)
        assert obs.dtype == np.float32
        # LiDAR portion should be in [0, 1]
        assert np.all(obs[10:] >= 0.0)
        assert np.all(obs[10:] <= 1.0)


# ============================================================================
# TEST SUITE: Reward Computation
# ============================================================================

class TestRewardComputation:
    """Tests for reward computation in RL environment."""

    def test_sparse_reward_no_goal(self):
        """Test sparse reward returns 0 when goal not reached."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='sparse')

        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': False}
        )

        assert reward == 0.0

    def test_sparse_reward_goal_reached(self):
        """Test sparse reward returns bonus when goal reached."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='sparse')

        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': True}
        )

        assert reward == RewardComputer.GOAL_REACHED_REWARD

    def test_dense_reward_progress(self):
        """Test dense reward increases when getting closer to goal."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        # Create observations where robot gets closer to goal
        prev_obs = np.zeros(10)
        prev_obs[8] = 2.0  # Distance to goal

        next_obs = np.zeros(10)
        next_obs[8] = 1.0  # Closer to goal
        next_obs[6] = 1.0  # goal_body_x (facing goal, angle=0)
        next_obs[7] = 0.0  # goal_body_y

        reward = computer.compute(
            obs=prev_obs,
            action=np.zeros(2),
            next_obs=next_obs,
            info={'goal_reached': False}
        )

        # Should get positive reward for progress
        assert reward > 0

    def test_dense_reward_regression(self):
        """Test dense reward decreases when getting farther from goal."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        # Create observations where robot gets farther from goal
        prev_obs = np.zeros(10)
        prev_obs[8] = 1.0   # Distance to goal

        next_obs = np.zeros(10)
        next_obs[8] = 2.0   # Farther from goal
        next_obs[6] = -2.0  # goal_body_x = dist*cos(pi) = -2.0
        next_obs[7] = 0.0   # goal_body_y = dist*sin(pi) ≈ 0

        reward = computer.compute(
            obs=prev_obs,
            action=np.zeros(2),
            next_obs=next_obs,
            info={'goal_reached': False}
        )

        # Should get negative or low reward for regression
        # (exact value depends on heading bonus calculation)
        assert reward < 1.0  # At least less than distance scale

    def test_collision_terminates_episode(self):
        """Test that collision causes episode termination."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        reward = computer.compute(
            obs=np.zeros(10), action=np.zeros(2), next_obs=np.zeros(10),
            info={'goal_reached': False, 'collision': True}
        )
        assert reward == RewardComputer.COLLISION_PENALTY

    def test_collision_reward_in_sparse_mode(self):
        """Test collision penalty in sparse mode."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='sparse')

        reward = computer.compute(
            obs=np.zeros(10), action=np.zeros(2), next_obs=np.zeros(10),
            info={'goal_reached': False, 'collision': True}
        )
        assert reward == RewardComputer.COLLISION_PENALTY


# ============================================================================
# TEST SUITE: Scene Manager
# ============================================================================

class TestSceneManagerRL:
    """Tests for SceneManager in RL context."""

    def test_workspace_bounds(self):
        """Test workspace bounds are set correctly."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)

        assert 'x' in manager.workspace_bounds
        assert 'y' in manager.workspace_bounds
        assert len(manager.workspace_bounds['x']) == 2
        assert len(manager.workspace_bounds['y']) == 2

    def test_custom_workspace_bounds(self):
        """Test custom workspace bounds are used."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        custom_bounds = {'x': [-5.0, 5.0], 'y': [-5.0, 5.0]}
        manager = SceneManager(mock_world, workspace_bounds=custom_bounds)

        assert manager.workspace_bounds['x'] == [-5.0, 5.0]
        assert manager.workspace_bounds['y'] == [-5.0, 5.0]

    def test_goal_reached_threshold(self):
        """Test goal reached with custom threshold."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)
        manager.goal_position = np.array([1.0, 1.0, 0.0])

        # Just outside default threshold (0.15)
        assert not manager.check_goal_reached(np.array([1.2, 1.0, 0.0]), threshold=0.15)

        # Inside threshold
        assert manager.check_goal_reached(np.array([1.1, 1.0, 0.0]), threshold=0.15)

    def test_obstacle_metadata_initialized_empty(self):
        """Test obstacle_metadata is initialized as empty list."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)
        assert manager.get_obstacle_metadata() == []

    def test_clear_obstacles_clears_metadata(self):
        """Test clear_obstacles also clears obstacle_metadata."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)
        # Manually add metadata
        manager.obstacle_metadata.append((np.array([1.0, 1.0]), 0.2))
        assert len(manager.obstacle_metadata) == 1

        manager.clear_obstacles()
        assert len(manager.obstacle_metadata) == 0


# ============================================================================
# TEST SUITE: Episode Termination
# ============================================================================

class TestEpisodeTermination:
    """Tests for episode termination conditions."""

    def test_goal_reached_terminates(self):
        """Test episode terminates when goal is reached."""
        # This would be tested by the RL env's _check_termination method
        # For now, test the scene manager's goal check
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)
        manager.goal_position = np.array([0.0, 0.0, 0.0])

        # At goal
        assert manager.check_goal_reached(np.array([0.0, 0.0, 0.0]))


# ============================================================================
# TEST SUITE: Action Scaling
# ============================================================================

class TestActionScaling:
    """Tests for action scaling in RL environment."""

    def test_action_mapper_output_range(self):
        """Test ActionMapper outputs in expected range."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()

        for key in ['w', 's', 'a', 'd', 'space']:
            action = mapper.map_key(key)
            assert -1.0 <= action[0] <= 1.0
            assert -1.0 <= action[1] <= 1.0

    def test_velocity_constants(self):
        """Test velocity constants are reasonable."""
        # These should match JetbotKeyboardController constants
        MAX_LINEAR_VELOCITY = 0.3   # m/s
        MAX_ANGULAR_VELOCITY = 1.0  # rad/s

        # Verify they're positive and reasonable
        assert MAX_LINEAR_VELOCITY > 0
        assert MAX_LINEAR_VELOCITY < 1.0  # Jetbot shouldn't go faster than 1 m/s
        assert MAX_ANGULAR_VELOCITY > 0
        assert MAX_ANGULAR_VELOCITY < 3.0  # Reasonable turn rate


# ============================================================================
# TEST SUITE: Integration
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_observation_reward_consistency(self):
        """Test observation and reward are computed consistently."""
        from jetbot_keyboard_control import ObservationBuilder, RewardComputer

        builder = ObservationBuilder()
        computer = RewardComputer(mode='dense')

        # Build two observations
        obs1 = builder.build(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.3,
            angular_velocity=0.0,
            goal_position=np.array([2.0, 0.0, 0.0]),
            goal_reached=False
        )

        obs2 = builder.build(
            robot_position=np.array([1.0, 0.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.3,
            angular_velocity=0.0,
            goal_position=np.array([2.0, 0.0, 0.0]),
            goal_reached=False
        )

        # Compute reward for transition
        reward = computer.compute(
            obs=obs1,
            action=np.array([1.0, 0.0]),
            next_obs=obs2,
            info={'goal_reached': False}
        )

        # Should get positive reward (got closer)
        assert reward > 0, "Should reward progress toward goal"

    def test_demo_record_load_cycle(self, tmp_path):
        """Test complete record and load cycle."""
        from jetbot_keyboard_control import DemoRecorder, DemoPlayer

        # Record
        recorder = DemoRecorder(obs_dim=10, action_dim=2)
        recorder.start_recording()

        for i in range(10):
            obs = np.random.randn(10).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            recorder.record_step(obs, action, float(i), False)

        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        filepath = tmp_path / "test_demo.npz"
        recorder.save(str(filepath))

        # Load
        player = DemoPlayer(str(filepath))

        assert player.num_episodes == 1
        assert player.total_frames == 10

        obs, actions = player.get_episode(0)
        assert len(obs) == 10
        assert len(actions) == 10


# ============================================================================
# TEST SUITE: JetbotNavigationEnv (with mocked Isaac Sim initialization)
# ============================================================================

class TestJetbotNavigationEnvInit:
    """Tests for JetbotNavigationEnv initialization with Isaac Sim mocked."""

    @pytest.fixture
    def env(self):
        """Create a JetbotNavigationEnv with _init_isaac_sim mocked out."""
        from jetbot_rl_env import JetbotNavigationEnv

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            # Call __init__ manually to set up spaces without Isaac Sim
            env.reward_mode = 'dense'
            env.max_episode_steps = 500
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            env.render_mode = 'human'
            env.headless = True
            env.goal_threshold = 0.15
            env.num_obstacles = 5
            env.min_goal_dist = 0.5

            from jetbot_keyboard_control import LidarSensor, ObservationBuilder, RewardComputer
            env.lidar_sensor = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

            obs_dim = 10 + 24
            from gymnasium import spaces
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            env.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

            env.obs_builder = ObservationBuilder(lidar_sensor=env.lidar_sensor)
            env.reward_computer = RewardComputer(mode='dense')
            env._step_count = 0
            env._prev_obs = None
            env._current_linear_vel = 0.0
            env._current_angular_vel = 0.0

            yield env

    def test_observation_space(self, env):
        assert env.observation_space.shape == (34,)

    def test_action_space(self, env):
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])

    def test_reward_mode_stored(self, env):
        assert env.reward_mode == 'dense'

    def test_max_episode_steps(self, env):
        assert env.max_episode_steps == 500


class TestCheckTermination:
    """Tests for _check_termination method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            yield env

    def test_goal_reached_terminates(self, env):
        info = {'goal_reached': True, 'collision': False}
        assert env._check_termination(info, np.array([0.0, 0.0, 0.0])) is True

    def test_collision_terminates(self, env):
        info = {'goal_reached': False, 'collision': True}
        assert env._check_termination(info, np.array([0.0, 0.0, 0.0])) is True

    def test_out_of_bounds_terminates(self, env):
        info = {'goal_reached': False, 'collision': False}
        # Way outside bounds
        assert env._check_termination(info, np.array([10.0, 0.0, 0.0])) is True

    def test_no_termination(self, env):
        info = {'goal_reached': False, 'collision': False}
        assert env._check_termination(info, np.array([0.0, 0.0, 0.0])) is False


class TestCheckTruncation:
    """Tests for _check_truncation method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.max_episode_steps = 500
            yield env

    def test_at_max_steps(self, env):
        env._step_count = 500
        assert env._check_truncation() is True

    def test_above_max_steps(self, env):
        env._step_count = 501
        assert env._check_truncation() is True

    def test_below_max_steps(self, env):
        env._step_count = 499
        assert env._check_truncation() is False

    def test_at_zero(self, env):
        env._step_count = 0
        assert env._check_truncation() is False


class TestIsOutOfBounds:
    """Tests for _is_out_of_bounds method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            yield env

    def test_within_bounds(self, env):
        assert not env._is_out_of_bounds(np.array([0.0, 0.0, 0.0]))

    def test_outside_x(self, env):
        assert env._is_out_of_bounds(np.array([5.0, 0.0, 0.0]))

    def test_outside_y(self, env):
        assert env._is_out_of_bounds(np.array([0.0, -5.0, 0.0]))

    def test_within_margin(self, env):
        """Position just beyond bounds but within 0.5 margin → not out of bounds."""
        assert not env._is_out_of_bounds(np.array([2.3, 0.0, 0.0]))

    def test_beyond_margin(self, env):
        """Position beyond bounds + 0.5 margin → out of bounds."""
        assert env._is_out_of_bounds(np.array([2.6, 0.0, 0.0]))

    def test_negative_outside(self, env):
        assert env._is_out_of_bounds(np.array([-3.0, 0.0, 0.0]))


class TestBuildObservation:
    """Tests for _build_observation method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv
        from jetbot_keyboard_control import LidarSensor, ObservationBuilder, SceneManager

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            env.goal_threshold = 0.15
            env.lidar_sensor = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
            env.obs_builder = ObservationBuilder(lidar_sensor=env.lidar_sensor)
            env._current_linear_vel = 0.0
            env._current_angular_vel = 0.0
            env.add_prev_action = False
            env._prev_action = np.zeros(2, dtype=np.float32)
            env.use_camera = False
            env._camera_features = None

            # Mock scene_manager and _get_robot_pose
            env.scene_manager = Mock()
            env.scene_manager.get_goal_position.return_value = np.array([1.0, 1.0, 0.0])
            env.scene_manager.check_goal_reached.return_value = False
            env.scene_manager.get_obstacle_metadata.return_value = []

            env.jetbot = Mock()

            yield env

    def test_output_shape(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            obs = env._build_observation()
            assert obs.shape == (34,)
            assert obs.dtype == np.float32

    def test_lidar_included(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            obs = env._build_observation()
            lidar_portion = obs[10:]
            assert len(lidar_portion) == 24
            # All lidar readings should be in [0, 1] (no obstacles)
            assert np.all(lidar_portion >= 0.0)
            assert np.all(lidar_portion <= 1.0)


class TestStep:
    """Tests for step method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv
        from jetbot_keyboard_control import (
            LidarSensor, ObservationBuilder, RewardComputer, SceneManager
        )

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.reward_mode = 'dense'
            env.max_episode_steps = 500
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            env.headless = True
            env.goal_threshold = 0.15
            env.cost_type = 'proximity'
            env.safe_mode = False
            env.lidar_sensor = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
            env.obs_builder = ObservationBuilder(lidar_sensor=env.lidar_sensor)
            env.reward_computer = RewardComputer(mode='dense')
            env._step_count = 0
            env._prev_obs = np.zeros(34, dtype=np.float32)
            env._prev_obs[8] = 2.0  # distance to goal
            env._current_linear_vel = 0.0
            env._current_angular_vel = 0.0
            env.add_prev_action = False
            env._prev_action = np.zeros(2, dtype=np.float32)
            env.use_camera = False
            env._camera_features = None
            env._wstep_ms_acc = 0.0
            env._wstep_n = 0

            env.MAX_LINEAR_VELOCITY = 0.3
            env.MAX_ANGULAR_VELOCITY = 1.0
            env.COLLISION_THRESHOLD = 0.08

            # Mock Isaac Sim objects
            env.world = Mock()
            env.jetbot = Mock()
            env.controller = Mock()
            env.controller.forward.return_value = Mock()
            env.scene_manager = Mock()
            env.scene_manager.get_goal_position.return_value = np.array([1.0, 1.0, 0.0])
            env.scene_manager.check_goal_reached.return_value = False
            env.scene_manager.get_obstacle_metadata.return_value = []

            yield env

    def test_step_count_increments(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            assert env._step_count == 0
            env.step(np.array([0.0, 0.0]))
            assert env._step_count == 1

    def test_returns_five_tuple(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            result = env.step(np.array([0.0, 0.0]))
            assert len(result) == 5
            obs, reward, terminated, truncated, info = result
            assert obs.shape == (34,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_action_clipping(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            # Pass actions outside [-1, 1]
            env.step(np.array([5.0, -5.0]))
            # Velocities should be clamped
            assert env._current_linear_vel == pytest.approx(0.3)   # 1.0 * 0.3
            assert env._current_angular_vel == pytest.approx(-1.0)  # -1.0 * 1.0


class TestReset:
    """Tests for reset method."""

    @pytest.fixture
    def env(self):
        from jetbot_rl_env import JetbotNavigationEnv
        from jetbot_keyboard_control import (
            LidarSensor, ObservationBuilder, RewardComputer
        )
        import gymnasium

        with patch.object(JetbotNavigationEnv, '_init_isaac_sim'):
            env = JetbotNavigationEnv.__new__(JetbotNavigationEnv)
            env.reward_mode = 'dense'
            env.max_episode_steps = 500
            env.workspace_bounds = {'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
            env.headless = True
            env.goal_threshold = 0.15
            env.lidar_sensor = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
            env.obs_builder = ObservationBuilder(lidar_sensor=env.lidar_sensor)
            env.reward_computer = RewardComputer(mode='dense')
            env._step_count = 100
            env._prev_obs = None
            env._current_linear_vel = 0.5
            env._current_angular_vel = 0.5
            env.add_prev_action = False
            env._prev_action = np.zeros(2, dtype=np.float32)
            env.use_camera = False
            env._camera_features = None

            env.START_POSITION = np.array([0.0, 0.0, 0.05])

            from gymnasium import spaces
            obs_dim = 34
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            env.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

            # Mock Isaac Sim objects
            env.world = Mock()
            env.jetbot = Mock()
            env.scene_manager = Mock()
            env.scene_manager.get_goal_position.return_value = np.array([1.0, 1.0, 0.0])
            env.scene_manager.check_goal_reached.return_value = False
            env.scene_manager.get_obstacle_metadata.return_value = []
            env.inflation_radius = 0.08

            # Need np_random for reset()
            env.np_random = np.random.default_rng(42)

            yield env

    def test_step_count_reset(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            with patch('jetbot_rl_env.astar_search', return_value=[(0, 0)]):
                with patch('jetbot_rl_env.OccupancyGrid.from_scene'):
                    # Patch super().reset to not call gymnasium.Env.reset
                    with patch('gymnasium.Env.reset', return_value=None):
                        env.reset()
            assert env._step_count == 0

    def test_returns_obs_and_info(self, env):
        with patch.object(type(env), '_get_robot_pose',
                         return_value=(np.array([0.0, 0.0, 0.05]), 0.0)):
            with patch('jetbot_rl_env.astar_search', return_value=[(0, 0)]):
                with patch('jetbot_rl_env.OccupancyGrid.from_scene'):
                    with patch('gymnasium.Env.reset', return_value=None):
                        result = env.reset()
            assert len(result) == 2
            obs, info = result
            assert obs.shape == (34,)
            assert isinstance(info, dict)
            assert 'is_success' in info
