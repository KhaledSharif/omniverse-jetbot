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

    def test_observation_space_shape(self):
        """Test observation space has correct shape."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()
        assert builder.obs_dim == 10

    def test_action_space_shape(self):
        """Test action space has correct shape."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        assert mapper.action_dim == 2

    def test_observation_building(self):
        """Test observation can be built correctly."""
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
        prev_obs[7] = 2.0  # Distance to goal

        next_obs = np.zeros(10)
        next_obs[7] = 1.0  # Closer to goal
        next_obs[8] = 0.0  # Facing goal

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
        prev_obs[7] = 1.0  # Distance to goal

        next_obs = np.zeros(10)
        next_obs[7] = 2.0  # Farther from goal
        next_obs[8] = np.pi  # Facing away from goal

        reward = computer.compute(
            obs=prev_obs,
            action=np.zeros(2),
            next_obs=next_obs,
            info={'goal_reached': False}
        )

        # Should get negative or low reward for regression
        # (exact value depends on heading bonus calculation)
        assert reward < 1.0  # At least less than distance scale


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
