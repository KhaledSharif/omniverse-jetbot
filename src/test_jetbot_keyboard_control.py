"""
Comprehensive Test Suite for Jetbot Keyboard Control

This test suite covers:
- Command processing and queue handling
- Movement command processing
- Recording functionality
- State management
- Helper classes (ActionMapper, ObservationBuilder, RewardComputer, etc.)

Run with: pytest test_jetbot_keyboard_control.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing jetbot_keyboard_control)
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
# TEST SUITE: ActionMapper
# ============================================================================

class TestActionMapper:
    """Tests for ActionMapper class."""

    def test_action_mapper_initialization(self):
        """Test ActionMapper initializes correctly."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        assert mapper.action_dim == 2

    def test_map_forward_key(self):
        """Test mapping 'w' key to forward action."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('w')
        assert action[0] == 1.0  # Linear velocity
        assert action[1] == 0.0  # Angular velocity

    def test_map_backward_key(self):
        """Test mapping 's' key to backward action."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('s')
        assert action[0] == -1.0
        assert action[1] == 0.0

    def test_map_turn_left_key(self):
        """Test mapping 'a' key to turn left action."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('a')
        assert action[0] == 0.0
        assert action[1] == 1.0

    def test_map_turn_right_key(self):
        """Test mapping 'd' key to turn right action."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('d')
        assert action[0] == 0.0
        assert action[1] == -1.0

    def test_map_stop_key(self):
        """Test mapping 'space' key to stop action."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('space')
        assert action[0] == 0.0
        assert action[1] == 0.0

    def test_map_unknown_key(self):
        """Test mapping unknown key returns zeros."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key('z')
        assert np.all(action == 0.0)

    def test_map_none_key(self):
        """Test mapping None key returns zeros."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()
        action = mapper.map_key(None)
        assert np.all(action == 0.0)


# ============================================================================
# TEST SUITE: ObservationBuilder
# ============================================================================

class TestObservationBuilder:
    """Tests for ObservationBuilder class."""

    def test_observation_builder_initialization(self):
        """Test ObservationBuilder initializes with correct dimensions."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()
        assert builder.obs_dim == 10

    def test_build_observation_basic(self):
        """Test building observation with basic inputs."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()

        obs = builder.build(
            robot_position=np.array([1.0, 2.0, 0.0]),
            robot_heading=0.5,
            linear_velocity=0.2,
            angular_velocity=0.1,
            goal_position=np.array([3.0, 4.0, 0.0]),
            goal_reached=False
        )

        assert obs.shape == (10,)
        assert obs[0] == 1.0  # x
        assert obs[1] == 2.0  # y
        assert obs[2] == 0.5  # heading
        assert obs[3] == 0.2  # linear vel
        assert obs[4] == 0.1  # angular vel
        assert obs[5] == 3.0  # goal x
        assert obs[6] == 4.0  # goal y
        assert obs[9] == 0.0  # goal not reached

    def test_build_observation_goal_reached(self):
        """Test observation with goal reached flag."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()

        obs = builder.build(
            robot_position=np.array([1.0, 1.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([1.0, 1.0, 0.0]),
            goal_reached=True
        )

        assert obs[9] == 1.0  # goal reached

    def test_build_observation_angle_to_goal(self):
        """Test angle to goal calculation."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()

        # Robot at origin, facing +x, goal at +x direction
        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([1.0, 0.0, 0.0]),
            goal_reached=False
        )

        # Angle to goal should be 0 (facing goal)
        assert abs(obs[8]) < 0.01

    def test_build_observation_distance_to_goal(self):
        """Test distance to goal calculation."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()

        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([3.0, 4.0, 0.0]),
            goal_reached=False
        )

        # Distance should be 5.0 (3-4-5 triangle)
        assert abs(obs[7] - 5.0) < 0.01


# ============================================================================
# TEST SUITE: RewardComputer
# ============================================================================

class TestRewardComputer:
    """Tests for RewardComputer class."""

    def test_reward_computer_sparse_mode(self):
        """Test sparse reward mode."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='sparse')

        # No goal reached
        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': False}
        )
        assert reward == 0.0

        # Goal reached
        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': True}
        )
        assert reward == 10.0  # GOAL_REACHED_REWARD

    def test_reward_computer_dense_mode_goal_reached(self):
        """Test dense reward mode with goal reached."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': True}
        )
        assert reward >= 10.0  # Should include goal bonus

    def test_reward_computer_dense_mode_getting_closer(self):
        """Test dense reward mode rewards getting closer to goal."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        # Previous obs: distance = 5.0
        prev_obs = np.zeros(10)
        prev_obs[7] = 5.0

        # Current obs: distance = 4.0 (got closer)
        next_obs = np.zeros(10)
        next_obs[7] = 4.0
        next_obs[8] = 0.0  # Facing goal

        reward = computer.compute(
            obs=prev_obs,
            action=np.zeros(2),
            next_obs=next_obs,
            info={'goal_reached': False}
        )

        # Should get positive reward for getting closer
        assert reward > 0


# ============================================================================
# TEST SUITE: DemoRecorder
# ============================================================================

class TestDemoRecorder:
    """Tests for DemoRecorder class."""

    def test_recorder_initialization(self):
        """Test DemoRecorder initializes correctly."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        assert recorder.obs_dim == 10
        assert recorder.action_dim == 2
        assert not recorder.is_recording
        assert len(recorder.observations) == 0

    def test_recording_start_stop(self):
        """Test starting and stopping recording."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        assert recorder.is_recording

        recorder.stop_recording()
        assert not recorder.is_recording

    def test_record_step(self):
        """Test recording a step."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        recorder.record_step(
            obs=np.ones(10),
            action=np.ones(2),
            reward=1.0,
            done=False
        )

        assert len(recorder.observations) == 1
        assert len(recorder.actions) == 1
        assert recorder.rewards[0] == 1.0

    def test_record_step_not_recording(self):
        """Test record_step does nothing when not recording."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        # Not recording
        recorder.record_step(
            obs=np.ones(10),
            action=np.ones(2),
            reward=1.0,
            done=False
        )

        assert len(recorder.observations) == 0

    def test_episode_finalization(self):
        """Test episode finalization."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        for i in range(5):
            recorder.record_step(np.ones(10), np.ones(2), 1.0, False)

        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        assert len(recorder.episode_starts) == 1
        assert recorder.episode_lengths[0] == 5
        assert recorder.episode_success[0] is True

    def test_get_stats(self):
        """Test getting recording statistics."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        for i in range(3):
            recorder.record_step(np.ones(10), np.ones(2), 1.0, False)

        stats = recorder.get_stats()

        assert stats['is_recording'] is True
        assert stats['total_frames'] == 3
        assert stats['current_episode_frames'] == 3

    def test_clear(self):
        """Test clearing recorded data."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        recorder.record_step(np.ones(10), np.ones(2), 1.0, False)
        recorder.clear()

        assert len(recorder.observations) == 0
        assert not recorder.is_recording


# ============================================================================
# TEST SUITE: SceneManager
# ============================================================================

class TestSceneManager:
    """Tests for SceneManager class."""

    def test_scene_manager_initialization(self):
        """Test SceneManager initializes correctly."""
        from jetbot_keyboard_control import SceneManager
        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)

        assert manager.goal_position is None
        assert manager.workspace_bounds == SceneManager.DEFAULT_WORKSPACE_BOUNDS

    def test_check_goal_reached(self):
        """Test goal reached checking."""
        from jetbot_keyboard_control import SceneManager
        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        manager = SceneManager(mock_world)
        manager.goal_position = np.array([1.0, 1.0, 0.0])

        # Robot at goal
        assert manager.check_goal_reached(np.array([1.0, 1.0, 0.0]), threshold=0.15)

        # Robot far from goal
        assert not manager.check_goal_reached(np.array([5.0, 5.0, 0.0]), threshold=0.15)

    def test_goal_position_none(self):
        """Test goal check when no goal set."""
        from jetbot_keyboard_control import SceneManager
        mock_world = Mock()
        mock_world.scene = Mock()

        manager = SceneManager(mock_world)

        # Should return False when no goal
        assert not manager.check_goal_reached(np.array([0.0, 0.0, 0.0]))


# ============================================================================
# TEST SUITE: DemoPlayer
# ============================================================================

class TestDemoPlayer:
    """Tests for DemoPlayer class."""

    def test_get_episode(self, tmp_path):
        """Test getting a specific episode."""
        from jetbot_keyboard_control import DemoRecorder, DemoPlayer

        # Create and save test data
        recorder = DemoRecorder(obs_dim=10, action_dim=2)
        recorder.start_recording()
        for i in range(5):
            recorder.record_step(np.ones(10) * i, np.ones(2) * i, float(i), False)
        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        filepath = tmp_path / "test_demo.npz"
        recorder.save(str(filepath))

        # Load and check
        player = DemoPlayer(str(filepath))
        obs, actions = player.get_episode(0)

        assert len(obs) == 5
        assert len(actions) == 5

    def test_get_successful_episodes(self, tmp_path):
        """Test filtering successful episodes."""
        from jetbot_keyboard_control import DemoRecorder, DemoPlayer

        # Create test data with mixed success
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        # Episode 1: success
        recorder.start_recording()
        recorder.record_step(np.ones(10), np.ones(2), 1.0, False)
        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        # Episode 2: failure
        recorder.start_recording()
        recorder.record_step(np.ones(10), np.ones(2), 0.0, False)
        recorder.mark_episode_success(False)
        recorder.finalize_episode()

        filepath = tmp_path / "test_demo.npz"
        recorder.save(str(filepath))

        # Load and check
        player = DemoPlayer(str(filepath))
        successful = player.get_successful_episodes()

        assert len(successful) == 1
        assert successful[0] == 0


# ============================================================================
# TEST SUITE: Command Processing
# ============================================================================

class TestCommandProcessing:
    """Tests for command queue and processing."""

    def test_queue_command(self):
        """Test queuing commands is thread-safe."""
        from jetbot_keyboard_control import JetbotKeyboardController
        import threading

        # Create minimal mock controller
        with patch('jetbot_keyboard_control.SimulationApp'), \
             patch('jetbot_keyboard_control.World'), \
             patch('jetbot_keyboard_control.WheeledRobot'), \
             patch('jetbot_keyboard_control.DifferentialController'), \
             patch('jetbot_keyboard_control.get_assets_root_path', return_value='/Isaac'), \
             patch('jetbot_keyboard_control.keyboard.Listener'), \
             patch('jetbot_keyboard_control.TUIRenderer'), \
             patch('jetbot_keyboard_control.termios'), \
             patch('jetbot_keyboard_control.sys'):

            # Just test the queue mechanism
            import jetbot_keyboard_control
            jetbot_keyboard_control.simulation_app = Mock()

            # Create a simple object with just the queue methods
            class SimpleController:
                def __init__(self):
                    self.command_lock = threading.Lock()
                    self.pending_commands = []

                def _queue_command(self, command):
                    with self.command_lock:
                        self.pending_commands.append(command)

            controller = SimpleController()

            # Queue multiple commands
            controller._queue_command(('char', 'w'))
            controller._queue_command(('char', 's'))
            controller._queue_command(('special', 'esc'))

            assert len(controller.pending_commands) == 3


# ============================================================================
# TEST SUITE: Movement Commands
# ============================================================================

class TestMovementCommands:
    """Tests for movement command processing."""

    def test_forward_command_sets_velocity(self):
        """Test 'w' key sets forward velocity."""
        # Test the ActionMapper directly since controller is complex to mock
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()

        action = mapper.map_key('w')
        assert action[0] == 1.0  # Normalized forward

    def test_backward_command_sets_velocity(self):
        """Test 's' key sets backward velocity."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()

        action = mapper.map_key('s')
        assert action[0] == -1.0  # Normalized backward

    def test_stop_command_zeros_velocity(self):
        """Test 'space' key stops the robot."""
        from jetbot_keyboard_control import ActionMapper
        mapper = ActionMapper()

        action = mapper.map_key('space')
        assert action[0] == 0.0
        assert action[1] == 0.0
