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
        """Test ObservationBuilder initializes with correct dimensions (no lidar)."""
        from jetbot_keyboard_control import ObservationBuilder
        builder = ObservationBuilder()
        assert builder.obs_dim == 10

    def test_observation_builder_with_lidar(self):
        """Test ObservationBuilder initializes with 34D when lidar provided."""
        from jetbot_keyboard_control import ObservationBuilder, LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
        builder = ObservationBuilder(lidar_sensor=lidar)
        assert builder.obs_dim == 34

    def test_build_observation_34d_with_lidar(self):
        """Test building 34D observation with lidar."""
        from jetbot_keyboard_control import ObservationBuilder, LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
        builder = ObservationBuilder(lidar_sensor=lidar)

        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.0]),
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
        # LiDAR readings should be normalized [0, 1]
        assert np.all(obs[10:] >= 0.0)
        assert np.all(obs[10:] <= 1.0)

    def test_build_observation_10d_without_lidar_params(self):
        """Test that 10D obs is returned when lidar is set but no metadata passed."""
        from jetbot_keyboard_control import ObservationBuilder, LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
        builder = ObservationBuilder(lidar_sensor=lidar)

        obs = builder.build(
            robot_position=np.array([0.0, 0.0, 0.0]),
            robot_heading=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            goal_position=np.array([1.0, 1.0, 0.0]),
            goal_reached=False,
        )
        # Without obstacle_metadata/workspace_bounds, falls back to base obs
        assert obs.shape == (10,)

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
        assert np.isclose(obs[3], 0.2)  # linear vel
        assert np.isclose(obs[4], 0.1)  # angular vel
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

    def test_reward_computer_collision_penalty(self):
        """Test collision returns -10 penalty."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': False, 'collision': True}
        )
        assert reward == RewardComputer.COLLISION_PENALTY

    def test_reward_computer_sparse_collision_penalty(self):
        """Test sparse mode also returns collision penalty."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='sparse')

        reward = computer.compute(
            obs=np.zeros(10),
            action=np.zeros(2),
            next_obs=np.zeros(10),
            info={'goal_reached': False, 'collision': True}
        )
        assert reward == RewardComputer.COLLISION_PENALTY

    def test_reward_computer_proximity_penalty(self):
        """Test proximity penalty when close to obstacle."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        prev_obs = np.zeros(10)
        prev_obs[7] = 2.0
        next_obs = np.zeros(10)
        next_obs[7] = 2.0
        next_obs[8] = 0.0

        # Close to obstacle (min_lidar < threshold)
        reward_close = computer.compute(
            obs=prev_obs, action=np.zeros(2), next_obs=next_obs,
            info={'goal_reached': False, 'min_lidar_distance': 0.1}
        )

        # Far from obstacles
        reward_far = computer.compute(
            obs=prev_obs, action=np.zeros(2), next_obs=next_obs,
            info={'goal_reached': False, 'min_lidar_distance': 1.0}
        )

        assert reward_close < reward_far, "Proximity penalty should reduce reward"

    def test_reward_computer_no_proximity_penalty_when_far(self):
        """Test no proximity penalty when far from obstacles."""
        from jetbot_keyboard_control import RewardComputer
        computer = RewardComputer(mode='dense')

        prev_obs = np.zeros(10)
        prev_obs[7] = 2.0
        next_obs = np.zeros(10)
        next_obs[7] = 2.0
        next_obs[8] = 0.0

        reward_no_lidar = computer.compute(
            obs=prev_obs, action=np.zeros(2), next_obs=next_obs,
            info={'goal_reached': False}
        )
        reward_far_lidar = computer.compute(
            obs=prev_obs, action=np.zeros(2), next_obs=next_obs,
            info={'goal_reached': False, 'min_lidar_distance': 2.0}
        )

        # Only difference should be time penalty (present in both)
        assert abs(reward_no_lidar - reward_far_lidar) < 0.001


# ============================================================================
# TEST SUITE: LidarSensor
# ============================================================================

class TestLidarSensor:
    """Tests for LidarSensor class."""

    def test_initialization(self):
        """Test LidarSensor initializes correctly."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)
        assert lidar.num_rays == 24
        assert lidar.fov_deg == 180.0
        assert lidar.max_range == 3.0

    def test_output_shape_and_dtype(self):
        """Test scan returns correct shape and dtype."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[],
            workspace_bounds={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
        )

        assert result.shape == (24,)
        assert result.dtype == np.float32

    def test_no_obstacles_returns_wall_or_max_range(self):
        """Test that with no obstacles, rays hit walls or return max_range."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[],
            workspace_bounds={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
        )

        # All distances should be positive and <= max_range
        assert np.all(result > 0)
        assert np.all(result <= 3.0)

    def test_obstacle_directly_ahead(self):
        """Test obstacle directly ahead is detected by center ray."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        # Place obstacle at (1.0, 0.0) with radius 0.1
        # Robot at origin facing +x
        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[(np.array([1.0, 0.0]), 0.1)],
            workspace_bounds={'x': [-5.0, 5.0], 'y': [-5.0, 5.0]}
        )

        # Center ray (index 12 for 24 rays, or close to it) should detect obstacle
        # The center ray should hit at distance ~0.9 (1.0 - radius)
        center_idx = lidar.num_rays // 2
        assert result[center_idx] < 1.5, "Center ray should detect nearby obstacle"

    def test_obstacle_to_the_left(self):
        """Test obstacle to the left is detected by left-side rays."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        # Robot at origin facing +x, obstacle at (0, 1) (left side)
        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[(np.array([0.0, 1.0]), 0.1)],
            workspace_bounds={'x': [-5.0, 5.0], 'y': [-5.0, 5.0]}
        )

        # Left rays (high indices, positive angle offset) should detect
        # Right rays (low indices) should not
        left_min = result[18:].min()
        right_min = result[:6].min()
        assert left_min < right_min, "Left rays should detect obstacle on the left"

    def test_obstacle_behind_not_detected(self):
        """Test obstacle behind robot (outside 180 FOV) is not detected."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        # Robot at origin facing +x, obstacle at (-1.0, 0.0) - behind
        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[(np.array([-1.0, 0.0]), 0.1)],
            workspace_bounds={'x': [-5.0, 5.0], 'y': [-5.0, 5.0]}
        )

        # No ray should detect the obstacle behind (all hit walls far away)
        # With large workspace, all rays should be > 2.0
        assert np.all(result > 1.5), "Obstacle behind should not be detected"

    def test_workspace_boundary_detection(self):
        """Test rays hitting workspace walls."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=10.0)

        # Robot at origin, walls at +/- 2.0
        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[],
            workspace_bounds={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
        )

        # Center ray (facing +x) should hit wall at x=2.0 -> distance=2.0
        center_idx = lidar.num_rays // 2
        assert abs(result[center_idx] - 2.0) < 0.01, "Center ray should hit wall at 2.0m"

    def test_multiple_obstacles_nearest_wins(self):
        """Test nearest obstacle per ray is reported."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=5.0)

        # Two obstacles ahead, near and far
        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[
                (np.array([1.0, 0.0]), 0.1),  # Near
                (np.array([3.0, 0.0]), 0.1),  # Far
            ],
            workspace_bounds={'x': [-5.0, 5.0], 'y': [-5.0, 5.0]}
        )

        center_idx = lidar.num_rays // 2
        # Should detect near obstacle, not far one
        assert result[center_idx] < 1.5, "Should detect nearest obstacle"

    def test_ray_at_fov_boundary(self):
        """Test rays at FOV boundary edges work correctly."""
        from jetbot_keyboard_control import LidarSensor
        lidar = LidarSensor(num_rays=24, fov_deg=180.0, max_range=3.0)

        result = lidar.scan(
            robot_position_2d=np.array([0.0, 0.0]),
            robot_heading=0.0,
            obstacle_metadata=[],
            workspace_bounds={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]}
        )

        # First and last rays should be at +/- 90 degrees
        # They should hit the y-walls at +/- 2.0
        assert result[0] > 0, "First ray should detect something"
        assert result[-1] > 0, "Last ray should detect something"


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

    def test_abandon_episode_discards_steps(self):
        """abandon_episode should remove in-progress steps without finalizing."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        recorder.start_recording()
        # Record a completed episode first
        for _ in range(3):
            recorder.record_step(np.ones(10), np.ones(2), 1.0, False)
        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        # Now record a partial episode and abandon it
        for _ in range(5):
            recorder.record_step(np.ones(10) * 2, np.ones(2), 0.5, False)
        recorder.abandon_episode()

        # Buffers should be back to end of first episode (3 frames)
        assert len(recorder.observations) == 3
        assert len(recorder.actions) == 3
        assert len(recorder.rewards) == 3
        assert len(recorder.dones) == 3
        # Finalized episode count unchanged
        assert len(recorder.episode_starts) == 1
        # Episode state reset
        assert recorder.current_episode_return == 0.0
        assert recorder._pending_success is None

    def test_abandon_episode_with_no_steps(self):
        """abandon_episode on empty episode should be a no-op."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)
        recorder.start_recording()
        recorder.abandon_episode()
        assert len(recorder.observations) == 0
        assert len(recorder.episode_starts) == 0

    def test_abandon_then_record_continues_normally(self):
        """After abandoning, new steps should be recorded as a fresh episode."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=10, action_dim=2)
        recorder.start_recording()

        # Record and abandon
        for _ in range(4):
            recorder.record_step(np.ones(10), np.ones(2), 1.0, False)
        recorder.abandon_episode()

        # Record a new valid episode
        for _ in range(2):
            recorder.record_step(np.ones(10) * 3, np.ones(2), 2.0, False)
        recorder.mark_episode_success(True)
        recorder.finalize_episode()

        assert len(recorder.episode_starts) == 1
        assert recorder.episode_lengths[0] == 2
        assert recorder.episode_success[0] is True

    def test_done_flag_true_on_collision(self):
        """Done flag should be True when recording a collision step."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=34, action_dim=2)
        recorder.start_recording()

        # Simulate normal steps (done=False)
        for _ in range(3):
            recorder.record_step(np.ones(34), np.ones(2), 0.1, False)

        # Simulate collision step (done=True)
        recorder.record_step(np.ones(34), np.ones(2), -10.0, True)
        recorder.mark_episode_success(False)
        recorder.finalize_episode()

        assert recorder.dones[0] is False
        assert recorder.dones[1] is False
        assert recorder.dones[2] is False
        assert recorder.dones[3] is True  # collision terminal

    def test_reward_includes_collision_penalty(self):
        """RewardComputer should return collision penalty when info has collision=True."""
        from jetbot_keyboard_control import RewardComputer
        rc = RewardComputer(mode='dense')

        obs = np.zeros(34)
        next_obs = np.zeros(34)
        info_collision = {'goal_reached': False, 'collision': True, 'min_lidar_distance': 0.05}
        info_no_collision = {'goal_reached': False, 'collision': False, 'min_lidar_distance': 1.0}

        reward_collision = rc.compute(obs, np.zeros(2), next_obs, info_collision)
        reward_normal = rc.compute(obs, np.zeros(2), next_obs, info_no_collision)

        assert reward_collision == -10.0, f"Collision should give -10.0, got {reward_collision}"
        assert reward_normal != -10.0, "Normal step should not give -10.0"

    def test_reward_includes_proximity_penalty(self):
        """RewardComputer should apply proximity penalty when min_lidar_distance is small."""
        from jetbot_keyboard_control import RewardComputer
        rc = RewardComputer(mode='dense')

        obs = np.zeros(34)
        obs[7] = 1.0  # distance to goal
        next_obs = np.zeros(34)
        next_obs[7] = 1.0

        # Close to obstacle (should have proximity penalty)
        info_close = {'goal_reached': False, 'collision': False, 'min_lidar_distance': 0.15}
        reward_close = rc.compute(obs, np.zeros(2), next_obs, info_close)

        # Far from obstacle (no proximity penalty)
        info_far = {'goal_reached': False, 'collision': False, 'min_lidar_distance': 1.0}
        reward_far = rc.compute(obs, np.zeros(2), next_obs, info_far)

        assert reward_close < reward_far, "Close to obstacle should have worse reward"

    def test_done_flag_false_on_timeout(self):
        """Done should NOT be True for timeout — timeout is truncation, not termination."""
        from jetbot_keyboard_control import DemoRecorder
        recorder = DemoRecorder(obs_dim=34, action_dim=2)
        recorder.start_recording()

        # Simulate a full episode of normal steps (timeout)
        for _ in range(5):
            recorder.record_step(np.ones(34), np.ones(2), -0.005, False)

        # Finalize as failure (timeout) — note all dones should be False
        recorder.mark_episode_success(False)
        recorder.finalize_episode()

        for i in range(5):
            assert recorder.dones[i] is False, f"Step {i} done should be False for timeout"


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


# ============================================================================
# TEST SUITE: AutoPilot
# ============================================================================

class TestAutoPilot:
    """Tests for AutoPilot class."""

    def test_facing_goal_drives_forward(self):
        """When facing goal straight ahead, should produce positive linear vel."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)

        obs = np.zeros(10)
        obs[7] = 1.0   # distance = 1m
        obs[8] = 0.0   # angle = 0 (facing goal)

        linear, angular = pilot.compute_action(obs)
        assert linear > 0.0, "Should drive forward when facing goal"
        assert abs(angular) < 0.1, "Should have near-zero angular vel when aligned"

    def test_goal_left_turns_left(self):
        """When goal is to the left, should produce positive angular vel."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)

        obs = np.zeros(10)
        obs[7] = 1.0
        obs[8] = np.pi / 4  # goal is 45 degrees to the left

        _, angular = pilot.compute_action(obs)
        assert angular > 0.0, "Should turn left (positive angular) when goal is left"

    def test_goal_right_turns_right(self):
        """When goal is to the right, should produce negative angular vel."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)

        obs = np.zeros(10)
        obs[7] = 1.0
        obs[8] = -np.pi / 4  # goal is 45 degrees to the right

        _, angular = pilot.compute_action(obs)
        assert angular < 0.0, "Should turn right (negative angular) when goal is right"

    def test_facing_away_low_linear_vel(self):
        """When facing away from goal, linear vel should be near zero."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)

        obs = np.zeros(10)
        obs[7] = 1.0
        obs[8] = np.pi  # facing completely away

        linear, _ = pilot.compute_action(obs)
        assert abs(linear) < 0.05, "Should have near-zero linear vel when facing away"

    def test_output_clipping(self):
        """Outputs should be clipped to velocity limits."""
        from jetbot_keyboard_control import AutoPilot
        max_lin = 0.3
        max_ang = 1.0
        pilot = AutoPilot(max_linear_vel=max_lin, max_angular_vel=max_ang,
                          noise_linear=0.0, noise_angular=0.0,
                          kp_angular=100.0)  # very high gain to force clipping

        obs = np.zeros(10)
        obs[7] = 1.0
        obs[8] = np.pi  # large angle

        linear, angular = pilot.compute_action(obs)
        assert abs(linear) <= max_lin + 1e-6
        assert abs(angular) <= max_ang + 1e-6

    def test_noise_variation(self):
        """With noise enabled, repeated calls should produce different outputs."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.03, noise_angular=0.15)

        obs = np.zeros(10)
        obs[7] = 1.0
        obs[8] = 0.0

        results = [pilot.compute_action(obs) for _ in range(20)]
        linear_vals = [r[0] for r in results]

        # With noise, not all values should be identical
        assert len(set(linear_vals)) > 1, "Noise should cause variation in outputs"

    def test_slowdown_near_goal(self):
        """Linear velocity should be reduced when close to goal."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)

        # Far from goal
        obs_far = np.zeros(10)
        obs_far[7] = 2.0
        obs_far[8] = 0.0
        linear_far, _ = pilot.compute_action(obs_far)

        # Close to goal
        obs_near = np.zeros(10)
        obs_near[7] = 0.1
        obs_near[8] = 0.0
        linear_near, _ = pilot.compute_action(obs_near)

        assert linear_near < linear_far, "Should slow down near goal"

    def test_compute_action_returns_tuple(self):
        """compute_action should return a tuple of two floats."""
        from jetbot_keyboard_control import AutoPilot
        pilot = AutoPilot()

        obs = np.zeros(10)
        obs[7] = 1.0

        result = pilot.compute_action(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ============================================================================
# TEST SUITE: AutoMode
# ============================================================================

class TestAutoMode:
    """Tests for automatic mode integration."""

    def _make_controller_attrs(self, automatic=True):
        """Create a minimal mock controller with automatic mode attributes."""
        from jetbot_keyboard_control import AutoPilot, DemoRecorder, ActionMapper, ObservationBuilder, RewardComputer, SceneManager

        class FakeController:
            MAX_LINEAR_VELOCITY = 0.3
            MAX_ANGULAR_VELOCITY = 1.0

            def __init__(self):
                self.automatic = automatic
                self.enable_recording = True
                self.recorder = DemoRecorder(obs_dim=10, action_dim=2)
                self.action_mapper = ActionMapper()
                self.obs_builder = ObservationBuilder()
                self.reward_computer = RewardComputer()
                self.auto_pilot = AutoPilot(
                    max_linear_vel=self.MAX_LINEAR_VELOCITY,
                    max_angular_vel=self.MAX_ANGULAR_VELOCITY
                ) if automatic else None
                self.auto_episode_count = 0
                self.auto_step_count = 0
                self.auto_max_episode_steps = 500
                self.num_episodes = 100
                self.continuous = False
                self.headless_tui = True
                self.current_linear_vel = 0.0
                self.current_angular_vel = 0.0
                self.should_exit = False

                # Set up scene manager with mock world
                mock_world = Mock()
                mock_world.scene = Mock()
                mock_world.scene.add = Mock(side_effect=lambda x: x)
                self.scene_manager = SceneManager(mock_world)
                self.scene_manager.goal_position = np.array([1.0, 1.0, 0.0])

        return FakeController()

    def test_automatic_forces_recording(self):
        """Automatic mode should force-enable recording."""
        # Test that AutoPilot is created when automatic=True
        ctrl = self._make_controller_attrs(automatic=True)
        assert ctrl.auto_pilot is not None
        assert ctrl.enable_recording is True
        assert ctrl.recorder is not None

    def test_wasd_ignored_in_auto_mode(self):
        """WASD movement commands should be ignored in automatic mode."""
        from jetbot_keyboard_control import JetbotKeyboardController
        import threading

        # Create a minimal object to test _process_commands logic
        class MinimalController:
            def __init__(self):
                self.automatic = True
                self.should_exit = False
                self.command_lock = threading.Lock()
                self.pending_commands = [('char', 'w'), ('char', 'a')]
                self.tui = Mock()
                self.recorder = None
                self.enable_recording = False
                self.scene_manager = Mock()
                self.camera_streamer = None

            def _process_movement_command(self, key):
                self.movement_called = True

            def _handle_recording_command(self, key):
                pass

            def _spawn_new_goal(self):
                pass

            def _reset_robot(self):
                pass

            def _toggle_camera_viewer(self):
                pass

        ctrl = MinimalController()
        ctrl.movement_called = False

        # Manually replicate the command processing logic
        with ctrl.command_lock:
            commands = ctrl.pending_commands.copy()
            ctrl.pending_commands.clear()

        for cmd_type, cmd_value in commands:
            if cmd_type == 'char':
                if cmd_value in ('w', 's', 'a', 'd', 'space') and not ctrl.automatic:
                    ctrl._process_movement_command(cmd_value)

        assert not ctrl.movement_called, "Movement commands should be ignored in automatic mode"

    def test_system_commands_still_work(self):
        """System commands (esc) should still work in automatic mode."""
        import threading

        class MinimalController:
            def __init__(self):
                self.automatic = True
                self.should_exit = False
                self.command_lock = threading.Lock()
                self.pending_commands = [('special', 'esc')]
                self.tui = Mock()

        ctrl = MinimalController()

        with ctrl.command_lock:
            commands = ctrl.pending_commands.copy()
            ctrl.pending_commands.clear()

        for cmd_type, cmd_value in commands:
            if cmd_type == 'special' and cmd_value == 'esc':
                ctrl.should_exit = True

        assert ctrl.should_exit, "Esc should still exit in automatic mode"

    def test_out_of_bounds_detection(self):
        """Out-of-bounds detection should work correctly."""
        from jetbot_keyboard_control import SceneManager

        mock_world = Mock()
        mock_world.scene = Mock()
        mock_world.scene.add = Mock(side_effect=lambda x: x)

        scene_manager = SceneManager(mock_world)
        bounds = scene_manager.workspace_bounds
        margin = 0.5

        # Within bounds
        pos_in = np.array([0.0, 0.0, 0.0])
        in_bounds = (
            pos_in[0] < bounds['x'][0] - margin or
            pos_in[0] > bounds['x'][1] + margin or
            pos_in[1] < bounds['y'][0] - margin or
            pos_in[1] > bounds['y'][1] + margin
        )
        assert not in_bounds

        # Out of bounds
        pos_out = np.array([5.0, 5.0, 0.0])
        out_bounds = (
            pos_out[0] < bounds['x'][0] - margin or
            pos_out[0] > bounds['x'][1] + margin or
            pos_out[1] < bounds['y'][0] - margin or
            pos_out[1] > bounds['y'][1] + margin
        )
        assert out_bounds

    def test_episode_counter_increment(self):
        """Episode counter should increment after episode end."""
        ctrl = self._make_controller_attrs(automatic=True)

        assert ctrl.auto_episode_count == 0

        # Simulate episode end
        ctrl.recorder.start_recording()
        # Record a few steps
        obs = np.zeros(10)
        for _ in range(5):
            ctrl.recorder.record_step(obs, np.zeros(2), 1.0, False)

        ctrl.recorder.mark_episode_success(True)
        ctrl.recorder.finalize_episode()
        ctrl.auto_episode_count += 1
        ctrl.auto_step_count = 0

        assert ctrl.auto_episode_count == 1

    def test_skip_unsolvable_does_not_increment_episode_count(self):
        """_skip_unsolvable_episode must not increment auto_episode_count."""
        from jetbot_keyboard_control import (
            DemoRecorder, AutoPilot, SceneManager,
            JetbotKeyboardController,
        )

        class FakeController:
            headless_tui = True
            auto_step_count = 0
            auto_episode_count = 0
            current_obs = np.zeros(10)
            MAX_CONSECUTIVE_SKIPS = 100

            def __init__(self):
                self._consecutive_skips = 0
                self.recorder = DemoRecorder(obs_dim=10, action_dim=2)
                self.recorder.start_recording()
                self.auto_pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)
                mock_world = Mock()
                mock_world.scene = Mock()
                mock_world.scene.add = Mock(side_effect=lambda x: x)
                self.scene_manager = SceneManager(mock_world)
                self.scene_manager.goal_position = np.array([1.0, 1.0, 0.0])

            def _reset_recording_episode(self):
                pass  # no Isaac Sim

            def _build_current_observation(self):
                return np.zeros(10)

            def _draw_debug_overlays(self):
                pass

            def _plan_autopilot_path(self):
                return False  # no Isaac Sim

            # Bind the real method
            _skip_unsolvable_episode = JetbotKeyboardController._skip_unsolvable_episode

        ctrl = FakeController()
        # Record some in-progress steps that should be abandoned
        for _ in range(7):
            ctrl.recorder.record_step(np.ones(10), np.zeros(2), 0.0, False)

        ctrl._skip_unsolvable_episode()

        assert ctrl.auto_episode_count == 0, "Episode count must not increment on skip"
        assert ctrl.auto_step_count == 0, "Step count must be reset on skip"
        assert len(ctrl.recorder.observations) == 0, "In-progress steps must be discarded"
        assert len(ctrl.recorder.episode_starts) == 0, "No episode should be finalized"

    def test_skip_unsolvable_episode_resets_autopilot(self):
        """_skip_unsolvable_episode must call auto_pilot.reset()."""
        from jetbot_keyboard_control import (
            DemoRecorder, AutoPilot, SceneManager,
            JetbotKeyboardController,
        )

        class FakeController:
            headless_tui = True
            auto_step_count = 5
            auto_episode_count = 2
            current_obs = np.zeros(10)
            MAX_CONSECUTIVE_SKIPS = 100

            def __init__(self):
                self._consecutive_skips = 0
                self.recorder = DemoRecorder(obs_dim=10, action_dim=2)
                self.recorder.start_recording()
                self.auto_pilot = AutoPilot(noise_linear=0.0, noise_angular=0.0)
                # Simulate fallback state
                self.auto_pilot._using_fallback = True
                self.auto_pilot._path = []
                mock_world = Mock()
                mock_world.scene = Mock()
                mock_world.scene.add = Mock(side_effect=lambda x: x)
                self.scene_manager = SceneManager(mock_world)
                self.scene_manager.goal_position = np.array([1.0, 1.0, 0.0])

            def _reset_recording_episode(self):
                pass

            def _build_current_observation(self):
                return np.zeros(10)

            def _draw_debug_overlays(self):
                pass

            def _plan_autopilot_path(self):
                return False  # still no path

            _skip_unsolvable_episode = JetbotKeyboardController._skip_unsolvable_episode

        ctrl = FakeController()
        ctrl._skip_unsolvable_episode()

        assert not ctrl.auto_pilot._using_fallback, "AutoPilot fallback flag must be cleared after reset"
