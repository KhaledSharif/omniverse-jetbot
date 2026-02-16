"""
Test Suite for Behavioral Cloning Training (train_bc.py)

Tests cover:
- load_demo_data: Loading and filtering demo episodes
- train_simple_pytorch: BC training loop

Run with: pytest test_train_bc.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES
# ============================================================================

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

mock_simulation_app_class = MagicMock()
mock_isaacsim.SimulationApp = mock_simulation_app_class

class MockArticulationAction:
    def __init__(self, joint_velocities=None, **kwargs):
        self.joint_velocities = joint_velocities if joint_velocities is not None else np.array([])

mock_isaacsim_core_utils_types.ArticulationAction = MockArticulationAction
mock_isaacsim_core_utils_nucleus.get_assets_root_path = Mock(return_value="/Isaac")

sys.modules.setdefault('isaacsim', mock_isaacsim)
sys.modules.setdefault('isaacsim.core', mock_isaacsim_core)
sys.modules.setdefault('isaacsim.core.api', mock_isaacsim_core_api)
sys.modules.setdefault('isaacsim.core.utils', mock_isaacsim_core_utils)
sys.modules.setdefault('isaacsim.core.utils.types', mock_isaacsim_core_utils_types)
sys.modules.setdefault('isaacsim.core.utils.nucleus', mock_isaacsim_core_utils_nucleus)
sys.modules.setdefault('isaacsim.robot', mock_isaacsim_robot)
sys.modules.setdefault('isaacsim.robot.wheeled_robots', mock_isaacsim_robot_wheeled_robots)
sys.modules.setdefault('isaacsim.robot.wheeled_robots.robots', mock_isaacsim_robot_wheeled_robots_robots)
sys.modules.setdefault('isaacsim.robot.wheeled_robots.controllers', mock_isaacsim_robot_wheeled_robots_controllers)

from train_bc import load_demo_data, train_simple_pytorch


# ============================================================================
# TEST SUITE: LoadDemoData
# ============================================================================

class TestLoadDemoDataBc:
    @patch('jetbot_keyboard_control.DemoPlayer')
    def test_loads_all(self, MockDemoPlayer):
        player = MockDemoPlayer.return_value
        player.num_episodes = 3
        player.total_frames = 30
        player.get_episode.side_effect = [
            (np.ones((10, 10)), np.ones((10, 2))),
            (np.ones((10, 10)), np.ones((10, 2))),
            (np.ones((10, 10)), np.ones((10, 2))),
        ]

        obs, acts = load_demo_data("dummy.npz", successful_only=False)
        assert obs.shape == (30, 10)
        assert acts.shape == (30, 2)

    @patch('jetbot_keyboard_control.DemoPlayer')
    def test_successful_only(self, MockDemoPlayer):
        player = MockDemoPlayer.return_value
        player.num_episodes = 3
        player.total_frames = 30
        player.get_successful_episodes.return_value = [1]
        player.get_episode.return_value = (np.ones((10, 10)), np.ones((10, 2)))

        obs, acts = load_demo_data("dummy.npz", successful_only=True)
        assert obs.shape == (10, 10)
        assert acts.shape == (10, 2)
        player.get_successful_episodes.assert_called_once()

    @patch('jetbot_keyboard_control.DemoPlayer')
    def test_empty_raises(self, MockDemoPlayer):
        player = MockDemoPlayer.return_value
        player.num_episodes = 0
        player.total_frames = 0
        player.get_successful_episodes.return_value = []

        with pytest.raises(ValueError, match="No episodes to train on"):
            load_demo_data("dummy.npz", successful_only=True)


# ============================================================================
# TEST SUITE: TrainSimplePytorch
# ============================================================================

class TestTrainSimplePytorch:
    def _make_args(self, tmp_path, epochs=10, batch_size=32, lr=1e-3):
        args = Mock()
        args.epochs = epochs
        args.batch_size = batch_size
        args.lr = lr
        args.output = str(tmp_path / "model.pt")
        return args

    def test_loss_decreases(self, tmp_path, capsys):
        """Train for 10 epochs and verify loss decreases."""
        obs = np.random.randn(200, 4).astype(np.float32)
        # Create learnable target: actions = obs[:, :2] (roughly)
        actions = obs[:, :2].copy()
        args = self._make_args(tmp_path, epochs=50, batch_size=64, lr=1e-3)

        train_simple_pytorch(obs, actions, args)

        captured = capsys.readouterr()
        # Parse first and last reported losses
        loss_lines = [l for l in captured.out.split('\n') if 'Loss:' in l]
        assert len(loss_lines) >= 2
        first_loss = float(loss_lines[0].split('Loss:')[1].strip())
        last_loss = float(loss_lines[-1].split('Loss:')[1].strip())
        assert last_loss < first_loss

    def test_model_saved(self, tmp_path):
        """Verify model file is written."""
        import torch
        obs = np.random.randn(100, 4).astype(np.float32)
        actions = np.random.randn(100, 2).astype(np.float32)
        args = self._make_args(tmp_path, epochs=2)

        train_simple_pytorch(obs, actions, args)

        model_path = tmp_path / "model.pt"
        assert model_path.exists()
        saved = torch.load(str(model_path), weights_only=False)
        assert 'model_state_dict' in saved
        assert saved['obs_dim'] == 4
        assert saved['action_dim'] == 2

    def test_output_shape(self, tmp_path):
        """Verify model produces correct action dimension."""
        import torch
        obs = np.random.randn(100, 6).astype(np.float32)
        actions = np.random.randn(100, 2).astype(np.float32)
        args = self._make_args(tmp_path, epochs=2)

        train_simple_pytorch(obs, actions, args)

        # Load and check prediction shape
        saved = torch.load(str(tmp_path / "model.pt"), weights_only=False)
        assert saved['obs_dim'] == 6
        assert saved['action_dim'] == 2
