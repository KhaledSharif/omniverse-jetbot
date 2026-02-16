"""
Test Suite for SAC/TQC Training Pipeline (train_sac.py)

Tests cover:
- validate_demo_data: Demo data validation
- load_demo_transitions: Reconstructing (obs, action, reward, next_obs, done) tuples
- inject_layernorm_into_critics: Post-hoc LayerNorm injection

Run with: pytest test_train_sac.py -v
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

from train_sac import (
    validate_demo_data,
    load_demo_transitions,
    inject_layernorm_into_critics,
)


# ============================================================================
# HELPERS
# ============================================================================

def make_demo_npz(path, num_episodes=20, steps_per_ep=50, obs_dim=34,
                  success_rate=0.5):
    """Create a synthetic demo NPZ file for testing."""
    total = num_episodes * steps_per_ep
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    actions = np.random.randn(total, 2).astype(np.float32)
    rewards = np.random.randn(total).astype(np.float32)
    episode_lengths = np.full(num_episodes, steps_per_ep)
    episode_success = np.zeros(num_episodes, dtype=bool)
    episode_success[:int(num_episodes * success_rate)] = True
    np.savez(path,
             observations=obs, actions=actions, rewards=rewards,
             episode_lengths=episode_lengths, episode_success=episode_success)
    return path


# ============================================================================
# TEST SUITE: ValidateDemoData
# ============================================================================

class TestValidateDemoDataSac:
    def test_valid_data(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=20, steps_per_ep=50, success_rate=0.5)
        result = validate_demo_data(str(path))
        assert result['episodes'] == 20
        assert result['transitions'] == 1000
        assert result['successful'] == 10

    def test_too_few_episodes(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=5, steps_per_ep=200, success_rate=1.0)
        with pytest.raises(ValueError, match="Need >= 10 episodes"):
            validate_demo_data(str(path))

    def test_too_few_transitions(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=10, steps_per_ep=10, success_rate=1.0)
        with pytest.raises(ValueError, match="Need >= 500 transitions"):
            validate_demo_data(str(path))

    def test_too_few_successful(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=20, steps_per_ep=50, success_rate=0.1)
        with pytest.raises(ValueError, match="Need >= 3 successful"):
            validate_demo_data(str(path))


# ============================================================================
# TEST SUITE: LoadDemoTransitions
# ============================================================================

class TestLoadDemoTransitions:
    def test_shapes(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=5, steps_per_ep=20, obs_dim=34)
        obs, actions, rewards, next_obs, dones = load_demo_transitions(str(path))
        assert obs.shape == (100, 34)
        assert actions.shape == (100, 2)
        assert rewards.shape == (100,)
        assert next_obs.shape == (100, 34)
        assert dones.shape == (100,)

    def test_next_obs_within_episode(self, tmp_path):
        """next_obs[t] == obs[t+1] within an episode."""
        path = tmp_path / "demo.npz"
        num_ep = 2
        steps = 10
        total = num_ep * steps
        obs_data = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        np.savez(path,
                 observations=obs_data,
                 actions=np.zeros((total, 2), dtype=np.float32),
                 rewards=np.zeros(total, dtype=np.float32),
                 episode_lengths=np.full(num_ep, steps))
        obs, _, _, next_obs, _ = load_demo_transitions(str(path))
        # Within episode 1 (indices 0-9), next_obs[t] should match obs[t+1]
        for t in range(steps - 1):
            np.testing.assert_array_equal(next_obs[t], obs[t + 1])
        # Within episode 2 (indices 10-19)
        for t in range(steps, 2 * steps - 1):
            np.testing.assert_array_equal(next_obs[t], obs[t + 1])

    def test_dones_at_boundaries(self, tmp_path):
        """dones=1.0 only at last step of each episode."""
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=3, steps_per_ep=10)
        _, _, _, _, dones = load_demo_transitions(str(path))
        # dones should be 1.0 at indices 9, 19, 29
        expected_done_indices = [9, 19, 29]
        for i in range(30):
            if i in expected_done_indices:
                assert dones[i] == 1.0, f"Expected done=1.0 at index {i}"
            else:
                assert dones[i] == 0.0, f"Expected done=0.0 at index {i}"

    def test_terminal_next_obs(self, tmp_path):
        """Last step's next_obs equals last obs (copy, since it's terminal)."""
        path = tmp_path / "demo.npz"
        num_ep = 2
        steps = 5
        total = num_ep * steps
        obs_data = np.arange(total * 2, dtype=np.float32).reshape(total, 2)
        np.savez(path,
                 observations=obs_data,
                 actions=np.zeros((total, 2), dtype=np.float32),
                 rewards=np.zeros(total, dtype=np.float32),
                 episode_lengths=np.full(num_ep, steps))
        obs, _, _, next_obs, _ = load_demo_transitions(str(path))
        # Terminal steps: indices 4 and 9
        np.testing.assert_array_equal(next_obs[4], obs[4])
        np.testing.assert_array_equal(next_obs[9], obs[9])


# ============================================================================
# TEST SUITE: InjectLayernormIntoCritics
# ============================================================================

class TestInjectLayernorm:
    def _count_layernorms(self, sequential):
        import torch.nn as nn
        return sum(1 for m in sequential if isinstance(m, nn.LayerNorm))

    def _count_linears(self, sequential):
        import torch.nn as nn
        return sum(1 for m in sequential if isinstance(m, nn.Linear))

    def test_tqc_injection(self):
        """Verify LayerNorm layers inserted after hidden Linear layers for TQC."""
        import torch
        import torch.nn as nn

        # Build mock TQC model with quantile_critics
        critic_net = nn.Sequential(
            nn.Linear(36, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 25),  # output layer (25 quantiles)
        )

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = Mock()
        model.critic.quantile_critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(36, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 25),
            )
            for _ in range(5)
        ])
        # Need to make hasattr check work for 'quantile_critics'
        model.critic.quantile_critics = model.critic.quantile_critics
        # Override hasattr by making critic a real object
        class FakeCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.quantile_critics = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(36, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 25),
                    )
                    for _ in range(5)
                ])
        model.critic = FakeCritic()
        model.critic_target = FakeCritic()

        inject_layernorm_into_critics(model)

        # Each critic should now have 2 LayerNorm layers (after each hidden Linear)
        for i, net in enumerate(model.critic.quantile_critics):
            ln_count = self._count_layernorms(net)
            assert ln_count == 2, f"Critic {i}: expected 2 LayerNorm, got {ln_count}"

    def test_sac_injection(self):
        """Verify LayerNorm layers inserted for SAC qf0/qf1 structure."""
        import torch
        import torch.nn as nn

        class FakeSACCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.qf0 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )
                self.qf1 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = FakeSACCritic()
        model.critic_target = FakeSACCritic()

        inject_layernorm_into_critics(model)

        assert self._count_layernorms(model.critic.qf0) == 2
        assert self._count_layernorms(model.critic.qf1) == 2

    def test_target_synced(self):
        """Verify critic_target matches critic state_dict after injection."""
        import torch
        import torch.nn as nn

        class FakeSACCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.qf0 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )
                self.qf1 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = FakeSACCritic()
        model.critic_target = FakeSACCritic()

        inject_layernorm_into_critics(model)

        # State dicts should match
        for key in model.critic.state_dict():
            torch.testing.assert_close(
                model.critic.state_dict()[key],
                model.critic_target.state_dict()[key],
            )

    def test_output_layer_no_layernorm(self):
        """No LayerNorm should be inserted after the final (output) Linear layer."""
        import torch
        import torch.nn as nn

        class FakeSACCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.qf0 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )
                self.qf1 = nn.Sequential(
                    nn.Linear(36, 256), nn.ReLU(),
                    nn.Linear(256, 1),
                )

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = FakeSACCritic()
        model.critic_target = FakeSACCritic()

        inject_layernorm_into_critics(model)

        # Last module should NOT be LayerNorm
        assert not isinstance(list(model.critic.qf0)[-1], nn.LayerNorm)
        assert not isinstance(list(model.critic.qf1)[-1], nn.LayerNorm)
