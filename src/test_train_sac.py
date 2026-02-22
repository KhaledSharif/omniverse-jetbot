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
    symlog,
    ChunkCVAEFeatureExtractor,
    TemporalCVAEFeatureExtractor,
    pretrain_chunk_cvae,
    CostReplayBuffer,
    MeanCostCritic,
    _create_safe_tqc_class,
    _create_dual_policy_class,
)
from demo_utils import extract_action_chunks, make_chunk_transitions, build_frame_stacks
from jetbot_rl_env import ChunkedEnvWrapper, FrameStackWrapper
from jetbot_keyboard_control import RewardComputer


# ============================================================================
# HELPERS
# ============================================================================

def make_demo_npz(path, num_episodes=20, steps_per_ep=50, obs_dim=34,
                  success_rate=0.5, include_costs=False):
    """Create a synthetic demo NPZ file for testing."""
    total = num_episodes * steps_per_ep
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    actions = np.random.randn(total, 2).astype(np.float32)
    rewards = np.random.randn(total).astype(np.float32)
    episode_lengths = np.full(num_episodes, steps_per_ep)
    episode_success = np.zeros(num_episodes, dtype=bool)
    episode_success[:int(num_episodes * success_rate)] = True
    # Build dones: 1.0 at end of each episode, 0.0 elsewhere
    dones = np.zeros(total, dtype=np.float32)
    offset = 0
    for length in episode_lengths:
        dones[offset + length - 1] = 1.0
        offset += length
    save_kwargs = dict(
        observations=obs, actions=actions, rewards=rewards, dones=dones,
        episode_lengths=episode_lengths, episode_success=episode_success,
    )
    if include_costs:
        costs = np.random.rand(total).astype(np.float32)
        metadata = {'obs_dim': obs_dim, 'action_dim': 2, 'has_cost': True,
                     'num_episodes': num_episodes, 'total_frames': total}
        save_kwargs['costs'] = costs
        save_kwargs['metadata'] = np.array(metadata, dtype=object)
    np.savez(path, **save_kwargs)
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

    def test_uses_recorded_dones(self, tmp_path):
        """When NPZ has dones field, use recorded dones instead of blanket episode-end."""
        path = tmp_path / "demo.npz"
        num_ep = 2
        steps = 5
        total = num_ep * steps
        obs_data = np.random.randn(total, 2).astype(np.float32)
        dones_data = np.zeros(total, dtype=bool)
        # Episode 1: collision at step 3 (index 2), episode ends at index 4
        dones_data[2] = True   # mid-episode collision
        dones_data[4] = True   # episode boundary
        # Episode 2: timeout (no true terminal), episode ends at index 9
        # dones_data[9] stays False — will be set to 1.0 by fallback
        np.savez(path,
                 observations=obs_data,
                 actions=np.zeros((total, 2), dtype=np.float32),
                 rewards=np.zeros(total, dtype=np.float32),
                 dones=dones_data,
                 episode_lengths=np.full(num_ep, steps))
        _, _, _, _, dones = load_demo_transitions(str(path))
        # Mid-episode collision should be preserved
        assert dones[2] == 1.0, "Collision done should be preserved from NPZ"
        assert dones[4] == 1.0, "Episode-end done should be preserved from NPZ"
        # Timeout fallback: last step of ep2 should be 1.0
        assert dones[9] == 1.0, "Timeout episode should fallback to done=1.0"
        # Non-terminal steps should be 0.0
        assert dones[0] == 0.0
        assert dones[1] == 0.0
        assert dones[5] == 0.0

    def test_legacy_npz_without_dones_key(self, tmp_path):
        """Legacy NPZ files without dones key should still load correctly."""
        path = tmp_path / "demo.npz"
        total = 10
        np.savez(path,
                 observations=np.random.randn(total, 2).astype(np.float32),
                 actions=np.zeros((total, 2), dtype=np.float32),
                 rewards=np.zeros(total, dtype=np.float32),
                 episode_lengths=np.array([5, 5]))
        _, _, _, _, dones = load_demo_transitions(str(path))
        # Should fallback to done=1.0 at episode boundaries
        assert dones[4] == 1.0
        assert dones[9] == 1.0
        assert dones[0] == 0.0
        assert dones[3] == 0.0


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

    def _count_ofn(self, sequential):
        return sum(1 for m in sequential if type(m).__name__ == 'OutputFeatureNorm')

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
        # and 1 OFN layer (before output Linear)
        for i, net in enumerate(model.critic.quantile_critics):
            ln_count = self._count_layernorms(net)
            assert ln_count == 2, f"Critic {i}: expected 2 LayerNorm, got {ln_count}"
            ofn_count = self._count_ofn(net)
            assert ofn_count == 1, f"Critic {i}: expected 1 OFN, got {ofn_count}"

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
        assert self._count_ofn(model.critic.qf0) == 1
        assert self._count_ofn(model.critic.qf1) == 1

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

        # Last module should be Linear (output), second-to-last should be OFN
        modules_qf0 = list(model.critic.qf0)
        modules_qf1 = list(model.critic.qf1)
        assert isinstance(modules_qf0[-1], nn.Linear), "Last module should be Linear"
        assert isinstance(modules_qf1[-1], nn.Linear), "Last module should be Linear"
        assert type(modules_qf0[-2]).__name__ == 'OutputFeatureNorm', "Second-to-last should be OFN"
        assert type(modules_qf1[-2]).__name__ == 'OutputFeatureNorm', "Second-to-last should be OFN"

    def test_ofn_produces_unit_norm(self):
        """Verify OFN output has unit L2 norm."""
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

        # Find the OFN module in qf0
        ofn = None
        for m in model.critic.qf0:
            if type(m).__name__ == 'OutputFeatureNorm':
                ofn = m
                break
        assert ofn is not None, "OFN not found"

        # Pass random input and verify unit norm
        x = torch.randn(8, 256)
        out = ofn(x)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-6, rtol=1e-6)

    def test_sac_5_critics_injection(self):
        """Verify injection works on SAC with 5 qf* networks (dynamic discovery)."""
        import torch
        import torch.nn as nn

        class FakeSAC5Critic(nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(5):
                    setattr(self, f'qf{i}', nn.Sequential(
                        nn.Linear(36, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 1),
                    ))

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = FakeSAC5Critic()
        model.critic_target = FakeSAC5Critic()

        inject_layernorm_into_critics(model)

        for i in range(5):
            net = getattr(model.critic, f'qf{i}')
            assert self._count_layernorms(net) == 2, f"qf{i}: expected 2 LayerNorm"
            assert self._count_ofn(net) == 1, f"qf{i}: expected 1 OFN"

    def test_tqc_alt_attr_critics(self):
        """Verify injection works when TQC uses 'critics' instead of 'quantile_critics'."""
        import torch
        import torch.nn as nn

        class FakeTQCCriticAlt(nn.Module):
            def __init__(self):
                super().__init__()
                self.critics = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(36, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, 25),
                    )
                    for _ in range(2)
                ])

        model = Mock()
        model.device = torch.device('cpu')
        model.lr_schedule = Mock(return_value=3e-4)
        model.critic = FakeTQCCriticAlt()
        model.critic_target = FakeTQCCriticAlt()

        inject_layernorm_into_critics(model)

        for i, net in enumerate(model.critic.critics):
            assert self._count_layernorms(net) == 2, f"Critic {i}: expected 2 LayerNorm"
            assert self._count_ofn(net) == 1, f"Critic {i}: expected 1 OFN"


# ============================================================================
# TEST SUITE: Symlog
# ============================================================================

class TestSymlog:
    def test_near_identity_at_zero(self):
        """symlog(0) == 0."""
        import torch
        x = torch.tensor([0.0])
        result = symlog(x)
        assert result.item() == pytest.approx(0.0, abs=1e-7)

    def test_compression_at_large_values(self):
        """Large values are compressed: |symlog(100)| < 100."""
        import torch
        x = torch.tensor([100.0])
        result = symlog(x)
        assert 0 < result.item() < 100.0

    def test_symmetry(self):
        """symlog(-x) == -symlog(x)."""
        import torch
        x = torch.tensor([1.0, 5.0, 10.0])
        assert torch.allclose(symlog(-x), -symlog(x))

    def test_lidar_range_preservation(self):
        """LiDAR values [0, 1] are nearly preserved (near-identity region)."""
        import torch
        x = torch.linspace(0, 1, 100)
        result = symlog(x)
        # symlog(x) = sign(x)*log(|x|+1) ≈ x for small x
        # For x=1: symlog(1) = log(2) ≈ 0.693, so values are compressed but close
        assert torch.all(result >= 0)
        assert torch.all(result <= 1.0)  # log(2) ≈ 0.693 < 1.0


# ============================================================================
# TEST SUITE: ChunkedEnvWrapper
# ============================================================================

class TestChunkedEnvWrapper:
    def _make_mock_env(self, obs_dim=34, act_dim=2):
        """Create a minimal gymnasium.Env subclass for testing."""
        import gymnasium as gym

        class _MinimalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

            def reset(self, **kwargs):
                return np.zeros(obs_dim, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}

        return _MinimalEnv()

    def test_action_space_shape(self):
        """Wrapper action space is (chunk_size * 2,)."""
        inner = self._make_mock_env()
        wrapper = ChunkedEnvWrapper(inner, chunk_size=5, gamma=0.99)
        assert wrapper.action_space.shape == (10,)

    def test_observation_space_unchanged(self):
        """Observation space is unchanged by wrapper."""
        inner = self._make_mock_env()
        wrapper = ChunkedEnvWrapper(inner, chunk_size=5, gamma=0.99)
        assert wrapper.observation_space.shape == (34,)

    def test_cumulative_reward(self):
        """Verify R_chunk = sum(gamma^i * r_i)."""
        inner = self._make_mock_env()
        k = 3
        gamma = 0.9
        rewards = [1.0, 2.0, 3.0]
        obs = np.zeros(34, dtype=np.float32)

        call_count = [0]
        def mock_step(action):
            r = rewards[call_count[0]]
            call_count[0] += 1
            return obs, r, False, False, {}

        inner.step = mock_step
        wrapper = ChunkedEnvWrapper(inner, chunk_size=k, gamma=gamma)

        action_flat = np.zeros(k * 2, dtype=np.float32)
        _, r_chunk, terminated, truncated, _ = wrapper.step(action_flat)

        expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0
        assert abs(r_chunk - expected) < 1e-6

    def test_early_termination(self):
        """Inner env terminates mid-chunk -> partial reward + done=True."""
        inner = self._make_mock_env()
        k = 5
        obs = np.zeros(34, dtype=np.float32)

        call_count = [0]
        def mock_step(action):
            call_count[0] += 1
            if call_count[0] == 2:
                return obs, 10.0, True, False, {}  # terminate at step 2
            return obs, 1.0, False, False, {}

        inner.step = mock_step
        wrapper = ChunkedEnvWrapper(inner, chunk_size=k, gamma=0.99)

        action_flat = np.zeros(k * 2, dtype=np.float32)
        _, r_chunk, terminated, truncated, _ = wrapper.step(action_flat)

        assert terminated is True
        assert call_count[0] == 2  # Only 2 inner steps executed
        expected = 1.0 + 0.99 * 10.0
        assert abs(r_chunk - expected) < 1e-6

    def test_full_chunk_execution(self):
        """All k sub-steps execute when no termination."""
        inner = self._make_mock_env()
        k = 4
        obs = np.zeros(34, dtype=np.float32)

        call_count = [0]
        def mock_step(action):
            call_count[0] += 1
            return obs, 1.0, False, False, {}

        inner.step = mock_step
        wrapper = ChunkedEnvWrapper(inner, chunk_size=k, gamma=0.99)

        action_flat = np.zeros(k * 2, dtype=np.float32)
        wrapper.step(action_flat)

        assert call_count[0] == k


# ============================================================================
# TEST SUITE: ChunkCVAEFeatureExtractor
# ============================================================================

class TestChunkCVAEFeatureExtractor:
    def _make_extractor(self, z_dim=8):
        import torch
        import gymnasium as gym
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        cls = ChunkCVAEFeatureExtractor.create(BaseFeaturesExtractor, z_dim=z_dim)
        features_dim = 96 + z_dim
        return cls(obs_space, features_dim=features_dim)

    def test_output_shape(self):
        """Feature extractor produces (batch, 104) from (batch, 34)."""
        import torch
        ext = self._make_extractor(z_dim=8)
        obs = torch.randn(8, 34)
        out = ext(obs)
        assert out.shape == (8, 104)

    def test_z_slot_is_zero(self):
        """Last z_dim entries are zeros during forward()."""
        import torch
        z_dim = 8
        ext = self._make_extractor(z_dim=z_dim)
        obs = torch.randn(4, 34)
        out = ext(obs)
        z_slot = out[:, -z_dim:]
        torch.testing.assert_close(z_slot, torch.zeros(4, z_dim))

    def test_encode_obs_shape(self):
        """encode_obs returns (batch, 96)."""
        import torch
        ext = self._make_extractor(z_dim=8)
        obs = torch.randn(4, 34)
        obs_features = ext.encode_obs(obs)
        assert obs_features.shape == (4, 96)

    def test_gradient_flow(self):
        """Gradients flow through state_mlp and lidar_mlp."""
        import torch
        ext = self._make_extractor(z_dim=8)
        ext.train()
        obs = torch.randn(4, 34)
        out = ext(obs)
        loss = out.sum()
        loss.backward()
        # Check state MLP gradients
        assert ext.state_mlp[0].weight.grad is not None
        # Check lidar MLP gradients
        assert ext.lidar_mlp[0].weight.grad is not None


# ============================================================================
# TEST SUITE: ExtractActionChunks
# ============================================================================

class TestExtractActionChunks:
    def test_chunk_count(self):
        """N_chunks = sum(max(0, L_i - k + 1)) for each episode."""
        # 2 episodes, 10 steps each, chunk_size=3
        # Each episode: 10 - 3 + 1 = 8 chunks, total = 16
        total = 20
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        ep_lengths = np.array([10, 10])

        chunk_obs, chunk_acts = extract_action_chunks(obs, actions, ep_lengths, chunk_size=3)
        assert len(chunk_obs) == 16
        assert chunk_acts.shape == (16, 6)  # 3 * 2 = 6

    def test_respects_episode_boundaries(self):
        """No chunk spans two episodes."""
        # 2 episodes of 5 steps each, chunk_size=3
        total = 10
        obs = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        actions = np.arange(total * 2, dtype=np.float32).reshape(total, 2)
        ep_lengths = np.array([5, 5])

        chunk_obs, chunk_acts = extract_action_chunks(obs, actions, ep_lengths, chunk_size=3)

        # Episode 1: chunks starting at t=0,1,2 (indices 0-4)
        # Episode 2: chunks starting at t=5,6,7 (indices 5-9)
        # Total: 6 chunks
        assert len(chunk_obs) == 6

        # First chunk of episode 2 should start at obs[5], not use obs from episode 1
        # The 4th chunk (index 3) is the first of episode 2
        np.testing.assert_array_equal(chunk_obs[3], obs[5])

    def test_short_episodes_skipped(self):
        """Episodes shorter than chunk_size produce no chunks."""
        total = 12
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        ep_lengths = np.array([2, 10])  # First ep too short for chunk_size=5

        chunk_obs, chunk_acts = extract_action_chunks(obs, actions, ep_lengths, chunk_size=5)
        # Only second episode: 10 - 5 + 1 = 6 chunks
        assert len(chunk_obs) == 6


# ============================================================================
# TEST SUITE: MakeChunkTransitions
# ============================================================================

class TestMakeChunkTransitions:
    def test_discounted_reward(self):
        """R_chunk matches hand-computed sum(gamma^i * r_i)."""
        total = 10
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        rewards = np.ones(total, dtype=np.float32)
        dones = np.zeros(total, dtype=np.float32)
        dones[9] = 1.0  # Terminal at end
        ep_lengths = np.array([10])
        gamma = 0.9
        k = 3

        c_obs, c_acts, c_rews, c_next, c_dones = make_chunk_transitions(
            obs, actions, rewards, dones, ep_lengths, k, gamma)

        # First chunk: r0 + 0.9*r1 + 0.81*r2 = 1 + 0.9 + 0.81 = 2.71
        expected = 1.0 + 0.9 + 0.81
        assert abs(c_rews[0] - expected) < 1e-6

    def test_next_obs_alignment(self):
        """next_obs is observation k steps ahead."""
        total = 10
        obs = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        actions = np.zeros((total, 2), dtype=np.float32)
        rewards = np.ones(total, dtype=np.float32)
        dones = np.zeros(total, dtype=np.float32)
        dones[9] = 1.0
        ep_lengths = np.array([10])
        k = 3

        c_obs, c_acts, c_rews, c_next, c_dones = make_chunk_transitions(
            obs, actions, rewards, dones, ep_lengths, k, 0.99)

        # First chunk starts at t=0, next_obs should be obs[3]
        np.testing.assert_array_equal(c_next[0], obs[3])
        # Second chunk starts at t=1, next_obs should be obs[4]
        np.testing.assert_array_equal(c_next[1], obs[4])

    def test_terminal_chunk(self):
        """done=True for chunks containing terminal steps."""
        total = 10
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        rewards = np.ones(total, dtype=np.float32)
        dones = np.zeros(total, dtype=np.float32)
        dones[9] = 1.0  # Terminal at last step
        ep_lengths = np.array([10])
        k = 3

        c_obs, c_acts, c_rews, c_next, c_dones = make_chunk_transitions(
            obs, actions, rewards, dones, ep_lengths, k, 0.99)

        # Last chunk starts at t=7 (covers steps 7,8,9), step 9 is terminal
        assert c_dones[-1] == 1.0
        # First chunk (steps 0,1,2) should not be terminal
        assert c_dones[0] == 0.0


# ============================================================================
# TEST SUITE: RewardComputer.compute_cost
# ============================================================================

class TestRewardComputerCost:
    def test_proximity_cost_triggered(self):
        """proximity cost = 1.0 when min_lidar_distance < 0.3."""
        info = {'min_lidar_distance': 0.2, 'collision': False}
        assert RewardComputer.compute_cost(info, cost_type='proximity') == 1.0

    def test_proximity_cost_safe(self):
        """proximity cost = 0.0 when min_lidar_distance >= 0.3."""
        info = {'min_lidar_distance': 0.5, 'collision': False}
        assert RewardComputer.compute_cost(info, cost_type='proximity') == 0.0

    def test_collision_cost_triggered(self):
        """collision cost = 1.0 when collision is True."""
        info = {'min_lidar_distance': 0.05, 'collision': True}
        assert RewardComputer.compute_cost(info, cost_type='collision') == 1.0

    def test_collision_cost_safe(self):
        """collision cost = 0.0 when collision is False."""
        info = {'min_lidar_distance': 0.2, 'collision': False}
        assert RewardComputer.compute_cost(info, cost_type='collision') == 0.0

    def test_both_cost_type(self):
        """both = max(proximity, collision)."""
        info = {'min_lidar_distance': 0.2, 'collision': False}
        assert RewardComputer.compute_cost(info, cost_type='both') == 1.0
        info2 = {'min_lidar_distance': 0.5, 'collision': True}
        assert RewardComputer.compute_cost(info2, cost_type='both') == 1.0
        info3 = {'min_lidar_distance': 0.5, 'collision': False}
        assert RewardComputer.compute_cost(info3, cost_type='both') == 0.0

    def test_safe_mode_skips_proximity_penalty(self):
        """safe_mode=True suppresses proximity penalty in dense reward."""
        rc = RewardComputer(mode='dense', safe_mode=True)
        obs = np.zeros(10)
        obs[7] = 1.0  # distance to goal
        next_obs = np.zeros(10)
        next_obs[7] = 0.95
        next_obs[8] = 0.1
        info = {'min_lidar_distance': 0.15, 'collision': False, 'goal_reached': False}
        reward_safe = rc.compute(obs, np.zeros(2), next_obs, info)

        rc_normal = RewardComputer(mode='dense', safe_mode=False)
        reward_normal = rc_normal.compute(obs, np.zeros(2), next_obs, info)

        # safe_mode should give higher reward (no proximity penalty subtracted)
        assert reward_safe > reward_normal


# ============================================================================
# TEST SUITE: ValidateDemoData with cost requirement
# ============================================================================

class TestValidateDemoDataCost:
    def test_require_cost_passes_with_cost_data(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=20, steps_per_ep=50,
                      success_rate=0.5, include_costs=True)
        result = validate_demo_data(str(path), require_cost=True)
        assert result['episodes'] == 20

    def test_require_cost_fails_without_cost_data(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=20, steps_per_ep=50, success_rate=0.5)
        with pytest.raises(ValueError, match="missing cost data"):
            validate_demo_data(str(path), require_cost=True)

    def test_no_require_cost_passes_without_cost_data(self, tmp_path):
        """When require_cost=False, missing cost data is fine."""
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=20, steps_per_ep=50, success_rate=0.5)
        result = validate_demo_data(str(path), require_cost=False)
        assert result['episodes'] == 20


# ============================================================================
# TEST SUITE: LoadDemoTransitions with costs
# ============================================================================

class TestLoadDemoTransitionsCosts:
    def test_load_costs_returns_6_tuple(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=5, steps_per_ep=20, include_costs=True)
        result = load_demo_transitions(str(path), load_costs=True)
        assert len(result) == 6
        obs, actions, rewards, next_obs, dones, costs = result
        assert costs.shape == (100,)
        assert costs.dtype == np.float32

    def test_load_costs_false_returns_5_tuple(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=5, steps_per_ep=20, include_costs=True)
        result = load_demo_transitions(str(path), load_costs=False)
        assert len(result) == 5

    def test_load_costs_fallback_zeros(self, tmp_path):
        """Missing costs key -> zeros array."""
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=5, steps_per_ep=20, include_costs=False)
        result = load_demo_transitions(str(path), load_costs=True)
        assert len(result) == 6
        costs = result[5]
        assert np.all(costs == 0.0)


# ============================================================================
# TEST SUITE: MakeChunkTransitions with costs
# ============================================================================

class TestMakeChunkTransitionsCosts:
    def test_chunk_cost_accumulation(self):
        """Chunk cost matches hand-computed sum(gamma^i * c_i)."""
        total = 10
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        rewards = np.ones(total, dtype=np.float32)
        dones = np.zeros(total, dtype=np.float32)
        dones[9] = 1.0
        costs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                         dtype=np.float32)
        ep_lengths = np.array([10])
        gamma = 0.9
        k = 3

        result = make_chunk_transitions(
            obs, actions, rewards, dones, ep_lengths, k, gamma, demo_costs=costs)
        assert len(result) == 6
        c_obs, c_acts, c_rews, c_next, c_dones, c_costs = result

        # First chunk (steps 0,1,2): c0 + 0.9*c1 + 0.81*c2 = 1.0 + 0 + 0.81 = 1.81
        expected = 1.0 + 0.9 * 0.0 + 0.81 * 1.0
        assert abs(c_costs[0] - expected) < 1e-6

    def test_no_costs_returns_5_tuple(self):
        """Without demo_costs, returns 5-tuple."""
        total = 10
        obs = np.random.randn(total, 34).astype(np.float32)
        actions = np.random.randn(total, 2).astype(np.float32)
        rewards = np.ones(total, dtype=np.float32)
        dones = np.zeros(total, dtype=np.float32)
        dones[9] = 1.0
        ep_lengths = np.array([10])

        result = make_chunk_transitions(
            obs, actions, rewards, dones, ep_lengths, 3, 0.9)
        assert len(result) == 5


# ============================================================================
# TEST SUITE: ChunkedEnvWrapper cost accumulation
# ============================================================================

class TestChunkedEnvWrapperCost:
    def _make_mock_env(self, obs_dim=34, act_dim=2):
        import gymnasium as gym

        class _MinimalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

            def reset(self, **kwargs):
                return np.zeros(obs_dim, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {'cost': 0.0}

        return _MinimalEnv()

    def test_cost_chunk_accumulated(self):
        """c_chunk = sum(gamma^i * cost_i) across sub-steps."""
        inner = self._make_mock_env()
        k = 3
        gamma = 0.9
        costs = [1.0, 0.0, 1.0]
        obs = np.zeros(34, dtype=np.float32)

        call_count = [0]
        def mock_step(action):
            c = costs[call_count[0]]
            call_count[0] += 1
            return obs, 1.0, False, False, {'cost': c}

        inner.step = mock_step
        wrapper = ChunkedEnvWrapper(inner, chunk_size=k, gamma=gamma)

        action_flat = np.zeros(k * 2, dtype=np.float32)
        _, _, _, _, info = wrapper.step(action_flat)

        expected_cost = 1.0 + 0.9 * 0.0 + 0.81 * 1.0
        assert abs(info['cost_chunk'] - expected_cost) < 1e-6

    def test_cost_chunk_on_early_termination(self):
        """Early termination -> partial cost chunk."""
        inner = self._make_mock_env()
        k = 5
        obs = np.zeros(34, dtype=np.float32)

        call_count = [0]
        def mock_step(action):
            call_count[0] += 1
            if call_count[0] == 2:
                return obs, 10.0, True, False, {'cost': 1.0}
            return obs, 1.0, False, False, {'cost': 0.5}

        inner.step = mock_step
        wrapper = ChunkedEnvWrapper(inner, chunk_size=k, gamma=0.99)

        action_flat = np.zeros(k * 2, dtype=np.float32)
        _, _, terminated, _, info = wrapper.step(action_flat)

        assert terminated is True
        expected_cost = 0.5 + 0.99 * 1.0
        assert abs(info['cost_chunk'] - expected_cost) < 1e-6


# ============================================================================
# TEST SUITE: CostReplayBuffer
# ============================================================================

class TestCostReplayBuffer:
    def test_add_and_sample(self):
        """Basic add/sample cycle works."""
        import torch
        device = torch.device('cpu')
        demo_costs = np.array([0.5, 1.0, 0.0, 0.5], dtype=np.float32)
        buf = CostReplayBuffer(100, device, demo_costs=demo_costs)

        # Add some online costs
        for c in [0.1, 0.2, 0.3]:
            buf.add(c)

        # Sample demo indices
        demo_idx = np.array([0, 2])
        online_idx = np.array([0, 1])
        result = buf.sample(demo_idx, online_idx)

        assert result.shape == (4, 1)
        # Demo costs: [0.5, 0.0], online costs: [0.1, 0.2]
        assert abs(result[0].item() - 0.5) < 1e-6
        assert abs(result[1].item() - 0.0) < 1e-6
        assert abs(result[2].item() - 0.1) < 1e-6
        assert abs(result[3].item() - 0.2) < 1e-6

    def test_demo_only_sample(self):
        """Sample with online_indices=None returns only demo costs."""
        import torch
        device = torch.device('cpu')
        demo_costs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf = CostReplayBuffer(100, device, demo_costs=demo_costs)

        demo_idx = np.array([0, 1, 2])
        result = buf.sample(demo_idx, None)
        assert result.shape == (3, 1)


# ============================================================================
# TEST SUITE: MeanCostCritic
# ============================================================================

class TestMeanCostCritic:
    def test_output_shape(self):
        """Cost critic produces list of n_critics tensors, each (batch, 1)."""
        import torch
        import gymnasium as gym
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        fe_cls = ChunkCVAEFeatureExtractor.get_class(BaseFeaturesExtractor, z_dim=8)
        features_dim = 104
        action_dim = 20  # chunk_size=10 * act_dim=2

        device = torch.device('cpu')
        critic, critic_target = MeanCostCritic.create(
            n_critics=2, features_dim=features_dim, action_dim=action_dim,
            fe_class=fe_cls, fe_kwargs=dict(features_dim=features_dim),
            obs_space=obs_space, device=device
        )

        obs = torch.randn(8, 34)
        actions = torch.randn(8, 20)
        outputs = critic(obs, actions)
        assert len(outputs) == 2
        assert outputs[0].shape == (8, 1)
        assert outputs[1].shape == (8, 1)

    def test_target_matches_initial(self):
        """critic_target should have same weights as critic initially."""
        import torch
        import gymnasium as gym
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        fe_cls = ChunkCVAEFeatureExtractor.get_class(BaseFeaturesExtractor, z_dim=8)

        device = torch.device('cpu')
        critic, critic_target = MeanCostCritic.create(
            n_critics=2, features_dim=104, action_dim=20,
            fe_class=fe_cls, fe_kwargs=dict(features_dim=104),
            obs_space=obs_space, device=device
        )

        for key in critic.state_dict():
            torch.testing.assert_close(
                critic.state_dict()[key],
                critic_target.state_dict()[key],
            )


# ============================================================================
# TEST SUITE: DualPolicyTQC (IBRL)
# ============================================================================

class TestDualPolicyTQC:
    """Tests for _create_dual_policy_class (IBRL dual-policy)."""

    def _make_base_cls(self):
        """Return a minimal stub that quacks like TQC/SAC for our tests."""
        import torch.nn as nn

        class _StubBase:
            pass

        return _StubBase

    def _make_dual_policy_instance(self):
        """Create a bare DualPolicyTQC instance without calling __init__."""
        import torch
        import torch.nn as nn

        DualPolicyClass = _create_dual_policy_class(self._make_base_cls())
        obj = object.__new__(DualPolicyClass)
        obj.il_actor = None
        obj.il_noise_std = 0.0
        obj.il_soft = False
        obj.il_beta = 10.0
        obj.device = torch.device('cpu')
        return obj, DualPolicyClass

    def _make_fake_tqc_critic(self, n_critics=2, n_quantiles=4, obs_dim=34, act_dim=2):
        """FakeCritic with quantile_critics and Identity features_extractor."""
        import torch
        import torch.nn as nn

        class _FakeTQCCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.features_extractor = nn.Identity()
                self.quantile_critics = nn.ModuleList([
                    nn.Linear(obs_dim + act_dim, n_quantiles)
                    for _ in range(n_critics)
                ])

        critic = _FakeTQCCritic()
        # Make each quantile critic return known constant outputs
        for i, net in enumerate(critic.quantile_critics):
            with torch.no_grad():
                nn.init.constant_(net.weight, 0.0)
                nn.init.constant_(net.bias, float(i + 1))  # net 0 → 1.0, net 1 → 2.0

        return critic

    def test_ibrl_min_q_tqc_path(self):
        """_ibrl_min_q returns (batch,) min-over-critics for TQC critic."""
        import torch

        obj, _ = self._make_dual_policy_instance()
        critic = self._make_fake_tqc_critic(n_critics=2, n_quantiles=4)
        obj.critic = critic

        batch = 3
        obs = torch.zeros(batch, 34)
        act = torch.zeros(batch, 2)

        result = obj._ibrl_min_q(obs, act)

        assert result.shape == (batch,), f"Expected ({batch},), got {result.shape}"
        # net 0 outputs all-ones → mean=1.0, net 1 outputs all-twos → mean=2.0
        # min over critics = 1.0
        torch.testing.assert_close(result, torch.ones(batch), atol=1e-5, rtol=1e-5)

    def test_ibrl_min_q_sac_path(self):
        """_ibrl_min_q with list-returning critic (SAC-style) → correct min."""
        import torch

        obj, _ = self._make_dual_policy_instance()

        # SAC-style critic that returns [q0, q1] as list
        class _SACCritic:
            def __call__(self, obs, act):
                batch = obs.shape[0]
                q0 = torch.full((batch, 1), 3.0)
                q1 = torch.full((batch, 1), 7.0)
                return [q0, q1]

            def __getattr__(self, name):
                # No quantile_critics attribute → returns None from _get_tqc_quantile_critics
                raise AttributeError(name)

        obj.critic = _SACCritic()

        batch = 5
        obs = torch.zeros(batch, 34)
        act = torch.zeros(batch, 2)

        result = obj._ibrl_min_q(obs, act)

        assert result.shape == (batch,)
        torch.testing.assert_close(result, torch.full((batch,), 3.0), atol=1e-5, rtol=1e-5)

    def test_greedy_selects_il_when_q_il_higher(self):
        """Greedy selection returns IL action when Q_IL > Q_RL."""
        import torch
        import numpy as np
        from unittest.mock import Mock

        obj, DualPolicyClass = self._make_dual_policy_instance()

        batch = 2
        a_rl = torch.zeros(batch, 4)          # known RL action (zeros)
        a_il = torch.ones(batch, 4) * 0.5     # known IL action (0.5s)

        obj.num_timesteps = 100
        obj._last_obs = np.zeros((batch, 34), dtype=np.float32)

        actor_mock = Mock()
        actor_mock._predict = Mock(return_value=a_rl)
        obj.actor = actor_mock

        il_actor_mock = Mock()
        il_actor_mock._predict = Mock(return_value=a_il)
        obj.il_actor = il_actor_mock

        obj.il_noise_std = 0.0  # no noise to ensure determinism

        policy_mock = Mock()
        policy_mock.unscale_action = lambda x: x
        obj.policy = policy_mock

        # Patch _ibrl_min_q: IL Q=5 (higher), RL Q=1
        call_count = [0]
        def mock_min_q(obs, action):
            val = 1.0 if call_count[0] == 0 else 5.0
            call_count[0] += 1
            return torch.full((batch,), val)

        obj._ibrl_min_q = mock_min_q

        action_np, buffer_action_np = DualPolicyClass._sample_action(
            obj, learning_starts=0
        )

        # Should select IL action (0.5s)
        np.testing.assert_allclose(buffer_action_np, a_il.numpy(), atol=1e-5)

    def test_greedy_selects_rl_when_q_rl_higher(self):
        """Greedy selection returns RL action when Q_RL > Q_IL."""
        import torch
        import numpy as np
        from unittest.mock import Mock

        obj, DualPolicyClass = self._make_dual_policy_instance()

        batch = 2
        a_rl = torch.zeros(batch, 4)
        a_il = torch.ones(batch, 4) * 0.5

        obj.num_timesteps = 100
        obj._last_obs = np.zeros((batch, 34), dtype=np.float32)

        actor_mock = Mock()
        actor_mock._predict = Mock(return_value=a_rl)
        obj.actor = actor_mock

        il_actor_mock = Mock()
        il_actor_mock._predict = Mock(return_value=a_il)
        obj.il_actor = il_actor_mock

        obj.il_noise_std = 0.0

        policy_mock = Mock()
        policy_mock.unscale_action = lambda x: x
        obj.policy = policy_mock

        # RL Q=5 (higher), IL Q=1
        call_count = [0]
        def mock_min_q(obs, action):
            val = 5.0 if call_count[0] == 0 else 1.0
            call_count[0] += 1
            return torch.full((batch,), val)

        obj._ibrl_min_q = mock_min_q

        action_np, buffer_action_np = DualPolicyClass._sample_action(
            obj, learning_starts=0
        )

        # Should select RL action (zeros)
        np.testing.assert_allclose(buffer_action_np, a_rl.numpy(), atol=1e-5)

    def test_soft_selection_uses_il_with_high_beta(self):
        """Soft (Boltzmann) selection with high beta strongly prefers IL when Q_IL >> Q_RL."""
        import torch
        import numpy as np
        from unittest.mock import Mock

        obj, DualPolicyClass = self._make_dual_policy_instance()

        obj.il_soft = True
        obj.il_beta = 1000.0   # very high temp → effectively deterministic toward best
        obj.il_noise_std = 0.0

        batch = 1
        a_rl = torch.zeros(batch, 4)
        a_il = torch.ones(batch, 4) * 0.5

        obj.num_timesteps = 100
        obj._last_obs = np.zeros((batch, 34), dtype=np.float32)

        actor_mock = Mock()
        actor_mock._predict = Mock(return_value=a_rl)
        obj.actor = actor_mock

        il_actor_mock = Mock()
        il_actor_mock._predict = Mock(return_value=a_il)
        obj.il_actor = il_actor_mock

        policy_mock = Mock()
        policy_mock.unscale_action = lambda x: x
        obj.policy = policy_mock

        # IL Q=5, RL Q=1 → with beta=1000, softmax heavily favours IL
        call_count = [0]
        def mock_min_q(obs, action):
            val = 1.0 if call_count[0] == 0 else 5.0
            call_count[0] += 1
            return torch.full((batch,), val)

        obj._ibrl_min_q = mock_min_q

        # Run multiple times to verify IL wins consistently
        wins_il = 0
        runs = 20
        for _ in range(runs):
            call_count[0] = 0
            _, buffer_action_np = DualPolicyClass._sample_action(
                obj, learning_starts=0
            )
            if np.allclose(buffer_action_np, a_il.numpy(), atol=0.1):
                wins_il += 1

        # With beta=1000 and Q_IL >> Q_RL, IL should win nearly every time
        assert wins_il >= 18, f"Expected IL to win >= 18/20 runs, got {wins_il}/20"


# ============================================================================
# TEST SUITE: FrameStackWrapper
# ============================================================================

class TestFrameStackWrapper:
    def _make_mock_env(self, obs_dim=34, act_dim=2):
        """Create a minimal gymnasium.Env subclass for testing."""
        import gymnasium as gym

        class _MinimalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
                self._step_count = 0

            def reset(self, **kwargs):
                self._step_count = 0
                return np.ones(obs_dim, dtype=np.float32) * 0.5, {}

            def step(self, action):
                self._step_count += 1
                obs = np.ones(obs_dim, dtype=np.float32) * self._step_count
                return obs, 1.0, False, False, {}

        return _MinimalEnv()

    def test_obs_space_shape(self):
        """Observation space is (n_frames * obs_dim,)."""
        inner = self._make_mock_env()
        wrapper = FrameStackWrapper(inner, n_frames=4)
        assert wrapper.observation_space.shape == (4 * 34,)

    def test_action_space_unchanged(self):
        """Action space is unchanged by wrapper."""
        inner = self._make_mock_env()
        wrapper = FrameStackWrapper(inner, n_frames=4)
        assert wrapper.action_space.shape == (2,)

    def test_reset_fills_all_frames(self):
        """After reset(), all frames contain the initial observation."""
        inner = self._make_mock_env()
        wrapper = FrameStackWrapper(inner, n_frames=3)
        obs, _ = wrapper.reset()

        assert obs.shape == (3 * 34,)
        # All three frames should be identical (the reset obs = 0.5 everywhere)
        frame0 = obs[:34]
        frame1 = obs[34:68]
        frame2 = obs[68:102]
        np.testing.assert_array_equal(frame0, frame1)
        np.testing.assert_array_equal(frame1, frame2)
        assert frame0[0] == 0.5

    def test_step_shifts_frames(self):
        """After step(), newest obs is in the last frame slot."""
        inner = self._make_mock_env()
        wrapper = FrameStackWrapper(inner, n_frames=3)
        obs, _ = wrapper.reset()

        # Step 1: inner returns obs filled with 1.0
        obs1, _, _, _, _ = wrapper.step(np.zeros(2))
        # Frames should be: [0.5, 0.5, 1.0]
        assert obs1[2 * 34] == 1.0  # Last frame, first element
        assert obs1[0] == 0.5       # First frame, first element

        # Step 2: inner returns obs filled with 2.0
        obs2, _, _, _, _ = wrapper.step(np.zeros(2))
        # Frames should be: [0.5, 1.0, 2.0]
        assert obs2[0] == 0.5       # Oldest frame
        assert obs2[34] == 1.0      # Middle frame
        assert obs2[2 * 34] == 2.0  # Newest frame

    def test_episode_boundary_no_leakage(self):
        """After reset(), previous episode's frames are cleared."""
        inner = self._make_mock_env()
        wrapper = FrameStackWrapper(inner, n_frames=3)
        wrapper.reset()

        # Take a few steps
        wrapper.step(np.zeros(2))
        wrapper.step(np.zeros(2))

        # Reset - all frames should be the new initial obs, not old steps
        obs, _ = wrapper.reset()
        frame0 = obs[:34]
        frame1 = obs[34:68]
        frame2 = obs[68:102]
        np.testing.assert_array_equal(frame0, frame1)
        np.testing.assert_array_equal(frame1, frame2)


# ============================================================================
# TEST SUITE: TemporalCVAEFeatureExtractor
# ============================================================================

class TestTemporalCVAEFeatureExtractor:
    def _make_extractor(self, n_frames=4, gru_hidden=128, z_dim=8):
        import torch
        import gymnasium as gym
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        obs_dim = n_frames * 34
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        cls = TemporalCVAEFeatureExtractor.create(
            BaseFeaturesExtractor, n_frames=n_frames,
            gru_hidden_dim=gru_hidden, z_dim=z_dim)
        features_dim = gru_hidden + z_dim
        return cls(obs_space, features_dim=features_dim)

    def test_output_shape(self):
        """Feature extractor produces (batch, gru_hidden + z_dim)."""
        import torch
        ext = self._make_extractor(n_frames=4, gru_hidden=128, z_dim=8)
        obs = torch.randn(8, 4 * 34)
        out = ext(obs)
        assert out.shape == (8, 136)

    def test_z_slot_is_zero(self):
        """Last z_dim entries are zeros during forward()."""
        import torch
        z_dim = 8
        ext = self._make_extractor(n_frames=4, gru_hidden=128, z_dim=z_dim)
        obs = torch.randn(4, 4 * 34)
        out = ext(obs)
        z_slot = out[:, -z_dim:]
        torch.testing.assert_close(z_slot, torch.zeros(4, z_dim))

    def test_encode_obs_shape(self):
        """encode_obs returns (batch, gru_hidden_dim)."""
        import torch
        ext = self._make_extractor(n_frames=4, gru_hidden=128, z_dim=8)
        obs = torch.randn(4, 4 * 34)
        obs_features = ext.encode_obs(obs)
        assert obs_features.shape == (4, 128)

    def test_obs_feature_dim_attribute(self):
        """_obs_feature_dim matches gru_hidden_dim."""
        ext = self._make_extractor(n_frames=4, gru_hidden=64, z_dim=8)
        assert ext._obs_feature_dim == 64

    def test_gradient_flow_through_gru(self):
        """Gradients flow through state_mlp, lidar_mlp, and GRU."""
        import torch
        ext = self._make_extractor(n_frames=4, gru_hidden=128, z_dim=8)
        ext.train()
        obs = torch.randn(4, 4 * 34)
        out = ext(obs)
        loss = out.sum()
        loss.backward()
        # Check state MLP gradients
        assert ext.state_mlp[0].weight.grad is not None
        # Check lidar MLP gradients
        assert ext.lidar_mlp[0].weight.grad is not None
        # Check GRU gradients
        assert ext.gru.weight_ih_l0.grad is not None

    def test_temporal_sensitivity(self):
        """Changing one frame changes the output (GRU is not ignoring input)."""
        import torch
        ext = self._make_extractor(n_frames=4, gru_hidden=128, z_dim=8)
        ext.eval()
        obs1 = torch.randn(1, 4 * 34)
        obs2 = obs1.clone()
        # Change only the last frame
        obs2[0, 3 * 34:] = torch.randn(34)
        with torch.no_grad():
            out1 = ext(obs1)
            out2 = ext(obs2)
        # Outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-6)


# ============================================================================
# TEST SUITE: build_frame_stacks
# ============================================================================

class TestBuildFrameStacks:
    def test_output_shape(self):
        """Output shape is (N, n_frames * obs_dim)."""
        total = 20
        obs = np.random.randn(total, 34).astype(np.float32)
        ep_lengths = np.array([10, 10])
        result = build_frame_stacks(obs, ep_lengths, n_frames=4)
        assert result.shape == (20, 4 * 34)

    def test_first_step_repeats(self):
        """First step of each episode repeats the initial obs for all frames."""
        total = 10
        obs = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        ep_lengths = np.array([5, 5])
        result = build_frame_stacks(obs, ep_lengths, n_frames=3)

        # First step of ep1 (index 0): all 3 frames should be obs[0]
        for f in range(3):
            np.testing.assert_array_equal(result[0, f * 34:(f + 1) * 34], obs[0])

        # First step of ep2 (index 5): all 3 frames should be obs[5]
        for f in range(3):
            np.testing.assert_array_equal(result[5, f * 34:(f + 1) * 34], obs[5])

    def test_later_step_correct_history(self):
        """Step t=3 with n_frames=3 should have frames [obs[1], obs[2], obs[3]]."""
        total = 10
        obs = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        ep_lengths = np.array([10])
        result = build_frame_stacks(obs, ep_lengths, n_frames=3)

        # Step t=3: frames should be obs[1], obs[2], obs[3]
        np.testing.assert_array_equal(result[3, :34], obs[1])
        np.testing.assert_array_equal(result[3, 34:68], obs[2])
        np.testing.assert_array_equal(result[3, 68:102], obs[3])

    def test_episode_boundary_no_cross(self):
        """Frame stacking does not cross episode boundaries."""
        total = 10
        obs = np.arange(total * 34, dtype=np.float32).reshape(total, 34)
        ep_lengths = np.array([5, 5])
        result = build_frame_stacks(obs, ep_lengths, n_frames=3)

        # Step t=1 of ep2 (index 6): n_frames=3 means we need t-2, t-1, t
        # t-2 = -1 < 0, so repeat obs[5]; t-1 = 0 → obs[5]; t = 1 → obs[6]
        np.testing.assert_array_equal(result[6, :34], obs[5])
        np.testing.assert_array_equal(result[6, 34:68], obs[5])
        np.testing.assert_array_equal(result[6, 68:102], obs[6])

    def test_n_frames_1_identity(self):
        """n_frames=1 returns the original observations unchanged."""
        total = 10
        obs = np.random.randn(total, 34).astype(np.float32)
        ep_lengths = np.array([5, 5])
        result = build_frame_stacks(obs, ep_lengths, n_frames=1)
        np.testing.assert_array_equal(result, obs)


# ============================================================================
# TEST SUITE: FrameStackWrapper + ChunkedEnvWrapper integration
# ============================================================================

class TestFrameStackChunkedIntegration:
    def _make_mock_env(self, obs_dim=34, act_dim=2):
        import gymnasium as gym

        class _MinimalEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

            def reset(self, **kwargs):
                return np.zeros(obs_dim, dtype=np.float32), {}

            def step(self, action):
                return np.ones(obs_dim, dtype=np.float32), 1.0, False, False, {}

        return _MinimalEnv()

    def test_combined_spaces(self):
        """FrameStack(n=4) + Chunked(k=5) → obs=(136,), action=(10,)."""
        inner = self._make_mock_env()
        stacked = FrameStackWrapper(inner, n_frames=4)
        chunked = ChunkedEnvWrapper(stacked, chunk_size=5, gamma=0.99)

        assert chunked.observation_space.shape == (4 * 34,)
        assert chunked.action_space.shape == (5 * 2,)

    def test_combined_step(self):
        """Combined wrapper executes correctly."""
        inner = self._make_mock_env()
        stacked = FrameStackWrapper(inner, n_frames=2)
        chunked = ChunkedEnvWrapper(stacked, chunk_size=3, gamma=0.99)

        obs, _ = chunked.reset()
        assert obs.shape == (2 * 34,)

        action = np.zeros(3 * 2, dtype=np.float32)
        obs, reward, terminated, truncated, info = chunked.step(action)
        assert obs.shape == (2 * 34,)
        # Reward should be accumulated over 3 sub-steps
        expected_reward = 1.0 + 0.99 * 1.0 + 0.99**2 * 1.0
        assert abs(reward - expected_reward) < 1e-6
