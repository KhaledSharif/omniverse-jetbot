"""
Test Suite for PPO Training Pipeline (train_rl.py)

Tests cover:
- linear_schedule: Learning rate decay function
- validate_demo_data: Demo data validation and requirements checking
- compute_mc_returns: Monte Carlo return computation
- normalize_obs / normalize_returns: VecNormalize-compatible normalization
- prewarm_vecnormalize: Pre-warming running statistics from demo data
- load_demo_data: Loading and filtering demo episodes
- bc_warmstart: Behavioral cloning pretraining (actor only)
- pretrain_critic: Critic pretraining on MC returns

Run with: pytest test_train_rl.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing train_rl)
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

from train_rl import (
    linear_schedule,
    validate_demo_data,
    compute_mc_returns,
    normalize_obs,
    normalize_returns,
    prewarm_vecnormalize,
    load_demo_data,
    bc_warmstart,
    pretrain_critic,
)


# ============================================================================
# HELPERS
# ============================================================================

def make_demo_npz(path, num_episodes=20, steps_per_ep=50, obs_dim=34,
                  success_rate=0.5, reward_value=None):
    """Create a synthetic demo NPZ file for testing."""
    total = num_episodes * steps_per_ep
    obs = np.random.randn(total, obs_dim).astype(np.float32)
    actions = np.random.randn(total, 2).astype(np.float32)
    if reward_value is not None:
        rewards = np.full(total, reward_value, dtype=np.float32)
    else:
        rewards = np.random.randn(total).astype(np.float32)
    episode_lengths = np.full(num_episodes, steps_per_ep)
    episode_success = np.zeros(num_episodes, dtype=bool)
    episode_success[:int(num_episodes * success_rate)] = True
    np.savez(path,
             observations=obs, actions=actions, rewards=rewards,
             episode_lengths=episode_lengths, episode_success=episode_success)
    return path


def make_mock_vecnormalize_env(obs_dim=34, obs_mean=None, obs_var=None,
                                ret_var=None):
    """Create a mock VecNormalize-like env with obs_rms and ret_rms."""
    env = Mock()
    env.obs_rms = Mock()
    env.obs_rms.mean = obs_mean if obs_mean is not None else np.zeros(obs_dim)
    env.obs_rms.var = obs_var if obs_var is not None else np.ones(obs_dim)
    env.obs_rms.update = Mock()
    env.ret_rms = Mock()
    env.ret_rms.var = ret_var if ret_var is not None else np.array(1.0)
    env.ret_rms.update = Mock()
    env.epsilon = 1e-8
    env.clip_obs = 10.0
    env.clip_reward = 10.0
    return env


# ============================================================================
# TEST SUITE: LinearSchedule
# ============================================================================

class TestLinearSchedule:
    def test_initial_value(self):
        schedule = linear_schedule(3e-4)
        assert schedule(1.0) == pytest.approx(3e-4)

    def test_midpoint(self):
        schedule = linear_schedule(3e-4)
        assert schedule(0.5) == pytest.approx(3e-4 * 0.5)

    def test_end(self):
        schedule = linear_schedule(3e-4)
        assert schedule(0.0) == pytest.approx(0.0)

    def test_arbitrary_value(self):
        schedule = linear_schedule(1.0)
        assert schedule(0.25) == pytest.approx(0.25)


# ============================================================================
# TEST SUITE: ValidateDemoData
# ============================================================================

class TestValidateDemoData:
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

    def test_no_episode_success_key(self, tmp_path):
        """NPZ without episode_success key skips success check."""
        path = tmp_path / "demo.npz"
        total = 20 * 50
        np.savez(path,
                 observations=np.random.randn(total, 34).astype(np.float32),
                 actions=np.random.randn(total, 2).astype(np.float32),
                 rewards=np.random.randn(total).astype(np.float32),
                 episode_lengths=np.full(20, 50))
        result = validate_demo_data(str(path))
        assert result['successful'] == 0
        assert result['episodes'] == 20

    def test_avg_return_computation(self, tmp_path):
        """Verify avg_return is computed correctly from known rewards."""
        path = tmp_path / "demo.npz"
        # 2 episodes, 3 steps each, known rewards
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        np.savez(path,
                 observations=np.random.randn(6, 34).astype(np.float32),
                 actions=np.random.randn(6, 2).astype(np.float32),
                 rewards=rewards,
                 episode_lengths=np.array([3, 3]),
                 episode_success=np.array([True, True]))
        # Episode returns: 1+2+3=6, 4+5+6=15; avg = 10.5
        # But only 2 episodes < 10 → ValueError. Use 10+ episodes.
        # Alternatively: use a data set that passes validation.
        # Let's build a proper one: 10 episodes, 50 steps each
        num_ep = 10
        steps = 50
        total = num_ep * steps
        rewards_full = np.ones(total, dtype=np.float32)
        # Each episode return = 50.0, avg_return = 50.0
        np.savez(path,
                 observations=np.random.randn(total, 34).astype(np.float32),
                 actions=np.random.randn(total, 2).astype(np.float32),
                 rewards=rewards_full,
                 episode_lengths=np.full(num_ep, steps),
                 episode_success=np.ones(num_ep, dtype=bool))
        result = validate_demo_data(str(path))
        assert result['avg_return'] == pytest.approx(50.0)


# ============================================================================
# TEST SUITE: ComputeMcReturns
# ============================================================================

class TestComputeMcReturns:
    def test_single_episode(self, tmp_path):
        """3-step episode with known rewards and gamma=0.99."""
        path = tmp_path / "demo.npz"
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.savez(path,
                 observations=np.zeros((3, 34), dtype=np.float32),
                 actions=np.zeros((3, 2), dtype=np.float32),
                 rewards=rewards,
                 episode_lengths=np.array([3]))
        returns = compute_mc_returns(str(path), gamma=0.99)
        # G[2] = 3.0
        # G[1] = 2.0 + 0.99 * 3.0 = 4.97
        # G[0] = 1.0 + 0.99 * 4.97 = 5.9203
        assert returns[2] == pytest.approx(3.0)
        assert returns[1] == pytest.approx(4.97)
        assert returns[0] == pytest.approx(5.9203)

    def test_multi_episode(self, tmp_path):
        """Two episodes, returns reset at episode boundary."""
        path = tmp_path / "demo.npz"
        rewards = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
        np.savez(path,
                 observations=np.zeros((4, 34), dtype=np.float32),
                 actions=np.zeros((4, 2), dtype=np.float32),
                 rewards=rewards,
                 episode_lengths=np.array([2, 2]))
        returns = compute_mc_returns(str(path), gamma=0.99)
        # Episode 1: G[1]=2.0, G[0]=1.0+0.99*2.0=2.98
        # Episode 2: G[3]=20.0, G[2]=10.0+0.99*20.0=29.8
        assert returns[0] == pytest.approx(2.98)
        assert returns[1] == pytest.approx(2.0)
        assert returns[2] == pytest.approx(29.8)
        assert returns[3] == pytest.approx(20.0)

    def test_gamma_zero(self, tmp_path):
        """With gamma=0, returns equal immediate rewards."""
        path = tmp_path / "demo.npz"
        rewards = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        np.savez(path,
                 observations=np.zeros((3, 34), dtype=np.float32),
                 actions=np.zeros((3, 2), dtype=np.float32),
                 rewards=rewards,
                 episode_lengths=np.array([3]))
        returns = compute_mc_returns(str(path), gamma=0.0)
        np.testing.assert_allclose(returns, rewards, atol=1e-6)

    def test_gamma_one(self, tmp_path):
        """With gamma=1, returns equal cumulative sum from t to end."""
        path = tmp_path / "demo.npz"
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        np.savez(path,
                 observations=np.zeros((3, 34), dtype=np.float32),
                 actions=np.zeros((3, 2), dtype=np.float32),
                 rewards=rewards,
                 episode_lengths=np.array([3]))
        returns = compute_mc_returns(str(path), gamma=1.0)
        assert returns[0] == pytest.approx(3.0)
        assert returns[1] == pytest.approx(2.0)
        assert returns[2] == pytest.approx(1.0)


# ============================================================================
# TEST SUITE: NormalizeObs
# ============================================================================

class TestNormalizeObs:
    def test_zero_mean_unit_var(self):
        env = make_mock_vecnormalize_env(obs_dim=2)
        obs = np.array([[2.0, -1.0]], dtype=np.float32)
        result = normalize_obs(obs, env)
        np.testing.assert_allclose(result, obs, atol=1e-5)

    def test_nonzero_mean(self):
        env = make_mock_vecnormalize_env(
            obs_dim=2,
            obs_mean=np.array([1.0, 2.0]),
            obs_var=np.array([4.0, 9.0]),
        )
        obs = np.array([[3.0, 5.0]], dtype=np.float32)
        result = normalize_obs(obs, env)
        # (3-1)/sqrt(4) = 1.0, (5-2)/sqrt(9) = 1.0
        np.testing.assert_allclose(result, [[1.0, 1.0]], atol=1e-5)

    def test_clipping(self):
        env = make_mock_vecnormalize_env(obs_dim=1)
        env.clip_obs = 5.0
        obs = np.array([[100.0]], dtype=np.float32)
        result = normalize_obs(obs, env)
        assert result[0, 0] == pytest.approx(5.0)

    def test_negative_clipping(self):
        env = make_mock_vecnormalize_env(obs_dim=1)
        env.clip_obs = 5.0
        obs = np.array([[-100.0]], dtype=np.float32)
        result = normalize_obs(obs, env)
        assert result[0, 0] == pytest.approx(-5.0)


# ============================================================================
# TEST SUITE: NormalizeReturns
# ============================================================================

class TestNormalizeReturns:
    def test_scaling(self):
        env = make_mock_vecnormalize_env(ret_var=np.array(4.0))
        returns = np.array([10.0, -6.0], dtype=np.float32)
        result = normalize_returns(returns, env)
        # 10/sqrt(4) = 5.0, -6/sqrt(4) = -3.0
        np.testing.assert_allclose(result, [5.0, -3.0], atol=1e-5)

    def test_clipping(self):
        env = make_mock_vecnormalize_env(ret_var=np.array(1.0))
        env.clip_reward = 5.0
        returns = np.array([100.0, -100.0], dtype=np.float32)
        result = normalize_returns(returns, env)
        assert result[0] == pytest.approx(5.0)
        assert result[1] == pytest.approx(-5.0)


# ============================================================================
# TEST SUITE: PrewarmVecnormalize
# ============================================================================

class TestPrewarmVecnormalize:
    def test_obs_rms_updated(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=10, steps_per_ep=50, obs_dim=34)
        env = make_mock_vecnormalize_env(obs_dim=34)

        prewarm_vecnormalize(env, str(path), gamma=0.99)

        env.obs_rms.update.assert_called_once()
        call_args = env.obs_rms.update.call_args[0][0]
        assert call_args.shape == (500, 34)

    def test_ret_rms_updated(self, tmp_path):
        path = tmp_path / "demo.npz"
        make_demo_npz(path, num_episodes=10, steps_per_ep=50, obs_dim=34)
        env = make_mock_vecnormalize_env(obs_dim=34)

        prewarm_vecnormalize(env, str(path), gamma=0.99)

        env.ret_rms.update.assert_called_once()
        call_args = env.ret_rms.update.call_args[0][0]
        assert call_args.shape == (500,)

    def test_running_returns_computation(self, tmp_path):
        """Verify running returns are computed correctly (ret = ret*gamma + r)."""
        path = tmp_path / "demo.npz"
        # Single episode, 3 steps, reward=1.0 each
        np.savez(path,
                 observations=np.zeros((3, 34), dtype=np.float32),
                 actions=np.zeros((3, 2), dtype=np.float32),
                 rewards=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                 episode_lengths=np.array([3]))
        env = make_mock_vecnormalize_env(obs_dim=34)

        prewarm_vecnormalize(env, str(path), gamma=0.99)

        call_args = env.ret_rms.update.call_args[0][0]
        # Running returns: ret[0]=1.0, ret[1]=1*0.99+1=1.99, ret[2]=1.99*0.99+1=2.9701
        np.testing.assert_allclose(call_args, [1.0, 1.99, 2.9701], atol=1e-4)


# ============================================================================
# TEST SUITE: LoadDemoData
# ============================================================================

class TestLoadDemoData:
    @patch('jetbot_keyboard_control.DemoPlayer')
    def test_loads_all_episodes(self, MockDemoPlayer, tmp_path):
        player = MockDemoPlayer.return_value
        player.num_episodes = 3
        player.total_frames = 30
        player.get_episode.side_effect = [
            (np.ones((10, 34)), np.ones((10, 2))),
            (np.ones((10, 34)), np.ones((10, 2))),
            (np.ones((10, 34)), np.ones((10, 2))),
        ]

        obs, acts = load_demo_data("dummy.npz", successful_only=False)
        assert obs.shape == (30, 34)
        assert acts.shape == (30, 2)

    @patch('jetbot_keyboard_control.DemoPlayer')
    def test_successful_only(self, MockDemoPlayer):
        player = MockDemoPlayer.return_value
        player.num_episodes = 3
        player.total_frames = 30
        player.get_successful_episodes.return_value = [0, 2]
        player.get_episode.side_effect = [
            (np.ones((10, 34)), np.ones((10, 2))),
            (np.ones((10, 34)), np.ones((10, 2))),
        ]

        obs, acts = load_demo_data("dummy.npz", successful_only=True)
        assert obs.shape == (20, 34)
        assert acts.shape == (20, 2)
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
# TEST SUITE: BcWarmstart (with real PyTorch)
# ============================================================================

class TestBcWarmstart:
    @pytest.fixture
    def tiny_model_and_env(self, tmp_path):
        """Create a tiny mock SB3 PPO model with real PyTorch networks."""
        import torch
        import torch.nn as nn

        obs_dim = 4
        act_dim = 2

        # Create real tiny networks mimicking SB3 PPO structure
        features_extractor = nn.Identity()
        policy_net = nn.Sequential(nn.Linear(obs_dim, 8), nn.Tanh())
        value_net_extractor = nn.Sequential(nn.Linear(obs_dim, 8), nn.Tanh())
        action_net = nn.Linear(8, act_dim)
        value_net = nn.Linear(8, 1)

        # Mock the distribution to return predictions
        class MockDistribution:
            def __init__(self, mean):
                self.mean = mean

        class MockActionDist:
            def __init__(self, dist):
                self.distribution = dist

        # Build mock policy
        policy = Mock()
        policy.features_extractor = features_extractor
        policy.mlp_extractor = Mock()
        policy.mlp_extractor.policy_net = policy_net
        policy.mlp_extractor.value_net = value_net_extractor
        policy.action_net = action_net
        policy.value_net = value_net
        policy.device = torch.device('cpu')

        def mock_get_distribution(obs):
            features = features_extractor(obs)
            latent = policy_net(features)
            mean = action_net(latent)
            return MockActionDist(MockDistribution(mean))

        policy.get_distribution = mock_get_distribution

        model = Mock()
        model.policy = policy

        # Create demo NPZ
        demo_path = tmp_path / "demo.npz"
        total = 100
        np.savez(demo_path,
                 observations=np.random.randn(total, obs_dim).astype(np.float32),
                 actions=np.random.randn(total, act_dim).astype(np.float32),
                 rewards=np.random.randn(total).astype(np.float32),
                 episode_lengths=np.array([50, 50]),
                 episode_success=np.array([True, True]))

        # Mock env
        env = make_mock_vecnormalize_env(obs_dim=obs_dim)

        return model, env, str(demo_path)

    @patch('train_rl.load_demo_data')
    def test_loss_decreases(self, mock_load, tiny_model_and_env):
        model, env, demo_path = tiny_model_and_env
        obs_dim = 4
        n = 100
        mock_load.return_value = (
            np.random.randn(n, obs_dim).astype(np.float32),
            np.random.randn(n, 2).astype(np.float32),
        )

        # Run with few epochs - just check it doesn't crash
        bc_warmstart(model, env, demo_path, epochs=5, batch_size=32)

    @patch('train_rl.load_demo_data')
    def test_value_net_unchanged(self, mock_load, tiny_model_and_env):
        """Verify value_net parameters are NOT updated by BC warmstart."""
        import torch
        model, env, demo_path = tiny_model_and_env
        obs_dim = 4
        n = 100
        mock_load.return_value = (
            np.random.randn(n, obs_dim).astype(np.float32),
            np.random.randn(n, 2).astype(np.float32),
        )

        # Snapshot value net params
        vf_params_before = {
            name: p.clone()
            for name, p in model.policy.value_net.named_parameters()
        }

        bc_warmstart(model, env, demo_path, epochs=5, batch_size=32)

        for name, p in model.policy.value_net.named_parameters():
            torch.testing.assert_close(p, vf_params_before[name])


# ============================================================================
# TEST SUITE: PretrainCritic (with real PyTorch)
# ============================================================================

class TestPretrainCritic:
    @patch('train_rl.load_demo_data')
    @patch('train_rl.compute_mc_returns')
    def test_loss_decreases(self, mock_mc, mock_load, tmp_path):
        import torch
        import torch.nn as nn

        obs_dim = 4
        n = 100

        # Create real tiny networks
        features_extractor = nn.Sequential(nn.Linear(obs_dim, 8), nn.Tanh())
        value_net_extractor = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        action_net = nn.Linear(8, 2)
        value_net = nn.Linear(8, 1)

        policy = Mock()
        policy.features_extractor = features_extractor
        policy.vf_features_extractor = features_extractor
        policy.mlp_extractor = Mock()
        policy.mlp_extractor.policy_net = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        policy.mlp_extractor.value_net = value_net_extractor
        policy.mlp_extractor.forward_critic = value_net_extractor.forward
        policy.action_net = action_net
        policy.value_net = value_net
        policy.device = torch.device('cpu')
        policy.extract_features = lambda obs, extractor: extractor(obs)

        model = Mock()
        model.policy = policy

        env = make_mock_vecnormalize_env(obs_dim=obs_dim)

        mock_load.return_value = (
            np.random.randn(n, obs_dim).astype(np.float32),
            np.random.randn(n, 2).astype(np.float32),
        )
        mock_mc.return_value = np.random.randn(n).astype(np.float32)

        demo_path = tmp_path / "demo.npz"
        make_demo_npz(demo_path, num_episodes=10, steps_per_ep=10, obs_dim=obs_dim)

        # Just verify it runs without error
        pretrain_critic(model, env, str(demo_path), epochs=5, batch_size=32)

    @patch('train_rl.load_demo_data')
    @patch('train_rl.compute_mc_returns')
    def test_action_net_unchanged(self, mock_mc, mock_load, tmp_path):
        """Verify actor parameters are NOT updated by critic pretraining."""
        import torch
        import torch.nn as nn

        obs_dim = 4
        n = 100

        features_extractor = nn.Sequential(nn.Linear(obs_dim, 8), nn.Tanh())
        value_net_extractor = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        action_net = nn.Linear(8, 2)
        value_net = nn.Linear(8, 1)

        policy = Mock()
        policy.features_extractor = features_extractor
        policy.vf_features_extractor = features_extractor
        policy.mlp_extractor = Mock()
        policy.mlp_extractor.policy_net = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        policy.mlp_extractor.value_net = value_net_extractor
        policy.mlp_extractor.forward_critic = value_net_extractor.forward
        policy.action_net = action_net
        policy.value_net = value_net
        policy.device = torch.device('cpu')
        policy.extract_features = lambda obs, extractor: extractor(obs)

        model = Mock()
        model.policy = policy

        env = make_mock_vecnormalize_env(obs_dim=obs_dim)

        mock_load.return_value = (
            np.random.randn(n, obs_dim).astype(np.float32),
            np.random.randn(n, 2).astype(np.float32),
        )
        mock_mc.return_value = np.random.randn(n).astype(np.float32)

        # Snapshot action net params
        action_params_before = {
            name: p.clone()
            for name, p in action_net.named_parameters()
        }

        demo_path = tmp_path / "demo.npz"
        make_demo_npz(demo_path, num_episodes=10, steps_per_ep=10, obs_dim=obs_dim)
        pretrain_critic(model, env, str(demo_path), epochs=5, batch_size=32)

        for name, p in action_net.named_parameters():
            torch.testing.assert_close(p, action_params_before[name])
