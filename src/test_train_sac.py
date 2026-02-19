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
    LidarMLPVAE,
    pretrain_lidar_vae,
    LidarVAEFeatureExtractor,
    bc_warmstart_sac,
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
# TEST SUITE: LidarMLPVAE
# ============================================================================

class TestLidarMLPVAE:
    def test_output_shapes(self):
        """Forward pass returns correct shapes for z, mu, logvar, recon."""
        import torch
        vae = LidarMLPVAE.create()
        x = torch.randn(8, 24)
        z, mu, logvar, recon = vae(x)
        assert z.shape == (8, 16)
        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)
        assert recon.shape == (8, 24)

    def test_encode_shapes(self):
        """Encode returns (mu, logvar) with correct shapes."""
        import torch
        vae = LidarMLPVAE.create()
        x = torch.randn(4, 24)
        mu, logvar = vae.encode(x)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_decode_shapes(self):
        """Decode returns reconstruction with correct shape."""
        import torch
        vae = LidarMLPVAE.create()
        z = torch.randn(4, 16)
        recon = vae.decode(z)
        assert recon.shape == (4, 24)

    def test_deterministic_eval(self):
        """In eval mode, reparameterize returns mu (deterministic)."""
        import torch
        vae = LidarMLPVAE.create()
        vae.eval()
        x = torch.randn(4, 24)
        z1, mu1, _, _ = vae(x)
        z2, mu2, _, _ = vae(x)
        torch.testing.assert_close(z1, z2)
        torch.testing.assert_close(z1, mu1)

    def test_stochastic_train(self):
        """In train mode, reparameterize is stochastic (z != mu with high probability)."""
        import torch
        vae = LidarMLPVAE.create()
        vae.train()
        x = torch.randn(32, 24)
        z, mu, _, _ = vae(x)
        # With 32 samples and 16 latent dims, z should differ from mu
        assert not torch.allclose(z, mu, atol=1e-6)

    def test_custom_dims(self):
        """Custom input_dim, hidden_dim, latent_dim work correctly."""
        import torch
        vae = LidarMLPVAE.create(input_dim=12, hidden_dim=64, latent_dim=8)
        x = torch.randn(4, 12)
        z, mu, logvar, recon = vae(x)
        assert z.shape == (4, 8)
        assert recon.shape == (4, 12)


# ============================================================================
# TEST SUITE: PretrainLidarVAE
# ============================================================================

class TestPretrainLidarVAE:
    def test_returns_eval_mode_vae(self):
        """Pretrained VAE is returned in eval mode."""
        demo_obs = np.random.randn(200, 34).astype(np.float32)
        vae = pretrain_lidar_vae(demo_obs, epochs=5, batch_size=64)
        assert not vae.training

    def test_reconstruction_improves(self):
        """Reconstruction loss decreases over training."""
        import torch
        demo_obs = np.random.randn(500, 34).astype(np.float32)
        # Untrained VAE
        vae_untrained = LidarMLPVAE.create()
        vae_untrained.eval()
        lidar = symlog(torch.tensor(demo_obs[:, 10:34]))
        with torch.no_grad():
            _, _, _, recon_before = vae_untrained(lidar)
            loss_before = torch.nn.functional.mse_loss(recon_before, lidar).item()

        # Trained VAE
        vae_trained = pretrain_lidar_vae(demo_obs, vae=LidarMLPVAE.create(), epochs=50, batch_size=64)
        with torch.no_grad():
            _, _, _, recon_after = vae_trained(lidar)
            loss_after = torch.nn.functional.mse_loss(recon_after, lidar).item()

        assert loss_after < loss_before, \
            f"Reconstruction should improve: {loss_after:.4f} >= {loss_before:.4f}"

    def test_accepts_prebuilt_vae(self):
        """Can pass in a pre-built VAE to pretrain."""
        demo_obs = np.random.randn(200, 34).astype(np.float32)
        vae = LidarMLPVAE.create(latent_dim=8)
        result = pretrain_lidar_vae(demo_obs, vae=vae, epochs=5, batch_size=64)
        assert result is vae
        assert result.latent_dim == 8


# ============================================================================
# TEST SUITE: LidarVAEFeatureExtractor
# ============================================================================

class TestLidarVAEFeatureExtractor:
    def _make_extractor(self, pretrained_vae=None):
        import torch
        import gymnasium as gym
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        cls = LidarVAEFeatureExtractor.create(BaseFeaturesExtractor)
        return cls(obs_space, features_dim=48, pretrained_vae=pretrained_vae)

    def test_output_shape(self):
        """Feature extractor produces 48D output from 34D input."""
        import torch
        ext = self._make_extractor()
        obs = torch.randn(8, 34)
        out = ext(obs)
        assert out.shape == (8, 48)

    def test_features_dim(self):
        """features_dim attribute is 48."""
        ext = self._make_extractor()
        assert ext.features_dim == 48

    def test_pretrained_weights(self):
        """Pretrained VAE weights are used when provided."""
        import torch
        vae = LidarMLPVAE.create()
        # Set a known weight
        vae.mu_head.bias.data.fill_(42.0)
        ext = self._make_extractor(pretrained_vae=vae)
        torch.testing.assert_close(
            ext.vae.mu_head.bias.data,
            torch.full_like(ext.vae.mu_head.bias.data, 42.0)
        )

    def test_eval_determinism(self):
        """In eval mode, same input produces same output."""
        import torch
        ext = self._make_extractor()
        ext.eval()
        obs = torch.randn(4, 34)
        out1 = ext(obs)
        out2 = ext(obs)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self):
        """Gradients flow through the feature extractor."""
        import torch
        ext = self._make_extractor()
        ext.train()
        obs = torch.randn(4, 34, requires_grad=False)
        out = ext(obs)
        loss = out.sum()
        loss.backward()
        # Check that state MLP has gradients
        assert ext.state_mlp[0].weight.grad is not None
        # Check that VAE encoder has gradients
        assert ext.vae.mu_head.weight.grad is not None


# ============================================================================
# TEST SUITE: BCWarmstartWithFeatureExtractor
# ============================================================================

class TestBCWarmstartWithFeatureExtractor:
    def test_include_feature_extractor_param_exists(self):
        """bc_warmstart_sac accepts include_feature_extractor parameter."""
        import inspect
        sig = inspect.signature(bc_warmstart_sac)
        assert 'include_feature_extractor' in sig.parameters
        # Default should be False
        assert sig.parameters['include_feature_extractor'].default is False
