"""Tests for demo_io.py — HDF5 incremental writer and unified demo reader.

Run with: pytest test_demo_io.py -v
"""

import pytest
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# ============================================================================
# MOCK ISAAC SIM MODULES (before importing anything that touches isaacsim)
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

mock_isaacsim.SimulationApp = MagicMock()

class MockArticulationAction:
    def __init__(self, joint_velocities=None, **kwargs):
        self.joint_velocities = joint_velocities if joint_velocities is not None else np.array([])

mock_isaacsim_core_utils_types.ArticulationAction = MockArticulationAction
mock_isaacsim_core_utils_nucleus.get_assets_root_path = MagicMock(return_value="/Isaac")

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

h5py = pytest.importorskip("h5py")


# ============================================================================
# Helpers
# ============================================================================

def _make_npz(path, n_steps=20, n_episodes=2, obs_dim=10, action_dim=2,
              include_costs=False):
    """Create a minimal NPZ demo file for testing."""
    obs = np.random.randn(n_steps, obs_dim).astype(np.float32)
    actions = np.random.randn(n_steps, action_dim).astype(np.float32)
    rewards = np.random.randn(n_steps).astype(np.float32)
    dones = np.zeros(n_steps, dtype=bool)
    costs = np.random.rand(n_steps).astype(np.float32) if include_costs else np.zeros(n_steps, dtype=np.float32)

    ep_len = n_steps // n_episodes
    ep_starts = np.array([i * ep_len for i in range(n_episodes)], dtype=np.int64)
    ep_lengths = np.full(n_episodes, ep_len, dtype=np.int64)
    ep_returns = np.random.randn(n_episodes).astype(np.float32)
    ep_success = np.array([True, False] * (n_episodes // 2 + 1))[:n_episodes]

    # Mark last step of each episode as done
    for i in range(n_episodes):
        dones[int(ep_starts[i]) + int(ep_lengths[i]) - 1] = True

    metadata = {'obs_dim': obs_dim, 'action_dim': action_dim,
                'num_episodes': n_episodes, 'total_frames': n_steps}
    if include_costs:
        metadata['has_cost'] = True

    np.savez_compressed(
        path,
        observations=obs, actions=actions, rewards=rewards,
        dones=dones, costs=costs,
        episode_starts=ep_starts, episode_lengths=ep_lengths,
        episode_returns=ep_returns, episode_success=ep_success,
        metadata=np.array(metadata, dtype=object),
    )
    return dict(obs=obs, actions=actions, rewards=rewards, dones=dones,
                costs=costs, ep_starts=ep_starts, ep_lengths=ep_lengths,
                ep_returns=ep_returns, ep_success=ep_success, metadata=metadata)


# ============================================================================
# Tests: open_demo with NPZ
# ============================================================================

class TestOpenDemoNpz:
    def test_open_demo_npz_reads_arrays(self, tmp_path):
        from demo_io import open_demo
        npz_path = str(tmp_path / "test.npz")
        ref = _make_npz(npz_path)

        data = open_demo(npz_path)
        np.testing.assert_array_equal(data['observations'], ref['obs'])
        np.testing.assert_array_equal(data['actions'], ref['actions'])
        assert 'observations' in data
        data.close()

    def test_open_demo_npz_metadata_item(self, tmp_path):
        from demo_io import open_demo
        npz_path = str(tmp_path / "test.npz")
        ref = _make_npz(npz_path)

        data = open_demo(npz_path)
        meta = data['metadata'].item()
        assert meta['obs_dim'] == 10
        assert meta['action_dim'] == 2
        data.close()

    def test_open_demo_npz_context_manager(self, tmp_path):
        from demo_io import open_demo
        npz_path = str(tmp_path / "test.npz")
        _make_npz(npz_path)

        with open_demo(npz_path) as data:
            assert len(data['observations']) == 20

    def test_open_demo_unsupported_extension(self, tmp_path):
        from demo_io import open_demo
        with pytest.raises(ValueError, match="Unsupported"):
            open_demo(str(tmp_path / "test.csv"))


# ============================================================================
# Tests: open_demo with HDF5
# ============================================================================

class TestOpenDemoHdf5:
    def test_open_demo_hdf5_reads_arrays(self, tmp_path):
        from demo_io import open_demo, convert_npz_to_hdf5
        npz_path = str(tmp_path / "test.npz")
        hdf5_path = str(tmp_path / "test.hdf5")
        ref = _make_npz(npz_path)
        convert_npz_to_hdf5(npz_path, hdf5_path)

        data = open_demo(hdf5_path)
        np.testing.assert_array_almost_equal(data['observations'], ref['obs'])
        np.testing.assert_array_almost_equal(data['actions'], ref['actions'])
        assert 'observations' in data
        data.close()

    def test_open_demo_hdf5_metadata_item(self, tmp_path):
        from demo_io import open_demo, convert_npz_to_hdf5
        npz_path = str(tmp_path / "test.npz")
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_npz(npz_path)
        convert_npz_to_hdf5(npz_path, hdf5_path)

        data = open_demo(hdf5_path)
        meta = data['metadata'].item()
        assert meta['obs_dim'] == 10
        assert meta['action_dim'] == 2
        data.close()

    def test_open_demo_hdf5_contains_metadata(self, tmp_path):
        from demo_io import open_demo, convert_npz_to_hdf5
        npz_path = str(tmp_path / "test.npz")
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_npz(npz_path)
        convert_npz_to_hdf5(npz_path, hdf5_path)

        data = open_demo(hdf5_path)
        assert 'metadata' in data
        assert 'observations' in data
        assert 'nonexistent' not in data
        data.close()


# ============================================================================
# Tests: HDF5DemoWriter
# ============================================================================

class TestHDF5DemoWriter:
    def test_writer_creates_file(self, tmp_path):
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)
        assert os.path.exists(path)
        assert writer.step_cursor == 0
        assert writer.ep_cursor == 0
        writer.close()

    def test_append_steps_incremental(self, tmp_path):
        """append_steps writes delta only, cursors advance correctly."""
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)

        # First batch
        obs1 = np.random.randn(5, 10).astype(np.float32)
        act1 = np.random.randn(5, 2).astype(np.float32)
        rew1 = np.random.randn(5).astype(np.float32)
        don1 = np.zeros(5, dtype=bool)
        cos1 = np.zeros(5, dtype=np.float32)
        writer.append_steps(obs1, act1, rew1, don1, cos1)
        assert writer.step_cursor == 5

        # Second batch
        obs2 = np.random.randn(3, 10).astype(np.float32)
        act2 = np.random.randn(3, 2).astype(np.float32)
        rew2 = np.random.randn(3).astype(np.float32)
        don2 = np.zeros(3, dtype=bool)
        cos2 = np.zeros(3, dtype=np.float32)
        writer.append_steps(obs2, act2, rew2, don2, cos2)
        assert writer.step_cursor == 8

        writer.flush()

        # Verify on disk
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape == (8, 10)
            np.testing.assert_array_almost_equal(f['observations'][:5], obs1)
            np.testing.assert_array_almost_equal(f['observations'][5:], obs2)

        writer.close()

    def test_append_steps_empty(self, tmp_path):
        """Appending zero steps is a no-op."""
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)
        writer.append_steps([], [], [], [], [])
        assert writer.step_cursor == 0
        writer.close()

    def test_append_episode(self, tmp_path):
        """Episode metadata rows accumulate."""
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)

        writer.append_episode(0, 10, 5.5, True)
        writer.append_episode(10, 8, 3.2, False)
        assert writer.ep_cursor == 2

        writer.flush()

        with h5py.File(path, 'r') as f:
            assert f['episode_starts'].shape == (2,)
            assert f['episode_starts'][0] == 0
            assert f['episode_starts'][1] == 10
            assert f['episode_lengths'][0] == 10
            assert f['episode_lengths'][1] == 8
            assert bool(f['episode_success'][0]) is True
            assert bool(f['episode_success'][1]) is False

        writer.close()

    def test_truncate_to(self, tmp_path):
        """truncate_to shrinks datasets for abandon_episode."""
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)

        obs = np.random.randn(10, 10).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        rew = np.random.randn(10).astype(np.float32)
        don = np.zeros(10, dtype=bool)
        cos = np.zeros(10, dtype=np.float32)
        writer.append_steps(obs, act, rew, don, cos)
        assert writer.step_cursor == 10

        writer.truncate_to(6)
        assert writer.step_cursor == 6

        writer.flush()
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape == (6, 10)
            np.testing.assert_array_almost_equal(f['observations'][:], obs[:6])

        writer.close()

    def test_resume_existing_file(self, tmp_path):
        """Opening an existing file resumes with correct cursors."""
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")

        # First session: write 5 steps + 1 episode
        writer1 = HDF5DemoWriter(path, obs_dim=10, action_dim=2)
        obs1 = np.random.randn(5, 10).astype(np.float32)
        act1 = np.random.randn(5, 2).astype(np.float32)
        writer1.append_steps(obs1, act1,
                             np.zeros(5, dtype=np.float32),
                             np.zeros(5, dtype=bool),
                             np.zeros(5, dtype=np.float32))
        writer1.append_episode(0, 5, 1.0, True)
        writer1.close()

        # Second session: resume, append more
        writer2 = HDF5DemoWriter(path, obs_dim=10, action_dim=2)
        assert writer2.step_cursor == 5
        assert writer2.ep_cursor == 1

        obs2 = np.random.randn(3, 10).astype(np.float32)
        act2 = np.random.randn(3, 2).astype(np.float32)
        writer2.append_steps(obs2, act2,
                             np.zeros(3, dtype=np.float32),
                             np.zeros(3, dtype=bool),
                             np.zeros(3, dtype=np.float32))
        writer2.append_episode(5, 3, 0.5, False)
        writer2.close()

        # Verify contiguous
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape == (8, 10)
            np.testing.assert_array_almost_equal(f['observations'][:5], obs1)
            np.testing.assert_array_almost_equal(f['observations'][5:], obs2)
            assert f['episode_starts'].shape == (2,)

    def test_set_metadata(self, tmp_path):
        from demo_io import HDF5DemoWriter
        path = str(tmp_path / "test.hdf5")
        writer = HDF5DemoWriter(path, obs_dim=10, action_dim=2)
        writer.set_metadata({'obs_dim': 10, 'action_dim': 2, 'has_cost': True})
        writer.close()

        with h5py.File(path, 'r') as f:
            assert f.attrs['obs_dim'] == 10
            assert f.attrs['has_cost'] == True


# ============================================================================
# Tests: convert_npz_to_hdf5
# ============================================================================

class TestConvertNpzToHdf5:
    def test_roundtrip_identical_arrays(self, tmp_path):
        """NPZ -> HDF5 -> verify identical arrays."""
        from demo_io import convert_npz_to_hdf5, open_demo
        npz_path = str(tmp_path / "test.npz")
        hdf5_path = str(tmp_path / "test.hdf5")
        ref = _make_npz(npz_path, include_costs=True)
        convert_npz_to_hdf5(npz_path, hdf5_path)

        data = open_demo(hdf5_path)
        np.testing.assert_array_almost_equal(data['observations'], ref['obs'])
        np.testing.assert_array_almost_equal(data['actions'], ref['actions'])
        np.testing.assert_array_almost_equal(data['rewards'], ref['rewards'])
        np.testing.assert_array_equal(data['dones'], ref['dones'])
        np.testing.assert_array_almost_equal(data['costs'], ref['costs'])
        np.testing.assert_array_equal(data['episode_starts'], ref['ep_starts'])
        np.testing.assert_array_equal(data['episode_lengths'], ref['ep_lengths'])
        np.testing.assert_array_almost_equal(data['episode_returns'], ref['ep_returns'])
        np.testing.assert_array_equal(data['episode_success'], ref['ep_success'])
        data.close()

    def test_convert_metadata_preserved(self, tmp_path):
        from demo_io import convert_npz_to_hdf5, open_demo
        npz_path = str(tmp_path / "test.npz")
        hdf5_path = str(tmp_path / "test.hdf5")
        _make_npz(npz_path)
        convert_npz_to_hdf5(npz_path, hdf5_path)

        data = open_demo(hdf5_path)
        meta = data['metadata'].item()
        assert meta['obs_dim'] == 10
        assert meta['num_episodes'] == 2
        data.close()


# ============================================================================
# Tests: DemoRecorder with HDF5
# ============================================================================

class TestDemoRecorderHdf5:
    def _record_episode(self, recorder, n_steps=10, success=True):
        """Helper: record one complete episode."""
        recorder.start_recording()
        for i in range(n_steps):
            recorder.record_step(
                obs=np.random.randn(recorder.obs_dim).astype(np.float32),
                action=np.random.randn(recorder.action_dim).astype(np.float32),
                reward=float(np.random.randn()),
                done=(i == n_steps - 1),
            )
        recorder.mark_episode_success(success)
        recorder.finalize_episode()

    def test_hdf5_checkpoint_writes_delta(self, tmp_path):
        """Checkpoint save only writes new data, not full rewrite."""
        from jetbot_keyboard_control import DemoRecorder

        path = str(tmp_path / "test.hdf5")
        recorder = DemoRecorder(obs_dim=10, action_dim=2, hdf5_path=path)

        # Record first episode
        self._record_episode(recorder, n_steps=5)
        recorder.save(path, finalize_pending=False)

        # Verify first flush
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape[0] == 5
            assert f['episode_starts'].shape[0] == 1

        # Record second episode
        self._record_episode(recorder, n_steps=3)
        recorder.save(path, finalize_pending=False)

        # Verify incremental append
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape[0] == 8
            assert f['episode_starts'].shape[0] == 2

        # Final save closes the writer
        recorder.save(path, finalize_pending=True)

    def test_hdf5_abandon_truncates(self, tmp_path):
        """abandon_episode truncates HDF5 when data was already flushed."""
        from jetbot_keyboard_control import DemoRecorder

        path = str(tmp_path / "test.hdf5")
        recorder = DemoRecorder(obs_dim=10, action_dim=2, hdf5_path=path)

        # Record and flush a complete episode
        self._record_episode(recorder, n_steps=5)
        recorder.save(path, finalize_pending=False)

        # Start another episode, record some steps, then flush (checkpoint)
        recorder.start_recording()
        for i in range(3):
            recorder.record_step(
                obs=np.random.randn(10).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                reward=0.0, done=False,
            )
        recorder.save(path, finalize_pending=False)

        # Data is now flushed to disk (8 steps total)
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape[0] == 8

        # Abandon the in-progress episode
        recorder.abandon_episode()

        # HDF5 should be truncated back to 5
        with h5py.File(path, 'r') as f:
            assert f['observations'].shape[0] == 5

        recorder.save(path, finalize_pending=True)

    def test_backward_compat_npz(self, tmp_path):
        """DemoRecorder without hdf5_path still saves NPZ correctly."""
        from jetbot_keyboard_control import DemoRecorder

        path = str(tmp_path / "test.npz")
        recorder = DemoRecorder(obs_dim=10, action_dim=2)

        self._record_episode(recorder, n_steps=5)
        recorder.save(path)

        # Verify NPZ is readable
        data = np.load(path, allow_pickle=True)
        assert data['observations'].shape == (5, 10)
        assert len(data['episode_starts']) == 1

    def test_hdf5_save_metadata(self, tmp_path):
        """Metadata is written as HDF5 root attributes."""
        from jetbot_keyboard_control import DemoRecorder
        from demo_io import open_demo

        path = str(tmp_path / "test.hdf5")
        recorder = DemoRecorder(obs_dim=10, action_dim=2, hdf5_path=path)
        self._record_episode(recorder, n_steps=5)
        recorder.save(path, metadata={'has_cost': True})

        data = open_demo(path)
        meta = data['metadata'].item()
        assert meta['obs_dim'] == 10
        assert meta['total_frames'] == 5
        assert meta['has_cost'] == True
        data.close()

    def test_hdf5_load_and_resume(self, tmp_path):
        """DemoRecorder.load() with HDF5 re-opens writer for appending."""
        from jetbot_keyboard_control import DemoRecorder

        path = str(tmp_path / "test.hdf5")
        recorder = DemoRecorder(obs_dim=10, action_dim=2, hdf5_path=path)
        self._record_episode(recorder, n_steps=5)
        recorder.save(path, metadata={})

        # Load and resume
        recorder2 = DemoRecorder.load(path)
        assert recorder2._hdf5_writer is not None
        assert len(recorder2.observations) == 5

        # Record more
        self._record_episode(recorder2, n_steps=3)
        recorder2.save(path, metadata={})

        with h5py.File(path, 'r') as f:
            assert f['observations'].shape[0] == 8
            assert f['episode_starts'].shape[0] == 2

    def test_clear_closes_hdf5_writer(self, tmp_path):
        """clear() closes and nullifies the HDF5 writer."""
        from jetbot_keyboard_control import DemoRecorder

        path = str(tmp_path / "test.hdf5")
        recorder = DemoRecorder(obs_dim=10, action_dim=2, hdf5_path=path)
        assert recorder._hdf5_writer is not None

        recorder.clear()
        assert recorder._hdf5_writer is None
