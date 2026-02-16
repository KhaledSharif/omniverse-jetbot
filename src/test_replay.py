"""
Test Suite for Demo Replay (replay.py)

Tests cover:
- show_info: Displaying demo statistics

Run with: pytest test_replay.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys

# ============================================================================
# MOCK ISAAC SIM MODULES (replay.py imports isaacsim inside visual_playback,
# but show_info doesn't need it. We still mock to be safe for module import.)
# ============================================================================

sys.modules.setdefault('isaacsim', MagicMock())
sys.modules.setdefault('isaacsim.core', MagicMock())
sys.modules.setdefault('isaacsim.core.api', MagicMock())
sys.modules.setdefault('isaacsim.core.utils', MagicMock())
sys.modules.setdefault('isaacsim.core.utils.types', MagicMock())
sys.modules.setdefault('isaacsim.core.utils.nucleus', MagicMock())
sys.modules.setdefault('isaacsim.robot', MagicMock())
sys.modules.setdefault('isaacsim.robot.wheeled_robots', MagicMock())
sys.modules.setdefault('isaacsim.robot.wheeled_robots.robots', MagicMock())
sys.modules.setdefault('isaacsim.robot.wheeled_robots.controllers', MagicMock())

from replay import show_info


# ============================================================================
# HELPERS
# ============================================================================

def make_replay_npz(path, num_episodes=5, steps_per_ep=20, obs_dim=10,
                    success_rate=0.6, include_returns=True, include_metadata=False):
    """Create a synthetic NPZ file matching replay.py's expected format."""
    total = num_episodes * steps_per_ep
    observations = np.random.randn(total, obs_dim).astype(np.float32)
    actions = np.random.randn(total, 2).astype(np.float32)

    episode_starts = np.arange(0, total, steps_per_ep)
    episode_lengths = np.full(num_episodes, steps_per_ep)
    episode_success = np.zeros(num_episodes, dtype=bool)
    episode_success[:int(num_episodes * success_rate)] = True

    save_dict = dict(
        observations=observations,
        actions=actions,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
        episode_success=episode_success,
    )

    if include_returns:
        episode_returns = np.random.randn(num_episodes).astype(np.float32) * 10
        save_dict['episode_returns'] = episode_returns

    if include_metadata:
        save_dict['metadata'] = {'obs_dim': obs_dim, 'action_dim': 2, 'version': '1.0'}

    np.savez(path, **save_dict)
    return path


# ============================================================================
# TEST SUITE: ShowInfo
# ============================================================================

class TestShowInfo:
    def test_prints_stats(self, tmp_path, capsys):
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=5, steps_per_ep=20, success_rate=0.6)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "Total Frames:" in captured.out
        assert "100" in captured.out  # 5 * 20
        assert "Total Episodes:" in captured.out
        assert "5" in captured.out
        assert "Successful:" in captured.out
        assert "3" in captured.out  # int(5 * 0.6) = 3

    def test_missing_returns(self, tmp_path, capsys):
        """NPZ without episode_returns should not crash."""
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=3, steps_per_ep=10, include_returns=False)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "Total Episodes:" in captured.out
        assert "3" in captured.out

    def test_with_metadata(self, tmp_path, capsys):
        """NPZ with metadata key should print metadata."""
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=2, steps_per_ep=10, include_metadata=True)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "Metadata:" in captured.out
        assert "obs_dim" in captured.out

    def test_episode_details_printed(self, tmp_path, capsys):
        """Verify per-episode table is printed."""
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=3, steps_per_ep=10)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "Episode Details:" in captured.out
        assert "SUCCESS" in captured.out or "FAILED" in captured.out

    def test_all_successful(self, tmp_path, capsys):
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=4, steps_per_ep=10, success_rate=1.0)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_all_failed(self, tmp_path, capsys):
        path = tmp_path / "demo.npz"
        make_replay_npz(path, num_episodes=4, steps_per_ep=10, success_rate=0.0)

        show_info(str(path))

        captured = capsys.readouterr()
        assert "Failed:" in captured.out
