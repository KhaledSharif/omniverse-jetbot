"""
Test Suite for Policy Evaluation (eval_policy.py)

Tests cover:
- compute_eval_metrics: Computing evaluation statistics from episode results

Run with: pytest test_eval_policy.py -v
"""

import pytest
import numpy as np

# eval_policy.py has no Isaac Sim imports at module level, only inside main()
from eval_policy import compute_eval_metrics


# ============================================================================
# TEST SUITE: ComputeEvalMetrics
# ============================================================================

class TestComputeEvalMetrics:
    def test_all_success(self):
        metrics = compute_eval_metrics(
            successes=10,
            total_rewards=[100.0] * 10,
            episode_lengths=[50] * 10,
            total_episodes=10,
        )
        assert metrics['success_rate'] == pytest.approx(100.0)
        assert metrics['avg_reward'] == pytest.approx(100.0)
        assert metrics['std_reward'] == pytest.approx(0.0)
        assert metrics['avg_length'] == pytest.approx(50.0)
        assert metrics['std_length'] == pytest.approx(0.0)
        assert metrics['min_reward'] == pytest.approx(100.0)
        assert metrics['max_reward'] == pytest.approx(100.0)

    def test_no_success(self):
        metrics = compute_eval_metrics(
            successes=0,
            total_rewards=[-10.0] * 10,
            episode_lengths=[500] * 10,
            total_episodes=10,
        )
        assert metrics['success_rate'] == pytest.approx(0.0)
        assert metrics['avg_reward'] == pytest.approx(-10.0)

    def test_mixed(self):
        rewards = [10.0, 20.0, 30.0, 40.0]
        lengths = [100, 200, 300, 400]
        metrics = compute_eval_metrics(
            successes=2,
            total_rewards=rewards,
            episode_lengths=lengths,
            total_episodes=4,
        )
        assert metrics['success_rate'] == pytest.approx(50.0)
        assert metrics['avg_reward'] == pytest.approx(25.0)
        assert metrics['std_reward'] == pytest.approx(np.std(rewards))
        assert metrics['avg_length'] == pytest.approx(250.0)
        assert metrics['std_length'] == pytest.approx(np.std(lengths))
        assert metrics['min_reward'] == pytest.approx(10.0)
        assert metrics['max_reward'] == pytest.approx(40.0)

    def test_single_episode(self):
        metrics = compute_eval_metrics(
            successes=1,
            total_rewards=[42.0],
            episode_lengths=[123],
            total_episodes=1,
        )
        assert metrics['success_rate'] == pytest.approx(100.0)
        assert metrics['avg_reward'] == pytest.approx(42.0)
        assert metrics['std_reward'] == pytest.approx(0.0)
        assert metrics['avg_length'] == pytest.approx(123.0)
        assert metrics['min_reward'] == pytest.approx(42.0)
        assert metrics['max_reward'] == pytest.approx(42.0)

    def test_negative_rewards(self):
        rewards = [-5.0, -10.0, -1.0]
        metrics = compute_eval_metrics(
            successes=0,
            total_rewards=rewards,
            episode_lengths=[10, 20, 30],
            total_episodes=3,
        )
        assert metrics['min_reward'] == pytest.approx(-10.0)
        assert metrics['max_reward'] == pytest.approx(-1.0)
        assert metrics['avg_reward'] == pytest.approx(np.mean(rewards))
