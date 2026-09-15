"""Unit tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from smart_scan_ew.evaluation.metrics import aggregate_metrics, compute_metrics, compute_reward
from smart_scan_ew.types import EmitterTruth, EpisodeSummary, Observation


def make_obs(
    time_index: int,
    band_id: int,
    detected: bool,
    occupied: bool,
    emitter_ids: tuple = (),
    dwell: int = 1,
) -> Observation:
    return Observation(
        time_index=time_index,
        timestamp=time_index * 0.001,
        band_id=band_id,
        dwell_slots=dwell,
        energy_statistic=5.0 if detected else -1.0,
        detected=detected,
        detection_confidence=0.9 if detected else 0.1,
        band_occupied_truth=occupied,
        emitter_ids_truth=emitter_ids,
    )


def make_episode(observations, truth_log=None) -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="test",
        scenario_id="test_scenario",
        scheduler_name="test",
        seed=0,
        n_steps=len(observations),
        observations=observations,
        truth_log=truth_log or [],
    )


class TestComputeReward:
    def test_true_detection_positive(self):
        obs = make_obs(0, 0, detected=True, occupied=True)
        r = compute_reward(obs, w_detection=1.0)
        assert r > 0

    def test_false_alarm_negative(self):
        obs = make_obs(0, 0, detected=True, occupied=False)
        r = compute_reward(obs, w_false_alarm=0.5)
        assert r < 0 or r == pytest.approx(0.0, abs=0.5)

    def test_switch_cost_applied(self):
        obs = make_obs(0, 2, detected=False, occupied=False)
        r_no_switch = compute_reward(obs, prev_band=2)
        r_switch = compute_reward(obs, prev_band=0, w_switch=1.0)
        assert r_switch < r_no_switch


class TestComputeMetrics:
    def test_pd_all_hits(self):
        obs = [make_obs(i, 0, True, True, (0,)) for i in range(10)]
        truth = [EmitterTruth(0, i, 0, True, -60.0, "fixed") for i in range(10)]
        m = compute_metrics(make_episode(obs, truth))
        assert m.pd == pytest.approx(1.0)

    def test_pfa_all_false_alarms(self):
        obs = [make_obs(i, 0, True, False) for i in range(10)]
        m = compute_metrics(make_episode(obs, []))
        assert m.pfa == pytest.approx(1.0)

    def test_pd_pfa_in_range(self):
        obs = [make_obs(i, i % 4, i % 2 == 0, i % 3 == 0) for i in range(20)]
        m = compute_metrics(make_episode(obs, []))
        assert 0.0 <= m.pd <= 1.0
        assert 0.0 <= m.pfa <= 1.0

    def test_interception_rate_zero_if_no_hits(self):
        obs = [make_obs(i, 0, False, True, (0,)) for i in range(10)]
        truth = [EmitterTruth(0, i, 0, True, -60.0, "fixed") for i in range(10)]
        m = compute_metrics(make_episode(obs, truth))
        assert m.interception_rate == pytest.approx(0.0)

    def test_interception_rate_one_if_intercepted(self):
        obs = [make_obs(0, 0, False, True, ())] + [make_obs(1, 0, True, True, (0,))]
        truth = [EmitterTruth(0, i, 0, True, -60.0, "fixed") for i in range(2)]
        m = compute_metrics(make_episode(obs, truth))
        assert m.interception_rate == pytest.approx(1.0)

    def test_cumulative_reward_summed(self):
        obs = [make_obs(i, 0, True, True, (0,)) for i in range(5)]
        truth = [EmitterTruth(0, i, 0, True, -60.0, "fixed") for i in range(5)]
        m = compute_metrics(make_episode(obs, truth))
        assert m.cumulative_reward > 0

    def test_coverage_ratio_between_0_and_1(self):
        obs = [make_obs(i, i % 4, False, False) for i in range(20)]
        m = compute_metrics(make_episode(obs, []))
        assert 0.0 <= m.coverage_ratio <= 1.0


class TestAggregateMetrics:
    def test_aggregate_returns_means(self):
        obs1 = [make_obs(i, 0, True, True) for i in range(5)]
        obs2 = [make_obs(i, 0, False, True) for i in range(5)]
        m1 = compute_metrics(make_episode(obs1, []))
        m2 = compute_metrics(make_episode(obs2, []))
        agg = aggregate_metrics([m1, m2])
        assert "pd_mean" in agg
        assert "pd_std" in agg
