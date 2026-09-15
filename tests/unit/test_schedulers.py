"""Unit tests for all schedulers."""

from __future__ import annotations

import numpy as np
import pytest

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.bandit import ThompsonSamplingScheduler
from smart_scan_ew.schedulers.greedy import GreedyOccupancyScheduler
from smart_scan_ew.schedulers.periodic import PeriodicityAwareScheduler
from smart_scan_ew.schedulers.random_scan import RandomScanScheduler
from smart_scan_ew.schedulers.round_robin import RoundRobinScheduler
from smart_scan_ew.schedulers.uniform import UniformSweepScheduler
from smart_scan_ew.types import BeliefState


def make_belief(n: int = 8, seed: int = 0) -> BeliefState:
    rng = np.random.default_rng(seed)
    return BeliefState(
        n_bands=n,
        occupancy_prob=rng.uniform(0.0, 1.0, n),
        occupancy_uncertainty=rng.uniform(0.0, 0.1, n),
        time_since_scan=rng.uniform(0, 50, n),
        time_since_hit=rng.uniform(0, 100, n),
        estimated_period=np.zeros(n),
        estimated_phase=np.zeros(n),
        period_confidence=np.zeros(n),
    )


@pytest.fixture
def receiver_cfg() -> ReceiverConfig:
    return ReceiverConfig(n_bands=8)


SCHEDULER_CLASSES = [
    UniformSweepScheduler,
    RandomScanScheduler,
    RoundRobinScheduler,
    GreedyOccupancyScheduler,
    ThompsonSamplingScheduler,
    PeriodicityAwareScheduler,
]


@pytest.mark.parametrize("SchedulerClass", SCHEDULER_CLASSES)
class TestSchedulerContract:
    """Every scheduler must satisfy the same basic contract."""

    def test_returns_valid_band(self, SchedulerClass, receiver_cfg):
        sched = SchedulerClass(receiver_cfg, seed=0)
        belief = make_belief(8)
        action = sched.select_action(belief)
        assert 0 <= action.band_id < 8

    def test_returns_valid_dwell(self, SchedulerClass, receiver_cfg):
        sched = SchedulerClass(receiver_cfg, seed=0)
        belief = make_belief(8)
        action = sched.select_action(belief)
        assert action.dwell_slots in receiver_cfg.dwell_choices

    def test_deterministic_same_seed(self, SchedulerClass, receiver_cfg):
        belief = make_belief(8, seed=42)
        s1 = SchedulerClass(receiver_cfg, seed=99)
        s2 = SchedulerClass(receiver_cfg, seed=99)
        a1 = s1.select_action(belief)
        a2 = s2.select_action(belief)
        assert a1.band_id == a2.band_id

    def test_starvation_guard_fires(self, SchedulerClass, receiver_cfg):
        """If a band exceeds max_revisit, it must be selected."""
        n = 8
        rng = np.random.default_rng(0)
        # Force band 5 to be extremely stale
        tss = np.full(n, 10.0)
        tss[5] = receiver_cfg.max_revisit_slots + 100.0

        belief = BeliefState(
            n_bands=n,
            occupancy_prob=np.ones(n) * 0.1,
            occupancy_uncertainty=np.zeros(n),
            time_since_scan=tss,
            time_since_hit=np.full(n, 10.0),
            estimated_period=np.zeros(n),
            estimated_phase=np.zeros(n),
            period_confidence=np.zeros(n),
        )
        sched = SchedulerClass(receiver_cfg, seed=0)
        action = sched.select_action(belief)
        assert action.band_id == 5

    def test_reset_works(self, SchedulerClass, receiver_cfg):
        sched = SchedulerClass(receiver_cfg, seed=0)
        belief = make_belief(8)
        sched.select_action(belief)
        sched.reset()  # should not raise


class TestUniformSweep:
    def test_cycles_through_all_bands(self, receiver_cfg):
        sched = UniformSweepScheduler(receiver_cfg, seed=0)
        belief = make_belief(8)
        bands = set()
        for _ in range(receiver_cfg.n_bands * 2):
            action = sched.select_action(belief)
            bands.add(action.band_id)
        assert bands == set(range(receiver_cfg.n_bands))


class TestRoundRobin:
    def test_selects_most_stale_band(self, receiver_cfg):
        n = 8
        tss = np.full(n, 5.0)
        tss[3] = 100.0  # band 3 most stale
        belief = BeliefState(
            n_bands=n,
            occupancy_prob=np.zeros(n),
            occupancy_uncertainty=np.zeros(n),
            time_since_scan=tss,
            time_since_hit=np.full(n, 5.0),
            estimated_period=np.zeros(n),
            estimated_phase=np.zeros(n),
            period_confidence=np.zeros(n),
        )
        sched = RoundRobinScheduler(receiver_cfg, seed=0)
        action = sched.select_action(belief)
        assert action.band_id == 3
