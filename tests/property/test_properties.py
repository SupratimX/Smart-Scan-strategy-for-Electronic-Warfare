"""Property-based tests using hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from smart_scan_ew.config import ReceiverConfig, ScenarioConfig
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY
from smart_scan_ew.types import Action, BeliefState

RECEIVER_CFG = ReceiverConfig(n_bands=8, max_revisit_slots=100)


def make_neutral_belief(n: int = 8) -> BeliefState:
    return BeliefState(
        n_bands=n,
        occupancy_prob=np.full(n, 0.5),
        occupancy_uncertainty=np.full(n, 0.01),
        time_since_scan=np.zeros(n),
        time_since_hit=np.full(n, 10.0),
        estimated_period=np.zeros(n),
        estimated_phase=np.zeros(n),
        period_confidence=np.zeros(n),
    )


# ---------------------------------------------------------------------------
# Property: Scheduler never selects an out-of-range band
# ---------------------------------------------------------------------------


@given(
    sched_name=st.sampled_from(list(SCHEDULER_REGISTRY)),
    seed=st.integers(0, 10_000),
    occ=st.lists(st.floats(0.0, 1.0), min_size=8, max_size=8),
)
@settings(max_examples=200, deadline=5000)
def test_scheduler_band_always_valid(sched_name, seed, occ):
    sched = SCHEDULER_REGISTRY[sched_name](RECEIVER_CFG, seed=seed)
    n = 8
    belief = BeliefState(
        n_bands=n,
        occupancy_prob=np.array(occ),
        occupancy_uncertainty=np.zeros(n),
        time_since_scan=np.zeros(n),
        time_since_hit=np.full(n, 5.0),
        estimated_period=np.zeros(n),
        estimated_phase=np.zeros(n),
        period_confidence=np.zeros(n),
    )
    action = sched.select_action(belief)
    assert 0 <= action.band_id < n


# ---------------------------------------------------------------------------
# Property: Dwell is always from the allowed set
# ---------------------------------------------------------------------------


@given(
    sched_name=st.sampled_from(list(SCHEDULER_REGISTRY)),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=200, deadline=5000)
def test_scheduler_dwell_always_valid(sched_name, seed):
    sched = SCHEDULER_REGISTRY[sched_name](RECEIVER_CFG, seed=seed)
    belief = make_neutral_belief()
    action = sched.select_action(belief)
    assert action.dwell_slots in RECEIVER_CFG.dwell_choices


# ---------------------------------------------------------------------------
# Property: Timestamps are monotonically non-decreasing
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 1000), n_steps=st.integers(1, 30))
@settings(max_examples=100, deadline=10000)
def test_simulator_timestamps_monotone(seed, n_steps):
    cfg = ReceiverConfig(n_bands=4)
    scenario = ScenarioConfig(scenario_id="prop_test", n_slots=n_steps * 4, emitters=[])
    sim = Simulator(scenario, cfg, seed=seed)
    prev_ts = -1.0
    for i in range(n_steps):
        if sim.done:
            break
        obs = sim.step(Action(band_id=i % 4, dwell_slots=1))
        assert obs.timestamp >= prev_ts - 1e-9, "Timestamp must be non-decreasing"
        prev_ts = obs.timestamp


# ---------------------------------------------------------------------------
# Property: Fixed seed reproduces identical episode
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 10_000))
@settings(max_examples=50, deadline=30000)
def test_fixed_seed_reproducible(seed):
    cfg = ReceiverConfig(n_bands=4)
    scenario = ScenarioConfig(scenario_id="repro_test", n_slots=20, emitters=[])

    def run(s):
        sim = Simulator(scenario, cfg, seed=s)
        results = []
        for i in range(5):
            if sim.done:
                break
            obs = sim.step(Action(band_id=i % 4, dwell_slots=1))
            results.append((obs.band_id, obs.detected, round(obs.energy_statistic, 8)))
        return results

    assert run(seed) == run(seed)


# ---------------------------------------------------------------------------
# Property: Metric values are in valid ranges
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 500))
@settings(max_examples=50, deadline=30000)
def test_metrics_in_valid_ranges(seed):
    from smart_scan_ew.evaluation.runner import EpisodeRunner
    from smart_scan_ew.schedulers.uniform import UniformSweepScheduler

    cfg = ReceiverConfig(n_bands=4)
    scenario = ScenarioConfig(
        scenario_id="range_test",
        n_slots=40,
        emitters=[{"kind": "fixed", "emitter_id": 0, "band_id": 1, "power_dbm": -55.0}],
    )
    sched = UniformSweepScheduler(cfg, seed=seed)
    runner = EpisodeRunner(scenario, cfg, sched, seed=seed)
    _, m = runner.run()

    assert 0.0 <= m.pd <= 1.0
    assert 0.0 <= m.pfa <= 1.0
    assert 0.0 <= m.interception_rate <= 1.0
    assert 0.0 <= m.coverage_ratio <= 1.0
    assert m.n_true_detections >= 0
    assert m.n_false_alarms >= 0
