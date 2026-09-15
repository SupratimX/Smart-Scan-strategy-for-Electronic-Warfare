"""Unit tests for the RF simulator core."""

from __future__ import annotations

import numpy as np
import pytest

from smart_scan_ew.config import ReceiverConfig, ScenarioConfig
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.types import Action


@pytest.fixture
def receiver_cfg() -> ReceiverConfig:
    return ReceiverConfig(n_bands=8, n_slots=100)


@pytest.fixture
def empty_scenario() -> ScenarioConfig:
    return ScenarioConfig(scenario_id="test_empty", n_slots=50, emitters=[])


@pytest.fixture
def fixed_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="test_fixed",
        n_slots=50,
        emitters=[{"kind": "fixed", "emitter_id": 0, "band_id": 3, "power_dbm": -50.0}],
    )


class TestSimulatorBasics:
    def test_step_returns_observation(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        obs = sim.step(Action(band_id=0, dwell_slots=1))
        assert obs.band_id == 0
        assert obs.dwell_slots == 1
        assert isinstance(obs.detected, bool)

    def test_episode_advances_slot(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        sim.step(Action(band_id=0, dwell_slots=2))
        # After dwell=2 slots, current slot should advance by at least 2
        assert sim.current_slot >= 2

    def test_done_after_all_slots(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        while not sim.done:
            sim.step(Action(band_id=0, dwell_slots=1))
        assert sim.done

    def test_step_after_done_raises(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        while not sim.done:
            sim.step(Action(band_id=0, dwell_slots=1))
        with pytest.raises(RuntimeError):
            sim.step(Action(band_id=0, dwell_slots=1))

    def test_deterministic_with_same_seed(self, empty_scenario, receiver_cfg):
        sim1 = Simulator(empty_scenario, receiver_cfg, seed=7)
        sim2 = Simulator(empty_scenario, receiver_cfg, seed=7)
        obs1 = sim1.step(Action(band_id=2, dwell_slots=1))
        obs2 = sim2.step(Action(band_id=2, dwell_slots=1))
        assert obs1.energy_statistic == pytest.approx(obs2.energy_statistic, abs=1e-9)
        assert obs1.detected == obs2.detected

    def test_different_seeds_different_results(self, empty_scenario, receiver_cfg):
        sim1 = Simulator(empty_scenario, receiver_cfg, seed=1)
        sim2 = Simulator(empty_scenario, receiver_cfg, seed=999)
        energies1 = [sim1.step(Action(band_id=0, dwell_slots=1)).energy_statistic
                     for _ in range(20)]
        energies2 = [sim2.step(Action(band_id=0, dwell_slots=1)).energy_statistic
                     for _ in range(20)]
        # With different seeds, results should differ at least sometimes
        assert not all(abs(a - b) < 1e-9 for a, b in zip(energies1, energies2))

    def test_invalid_band_falls_back_to_zero(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        obs = sim.step(Action(band_id=999, dwell_slots=1))
        assert obs.band_id == 0  # fallback

    def test_timestamps_monotonic(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        for _ in range(10):
            if sim.done:
                break
            sim.step(Action(band_id=0, dwell_slots=1))
        timestamps = [o.timestamp for o in sim.observations]
        assert all(b >= a for a, b in zip(timestamps, timestamps[1:]))

    def test_fixed_emitter_truth_logged(self, fixed_scenario, receiver_cfg):
        sim = Simulator(fixed_scenario, receiver_cfg, seed=42)
        while not sim.done:
            sim.step(Action(band_id=3, dwell_slots=1))
        # Truth log should contain entries for emitter 0
        emitter_ids = {t.emitter_id for t in sim._internal_truth_log}
        assert 0 in emitter_ids

    def test_truth_fields_in_observation(self, fixed_scenario, receiver_cfg):
        sim = Simulator(fixed_scenario, receiver_cfg, seed=42)
        obs = sim.step(Action(band_id=3, dwell_slots=2))
        # Band 3 has an active emitter, so band_occupied_truth should be True
        assert obs.band_occupied_truth is True

    def test_reset_clears_state(self, empty_scenario, receiver_cfg):
        sim = Simulator(empty_scenario, receiver_cfg, seed=42)
        sim.step(Action(band_id=0, dwell_slots=5))
        assert sim.current_slot > 0
        sim.reset(seed=42)
        assert sim.current_slot == 0
        assert len(sim.observations) == 0
