"""Integration tests: full episode pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smart_scan_ew.config import (
    BenchmarkSuiteConfig,
    ReceiverConfig,
    ScenarioConfig,
    load_receiver_config,
    load_scenario_config,
)
from smart_scan_ew.evaluation.metrics import aggregate_metrics
from smart_scan_ew.evaluation.runner import EpisodeRunner, run_benchmark
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY
from smart_scan_ew.storage.event_log import save_all_metrics_json, save_episode_jsonl


@pytest.fixture
def receiver_cfg() -> ReceiverConfig:
    return ReceiverConfig(n_bands=8, max_revisit_slots=100)


@pytest.fixture
def periodic_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="integ_periodic",
        n_slots=200,
        seed=42,
        emitters=[
            {"kind": "fixed",    "emitter_id": 0, "band_id": 2, "power_dbm": -55.0},
            {"kind": "periodic", "emitter_id": 1, "band_id": 5,
             "power_dbm": -58.0, "period_slots": 20, "duty_cycle": 0.3,
             "phase_offset_slots": 0, "jitter_slots": 1.0},
            {"kind": "burst",    "emitter_id": 2, "band_id": 7,
             "power_dbm": -62.0, "burst_duration_slots": 2, "inter_burst_slots": 20},
        ],
    )


class TestFullEpisodePipeline:
    @pytest.mark.parametrize("sched_name", list(SCHEDULER_REGISTRY))
    def test_episode_completes(self, sched_name, receiver_cfg, periodic_scenario):
        sched = SCHEDULER_REGISTRY[sched_name](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        episode, metrics = runner.run()
        assert len(episode.observations) > 0
        assert metrics.n_steps > 0

    @pytest.mark.parametrize("sched_name", list(SCHEDULER_REGISTRY))
    def test_truth_never_empty_when_emitters_exist(self, sched_name, receiver_cfg, periodic_scenario):
        sched = SCHEDULER_REGISTRY[sched_name](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        episode, _ = runner.run()
        assert len(episode.truth_log) > 0

    def test_metrics_pd_and_pfa_in_range(self, receiver_cfg, periodic_scenario):
        sched = SCHEDULER_REGISTRY["uniform_sweep"](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        _, metrics = runner.run()
        assert 0.0 <= metrics.pd <= 1.0
        assert 0.0 <= metrics.pfa <= 1.0

    def test_periodicity_aware_intercepts_periodic_emitter(
        self, receiver_cfg, periodic_scenario
    ):
        """Periodicity-aware scheduler should intercept the periodic emitter."""
        sched = SCHEDULER_REGISTRY["periodicity_aware"](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        _, metrics = runner.run()
        # At least one emitter should be intercepted
        assert metrics.n_emitters_intercepted >= 1

    def test_uniform_sweep_all_bands_visited(self, receiver_cfg, periodic_scenario):
        sched = SCHEDULER_REGISTRY["uniform_sweep"](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        episode, _ = runner.run()
        visited_bands = {o.band_id for o in episode.observations}
        # In 200 slots with 8 bands, all bands should be visited
        assert len(visited_bands) == receiver_cfg.n_bands

    def test_replay_reproducibility(self, receiver_cfg, periodic_scenario):
        """Same seed must produce identical episodes."""
        def run_once(seed):
            sched = SCHEDULER_REGISTRY["thompson_sampling"](receiver_cfg, seed=seed)
            runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=seed)
            episode, m = runner.run()
            return [o.detected for o in episode.observations], m.pd

        dets1, pd1 = run_once(77)
        dets2, pd2 = run_once(77)
        assert dets1 == dets2
        assert pd1 == pytest.approx(pd2)


class TestBenchmarkRunner:
    def test_run_benchmark_returns_all_metrics(
        self, receiver_cfg, periodic_scenario, tmp_path
    ):
        suite = BenchmarkSuiteConfig(
            suite_id="test_suite",
            scenario_configs=["configs/scenarios/periodic_emitters.yaml"],
            scheduler_names=["uniform_sweep", "round_robin"],
            n_seeds=2,
            base_seed=0,
        )
        # Write scenario yaml so the runner can load it
        import yaml
        scenario_yaml = tmp_path / "periodic_emitters.yaml"
        scenario_yaml.write_text(
            yaml.dump(periodic_scenario.model_dump()), encoding="utf-8"
        )
        # Patch suite config to point to tmp file
        suite.scenario_configs = [str(scenario_yaml)]

        all_metrics = run_benchmark(
            suite_cfg=suite,
            receiver_cfg=receiver_cfg,
            scheduler_factory=SCHEDULER_REGISTRY,
            project_root=".",
            verbose=False,
        )
        # 2 schedulers × 2 seeds × 1 scenario = 4 episodes
        assert len(all_metrics) == 4


class TestStoragePipeline:
    def test_save_and_reload_metrics(self, receiver_cfg, periodic_scenario, tmp_path):
        sched = SCHEDULER_REGISTRY["greedy_occupancy"](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        episode, metrics = runner.run()

        out_file = tmp_path / "all_metrics.json"
        save_all_metrics_json([metrics], out_file)

        loaded = json.loads(out_file.read_text())
        assert len(loaded) == 1
        assert abs(loaded[0]["pd"] - metrics.pd) < 1e-9

    def test_save_episode_jsonl(self, receiver_cfg, periodic_scenario, tmp_path):
        sched = SCHEDULER_REGISTRY["random_scan"](receiver_cfg, seed=0)
        runner = EpisodeRunner(periodic_scenario, receiver_cfg, sched, seed=0)
        episode, _ = runner.run()

        ep_file = save_episode_jsonl(episode, tmp_path)
        lines = ep_file.read_text().strip().splitlines()
        assert len(lines) == len(episode.observations)


class TestConfigLoading:
    def test_load_receiver_config(self):
        cfg = load_receiver_config("configs/receiver/baseline.yaml")
        assert cfg.n_bands == 32
        assert cfg.retune_delay_slots == 1

    def test_load_scenario_config_periodic(self):
        cfg = load_scenario_config("configs/scenarios/periodic_emitters.yaml")
        assert cfg.scenario_id == "periodic_emitters"
        assert len(cfg.emitters) == 3

    def test_load_scenario_config_dense(self):
        cfg = load_scenario_config("configs/scenarios/dense_environment.yaml")
        assert len(cfg.emitters) == 6
