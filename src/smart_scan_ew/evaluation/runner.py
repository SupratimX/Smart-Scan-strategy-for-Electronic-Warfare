"""Episode runner: orchestrates simulator + scheduler + belief state + metrics."""

from __future__ import annotations

import uuid

import numpy as np

from smart_scan_ew.config import (
    BenchmarkSuiteConfig,
    ReceiverConfig,
    ScenarioConfig,
    load_scenario_config,
)
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.evaluation.metrics import EpisodeMetrics, compute_metrics, compute_reward
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.state_estimation.belief_state import BeliefStateEstimator
from smart_scan_ew.types import EpisodeSummary


class EpisodeRunner:
    """Runs one complete episode and returns an EpisodeSummary + EpisodeMetrics.

    The runner is the ONLY component allowed to read simulator truth after
    the episode ends.  During execution, it withholds truth from the scheduler
    by never passing it to ``scheduler.select_action()``.
    """

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        receiver_cfg: ReceiverConfig,
        scheduler: BaseScheduler,
        seed: int = 0,
        reward_weights: dict | None = None,
    ) -> None:
        self.scenario_cfg = scenario_cfg
        self.receiver_cfg = receiver_cfg
        self.scheduler = scheduler
        self.seed = seed
        self.reward_weights = reward_weights or {}

    def run(self) -> tuple[EpisodeSummary, EpisodeMetrics]:
        """Execute the full episode and return summary + metrics."""
        episode_id = str(uuid.uuid4())[:8]
        sim = Simulator(self.scenario_cfg, self.receiver_cfg, seed=self.seed)
        belief = BeliefStateEstimator(
            n_bands=self.receiver_cfg.n_bands,
            discount=0.99,
            max_revisit_slots=self.receiver_cfg.max_revisit_slots,
        )
        self.scheduler.reset()

        last_obs = None
        prev_band = None

        while not sim.done:
            bs = belief.snapshot()
            action = self.scheduler.select_action(bs, last_obs)
            obs = sim.step(action)

            # Assign reward (uses truth fields stored in obs — runner-only)
            obs.reward = compute_reward(
                obs,
                w_detection=self.reward_weights.get("w_detection", 1.0),
                w_false_alarm=self.reward_weights.get("w_false_alarm", 0.5),
                w_intercept_time=self.reward_weights.get("w_intercept_time", 0.2),
                w_coverage=self.reward_weights.get("w_coverage", 0.1),
                w_switch=self.reward_weights.get("w_switch", 0.05),
                prev_band=prev_band,
            )

            # Update belief and scheduler (policy-visible only)
            belief.update(obs)
            self.scheduler.update(obs)

            last_obs = obs
            prev_band = obs.band_id

        # Build episode summary (truth log only accessible here)
        episode = EpisodeSummary(
            episode_id=episode_id,
            scenario_id=self.scenario_cfg.scenario_id,
            scheduler_name=self.scheduler.name,
            seed=self.seed,
            n_steps=len(sim.observations),
            observations=list(sim.observations),
            truth_log=list(sim._internal_truth_log),  # runner-only access
            metadata={
                "scenario_cfg": self.scenario_cfg.model_dump(),
                "scheduler": self.scheduler.name,
            },
        )
        metrics = compute_metrics(episode, self.reward_weights)
        return episode, metrics


def run_benchmark(
    suite_cfg: BenchmarkSuiteConfig,
    receiver_cfg: ReceiverConfig,
    scheduler_factory: dict[str, type[BaseScheduler]],
    project_root: str = ".",
    verbose: bool = True,
) -> list[EpisodeMetrics]:
    """Run the full benchmark suite across all scenario × scheduler × seed combinations."""
    import os

    all_metrics: list[EpisodeMetrics] = []
    for scenario_path in suite_cfg.scenario_configs:
        full_path = os.path.join(project_root, scenario_path)
        scenario_cfg = load_scenario_config(full_path)

        for sched_name in suite_cfg.scheduler_names:
            if sched_name not in scheduler_factory:
                if verbose:
                    print(f"  [skip] scheduler '{sched_name}' not registered")
                continue
            SchedulerClass = scheduler_factory[sched_name]

            seed_metrics: list[EpisodeMetrics] = []
            for s_offset in range(suite_cfg.n_seeds):
                seed = suite_cfg.base_seed + s_offset
                scheduler = SchedulerClass(receiver_cfg, seed=seed)
                runner = EpisodeRunner(
                    scenario_cfg=scenario_cfg,
                    receiver_cfg=receiver_cfg,
                    scheduler=scheduler,
                    seed=seed,
                    reward_weights=suite_cfg.reward_weights,
                )
                _, m = runner.run()
                all_metrics.append(m)
                seed_metrics.append(m)
                if verbose:
                    print(
                        f"  {sched_name:25s} | {scenario_cfg.scenario_id:25s} | "
                        f"seed={seed} | Pd={m.pd:.3f} Pfa={m.pfa:.3f} "
                        f"IR={m.interception_rate:.3f} "
                        f"T_int={m.mean_intercept_time:.1f}"
                    )

            # Print per-scheduler summary (mean ± std) across all seeds
            if verbose and len(seed_metrics) > 1:
                ir_vals = np.array([m.interception_rate for m in seed_metrics])
                ti_vals = np.array(
                    [
                        m.mean_intercept_time
                        for m in seed_metrics
                        if m.mean_intercept_time == m.mean_intercept_time
                    ]
                )
                summary = (
                    f"  {'[ summary ]':>25s}   {scenario_cfg.scenario_id:25s}   "
                    f"avg_IR={ir_vals.mean():.3f} ±{ir_vals.std():.3f}"
                )
                if len(ti_vals):
                    summary += f"  avg_T_int={ti_vals.mean():.1f} ±{ti_vals.std():.1f}"
                print(summary)
    return all_metrics
