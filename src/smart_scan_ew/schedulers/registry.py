"""Scheduler registry and factory."""

from __future__ import annotations

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.bandit import ThompsonSamplingScheduler
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.schedulers.greedy import GreedyOccupancyScheduler
from smart_scan_ew.schedulers.periodic import PeriodicityAwareScheduler
from smart_scan_ew.schedulers.random_scan import RandomScanScheduler
from smart_scan_ew.schedulers.round_robin import RoundRobinScheduler
from smart_scan_ew.schedulers.uniform import UniformSweepScheduler

SCHEDULER_REGISTRY: dict[str, type[BaseScheduler]] = {
    "uniform_sweep": UniformSweepScheduler,
    "random_scan": RandomScanScheduler,
    "round_robin": RoundRobinScheduler,
    "greedy_occupancy": GreedyOccupancyScheduler,
    "thompson_sampling": ThompsonSamplingScheduler,
    "periodicity_aware": PeriodicityAwareScheduler,
}


def make_scheduler(name: str, receiver_cfg: ReceiverConfig, seed: int = 0) -> BaseScheduler:
    """Instantiate a scheduler by name."""
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULER_REGISTRY)}")
    return SCHEDULER_REGISTRY[name](receiver_cfg, seed=seed)
