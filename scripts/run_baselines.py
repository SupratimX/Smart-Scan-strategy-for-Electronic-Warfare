#!/usr/bin/env python
"""run_baselines.py — Run all schedulers on a benchmark suite and save results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from smart_scan_ew.config import load_benchmark_suite, load_receiver_config
from smart_scan_ew.evaluation.runner import run_benchmark
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY
from smart_scan_ew.storage.event_log import save_all_metrics_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all schedulers on a benchmark suite.")
    parser.add_argument("--suite", default="configs/evaluation/benchmark_suite.yaml")
    parser.add_argument("--receiver", default="configs/receiver/baseline.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed in suite config")
    parser.add_argument("--output-dir", default="artifacts/results")
    args = parser.parse_args()

    suite_cfg = load_benchmark_suite(args.suite)
    receiver_cfg = load_receiver_config(args.receiver)
    if args.seed is not None:
        suite_cfg.base_seed = args.seed

    print(f"\n{'=' * 70}")
    print(f"  Benchmark suite: {suite_cfg.suite_id}")
    print(f"  Scenarios      : {len(suite_cfg.scenario_configs)}")
    print(f"  Schedulers     : {suite_cfg.scheduler_names}")
    print(f"  Seeds          : {suite_cfg.n_seeds} (base={suite_cfg.base_seed})")
    print(f"{'=' * 70}\n")

    all_metrics = run_benchmark(
        suite_cfg=suite_cfg,
        receiver_cfg=receiver_cfg,
        scheduler_factory=SCHEDULER_REGISTRY,
        project_root=".",
        verbose=True,
    )

    out_path = Path(args.output_dir) / "all_metrics.json"
    save_all_metrics_json(all_metrics, out_path)
    print(f"\nMetrics written to: {out_path}")
    print(f"Total episodes: {len(all_metrics)}")


if __name__ == "__main__":
    main()
