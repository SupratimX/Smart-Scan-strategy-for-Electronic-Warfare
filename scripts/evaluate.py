#!/usr/bin/env python
"""evaluate.py — Load saved metrics and print a summary table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

import numpy as np

from smart_scan_ew.evaluation.metrics import EpisodeMetrics, aggregate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved benchmark metrics.")
    parser.add_argument("--results-dir", default="artifacts/results")
    args = parser.parse_args()

    results_path = Path(args.results_dir) / "all_metrics.json"
    if not results_path.exists():
        print(f"No results found at {results_path}. Run run_baselines.py first.")
        sys.exit(1)

    raw = json.loads(results_path.read_text())
    all_metrics: list[EpisodeMetrics] = []
    for d in raw:
        m = EpisodeMetrics.__new__(EpisodeMetrics)
        m.__dict__.update(d)
        all_metrics.append(m)

    # Group by scheduler
    by_sched: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_sched.setdefault(m.scheduler_name, []).append(m)

    print(f"\n{'='*90}")
    print(f"{'Scheduler':<26} {'Pd':>6} {'Pfa':>6} {'IR':>6} "
          f"{'MeanT':>8} {'P95T':>8} {'Cov':>6} {'Reward':>9}")
    print(f"{'='*90}")

    for name, mlist in by_sched.items():
        agg = aggregate_metrics(mlist)
        print(
            f"{name:<26} "
            f"{agg.get('pd_mean', float('nan')):6.3f} "
            f"{agg.get('pfa_mean', float('nan')):6.3f} "
            f"{agg.get('interception_rate_mean', float('nan')):6.3f} "
            f"{agg.get('mean_intercept_time_mean', float('nan')):8.1f} "
            f"{agg.get('p95_intercept_time_mean', float('nan')):8.1f} "
            f"{agg.get('coverage_ratio_mean', float('nan')):6.3f} "
            f"{agg.get('cumulative_reward_mean', float('nan')):9.1f}"
        )
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
