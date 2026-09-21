#!/usr/bin/env python
"""generate_dataset.py — Generate synthetic episode data for one scenario."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_scan_ew.config import load_receiver_config, load_scenario_config
from smart_scan_ew.evaluation.runner import EpisodeRunner
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY, make_scheduler
from smart_scan_ew.storage.event_log import save_episode_jsonl, save_metrics_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic episode dataset.")
    parser.add_argument("--config", required=True, help="Path to scenario YAML")
    parser.add_argument("--receiver", default="configs/receiver/baseline.yaml")
    parser.add_argument("--scheduler", default="uniform_sweep", choices=list(SCHEDULER_REGISTRY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    scenario_cfg = load_scenario_config(args.config)
    receiver_cfg = load_receiver_config(args.receiver)
    scheduler = make_scheduler(args.scheduler, receiver_cfg, seed=args.seed)

    runner = EpisodeRunner(scenario_cfg, receiver_cfg, scheduler, seed=args.seed)
    episode, metrics = runner.run()

    out = Path(args.output_dir)
    ep_file = save_episode_jsonl(episode, out)
    m_file = save_metrics_json(metrics, out)

    print(f"Episode saved: {ep_file}")
    print(f"Metrics saved: {m_file}")
    print(
        f"  Pd={metrics.pd:.3f} Pfa={metrics.pfa:.3f} "
        f"IR={metrics.interception_rate:.3f} "
        f"MeanT={metrics.mean_intercept_time:.1f}"
    )


if __name__ == "__main__":
    main()
