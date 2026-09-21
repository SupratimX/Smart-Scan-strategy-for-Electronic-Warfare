"""CLI script: evaluate all schedulers on the Turing Synthetic Radar Dataset.

Uses HuggingFace streaming to fetch only the records needed (~few MB),
never downloading the full 70 GB dataset.

Usage
-----
    # Online (streams from HuggingFace):
    python scripts/load_turing_dataset.py

    # Offline / demo fallback (built-in stub, no internet needed):
    python scripts/load_turing_dataset.py --offline

    # Custom sample size and episode length:
    python scripts/load_turing_dataset.py --n-records 200 --n-slots 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the src package is importable when running as a script
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.data.turing_loader import load_turing_scenario
from smart_scan_ew.evaluation.reports import generate_html_report
from smart_scan_ew.evaluation.runner import EpisodeRunner
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Smart Scan schedulers on the Turing Synthetic Radar Dataset"
    )
    p.add_argument(
        "--n-records",
        type=int,
        default=500,
        help="Number of dataset records to stream (default: 500, ~few MB transferred)",
    )
    p.add_argument(
        "--n-slots",
        type=int,
        default=1000,
        help="Episode length in time slots (default: 1000)",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Independent seeds per scheduler (default: 3)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force offline stub — no network call, useful during demos",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "artifacts" / "results" / "turing",
        help="Directory to write results and HTML report",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load (or stub) the Turing scenario                               #
    # ------------------------------------------------------------------ #
    print(
        f"\n{'=' * 60}\n"
        f"  Turing Synthetic Radar Dataset Evaluation\n"
        f"  Mode     : {'OFFLINE STUB' if args.offline else 'streaming (HuggingFace)'}\n"
        + (f"  Records  : {args.n_records} streamed (~few MB)\n" if not args.offline else "")
        + f"  n_slots  : {args.n_slots}\n"
        f"  n_seeds  : {args.n_seeds}\n"
        f"{'=' * 60}\n"
    )

    scenario_cfg = load_turing_scenario(
        n_records=args.n_records,
        n_slots=args.n_slots,
        seed=42,
        force_offline=args.offline,
    )
    print(f"  Scenario : {scenario_cfg.scenario_id}")
    print(f"  Emitters : {len(scenario_cfg.emitters)}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Run all registered schedulers                                    #
    # ------------------------------------------------------------------ #
    receiver_cfg = ReceiverConfig()  # default 32-band config
    from smart_scan_ew.evaluation.metrics import EpisodeMetrics

    all_metrics: list[EpisodeMetrics] = []

    for sched_name, SchedulerClass in SCHEDULER_REGISTRY.items():
        seed_metrics = []
        for seed_offset in range(args.n_seeds):
            seed = 100 + seed_offset
            scheduler = SchedulerClass(receiver_cfg, seed=seed)
            runner = EpisodeRunner(
                scenario_cfg=scenario_cfg,
                receiver_cfg=receiver_cfg,
                scheduler=scheduler,
                seed=seed,
            )
            _, m = runner.run()
            all_metrics.append(m)
            seed_metrics.append(m)
            print(
                f"  {sched_name:25s} | seed={seed} | "
                f"Pd={m.pd:.3f}  Pfa={m.pfa:.3f}  "
                f"IR={m.interception_rate:.3f}  "
                f"T_int={m.mean_intercept_time:.1f}"
            )

        import numpy as np

        ir_vals = [m.interception_rate for m in seed_metrics]
        print(
            f"  {'[ summary ]':>25s}           "
            f"avg_IR={np.mean(ir_vals):.3f} ±{np.std(ir_vals):.3f}\n"
        )

    # ------------------------------------------------------------------ #
    # 3. Save JSON results + HTML report                                  #
    # ------------------------------------------------------------------ #
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "results.json"
    results_data = [
        {
            "scheduler": m.scheduler_name,
            "seed": m.seed,
            "pd": m.pd,
            "pfa": m.pfa,
            "interception_rate": m.interception_rate,
            "mean_intercept_time": m.mean_intercept_time,
            "p95_intercept_time": m.p95_intercept_time,
            "coverage_ratio": m.coverage_ratio,
            "cumulative_reward": m.cumulative_reward,
        }
        for m in all_metrics
    ]
    results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"\n  Results saved  -> {results_path}")

    report_path = args.output_dir / "turing_benchmark.html"
    generate_html_report(
        all_metrics,
        output_path=report_path,
        subtitle=f"Turing Radar Dataset — {scenario_cfg.scenario_id}",
        n_seeds=args.n_seeds,
    )
    print(f"  HTML report    -> {report_path}")
    print(f"\n{'=' * 60}\n  Done.\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
