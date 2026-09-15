#!/usr/bin/env python
"""run_all.py — ONE-COMMAND ACCEPTANCE TEST.

Generates data → evaluates all schedulers → prints results → writes HTML report.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --suite configs/evaluation/benchmark_suite.yaml --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_step(desc: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {desc}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command acceptance test.")
    parser.add_argument("--suite", default="configs/evaluation/benchmark_suite.yaml")
    parser.add_argument("--receiver", default="configs/receiver/baseline.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--report", default="artifacts/reports/benchmark.html")
    args = parser.parse_args()

    t0 = time.time()
    py = sys.executable

    # Step 1: Generate a sample dataset with the default scenario
    run_step("Step 1/4: Generating sample dataset (periodic_emitters)", [
        py, "scripts/generate_dataset.py",
        "--config", "configs/scenarios/periodic_emitters.yaml",
        "--receiver", args.receiver,
        "--seed", str(args.seed),
        "--output-dir", "data/raw",
    ])

    # Step 2: Run full benchmark suite
    run_step("Step 2/4: Running full benchmark (all schedulers × all scenarios × 5 seeds)", [
        py, "scripts/run_baselines.py",
        "--suite", args.suite,
        "--receiver", args.receiver,
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
    ])

    # Step 3: Print evaluation summary
    run_step("Step 3/4: Evaluating and printing summary table", [
        py, "scripts/evaluate.py",
        "--results-dir", args.output_dir,
    ])

    # Step 4: Generate HTML report
    run_step("Step 4/4: Generating HTML report", [
        py, "scripts/make_report.py",
        "--results-dir", args.output_dir,
        "--output", args.report,
    ])

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  [OK] All steps completed in {elapsed:.1f}s")
    print(f"  Report: {Path(args.report).resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
