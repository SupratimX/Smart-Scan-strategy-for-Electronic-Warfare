#!/usr/bin/env python
"""make_report.py — Generate HTML benchmark report from saved metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from smart_scan_ew.evaluation.metrics import EpisodeMetrics
from smart_scan_ew.evaluation.reports import generate_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML report from saved metrics.")
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--output", default="artifacts/reports/benchmark.html")
    parser.add_argument("--title", default="Smart Scan EW — Benchmark Report")
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

    out = generate_html_report(all_metrics, args.output, subtitle=args.title)
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
