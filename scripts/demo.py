#!/usr/bin/env python
"""demo.py — Interactive / terminal visualizer demonstrating adaptive scanning in action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smart_scan_ew.config import load_receiver_config, load_scenario_config
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY, make_scheduler
from smart_scan_ew.state_estimation.belief_state import BeliefStateEstimator


def render_belief_bars(occupancy: list[float], selected_band: int) -> str:
    """Render a compact ASCII occupancy belief bar for each band."""
    bars = []
    blocks = [".", "-", "=", "+", "#", "@"]
    for idx, p in enumerate(occupancy):
        level = int(p * 5.0)
        char = blocks[min(max(level, 0), 5)]
        if idx == selected_band:
            bars.append(f"[{char}]")
        else:
            bars.append(f" {char} ")
    return "".join(bars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live EW Receiver Scheduler Demo")
    parser.add_argument("--scenario", default="configs/scenarios/periodic_emitters.yaml")
    parser.add_argument("--receiver", default="configs/receiver/baseline.yaml")
    parser.add_argument(
        "--scheduler",
        default="thompson_sampling",
        choices=list(SCHEDULER_REGISTRY),
        help="Scheduler algorithm to run",
    )
    parser.add_argument("--steps", type=int, default=25, help="Number of steps to display")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenario_cfg = load_scenario_config(args.scenario)
    receiver_cfg = load_receiver_config(args.receiver)
    scheduler = make_scheduler(args.scheduler, receiver_cfg, seed=args.seed)
    sim = Simulator(scenario_cfg, receiver_cfg, seed=args.seed)
    belief_est = BeliefStateEstimator(
        n_bands=receiver_cfg.n_bands,
        discount=0.99,
        max_revisit_slots=receiver_cfg.max_revisit_slots,
    )
    scheduler.reset()

    print("=" * 85)
    print("  SMART SCAN EW -- LIVE RECEIVER SCANNER DEMO")
    print(f"  Algorithm : {args.scheduler}")
    print(f"  Scenario  : {scenario_cfg.scenario_id} ({len(scenario_cfg.emitters)} emitters)")
    print(f"  Spectrum  : {receiver_cfg.n_bands} frequency bands (0 - {receiver_cfg.n_bands - 1})")
    print("=" * 85)
    print(" Slot | Band | Dwell | Energy (dB) | Status       | Belief Distribution (Bands 0-7)")
    print("-" * 85)

    total_hits = 0
    total_fa = 0
    total_misses = 0
    last_obs = None

    step_count = 0
    while not sim.done and step_count < args.steps:
        bs = belief_est.snapshot()
        action = scheduler.select_action(bs, last_obs)
        obs = sim.step(action)
        belief_est.update(obs)
        scheduler.update(obs)
        last_obs = obs
        step_count += 1

        if obs.detected and obs.band_occupied_truth:
            status = "[HIT SIGNAL]"
            total_hits += 1
        elif obs.detected and not obs.band_occupied_truth:
            status = "[FALSE ALARM]"
            total_fa += 1
        elif not obs.detected and obs.band_occupied_truth:
            status = "[MISSED SIG]"
            total_misses += 1
        else:
            status = " Noise only "

        occ_list = bs.occupancy_prob.tolist()[:8]
        bars = render_belief_bars(occ_list, action.band_id)

        print(
            f" {obs.time_index:4d} |  {action.band_id:2d}  |   {action.dwell_slots}   | "
            f"  {obs.energy_statistic:6.1f} dB  | {status:12s} | {bars}"
        )

    print("-" * 85)
    print(f" Demo Summary ({step_count} scan actions):")
    print(f"   True Detections (Hits): {total_hits}")
    print(f"   False Alarms:           {total_fa}")
    print(f"   Missed Signals:         {total_misses}")
    print("=" * 85)
    print(" Legend: [X] indicates scanned band in this step.")
    print("         '.' = low occupancy belief, '#'/'@' = high occupancy belief.")
    print("=" * 85)


if __name__ == "__main__":
    main()
