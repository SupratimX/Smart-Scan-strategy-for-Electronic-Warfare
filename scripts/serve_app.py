#!/usr/bin/env python
"""serve_app.py — Lightweight interactive web dashboard server for Smart Scan EW."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add src to python path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "src"))

from smart_scan_ew.config import (
    load_receiver_config,
    load_scenario_config,
)
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.evaluation.metrics import compute_metrics, compute_reward
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY, make_scheduler
from smart_scan_ew.state_estimation.belief_state import BeliefStateEstimator


def clean_nan(obj):
    """Recursively replace NaN and inf values with None and convert numpy types for valid JSON serialization."""
    import numpy as np
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return clean_nan(obj.tolist())
    elif isinstance(obj, dict):
        return {str(k): clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    return obj


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the web UI and simulation API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR / "web"), **kwargs)

    def log_message(self, format, *args):
        # Keep console output concise
        if "api" in args[0]:
            print(f"[API] {args[0]}")

    def do_GET(self):
        try:
            self._handle_get()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            self._send_error(500, str(ex))

    def _handle_get(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._send_json({"status": "ready", "version": "0.1.0"})
            return

        if path == "/api/schedulers":
            self._send_json({
                "schedulers": [
                    {
                        "id": "thompson_sampling",
                        "name": "Thompson Sampling (Bayesian Bandit)",
                        "type": "Cognitive AI",
                        "desc": "Beta-Bernoulli bandit balancing exploration vs exploitation with recency discounting.",
                    },
                    {
                        "id": "periodicity_aware",
                        "name": "Periodicity-Aware Forecaster",
                        "type": "Signal Exploitation",
                        "desc": "Autocorrelation pulse-period estimator predicting future dwell arrival times.",
                    },
                    {
                        "id": "greedy_occupancy",
                        "name": "Greedy Occupancy Scanner",
                        "type": "Heuristic",
                        "desc": "Prioritizes bands with highest instantaneous posterior occupancy.",
                    },
                    {
                        "id": "round_robin",
                        "name": "Round Robin (Starvation Minimizer)",
                        "type": "Baseline",
                        "desc": "Selects the most stale band exceeding revisit bounds.",
                    },
                    {
                        "id": "uniform_sweep",
                        "name": "Uniform Sweep (Sequential)",
                        "type": "Standard Radar Scan",
                        "desc": "Sequential frequency sweep across spectrum (Band 0 -> 31).",
                    },
                    {
                        "id": "random_scan",
                        "name": "Random Walk Scan",
                        "type": "Stochastic Baseline",
                        "desc": "Uniformly random band selection with dwell choices.",
                    },
                ]
            })
            return

        if path == "/api/scenarios":
            scenarios_dir = ROOT_DIR / "configs" / "scenarios"
            scenarios = []
            for yml in sorted(scenarios_dir.glob("*.yaml")):
                cfg = load_scenario_config(yml)
                scenarios.append({
                    "id": yml.stem,
                    "filename": yml.name,
                    "n_slots": cfg.n_slots,
                    "n_emitters": len(cfg.emitters),
                    "emitters": [
                        {
                            "id": e.emitter_id,
                            "type": e.emitter_type,
                            "band_id": getattr(e, "band_id", getattr(e, "initial_band_id", 0)),
                            "power_dbm": e.power_dbm,
                            "period": getattr(e, "period", None),
                            "duty_cycle": getattr(e, "duty_cycle", None),
                        }
                        for e in cfg.emitters
                    ],
                })
            self._send_json({"scenarios": scenarios})
            return

        if path == "/api/benchmark-data":
            metrics_file = ROOT_DIR / "artifacts" / "results" / "all_metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text(encoding="utf-8"))
                    self._send_json(clean_nan(data))
                    return
                except Exception as ex:
                    self._send_error(500, f"Error reading metrics: {ex}")
                    return
            else:
                self._send_json([])
                return

        if path == "/api/simulate":
            query = parse_qs(parsed.query)
            scenario_name = query.get("scenario", ["periodic_emitters"])[0]
            scheduler_name = query.get("scheduler", ["thompson_sampling"])[0]
            seed = int(query.get("seed", [42])[0])
            max_steps = int(query.get("steps", [150])[0])

            scenario_file = ROOT_DIR / "configs" / "scenarios" / f"{scenario_name}.yaml"
            if not scenario_file.exists():
                self._send_error(404, f"Scenario {scenario_name} not found")
                return

            receiver_file = ROOT_DIR / "configs" / "receiver" / "baseline.yaml"
            scenario_cfg = load_scenario_config(scenario_file)
            receiver_cfg = load_receiver_config(receiver_file)

            scheduler = make_scheduler(scheduler_name, receiver_cfg, seed=seed)
            sim = Simulator(scenario_cfg, receiver_cfg, seed=seed)
            belief_est = BeliefStateEstimator(
                n_bands=receiver_cfg.n_bands,
                discount=0.99,
                max_revisit_slots=receiver_cfg.max_revisit_slots,
            )
            scheduler.reset()

            steps = []
            cum_reward = 0.0
            last_obs = None
            prev_band = None

            step_idx = 0
            while not sim.done and step_idx < max_steps:
                bs = belief_est.snapshot()
                action = scheduler.select_action(bs, last_obs)
                obs = sim.step(action)

                # Assign reward
                r = compute_reward(obs, prev_band=prev_band)
                obs.reward = r
                cum_reward += r

                belief_est.update(obs)
                scheduler.update(obs)

                is_hit = obs.detected and obs.band_occupied_truth
                is_fa = obs.detected and not obs.band_occupied_truth
                is_miss = not obs.detected and obs.band_occupied_truth

                steps.append({
                    "step_idx": step_idx,
                    "slot": obs.time_index,
                    "timestamp": round(obs.timestamp, 4),
                    "band_id": obs.band_id,
                    "center_freq_mhz": receiver_cfg.bands[obs.band_id].center_freq_mhz,
                    "dwell_slots": obs.dwell_slots,
                    "energy_statistic": round(obs.energy_statistic, 2),
                    "measured_power_dbm": round(receiver_cfg.bands[obs.band_id].noise_floor_dbm + obs.energy_statistic, 1),
                    "detected": obs.detected,
                    "confidence": round(obs.detection_confidence, 3),
                    "truth_occupied": obs.band_occupied_truth,
                    "truth_snr_db": round(obs.snr_truth_db, 1) if not math.isinf(obs.snr_truth_db) else -99.0,
                    "is_hit": is_hit,
                    "is_fa": is_fa,
                    "is_miss": is_miss,
                    "reward": round(r, 3),
                    "cum_reward": round(cum_reward, 3),
                    "occupancy_prob": [round(float(p), 3) for p in bs.occupancy_prob],
                    "uncertainty": [round(float(u), 4) for u in bs.occupancy_uncertainty],
                })

                last_obs = obs
                prev_band = obs.band_id
                step_idx += 1

            from smart_scan_ew.types import EpisodeSummary
            episode = EpisodeSummary(
                episode_id="web_sim",
                scenario_id=scenario_cfg.scenario_id,
                scheduler_name=scheduler.name,
                seed=seed,
                n_steps=len(sim.observations),
                observations=list(sim.observations),
                truth_log=list(sim._internal_truth_log),
                metadata={},
            )
            metrics = compute_metrics(episode)

            response = {
                "scenario_id": scenario_name,
                "scheduler_id": scheduler_name,
                "seed": seed,
                "n_steps": len(steps),
                "n_bands": receiver_cfg.n_bands,
                "bands": [
                    {"id": b.band_id, "freq_mhz": b.center_freq_mhz}
                    for b in receiver_cfg.bands
                ],
                "steps": steps,
                "summary": {
                    "pd": round(metrics.pd, 3) if not math.isnan(metrics.pd) else 0.0,
                    "pfa": round(metrics.pfa, 4) if not math.isnan(metrics.pfa) else 0.0,
                    "interception_rate": round(metrics.interception_rate, 3),
                    "mean_intercept_time": round(metrics.mean_intercept_time, 1)
                    if metrics.mean_intercept_time and not math.isnan(metrics.mean_intercept_time)
                    else None,
                    "cumulative_reward": round(cum_reward, 2),
                    "total_hits": metrics.n_true_detections,
                    "total_fa": metrics.n_false_alarms,
                    "total_opportunities": metrics.n_true_opportunities,
                },
            }
            self._send_json(clean_nan(response))
            return

        super().do_GET()

    def _send_json(self, data: dict | list):
        body = json.dumps(data).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Smart Scan EW Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardRequestHandler)
    url = f"http://localhost:{args.port}"
    print("=" * 70)
    print("  SMART SCAN EW -- INTERACTIVE TACTICAL COCKPIT & WEB DASHBOARD")
    print(f"  Server listening at: {url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 70)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    main()
