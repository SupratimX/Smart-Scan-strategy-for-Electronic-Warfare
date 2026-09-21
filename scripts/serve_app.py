#!/usr/bin/env python
"""serve_app.py — Lightweight interactive web dashboard server for Smart Scan EW (FastAPI)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add src to python path
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR / "src"))

from smart_scan_ew.config import load_receiver_config, load_scenario_config
from smart_scan_ew.environment.simulator import Simulator
from smart_scan_ew.evaluation.metrics import compute_metrics, compute_reward
from smart_scan_ew.schedulers.registry import make_scheduler
from smart_scan_ew.state_estimation.belief_state import BeliefStateEstimator
from smart_scan_ew.types import EpisodeSummary

app = FastAPI(title="Smart Scan EW Web Dashboard", version="0.1.0")

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_nan(obj):
    """Recursively replace NaN and inf values with None for valid JSON serialization."""
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


@app.get("/api/status")
async def api_status():
    return {"status": "ready", "version": "0.1.0"}


@app.get("/api/schedulers")
async def api_schedulers():
    return {
        "schedulers": [
            {
                "id": "thompson_sampling",
                "name": "Thompson Sampling (Bayesian Bandit)",
                "type": "Cognitive AI",
                "desc": "Beta-Bernoulli bandit balancing exploration vs exploitation.",
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
                "desc": "Sequential frequency sweep across spectrum.",
            },
            {
                "id": "random_scan",
                "name": "Random Walk Scan",
                "type": "Stochastic Baseline",
                "desc": "Uniformly random band selection.",
            },
        ]
    }


@app.get("/api/scenarios")
async def api_scenarios():
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
                    "type": getattr(e, "kind", getattr(e, "emitter_type", "unknown")),
                    "band_id": getattr(e, "band_id", getattr(e, "initial_band_id", 0)),
                    "power_dbm": e.power_dbm,
                    "period": getattr(e, "period", None),
                    "duty_cycle": getattr(e, "duty_cycle", None),
                }
                for e in cfg.emitters
            ],
        })
    return {"scenarios": scenarios}


@app.get("/api/benchmark-data")
async def api_benchmark_data():
    candidate_files = [
        ROOT_DIR / "artifacts" / "results" / "all_metrics.json",
        ROOT_DIR / "artifacts" / "results" / "turing" / "results.json",
        ROOT_DIR / "artifacts" / "results" / "training" / "results.json",
    ]
    merged = []
    for mf in candidate_files:
        if mf.exists():
            try:
                raw = json.loads(mf.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    for rec in raw:
                        if "scheduler_name" not in rec and "scheduler" in rec:
                            rec["scheduler_name"] = rec["scheduler"]
                        if "scenario_id" not in rec:
                            rec["scenario_id"] = mf.parent.name
                        if "seed" not in rec:
                            rec["seed"] = 0
                    merged.extend(raw)
            except Exception:
                pass
    return JSONResponse(content=clean_nan(merged))


@app.get("/api/simulate")
async def api_simulate(
    scenario: str = Query("periodic_emitters"),
    scheduler: str = Query("thompson_sampling"),
    seed: int = Query(42),
    steps: int = Query(150),
):
    scenario_file = ROOT_DIR / "configs" / "scenarios" / f"{scenario}.yaml"
    if not scenario_file.exists():
        raise HTTPException(status_code=404, detail=f"Scenario {scenario} not found")

    receiver_file = ROOT_DIR / "configs" / "receiver" / "baseline.yaml"
    scenario_cfg = load_scenario_config(scenario_file)
    receiver_cfg = load_receiver_config(receiver_file)

    sched = make_scheduler(scheduler, receiver_cfg, seed=seed)
    sim = Simulator(scenario_cfg, receiver_cfg, seed=seed)
    belief_est = BeliefStateEstimator(
        n_bands=receiver_cfg.n_bands,
        discount=0.99,
        max_revisit_slots=receiver_cfg.max_revisit_slots,
    )
    sched.reset()

    sim_steps = []
    cum_reward = 0.0
    last_obs = None
    prev_band = None

    step_idx = 0
    while not sim.done and step_idx < steps:
        bs = belief_est.snapshot()
        action = sched.select_action(bs, last_obs)
        obs = sim.step(action)

        r = compute_reward(obs, prev_band=prev_band)
        obs.reward = r
        cum_reward += r

        belief_est.update(obs)
        sched.update(obs)

        is_hit = obs.detected and obs.band_occupied_truth
        is_fa = obs.detected and not obs.band_occupied_truth
        is_miss = not obs.detected and obs.band_occupied_truth

        sim_steps.append({
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

    episode = EpisodeSummary(
        episode_id="web_sim",
        scenario_id=scenario_cfg.scenario_id,
        scheduler_name=sched.name,
        seed=seed,
        n_steps=len(sim.observations),
        observations=list(sim.observations),
        truth_log=list(sim._internal_truth_log),
        metadata={},
    )
    metrics = compute_metrics(episode)

    response = {
        "scenario_id": scenario,
        "scheduler_id": scheduler,
        "seed": seed,
        "n_steps": len(sim_steps),
        "n_bands": receiver_cfg.n_bands,
        "bands": [{"id": b.band_id, "freq_mhz": b.center_freq_mhz} for b in receiver_cfg.bands],
        "steps": sim_steps,
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
    return JSONResponse(content=clean_nan(response))


# Serve the frontend files at the root
frontend_path = ROOT_DIR.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


def main():
    parser = argparse.ArgumentParser(description="Smart Scan EW Web Dashboard Server (FastAPI)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print("=" * 70)
    print("  SMART SCAN EW -- INTERACTIVE TACTICAL COCKPIT & WEB DASHBOARD")
    print(f"  Server listening at: {url}")
    print("  Powered by FastAPI & Uvicorn")
    print("  Press Ctrl+C to stop.")
    print("=" * 70)

    if not args.no_browser:
        webbrowser.open(url)

    # Note: For production, you'd run `uvicorn serve_app:app --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
