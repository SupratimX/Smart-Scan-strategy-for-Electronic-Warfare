"""JSONL event logger for storing observation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smart_scan_ew.evaluation.metrics import EpisodeMetrics
from smart_scan_ew.types import EpisodeSummary, Observation


def _obs_to_dict(obs: Observation) -> dict[str, Any]:
    return {
        "time_index": obs.time_index,
        "timestamp": obs.timestamp,
        "band_id": obs.band_id,
        "dwell_slots": obs.dwell_slots,
        "energy_statistic": obs.energy_statistic,
        "detected": obs.detected,
        "detection_confidence": obs.detection_confidence,
        "mode": obs.mode,
        "reward": obs.reward,
        # Truth fields stored for offline analysis only
        "band_occupied_truth": obs.band_occupied_truth,
        "emitter_ids_truth": list(obs.emitter_ids_truth),
        "snr_truth_db": obs.snr_truth_db,
    }


def save_episode_jsonl(episode: EpisodeSummary, output_dir: str | Path) -> Path:
    """Write one JSONL file per episode with all observations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{episode.episode_id}_{episode.scheduler_name}.jsonl"
    with fname.open("w", encoding="utf-8") as fh:
        for obs in episode.observations:
            fh.write(json.dumps(_obs_to_dict(obs)) + "\n")
    return fname


def save_metrics_json(metrics: EpisodeMetrics, output_dir: str | Path) -> Path:
    """Write episode metrics as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"metrics_{metrics.episode_id}_{metrics.scheduler_name}.json"
    data = {k: v for k, v in metrics.__dict__.items()}
    fname.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return fname


def save_all_metrics_json(all_metrics: list[EpisodeMetrics], output_path: str | Path) -> Path:
    """Write all metrics to a single JSON array file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{k: v for k, v in m.__dict__.items()} for m in all_metrics]
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output_path
