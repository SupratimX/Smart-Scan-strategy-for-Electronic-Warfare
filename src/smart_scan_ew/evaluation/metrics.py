"""Full metrics suite for ES scheduler evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from smart_scan_ew.types import EmitterTruth, EpisodeSummary, Observation


# ---------------------------------------------------------------------------
# Reward helper
# ---------------------------------------------------------------------------

def compute_reward(
    obs: Observation,
    w_detection: float = 1.0,
    w_false_alarm: float = 0.5,
    w_intercept_time: float = 0.2,
    w_coverage: float = 0.1,
    w_switch: float = 0.05,
    prev_band: Optional[int] = None,
    switch_cost: float = 1.0,
) -> float:
    """Composite reward for one observation step."""
    r = 0.0
    if obs.detected and obs.band_occupied_truth:
        r += w_detection
    if obs.detected and not obs.band_occupied_truth:
        r -= w_false_alarm
    r += w_coverage * obs.dwell_slots
    if prev_band is not None and prev_band != obs.band_id:
        r -= w_switch * switch_cost
    return float(r)


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """All metrics for a single completed episode."""

    episode_id: str
    scenario_id: str
    scheduler_name: str
    seed: int
    n_steps: int
    n_bands: int

    # Detection / false alarm
    n_true_detections: int = 0
    n_false_alarms: int = 0
    n_true_opportunities: int = 0   # steps where band was occupied
    n_empty_observations: int = 0   # steps where band was empty

    pd: float = 0.0          # P(detection)
    pfa: float = 0.0         # P(false alarm)

    # Interception
    n_emitters_total: int = 0
    n_emitters_intercepted: int = 0
    interception_rate: float = 0.0

    # Intercept times (slots from emitter first activation to first detection)
    intercept_times: list[float] = field(default_factory=list)
    mean_intercept_time: float = float("nan")
    median_intercept_time: float = float("nan")
    p95_intercept_time: float = float("nan")

    # Coverage
    coverage_ratio: float = 0.0

    # Revisit intervals per band
    mean_revisit_interval: float = float("nan")
    max_revisit_interval: float = float("nan")

    # Cumulative reward
    cumulative_reward: float = 0.0
    mean_reward_per_step: float = float("nan")

    # Prediction accuracy (if occupancy estimates available)
    prediction_accuracy: float = float("nan")


def compute_metrics(episode: EpisodeSummary, reward_weights: dict | None = None) -> EpisodeMetrics:
    """Compute all metrics for a completed episode.

    This function reads both observations (policy-visible) and truth_log
    (simulator truth).  It is ONLY called by the EpisodeRunner after the
    episode ends — never during scheduling.
    """
    weights = reward_weights or {}
    obs_list = episode.observations
    truth_log = episode.truth_log

    n_bands = max((o.band_id for o in obs_list), default=0) + 1
    n_steps = len(obs_list)

    # ---- Build truth lookup: (slot, band_id) -> occupied -----------------
    # For each slot, which bands had active emitters?
    truth_by_slot: dict[int, set[int]] = {}
    emitter_first_active: dict[int, int] = {}  # emitter_id -> first active slot

    for t in truth_log:
        if t.active:
            truth_by_slot.setdefault(t.time_index, set()).add(t.band_id)
            if t.emitter_id not in emitter_first_active:
                emitter_first_active[t.emitter_id] = t.time_index

    # ---- Per-step classification ------------------------------------------
    true_detections = 0
    false_alarms = 0
    true_opportunities = 0
    empty_obs = 0
    cumulative_reward = 0.0
    prev_band = None

    band_last_visit: dict[int, int] = {}
    band_visit_intervals: dict[int, list[float]] = {b: [] for b in range(n_bands)}
    total_dwell = 0

    for obs in obs_list:
        # Truth at observation slot
        occupied = obs.band_occupied_truth  # stored by simulator

        if occupied:
            true_opportunities += 1
            if obs.detected:
                true_detections += 1
        else:
            empty_obs += 1
            if obs.detected:
                false_alarms += 1

        # Reward
        r = compute_reward(
            obs,
            w_detection=weights.get("w_detection", 1.0),
            w_false_alarm=weights.get("w_false_alarm", 0.5),
            w_intercept_time=weights.get("w_intercept_time", 0.2),
            w_coverage=weights.get("w_coverage", 0.1),
            w_switch=weights.get("w_switch", 0.05),
            prev_band=prev_band,
        )
        cumulative_reward += r
        prev_band = obs.band_id

        # Revisit intervals
        slot = obs.time_index
        b = obs.band_id
        if b in band_last_visit:
            band_visit_intervals[b].append(slot - band_last_visit[b])
        band_last_visit[b] = slot
        total_dwell += obs.dwell_slots

    # ---- Interception times -----------------------------------------------
    # Find first detection per emitter_id (using truth_ids in observations)
    emitter_first_detected: dict[int, int] = {}
    for obs in obs_list:
        if obs.detected and obs.band_occupied_truth:
            for eid in obs.emitter_ids_truth:
                if eid not in emitter_first_detected:
                    emitter_first_detected[eid] = obs.time_index

    all_emitter_ids = set(emitter_first_active.keys())
    n_emitters = len(all_emitter_ids)
    n_intercepted = len(emitter_first_detected)

    intercept_times: list[float] = []
    for eid in all_emitter_ids:
        if eid in emitter_first_detected:
            t_intercept = emitter_first_detected[eid]
            t_active = emitter_first_active[eid]
            intercept_times.append(float(t_intercept - t_active))

    # ---- Derived metrics --------------------------------------------------
    pd = true_detections / true_opportunities if true_opportunities > 0 else 0.0
    pfa = false_alarms / empty_obs if empty_obs > 0 else 0.0
    interception_rate = n_intercepted / n_emitters if n_emitters > 0 else 0.0

    n_total_band_slots = n_bands * (obs_list[-1].time_index + obs_list[-1].dwell_slots
                                    if obs_list else 1)
    coverage = total_dwell / max(n_total_band_slots, 1)

    all_intervals: list[float] = []
    for ivs in band_visit_intervals.values():
        all_intervals.extend(ivs)

    mean_revisit = float(np.mean(all_intervals)) if all_intervals else float("nan")
    max_revisit = float(np.max(all_intervals)) if all_intervals else float("nan")

    return EpisodeMetrics(
        episode_id=episode.episode_id,
        scenario_id=episode.scenario_id,
        scheduler_name=episode.scheduler_name,
        seed=episode.seed,
        n_steps=n_steps,
        n_bands=n_bands,
        n_true_detections=true_detections,
        n_false_alarms=false_alarms,
        n_true_opportunities=true_opportunities,
        n_empty_observations=empty_obs,
        pd=float(pd),
        pfa=float(pfa),
        n_emitters_total=n_emitters,
        n_emitters_intercepted=n_intercepted,
        interception_rate=float(interception_rate),
        intercept_times=intercept_times,
        mean_intercept_time=float(np.mean(intercept_times)) if intercept_times else float("nan"),
        median_intercept_time=float(np.median(intercept_times)) if intercept_times else float("nan"),
        p95_intercept_time=float(np.percentile(intercept_times, 95)) if intercept_times else float("nan"),
        coverage_ratio=float(coverage),
        mean_revisit_interval=mean_revisit,
        max_revisit_interval=max_revisit,
        cumulative_reward=float(cumulative_reward),
        mean_reward_per_step=float(cumulative_reward / n_steps) if n_steps > 0 else float("nan"),
    )


def aggregate_metrics(metrics_list: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate a list of EpisodeMetrics into mean ± std summary."""
    if not metrics_list:
        return {}

    fields = [
        "pd", "pfa", "interception_rate",
        "mean_intercept_time", "median_intercept_time", "p95_intercept_time",
        "coverage_ratio", "mean_revisit_interval", "max_revisit_interval",
        "cumulative_reward", "mean_reward_per_step",
    ]
    result: dict[str, float] = {}
    for f in fields:
        vals = [getattr(m, f) for m in metrics_list if not np.isnan(getattr(m, f))]
        if vals:
            result[f"{f}_mean"] = float(np.mean(vals))
            result[f"{f}_std"] = float(np.std(vals))
            result[f"{f}_median"] = float(np.median(vals))
        else:
            result[f"{f}_mean"] = float("nan")
    return result
