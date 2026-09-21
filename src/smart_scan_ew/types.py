"""Core data types for the Smart Scan EW scheduler.

All dataclasses are frozen (immutable) where appropriate so that they
can be passed safely across module boundaries without accidental mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """Description of one logical frequency band visible to the receiver."""

    band_id: int
    """Zero-based band index."""
    center_freq_mhz: float
    """Centre frequency in MHz."""
    bandwidth_mhz: float = 1.0
    """Occupied bandwidth in MHz."""
    noise_floor_dbm: float = -100.0
    """Receiver noise floor for this band (dBm)."""
    sensitivity_dbm: float = -90.0
    """Minimum detectable signal level (dBm)."""
    priority: float = 1.0
    """Operator-assigned priority weight (higher = more important)."""
    enabled: bool = True
    """Whether the band is available for scheduling."""
    switch_cost: float = 1.0
    """Relative cost of retuning to this band (normalised)."""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """Scheduler action: observe a band for a given dwell duration."""

    band_id: int
    """Target band to observe."""
    dwell_slots: int = 1
    """Number of time slots to dwell (≥ 1)."""
    mode: str = "energy"
    """Receiver mode string (for logging; only 'energy' used in MVP)."""


# ---------------------------------------------------------------------------
# Observation (policy-visible)
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """What the scheduler sees after one dwell on a band.

    Simulator truth fields (``*_truth``) are populated by the
    :class:`~smart_scan_ew.environment.simulator.Simulator` but are
    **never** read by any scheduler.  They exist only so the episode
    runner can compute metrics against truth at episode end.
    """

    time_index: int
    """Discrete time step at the start of the dwell."""
    timestamp: float
    """Simulation time in seconds."""
    band_id: int
    """Band that was observed."""
    dwell_slots: int
    """Actual dwell duration (time slots)."""
    energy_statistic: float
    """Normalised energy measurement (dB above noise floor)."""
    detected: bool
    """Energy-detector output (True = detection declared)."""
    detection_confidence: float
    """Estimated detector confidence in [0, 1]."""
    mode: str = "energy"

    # ---- Truth fields (kept separate; never read by schedulers) --------
    band_occupied_truth: bool = False
    """Simulator truth: was any emitter active in this band during the dwell?"""
    emitter_ids_truth: tuple[int, ...] = field(default_factory=tuple)
    """Simulator truth: IDs of active emitters."""
    snr_truth_db: float = -np.inf
    """Simulator truth: signal-to-noise ratio experienced."""
    reward: float = 0.0
    """Immediate reward assigned by the runner after the observation."""


# ---------------------------------------------------------------------------
# Emitter truth record (used only inside simulator and by runner)
# ---------------------------------------------------------------------------


@dataclass
class EmitterTruth:
    """Ground-truth state of one emitter at one time step."""

    emitter_id: int
    time_index: int
    band_id: int
    active: bool
    power_dbm: float
    mode: str  # "fixed" | "periodic" | "burst" | "agile"


# ---------------------------------------------------------------------------
# Belief state (policy-visible summary)
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """Per-band belief-state summary exposed to the scheduler."""

    n_bands: int
    """Number of bands."""
    occupancy_prob: np.ndarray
    """Estimated P(band active) for each band, shape (n_bands,)."""
    occupancy_uncertainty: np.ndarray
    """Variance of the occupancy estimate, shape (n_bands,)."""
    time_since_scan: np.ndarray
    """Slots since each band was last observed, shape (n_bands,)."""
    time_since_hit: np.ndarray
    """Slots since each band last had a true hit, shape (n_bands,)."""
    estimated_period: np.ndarray
    """Estimated periodicity (in slots) per band; 0 = unknown."""
    estimated_phase: np.ndarray
    """Estimated phase offset (in slots) per band; 0 = unknown."""
    period_confidence: np.ndarray
    """Confidence in the period estimate [0, 1], shape (n_bands,)."""

    def to_feature_vector(self) -> np.ndarray:
        """Flatten belief state into a 1-D numeric feature vector."""
        return np.concatenate(
            [
                self.occupancy_prob,
                self.occupancy_uncertainty,
                self.time_since_scan,
                self.time_since_hit,
                self.estimated_period,
                self.estimated_phase,
                self.period_confidence,
            ]
        )


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSummary:
    """High-level summary of a completed simulation episode."""

    episode_id: str
    scenario_id: str
    scheduler_name: str
    seed: int
    n_steps: int
    observations: list[Observation] = field(default_factory=list)
    truth_log: list[EmitterTruth] = field(default_factory=list)
    """Complete emitter truth — only used by runner/metrics, never by schedulers."""
    metadata: dict[str, Any] = field(default_factory=dict)
