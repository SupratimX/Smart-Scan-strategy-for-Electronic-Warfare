"""Receiver model: tuning, dwell, and detection probability."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.environment.propagation import pd_from_snr


class ReceiverModel:
    """Represents the physical receiver constraints and detection statistics.

    The receiver model is query-only — it does not store scheduling state.
    All stateful decisions live in the simulator.
    """

    def __init__(self, cfg: ReceiverConfig) -> None:
        self.cfg = cfg
        self.n_bands = cfg.n_bands
        self.slot_duration_s = cfg.slot_duration_s
        self.min_dwell = cfg.min_dwell_slots
        self.max_dwell = cfg.max_dwell_slots
        self.dwell_choices = sorted(cfg.dwell_choices)
        self.retune_delay = cfg.retune_delay_slots
        self.false_alarm_rate = cfg.false_alarm_rate
        self.pd_at_snr0 = cfg.pd_at_snr0
        self.snr_threshold_db = cfg.snr_threshold_db

    # ------------------------------------------------------------------
    # Constraint helpers (used by base scheduler)
    # ------------------------------------------------------------------

    def valid_band(self, band_id: int) -> bool:
        return 0 <= band_id < self.n_bands and self.cfg.bands[band_id].enabled

    def clamp_dwell(self, dwell_slots: int) -> int:
        return max(self.min_dwell, min(self.max_dwell, dwell_slots))

    def nearest_valid_dwell(self, dwell_slots: int) -> int:
        """Return the closest allowed dwell from ``dwell_choices``."""
        return min(self.dwell_choices, key=lambda d: abs(d - dwell_slots))

    # ------------------------------------------------------------------
    # Detection probability
    # ------------------------------------------------------------------

    def probability_of_detection(self, snr_db: float) -> float:
        """P(detection | signal present at this SNR)."""
        return min(1.0, pd_from_snr(snr_db, self.snr_threshold_db, self.pd_at_snr0))

    def detection_decision(
        self,
        snr_db: float,
        signal_present: bool,
        rng: np.random.Generator,
    ) -> tuple[bool, float]:
        """Draw a detection decision and return (detected, confidence).

        Parameters
        ----------
        snr_db:
            Received SNR (dB).
        signal_present:
            True if a signal actually exists (used to select the right distribution).
        rng:
            Per-episode random generator (keeps simulation deterministic).
        """
        if signal_present:
            pd = self.probability_of_detection(snr_db)
            detected = bool(rng.random() < pd)
            confidence = pd if detected else 1.0 - pd
        else:
            # False-alarm draw
            detected = bool(rng.random() < self.false_alarm_rate)
            confidence = self.false_alarm_rate if detected else 1.0 - self.false_alarm_rate
        return detected, float(confidence)
