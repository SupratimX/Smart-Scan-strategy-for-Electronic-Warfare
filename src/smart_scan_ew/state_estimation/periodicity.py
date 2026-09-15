"""Periodicity estimator using autocorrelation of detection times."""

from __future__ import annotations

import numpy as np


class PeriodicityEstimator:
    """Per-band periodicity estimator.

    Stores the last ``window`` detection times and estimates the period
    using autocorrelation of the inter-detection interval sequence.

    Parameters
    ----------
    n_bands:
        Number of frequency bands.
    window:
        Maximum number of detection events to retain per band.
    min_detections:
        Minimum detections needed before reporting a period estimate.
    max_period_slots:
        Maximum period to consider (limits autocorrelation lag).
    """

    def __init__(
        self,
        n_bands: int,
        window: int = 200,
        min_detections: int = 4,
        max_period_slots: int = 100,
    ) -> None:
        self.n_bands = n_bands
        self.window = window
        self.min_detections = min_detections
        self.max_period_slots = max_period_slots

        # Ring buffers: list of detection slots per band
        self._hits: list[list[int]] = [[] for _ in range(n_bands)]

        # Cached estimates (updated lazily on hit)
        self._period: np.ndarray = np.zeros(n_bands)
        self._phase: np.ndarray = np.zeros(n_bands)
        self._confidence: np.ndarray = np.zeros(n_bands)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_hit(self, band_id: int, slot: int) -> None:
        """Record a detection event on ``band_id`` at time ``slot``."""
        buf = self._hits[band_id]
        buf.append(slot)
        if len(buf) > self.window:
            buf.pop(0)
        self._update_estimate(band_id)

    def record_miss(self, band_id: int, slot: int) -> None:  # noqa: ARG002
        """Record a miss (no update to period estimate in MVP)."""

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _update_estimate(self, band_id: int) -> None:
        hits = self._hits[band_id]
        if len(hits) < self.min_detections:
            self._confidence[band_id] = 0.0
            return

        # Inter-detection intervals
        idi = np.diff(hits)
        if len(idi) == 0:
            self._confidence[band_id] = 0.0
            return

        # Autocorrelation on integer-binned IDI histogram
        max_lag = min(self.max_period_slots, int(idi.max()) + 1)
        if max_lag < 2:
            self._confidence[band_id] = 0.0
            return

        counts, _ = np.histogram(idi, bins=np.arange(1, max_lag + 2))
        if counts.sum() == 0:
            self._confidence[band_id] = 0.0
            return

        # Full autocorrelation and find dominant lag
        ac = np.correlate(counts, counts, mode="full")
        ac = ac[len(ac) // 2 :]  # non-negative lags
        ac = ac[1:]  # skip lag-0
        if len(ac) == 0:
            self._confidence[band_id] = 0.0
            return

        best_lag = int(np.argmax(ac)) + 1
        self._period[band_id] = float(best_lag)

        # Phase: most recent hit modulo estimated period
        if best_lag > 0:
            self._phase[band_id] = float(hits[-1] % best_lag)

        # Confidence: normalised peak amplitude
        ac_norm = ac / (ac[0] + 1e-9)
        self._confidence[band_id] = float(np.clip(ac_norm[best_lag - 1], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def period(self) -> np.ndarray:
        """Estimated period (slots) per band; 0 = unknown."""
        return self._period.copy()

    @property
    def phase(self) -> np.ndarray:
        """Estimated phase offset (slots) per band."""
        return self._phase.copy()

    @property
    def confidence(self) -> np.ndarray:
        """Confidence in period estimate [0, 1] per band."""
        return self._confidence.copy()

    def predicted_active_in(self, band_id: int, current_slot: int) -> float:
        """Return the number of slots until the next predicted active window.

        Returns ``float('inf')`` if no reliable period is known.
        """
        conf = self._confidence[band_id]
        period = self._period[band_id]
        if conf < 0.3 or period < 2:
            return float("inf")
        phase = self._phase[band_id]
        # Slots until next predicted rising edge
        slots_into = (current_slot - phase) % period
        return float(period - slots_into)

    def reset(self) -> None:
        self._hits = [[] for _ in range(self.n_bands)]
        self._period[:] = 0.0
        self._phase[:] = 0.0
        self._confidence[:] = 0.0
