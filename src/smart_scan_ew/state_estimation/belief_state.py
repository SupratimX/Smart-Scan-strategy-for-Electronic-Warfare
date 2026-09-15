"""Unified belief state that aggregates all per-band estimates."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.state_estimation.occupancy import BetaBernoulliEstimator
from smart_scan_ew.state_estimation.periodicity import PeriodicityEstimator
from smart_scan_ew.types import BeliefState, Observation


class BeliefStateEstimator:
    """Maintains and updates the full belief state for the scheduler.

    This is the ONLY object schedulers interact with for state information.
    It never contains or exposes simulator truth.

    Parameters
    ----------
    n_bands:
        Number of frequency bands.
    discount:
        Occupancy estimator discount factor.
    """

    def __init__(
        self,
        n_bands: int,
        discount: float = 0.99,
        max_revisit_slots: int = 200,
    ) -> None:
        self.n_bands = n_bands
        self.max_revisit = max_revisit_slots

        self._occupancy = BetaBernoulliEstimator(n_bands, discount=discount)
        self._periodicity = PeriodicityEstimator(n_bands)

        # Timing trackers
        self._time_since_scan: np.ndarray = np.zeros(n_bands)
        self._time_since_hit: np.ndarray = np.full(n_bands, np.inf)
        self._current_slot: int = 0

    # ------------------------------------------------------------------
    # Update after each observation
    # ------------------------------------------------------------------

    def update(self, obs: Observation) -> None:
        """Incorporate the latest observation into the belief state."""
        band = obs.band_id
        slot = obs.time_index
        slots_elapsed = obs.dwell_slots

        # Advance time-since-scan for all bands
        self._time_since_scan += slots_elapsed
        self._time_since_hit += slots_elapsed

        # Reset for observed band
        self._time_since_scan[band] = 0.0
        self._current_slot = slot + slots_elapsed

        # Occupancy update
        # Uninformative miss weight: if band was unscanned for a long time,
        # a miss is less informative (emitter might have been active earlier)
        miss_weight = 1.0
        if not obs.detected:
            # Reduce miss weight proportional to time gap
            staleness = min(self._time_since_scan[band] / self.max_revisit, 1.0)
            miss_weight = 1.0 - 0.5 * staleness

        self._occupancy.update(band, obs.detected, weight=miss_weight)

        # Periodicity update
        if obs.detected:
            self._periodicity.record_hit(band, slot)
            self._time_since_hit[band] = 0.0
        else:
            self._periodicity.record_miss(band, slot)

    # ------------------------------------------------------------------
    # Query: produce BeliefState snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> BeliefState:
        """Return the current belief state (immutable snapshot)."""
        tsh = np.where(np.isinf(self._time_since_hit), self.max_revisit * 10,
                       self._time_since_hit)
        return BeliefState(
            n_bands=self.n_bands,
            occupancy_prob=self._occupancy.mean.copy(),
            occupancy_uncertainty=self._occupancy.variance.copy(),
            time_since_scan=self._time_since_scan.copy(),
            time_since_hit=tsh.copy(),
            estimated_period=self._periodicity.period,
            estimated_phase=self._periodicity.phase,
            period_confidence=self._periodicity.confidence,
        )

    def sample_occupancy(self, rng: np.random.Generator) -> np.ndarray:
        """Thompson sample: draw one P(active) per band from posteriors."""
        return self._occupancy.sample(rng)

    def predicted_slots_until_active(self, band_id: int) -> float:
        return self._periodicity.predicted_active_in(band_id, self._current_slot)

    def reset(self) -> None:
        self._occupancy.reset()
        self._periodicity.reset()
        self._time_since_scan[:] = 0.0
        self._time_since_hit[:] = np.inf
        self._current_slot = 0
