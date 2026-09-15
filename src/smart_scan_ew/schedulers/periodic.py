"""Periodicity-aware scheduler: phase prediction + Thompson sampling + forced exploration."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class PeriodicityAwareScheduler(BaseScheduler):
    """Hybrid scheduler that combines three policies:

    1. **Periodicity exploitation** — if a band has a reliable period
       estimate, prioritise it when the predicted active window is near.
    2. **Thompson sampling** — bandit arm selection when no band has a
       reliable period.
    3. **Forced exploration** — every ``explore_every`` steps, pick a
       random low-priority band to prevent starvation and model drift.

    Parameters
    ----------
    phase_window_slots:
        Bands are considered "due" if predicted_slots_until_active ≤ this.
    explore_fraction:
        Probability of a random exploration step regardless of predictions.
    min_period_confidence:
        Minimum confidence required to act on a period prediction.
    """

    name = "periodicity_aware"

    def __init__(
        self,
        receiver_cfg: ReceiverConfig,
        seed: int = 0,
        dwell_slots: int = 1,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        discount: float = 0.99,
        phase_window_slots: float = 3.0,
        explore_fraction: float = 0.10,
        min_period_confidence: float = 0.30,
    ) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.discount = discount
        self.phase_window = phase_window_slots
        self.explore_frac = explore_fraction
        self.min_conf = min_period_confidence

        self._alpha = np.full(self.n_bands, alpha0)
        self._beta = np.full(self.n_bands, beta0)
        self._consecutive_misses = np.zeros(self.n_bands, dtype=int)

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        # ---- Forced exploration ----------------------------------------
        if self._rng.random() < self.explore_frac:
            band_id = int(self._rng.choice(self._enabled_bands))
            return Action(band_id=band_id, dwell_slots=self._dwell)

        # ---- Periodic exploitation -------------------------------------
        enabled = np.array(self._enabled_bands)
        best_periodic: int | None = None
        best_urgency: float = self.phase_window + 1

        for b in enabled:
            conf = belief.period_confidence[b]
            period = belief.estimated_period[b]
            if conf < self.min_conf or period < 2:
                continue
            slots_until = belief.estimated_period[b] - (
                (belief.time_since_scan[b]) % max(belief.estimated_period[b], 1)
            )
            # Increase window if consecutive misses suggest jitter
            window = self.phase_window * (1 + 0.5 * min(self._consecutive_misses[b], 4))
            if slots_until <= window and slots_until < best_urgency:
                best_urgency = slots_until
                best_periodic = int(b)

        if best_periodic is not None:
            return Action(band_id=best_periodic, dwell_slots=self._dwell)

        # ---- Thompson sampling fallback --------------------------------
        self._alpha = self.alpha0 + self.discount * (self._alpha - self.alpha0)
        self._beta = self.beta0 + self.discount * (self._beta - self.beta0)
        samples = self._rng.beta(self._alpha[enabled], self._beta[enabled])
        band_id = int(enabled[np.argmax(samples)])
        return Action(band_id=band_id, dwell_slots=self._dwell)

    def _on_observation(self, obs: Observation) -> None:
        b = obs.band_id
        if obs.detected:
            self._alpha[b] += 1.0
            self._consecutive_misses[b] = 0
        else:
            self._beta[b] += 1.0
            self._consecutive_misses[b] += 1

    def reset(self) -> None:
        super().reset()
        self._alpha = np.full(self.n_bands, self.alpha0)
        self._beta = np.full(self.n_bands, self.beta0)
        self._consecutive_misses = np.zeros(self.n_bands, dtype=int)
