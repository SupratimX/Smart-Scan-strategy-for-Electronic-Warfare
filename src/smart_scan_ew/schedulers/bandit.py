"""Thompson-Sampling bandit scheduler."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class ThompsonSamplingScheduler(BaseScheduler):
    """Non-stationary Thompson-sampling bandit over frequency bands.

    Each band is an arm with a Beta(alpha, beta) posterior on its reward
    probability.  On every step the scheduler samples one value per arm
    and selects the arm with the highest sample.

    Non-stationarity is handled by applying a multiplicative discount to
    the accumulated counts every ``discount_every`` steps, which shrinks
    counts back toward the prior and forgets old observations.

    Assumption: the belief-state occupancy estimator already maintains its
    own posteriors.  This scheduler maintains *separate* bandit counts that
    track scheduler reward (detection = hit) rather than occupancy.
    """

    name = "thompson_sampling"

    def __init__(
        self,
        receiver_cfg: ReceiverConfig,
        seed: int = 0,
        dwell_slots: int = 1,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        discount: float = 0.99,
    ) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.discount = discount
        self._alpha = np.full(self.n_bands, alpha0)
        self._beta = np.full(self.n_bands, beta0)

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        # Apply discount to all counts
        self._alpha = self.alpha0 + self.discount * (self._alpha - self.alpha0)
        self._beta = self.beta0 + self.discount * (self._beta - self.beta0)

        # Thompson sample
        enabled = np.array(self._enabled_bands)
        samples = self._rng.beta(self._alpha[enabled], self._beta[enabled])
        band_id = int(enabled[np.argmax(samples)])
        return Action(band_id=band_id, dwell_slots=self._dwell)

    def _on_observation(self, obs: Observation) -> None:
        b = obs.band_id
        if obs.detected:
            self._alpha[b] += 1.0
        else:
            self._beta[b] += 1.0

    def reset(self) -> None:
        super().reset()
        self._alpha = np.full(self.n_bands, self.alpha0)
        self._beta = np.full(self.n_bands, self.beta0)
