"""Beta-Bernoulli occupancy estimator (per-band)."""

from __future__ import annotations

import numpy as np


class BetaBernoulliEstimator:
    """Maintains a Beta posterior for P(band active) for each band.

    The prior is Beta(alpha0, beta0).  After each observation the
    posterior is updated with a discounted count to handle non-stationarity.

    Parameters
    ----------
    n_bands:
        Number of frequency bands.
    alpha0, beta0:
        Prior Beta parameters (default: uniform prior).
    discount:
        Multiplicative discount applied to counts each step (0 < discount ≤ 1).
        Values < 1 implement a sliding-window effect.
    """

    def __init__(
        self,
        n_bands: int,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        discount: float = 0.99,
    ) -> None:
        self.n_bands = n_bands
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.discount = discount
        self.alpha = np.full(n_bands, alpha0)
        self.beta = np.full(n_bands, beta0)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, band_id: int, detected: bool, weight: float = 1.0) -> None:
        """Update the posterior for ``band_id``.

        Parameters
        ----------
        band_id:
            Which band was observed.
        detected:
            Whether the detector fired.
        weight:
            Evidence weight in [0, 1].  Use < 1 for uninformative misses.
        """
        # Apply discount to all bands (sliding-window effect)
        self.alpha = self.alpha0 + self.discount * (self.alpha - self.alpha0)
        self.beta = self.beta0 + self.discount * (self.beta - self.beta0)
        # Likelihood update for the observed band
        if detected:
            self.alpha[band_id] += weight
        else:
            self.beta[band_id] += weight

    def update_batch(self, band_id: int, n_hits: int, n_misses: int) -> None:
        """Bulk update (used for replay-based initialisation)."""
        self.alpha[band_id] += n_hits
        self.beta[band_id] += n_misses

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """Posterior mean P(band active), shape (n_bands,)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> np.ndarray:
        """Posterior variance, shape (n_bands,)."""
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one occupancy probability per band from the posterior."""
        return rng.beta(self.alpha, self.beta)

    def reset(self) -> None:
        self.alpha = np.full(self.n_bands, self.alpha0)
        self.beta = np.full(self.n_bands, self.beta0)
