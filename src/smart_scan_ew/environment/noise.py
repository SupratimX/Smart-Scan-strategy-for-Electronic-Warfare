"""Noise model for the RF environment."""

from __future__ import annotations

import numpy as np


class NoiseModel:
    """Adds band-dependent noise and false-alarm interference.

    Parameters
    ----------
    n_bands:
        Number of frequency bands.
    base_noise_floor_dbm:
        Default noise floor (dBm) applied to all bands.
    noise_std_db:
        Standard deviation of per-sample Gaussian noise (dB).
    interference_bands:
        Band IDs with an additional +10 dB noise uplift (co-channel interference).
    false_alarm_rate:
        Probability that an empty band triggers a false alarm per dwell.
    rng:
        Seeded NumPy random generator.
    """

    INTERFERENCE_UPLIFT_DB: float = 10.0

    def __init__(
        self,
        n_bands: int,
        base_noise_floor_dbm: float = -100.0,
        noise_std_db: float = 2.0,
        interference_bands: list[int] | None = None,
        false_alarm_rate: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_bands = n_bands
        self.base_noise_floor_dbm = base_noise_floor_dbm
        self.noise_std_db = noise_std_db
        self.interference_bands: set[int] = set(interference_bands or [])
        self.false_alarm_rate = false_alarm_rate
        self._rng = rng if rng is not None else np.random.default_rng(0)

        # Pre-compute per-band noise floors
        self.band_noise_floor = np.full(n_bands, base_noise_floor_dbm)
        for b in self.interference_bands:
            if 0 <= b < n_bands:
                self.band_noise_floor[b] += self.INTERFERENCE_UPLIFT_DB

    def sample_energy(self, band_id: int, signal_power_dbm: float | None) -> float:
        """Sample an energy measurement for a given band.

        Parameters
        ----------
        band_id:
            Which band is being measured.
        signal_power_dbm:
            True signal power (dBm) if a signal is present, or None if absent.

        Returns
        -------
        float
            Observed energy in dB above the local noise floor.
        """
        noise_floor = self.band_noise_floor[band_id]
        additive_noise = self._rng.normal(0.0, self.noise_std_db)

        if signal_power_dbm is not None:
            # Signal + noise
            signal_above = signal_power_dbm - noise_floor
            raw_energy = signal_above + additive_noise
        else:
            # Noise only (expected 0 dB above floor)
            raw_energy = additive_noise

        return float(raw_energy)

    def false_alarm_this_dwell(self, band_id: int) -> bool:  # noqa: ARG002
        """Return True if a false alarm occurs on an empty dwell."""
        return bool(self._rng.random() < self.false_alarm_rate)
