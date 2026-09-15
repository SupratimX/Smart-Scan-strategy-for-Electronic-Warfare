"""Propagation and SNR model."""

from __future__ import annotations

import numpy as np


def compute_snr_db(
    power_dbm: float,
    noise_floor_dbm: float,
    path_loss_db: float = 0.0,
) -> float:
    """Return signal-to-noise ratio in dB.

    Parameters
    ----------
    power_dbm:
        Emitter transmit power (dBm).
    noise_floor_dbm:
        Receiver noise floor for the band (dBm).
    path_loss_db:
        Total propagation loss (dB).  Defaults to 0 (co-located / short range).
    """
    received_dbm = power_dbm - path_loss_db
    return float(received_dbm - noise_floor_dbm)


def pd_from_snr(snr_db: float, snr_threshold_db: float, pd_at_threshold: float = 0.9) -> float:
    """Sigmoid-like P(detection) as a function of SNR.

    At ``snr_threshold_db`` the probability equals ``pd_at_threshold``.
    Decays to near-zero below threshold and saturates near 1 above it.
    """
    delta = snr_db - snr_threshold_db
    # Logistic function centred at threshold, slope ≈ 0.5
    return float(1.0 / (1.0 + np.exp(-0.5 * delta)) * pd_at_threshold / 0.5)
