"""Energy detector with configurable threshold."""

from __future__ import annotations

from smart_scan_ew.types import Observation


class EnergyDetector:
    """Simple energy-threshold detector.

    The detector is applied to the raw energy statistic produced by the
    :class:`~smart_scan_ew.environment.simulator.Simulator`.  The
    simulator already applies the detection decision internally; this
    class provides the post-processing feature record used by the belief
    state estimator.
    """

    def __init__(self, threshold_db: float = 5.0) -> None:
        self.threshold_db = threshold_db

    def extract_features(self, obs: Observation) -> dict[str, float]:
        """Return a feature dictionary from a raw observation."""
        above_threshold = float(obs.energy_statistic - self.threshold_db)
        return {
            "energy_db": obs.energy_statistic,
            "above_threshold_db": above_threshold,
            "detected": float(obs.detected),
            "confidence": obs.detection_confidence,
            "dwell_slots": float(obs.dwell_slots),
            "band_id": float(obs.band_id),
        }

    @staticmethod
    def compute_neyman_pearson_threshold(pfa_target: float, noise_std_db: float = 2.0) -> float:
        """Compute threshold (dB) for a target false-alarm probability.

        Assumes noise-only energy is Gaussian with mean 0 and std ``noise_std_db``.
        """
        from scipy.stats import norm  # lazy import for optional scipy

        return float(norm.ppf(1.0 - pfa_target) * noise_std_db)
