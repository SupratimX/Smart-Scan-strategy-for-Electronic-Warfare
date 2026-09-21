"""Random scan scheduler."""

from __future__ import annotations

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class RandomScanScheduler(BaseScheduler):
    """Selects bands uniformly at random from enabled bands."""

    name = "random_scan"

    def __init__(self, receiver_cfg: ReceiverConfig, seed: int = 0, dwell_slots: int = 1) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        band_id = int(self._rng.choice(self._enabled_bands))
        return Action(band_id=band_id, dwell_slots=self._dwell)
