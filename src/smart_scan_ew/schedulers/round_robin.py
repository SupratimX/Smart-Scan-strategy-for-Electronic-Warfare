"""Round-robin scheduler with starvation prevention."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class RoundRobinScheduler(BaseScheduler):
    """Circular scan that enforces a maximum revisit interval.

    The starvation guard in BaseScheduler covers the hard constraint;
    this scheduler adds a soft priority boost for long-unvisited bands.
    """

    name = "round_robin"

    def __init__(self, receiver_cfg: ReceiverConfig, seed: int = 0, dwell_slots: int = 1) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots
        self._idx = 0

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        # Pick the band that has been unvisited the longest
        tss = belief.time_since_scan
        enabled = np.array(self._enabled_bands)
        band_id = int(enabled[np.argmax(tss[enabled])])
        return Action(band_id=band_id, dwell_slots=self._dwell)
