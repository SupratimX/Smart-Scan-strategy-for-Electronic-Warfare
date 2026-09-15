"""Greedy occupancy scheduler."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class GreedyOccupancyScheduler(BaseScheduler):
    """Always selects the band with the highest estimated occupancy probability.

    A simple closed-loop baseline that uses the belief state but makes no
    exploration-exploitation trade-off.
    """

    name = "greedy_occupancy"

    def __init__(self, receiver_cfg: ReceiverConfig, seed: int = 0, dwell_slots: int = 1) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        enabled = np.array(self._enabled_bands)
        scores = belief.occupancy_prob[enabled]
        band_id = int(enabled[np.argmax(scores)])
        return Action(band_id=band_id, dwell_slots=self._dwell)
