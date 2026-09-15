"""Uniform sweep scheduler."""

from __future__ import annotations

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.schedulers.base import BaseScheduler
from smart_scan_ew.types import Action, BeliefState, Observation


class UniformSweepScheduler(BaseScheduler):
    """Sequentially visits all enabled bands with a fixed dwell.

    This is the canonical open-loop baseline.
    """

    name = "uniform_sweep"

    def __init__(self, receiver_cfg: ReceiverConfig, seed: int = 0, dwell_slots: int = 1) -> None:
        super().__init__(receiver_cfg, seed)
        self._dwell = dwell_slots
        self._idx = 0

    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        band_id = self._enabled_bands[self._idx % len(self._enabled_bands)]
        self._idx += 1
        return Action(band_id=band_id, dwell_slots=self._dwell)

    def reset(self) -> None:
        super().reset()
        self._idx = 0
