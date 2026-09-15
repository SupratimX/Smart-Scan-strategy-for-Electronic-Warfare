"""Emitter truth models.

Each emitter class determines whether it is *active* on a given band at a
given time slot.  These classes are **private to the simulator** and must
never be imported by any scheduler.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from smart_scan_ew.config import (
    AgileEmitterConfig,
    BurstEmitterConfig,
    EmitterConfig,
    FixedEmitterConfig,
    PeriodicEmitterConfig,
)


class BaseEmitter(ABC):
    """Abstract base for all emitter models."""

    def __init__(self, emitter_id: int, power_dbm: float,
                 active_from: int, active_until: Optional[int]) -> None:
        self.emitter_id = emitter_id
        self.power_dbm = power_dbm
        self.active_from = active_from
        self.active_until = active_until  # None = open-ended

    def _in_lifetime(self, slot: int) -> bool:
        if slot < self.active_from:
            return False
        if self.active_until is not None and slot >= self.active_until:
            return False
        return True

    @abstractmethod
    def is_active(self, slot: int, rng: np.random.Generator) -> tuple[bool, int]:
        """Return (active, band_id) at this time slot."""


class FixedEmitter(BaseEmitter):
    """Always-on emitter on a single band."""

    def __init__(self, cfg: FixedEmitterConfig) -> None:
        super().__init__(cfg.emitter_id, cfg.power_dbm,
                         cfg.active_from_slot, cfg.active_until_slot)
        self.band_id = cfg.band_id

    def is_active(self, slot: int, rng: np.random.Generator) -> tuple[bool, int]:
        return self._in_lifetime(slot), self.band_id


class PeriodicEmitter(BaseEmitter):
    """Periodic emitter: on for ``duty_cycle * period`` slots, off otherwise."""

    def __init__(self, cfg: PeriodicEmitterConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.emitter_id, cfg.power_dbm,
                         cfg.active_from_slot, cfg.active_until_slot)
        self.band_id = cfg.band_id
        self.period = cfg.period_slots
        self.duty = cfg.duty_cycle
        self.phase = cfg.phase_offset_slots
        self.jitter_std = cfg.jitter_slots
        # Pre-compute per-period jitter offsets for determinism
        self._rng = rng
        self._on_slots = max(1, round(self.duty * self.period))

    def is_active(self, slot: int, rng: np.random.Generator) -> tuple[bool, int]:
        if not self._in_lifetime(slot):
            return False, self.band_id
        # Position within the current period (with phase offset)
        phase_pos = (slot - self.phase) % self.period
        # Simple deterministic jitter: shift on-window by a per-period offset
        period_idx = (slot - self.phase) // self.period
        jitter = 0
        if self.jitter_std > 0:
            # Deterministic per-period jitter using a seeded hash
            seed_val = int(hash((self.emitter_id, period_idx)) & 0xFFFF_FFFF)
            local_rng = np.random.default_rng(seed_val)
            jitter = int(round(local_rng.normal(0, self.jitter_std)))
        on_start = jitter % self.period
        on_end = on_start + self._on_slots
        active = (on_start <= phase_pos < on_end) or (
            on_end > self.period and phase_pos < on_end - self.period
        )
        return active, self.band_id


class BurstEmitter(BaseEmitter):
    """Burst emitter: short bursts separated by long gaps."""

    def __init__(self, cfg: BurstEmitterConfig) -> None:
        super().__init__(cfg.emitter_id, cfg.power_dbm,
                         cfg.active_from_slot, cfg.active_until_slot)
        self.band_id = cfg.band_id
        self.burst_dur = cfg.burst_duration_slots
        self.gap = cfg.inter_burst_slots
        self.cycle = self.burst_dur + self.gap

    def is_active(self, slot: int, rng: np.random.Generator) -> tuple[bool, int]:
        if not self._in_lifetime(slot):
            return False, self.band_id
        phase = slot % self.cycle
        return phase < self.burst_dur, self.band_id


class AgileEmitter(BaseEmitter):
    """Frequency-agile emitter that hops between bands."""

    def __init__(self, cfg: AgileEmitterConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.emitter_id, cfg.power_dbm,
                         cfg.active_from_slot, cfg.active_until_slot)
        self.band_ids = list(cfg.band_ids)
        self.hop_interval = cfg.hop_interval_slots
        self.pattern = cfg.hop_pattern
        self._n = len(self.band_ids)
        # Pre-sample hop sequence for reproducibility
        if self.pattern == "random":
            self._hop_seq: list[int] = rng.choice(
                self.band_ids, size=100_000, replace=True
            ).tolist()
        elif self.pattern == "sequential":
            repeats = 100_000 // self._n + 1
            self._hop_seq = (self.band_ids * repeats)[:100_000]
        else:  # pseudo_random
            self._hop_seq = rng.permutation(self.band_ids * (100_000 // self._n + 1)).tolist()[:100_000]

    def _current_band(self, slot: int) -> int:
        hop_index = (slot // self.hop_interval) % len(self._hop_seq)
        return self._hop_seq[hop_index]

    def is_active(self, slot: int, rng: np.random.Generator) -> tuple[bool, int]:
        if not self._in_lifetime(slot):
            return False, self._current_band(slot)
        return True, self._current_band(slot)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_emitter(cfg: EmitterConfig, rng: np.random.Generator) -> BaseEmitter:
    """Instantiate the correct emitter class from a config object."""
    if isinstance(cfg, FixedEmitterConfig):
        return FixedEmitter(cfg)
    if isinstance(cfg, PeriodicEmitterConfig):
        return PeriodicEmitter(cfg, rng)
    if isinstance(cfg, BurstEmitterConfig):
        return BurstEmitter(cfg)
    if isinstance(cfg, AgileEmitterConfig):
        return AgileEmitter(cfg, rng)
    raise TypeError(f"Unknown emitter config type: {type(cfg)}")
