"""Abstract base scheduler with constraint enforcement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smart_scan_ew.config import ReceiverConfig
from smart_scan_ew.types import Action, BeliefState, Observation


class BaseScheduler(ABC):
    """All schedulers inherit from this class.

    Enforces:
    - Only valid, enabled bands are selected.
    - Dwell is clamped to the nearest allowed value.
    - If the starvation guard fires, a starved band is chosen instead.

    Schedulers must implement :meth:`_select_action` and must NOT read any
    truth field from observations.
    """

    name: str = "base"

    def __init__(self, receiver_cfg: ReceiverConfig, seed: int = 0) -> None:
        self.receiver_cfg = receiver_cfg
        self.n_bands = receiver_cfg.n_bands
        self.dwell_choices = receiver_cfg.dwell_choices
        self.max_revisit = receiver_cfg.max_revisit_slots
        self._rng = np.random.default_rng(seed)
        self._enabled_bands: list[int] = [b.band_id for b in receiver_cfg.bands if b.enabled]

    # ------------------------------------------------------------------
    # Public interface (DO NOT OVERRIDE)
    # ------------------------------------------------------------------

    def select_action(self, belief: BeliefState, last_obs: Observation | None = None) -> Action:
        """Select next action with constraint enforcement.

        Parameters
        ----------
        belief:
            Current belief state snapshot.
        last_obs:
            Most recent observation (may be None at episode start).
        """
        # Starvation guard: if any band exceeds max_revisit force it first
        starved = np.where(belief.time_since_scan >= self.max_revisit)[0]
        starved_enabled = [b for b in starved if b in self._enabled_bands]
        if starved_enabled:
            band_id = int(self._rng.choice(starved_enabled))
            dwell = self.dwell_choices[0]
            return self._make_action(band_id, dwell)

        action = self._select_action(belief, last_obs)
        return self._enforce_constraints(action)

    def update(self, obs: Observation) -> None:
        """Called after each observation so the scheduler can learn."""
        self._on_observation(obs)

    # ------------------------------------------------------------------
    # Override these
    # ------------------------------------------------------------------

    @abstractmethod
    def _select_action(self, belief: BeliefState, last_obs: Observation | None) -> Action:
        """Subclass implements the scheduling policy here."""

    def _on_observation(self, obs: Observation) -> None:
        """Optional hook: called after each step with the new observation."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enforce_constraints(self, action: Action) -> Action:
        band_id = action.band_id
        if band_id not in self._enabled_bands:
            band_id = self._enabled_bands[0]
        dwell = min(
            self.dwell_choices,
            key=lambda d: abs(d - action.dwell_slots),
        )
        return Action(band_id=band_id, dwell_slots=dwell, mode=action.mode)

    def _make_action(self, band_id: int, dwell_slots: int, mode: str = "energy") -> Action:
        return self._enforce_constraints(
            Action(band_id=band_id, dwell_slots=dwell_slots, mode=mode)
        )

    def reset(self) -> None:
        """Reset any internal scheduler state for a new episode."""
        self._rng = np.random.default_rng(self._rng.integers(0, 2**31))
