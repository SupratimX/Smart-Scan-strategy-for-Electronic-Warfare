"""Main discrete-time RF simulator.

The simulator owns the truth log and is the ONLY place that reads emitter
truth.  Schedulers receive only Observation objects (no truth fields during
execution).  The runner accesses truth after the episode ends for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from smart_scan_ew.config import ReceiverConfig, ScenarioConfig
from smart_scan_ew.environment.propagation import compute_snr_db
from smart_scan_ew.environment.scenario import Scenario
from smart_scan_ew.types import Action, EmitterTruth, Observation


@dataclass
class SimulatorState:
    """Mutable state advanced by the simulator each step."""
    current_slot: int = 0
    current_band: int = 0
    retune_slots_remaining: int = 0


class Simulator:
    """Seeded discrete-time RF environment simulator.

    Usage::

        sim = Simulator(scenario_cfg, receiver_cfg, seed=42)
        obs = sim.step(Action(band_id=3, dwell_slots=2))
        truth_log = sim.truth_log   # access ONLY after episode end
    """

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        receiver_cfg: ReceiverConfig,
        seed: int | None = None,
    ) -> None:
        self._scenario = Scenario(scenario_cfg, receiver_cfg, seed)
        self._receiver = self._scenario.receiver
        self._noise = self._scenario.noise_model
        self._rng = np.random.default_rng(
            seed if seed is not None else scenario_cfg.seed
        )
        self._state = SimulatorState()

        # Truth log — private; accessed only by EpisodeRunner after episode end
        self._truth_log: list[EmitterTruth] = []
        self._observations: list[Observation] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_bands(self) -> int:
        return self._receiver.n_bands

    @property
    def n_slots(self) -> int:
        return self._scenario.n_slots

    @property
    def current_slot(self) -> int:
        return self._state.current_slot

    @property
    def done(self) -> bool:
        return self._state.current_slot >= self._scenario.n_slots

    # Truth access — intentionally named to discourage casual use
    @property
    def _internal_truth_log(self) -> list[EmitterTruth]:
        """PRIVATE: only the EpisodeRunner may read this after episode end."""
        return self._truth_log

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    def step(self, action: Action) -> Observation:
        """Advance the simulator by one action (dwell on a band).

        Invariants enforced
        -------------------
        * Band must be valid and enabled.
        * Dwell is clamped to [min_dwell, max_dwell].
        * Retuning delay is consumed before the dwell begins.
        * Episode must not be done.

        Returns
        -------
        Observation
            Policy-visible observation (truth fields populated but should NOT
            be read by schedulers during execution).
        """
        if self.done:
            raise RuntimeError("Episode is finished; call reset() to start a new one.")

        # ---- Enforce receiver constraints ---------------------------------
        band_id = action.band_id
        if not self._receiver.valid_band(band_id):
            # Fallback: use band 0
            band_id = 0
        dwell = self._receiver.nearest_valid_dwell(action.dwell_slots)

        slot = self._state.current_slot

        # ---- Consume retuning delay if switching band --------------------
        retune_cost = 0
        if band_id != self._state.current_band:
            retune_cost = self._receiver.retune_delay
            # Advance slot counter through dead time
            self._advance_truth_log(slot, slot + retune_cost, band_id=None)
            slot += retune_cost

        # ---- Dwell: collect truth for each slot of the dwell --------------
        occupied = False
        best_power_dbm: Optional[float] = None
        active_emitter_ids: list[int] = []

        for dwell_slot in range(slot, min(slot + dwell, self._scenario.n_slots)):
            truth_at = self._scenario.emitter_truth_at(dwell_slot)
            for em in self._scenario.emitters:
                em_active, em_band = em.is_active(dwell_slot, self._rng)
                truth_entry = EmitterTruth(
                    emitter_id=em.emitter_id,
                    time_index=dwell_slot,
                    band_id=em_band,
                    active=em_active,
                    power_dbm=em.power_dbm if em_active else -np.inf,
                    mode=type(em).__name__.replace("Emitter", "").lower(),
                )
                self._truth_log.append(truth_entry)
                if em_active and em_band == band_id:
                    occupied = True
                    if best_power_dbm is None or em.power_dbm > best_power_dbm:
                        best_power_dbm = em.power_dbm
                    if em.emitter_id not in active_emitter_ids:
                        active_emitter_ids.append(em.emitter_id)

        # ---- Sample energy measurement ------------------------------------
        band_cfg = self._receiver.cfg.bands[band_id]
        if occupied and best_power_dbm is not None:
            snr_db = compute_snr_db(best_power_dbm, band_cfg.noise_floor_dbm)
            energy = self._noise.sample_energy(band_id, best_power_dbm)
        else:
            snr_db = -np.inf
            energy = self._noise.sample_energy(band_id, None)

        # ---- Detection decision -------------------------------------------
        detected, confidence = self._receiver.detection_decision(
            snr_db=snr_db,
            signal_present=occupied,
            rng=self._rng,
        )

        # ---- Build observation -------------------------------------------
        end_slot = min(slot + dwell, self._scenario.n_slots)
        obs = Observation(
            time_index=slot,
            timestamp=slot * self._receiver.slot_duration_s,
            band_id=band_id,
            dwell_slots=end_slot - slot,
            energy_statistic=float(energy),
            detected=detected,
            detection_confidence=float(confidence),
            mode=action.mode,
            # Truth fields — populated for runner use only
            band_occupied_truth=occupied,
            emitter_ids_truth=tuple(active_emitter_ids),
            snr_truth_db=float(snr_db),
        )

        # ---- Advance state -----------------------------------------------
        self._state.current_slot = end_slot
        self._state.current_band = band_id
        self._observations.append(obs)
        return obs

    def _advance_truth_log(
        self, start: int, end: int, band_id: Optional[int]
    ) -> None:
        """Log truth for dead-time slots (no observation)."""
        for slot in range(start, min(end, self._scenario.n_slots)):
            for em in self._scenario.emitters:
                em_active, em_band = em.is_active(slot, self._rng)
                self._truth_log.append(EmitterTruth(
                    emitter_id=em.emitter_id,
                    time_index=slot,
                    band_id=em_band,
                    active=em_active,
                    power_dbm=em.power_dbm if em_active else -np.inf,
                    mode=type(em).__name__.replace("Emitter", "").lower(),
                ))

    def reset(self, seed: int | None = None) -> None:
        """Reset the simulator for a new episode."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = SimulatorState()
        self._truth_log.clear()
        self._observations.clear()
