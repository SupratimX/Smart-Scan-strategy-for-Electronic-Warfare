"""Scenario: wires together emitters and receiver for one episode."""

from __future__ import annotations

import numpy as np

from smart_scan_ew.config import ReceiverConfig, ScenarioConfig
from smart_scan_ew.environment.emitter import BaseEmitter, make_emitter
from smart_scan_ew.environment.noise import NoiseModel
from smart_scan_ew.environment.receiver import ReceiverModel


class Scenario:
    """Fully initialised scenario ready for the simulator to step through."""

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        receiver_cfg: ReceiverConfig,
        seed: int | None = None,
    ) -> None:
        self.scenario_cfg = scenario_cfg
        self.receiver_cfg = receiver_cfg
        self.seed = seed if seed is not None else scenario_cfg.seed
        self.rng = np.random.default_rng(self.seed)

        self.receiver = ReceiverModel(receiver_cfg)
        self.noise_model = NoiseModel(
            n_bands=receiver_cfg.n_bands,
            base_noise_floor_dbm=receiver_cfg.bands[0].noise_floor_dbm
            if receiver_cfg.bands else -100.0,
            noise_std_db=scenario_cfg.noise_std_db,
            interference_bands=scenario_cfg.interference_bands,
            false_alarm_rate=receiver_cfg.false_alarm_rate,
            rng=np.random.default_rng(self.seed + 1),
        )
        self.emitters: list[BaseEmitter] = [
            make_emitter(e, np.random.default_rng(self.seed + 100 + i))
            for i, e in enumerate(scenario_cfg.emitters)
        ]
        self.n_slots = scenario_cfg.n_slots
        self.n_bands = receiver_cfg.n_bands

    def emitter_truth_at(self, slot: int) -> dict[int, tuple[bool, float]]:
        """Return {band_id: (active, power_dbm)} for all emitters at ``slot``.

        Only the first active emitter per band is recorded (highest power wins
        in a future version; MVP takes first-found).
        """
        truth: dict[int, tuple[bool, float]] = {}
        rng_slot = np.random.default_rng(self.seed + slot)  # reproducible per slot
        for em in self.emitters:
            active, band_id = em.is_active(slot, rng_slot)
            if active and band_id not in truth:
                truth[band_id] = (True, em.power_dbm)
        return truth
