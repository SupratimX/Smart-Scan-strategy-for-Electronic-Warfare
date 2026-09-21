"""Configuration loading and validation using Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Band configuration
# ---------------------------------------------------------------------------


class BandConfig(BaseModel):
    band_id: int
    center_freq_mhz: float
    bandwidth_mhz: float = 1.0
    noise_floor_dbm: float = -100.0
    sensitivity_dbm: float = -90.0
    priority: float = 1.0
    enabled: bool = True
    switch_cost: float = 1.0


# ---------------------------------------------------------------------------
# Receiver configuration
# ---------------------------------------------------------------------------


class ReceiverConfig(BaseModel):
    n_bands: int = 32
    """Number of logical frequency bands."""
    slot_duration_s: float = 0.001
    """Duration of one time slot in seconds."""
    min_dwell_slots: int = 1
    """Minimum dwell in slots."""
    max_dwell_slots: int = 4
    """Maximum dwell in slots."""
    dwell_choices: list[int] = Field(default_factory=lambda: [1, 2, 4])
    """Allowed dwell durations in slots."""
    retune_delay_slots: int = 1
    """Dead-time slots incurred when switching between bands."""
    false_alarm_rate: float = 0.05
    """Per-dwell P(FA) when band is empty."""
    pd_at_snr0: float = 0.9
    """P(detection) when SNR equals the threshold."""
    snr_threshold_db: float = 10.0
    """Energy detector SNR threshold in dB."""
    max_revisit_slots: int = 200
    """Maximum slots before a band must be visited (starvation guard)."""
    bands: list[BandConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def generate_bands_if_empty(self) -> ReceiverConfig:
        if not self.bands:
            start_mhz = 100.0
            step_mhz = 10.0
            self.bands = [
                BandConfig(
                    band_id=i,
                    center_freq_mhz=start_mhz + i * step_mhz,
                )
                for i in range(self.n_bands)
            ]
        return self


# ---------------------------------------------------------------------------
# Emitter configurations
# ---------------------------------------------------------------------------


class FixedEmitterConfig(BaseModel):
    emitter_id: int
    kind: Literal["fixed"] = "fixed"
    band_id: int
    power_dbm: float = -60.0
    active_from_slot: int = 0
    active_until_slot: int | None = None  # None = until end of episode


class PeriodicEmitterConfig(BaseModel):
    emitter_id: int
    kind: Literal["periodic"] = "periodic"
    band_id: int
    power_dbm: float = -60.0
    period_slots: int = 20
    """Transmission period in slots."""
    duty_cycle: float = 0.3
    """Fraction of period the emitter is on."""
    phase_offset_slots: int = 0
    """Initial phase offset in slots."""
    jitter_slots: float = 0.0
    """Std. dev. of timing jitter in slots."""
    active_from_slot: int = 0
    active_until_slot: int | None = None


class BurstEmitterConfig(BaseModel):
    emitter_id: int
    kind: Literal["burst"] = "burst"
    band_id: int
    power_dbm: float = -60.0
    burst_duration_slots: int = 3
    """Length of each burst."""
    inter_burst_slots: int = 30
    """Gap between bursts."""
    active_from_slot: int = 0
    active_until_slot: int | None = None


class AgileEmitterConfig(BaseModel):
    emitter_id: int
    kind: Literal["agile"] = "agile"
    band_ids: list[int]
    """Set of bands the emitter hops among."""
    power_dbm: float = -60.0
    hop_interval_slots: int = 10
    """Slots between frequency hops."""
    hop_pattern: Literal["random", "sequential", "pseudo_random"] = "random"
    active_from_slot: int = 0
    active_until_slot: int | None = None


class SpatiallyScanningEmitterConfig(BaseModel):
    emitter_id: int
    kind: Literal["spatial_scan"] = "spatial_scan"
    band_id: int
    """Frequency band the emitter transmits on."""
    power_dbm: float = -60.0
    scan_period_slots: int = 60
    """Slots for one full antenna rotation (e.g. 60 slots = one scan cycle)."""
    beam_dwell_fraction: float = 0.05
    """Fraction of the scan period the beam is pointed at our receiver.

    A typical surveillance radar has a 2-4 % dwell fraction.
    E.g. scan_period_slots=60, beam_dwell_fraction=0.05 → beam on for 3 slots.
    """
    phase_offset_slots: int = 0
    """Initial phase offset (slot within the scan cycle when beam first hits)."""
    active_from_slot: int = 0
    active_until_slot: int | None = None


EmitterConfig = (
    FixedEmitterConfig
    | PeriodicEmitterConfig
    | BurstEmitterConfig
    | AgileEmitterConfig
    | SpatiallyScanningEmitterConfig
)


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    scenario_id: str
    description: str = ""
    seed: int = 42
    n_slots: int = 1000
    """Episode length in time slots."""
    emitters: list[Any] = Field(default_factory=list)
    """List of emitter config dicts (parsed below)."""
    noise_std_db: float = 2.0
    """Std. dev. of channel noise added to energy measurements (dB)."""
    interference_bands: list[int] = Field(default_factory=list)
    """Bands with elevated noise floor (+10 dB)."""

    @model_validator(mode="after")
    def parse_emitters(self) -> ScenarioConfig:
        parsed: list[Any] = []
        for e in self.emitters:
            kind = e.get("kind", "fixed") if isinstance(e, dict) else e.kind
            if kind == "fixed":
                parsed.append(FixedEmitterConfig(**e) if isinstance(e, dict) else e)
            elif kind == "periodic":
                parsed.append(PeriodicEmitterConfig(**e) if isinstance(e, dict) else e)
            elif kind == "burst":
                parsed.append(BurstEmitterConfig(**e) if isinstance(e, dict) else e)
            elif kind == "agile":
                parsed.append(AgileEmitterConfig(**e) if isinstance(e, dict) else e)
            elif kind == "spatial_scan":
                parsed.append(SpatiallyScanningEmitterConfig(**e) if isinstance(e, dict) else e)
            else:
                raise ValueError(f"Unknown emitter kind: {kind}")
        self.emitters = parsed
        return self


# ---------------------------------------------------------------------------
# Benchmark suite configuration
# ---------------------------------------------------------------------------


class BenchmarkSuiteConfig(BaseModel):
    suite_id: str = "default"
    description: str = ""
    scenario_configs: list[str] = Field(default_factory=list)
    """Paths to scenario YAML files (relative to project root)."""
    scheduler_names: list[str] = Field(
        default_factory=lambda: [
            "uniform_sweep",
            "random_scan",
            "round_robin",
            "greedy_occupancy",
            "thompson_sampling",
            "periodicity_aware",
        ]
    )
    n_seeds: int = 5
    """Number of independent seeds per scenario-scheduler combination."""
    base_seed: int = 0
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "w_detection": 1.0,
            "w_false_alarm": 0.5,
            "w_intercept_time": 0.2,
            "w_coverage": 0.1,
            "w_switch": 0.05,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_receiver_config(path: str | Path) -> ReceiverConfig:
    return ReceiverConfig(**load_yaml(path))


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    return ScenarioConfig(**load_yaml(path))


def load_benchmark_suite(path: str | Path) -> BenchmarkSuiteConfig:
    return BenchmarkSuiteConfig(**load_yaml(path))
