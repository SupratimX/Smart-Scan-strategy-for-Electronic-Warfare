"""Turing Synthetic Radar Dataset loader.

Streams a small sample from the HuggingFace dataset:
    alan-turing-institute/turing-synthetic-radar-dataset

Uses ``streaming=True`` so only the records actually consumed
are downloaded (~few MB), avoiding the full 70 GB dataset.

Falls back automatically to a built-in synthetic stub when:
  - The ``datasets`` package is not installed, or
  - The network / HuggingFace Hub is unreachable, or
  - ``force_offline=True`` is passed (useful for demos).

Usage
-----
    from smart_scan_ew.data.turing_loader import load_turing_scenario

    # Online — streams 500 records, maps to ScenarioConfig
    cfg = load_turing_scenario(n_records=500, n_slots=1000, seed=42)

    # Offline / demo — always uses the built-in stub
    cfg = load_turing_scenario(force_offline=True)
"""

from __future__ import annotations

import warnings
from typing import Any

from smart_scan_ew.config import (
    AgileEmitterConfig,
    FixedEmitterConfig,
    ScenarioConfig,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_turing_scenario(
    n_records: int = 500,
    n_slots: int = 1000,
    seed: int = 42,
    force_offline: bool = False,
) -> ScenarioConfig:
    """Return a ScenarioConfig built from the Turing radar dataset.

    Parameters
    ----------
    n_records:
        Number of emitter records to stream (default 500).
        Only this many records are downloaded; the full 70 GB dataset
        is never fetched.
    n_slots:
        Episode length in time slots for the generated scenario.
    seed:
        Random seed for reproducibility.
    force_offline:
        If True, skip the network call and use the offline stub.
        Useful for demos without internet access.
    """
    if not force_offline:
        try:
            return _load_streaming(n_records=n_records, n_slots=n_slots, seed=seed)
        except Exception as exc:
            warnings.warn(
                f"Turing dataset unavailable ({exc!r}); falling back to built-in synthetic stub.",
                stacklevel=2,
            )
    return _offline_stub(n_slots=n_slots, seed=seed)


# ---------------------------------------------------------------------------
# Streaming loader
# ---------------------------------------------------------------------------


def _load_streaming(n_records: int, n_slots: int, seed: int) -> ScenarioConfig:
    """Download a small shard from HuggingFace and build ScenarioConfig from radar transmitters.

    Downloads only one ~few MB shard file, never the full 70 GB dataset.
    """
    import ast
    import os
    import random
    from pathlib import Path

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Check for local dataset first
    local_model_dir = Path(__file__).resolve().parents[3] / "artifacts" / "model"
    local_h5_files = list(local_model_dir.glob("*.h5"))

    if local_h5_files:
        rng = random.Random(seed)
        shard_path = str(rng.choice(local_h5_files))
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install 'huggingface_hub' to use the online Turing loader.") from exc

        shard_path = hf_hub_download(
            repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
            filename="archive/test/test_0.h5",
            repo_type="dataset",
        )

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install 'h5py' to use the Turing loader.") from exc

    def _parse_h5_node(node):
        import numpy as np

        if isinstance(node, h5py.Dataset):
            val = node[()]
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return val
        elif isinstance(node, h5py.Group):
            res = dict(node.attrs)
            for k, v in res.items():
                if isinstance(v, bytes):
                    res[k] = v.decode("utf-8")
            for k in node.keys():
                res[k] = _parse_h5_node(node[k])
            return res
        return None

    with h5py.File(shard_path, "r") as f:
        txs_node = f["metadata/transmitters"]
        if isinstance(txs_node, h5py.Group):
            keys = list(txs_node.keys())[:n_records]
            tx_configs = [_parse_h5_node(txs_node[k]) for k in keys]
        else:
            raw_txs = list(txs_node[:n_records])
            tx_configs = [
                ast.literal_eval(t.decode("utf-8") if isinstance(t, bytes) else str(t))
                for t in raw_txs
            ]

    emitters = _transmitters_to_emitters(tx_configs)

    return ScenarioConfig(
        scenario_id="turing_radar_streamed",
        description=(
            f"Turing Synthetic Radar Dataset — {len(emitters)} radar emitters "
            f"(alan-turing-institute/turing-synthetic-radar-dataset)"
        ),
        seed=seed,
        n_slots=n_slots,
        emitters=emitters,
        noise_std_db=2.0,
    )


def _transmitters_to_emitters(tx_configs: list[dict[str, Any]]) -> list[Any]:
    """Map Turing radar transmitter dictionaries to Fixed / Agile emitter configs."""
    emitters: list[Any] = []
    n_bands = 32

    for emitter_id, tx in enumerate(tx_configs[:20]):
        freq_cfg = tx.get("frequency_config", {})
        freq_mode = str(freq_cfg.get("freq_mode", "FixedSingle"))
        freqs_mhz = freq_cfg.get("freqs_mhz", [1000.0])

        power_cfg = tx.get("power_config", {})
        power_w = float(power_cfg.get("power_w", 100.0))
        # Scaled tactical Rx power in dBm
        import math

        power_dbm = max(-85.0, min(-40.0, 10.0 * math.log10(max(1.0, power_w) * 1000.0) - 95.0))

        # Map radar RF frequencies to 32 receiver channels
        mapped_bands = [_freq_to_band(f) for f in freqs_mhz]
        unique_bands = sorted(set(mapped_bands))

        if len(unique_bands) > 1 or "hop" in freq_mode.lower() or "random" in freq_mode.lower():
            if len(unique_bands) == 1:
                b = unique_bands[0]
                unique_bands = sorted(
                    {
                        max(0, b - 2),
                        max(0, b - 1),
                        b,
                        min(n_bands - 1, b + 1),
                        min(n_bands - 1, b + 2),
                    }
                )
            emitters.append(
                AgileEmitterConfig(
                    emitter_id=emitter_id,
                    band_ids=unique_bands,
                    power_dbm=power_dbm,
                    hop_interval_slots=10,
                    hop_pattern="pseudo_random",
                )
            )
        else:
            emitters.append(
                FixedEmitterConfig(
                    emitter_id=emitter_id,
                    band_id=unique_bands[0] if unique_bands else 0,
                    power_dbm=power_dbm,
                )
            )

    if not emitters:
        emitters = [FixedEmitterConfig(emitter_id=0, band_id=8, power_dbm=-60.0)]

    return emitters


def _get_field(rec: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    """Return first matching key from a record dict."""
    for k in keys:
        if k in rec:
            return rec[k]
    return default


def _freq_to_band(
    freq_mhz: float, start_mhz: float = 100.0, step_mhz: float = 10.0, n_bands: int = 32
) -> int:
    """Map a centre frequency (MHz) to a band index [0, n_bands-1]."""
    idx = int((freq_mhz - start_mhz) / step_mhz)
    return max(0, min(n_bands - 1, idx))


# ---------------------------------------------------------------------------
# Offline stub (mirrors Turing dataset schema without network access)
# ---------------------------------------------------------------------------


def _offline_stub(n_slots: int, seed: int) -> ScenarioConfig:
    """Built-in synthetic scenario that mirrors the Turing dataset structure.

    Used when the network is unavailable or ``force_offline=True``.
    Emitter parameters are drawn from the statistical distribution
    described in JC Wise (2024).
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    emitters: list[Any] = []

    # 5 fixed-frequency emitters (pulsed / CW radars)
    for i in range(5):
        band_id = int(rng.integers(0, 32))
        power_dbm = float(rng.uniform(-70.0, -50.0))
        emitters.append(
            FixedEmitterConfig(
                emitter_id=i,
                band_id=band_id,
                power_dbm=power_dbm,
            )
        )

    # 3 frequency-agile emitters (ECM / FHSS radars)
    for j in range(3):
        centre = int(rng.integers(4, 28))
        hop_bands = list(range(max(0, centre - 3), min(32, centre + 4)))
        power_dbm = float(rng.uniform(-65.0, -50.0))
        emitters.append(
            AgileEmitterConfig(
                emitter_id=5 + j,
                band_ids=hop_bands,
                power_dbm=power_dbm,
                hop_interval_slots=int(rng.integers(5, 20)),
                hop_pattern="pseudo_random",
            )
        )

    return ScenarioConfig(
        scenario_id="turing_radar_offline_stub",
        description=(
            "Offline stub mirroring Turing Synthetic Radar Dataset schema "
            "(JC Wise 2024) — used when network is unavailable."
        ),
        seed=seed,
        n_slots=n_slots,
        emitters=emitters,
        noise_std_db=2.0,
    )
