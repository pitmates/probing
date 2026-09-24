"""Single-config front door for roofline tuning.

The only knob a user should need is ``PROBING_TORCH_PROFILER_ANALYSIS=roofline``.
Everything vendor/calibration specific is optional and collapses into one
variable, ``PROBING_TORCH_ROOFLINE_CONFIG``, which accepts inline JSON or a
path to a JSON file::

    {
      "backend": "rocm",
      "rocm_enabled": true,
      "rocprof_cmd": "rocprof --output {output} --basenames on --stats",
      "probe_cmd": "rocprofv2 --list-counters",
      "metrics": ["TCC_EA_RDREQ_32B", "TCC_EA_RDREQ", "TCC_EA_WRREQ_64B", "TCC_EA_WRREQ"],
      "peaks": {"backend": "rocm", "device_arch": "gfx936",
                "peaks": {"fp16_tensor_dense": {"peak_flops": 312e12, "peak_bytes_per_sec": 1.6e12}}},
      "flop_weights": {"SQ_INSTS_VALU": 2, "SQ_INSTS_SALU": 1}
    }

The legacy per-vendor environment variables remain as fallbacks so existing
deployments keep working, but new setup does not need them.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any


CONFIG_ENV = "PROBING_TORCH_ROOFLINE_CONFIG"


@dataclass(frozen=True)
class RooflineConfig:
    rocm_enabled: bool = True
    backend: str = "auto"
    rocprof_cmd: str = ""
    probe_cmd: str = ""
    metrics: tuple[str, ...] = ()
    peaks: dict[str, Any] | None = None
    flop_weights: dict[str, int] | None = None


def _parse_metrics(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        names = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    else:
        return ()
    return tuple(dict.fromkeys(names))


def _parse_weights(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    weights: dict[str, int] = {}
    for name, weight in value.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            continue
        if isinstance(weight, int):
            if weight < 0:
                continue
            weights[name.strip()] = weight
            continue
        if not math.isfinite(weight) or weight < 0:
            continue
        weights[name.strip()] = int(weight)
    return weights


def _read_raw_config() -> dict[str, Any]:
    raw = os.environ.get(CONFIG_ENV, "").lstrip("\ufeff").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        payload = raw
    else:
        try:
            with open(raw, encoding="utf-8") as handle:
                payload = handle.read()
        except OSError:
            return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_roofline_config() -> RooflineConfig:
    """Read ``PROBING_TORCH_ROOFLINE_CONFIG`` with legacy env fallbacks."""
    raw = _read_raw_config()

    rocm_enabled = raw.get("rocm_enabled", raw.get("enabled", True))
    if not isinstance(rocm_enabled, bool):
        rocm_enabled = True

    backend = raw.get("backend", "auto")
    if not isinstance(backend, str) or not backend.strip():
        backend = "auto"
    backend = backend.strip().lower()

    rocprof_cmd = raw.get("rocprof_cmd", "")
    if not isinstance(rocprof_cmd, str):
        rocprof_cmd = ""
    rocprof_cmd = rocprof_cmd.strip()

    probe_cmd = raw.get("probe_cmd", "")
    if not isinstance(probe_cmd, str):
        probe_cmd = ""
    probe_cmd = probe_cmd.strip()

    peaks = raw.get("peaks") if isinstance(raw.get("peaks"), dict) else None
    flop_weights = _parse_weights(raw.get("flop_weights"))

    return RooflineConfig(
        rocm_enabled=rocm_enabled,
        backend=backend,
        rocprof_cmd=rocprof_cmd,
        probe_cmd=probe_cmd,
        metrics=_parse_metrics(raw.get("metrics")),
        peaks=peaks,
        flop_weights=flop_weights,
    )
