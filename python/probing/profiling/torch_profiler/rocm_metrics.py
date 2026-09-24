"""Pure ROCm counter-name and byte-conversion helpers.

The conversion formulas are draft and are intentionally isolated here so the
eventual ``rocprofiler`` sidecar parser can use the same definitions as the
capability probe and unit tests.
"""

from __future__ import annotations

import json
import math
import os
from typing import Mapping


ROCM_DEFAULT_METRICS: tuple[str, ...] = (
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM_WR",
    "SQ_INSTS_VMEM_RD",
    "TCC_EA_RDREQ_32B",
    "TCC_EA_WRREQ_64B",
    "TCC_EA_RDREQ",
    "TCC_EA_WRREQ",
)

_READ_32B = "TCC_EA_RDREQ_32B"
_READ_TOTAL = "TCC_EA_RDREQ"
_WRITE_64B = "TCC_EA_WRREQ_64B"
_WRITE_TOTAL = "TCC_EA_WRREQ"

ROCM_FLOP_WEIGHTS_ENV = "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON"

ROCM_DRAM_METRICS: tuple[str, ...] = (
    _READ_32B,
    _READ_TOTAL,
    _WRITE_64B,
    _WRITE_TOTAL,
)


def _nonnegative(value: int | float | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def rocm_dram_bytes(metrics: Mapping[str, int | float]) -> int | None:
    """Return DRAM bytes read + written, or ``None`` when required counters are absent.

    ROCm counter semantics from the phase-0 counter catalog:
    - ``TCC_EA_RDREQ`` counts 32B and 64B read requests; ``TCC_EA_RDREQ_32B``
      counts only the 32B subset.
    - ``TCC_EA_WRREQ`` counts 32B and 64B write requests; ``TCC_EA_WRREQ_64B``
      counts only the 64B subset.
    """
    read_32b = _nonnegative(metrics.get(_READ_32B))
    read_total = _nonnegative(metrics.get(_READ_TOTAL))
    write_64b = _nonnegative(metrics.get(_WRITE_64B))
    write_total = _nonnegative(metrics.get(_WRITE_TOTAL))
    if any(value is None for value in (read_32b, read_total, write_64b, write_total)):
        return None

    read_bytes = 32 * read_32b + 64 * max(read_total - read_32b, 0)
    write_bytes = 32 * max(write_total - write_64b, 0) + 64 * write_64b
    return read_bytes + write_bytes


def rocm_missing_metrics(
    metrics: Mapping[str, int | float],
    names: tuple[str, ...] | list[str],
) -> list[str]:
    """Return counter names that are absent or not coercible to a count.

    Unlike a plain key-membership check, this also treats an empty or
    non-numeric value as missing so the diagnostic list stays complete.
    """
    return [name for name in names if _nonnegative(metrics.get(name)) is None]


def rocm_flop_weights() -> dict[str, int]:
    """Parse explicit instruction-to-FLOP weights from config or environment.

    ``PROBING_TORCH_ROOFLINE_CONFIG.flop_weights`` maps a ROCm instruction
    counter name to FLOPs per instruction, for example
    ``{"SQ_INSTS_VALU": 2, "SQ_INSTS_SALU": 1}``. The legacy
    ``PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON`` env is the fallback.
    When neither is set or invalid the result is empty and no FLOPs are
    fabricated.
    """
    from .config import load_roofline_config

    config = load_roofline_config()
    if config.flop_weights is not None:
        parsed = config.flop_weights
    else:
        raw = os.environ.get(ROCM_FLOP_WEIGHTS_ENV, "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}

    weights: dict[str, int] = {}
    for name, value in parsed.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value < 0:
                continue
            weights[name.strip()] = value
            continue
        if not math.isfinite(value) or value < 0:
            continue
        weights[name.strip()] = int(value)
    return weights


def rocm_instruction_flops(
    metrics: Mapping[str, int | float],
    weights: dict[str, int] | None = None,
) -> int | None:
    """Return calibrated FLOPs, or ``None`` when no weights are configured.

    v1 ships with no default weights, so ROCm roofline produces counter facts
    only until an operator supplies the calibration JSON for their device.
    """
    if weights is None:
        weights = rocm_flop_weights()
    if not weights:
        return None
    total = 0.0
    matched = False
    for name, weight in weights.items():
        value = metrics.get(name)
        if value is None:
            continue
        try:
            count = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(count) or count < 0:
            continue
        matched = True
        total += count * weight
    return int(total) if matched else None
