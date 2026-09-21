"""Pure ROCm counter-name and byte-conversion helpers.

The conversion formulas are draft and are intentionally isolated here so the
eventual ``rocprofiler`` sidecar parser can use the same definitions as the
capability probe and unit tests.
"""

from __future__ import annotations

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


def rocm_instruction_flops(_metrics: Mapping[str, int | float]) -> int | None:
    """Return ``None``: instruction-to-FLOP weights are not calibrated yet.

    The v1 implementation must not fabricate an FLOPs value for ROCm before the
    fixture/E2E validation described in ``roofline-backends.zh.md``.
    """
    return None
