"""Unit tests for ROCm counter helpers."""

from __future__ import annotations

from probing.profiling.torch_profiler.rocm_metrics import (
    rocm_dram_bytes,
    rocm_instruction_flops,
)


def test_rocm_dram_bytes_converts_read_and_write_requests():
    assert (
        rocm_dram_bytes(
            {
                "TCC_EA_RDREQ_32B": 10,
                "TCC_EA_RDREQ": 15,
                "TCC_EA_WRREQ_64B": 10,
                "TCC_EA_WRREQ": 20,
            }
        )
        == (10 * 32 + 5 * 64) + (10 * 32 + 10 * 64)
    )


def test_rocm_dram_bytes_requires_all_four_counters():
    assert (
        rocm_dram_bytes(
            {
                "TCC_EA_RDREQ_32B": 10,
                "TCC_EA_RDREQ": 15,
                "TCC_EA_WRREQ": 20,
            }
        )
        is None
    )


def test_rocm_instruction_flops_stays_uncalibrated():
    assert rocm_instruction_flops({"SQ_INSTS_VALU": 100}) is None
