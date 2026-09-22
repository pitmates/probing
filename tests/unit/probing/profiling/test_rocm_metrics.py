"""Unit tests for ROCm counter helpers."""

from __future__ import annotations

import json

from probing.profiling.torch_profiler.rocm_metrics import (
    rocm_dram_bytes,
    rocm_flop_weights,
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


def test_rocm_flop_weights_parse_calibration(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        json.dumps({"SQ_INSTS_VALU": 2, "SQ_INSTS_SALU": 1}),
    )
    assert rocm_flop_weights() == {"SQ_INSTS_VALU": 2, "SQ_INSTS_SALU": 1}
    assert rocm_instruction_flops({"SQ_INSTS_VALU": 4, "SQ_INSTS_SALU": 3}) == 11


def test_rocm_flop_weights_invalid_env_is_empty(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        "not-json",
    )
    assert rocm_flop_weights() == {}
    assert rocm_instruction_flops({"SQ_INSTS_VALU": 4}) is None


def test_rocm_flop_weights_ignore_non_finite(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        json.dumps({"SQ_INSTS_VALU": float("inf"), "SQ_INSTS_SALU": float("nan")}),
    )
    assert rocm_flop_weights() == {}


def test_rocm_instruction_flops_none_when_weighted_metrics_missing(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        json.dumps({"SQ_INSTS_VALU": 2}),
    )
    assert rocm_instruction_flops({"TCC_EA_RDREQ": 1}) is None
