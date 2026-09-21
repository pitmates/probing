"""Unit tests for the ROCm rocprofiler sidecar artifact parser."""

from __future__ import annotations

import json

import pytest

from probing.profiling.torch_profiler.backends import (
    BackendInfo,
    RocmRooflineBackend,
)
from probing.profiling.torch_profiler.rocm_sidecar import (
    SIDECAR_FORMAT,
    build_counter_records,
    parse_counter_artifact,
    parse_counter_csv,
)

DRAM_METRICS = {
    "TCC_EA_RDREQ_32B": 10,
    "TCC_EA_RDREQ": 15,
    "TCC_EA_WRREQ_64B": 10,
    "TCC_EA_WRREQ": 20,
}

DRAM_BYTES = (10 * 32 + 5 * 64) + (10 * 32 + 10 * 64)


def _document(rows):
    return {"format": SIDECAR_FORMAT, "counters": rows}


def test_parse_counter_artifact_normalizes_rows():
    rows = parse_counter_artifact(
        json.dumps(
            _document(
                [
                    {
                        "kernel_name": "gemm_kernel",
                        "correlation_id": 7,
                        "op_name": "aten::mm",
                        "duration_ns": 1234000,
                        "metrics": DRAM_METRICS,
                    }
                ]
            )
        )
    )
    assert rows == [
        {
            "kernel_name": "gemm_kernel",
            "op_name": "aten::mm",
            "correlation_id": 7,
            "duration_ns": 1234000,
            "calls": 1,
            "metrics": DRAM_METRICS,
        }
    ]


def test_parse_counter_artifact_rejects_missing_kernel_name():
    with pytest.raises(ValueError, match="missing kernel_name"):
        parse_counter_artifact(_document([{"metrics": DRAM_METRICS}]))


def test_parse_counter_artifact_rejects_wrong_format():
    with pytest.raises(ValueError, match="unsupported sidecar format"):
        parse_counter_artifact({"format": "other", "counters": []})


def test_build_counter_records_leaves_flops_uncalibrated():
    counters, missing = build_counter_records(
        parse_counter_artifact(
            _document(
                [
                    {
                        "kernel_name": "gemm_kernel",
                        "op_name": "aten::mm",
                        "duration_ns": 1500,
                        "metrics": DRAM_METRICS,
                    }
                ]
            )
        ),
        capture_id="c1",
        local_step=3,
        global_step=3,
        rank=0,
        role="",
    )
    assert len(counters) == 1
    assert missing == []
    row = counters[0]
    assert row.flops is None
    assert row.duration_us == 1
    assert row.dram_bytes == DRAM_BYTES
    assert json.loads(row.metrics) == DRAM_METRICS


def test_build_counter_records_reports_missing_dram_metrics():
    counters, missing = build_counter_records(
        parse_counter_artifact(
            _document([{"kernel_name": "k", "duration_ns": 1, "metrics": {"SQ_INSTS_VALU": 3}}])
        ),
        capture_id="c",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert len(counters) == 1
    assert counters[0].dram_bytes is None
    assert counters[0].flops is None
    assert set(missing) == {
        "TCC_EA_RDREQ_32B",
        "TCC_EA_RDREQ",
        "TCC_EA_WRREQ_64B",
        "TCC_EA_WRREQ",
    }


def test_build_counter_records_respects_calls_field():
    counters, missing = build_counter_records(
        parse_counter_artifact(
            _document(
                [
                    {
                        "kernel_name": "k",
                        "calls": 3,
                        "duration_ns": 1000,
                        "metrics": DRAM_METRICS,
                    }
                ]
            )
        ),
        capture_id="c",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert missing == []
    assert counters[0].calls == 3


def _backend():
    return RocmRooflineBackend(
        BackendInfo(
            vendor="amd",
            device_model="",
            device_arch="gfx936",
            counter_source="rocm",
        )
    )


def test_rocm_backend_materializes_counter_facts():
    result = _backend().compile_counter_rows(
        _document(
            [
                {
                    "kernel_name": "k1",
                    "op_name": "aten::mm",
                    "duration_ns": 2000,
                    "metrics": DRAM_METRICS,
                },
                {"kernel_name": "k2", "duration_ns": 3000, "metrics": DRAM_METRICS},
            ]
        ),
        capture_id="c1",
        local_step=2,
        global_step=2,
        rank=0,
        role="",
    )
    assert result.quality == "partial"
    assert result.counter_events == 2
    assert result.associated_kernels == 1
    assert result.unassociated_kernels == 1
    assert result.rooflines == []
    assert len(result.counters) == 2
    assert all(counter.flops is None for counter in result.counters)


def test_rocm_backend_does_not_join_on_correlation_id_alone():
    result = _backend().compile_counter_rows(
        _document(
            [
                {
                    "kernel_name": "k1",
                    "correlation_id": 17,
                    "duration_ns": 1000,
                    "metrics": DRAM_METRICS,
                }
            ]
        ),
        capture_id="c",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.associated_kernels == 0
    assert result.unassociated_kernels == 1


def test_rocm_backend_empty_artifacts_is_unavailable():
    result = _backend().compile_counter_rows(
        [],
        capture_id="c1",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.quality == "unavailable"
    assert result.counters == []


def test_rocm_backend_malformed_artifact_is_unavailable():
    result = _backend().compile_counter_rows(
        {"format": "not-rocm"},
        capture_id="c1",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.quality == "unavailable"
    assert result.counters == []
    assert "parse failed" in result.error


def test_parse_counter_csv_extracts_kernel_and_metrics():
    text = """Index,KernelName,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ,SQ_INSTS_VALU
0,gemm_kernel,12345,10,15,10,20,512
"""
    rows = parse_counter_csv(text)
    assert rows == [
        {
            "kernel_name": "gemm_kernel",
            "op_name": "",
            "correlation_id": None,
            "duration_ns": 12345,
            "calls": 1,
            "metrics": {
                "TCC_EA_RDREQ_32B": 10,
                "TCC_EA_RDREQ": 15,
                "TCC_EA_WRREQ_64B": 10,
                "TCC_EA_WRREQ": 20,
                "SQ_INSTS_VALU": 512,
            },
        }
    ]


def test_parse_counter_csv_preserves_correlation_id():
    text = """KernelName,correlation_id,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
k,41,1000,10,15,10,20
"""
    rows = parse_counter_csv(text)
    assert len(rows) == 1
    assert rows[0]["correlation_id"] == 41


def test_parse_counter_csv_tolerates_empty_metric_cells():
    text = """KernelName,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
k,1000,10,,10,20
"""
    rows = parse_counter_csv(text)
    assert len(rows) == 1
    assert rows[0]["metrics"]["TCC_EA_WRREQ_64B"] == 10
    assert "TCC_EA_RDREQ" not in rows[0]["metrics"]


def test_parse_counter_csv_rejects_missing_kernel_column():
    with pytest.raises(ValueError, match="KernelName"):
        parse_counter_csv("DurationNs,TCC_EA_RDREQ\n1,2\n")


def test_parse_counter_artifact_detects_csv():
    text = """KernelName,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
k,1000,10,15,10,20
"""
    rows = parse_counter_artifact(text)
    assert len(rows) == 1
    assert rows[0]["kernel_name"] == "k"


def test_rocm_backend_compiles_csv_facts():
    text = """Index,KernelName,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
0,k1,2000,10,15,10,20
"""
    result = _backend().compile_counter_rows(
        text,
        capture_id="c1",
        local_step=2,
        global_step=2,
        rank=0,
        role="",
    )
    assert result.quality == "partial"
    assert result.counter_events == 1
    assert result.counters[0].dram_bytes == DRAM_BYTES
    assert result.counters[0].flops is None
