"""Unit tests for roofline backend detection and capability probing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from probing.profiling.torch_profiler.backends import (
    BackendInfo,
    CudaRooflineBackend,
    RocmRooflineBackend,
    UnavailableRooflineBackend,
    create_roofline_backend,
    detect_backend,
    selected_backend,
)


def _backend() -> RocmRooflineBackend:
    return RocmRooflineBackend(
        BackendInfo("amd", "BW", "gfx936", "rocm")
    )


def _fake_torch(*, hip=None, cuda=None, cuda_available=True, props=None):
    fake = MagicMock()
    fake.version.hip = hip
    fake.version.cuda = cuda
    fake.cuda.is_available.return_value = cuda_available
    fake.cuda.get_device_properties.return_value = props or SimpleNamespace(
        name="TestGPU",
        major=9,
        minor=0,
        gcnArchName="gfx936",
    )
    return fake


def test_detect_cuda_uses_available_device():
    info = detect_backend(_fake_torch(cuda="12.1"))
    assert info.vendor == "nvidia"
    assert info.counter_source == "cuda"
    assert info.device_model == "TestGPU"
    assert info.device_arch == "sm_90"


def test_detect_rocm_reads_gcn_arch():
    info = detect_backend(_fake_torch(hip="6.3.26093"))
    assert info.vendor == "amd"
    assert info.counter_source == "rocm"
    assert info.device_arch == "gfx936"


def test_rocm_probe_disabled_is_unavailable(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", raising=False)
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    result = RocmRooflineBackend(info).probe(_fake_torch(hip="6.3.26093"))
    assert result.status == "unavailable"
    assert "disabled" in result.error


def test_rocm_probe_enabled_without_command_is_unavailable(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    result = RocmRooflineBackend(info).probe(_fake_torch(hip="6.3.26093"))
    assert result.status == "unavailable"
    assert "ROCPROF_CMD" in result.error
    assert "{output}" in result.error


def test_rocm_probe_enabled_with_command_is_ok(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD",
        "rocprofv2 --target-process {pid} --output {output}",
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    result = RocmRooflineBackend(info).probe(_fake_torch(hip="6.3.26093"))
    assert result.status == "ok"
    assert result.error == ""


def test_rocm_platform_peaks_reads_rocm_config(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON",
        json.dumps(
            {
                "backend": "rocm",
                "device_arch": "gfx936",
                "peaks": {
                    "fp16_tensor_dense": {
                        "peak_flops": 312e12,
                        "peak_bytes_per_sec": 1.6e12,
                    }
                },
            }
        ),
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        json.dumps(
            {"fp16_tensor_dense": {"peak_flops": 1, "peak_bytes_per_sec": 1}}
        ),
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    assert RocmRooflineBackend(info).platform_peaks(info) == (312e12, 1.6e12)


def test_rocm_zero_peaks_mean_uncalibrated(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON",
        json.dumps(
            {
                "backend": "rocm",
                "device_arch": "gfx936",
                "peaks": {
                    "fp16_tensor_dense": {
                        "peak_flops": 0,
                        "peak_bytes_per_sec": 0,
                    }
                },
            }
        ),
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    assert RocmRooflineBackend(info).platform_peaks(info) == (None, None)


def test_rocm_peaks_reject_backend_mismatch(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON",
        json.dumps(
            {
                "backend": "cuda",
                "device_arch": "gfx936",
                "peaks": {
                    "fp16_tensor_dense": {
                        "peak_flops": 1,
                        "peak_bytes_per_sec": 1,
                    }
                },
            }
        ),
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    assert RocmRooflineBackend(info).platform_peaks(info) == (None, None)


def test_selected_backend_explicit_override_reports_diagnostic(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_BACKEND", "cuda")
    info = selected_backend(_fake_torch(hip="6.3.26093"))
    assert info.counter_source == "none"
    assert "non-CUDA" in info.diagnostic


def test_create_roofline_backend_dispatch(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_BACKEND", "auto")
    assert isinstance(
        create_roofline_backend(_fake_torch(cuda="12.1")),
        CudaRooflineBackend,
    )
    assert isinstance(
        create_roofline_backend(_fake_torch(hip="6.3.26093")),
        RocmRooflineBackend,
    )


def test_cuda_build_profiler_kwargs_adds_experimental_config():
    config = object()
    backend = CudaRooflineBackend(BackendInfo("nvidia", "TestGPU", "sm_90", "cuda"))
    backend._experimental_config = config
    kwargs = backend.build_profiler_kwargs({"with_flops": False}, _fake_torch(cuda="12.1"))
    assert kwargs["with_flops"] is False
    assert kwargs["experimental_config"] is config


def test_cuda_compile_counter_rows_dispatch_returns_empty_unavailable():
    backend = CudaRooflineBackend(BackendInfo("nvidia", "TestGPU", "sm_90", "cuda"))
    result = backend.compile_counter_rows(
        [],
        capture_id="cap",
        local_step=1,
        global_step=1,
        rank=0,
        role="dp=0",
    )
    assert result.quality == "unavailable"
    assert result.counter_events == 0


def test_cuda_probe_missing_experimental_config_raises():
    fake = _fake_torch(cuda="12.1")
    fake.profiler._ExperimentalConfig = None
    info = BackendInfo("nvidia", "TestGPU", "sm_90", "cuda")
    with pytest.raises(RuntimeError, match="_ExperimentalConfig"):
        CudaRooflineBackend(info).probe(fake)


def test_unavailable_backend_carries_diagnostic():
    info = BackendInfo("unknown", "", "unknown", "none", "requested cuda on non-CUDA")
    backend = UnavailableRooflineBackend(info)
    result = backend.probe(_fake_torch(cuda_available=False))
    assert result.status == "unavailable"
    assert result.error == info.diagnostic
    compiled = backend.compile_counter_rows(
        [],
        capture_id="cap",
        local_step=1,
        global_step=1,
        rank=0,
        role="dp=0",
    )
    assert compiled.error == info.diagnostic

def test_rocm_probe_runs_probe_command_when_configured(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD", "rocprofv2 --output {output}"
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_PROBE_CMD", "rocprofv2 --list-counters"
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.backends._run_rocm_probe",
        lambda command, timeout_s=15.0: (False, "rejected"),
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    result = RocmRooflineBackend(info).probe(_fake_torch(hip="6.3.26093"))
    assert result.status == "unavailable"
    assert "probe failed" in result.error


def test_rocm_probe_ok_when_probe_command_succeeds(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD", "rocprofv2 --output {output}"
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_PROBE_CMD", "rocprofv2 --list-counters"
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.backends._run_rocm_probe",
        lambda command, timeout_s=15.0: (True, ""),
    )
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    result = RocmRooflineBackend(info).probe(_fake_torch(hip="6.3.26093"))
    assert result.status == "ok"


def test_rocm_backend_builds_roofline_when_calibrated(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        json.dumps({"SQ_INSTS_VALU": 2}),
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON",
        json.dumps(
            {
                "backend": "rocm",
                "device_arch": "gfx936",
                "peaks": {
                    "fp16_tensor_dense": {
                        "peak_flops": 1000,
                        "peak_bytes_per_sec": 10000,
                    }
                },
            }
        ),
    )
    backend = _backend()
    result = backend.compile_counter_rows(
        _rocm_document(
            [
                {
                    "kernel_name": "k",
                    "op_name": "aten::mm",
                    "duration_ns": 1000000,
                    "metrics": {
                        "SQ_INSTS_VALU": 500,
                        "TCC_EA_RDREQ_32B": 10,
                        "TCC_EA_RDREQ": 15,
                        "TCC_EA_WRREQ_64B": 10,
                        "TCC_EA_WRREQ": 20,
                    },
                }
            ]
        ),
        capture_id="c",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.counters[0].flops == 1000
    assert len(result.rooflines) == 1
    assert result.rooflines[0].flops == 1000
    assert result.rooflines[0].bottleneck in {"compute", "memory", "balanced"}


def test_rocm_backend_quality_partial_when_no_roofline_rows(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON",
        json.dumps({"SQ_INSTS_VALU": 2}),
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON",
        json.dumps(
            {
                "backend": "rocm",
                "device_arch": "gfx936",
                "peaks": {
                    "fp16_tensor_dense": {
                        "peak_flops": 1000,
                        "peak_bytes_per_sec": 10000,
                    }
                },
            }
        ),
    )
    backend = _backend()
    result = backend.compile_counter_rows(
        _rocm_document(
            [
                {
                    "kernel_name": "k",
                    "op_name": "aten::mm",
                    "metrics": {
                        "SQ_INSTS_VALU": 500,
                        "TCC_EA_RDREQ_32B": 10,
                        "TCC_EA_RDREQ": 15,
                        "TCC_EA_WRREQ_64B": 10,
                        "TCC_EA_WRREQ": 20,
                    },
                }
            ]
        ),
        capture_id="c",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.counters[0].flops == 1000
    assert result.rooflines == []
    assert result.quality == "partial"


def _rocm_document(rows):
    from probing.profiling.torch_profiler.rocm_sidecar import SIDECAR_FORMAT

    return {"format": SIDECAR_FORMAT, "counters": rows}
