"""Slow ROCm roofline E2E (opt-in).

Runs only when an operator sets ``PROBING_TORCH_ROOFLINE_ROCM_E2E=1`` on a real
DCU/ROCm node and supplies the experimental sidecar command template. The test
exercises the real ``rocprofiler`` launch/collect/parse link without requiring
the full training control loop.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("PROBING_TORCH_ROOFLINE_ROCM_E2E") != "1",
        reason="requires a real DCU/ROCm rocprofiler environment",
    ),
]


def test_rocm_sidecar_produces_counter_facts() -> None:
    from probing.profiling.torch_profiler.backends import (
        BackendInfo,
        RocmRooflineBackend,
    )
    from probing.profiling.torch_profiler.rocm_runner import (
        RocmSidecarSession,
        sidecar_command,
        sidecar_enabled,
    )

    assert sidecar_enabled(), "PROBING_TORCH_ROOFLINE_ROCM_PROFILE must be 1"
    assert sidecar_command(), "PROBING_TORCH_ROOFLINE_ROCPROF_CMD must be set"

    session = RocmSidecarSession()
    assert session.start() == "", session.start()

    rows, error = session.collect(timeout_s=float(os.getenv("PROBING_TORCH_ROOFLINE_ROCM_E2E_TIMEOUT", "60")))
    assert error == "", error
    assert rows, "rocprofiler produced no counter rows"

    backend = RocmRooflineBackend(
        BackendInfo(
            vendor="amd",
            device_model=os.getenv("PROBING_TORCH_ROOFLINE_ROCM_E2E_MODEL", ""),
            device_arch=os.getenv("PROBING_TORCH_ROOFLINE_ROCM_E2E_ARCH", "unknown"),
            counter_source="rocm",
        )
    )
    result = backend.compile_counter_rows(
        rows,
        capture_id="rocm-e2e",
        local_step=1,
        global_step=1,
        rank=0,
        role="",
    )
    assert result.counter_events == len(rows)
    assert result.counters
    assert all(counter.flops is None or counter.flops >= 0 for counter in result.counters)
