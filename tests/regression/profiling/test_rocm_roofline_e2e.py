"""Slow ROCm roofline E2E (opt-in).

Runs only when an operator sets ``PROBING_TORCH_ROOFLINE_ROCM_E2E=1`` on a real
DCU/ROCm node and has already produced whole-run counter artifacts with the
launcher wrapper. This validates the offline import + compile link without the
full training control loop.
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


def test_rocm_wrapper_imports_counter_facts() -> None:
    from probing.profiling.torch_profiler.backends import (
        BackendInfo,
        RocmRooflineBackend,
    )
    from probing.profiling.torch_profiler.rocm_runner import (
        import_artifact_rows,
        sidecar_command,
        sidecar_enabled,
    )

    assert sidecar_enabled(), "rocm roofline wrapper collection must be enabled"
    assert sidecar_command(), "rocm rocprof wrapper command template must be available"

    artifact_dir = os.getenv("PROBING_TORCH_ROOFLINE_ARTIFACT_DIR", "")
    assert artifact_dir, "PROBING_TORCH_ROOFLINE_ARTIFACT_DIR must point at rocprof artifacts"

    rows, error = import_artifact_rows(artifact_dir, rank=0, finalized=True)
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
