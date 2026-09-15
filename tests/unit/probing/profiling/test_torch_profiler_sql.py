"""Unit tests for profile SQL row providers."""

from __future__ import annotations

from probing.profiling.torch_profiler.session_store import (
    CaptureRecord,
    CounterRecord,
    HotspotRecord,
    RooflineRecord,
)
from probing.profiling.torch_profiler.session_store import get_session_store
from probing.profiling.torch_profiler import sql as profile_sql


def test_profile_rows_roundtrip():
    store = get_session_store()
    store.add_capture(
        CaptureRecord(
            capture_id="cap1",
            local_step=5,
            global_step=5,
            rank=1,
            trigger="unit",
            status="completed",
            wall_us=1000,
        ),
        [
            HotspotRecord(
                capture_id="cap1",
                local_step=5,
                bucket_kind="kernel",
                bucket_name="gemm",
                self_us=800,
                pct_of_capture=0.8,
            )
        ],
        [
            CounterRecord(
                capture_id="cap1",
                kernel_name="gemm",
                op_name="aten::linear",
                calls=2,
                flops=100,
                dram_bytes=200,
            )
        ],
        [
            RooflineRecord(
                capture_id="cap1",
                kernel_name="gemm",
                op_name="aten::linear",
                calls=2,
                flops=100,
                dram_bytes=200,
                arithmetic_intensity=0.5,
            )
        ],
    )

    captures = profile_sql.profile_capture_rows()
    hotspots = profile_sql.profile_hotspot_rows()
    assert len(captures) == 1
    assert captures[0]["capture_id"] == "cap1"
    assert captures[0]["local_step"] == 5
    assert captures[0]["truncated"] == 0
    assert len(hotspots) == 1
    assert hotspots[0]["bucket_name"] == "gemm"
    assert hotspots[0]["pct_of_capture"] == 0.8
    counters = profile_sql.profile_counter_rows()
    rooflines = profile_sql.profile_roofline_rows()
    assert counters[0]["kernel_name"] == "gemm"
    assert counters[0]["flops"] == 100
    assert rooflines[0]["arithmetic_intensity"] == 0.5
