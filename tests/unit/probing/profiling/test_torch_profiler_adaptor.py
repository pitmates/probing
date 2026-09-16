"""Unit tests for torch_profiler Kineto → hotspot adaptor."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import tempfile
from unittest.mock import MagicMock

import pytest

from probing.profiling.torch_profiler.adaptor import (
    _bucket_kind_for_name,
    compile_from_profiler,
    compile_key_averages,
    compile_roofline_from_trace,
)
from probing.profiling.torch_profiler.session_store import (
    CaptureRecord,
    CounterRecord,
    HotspotRecord,
    RooflineRecord,
    SessionStore,
    roofline_peaks,
)


@dataclass
class _FakeTimeRange:
    start: int
    end: int

    def elapsed_us(self) -> int:
        return self.end - self.start


@dataclass
class _FakeEvent:
    key: str
    self_cuda_time_total: int = 0
    self_cpu_time_total: int = 0
    cuda_time_total: int = 0
    cpu_time_total: int = 0
    count: int = 1
    args: dict = field(default_factory=dict)
    id: int | None = None
    time_range: _FakeTimeRange | None = None
    stack: tuple[str, ...] = ()


@pytest.fixture
def stub_coords(monkeypatch):
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.row_fields",
        lambda _snap=None: {
            "local_step": 7,
            "global_step": 7,
            "rank": 2,
            "world_size": 8,
        },
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.current_role",
        lambda: "dp=2",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nccl:all_reduce", "collective"),
        ("Memcpy HtoD", "memcpy"),
        ("cudaDeviceSynchronize", "cuda_runtime"),
        ("cudaLaunchKernel", "cuda_runtime"),
        ("aten::mm", "cpu_op"),
        ("autograd::engine", "cpu_op"),
        ("void at::native::vectorized_elementwise_kernel", "kernel"),
    ],
)
def test_bucket_kind_mapping(name, expected):
    assert _bucket_kind_for_name(name) == expected


def test_zero_roofline_peaks_are_unavailable(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":0,"peak_bytes_per_sec":10000}}',
    )
    assert roofline_peaks() == (None, None)


def test_compile_key_averages_buckets_and_pct(stub_coords):
    events = [
        _FakeEvent("nccl:all_reduce", self_cuda_time_total=300, cuda_time_total=400),
        _FakeEvent("aten::mm", self_cpu_time_total=100, cpu_time_total=150),
        _FakeEvent("Memcpy HtoD (Pinned -> Device)", self_cuda_time_total=50),
    ]
    capture, hotspots = compile_key_averages(
        events,
        trigger="test",
        steps_profiled=1,
        started_at_us=1_000_000,
        ended_at_us=1_500_000,
    )
    assert capture.status == "completed"
    assert capture.local_step == 7
    assert capture.rank == 2
    assert capture.wall_us == 500_000
    kinds = {h.bucket_kind for h in hotspots}
    assert kinds == {"collective", "cpu_op", "memcpy"}
    total_pct = sum(h.pct_of_capture for h in hotspots)
    assert abs(total_pct - 450 / 500_000) < 1e-6
    top = max(hotspots, key=lambda h: h.self_us)
    assert top.bucket_kind == "collective"


def test_compile_key_averages_merges_duplicate_buckets(stub_coords):
    events = [
        _FakeEvent("aten::mm", self_cpu_time_total=40, count=2),
        _FakeEvent("aten::mm", self_cpu_time_total=60, count=3),
    ]
    _, hotspots = compile_key_averages(
        events,
        trigger="test",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=100,
    )
    assert len(hotspots) == 1
    assert hotspots[0].self_us == 100
    assert hotspots[0].calls == 5


def test_compile_key_averages_truncation(monkeypatch, stub_coords):
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor._max_events",
        lambda: 2,
    )
    events = [_FakeEvent(f"op{i}", self_cpu_time_total=10) for i in range(5)]
    capture, hotspots = compile_key_averages(
        events,
        trigger="test",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=100,
    )
    assert capture.truncated is True
    assert capture.event_count == 5
    assert len(hotspots) == 2


def test_compile_from_profiler_uses_key_averages(stub_coords):
    profiler = MagicMock()
    profiler.key_averages.return_value = [
        _FakeEvent("aten::add", self_cpu_time_total=25),
    ]
    capture, hotspots, counters, rooflines, quality = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
    )
    assert capture.trigger == "unit"
    assert len(hotspots) == 1
    assert hotspots[0].bucket_name == "aten::add"
    assert counters == []
    assert rooflines == []
    assert quality == "unavailable"


def test_compile_from_profiler_falls_back_to_events(stub_coords):
    profiler = MagicMock()
    profiler.key_averages.side_effect = RuntimeError("not ready")
    profiler.events.return_value = [_FakeEvent("kernel_a", self_cuda_time_total=10)]
    capture, hotspots, _, _, _ = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
    )
    assert capture.status == "completed"
    assert capture.error == ""
    assert len(hotspots) == 1
    assert hotspots[0].bucket_name == "kernel_a"


def test_session_store_bounded(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_PROFILER_MAX_SESSIONS", "2")
    store = SessionStore(max_sessions=2)
    for i in range(3):
        cid = f"c{i}"
        store.add_capture(
            CaptureRecord(capture_id=cid, status="completed"),
            [HotspotRecord(capture_id=cid, bucket_name="k", self_us=1)],
            [CounterRecord(capture_id=cid, kernel_name="k", calls=1)],
            [RooflineRecord(capture_id=cid, kernel_name="k", calls=1)],
        )
    assert len(store.captures()) == 2
    assert store.captures()[0].capture_id == "c1"
    assert all(h.capture_id != "c0" for h in store.hotspots())
    assert all(c.capture_id != "c0" for c in store.counters())
    assert all(r.capture_id != "c0" for r in store.rooflines())


def test_compile_counter_and_roofline(stub_coords, monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_METRICS",
        ",".join(
            [
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ]
        ),
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":10000}}',
    )
    raw_events = [
        _FakeEvent(
            "aten::linear",
            args={"External id": 10},
            time_range=_FakeTimeRange(100, 150),
        ),
        _FakeEvent(
            "gemm",
            args={
                "cat": "cuda_profiler_range",
                "External id": 10,
                "ts": 200,
                "dur": 1_000,
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 3,
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 4,
                "dram__bytes_read.sum": 10,
                "dram__bytes_write.sum": 15,
                "sm__warps_active.avg.pct_of_peak_sustained_active": 0.5,
            },
            time_range=_FakeTimeRange(200, 1_200),
        ),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    capture, hotspots, counters, rooflines, quality = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
        analysis="roofline",
    )
    assert quality == "ok"
    assert capture.analysis == "roofline"
    assert capture.roofline_quality == "ok"
    assert capture.roofline_counter_events == 1
    assert capture.roofline_associated_kernels == 1
    assert capture.roofline_unassociated_kernels == 0
    assert capture.roofline_missing_metrics == "[]"
    assert capture.roofline_parser_version.startswith("probing-roofline-v1+hta-")
    assert len(counters) == len(rooflines) == 1
    counter = counters[0]
    assert counter.op_name == "aten::linear"
    assert counter.calls == 1
    assert counter.duration_us == 1_000
    assert counter.flops == 3 * 2 + 4
    assert counter.dram_bytes == 25
    assert (
        counter.metrics == '{"sm__warps_active.avg.pct_of_peak_sustained_active":0.5}'
    )
    roofline = rooflines[0]
    assert roofline.arithmetic_intensity == 10 / 25
    assert roofline.achieved_flops == 10_000
    assert roofline.achieved_bytes_per_sec == 25_000
    assert roofline.bottleneck == "compute"


def _counter_trace_events():
    return [
        {
            "name": "aten::linear",
            "cat": "cpu_op",
            "ts": 100,
            "dur": 50,
            "args": {"External id": 10},
        },
        {
            "name": "gemm",
            "cat": "cuda_profiler_range",
            "ts": 200,
            "dur": 1_000,
            "args": {
                "External id": 10,
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 3,
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 4,
                "dram__bytes_read.sum": 10,
                "dram__bytes_write.sum": 15,
            },
        },
    ]


def test_counter_fallback_association_does_not_inflate_quality(
    stub_coords, monkeypatch
):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_METRICS", raising=False)
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":100}}',
    )
    raw_events = [
        _FakeEvent("aten::mm", time_range=_FakeTimeRange(100, 120)),
        _FakeEvent(
            "fallback_kernel",
            time_range=_FakeTimeRange(150, 160),
            args={
                "cat": "cuda_profiler_range",
                "ts": 150,
            },
        ),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    capture, _, counters, rooflines, quality = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
        analysis="roofline",
    )
    assert quality == "partial"
    assert capture.roofline_associated_kernels == 0
    assert capture.roofline_unassociated_kernels == 1
    assert counters[0].op_name == "aten::mm"
    assert rooflines[0].data_quality == "partial"


def test_counter_external_id_matches_trace_parity(stub_coords, monkeypatch):
    trace_events = _counter_trace_events()
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":10000}}',
    )
    exporter = MagicMock()

    def export_trace(path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": trace_events}, handle)

    exporter.export_chrome_trace.side_effect = export_trace
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        exporter.export_chrome_trace(handle.name)
        with open(handle.name, encoding="utf-8") as trace_handle:
            exported_trace = trace_handle.read()

    offline = compile_roofline_from_trace(
        exported_trace,
        capture_id="offline",
        local_step=7,
        global_step=7,
        rank=2,
        role="dp=2",
    )
    online_events = [
        _FakeEvent(
            event["name"],
            time_range=_FakeTimeRange(event["ts"], event["ts"] + event["dur"]),
            args={"cat": event["cat"], **event["args"]},
        )
        for event in trace_events
    ]
    profiler = MagicMock()
    profiler.events.return_value = online_events
    profiler.key_averages.return_value = []
    _, _, counters, rooflines, _ = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
        analysis="roofline",
    )
    assert [(row.kernel_name, row.op_stack, row.calls) for row in counters] == [
        (row.kernel_name, row.op_stack, row.calls) for row in offline.counters
    ]
    assert [
        (
            row.op_name,
            row.kernel_name,
            row.calls,
            row.self_duration_us,
            row.flops,
            row.dram_bytes,
            row.arithmetic_intensity,
            row.achieved_flops,
            row.achieved_bytes_per_sec,
            row.boundedness,
            row.bottleneck,
            row.data_quality,
        )
        for row in rooflines
    ] == [
        (
            row.op_name,
            row.kernel_name,
            row.calls,
            row.self_duration_us,
            row.flops,
            row.dram_bytes,
            row.arithmetic_intensity,
            row.achieved_flops,
            row.achieved_bytes_per_sec,
            row.boundedness,
            row.bottleneck,
            row.data_quality,
        )
        for row in offline.rooflines
    ]
    assert counters[0].kernel_name == "gemm"
    assert json.loads(counters[0].op_stack) == ["aten::linear"]
    assert counters[0].flops == 10
    assert counters[0].dram_bytes == 25
    assert offline.associated_kernels == 1


def test_roofline_unavailable_capture_reports_diagnostic_error(
    stub_coords, monkeypatch
):
    profiler = MagicMock()
    profiler.events.return_value = []
    profiler.key_averages.return_value = []
    capture, _, _, _, quality = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=10,
        analysis="roofline",
    )
    assert quality == "unavailable"
    assert "no cuda_profiler_range events" in capture.error


def test_sequential_cpu_ops_do_not_accumulate_into_one_stack(
    stub_coords, monkeypatch
):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":10000}}',
    )
    raw_events = [
        _FakeEvent(
            "aten::mm",
            args={"External id": 10},
            time_range=_FakeTimeRange(100, 150),
        ),
        _FakeEvent(
            "aten::add",
            args={"External id": 11},
            time_range=_FakeTimeRange(160, 190),
        ),
        _FakeEvent(
            "add_kernel",
            args={"cat": "cuda_profiler_range", "External id": 11},
            time_range=_FakeTimeRange(200, 210),
        ),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    _, _, counters, _, _ = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=250,
        analysis="roofline",
    )
    assert json.loads(counters[0].op_stack) == ["aten::add"]


def test_cpu_event_stack_is_preserved(stub_coords, monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":10000}}',
    )
    raw_events = [
        _FakeEvent(
            "aten::add",
            args={"External id": 11},
            stack=("Module.forward", "aten::add"),
            time_range=_FakeTimeRange(100, 150),
        ),
        _FakeEvent(
            "add_kernel",
            args={"cat": "cuda_profiler_range", "External id": 11},
            time_range=_FakeTimeRange(200, 210),
        ),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    _, _, counters, _, _ = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=250,
        analysis="roofline",
    )
    assert json.loads(counters[0].op_stack) == ["Module.forward", "aten::add"]
    assert counters[0].top_level_op == "Module.forward"
    assert counters[0].bottom_level_op == "aten::add"


def test_cpu_event_stack_appends_op_when_missing(stub_coords, monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":10000}}',
    )
    raw_events = [
        _FakeEvent(
            "aten::add",
            args={"External id": 11},
            stack=("Module.forward", "Module.inner"),
            time_range=_FakeTimeRange(100, 150),
        ),
        _FakeEvent(
            "add_kernel",
            args={"cat": "cuda_profiler_range", "External id": 11},
            time_range=_FakeTimeRange(200, 210),
        ),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    _, _, counters, _, _ = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=250,
        analysis="roofline",
    )
    assert json.loads(counters[0].op_stack) == ["Module.forward", "Module.inner", "aten::add"]


def test_offline_trace_launch_fallback_uses_timestamp_order(monkeypatch):
    trace_events = [
        {
            "name": "fallback_kernel",
            "cat": "cuda_profiler_range",
            "ts": 200,
            "dur": 10,
        },
        {
            "name": "aten::mm",
            "cat": "cpu_op",
            "ts": 100,
            "dur": 50,
        },
    ]
    offline = compile_roofline_from_trace(
        {"traceEvents": trace_events},
        capture_id="offline",
        local_step=7,
        global_step=7,
        rank=2,
        role="dp=2",
    )
    assert offline.counters[0].op_name == "aten::mm"
    assert offline.associated_kernels == 0
    assert offline.unassociated_kernels == 1
    assert offline.quality == "partial"


def test_truncated_roofline_keeps_facts_but_not_efficiency(stub_coords, monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_MAX_EVENTS", "1000")
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor._roofline_max_events",
        lambda: 2,
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_PEAKS_JSON",
        '{"fp16_tensor_dense":{"peak_flops":1000,"peak_bytes_per_sec":100}}',
    )
    raw_events = [
        _FakeEvent(
            "aten::mm",
            args={"External id": 10},
            time_range=_FakeTimeRange(100, 150),
        ),
        _FakeEvent(
            "gemm",
            args={
                "cat": "cuda_profiler_range",
                "External id": 10,
                "ts": 200,
                "dur": 10,
            },
            time_range=_FakeTimeRange(200, 210),
        ),
        _FakeEvent("aten::add", time_range=_FakeTimeRange(300, 320)),
    ]
    profiler = MagicMock()
    profiler.events.return_value = raw_events
    profiler.key_averages.return_value = []
    capture, _, counters, rooflines, quality = compile_from_profiler(
        profiler,
        trigger="unit",
        steps_profiled=1,
        started_at_us=0,
        ended_at_us=50,
        analysis="roofline",
    )
    assert quality == "truncated"
    assert capture.roofline_quality == "truncated"
    assert len(counters) == 1
    assert rooflines == []
