"""Compile torch.profiler output into profile_capture + profile_hotspot rows."""

from __future__ import annotations

import logging
import os
import time
import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from probing.parallel import current_role
from probing.tracing.coordinates import row_fields
from probing.tracing import step

from .session_store import (
    CaptureRecord,
    CounterRecord,
    HotspotRecord,
    RooflineRecord,
    roofline_peaks,
)

logger = logging.getLogger(__name__)


def _max_events() -> int:
    raw = os.environ.get("PROBING_TORCH_PROFILER_MAX_EVENTS", "200000").strip()
    try:
        value = int(raw)
    except ValueError:
        return 200000
    return max(value, 1000)


ROOFLINE_PARSER_VERSION = (
    "probing-roofline-v1+hta-52f86de6bdbdaa996102bc017429283bf2b24b9d"
)

_SASS_METRICS: tuple[tuple[str, int], ...] = (
    ("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 2),
    ("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 1),
    ("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 1),
    ("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 2),
    ("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 1),
    ("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 1),
    ("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 2),
    ("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum", 1),
    ("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum", 1),
)
_SASS_METRIC_NAMES = {name for name, _ in _SASS_METRICS}
DEFAULT_ROOFLINE_METRICS = (
    *(name for name, _ in _SASS_METRICS),
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)


@dataclass
class _BucketAgg:
    bucket_kind: str
    bucket_name: str
    self_us: int = 0
    wall_us: int = 0
    calls: int = 0


@dataclass
class _CounterAgg:
    calls: int = 0
    duration_us: int = 0
    flops: int = 0
    dram_bytes: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    op_stack: list[str] = field(default_factory=list)


@dataclass
class _RooflineCompileResult:
    counters: list[CounterRecord]
    rooflines: list[RooflineRecord]
    quality: str
    counter_events: int
    associated_kernels: int
    unassociated_kernels: int
    missing_metrics: list[str]


def _bucket_kind_for_name(name: str) -> str:
    lower = name.lower()
    if "nccl" in lower:
        return "collective"
    if "memcpy" in lower or lower.startswith("memcpy"):
        return "memcpy"
    if "cudadevicesynchronize" in lower or "cudastreamsynchronize" in lower:
        return "cuda_runtime"
    if lower.startswith("cuda") and (
        "launch" in lower or "malloc" in lower or "free" in lower or "sync" in lower
    ):
        return "cuda_runtime"
    if lower.startswith("aten::") or lower.startswith("autograd::"):
        return "cpu_op"
    return "kernel"


def _roofline_max_events() -> int:
    raw = os.environ.get("PROBING_TORCH_ROOFLINE_MAX_EVENTS", "200000").strip()
    try:
        value = int(raw)
    except ValueError:
        return 200_000
    return max(value, 1000)


def _balanced_threshold() -> float:
    raw = os.environ.get("PROBING_TORCH_ROOFLINE_BALANCED_THRESHOLD", "0.9").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.9
    if value <= 0.0 or value >= 1.0:
        return 0.9
    return value


def _roofline_metrics() -> tuple[str, ...]:
    raw = os.environ.get("PROBING_TORCH_ROOFLINE_METRICS", "").strip()
    if not raw:
        return DEFAULT_ROOFLINE_METRICS
    metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
    return metrics or DEFAULT_ROOFLINE_METRICS


def selected_profiler_analysis(analysis: str | None) -> str:
    if analysis is not None:
        return analysis.strip().lower() or "none"
    return os.environ.get("PROBING_TORCH_PROFILER_ANALYSIS", "none").strip().lower()


def roofline_analysis_enabled(analysis: str | None) -> bool:
    selected = selected_profiler_analysis(analysis)
    return "roofline" in {item.strip() for item in selected.split(",") if item.strip()}


def _event_args(event: Any) -> dict[str, Any]:
    args = getattr(event, "args", None)
    return args if isinstance(args, dict) else {}


def _event_category(event: Any) -> str:
    args = _event_args(event)
    value = args.get("cat") or getattr(event, "category", "")
    return str(value) if value else ""


def _counter_value(event: Any, metric: str) -> int | None:
    value = _event_args(event).get(metric)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _counter_float_value(event: Any, metric: str) -> float | None:
    value = _event_args(event).get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _counter_kernel_name(event: Any) -> str:
    return _event_name(event)


def _event_external_id(event: Any) -> int | None:
    value = getattr(event, "id", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cpu_op_name(event: Any) -> str | None:
    name = _event_name(event)
    return name if name.startswith(("aten::", "autograd::")) else None


def _event_self_us(event: Any) -> int:
    cuda = int(getattr(event, "self_cuda_time_total", 0) or 0)
    cpu = int(getattr(event, "self_cpu_time_total", 0) or 0)
    return cuda if cuda > 0 else cpu


def _event_name(event: Any) -> str:
    for attr in ("key", "name"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return "unknown"


def _event_self_us(event: Any) -> int:
    cuda = int(getattr(event, "self_cuda_time_total", 0) or 0)
    cpu = int(getattr(event, "self_cpu_time_total", 0) or 0)
    return cuda if cuda > 0 else cpu


def _event_wall_us(event: Any) -> int:
    cuda = int(getattr(event, "cuda_time_total", 0) or 0)
    cpu = int(getattr(event, "cpu_time_total", 0) or 0)
    return cuda if cuda > 0 else cpu


def _event_calls(event: Any) -> int:
    return max(int(getattr(event, "count", 0) or 0), 1)


def compile_key_averages(
    events: list[Any],
    *,
    trigger: str,
    steps_profiled: int,
    started_at_us: int,
    ended_at_us: int,
    capture_id: Optional[str] = None,
    status: str = "completed",
    error: str = "",
) -> tuple[CaptureRecord, list[HotspotRecord]]:
    """Build capture + hotspot rows from profiler key_averages() events."""
    coords = row_fields(step.snapshot())
    rank = int(coords.get("rank", -1))
    local_step = int(coords.get("local_step", -1))
    global_step = int(coords.get("global_step", -1))

    truncated = False
    original_event_count = len(events)
    max_events = _max_events()
    if original_event_count > max_events:
        truncated = True
        events = events[:max_events]

    aggs: dict[tuple[str, str], _BucketAgg] = {}
    for event in events:
        name = _event_name(event)
        kind = _bucket_kind_for_name(name)
        key = (kind, name)
        agg = aggs.get(key)
        if agg is None:
            agg = _BucketAgg(bucket_kind=kind, bucket_name=name)
            aggs[key] = agg
        agg.self_us += _event_self_us(event)
        agg.wall_us += _event_wall_us(event)
        agg.calls += _event_calls(event)

    total_self_us = sum(a.self_us for a in aggs.values())
    wall_us = max(ended_at_us - started_at_us, 0)
    if wall_us <= 0:
        wall_us = total_self_us

    capture = CaptureRecord(
        capture_id=capture_id or uuid.uuid4().hex,
        local_step=local_step,
        global_step=global_step,
        rank=rank,
        world_size=int(coords.get("world_size", -1)),
        role=current_role(),
        trigger=trigger,
        steps_profiled=steps_profiled,
        wall_us=wall_us,
        started_at_us=started_at_us,
        ended_at_us=ended_at_us,
        status=status,
        truncated=truncated,
        event_count=original_event_count,
        error=error,
    )

    hotspots: list[HotspotRecord] = []
    denom = wall_us if wall_us > 0 else max(total_self_us, 1)
    for agg in sorted(aggs.values(), key=lambda a: a.self_us, reverse=True):
        hotspots.append(
            HotspotRecord(
                capture_id=capture.capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                bucket_kind=agg.bucket_kind,
                bucket_name=agg.bucket_name,
                self_us=agg.self_us,
                wall_us=agg.wall_us,
                calls=agg.calls,
                pct_of_capture=agg.self_us / denom,
            )
        )
    return capture, hotspots


def compile_from_profiler(
    profiler: Any,
    *,
    trigger: str,
    steps_profiled: int,
    started_at_us: int,
    ended_at_us: Optional[int] = None,
    capture_id: Optional[str] = None,
    status: str = "completed",
    error: str = "",
    analysis: str | None = None,
) -> tuple[CaptureRecord, list[HotspotRecord], list[CounterRecord], list[RooflineRecord], str]:
    """Compile a finished torch.profiler profile into SQL rows."""
    end_us = ended_at_us if ended_at_us is not None else _now_us()
    events: list[Any] = []
    raw_events: list[Any] = []
    out_status = status
    out_error = error
    try:
        raw = profiler.events()
        raw_events = list(raw) if raw is not None else []
    except Exception as exc:
        logger.debug("profiler.events failed for roofline: %s", exc)

    try:
        averages = profiler.key_averages()
        events = list(averages) if averages is not None else []
    except Exception as exc:
        logger.debug("profiler.key_averages failed: %s", exc)
        out_error = out_error or str(exc)

    if not events:
        try:
            raw = profiler.events()
            events = list(raw) if raw is not None else []
        except Exception as exc:
            logger.debug("profiler.events fallback failed: %s", exc)
            out_error = out_error or str(exc)

    if not events and out_error and out_status == "completed":
        out_status = "failed"
    elif events and out_status == "completed":
        out_error = ""

    capture, hotspots = compile_key_averages(
        events,
        trigger=trigger,
        steps_profiled=steps_profiled,
        started_at_us=started_at_us,
        ended_at_us=end_us,
        capture_id=capture_id,
        status=out_status,
        error=out_error,
    )
    selected_analysis = selected_profiler_analysis(analysis)
    capture.analysis = selected_analysis
    if roofline_analysis_enabled(selected_analysis):
        result = _compile_counter_rows(
            raw_events,
            capture_id=capture.capture_id,
            local_step=capture.local_step,
            global_step=capture.global_step,
            rank=capture.rank,
            role=capture.role,
        )
        capture.roofline_quality = result.quality
        capture.roofline_counter_events = result.counter_events
        capture.roofline_associated_kernels = result.associated_kernels
        capture.roofline_unassociated_kernels = result.unassociated_kernels
        capture.roofline_missing_metrics = json.dumps(result.missing_metrics)
        capture.roofline_parser_version = ROOFLINE_PARSER_VERSION
        counters = result.counters
        rooflines = result.rooflines
        roofline_quality = result.quality
    else:
        counters = []
        rooflines = []
        roofline_quality = "unavailable"
    return capture, hotspots, counters, rooflines, roofline_quality


def _compile_counter_rows(
    raw_events: list[Any],
    *,
    capture_id: str,
    local_step: int,
    global_step: int,
    rank: int,
    role: str,
) -> _RooflineCompileResult:
    max_events = _roofline_max_events()
    truncated = len(raw_events) > max_events
    scoped_events = raw_events[:max_events]

    cpu_by_external_id: dict[int, list[str]] = {}
    cpu_stacks: list[tuple[int, list[str]]] = []
    counter_events: list[Any] = []
    metrics = _roofline_metrics()
    metric_names = set(metrics)
    sass_names = {name for name, _ in _SASS_METRICS}
    dram_names = {"dram__bytes_read.sum", "dram__bytes_write.sum"}
    missing_metrics: set[str] = set()

    for event in scoped_events:
        category = _event_category(event)
        if category == "cuda_profiler_range":
            counter_events.append(event)
            continue
        op_name = _cpu_op_name(event)
        if op_name is None:
            continue
        external_id = _event_external_id(event)
        op_stack = [*cpu_stacks[-1][1]] if cpu_stacks else []
        op_stack.append(op_name)
        if external_id is not None:
            cpu_by_external_id.setdefault(external_id, []).append(op_name)
        cpu_stacks.append((external_id if external_id is not None else -1, op_stack))

    aggs: dict[tuple[str, tuple[str, ...]], _CounterAgg] = {}
    unassociated = 0
    associated = 0
    for event in counter_events:
        kernel_name = _counter_kernel_name(event)
        external_id = _event_external_id(event)
        op_names = cpu_by_external_id.get(external_id) if external_id is not None else None
        if not op_names:
            op_names = [cpu_stacks[-1][1][-1]] if cpu_stacks else []
        if not op_names:
            op_names = [""]
            unassociated += 1
        else:
            associated += 1
        op_stack = tuple(op_names)
        key = (kernel_name, op_stack)
        agg = aggs.setdefault(key, _CounterAgg())
        agg.calls += 1
        agg.duration_us += _event_self_us(event)
        for name, weight in _SASS_METRICS:
            if name not in metric_names:
                continue
            value = _counter_value(event, name)
            if value is None:
                missing_metrics.add(name)
                continue
            agg.flops += value * weight
        for name in dram_names:
            if name not in metric_names:
                continue
            value = _counter_value(event, name)
            if value is None:
                missing_metrics.add(name)
                continue
            agg.dram_bytes += value
        for name in sorted(metric_names - sass_names - dram_names):
            value = _counter_float_value(event, name)
            if value is None:
                missing_metrics.add(name)
                continue
            agg.metrics[name] = agg.metrics.get(name, 0.0) + value

    if not counter_events:
        quality = "unavailable"
    elif truncated:
        quality = "truncated"
    elif missing_metrics or unassociated:
        quality = "partial"
    else:
        quality = "ok"

    counters: list[CounterRecord] = []
    rooflines: list[RooflineRecord] = []
    peak_flops, peak_bytes = roofline_peaks()
    threshold = _balanced_threshold()
    for (kernel_name, op_stack), agg in sorted(aggs.items()):
        op_name = op_stack[-1]
        counters.append(
            CounterRecord(
                capture_id=capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                role=role,
                kernel_name=kernel_name,
                op_name=op_name,
                top_level_op=op_stack[0] if op_stack else "",
                bottom_level_op=op_name,
                op_stack=json.dumps(op_stack, ensure_ascii=False),
                calls=agg.calls,
                duration_us=agg.duration_us,
                flops=agg.flops,
                dram_bytes=agg.dram_bytes,
                metrics=json.dumps(agg.metrics, separators=(",", ":")),
            )
        )
        arithmetic_intensity = (
            agg.flops / agg.dram_bytes if agg.dram_bytes > 0 else None
        )
        duration_sec = agg.duration_us / 1_000_000
        achieved_flops = agg.flops / duration_sec if duration_sec > 0 else None
        achieved_bytes = agg.dram_bytes / duration_sec if duration_sec > 0 else None
        boundedness: float | None = None
        bottleneck = "unknown"
        if peak_flops and peak_bytes and achieved_flops and achieved_bytes:
            compute_eff = achieved_flops / peak_flops
            memory_eff = achieved_bytes / peak_bytes
            boundedness = min(compute_eff, memory_eff)
            if abs(compute_eff - memory_eff) < (1.0 - threshold):
                bottleneck = "balanced"
            elif compute_eff < memory_eff:
                bottleneck = "compute"
            else:
                bottleneck = "memory"
        rooflines.append(
            RooflineRecord(
                capture_id=capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                role=role,
                op_name=op_name,
                kernel_name=kernel_name,
                calls=agg.calls,
                self_duration_us=agg.duration_us,
                flops=agg.flops,
                dram_bytes=agg.dram_bytes,
                arithmetic_intensity=arithmetic_intensity,
                achieved_flops=achieved_flops,
                achieved_bytes_per_sec=achieved_bytes,
                peak_flops=peak_flops,
                peak_bytes_per_sec=peak_bytes,
                boundedness=boundedness,
                bottleneck=bottleneck,
                data_quality=quality,
            )
        )
    return _RooflineCompileResult(
        counters=counters,
        rooflines=rooflines,
        quality=quality,
        counter_events=len(counter_events),
        associated_kernels=associated,
        unassociated_kernels=unassociated,
        missing_metrics=sorted(missing_metrics),
    )


def _now_us() -> int:
    return int(time.time() * 1_000_000)
