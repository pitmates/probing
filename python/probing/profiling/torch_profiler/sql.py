"""SQL row providers for python.profile_* virtual tables."""

from __future__ import annotations

from typing import Any

import probing

from .session_store import (
    CaptureRecord,
    CounterRecord,
    HotspotRecord,
    RooflineRecord,
    get_session_store,
)

_CAPTURE_DOC = (
    "On-demand torch.profiler capture anchor (step, rank, quality metadata). "
    "One row per profile window."
)
_HOTSPOT_DOC = (
    "Conclusion fact table: aggregated kernel/op time buckets per capture. "
    "Fed by KinetoSqlAdaptor at finalize."
)
_COUNTER_DOC = (
    "Kernel-level CUPTI counter facts for a torch.profiler capture. "
    "Requires a capture with analysis=roofline."
)
_ROOFLINE_DOC = (
    "Operator/kernel roofline conclusions derived from profile_counter. "
    "Requires a capture with analysis=roofline."
)

_COLUMN_DOCS: dict[str, dict[str, str]] = {
    "profile_capture": {
        "capture_id": "Unique capture id (join key)",
        "local_step": "Per-rank training step at finalize",
        "global_step": "Global training step at finalize",
        "rank": "torch.distributed rank (-1 unknown)",
        "world_size": "World size (-1 unknown)",
        "role": "Parallel role key (dp=…,pp=…)",
        "trigger": "Who started capture (manual, skill, http)",
        "steps_profiled": "Profiler window length in optimizer steps",
        "wall_us": "Capture wall time (microseconds); Q2 denominator",
        "started_at_us": "Capture start (epoch microseconds)",
        "ended_at_us": "Capture end (epoch microseconds)",
        "status": "running | completed | failed",
        "truncated": "1 if event list was truncated",
        "event_count": "Raw profiler event count before aggregation",
        "error": "Failure message when status=failed",
        "analysis": "Capture-selected analysis features (none | roofline)",
        "roofline_quality": "ok | partial | truncated | unavailable",
        "roofline_counter_events": "CUPTI counter events parsed",
        "roofline_associated_kernels": "Counter kernels associated with an operator",
        "roofline_unassociated_kernels": "Counter kernels without an operator association",
        "roofline_missing_metrics": "Missing required metric names as JSON array",
        "roofline_parser_version": "Roofline parser and HTA alignment version",
    },
    "profile_hotspot": {
        "capture_id": "FK to profile_capture",
        "local_step": "Training step (query without capture_id)",
        "global_step": "Global step",
        "rank": "Rank that produced this row",
        "bucket_kind": "kernel | cpu_op | memcpy | cuda_runtime | collective | other",
        "bucket_name": "Kernel or op name",
        "self_us": "Self time microseconds (primary sort key)",
        "wall_us": "Wall/subtree time microseconds",
        "calls": "Invocation count in capture",
        "pct_of_capture": "self_us / capture.wall_us",
        "module_hint": "Module hint from stack (v2)",
    },
    "profile_counter": {
        "capture_id": "FK to profile_capture",
        "local_step": "Training step at finalize",
        "global_step": "Global training step",
        "rank": "Rank that produced this row",
        "role": "Parallel role key",
        "kernel_name": "GPU kernel name",
        "op_name": "Associated PyTorch CPU operator",
        "top_level_op": "Outermost operator in the local op stack",
        "bottom_level_op": "Innermost operator in the local op stack",
        "op_stack": "Operator stack as a JSON array (diagnostic only)",
        "calls": "Kernel invocation count",
        "duration_us": "Kernel duration (microseconds)",
        "flops": "FLOPs from SASS instruction counters",
        "dram_bytes": "DRAM bytes read + written",
        "metrics": "Extra requested metrics as a JSON object",
    },
    "profile_roofline": {
        "capture_id": "FK to profile_capture",
        "local_step": "Training step at finalize",
        "global_step": "Global training step",
        "rank": "Rank that produced this row",
        "role": "Parallel role key",
        "op_name": "Associated PyTorch CPU operator",
        "kernel_name": "GPU kernel name",
        "calls": "Kernel invocation count",
        "self_duration_us": "Kernel duration (microseconds)",
        "flops": "FLOPs total",
        "dram_bytes": "DRAM bytes total",
        "arithmetic_intensity": "FLOPs / DRAM bytes",
        "achieved_flops": "FLOPs per second",
        "achieved_bytes_per_sec": "DRAM bytes per second",
        "peak_flops": "Reference peak FLOPs per second",
        "peak_bytes_per_sec": "Reference peak DRAM bandwidth",
        "peak_flops_kind": "Peak reference kind (fp16_tensor_dense in v1)",
        "boundedness": "min(compute efficiency, memory efficiency)",
        "bottleneck": "compute | memory | balanced | unknown",
        "data_quality": "ok | partial | truncated | unavailable",
    },
}

_DOCS_REGISTERED = False


def _register_docs_once() -> None:
    global _DOCS_REGISTERED
    if _DOCS_REGISTERED:
        return
    probing.register_table_docs(
        "python.profile_capture", _CAPTURE_DOC, _COLUMN_DOCS["profile_capture"]
    )
    probing.register_table_docs(
        "python.profile_hotspot", _HOTSPOT_DOC, _COLUMN_DOCS["profile_hotspot"]
    )
    probing.register_table_docs(
        "python.profile_counter", _COUNTER_DOC, _COLUMN_DOCS["profile_counter"]
    )
    probing.register_table_docs(
        "python.profile_roofline", _ROOFLINE_DOC, _COLUMN_DOCS["profile_roofline"]
    )
    _DOCS_REGISTERED = True


def _capture_to_dict(row: CaptureRecord) -> dict[str, Any]:
    return {
        "capture_id": row.capture_id,
        "local_step": row.local_step,
        "global_step": row.global_step,
        "rank": row.rank,
        "world_size": row.world_size,
        "role": row.role,
        "trigger": row.trigger,
        "steps_profiled": row.steps_profiled,
        "wall_us": row.wall_us,
        "started_at_us": row.started_at_us,
        "ended_at_us": row.ended_at_us,
        "status": row.status,
        "truncated": 1 if row.truncated else 0,
        "event_count": row.event_count,
        "error": row.error,
        "analysis": row.analysis,
        "roofline_quality": row.roofline_quality,
        "roofline_counter_events": row.roofline_counter_events,
        "roofline_associated_kernels": row.roofline_associated_kernels,
        "roofline_unassociated_kernels": row.roofline_unassociated_kernels,
        "roofline_missing_metrics": row.roofline_missing_metrics,
        "roofline_parser_version": row.roofline_parser_version,
    }


def _hotspot_to_dict(row: HotspotRecord) -> dict[str, Any]:
    return {
        "capture_id": row.capture_id,
        "local_step": row.local_step,
        "global_step": row.global_step,
        "rank": row.rank,
        "bucket_kind": row.bucket_kind,
        "bucket_name": row.bucket_name,
        "self_us": row.self_us,
        "wall_us": row.wall_us,
        "calls": row.calls,
        "pct_of_capture": row.pct_of_capture,
        "module_hint": row.module_hint,
    }


def _counter_to_dict(row: CounterRecord) -> dict[str, Any]:
    return {
        "capture_id": row.capture_id,
        "local_step": row.local_step,
        "global_step": row.global_step,
        "rank": row.rank,
        "role": row.role,
        "kernel_name": row.kernel_name,
        "op_name": row.op_name,
        "top_level_op": row.top_level_op,
        "bottom_level_op": row.bottom_level_op,
        "op_stack": row.op_stack,
        "calls": row.calls,
        "duration_us": row.duration_us,
        "flops": row.flops,
        "dram_bytes": row.dram_bytes,
        "metrics": row.metrics,
    }


def _roofline_to_dict(row: RooflineRecord) -> dict[str, Any]:
    return {
        "capture_id": row.capture_id,
        "local_step": row.local_step,
        "global_step": row.global_step,
        "rank": row.rank,
        "role": row.role,
        "op_name": row.op_name,
        "kernel_name": row.kernel_name,
        "calls": row.calls,
        "self_duration_us": row.self_duration_us,
        "flops": row.flops,
        "dram_bytes": row.dram_bytes,
        "arithmetic_intensity": row.arithmetic_intensity,
        "achieved_flops": row.achieved_flops,
        "achieved_bytes_per_sec": row.achieved_bytes_per_sec,
        "peak_flops": row.peak_flops,
        "peak_bytes_per_sec": row.peak_bytes_per_sec,
        "peak_flops_kind": row.peak_flops_kind,
        "boundedness": row.boundedness,
        "bottleneck": row.bottleneck,
        "data_quality": row.data_quality,
    }


def profile_capture_rows() -> list[dict[str, Any]]:
    """Rows for ``SELECT * FROM python.profile_capture``."""
    _register_docs_once()
    return [_capture_to_dict(c) for c in get_session_store().captures()]


def profile_hotspot_rows() -> list[dict[str, Any]]:
    """Rows for ``SELECT * FROM python.profile_hotspot``."""
    _register_docs_once()
    return [_hotspot_to_dict(h) for h in get_session_store().hotspots()]


def profile_counter_rows() -> list[dict[str, Any]]:
    """Rows for ``SELECT * FROM python.profile_counter``."""
    _register_docs_once()
    return [_counter_to_dict(c) for c in get_session_store().counters()]


def profile_roofline_rows() -> list[dict[str, Any]]:
    """Rows for ``SELECT * FROM python.profile_roofline``."""
    _register_docs_once()
    return [_roofline_to_dict(r) for r in get_session_store().rooflines()]
