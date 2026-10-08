"""Offline ROCm counter-artifact parser for the roofline backend.

This module is the in-process contract half: it validates and normalizes
counter artifacts into python.profile_counter fact rows. Artifact production
(the launcher wrapper and its artifact lifecycle) lives in
``rocm_runner.py``; this module does not launch rocprofiler, manage
subprocesses, clean up files, or perform cluster fan-out.

Supported inputs:

- probing-rocm-sidecar-v1 JSON documents,
- legacy rocprof CSV,
- already-normalized row dicts.

JSON format (draft, subject to fixture/E2E validation)::

    {
      "format": "probing-rocm-sidecar-v1",
      "counters": [
        {
          "kernel_name": "Cijk_Alik_Bljk_HBH",
          "correlation_id": 41,
          "op_name": "aten::mm",
          "calls": 3,
          "duration_ns": 12345,
          "metrics": {
            "SQ_INSTS_VALU": 512,
            "TCC_EA_RDREQ_32B": 10,
            "TCC_EA_RDREQ": 12,
            "TCC_EA_WRREQ_64B": 8,
            "TCC_EA_WRREQ": 16
          }
        }
      ]
    }

CSV supports the legacy rocprof layout::

    Index,KernelName,correlation_id,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
    0,gemm_kernel,41,12345,10,12,8,16

and the rocprofv2 ``--plugin file`` layout, whose header names each counter
once but whose data rows repeat the counter block once per hardware instance.
Those per-instance cells are summed back into a single value per counter, and
``Start_Timestamp`` / ``End_Timestamp`` (nanoseconds) provide duration and
timestamp facts.

The first row is the header. A KernelName column is required; correlation_id,
calls, DurationNs, and an optional op_name column are recognized, and every
remaining non-metadata column is treated as a numeric counter. Empty or
non-numeric counter cells are recorded as missing metrics rather than failing
the whole artifact.

FLOPs are intentionally None until instruction weights are calibrated; only
DRAM bytes (via rocm_metrics.rocm_dram_bytes) and raw counter values are
materialized. correlation_id is preserved on rows when the artifact carries it
so association can prefer it over name-only or timestamp fallbacks.
"""

from __future__ import annotations

import csv
import io
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from .rocm_metrics import (
    ROCM_DRAM_METRICS,
    rocm_dram_bytes,
    rocm_flop_weights,
    rocm_instruction_flops,
    rocm_missing_metrics,
)
from .session_store import CounterRecord, RooflineRecord

SIDECAR_FORMAT = "probing-rocm-sidecar-v1"

_IGNORED_CSV_COLUMNS = {
    "index",
    "gfx",
    "grid",
    "workgroup",
    "wave_size",
    "vgpr",
    "sgpr",
    "lmem",
    "spill",
    "mem",
    "timestamp",
    "device_id",
    "queue_index",
    "pid",
    "tid",
    "dispatch_id",
    "gpu_id",
    "queue_id",
    "grd",
    "wgr",
    "lds",
    "scr",
    "arch_vgpr",
    "accum_vgpr",
    "sig",
    "obj",
    "start_timestamp",
    "end_timestamp",
}

_DURATION_COLUMNS = ("durationns", "duration_ns", "duration", "kernelduration")
_TIMESTAMP_COLUMNS = ("timestamp", "ts")
_TIMESTAMP_NS_COLUMNS = ("timestamp_ns", "timestampns")
_START_TIMESTAMP_COLUMNS = ("start_timestamp", "starttimestamp")
_END_TIMESTAMP_COLUMNS = ("end_timestamp", "endtimestamp")


@dataclass
class _RooflineCompileResult:
    counters: list[CounterRecord]
    rooflines: list[RooflineRecord]
    quality: str
    counter_events: int
    associated_kernels: int
    unassociated_kernels: int
    missing_metrics: list[str]
    error: str = ""


def _timeline_event_args(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        args = event.get("args")
        return dict(args) if isinstance(args, dict) else {}
    args = getattr(event, "args", None)
    return args if isinstance(args, dict) else {}


def _timeline_event_name(event: Any) -> str:
    if isinstance(event, dict):
        for key in ("name", "key"):
            value = event.get(key)
            if value:
                return str(value)
        return "unknown"
    for attr in ("key", "name"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return "unknown"


def _timeline_event_external_id(event: Any) -> int | None:
    value = _timeline_event_args(event).get("External id")
    if value is None and isinstance(event, dict):
        value = event.get("external_id")
    if value is None:
        value = getattr(event, "external_id", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _timeline_event_timestamp_us(event: Any) -> int | None:
    if isinstance(event, dict):
        candidates = (
            event.get("ts"),
            event.get("timestamp_us"),
            event.get("start_time"),
        )
    else:
        candidates = (
            getattr(getattr(event, "time_range", None), "start", None),
            getattr(event, "start_us", None),
            getattr(event, "start_time", None),
            getattr(event, "timestamp_us", None),
            _timeline_event_args(event).get("ts"),
        )
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _timeline_cpu_op_name(event: Any) -> str | None:
    name = _timeline_event_name(event)
    return name if name.startswith(("aten::", "autograd::")) else None


def _timeline_kernel_name(event: Any) -> str | None:
    """Return the kernel name for a Kineto CUDA/ROCm activity event.

    Chrome-trace dicts must be categorized as a kernel. ``FunctionEvent``
    objects mirror the CUDA counter path: any event that is not an
    ``aten::``/``autograd::`` CPU op is treated as a potential kernel, and
    callers ignore events without a usable external id.
    """
    if isinstance(event, dict):
        if str(event.get("cat") or "").lower() != "kernel":
            return None
        name = _timeline_event_name(event)
        return name if name != "unknown" else None
    name = _timeline_event_name(event)
    if not name or name == "unknown":
        return None
    if name.startswith(("aten::", "autograd::")):
        return None
    return name


def _normalize_kernel_name(name: str) -> str:
    """Normalize a rocprof kernel name for matching Kineto kernel events."""
    text = name.strip()
    if text.endswith("(.kd)"):
        text = text[: -len("(.kd)")].strip()
    return text


def _timeline_event_op_stack(event: Any, op_name: str) -> tuple[str, ...]:
    if isinstance(event, dict):
        stack = event.get("stack")
    else:
        stack = getattr(event, "stack", None)
    names: list[str] = []
    if stack is not None:
        for frame in stack:
            if isinstance(frame, str):
                frame_name = frame
            elif isinstance(frame, dict):
                frame_name = frame.get("name") or frame.get("key") or ""
            else:
                frame_name = _timeline_event_name(frame)
            if frame_name:
                names.append(str(frame_name))
    stack = tuple(names) if names else ()
    if op_name not in stack:
        stack = (*stack, op_name)
    return stack


def join_rocm_rows_with_timeline(
    rows: list[dict[str, Any]], timeline_events: list[Any]
) -> list[dict[str, Any]]:
    """Associate ROCm counter rows with Kineto CPU ops.

    Preference order matches ``roofline-backends.zh.md``: keep an explicit
    ``op_name``, otherwise match by ``correlation_id``, then by the kernel
    name recorded in the Kineto timeline, and finally fall back to the
    nearest preceding CPU launch by timestamp. The timestamp fallback only
    runs when the row timestamp lies inside the observed timeline range, so a
    device-clock rocprof timestamp is never joined onto a host-clock Kineto
    trace. Rows are mutated in place and returned.
    """
    cpu_by_external_id: dict[int, list[tuple[str, ...]]] = {}
    external_id_by_kernel_name: dict[str, int] = {}
    launch_stack_by_timestamp: list[tuple[int, tuple[str, ...]]] = []
    for event in timeline_events or []:
        op_name = _timeline_cpu_op_name(event)
        external_id = _timeline_event_external_id(event)
        timestamp = _timeline_event_timestamp_us(event)
        if op_name is not None:
            stack = _timeline_event_op_stack(event, op_name)
            if external_id is not None:
                cpu_by_external_id.setdefault(external_id, []).append(tuple(stack))
            if timestamp is not None:
                launch_stack_by_timestamp.append((timestamp, tuple(stack)))
            continue
        kernel_name = _timeline_kernel_name(event)
        if kernel_name is not None and external_id is not None:
            external_id_by_kernel_name.setdefault(kernel_name, external_id)
            normalized_name = _normalize_kernel_name(kernel_name)
            if normalized_name != kernel_name:
                external_id_by_kernel_name.setdefault(normalized_name, external_id)

    launch_stack_by_timestamp.sort(key=lambda item: item[0])
    timestamps = [item[0] for item in launch_stack_by_timestamp]
    for row in rows:
        if row.get("op_name"):
            continue
        stack: tuple[str, ...] | None = None
        correlation_id = row.get("correlation_id")
        if correlation_id is not None:
            try:
                correlation_id = int(correlation_id)
            except (TypeError, ValueError):
                correlation_id = None
        if correlation_id is not None:
            stacks = cpu_by_external_id.get(correlation_id)
            if stacks:
                stack = stacks[-1]
        if stack is None:
            row_kernel_name = row.get("kernel_name")
            if row_kernel_name:
                row_kernel_name = str(row_kernel_name)
                external_id = external_id_by_kernel_name.get(row_kernel_name)
                if external_id is None:
                    normalized_name = _normalize_kernel_name(row_kernel_name)
                    if normalized_name != row_kernel_name:
                        external_id = external_id_by_kernel_name.get(normalized_name)
                if external_id is not None:
                    stacks = cpu_by_external_id.get(external_id)
                    if stacks:
                        stack = stacks[-1]
        if stack is None and timestamps:
            row_timestamp = row.get("timestamp_us")
            if row_timestamp is not None:
                try:
                    row_timestamp = int(row_timestamp)
                except (TypeError, ValueError):
                    row_timestamp = None
                if (
                    row_timestamp is not None
                    and timestamps[0] <= row_timestamp <= timestamps[-1]
                ):
                    position = bisect_right(timestamps, row_timestamp)
                    if position:
                        stack = launch_stack_by_timestamp[position - 1][1]
        if stack:
            row["op_name"] = stack[-1]
            row["op_stack"] = list(stack)
    return rows


def _coerce_number(value: str) -> int | float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _find_column(headers: list[str], candidates: tuple[str, ...]) -> int | None:
    lowered = {name.lower(): index for index, name in enumerate(headers)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_counter_csv(text: str) -> list[dict[str, Any]]:
    """Parse legacy and rocprofv2 counter CSV into normalized rows."""
    if not text.strip():
        return []
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    headers = [cell.strip() for cell in rows[0]]
    kernel_column = _find_column(headers, ("kernelname", "kernel_name"))
    if kernel_column is None:
        raise ValueError("rocm CSV is missing a KernelName column")
    duration_column = _find_column(headers, _DURATION_COLUMNS)
    op_column = _find_column(headers, ("opname", "op_name"))
    correlation_column = _find_column(headers, ("correlation_id", "correlationid"))
    calls_column = _find_column(headers, ("calls",))
    timestamp_column = _find_column(headers, _TIMESTAMP_COLUMNS)
    timestamp_ns_column = _find_column(headers, _TIMESTAMP_NS_COLUMNS)
    start_timestamp_column = _find_column(headers, _START_TIMESTAMP_COLUMNS)
    end_timestamp_column = _find_column(headers, _END_TIMESTAMP_COLUMNS)
    metadata_columns = {
        kernel_column,
        duration_column,
        op_column,
        correlation_column,
        calls_column,
        timestamp_column,
        timestamp_ns_column,
        start_timestamp_column,
        end_timestamp_column,
    }
    metric_columns = [
        (index, name)
        for index, name in enumerate(headers)
        if index not in metadata_columns
        and name.strip().lower() not in _IGNORED_CSV_COLUMNS
    ]

    # rocprofv2 repeats each configured counter once per hardware instance in
    # the data rows while the header only names the counter once. Detect that
    # shape explicitly; legacy artifacts keep one cell per named column.
    modern_counter_names: list[str] = []
    counter_start: int | None = None
    if start_timestamp_column is not None and correlation_column is not None:
        counter_start = correlation_column + 1
        for index, name in enumerate(headers[counter_start:], start=counter_start):
            if name.strip() and name.strip().lower() not in _IGNORED_CSV_COLUMNS:
                modern_counter_names.append(name.strip())
        if not modern_counter_names:
            counter_start = None
            modern_counter_names = []

    if modern_counter_names:
        metric_columns = [
            (counter_start + i, name)
            for i, name in enumerate(modern_counter_names)
        ]
    if not metric_columns:
        raise ValueError("rocm CSV has no metric columns")

    parsed: list[dict[str, Any]] = []
    for row in rows[1:]:
        padded = list(row) + [""] * (len(headers) - len(row))
        kernel_name = padded[kernel_column].strip()
        if not kernel_name:
            continue

        metrics: dict[str, int | float] = {}
        if modern_counter_names and counter_start is not None:
            counter_cells = padded[counter_start:]
            names_count = len(modern_counter_names)
            if len(counter_cells) % names_count == 0:
                for index, cell in enumerate(counter_cells):
                    value = _coerce_number(cell)
                    if value is None:
                        continue
                    name = modern_counter_names[index % names_count]
                    metrics[name] = metrics.get(name, 0) + value
            else:
                # Non-replicated or truncated tail: match named columns by
                # position and skip the unaligned remainder rather than
                # misattributing counters.
                for offset, name in enumerate(modern_counter_names):
                    value = _coerce_number(
                        counter_cells[offset] if offset < len(counter_cells) else ""
                    )
                    if value is not None:
                        metrics[name] = value
        else:
            for index, name in metric_columns:
                value = _coerce_number(padded[index])
                if value is not None:
                    metrics[name] = value

        duration_ns = None
        if start_timestamp_column is not None and end_timestamp_column is not None:
            start_value = _coerce_number(padded[start_timestamp_column])
            end_value = _coerce_number(padded[end_timestamp_column])
            if start_value is not None and end_value is not None:
                duration_ns = max(int(end_value) - int(start_value), 0)
        elif duration_column is not None:
            duration_ns = _coerce_number(padded[duration_column])
        op_name = padded[op_column].strip() if op_column is not None else ""
        correlation_id = None
        if correlation_column is not None and padded[correlation_column].strip():
            correlation_id = _coerce_number(padded[correlation_column])
        calls = (
            _positive_int(padded[calls_column])
            if calls_column is not None and padded[calls_column].strip()
            else 1
        )
        timestamp_us = _timestamp_us_from_columns(
            padded,
            timestamp_column,
            timestamp_ns_column,
            start_timestamp_column,
        )
        parsed.append(
            {
                "kernel_name": kernel_name,
                "op_name": op_name,
                "correlation_id": correlation_id,
                "duration_ns": duration_ns,
                "timestamp_us": timestamp_us,
                "calls": calls,
                "metrics": metrics,
            }
        )
    return parsed


def _parse_document(document: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("sidecar document must be a JSON object")
    if document.get("format") != SIDECAR_FORMAT:
        raise ValueError(
            f"unsupported sidecar format {document.get('format')!r}; "
            f"expected {SIDECAR_FORMAT!r}"
        )
    counters = document.get("counters")
    if not isinstance(counters, list):
        raise ValueError("sidecar document must contain a 'counters' array")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(counters):
        if not isinstance(item, dict):
            raise ValueError(f"counter row {index} must be an object")
        kernel_name = item.get("kernel_name")
        if not isinstance(kernel_name, str) or not kernel_name.strip():
            raise ValueError(f"counter row {index} is missing kernel_name")
        metrics = item.get("metrics")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise ValueError(f"counter row {index} metrics must be an object")
        rows.append(
            {
                "kernel_name": kernel_name.strip(),
                "op_name": item.get("op_name") or "",
                "correlation_id": item.get("correlation_id"),
                "duration_ns": item.get("duration_ns"),
                "timestamp_us": _row_timestamp_us(item),
                "calls": _positive_int(item.get("calls")),
                "metrics": metrics,
            }
        )
    return rows


def parse_counter_artifact(payload: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a sidecar artifact and return per-kernel rows.

    Accepts a JSON string, a JSON document, a normalized CSV string, or an
    already-normalized list of row dicts.
    """
    if isinstance(payload, str):
        text = payload.lstrip(chr(0xFEFF)).strip()
        if text.startswith("{"):
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                return parse_counter_csv(payload)
            return _parse_document(document)
        return parse_counter_csv(payload)
    if isinstance(payload, dict):
        return _parse_document(payload)
    if isinstance(payload, (list, tuple)) and all(
        isinstance(row, dict) for row in payload
    ):
        return _parse_document({"format": SIDECAR_FORMAT, "counters": list(payload)})
    raise ValueError("sidecar payload must be JSON, CSV, or a list of row dicts")


def load_timeline_events(path: str) -> list[dict[str, Any]]:
    """Load Kineto timeline events from a JSON file for offline association.

    Accepts either a Chrome-trace document (``{"traceEvents": [...]}``) or a
    bare list of event objects. Each returned element is a plain dict that
    ``join_rocm_rows_with_timeline`` can consume: ``name`` selects CPU ops,
    ``ts`` (microseconds) supplies the host launch timestamp, and
    ``args``/``external_id``/``stack`` are optional.
    """
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            payload = handle.read()
    except OSError:
        return []
    text = payload.lstrip(chr(0xFEFF)).strip()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(document, dict) and isinstance(document.get("traceEvents"), list):
        return [item for item in document["traceEvents"] if isinstance(item, dict)]
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    return []


def _timestamp_us_from_columns(
    padded: list[str],
    timestamp_column: int | None,
    timestamp_ns_column: int | None,
    start_timestamp_column: int | None = None,
) -> int | None:
    if start_timestamp_column is not None:
        start_value = _coerce_number(padded[start_timestamp_column])
        if start_value is not None and start_value >= 0:
            return int(start_value // 1000)
    if timestamp_ns_column is not None:
        value = _coerce_number(padded[timestamp_ns_column])
        if value is not None and value >= 0:
            return int(value // 1000)
    if timestamp_column is not None:
        value = _coerce_number(padded[timestamp_column])
        if value is not None and value >= 0:
            return int(value)
    return None


def _row_timestamp_us(item: dict[str, Any]) -> int | None:
    if item.get("timestamp_ns") is not None:
        value = _coerce_number(str(item.get("timestamp_ns")))
        if value is not None and value >= 0:
            return int(value // 1000)
    if item.get("timestamp_us") is not None:
        value = _coerce_number(str(item.get("timestamp_us")))
        if value is not None and value >= 0:
            return int(value)
    return None


def _duration_us(duration_ns: Any) -> int | None:
    if duration_ns is None:
        return None
    try:
        value = float(duration_ns)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return int(value // 1000)


def build_counter_records(
    rows: list[dict[str, Any]],
    *,
    capture_id: str,
    local_step: int,
    global_step: int,
    rank: int,
    role: str,
) -> tuple[list[CounterRecord], list[str]]:
    """Compile normalized rows into profile_counter fact rows.

    Returns (counters, missing_metrics). flops is ``None`` unless the operator
    supplies ``PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON``.
    """
    counters: list[CounterRecord] = []
    missing_metrics: set[str] = set()
    flop_weights = rocm_flop_weights()
    for row in rows:
        metrics = row["metrics"]
        dram_bytes = rocm_dram_bytes(metrics)
        if dram_bytes is None:
            missing_metrics.update(rocm_missing_metrics(metrics, ROCM_DRAM_METRICS))
        if flop_weights:
            missing_metrics.update(rocm_missing_metrics(metrics, list(flop_weights)))
        op_name = row.get("op_name") or ""
        op_stack = _normalize_op_stack(row.get("op_stack"), op_name)
        top_level_op = op_stack[0] if op_stack else op_name
        bottom_level_op = op_stack[-1] if op_stack else op_name
        counters.append(
            CounterRecord(
                capture_id=capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                role=role,
                kernel_name=row["kernel_name"],
                op_name=op_name,
                top_level_op=top_level_op,
                bottom_level_op=bottom_level_op,
                op_stack=json.dumps(op_stack, ensure_ascii=False),
                calls=_positive_int(row.get("calls")),
                duration_us=_duration_us(row.get("duration_ns")),
                flops=rocm_instruction_flops(metrics, flop_weights),
                dram_bytes=dram_bytes,
                metrics=json.dumps(metrics, separators=(",", ":")),
            )
        )
    return counters, sorted(missing_metrics)


def _normalize_op_stack(raw: Any, op_name: str) -> list[str]:
    """Coerce an ``op_stack`` value to a list of strings.

    Artifacts normally provide a JSON list, but a JSON string can leak through
    from hand-built rows; ``list("aten::mm")`` would otherwise split into
    characters.
    """
    if raw is None:
        stack: list[str] = []
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [raw]
        stack = [str(item) for item in parsed] if isinstance(parsed, list) else [raw]
    elif isinstance(raw, (list, tuple)):
        stack = [str(item) for item in raw]
    else:
        stack = [str(raw)]
    return stack or ([op_name] if op_name else [])
