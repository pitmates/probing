"""Offline ROCm counter-artifact parser for the roofline backend.

This module is the in-process contract half: it validates and normalizes
counter artifacts into python.profile_counter fact rows. Artifact producers
(the experimental rocprofiler sidecar and its temp-file lifecycle) are not
implemented here or anywhere in the current phase, so inputs today come from
unit fixtures and direct caller payloads only. This module does not launch
rocprofiler, manage subprocesses, clean up files, or perform cluster fan-out.

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

CSV format mirrors the legacy rocprof CSV layout::

    Index,KernelName,correlation_id,DurationNs,TCC_EA_RDREQ_32B,TCC_EA_RDREQ,TCC_EA_WRREQ_64B,TCC_EA_WRREQ
    0,gemm_kernel,41,12345,10,12,8,16

The first row is the header. A KernelName column is required; correlation_id,
calls, DurationNs, and an optional op_name column are recognized, and every
remaining non-metadata column is treated as a numeric counter. Empty or
non-numeric counter cells are recorded as missing metrics rather than failing
the whole artifact. The CSV column contract is still draft and must be
revalidated against a real rocprof sample before it can graduate past
experimental.

FLOPs are intentionally None until instruction weights are calibrated; only
DRAM bytes (via rocm_metrics.rocm_dram_bytes) and raw counter values are
materialized. correlation_id is preserved on rows when the artifact carries it
so association can prefer it over name-only or timestamp fallbacks.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .rocm_metrics import ROCM_DRAM_METRICS, rocm_dram_bytes
from .session_store import CounterRecord

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
}

_DURATION_COLUMNS = ("durationns", "duration_ns", "duration", "kernelduration")


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
    """Parse legacy rocprof-style CSV into normalized per-kernel rows."""
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
    metric_columns = [
        (index, name)
        for index, name in enumerate(headers)
        if index
        not in {kernel_column, duration_column, op_column, correlation_column, calls_column}
        and name.strip().lower() not in _IGNORED_CSV_COLUMNS
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
        for index, name in metric_columns:
            value = _coerce_number(padded[index])
            if value is not None:
                metrics[name] = value
        duration_ns = None
        if duration_column is not None:
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
        parsed.append(
            {
                "kernel_name": kernel_name,
                "op_name": op_name,
                "correlation_id": correlation_id,
                "duration_ns": duration_ns,
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

    Returns (counters, missing_metrics). flops stays None because ROCm
    instruction-to-FLOP weights are not calibrated yet.
    """
    counters: list[CounterRecord] = []
    missing_metrics: set[str] = set()
    for row in rows:
        metrics = row["metrics"]
        dram_bytes = rocm_dram_bytes(metrics)
        if dram_bytes is None:
            missing_metrics.update(
                name for name in ROCM_DRAM_METRICS if name not in metrics
            )
        op_name = row.get("op_name") or ""
        op_stack = [op_name] if op_name else []
        counters.append(
            CounterRecord(
                capture_id=capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                role=role,
                kernel_name=row["kernel_name"],
                op_name=op_name,
                top_level_op=op_name,
                bottom_level_op=op_name,
                op_stack=json.dumps(op_stack, ensure_ascii=False),
                calls=_positive_int(row.get("calls")),
                duration_us=_duration_us(row.get("duration_ns")),
                flops=None,
                dram_bytes=dram_bytes,
                metrics=json.dumps(metrics, separators=(",", ":")),
            )
        )
    return counters, sorted(missing_metrics)
