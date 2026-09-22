"""Real-DCU validation spike for the ROCm roofline sidecar path.

This is not a unit test; it runs on a ROCm/DCU node with ``probing`` installed
and exercises the production code path end to end:

    RocmSidecarSession.start() -> run microbenchmark kernels -> collect()
    -> parse_counter_artifact() -> build_counter_records()
    -> RocmRooflineBackend.compile_counter_rows()

The operator must supply a working rocprofiler command via
``PROBING_TORCH_ROOFLINE_ROCPROF_CMD`` (a shell template with ``{output}`` and
``{pid}`` placeholders), for example::

    rocprof --output {output} --basenames on --stats

Run::

    PROBING_TORCH_ROOFLINE_ROCM_PROFILE=1 \
    PROBING_TORCH_ROOFLINE_ROCPROF_CMD='rocprof --output {output} --basenames on --stats' \
    python -m probing.profiling.torch_profiler.rocm_e2e_spike --workload all --output /tmp/rocm_e2e.json

Use ``--list-counters`` to check whether the required metric names exist in
``rocprofv2 --list-counters`` before attempting a capture.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any


def _required_metrics() -> tuple[str, ...]:
    from probing.profiling.torch_profiler.rocm_metrics import ROCM_DEFAULT_METRICS

    return ROCM_DEFAULT_METRICS


def _detect() -> tuple[dict[str, Any], Any]:
    import torch

    from probing.profiling.torch_profiler.backends import detect_backend

    info = detect_backend(torch)
    return asdict(info), torch


def list_counters(list_cmd: str) -> dict[str, Any]:
    required = _required_metrics()
    try:
        completed = subprocess.run(
            list_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "required": list(required)}

    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    found = [name for name in required if name in output]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "found": found,
        "missing": [name for name in required if name not in found],
    }


def _run_workload(name: str, device: str, dtype: Any) -> None:
    import torch

    if name in {"gemm", "all"}:
        a = torch.randn(4096, 4096, device=device, dtype=dtype)
        b = torch.randn(4096, 4096, device=device, dtype=dtype)
        torch.matmul(a, b)
    if name in {"elementwise", "all"}:
        x = torch.randn(1 << 22, device=device, dtype=dtype)
        _ = x * 2.0
    torch.cuda.synchronize()


def run_capture(
    workload: str,
    timeout_s: float,
    warmup_s: float,
) -> dict[str, Any]:
    import torch

    from probing.profiling.torch_profiler.backends import (
        BackendInfo,
        RocmRooflineBackend,
    )
    from probing.profiling.torch_profiler.rocm_runner import (
        RocmSidecarSession,
        sidecar_command,
        sidecar_enabled,
    )

    info_dict, torch_module = _detect()
    backend_info = BackendInfo(
        vendor=info_dict["vendor"],
        device_model=info_dict["device_model"],
        device_arch=info_dict["device_arch"],
        counter_source=info_dict["counter_source"],
    )
    backend = RocmRooflineBackend(backend_info)
    capability = backend.probe(torch_module)

    report: dict[str, Any] = {
        "backend": info_dict,
        "capability": asdict(capability) if hasattr(capability, "status") else None,
        "workload": workload,
    }

    if not sidecar_enabled():
        report["error"] = "PROBING_TORCH_ROOFLINE_ROCM_PROFILE must be set to 1"
        return report
    if not sidecar_command():
        report["error"] = "PROBING_TORCH_ROOFLINE_ROCPROF_CMD is not set"
        return report

    device = "cuda"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    session = RocmSidecarSession()
    start_error = session.start()
    report["sidecar_start_error"] = start_error
    if start_error:
        return report

    try:
        if warmup_s > 0:
            time.sleep(warmup_s)
        _run_workload(workload, device, dtype)
        rows, collect_error = session.collect(timeout_s=timeout_s)
    finally:
        if session.active:
            session.cleanup()

    report["collect_error"] = collect_error
    report["parsed_rows"] = len(rows)
    report["first_rows"] = rows[:5]

    if collect_error:
        return report

    result = backend.compile_counter_rows(
        rows,
        capture_id="rocm-e2e-spike",
        local_step=0,
        global_step=0,
        rank=0,
        role="",
    )
    report["compile"] = {
        "quality": result.quality,
        "counter_events": result.counter_events,
        "associated_kernels": result.associated_kernels,
        "unassociated_kernels": result.unassociated_kernels,
        "missing_metrics": result.missing_metrics,
        "error": result.error,
    }
    report["counters"] = [asdict(item) for item in result.counters[:20]]
    report["rooflines"] = [asdict(item) for item in result.rooflines[:20]]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-counters",
        action="store_true",
        help="list required counters via rocprofv2 and exit",
    )
    parser.add_argument(
        "--list-cmd",
        default="rocprofv2 --list-counters",
        help="shell command for --list-counters (default: rocprofv2 --list-counters)",
    )
    parser.add_argument("--workload", default="all", choices=["gemm", "elementwise", "all"])
    parser.add_argument("--output", default="", help="write JSON report to this path")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.list_counters:
        report = list_counters(args.list_cmd)
    else:
        report = run_capture(args.workload, args.timeout, args.warmup)

    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
