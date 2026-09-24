"""Real-DCU validation spike for the ROCm whole-run roofline path.

This is not a unit test; it runs on a ROCm/DCU node with ``probing`` installed.

Two validation steps are supported:

1. Render the launch wrapper (helps the operator run the training command
   under ``rocprof``)::

       python -m probing.profiling.torch_profiler.rocm_e2e_spike --wrap-cmd \
           --pmc /path/to/pmc.txt --out-dir /path/to/artifacts/rank0 \
           --app "python bench_train.py"

2. Import offline counter artifacts and compile roofline rows::

       python -m probing.profiling.torch_profiler.rocm_e2e_spike \
           --artifact-dir /path/to/artifacts --rank 0

Use ``--list-counters`` to check whether the required metric names exist in
``rocprofv2 --list-counters`` before attempting a capture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from typing import Any


def _required_metrics() -> tuple[str, ...]:
    from probing.profiling.torch_profiler.rocm_metrics import ROCM_DEFAULT_METRICS
    from probing.profiling.torch_profiler.config import load_roofline_config

    config = load_roofline_config()
    return config.metrics or ROCM_DEFAULT_METRICS


def _detect() -> tuple[dict[str, Any], Any]:
    import torch

    from probing.profiling.torch_profiler.backends import detect_backend

    info = detect_backend(torch)
    return asdict(info), torch


def _backend() -> tuple[Any, Any, dict[str, Any]]:
    from probing.profiling.torch_profiler.backends import BackendInfo, RocmRooflineBackend

    info_dict, torch_module = _detect()
    backend = RocmRooflineBackend(BackendInfo(**info_dict))
    return backend, torch_module, info_dict


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


def render_wrap_cmd(pmc: str, out_dir: str, app: str) -> dict[str, Any]:
    from probing.profiling.torch_profiler.rocm_runner import (
        artifact_root,
        sidecar_command,
        sidecar_enabled,
        wrap_command,
    )

    report: dict[str, Any] = {
        "enabled": sidecar_enabled(),
        "template": sidecar_command(),
        "artifact_root_default": artifact_root(),
    }
    rendered = wrap_command(pmc=pmc or None, output=out_dir or None, app=app or None)
    if rendered is None:
        report["error"] = "no rocprof wrapper command template configured"
    else:
        report["wrap_command"] = rendered
    return report


def run_import(artifact_dir: str, rank: int) -> dict[str, Any]:
    from probing.profiling.torch_profiler.rocm_runner import import_artifact_rows

    backend, torch_module, info_dict = _backend()
    capability = backend.probe(torch_module)
    report: dict[str, Any] = {
        "backend": info_dict,
        "capability": asdict(capability),
        "artifact_dir": artifact_dir,
        "rank": rank,
    }

    rows, import_error = import_artifact_rows(artifact_dir, rank, finalized=True)
    report["import_error"] = import_error
    report["parsed_rows"] = len(rows)
    report["first_rows"] = rows[:5]
    if import_error:
        return report

    result = backend.compile_counter_rows(
        rows,
        capture_id="rocm-e2e-spike",
        local_step=0,
        global_step=0,
        rank=rank,
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
    parser.add_argument(
        "--wrap-cmd",
        action="store_true",
        help="render the rocprof wrapper command and exit",
    )
    parser.add_argument("--pmc", default="", help="metrics file for --wrap-cmd")
    parser.add_argument("--out-dir", default="", help="output directory for --wrap-cmd")
    parser.add_argument("--app", default="", help="wrapped application for --wrap-cmd")
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="import counter artifacts from this directory root",
    )
    parser.add_argument("--rank", type=int, default=0, help="rank subdirectory to import")
    parser.add_argument("--output", default="", help="write JSON report to this path")
    args = parser.parse_args(argv)

    if args.list_counters:
        report = list_counters(args.list_cmd)
    elif args.wrap_cmd:
        report = render_wrap_cmd(args.pmc, args.out_dir, args.app)
    elif args.artifact_dir:
        report = run_import(args.artifact_dir, args.rank)
    else:
        parser.error("one of --list-counters, --wrap-cmd, or --artifact-dir is required")

    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
