"""Single-command ROCm whole-run roofline launcher.

DTK ``rocprof`` / ``rocprofv2`` cannot attach to a running process, so the
operator has to wrap the training launch command. This entry point collapses
that wrapping into one command: the only thing the user supplies is the
training command itself. The metrics file, output directory, and rank mapping
come from the roofline config with defaults.

Usage::

    PROBING_TORCH_ROOFLINE_ARTIFACT_DIR=/path/to/artifacts \
        probing-roofline -- torchrun --nproc_per_node=1 train.py

or equivalently without installing the console script::

    python -m probing.profiling.torch_profiler.rocm_wrap -- torchrun ...
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Optional, Sequence

from .config import ARTIFACT_DIR_ENV, load_roofline_config
from .rocm_metrics import ROCM_DEFAULT_METRICS
from .rocm_runner import sidecar_enabled, wrap_command

DEFAULT_PMC_FILENAME = "rocm_pmc_default.txt"


def default_pmc_text(metrics: Sequence[str]) -> str:
    """Render the ``rocprof -i`` counter file for a metric list."""
    return "pmc: " + " ".join(metrics) + "\n"


def output_dir(artifact_dir: str, rank: int, launch_ts: str) -> str:
    """Return the per-rank launch output directory used by offline import."""
    return os.path.join(artifact_dir, f"rank{rank}", launch_ts)


def _resolve_pmc(
    artifact_dir: str, pmc_override: str, metrics: Sequence[str]
) -> str:
    if pmc_override:
        return pmc_override
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, DEFAULT_PMC_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(default_pmc_text(metrics))
    return path


def build_wrapped_command(
    artifact_dir: str,
    rank: int,
    launch_ts: str,
    app_args: Sequence[str],
    *,
    pmc: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Render the whole-run wrapper command, creating a default pmc if needed."""
    if not app_args:
        raise ValueError("no application command supplied after --")
    metrics = tuple(metrics or ROCM_DEFAULT_METRICS)
    pmc_path = _resolve_pmc(artifact_dir, pmc or "", metrics)
    out_dir = output_dir(artifact_dir, rank, launch_ts)
    os.makedirs(out_dir, exist_ok=True)
    app = shlex.join(list(app_args))
    return wrap_command(pmc=pmc_path, output=out_dir, app=app)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probing-roofline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rank", type=int, default=0, help="rank subdirectory (default: 0)")
    parser.add_argument("--launch-ts", default="", help="launch timestamp directory (default: epoch seconds)")
    parser.add_argument("--pmc", default="", help="rocprof -i counter file (default: generated from configured metrics)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rendered rocprof wrapper and exit without running it",
    )
    parser.add_argument("app", nargs=argparse.REMAINDER, help="training command after --")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    # The only knob TorchProbe needs to enable this path.
    os.environ.setdefault("PROBING_TORCH_PROFILER_ANALYSIS", "roofline")

    config = load_roofline_config()
    artifact_dir = config.artifact_dir or os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not artifact_dir:
        print(
            "roofline artifact directory is not configured; set "
            "PROBING_TORCH_ROOFLINE_ARTIFACT_DIR or "
            "PROBING_TORCH_ROOFLINE_CONFIG.artifact_dir",
            file=sys.stderr,
        )
        return 2

    if not sidecar_enabled():
        print(
            "rocprof wrapper collection is disabled; set "
            "PROBING_TORCH_ROOFLINE_CONFIG rocm_enabled=true",
            file=sys.stderr,
        )
        return 2

    metrics = config.metrics or ROCM_DEFAULT_METRICS
    launch_ts = args.launch_ts or str(int(time.time()))

    try:
        rendered = build_wrapped_command(
            artifact_dir,
            args.rank,
            launch_ts,
            args.app,
            pmc=args.pmc,
            metrics=metrics,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if rendered is None:
        print(
            "rocprof wrapper collection is disabled; set "
            "PROBING_TORCH_ROOFLINE_CONFIG rocm_enabled=true",
            file=sys.stderr,
        )
        return 2

    print(rendered, file=sys.stderr)
    if args.dry_run:
        return 0

    argv_run = shlex.split(rendered)
    return subprocess.call(argv_run)


if __name__ == "__main__":
    raise SystemExit(main())
