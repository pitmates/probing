"""Single-command ROCm whole-run roofline launcher.

DTK ``rocprof`` / ``rocprofv2`` cannot attach to a running process, so the
operator has to wrap the training launch command. This entry point collapses
that wrapping into one command: the only thing the user supplies is the
training command itself. The metrics file, output directory, and rank mapping
come from the roofline config with defaults. By default the generated
``-i`` counter file adds ``range: 0:<dispatch_cap>`` (2000) and the command
injects ``--flush-interval <ms>`` (1000), bounding whole-run collection to the
first dispatches instead of writing one counter row per kernel for the entire
run. Set the ``dispatch_cap`` / ``flush_interval_ms`` keys in
``PROBING_TORCH_ROOFLINE_CONFIG`` to tune, or ``dispatch_cap=0`` to collect the
whole run.

Usage::

    PROBING_TORCH_ROOFLINE_ARTIFACT_DIR=/path/to/artifacts \
        probing-roofline -- python train.py

Short collection window (requires the training script to honor ``--steps``)::

    PROBING_TORCH_ROOFLINE_ARTIFACT_DIR=/path/to/artifacts \
        probing-roofline --steps 20 -- python train.py --steps 500

Instruction counters are opt-in; the default metric set is DRAM-only::

    PROBING_TORCH_ROOFLINE_ARTIFACT_DIR=/path/to/artifacts \
        probing-roofline --with-instruction-counters -- python train.py

``torchrun`` / ``python -m torch.distributed.run`` launches are detected
automatically and wrapped per rank (single-node multi-GPU)::

    PROBING_TORCH_ROOFLINE_ARTIFACT_DIR=/path/to/artifacts \
        probing-roofline -- torchrun --nproc_per_node=4 train.py

Without installing the console script::

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
from .rocm_metrics import ROCM_DEFAULT_METRICS, ROCM_INSTRUCTION_METRICS
from .rocm_runner import (
    DEFAULT_ROCM_ROCPROF_CMD,
    inject_flush_interval,
    inject_trace_period,
    sidecar_command,
    sidecar_enabled,
    wrap_command,
)

DEFAULT_PMC_FILENAME = "rocm_pmc_default.txt"
RANK_WRAPPER_FILENAME = "rocm_rank_wrap.sh"
DEFAULT_ARTIFACT_DIRNAME = "roofline_artifacts"

# torchrun options that consume a following value. ``--opt=value`` is handled
# separately because it never consumes the next token. Flag options (for
# example ``--standalone`` / ``--no_python``) are intentionally absent.
TORCHRUN_VALUE_OPTIONS = frozenset(
    {
        "--nnodes",
        "--nproc-per-node",
        "--nproc_per_node",
        "--rdzv-backend",
        "--rdzv-endpoint",
        "--rdzv-id",
        "--rdzv-conf",
        "--max-restarts",
        "--monitor-interval",
        "--start-method",
        "--role",
        "--tee",
        "--local-addr",
        "--log-dir",
        "--redirects",
        "--master-addr",
        "--master-port",
        "--node-rank",
    }
)


def default_pmc_text(
    metrics: Sequence[str], dispatch_cap: Optional[int] = None
) -> str:
    """Render the ``rocprof -i`` counter file for a metric list.

    ``dispatch_cap`` bounds the number of kernel dispatches recorded by adding
    one ``range: 0:<cap>`` line. The input file keeps exactly one ``pmc:`` row,
    so the counters are still collected in a single hardware pass; the range
    only suppresses counter rows for later dispatches. A non-positive value
    disables the cap.
    """
    if dispatch_cap is None:
        dispatch_cap = load_roofline_config().dispatch_cap
    lines = ["pmc: " + " ".join(metrics)]
    if dispatch_cap and dispatch_cap > 0:
        lines.append(f"range: 0:{int(dispatch_cap)}")
    return "\n".join(lines) + "\n"


def output_dir(artifact_dir: str, rank: int, launch_ts: str) -> str:
    """Return the per-rank launch output directory used by offline import."""
    return os.path.join(artifact_dir, f"rank{rank}", launch_ts)


def apply_steps_arg(app_args: Sequence[str], steps: int) -> list[str]:
    """Return ``app_args`` with ``--steps <steps>`` replaced or appended.

    The wrapper cannot stop ``rocprof`` mid-run; it relies on the wrapped
    training script honoring ``--steps`` so collection ends after a short
    window. Existing ``--steps`` and ``--steps=<value>`` arguments are
    replaced, otherwise the flag is appended.
    """
    tokens = list(app_args)
    if steps <= 0:
        return tokens
    result: list[str] = []
    replaced = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--steps":
            result.extend(["--steps", str(steps)])
            index += 2
            replaced = True
        elif token.startswith("--steps="):
            result.append(f"--steps={steps}")
            index += 1
            replaced = True
        else:
            result.append(token)
            index += 1
    if not replaced:
        result.extend(["--steps", str(steps)])
    return result


def _resolve_pmc(
    artifact_dir: str,
    pmc_override: str,
    metrics: Sequence[str],
    dispatch_cap: Optional[int] = None,
) -> str:
    if pmc_override:
        return pmc_override
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, DEFAULT_PMC_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(default_pmc_text(metrics, dispatch_cap=dispatch_cap))
    return path


def build_wrapped_command(
    artifact_dir: str,
    rank: int,
    launch_ts: str,
    app_args: Sequence[str],
    *,
    pmc: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    dispatch_cap: Optional[int] = None,
    flush_interval_ms: Optional[int] = None,
    trace_period: Optional[str] = None,
) -> Optional[str]:
    """Render the whole-run wrapper command, creating a default pmc if needed."""
    if not app_args:
        raise ValueError("no application command supplied after --")
    metrics = tuple(metrics or ROCM_DEFAULT_METRICS)
    pmc_path = _resolve_pmc(artifact_dir, pmc or "", metrics, dispatch_cap=dispatch_cap)
    out_dir = output_dir(artifact_dir, rank, launch_ts)
    os.makedirs(out_dir, exist_ok=True)
    app = shlex.join(list(app_args))
    return wrap_command(
        pmc=pmc_path,
        output=out_dir,
        app=app,
        flush_interval_ms=flush_interval_ms,
        trace_period=trace_period,
    )


def is_torchrun_command(tokens: Sequence[str]) -> bool:
    """Return whether ``tokens`` is a torchrun-style launcher command."""
    if not tokens:
        return False
    head = tokens[0]
    if os.path.basename(head) == "torchrun":
        return True
    return (
        head in {"python", "python3"}
        and len(tokens) >= 3
        and tokens[1] == "-m"
        and tokens[2] == "torch.distributed.run"
    )


def _strip_torchrun_launcher(
    tokens: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Return ``(launcher_prefix, torchrun_options_and_script)``."""
    head = tokens[0]
    if os.path.basename(head) == "torchrun":
        return [head], list(tokens[1:])
    if (
        head in {"python", "python3"}
        and len(tokens) >= 3
        and tokens[1] == "-m"
        and tokens[2] == "torch.distributed.run"
    ):
        return list(tokens[:3]), list(tokens[3:])
    raise ValueError("expected a torchrun or python -m torch.distributed.run command")


def split_torchrun_command(
    tokens: Sequence[str],
) -> tuple[list[str], str, list[str]]:
    """Split torchrun args into ``(options, training_script, script_args)``."""
    opts: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            i += 1
            break
        if token.startswith("-") and token != "-":
            opts.append(token)
            if "=" not in token and token in TORCHRUN_VALUE_OPTIONS:
                i += 1
                if i < len(tokens):
                    opts.append(tokens[i])
        else:
            break
        i += 1
    if i >= len(tokens):
        raise ValueError("no training script found in torchrun command")
    return opts, tokens[i], list(tokens[i + 1 :])


def render_rank_wrapper(
    *,
    script: str,
    python_executable: str,
    pmc: str,
    artifact_dir: str,
    launch_ts: str,
    flush_interval_ms: Optional[int] = None,
    trace_period: Optional[str] = None,
) -> str:
    """Render the per-rank shell wrapper torchrun executes via ``--no_python``."""
    command = sidecar_command() or DEFAULT_ROCM_ROCPROF_CMD
    app_ref = f"{shlex.quote(python_executable)} -u {shlex.quote(script)} \"$@\""
    rocprof_line = command.replace("{pmc}", shlex.quote(pmc)).replace(
        "{output}", '"$OUT"'
    )
    rocprof_line = inject_flush_interval(rocprof_line, flush_interval_ms)
    if trace_period is None:
        trace_period = load_roofline_config().trace_period
    rocprof_line = inject_trace_period(rocprof_line, trace_period)
    rocprof_line = rocprof_line.replace("{app}", app_ref)
    out_assign = f"OUT={shlex.quote(artifact_dir)}/rank$RANK/{shlex.quote(launch_ts)}"
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            'RANK="${RANK:-${LOCAL_RANK:-0}}"',
            out_assign,
            'mkdir -p "$OUT"',
            f"exec {rocprof_line}",
            "",
        ]
    )


def build_torchrun_launch(
    app_args: Sequence[str],
    *,
    artifact_dir: str,
    launch_ts: str,
    pmc: str,
    python_executable: Optional[str] = None,
    flush_interval_ms: Optional[int] = None,
    trace_period: Optional[str] = None,
) -> tuple[list[str], str, str]:
    """Return ``(torchrun_command, wrapper_path, wrapper_text)`` for per-rank wrap."""
    if not app_args:
        raise ValueError("no torchrun command supplied after --")
    launcher, rest = _strip_torchrun_launcher(list(app_args))
    opts, script, script_args = split_torchrun_command(rest)
    wrapper_text = render_rank_wrapper(
        script=script,
        python_executable=python_executable or sys.executable,
        pmc=pmc,
        artifact_dir=artifact_dir,
        launch_ts=launch_ts,
        flush_interval_ms=flush_interval_ms,
        trace_period=trace_period,
    )
    wrapper_path = os.path.join(artifact_dir, RANK_WRAPPER_FILENAME)
    command = [*launcher, *opts, "--no_python", wrapper_path, *script_args]
    return command, wrapper_path, wrapper_text


def run_torchrun(
    app_args: Sequence[str],
    *,
    artifact_dir: str,
    launch_ts: str,
    pmc: str,
    dry_run: bool,
    flush_interval_ms: Optional[int] = None,
    trace_period: Optional[str] = None,
) -> int:
    """Write the rank wrapper and launch torchrun, or print it for ``--dry-run``."""
    command, wrapper_path, wrapper_text = build_torchrun_launch(
        app_args,
        artifact_dir=artifact_dir,
        launch_ts=launch_ts,
        pmc=pmc,
        flush_interval_ms=flush_interval_ms,
        trace_period=trace_period,
    )
    if dry_run:
        print(f"# rank wrapper: {wrapper_path}", file=sys.stderr)
        print(wrapper_text, file=sys.stderr)
        print(shlex.join(command), file=sys.stderr)
        return 0
    with open(wrapper_path, "w", encoding="utf-8") as handle:
        handle.write(wrapper_text)
    os.chmod(wrapper_path, 0o755)
    return subprocess.call(command)


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
        "--steps",
        type=int,
        default=0,
        help="replace/append --steps N on the training command to shorten collection",
    )
    parser.add_argument(
        "--with-instruction-counters",
        action="store_true",
        help="add SQ_INSTS_* instruction counters to the default DRAM metric set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rendered rocprof wrapper and exit without running it",
    )
    parser.add_argument("app", nargs=argparse.REMAINDER, help="training command after --")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    # argparse.REMAINDER keeps the ``--`` separator; strip it so ``args.app`` is
    # the actual launcher command (e.g. ``torchrun`` or ``python``).
    app_args = list(args.app)
    if app_args and app_args[0] == "--":
        app_args = app_args[1:]
    if args.steps > 0:
        app_args = apply_steps_arg(app_args, args.steps)

    # The only knob TorchProbe needs to enable this path.
    os.environ.setdefault("PROBING_TORCH_PROFILER_ANALYSIS", "roofline")

    config = load_roofline_config()
    artifact_dir = config.artifact_dir or os.environ.get(ARTIFACT_DIR_ENV, "").strip()
    if not artifact_dir:
        artifact_dir = os.path.join(os.getcwd(), DEFAULT_ARTIFACT_DIRNAME)
        os.environ[ARTIFACT_DIR_ENV] = artifact_dir
        print(
            f"roofline artifact directory defaults to {artifact_dir} "
            f"(override with {ARTIFACT_DIR_ENV})",
            file=sys.stderr,
        )

    if not sidecar_enabled():
        print(
            "rocprof wrapper collection is disabled; set "
            "PROBING_TORCH_ROOFLINE_CONFIG rocm_enabled=true",
            file=sys.stderr,
        )
        return 2

    if config.trace_period:
        print(
            "warning: PROBING_TORCH_ROOFLINE_TRACE_PERIOD / config trace_period "
            "bounds the rocprofv2 trace window, not the --plugin file counter "
            "stream. On DTK 26.04 rocprofv2 this has been observed to produce "
            "an empty pmc_1 directory with no results_*.csv. Prefer a short "
            "--steps run without trace_period for counter roofline.",
            file=sys.stderr,
        )

    metrics = config.metrics or ROCM_DEFAULT_METRICS
    if args.with_instruction_counters:
        metrics = tuple(dict.fromkeys((*metrics, *ROCM_INSTRUCTION_METRICS)))
    launch_ts = args.launch_ts or str(int(time.time()))

    try:
        if is_torchrun_command(app_args):
            pmc_path = _resolve_pmc(
                artifact_dir,
                args.pmc,
                metrics,
                dispatch_cap=config.dispatch_cap,
            )
            return run_torchrun(
                app_args,
                artifact_dir=artifact_dir,
                launch_ts=launch_ts,
                pmc=pmc_path,
                dry_run=args.dry_run,
                flush_interval_ms=config.flush_interval_ms,
                trace_period=config.trace_period,
            )

        rendered = build_wrapped_command(
            artifact_dir,
            args.rank,
            launch_ts,
            app_args,
            pmc=args.pmc,
            metrics=metrics,
            dispatch_cap=config.dispatch_cap,
            flush_interval_ms=config.flush_interval_ms,
            trace_period=config.trace_period,
        )
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
        return subprocess.call(shlex.split(rendered))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
