"""ROCm whole-run wrapper rendering and offline counter-artifact importer.

DTK ``rocprof`` / ``rocprofv2`` can only wrap-launch an application; they do
not attach to a running training process. The v1 model therefore splits the
pipeline in two:

* The launcher renders ``wrap_command()`` and runs

  ``rocprof -i {pmc} --timestamp on -d {artifact_dir}/rank{rank}/{launch_ts} <app>``

  around the training process, producing per-rank counter artifacts for the
  entire run.

* The in-process collector calls ``import_artifact_rows()`` at finalize to find
  the newest artifact under the configured directory, parse it with
  ``rocm_sidecar``, and remove it unless retention is requested.

This module no longer launches an in-window sidecar subprocess.
``sidecar_enabled`` / ``sidecar_command`` are kept as legacy names for
configuration compatibility; they now gate and return the wrapper command
template.
"""

from __future__ import annotations

import os
import shlex
import shutil
from typing import Any, Optional

from .config import (
    ARTIFACT_DIR_ENV,
    KEEP_ARTIFACTS_ENV,
    load_roofline_config,
)
from .rocm_sidecar import parse_counter_artifact

_ARTIFACT_SUFFIXES = {".json", ".csv", ".jsonl"}

DEFAULT_ROCM_ROCPROF_CMD = "rocprof -i {pmc} --timestamp on -d {output} {app}"


def sidecar_enabled() -> bool:
    """Return whether ROCm wrapper collection is enabled (legacy name)."""
    config = load_roofline_config()
    if not config.rocm_enabled:
        return False
    legacy = os.environ.get("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "").strip().lower()
    return legacy not in {"0", "false", "no", "off"}


def sidecar_command() -> Optional[str]:
    """Return the configured wrapper command template (legacy name)."""
    config = load_roofline_config()
    if config.rocprof_cmd:
        return config.rocprof_cmd
    legacy = os.environ.get("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", "").strip()
    if legacy:
        return legacy
    return DEFAULT_ROCM_ROCPROF_CMD


def wrap_command(
    *,
    pmc: Optional[str] = None,
    output: Optional[str] = None,
    app: Optional[str] = None,
) -> Optional[str]:
    """Render the wrapper template with shell-quoted replacements.

    Unset placeholders are left in the template so callers can dry-run partial
    renders for documentation or diagnostics.
    """
    command = sidecar_command()
    if command is None:
        return None
    rendered = command
    for placeholder, value in (
        ("{pmc}", pmc),
        ("{output}", output),
        ("{app}", app),
    ):
        if value is not None:
            rendered = rendered.replace(placeholder, shlex.quote(value))
    return rendered


def artifact_root() -> str:
    """Resolve the artifact directory from config or the dedicated env."""
    config = load_roofline_config()
    if config.artifact_dir:
        return config.artifact_dir
    return os.environ.get(ARTIFACT_DIR_ENV, "").strip()


def keep_artifacts() -> bool:
    """Retain artifacts after import when requested by config/env."""
    config = load_roofline_config()
    if config.keep_artifacts:
        return True
    return os.environ.get(KEEP_ARTIFACTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _artifact_files(outdir: str) -> list[str]:
    if not outdir or not os.path.isdir(outdir):
        return []
    found: list[str] = []
    for root, _dirs, names in os.walk(outdir):
        for name in names:
            if os.path.splitext(name)[1].lower() in _ARTIFACT_SUFFIXES:
                found.append(os.path.join(root, name))
    return found


def discover_artifact_files(artifact_root_value: str, rank: int) -> list[str]:
    """Return the artifact files for ``rank``, newest-first call site order."""
    if not artifact_root_value:
        return []
    roots: list[str] = []
    if rank >= 0:
        roots.append(os.path.join(artifact_root_value, f"rank{rank}"))
    roots.append(artifact_root_value)
    for root in roots:
        files = _artifact_files(root)
        if files:
            return files
    return []


def import_artifact_rows(
    artifact_root_value: str,
    rank: int,
    *,
    keep: Optional[bool] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Discover, parse, and clean up the latest ROCm counter artifact.

    Returns ``(rows, error)``. ``error`` is empty on success. The newest JSON,
    CSV, or JSONL file under ``artifact_root_value/rank{rank}`` (falling back to
    ``artifact_root_value`` for single-rank benches) is parsed; on success the
    containing launch directory is removed unless retention is requested.
    """
    if not artifact_root_value:
        return [], "rocm roofline artifact directory is not configured"

    files = discover_artifact_files(artifact_root_value, rank)
    if not files:
        return [], (
            f"no rocm counter artifact directory found under "
            f"{artifact_root_value}/rank{rank}"
        )

    latest = max(files, key=os.path.getmtime)
    try:
        payload = open(latest, encoding="utf-8").read()
    except OSError as exc:
        return [], f"failed to read rocprofiler artifact {os.path.basename(latest)}: {exc}"

    try:
        rows = parse_counter_artifact(payload)
    except (ValueError, TypeError) as exc:
        return [], f"rocm artifact parse failed: {exc}"

    if not rows:
        return [], (
            f"rocprofiler artifact {os.path.basename(latest)} contained no counter rows"
        )

    remove = keep_artifacts() if keep is None else keep
    if not remove:
        parent = os.path.dirname(latest)
        if os.path.abspath(parent) != os.path.abspath(artifact_root_value):
            shutil.rmtree(parent, ignore_errors=True)
        else:
            for path in files:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return rows, ""
