"""ROCm whole-run wrapper rendering and offline counter-artifact importer.

DTK ``rocprof`` / ``rocprofv2`` can only wrap-launch an application; they do
not attach to a running training process. The v1 model therefore splits the
pipeline in two:

* The operator or a launcher renders ``wrap_command()`` and runs

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
    FINALIZED_ENV,
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


def artifact_finalized() -> bool:
    """Return whether the whole-run artifact is confirmed finalized.

    In v1 the wrapper collects for the entire run, so an in-process
    ``profile/start`` finalize may race a still-writing ``rocprof``. Import
    must be gated on an explicit ``PROBING_TORCH_ROOFLINE_FINALIZED=1`` (or
    config ``finalized: true``) signal; otherwise the offline ``rocm_e2e_spike
    --artifact-dir`` path is the post-run import surface.
    """
    config = load_roofline_config()
    if config.finalized:
        return True
    return os.environ.get(FINALIZED_ENV, "").strip().lower() in {
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
    """Return candidate artifact files for ``rank``, unsorted.

    Files under ``artifact_root_value/rank{rank}`` take precedence; when that
    directory has no artifacts the caller falls back to
    ``artifact_root_value`` (single-rank benches). Ordering is left to the
    caller, which sorts by mtime before import.
    """
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
    finalized: Optional[bool] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Discover, parse, and clean up a finalized ROCm counter artifact.

    Returns ``(rows, error)``. ``error`` is empty on success. Unless
    ``finalized`` is explicitly true (or ``PROBING_TORCH_ROOFLINE_FINALIZED=1``),
    import refuses to run: the v1 wrapper collects for the whole run, so an
    in-process finalize could otherwise read a half-written file and silently
    surface truncated counters as ``partial``.

    Candidates are tried newest-first; a candidate that fails to parse or
    yields no rows is skipped so a stray component file (e.g. a ``.json``
    alongside ``results.csv``) does not mask the real counter artifact. On
    success the containing launch directory is removed unless retention is
    requested.
    """
    if not artifact_root_value:
        return [], "rocm roofline artifact directory is not configured"

    if finalized is None:
        finalized = artifact_finalized()
    if not finalized:
        return [], (
            "rocm counter artifact is not finalized yet; the whole-run wrapper "
            "may still be writing. Import after collection via "
            "``rocm_e2e_spike --artifact-dir``, or set "
            "PROBING_TORCH_ROOFLINE_FINALIZED=1 once collection has finished."
        )

    files = discover_artifact_files(artifact_root_value, rank)
    if not files:
        return [], (
            f"no rocm counter artifact directory found under "
            f"{artifact_root_value}/rank{rank}"
        )

    ordered = sorted(files, key=os.path.getmtime, reverse=True)
    rows: list[dict[str, Any]] = []
    consumed: Optional[str] = None
    last_error = ""
    for path in ordered:
        try:
            payload = open(path, encoding="utf-8").read()
        except OSError as exc:
            last_error = f"failed to read rocprofiler artifact {os.path.basename(path)}: {exc}"
            continue
        try:
            rows = parse_counter_artifact(payload)
        except (ValueError, TypeError) as exc:
            last_error = f"rocm artifact parse failed: {exc}"
            continue
        if rows:
            consumed = path
            break
        last_error = f"rocprofiler artifact {os.path.basename(path)} contained no counter rows"

    if consumed is None:
        return [], last_error or "rocm counter artifacts contained no counter rows"

    remove = keep_artifacts() if keep is None else keep
    if not remove:
        parent = os.path.dirname(consumed)
        if os.path.abspath(parent) != os.path.abspath(artifact_root_value):
            shutil.rmtree(parent, ignore_errors=True)
        else:
            for path in files:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return rows, ""
