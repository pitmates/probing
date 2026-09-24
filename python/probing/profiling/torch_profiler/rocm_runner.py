"""Experimental ROCm rocprofiler sidecar session runner.

Runs a rocprofiler command for the duration of one capture, collects the
normalized JSON/CSV artifact, and removes the temporary output directory. This
owns the launch / collect / cleanup half of the ROCm roofline pipeline.
``ProfilerController`` starts and collects one session per capture window for
the rocm backend.

The command template lives in ``PROBING_TORCH_ROOFLINE_CONFIG`` (or the legacy
``PROBING_TORCH_ROOFLINE_ROCPROF_CMD`` env); when neither is set a conservative
default is used. Its ``{output}`` placeholder is replaced with a per-session
temp directory and ``{pid}`` with the current process id. A failing command
degrades to an explicit error string instead of fabricating counter rows.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from typing import Any, Optional

from .rocm_sidecar import parse_counter_artifact
from .config import load_roofline_config

_ARTIFACT_SUFFIXES = {".json", ".csv", ".jsonl"}
_TERMINATE_TIMEOUT_S = 5.0

DEFAULT_ROCM_ROCPROF_CMD = "rocprof --output {output} --basenames on --stats"


def sidecar_enabled() -> bool:
    config = load_roofline_config()
    if not config.enabled:
        return False
    legacy = os.environ.get("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "").strip().lower()
    return legacy not in {"0", "false", "no", "off"}


def sidecar_command() -> Optional[str]:
    config = load_roofline_config()
    if config.rocprof_cmd:
        return config.rocprof_cmd
    legacy = os.environ.get("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", "").strip()
    if legacy:
        return legacy
    return DEFAULT_ROCM_ROCPROF_CMD


class RocmSidecarSession:
    """One rocprofiler run: start a subprocess, collect its artifact, clean up."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._outdir: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._proc is not None

    def start(self) -> str:
        """Launch the configured command; return "" on success or a diagnostic."""
        if self._proc is not None:
            return "rocm roofline sidecar is already running"
        if not sidecar_enabled():
            return (
                "rocm roofline sidecar is disabled; enable it via "
                "PROBING_TORCH_ROOFLINE_CONFIG (enabled=true)"
            )
        command = sidecar_command()
        if command is None:
            return "rocm roofline sidecar has no rocprof command template"
        try:
            outdir = tempfile.mkdtemp(prefix="probing-rocm-artifacts-")
            rendered = command.replace("{output}", shlex.quote(outdir))
            rendered = rendered.replace("{pid}", str(os.getpid()))
            popen_kwargs: dict[str, Any] = {
                "shell": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            self._proc = subprocess.Popen(rendered, **popen_kwargs)
            self._outdir = outdir
            return ""
        except Exception as exc:
            self.cleanup()
            return f"failed to launch rocprofiler sidecar: {exc}"

    def collect(self, timeout_s: float = 30.0) -> tuple[list[dict[str, Any]], str]:
        """Stop the sidecar, parse its latest artifact, and clean up temp files."""
        if self._proc is None:
            return [], "rocm roofline sidecar was not started"
        proc = self._proc
        outdir = self._outdir or ""
        try:
            if proc.poll() is None:
                self._terminate(proc)
            stdout, stderr = proc.communicate(timeout=timeout_s)
            if proc.returncode != 0:
                detail = (stderr or "").strip() or (stdout or "").strip()
                return [], f"rocprofiler exited with {proc.returncode}: {detail[:400]}"
            files = _artifact_files(outdir)
            if not files:
                return [], "rocprofiler produced no JSON/CSV counter artifacts"
            latest = max(files, key=os.path.getmtime)
            try:
                payload = open(latest, encoding="utf-8").read()
            except OSError as exc:
                return [], f"failed to read rocprofiler artifact {os.path.basename(latest)}: {exc}"
            rows = parse_counter_artifact(payload)
            if not rows:
                return [], (
                    f"rocprofiler artifact {os.path.basename(latest)} "
                    "contained no counter rows"
                )
            return rows, ""
        except subprocess.TimeoutExpired:
            return [], f"rocprofiler did not stop within {timeout_s}s"
        except Exception as exc:
            return [], f"rocprofiler artifact collection failed: {exc}"
        finally:
            self.cleanup()

    def _terminate(self, proc: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
        else:
            proc.terminate()

    def cleanup(self) -> None:
        proc = self._proc
        self._proc = None
        outdir = self._outdir
        self._outdir = None
        if proc is not None and proc.poll() is None:
            self._terminate(proc)
            try:
                proc.wait(timeout=_TERMINATE_TIMEOUT_S)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if outdir:
            shutil.rmtree(outdir, ignore_errors=True)


def _artifact_files(outdir: str) -> list[str]:
    if not outdir or not os.path.isdir(outdir):
        return []
    found: list[str] = []
    for root, _dirs, names in os.walk(outdir):
        for name in names:
            if os.path.splitext(name)[1].lower() in _ARTIFACT_SUFFIXES:
                found.append(os.path.join(root, name))
    return found
