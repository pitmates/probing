"""Unit tests for the experimental ROCm rocprofiler sidecar session runner."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from probing.profiling.torch_profiler.rocm_runner import (
    RocmSidecarSession,
    sidecar_command,
    sidecar_enabled,
)

DOC = {
    "format": "probing-rocm-sidecar-v1",
    "counters": [
        {
            "kernel_name": "gemm_kernel",
            "duration_ns": 1000,
            "metrics": {
                "TCC_EA_RDREQ_32B": 10,
                "TCC_EA_RDREQ": 15,
                "TCC_EA_WRREQ_64B": 10,
                "TCC_EA_WRREQ": 20,
            },
        }
    ],
}


def _fake_script(tmp_path, marker=None, outcome="ok"):
    lines = ["import json, sys", "out = sys.argv[1]"]
    if marker is not None:
        lines.append("open(" + repr(str(marker)) + ", 'w').write(out)")
    if outcome == "ok":
        lines.append("doc = " + json.dumps(DOC, separators=(",", ":")))
        lines.append("open(out + '/artifact.json', 'w').write(json.dumps(doc))")
    elif outcome == "exit-1":
        lines.append("sys.exit(1)")
    script = tmp_path / "fake_rocprof.py"
    script.write_text("\n".join(lines) + "\n")
    return script


def _configure(monkeypatch, script):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD",
        "{} {} {{output}}".format(shlex.quote(sys.executable), shlex.quote(str(script))),
    )


def test_start_collect_parses_artifact_and_cleans_up(monkeypatch, tmp_path):
    marker = tmp_path / "outdir.txt"
    _configure(monkeypatch, _fake_script(tmp_path, marker=marker))
    session = RocmSidecarSession()

    assert session.start() == ""
    assert session.active

    rows, error = session.collect(timeout_s=10)
    assert error == ""
    assert [row["kernel_name"] for row in rows] == ["gemm_kernel"]
    assert not session.active
    assert not Path(marker.read_text()).exists()


def test_start_when_disabled_returns_error(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "0")
    assert sidecar_enabled() is False

    session = RocmSidecarSession()
    assert "disabled" in session.start()
    rows, error = session.collect()
    assert rows == []
    assert "was not started" in error


def test_start_without_command_returns_error(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    assert sidecar_command() is None

    session = RocmSidecarSession()
    assert session.start() == "PROBING_TORCH_ROOFLINE_ROCPROF_CMD is not set"


def test_collect_reports_missing_artifact_files(monkeypatch, tmp_path):
    _configure(monkeypatch, _fake_script(tmp_path, outcome="no-files"))
    session = RocmSidecarSession()
    assert session.start() == ""

    rows, error = session.collect(timeout_s=10)
    assert rows == []
    assert "no JSON/CSV counter artifacts" in error


def test_collect_reports_command_failure(monkeypatch, tmp_path):
    _configure(monkeypatch, _fake_script(tmp_path, outcome="exit-1"))
    session = RocmSidecarSession()
    assert session.start() == ""

    rows, error = session.collect(timeout_s=10)
    assert rows == []
    assert "exited with 1" in error
