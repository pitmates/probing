"""Unit tests for ROCm wrapper rendering and offline artifact import."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from probing.profiling.torch_profiler.rocm_runner import (
    DEFAULT_ROCM_ROCPROF_CMD,
    artifact_finalized,
    artifact_root,
    import_artifact_rows,
    keep_artifacts,
    sidecar_command,
    sidecar_enabled,
    wrap_command,
)

_DOC = {
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


def _write_artifact(root: Path, name: str = "artifact.json") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(_DOC), encoding="utf-8")
    return path


def test_sidecar_command_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    assert sidecar_command() == DEFAULT_ROCM_ROCPROF_CMD


def test_sidecar_enabled_respects_legacy_env(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "0")
    assert sidecar_enabled() is False
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ROCM_PROFILE", "1")
    assert sidecar_enabled() is True


def test_wrap_command_quotes_replacements(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD",
        "python {app} -o {output} -i {pmc}",
    )
    rendered = wrap_command(
        pmc="/tmp/a b.txt",
        output="/tmp/o dir",
        app="bench.py --steps 2",
    )
    assert shlex.split(rendered) == [
        "python",
        "bench.py",
        "--steps",
        "2",
        "-o",
        "/tmp/o dir",
        "-i",
        "/tmp/a b.txt",
    ]


def test_wrap_command_leaves_unsupplied_placeholders(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    rendered = wrap_command(pmc="pmc.txt")
    assert rendered is not None
    assert "pmc.txt" in rendered
    assert "{output}" in rendered
    assert "{app}" in rendered


def test_import_artifact_rows_parses_and_cleanups(tmp_path):
    launch = tmp_path / "rank0" / "launch-1"
    _write_artifact(launch)

    rows, error = import_artifact_rows(str(tmp_path), rank=0, finalized=True)
    assert error == ""
    assert [row["kernel_name"] for row in rows] == ["gemm_kernel"]
    assert not launch.exists()
    assert (tmp_path / "rank0").exists()


def test_import_artifact_rows_keeps_when_requested(tmp_path):
    launch = tmp_path / "rank0" / "launch-1"
    path = _write_artifact(launch)

    rows, error = import_artifact_rows(str(tmp_path), rank=0, keep=True, finalized=True)
    assert error == ""
    assert rows
    assert path.exists()


def test_import_artifact_rows_reports_missing_directory(tmp_path):
    rows, error = import_artifact_rows(str(tmp_path), rank=0, finalized=True)
    assert rows == []
    assert "no rocm counter artifact directory" in error


def test_import_artifact_rows_refuses_unfinalized(tmp_path):
    launch = tmp_path / "rank0" / "launch-1"
    _write_artifact(launch)

    rows, error = import_artifact_rows(str(tmp_path), rank=0)
    assert rows == []
    assert "not finalized" in error
    assert launch.exists()


def test_artifact_finalized_respects_env(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_FINALIZED", "1")
    assert artifact_finalized() is True
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_FINALIZED", "0")
    assert artifact_finalized() is False


def test_import_artifact_rows_skips_stray_component(tmp_path):
    launch = tmp_path / "rank0" / "launch-1"
    launch.mkdir(parents=True)
    _write_artifact(launch, name="results.csv")
    # A newer stray JSON component must be skipped in favor of the counter CSV.
    (launch / "metadata.json").write_text('{"component": true}', encoding="utf-8")

    rows, error = import_artifact_rows(str(tmp_path), rank=0, finalized=True)
    assert error == ""
    assert [row["kernel_name"] for row in rows] == ["gemm_kernel"]


def test_artifact_root_prefers_env(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_ARTIFACT_DIR", "/artifacts")
    assert artifact_root() == "/artifacts"


def test_keep_artifacts_env(monkeypatch):
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_KEEP_ARTIFACTS", "1")
    assert keep_artifacts() is True
    monkeypatch.setenv("PROBING_TORCH_ROOFLINE_KEEP_ARTIFACTS", "0")
    assert keep_artifacts() is False
