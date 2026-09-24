"""Offline dry-run tests for the ROCm roofline validation spike.

These exercise ``rocm_e2e_spike`` without a real DCU or ``rocprof``: wrapper
template rendering and the import+compile link are validated against the
checked-in counter fixtures. The actual ``rocprof`` subprocess and a live GPU
are exercised separately by the opt-in ``slow`` regression test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from probing.profiling.torch_profiler import rocm_e2e_spike as spike
from probing.profiling.torch_profiler import rocm_runner
from probing.profiling.torch_profiler.backends import (
    BackendInfo,
    RocmRooflineBackend,
)

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "rocm"


def test_render_wrap_cmd_reports_template_and_command(monkeypatch):
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_ROCPROF_CMD",
        "rocprof -i {pmc} --timestamp on -d {output} {app}",
    )
    report = spike.render_wrap_cmd(
        pmc="/tmp/a b.txt",
        out_dir="/tmp/o dir",
        app="python bench.py --steps 2",
    )
    assert report["enabled"] is True
    assert report["template"].startswith("rocprof -i {pmc}")
    assert report["artifact_root_default"] == ""
    assert "error" not in report
    assert report["wrap_command"] == (
        "rocprof -i '/tmp/a b.txt' --timestamp on -d '/tmp/o dir' "
        "'python bench.py --steps 2'"
    )


def test_render_wrap_cmd_leaves_missing_placeholders(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    report = spike.render_wrap_cmd(pmc="pmc.txt", out_dir="", app="")
    assert report["wrap_command"] == "rocprof -i pmc.txt --timestamp on -d {output} {app}"


def test_render_wrap_cmd_reports_missing_template(monkeypatch):
    monkeypatch.setattr(rocm_runner, "sidecar_command", lambda: None)
    report = spike.render_wrap_cmd("", "", "")
    assert "no rocprof wrapper command template configured" in report["error"]


def test_run_import_dry_run_compiles_fixture(monkeypatch, tmp_path):
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    backend = RocmRooflineBackend(info)
    monkeypatch.setattr(spike, "_backend", lambda: (backend, object(), info.__dict__))

    payload = (FIXTURES / "counter_artifact_v1.json").read_text(encoding="utf-8")
    rank_dir = tmp_path / "rank0" / "launch-1"
    rank_dir.mkdir(parents=True)
    (rank_dir / "artifact.json").write_text(payload, encoding="utf-8")

    report = spike.run_import(str(tmp_path), rank=0)
    assert report["import_error"] == ""
    assert report["parsed_rows"] == 3
    assert report["compile"]["quality"] == "partial"
    assert report["compile"]["counter_events"] == 3
    assert report["compile"]["missing_metrics"] == []
    assert len(report["counters"]) == 3
    assert all(item["flops"] is None for item in report["counters"])


def test_run_import_reports_missing_artifact(monkeypatch, tmp_path):
    info = BackendInfo("amd", "BW", "gfx936", "rocm")
    backend = RocmRooflineBackend(info)
    monkeypatch.setattr(spike, "_backend", lambda: (backend, object(), info.__dict__))

    report = spike.run_import(str(tmp_path), rank=0)
    assert "no rocm counter artifact directory" in report["import_error"]
    assert report["parsed_rows"] == 0


def test_list_counters_parses_found_and_missing(monkeypatch):
    monkeypatch.setattr(spike, "_required_metrics", lambda: ("A", "B"))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=a[0] if a else "",
            returncode=0,
            stdout="A metric present\n",
            stderr="",
        ),
    )
    report = spike.list_counters("rocprofv2 --list-counters")
    assert report["ok"] is True
    assert report["found"] == ["A"]
    assert report["missing"] == ["B"]