"""Unit tests for the single-command ROCm roofline launcher.

These tests exercise rendering and default-file generation without invoking
``rocprof`` or a real DCU.
"""

from __future__ import annotations

import os

from probing.profiling.torch_profiler import rocm_wrap


def test_default_pmc_text_lists_metrics():
    text = rocm_wrap.default_pmc_text(["A", "B"])
    assert text == "pmc: A B\n"


def test_output_dir_uses_rank_and_launch_ts():
    assert rocm_wrap.output_dir("/art", 3, "123") == os.path.join(
        "/art", "rank3", "123"
    )


def test_build_wrapped_command_generates_pmc_and_outdir(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def fake_wrap_command(*, pmc, output, app):
        captured["pmc"] = pmc
        captured["output"] = output
        captured["app"] = app
        return "RENDERED"

    monkeypatch.setattr(rocm_wrap, "wrap_command", fake_wrap_command)
    rendered = rocm_wrap.build_wrapped_command(
        str(tmp_path),
        0,
        "42",
        ["python", "train.py", "--steps", "2"],
        metrics=["M1", "M2"],
    )
    assert rendered == "RENDERED"
    assert captured["pmc"].endswith(rocm_wrap.DEFAULT_PMC_FILENAME)
    assert captured["output"] == str(tmp_path / "rank0" / "42")
    assert captured["app"] == "python train.py --steps 2"

    pmc_text = open(captured["pmc"], encoding="utf-8").read()
    assert pmc_text == "pmc: M1 M2\n"
    assert (tmp_path / "rank0" / "42").is_dir()


def test_build_wrapped_command_requires_app(monkeypatch, tmp_path):
    monkeypatch.setattr(rocm_wrap, "wrap_command", lambda **kwargs: "RENDERED")
    import pytest

    with pytest.raises(ValueError):
        rocm_wrap.build_wrapped_command(str(tmp_path), 0, "42", [])


def test_main_dry_run_renders_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(
        rocm_wrap.ARTIFACT_DIR_ENV, str(tmp_path)
    )
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)

    def fake_wrap_command(*, pmc, output, app):
        return "rocprof -i pmc.txt -d out app"

    monkeypatch.setattr(rocm_wrap, "wrap_command", fake_wrap_command)
    code = rocm_wrap.main(["--dry-run", "--", "python", "train.py"])
    assert code == 0
    captured = capsys.readouterr()
    assert "rocprof -i pmc.txt -d out app" in captured.err


def test_main_missing_artifact_dir(monkeypatch, capsys):
    monkeypatch.delenv(rocm_wrap.ARTIFACT_DIR_ENV, raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    code = rocm_wrap.main(["--dry-run", "--", "python", "train.py"])
    assert code == 2
    assert "artifact directory is not configured" in capsys.readouterr().err
