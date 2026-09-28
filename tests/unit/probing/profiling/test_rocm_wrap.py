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


def test_main_defaults_artifact_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(rocm_wrap.ARTIFACT_DIR_ENV, raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    def fake_wrap_command(*, pmc, output, app):
        return "rocprof -i pmc.txt -d out app"

    monkeypatch.setattr(rocm_wrap, "wrap_command", fake_wrap_command)
    code = rocm_wrap.main(["--dry-run", "--", "python", "train.py"])
    assert code == 0
    captured = capsys.readouterr()
    assert "roofline_artifacts" in captured.err
    assert "rocprof -i pmc.txt -d out app" in captured.err
    assert os.environ.get(rocm_wrap.ARTIFACT_DIR_ENV) is not None


def test_is_torchrun_command_variants():
    assert rocm_wrap.is_torchrun_command(["torchrun", "--nproc_per_node", "4"])
    assert rocm_wrap.is_torchrun_command(
        ["python", "-m", "torch.distributed.run", "train.py"]
    )
    assert rocm_wrap.is_torchrun_command(
        ["python3", "-m", "torch.distributed.run", "train.py"]
    )
    assert not rocm_wrap.is_torchrun_command(["python", "train.py"])
    assert not rocm_wrap.is_torchrun_command([])


def test_split_torchrun_command_separates_options_script_args():
    opts, script, script_args = rocm_wrap.split_torchrun_command(
        ["--nproc_per_node", "4", "train.py", "--steps", "2"]
    )
    assert opts == ["--nproc_per_node", "4"]
    assert script == "train.py"
    assert script_args == ["--steps", "2"]


def test_split_torchrun_command_equals_form_does_not_consume_next():
    opts, script, script_args = rocm_wrap.split_torchrun_command(
        ["--nproc-per-node=4", "--standalone", "train.py"]
    )
    assert opts == ["--nproc-per-node=4", "--standalone"]
    assert script == "train.py"
    assert script_args == []


def test_split_torchrun_command_requires_script():
    import pytest

    with pytest.raises(ValueError):
        rocm_wrap.split_torchrun_command(["--nproc_per_node", "4"])


def test_render_rank_wrapper_embeds_rank_and_script(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    text = rocm_wrap.render_rank_wrapper(
        script="/train/train.py",
        python_executable="/usr/bin/python3",
        pmc="/art/pmc.txt",
        artifact_dir="/art",
        launch_ts="42",
    )
    assert text.startswith("#!/bin/bash")
    assert 'RANK="${RANK:-${LOCAL_RANK:-0}}"' in text
    assert "/art/rank$RANK/42" in text
    assert "rocprofv2 -i /art/pmc.txt --plugin file -d \"$OUT\"" in text
    assert "/usr/bin/python3 -u /train/train.py \"$@\"" in text


def test_build_torchrun_launch_injects_no_python_and_wrapper(monkeypatch):
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    command, wrapper_path, wrapper_text = rocm_wrap.build_torchrun_launch(
        ["torchrun", "--nproc_per_node", "4", "train.py", "--steps", "2"],
        artifact_dir="/art",
        launch_ts="42",
        pmc="/art/pmc.txt",
        python_executable="/usr/bin/python3",
    )
    assert command == [
        "torchrun",
        "--nproc_per_node",
        "4",
        "--no_python",
        wrapper_path,
        "--steps",
        "2",
    ]
    assert wrapper_path == os.path.join("/art", rocm_wrap.RANK_WRAPPER_FILENAME)
    assert "train.py" in wrapper_text


def test_main_torchrun_dry_run_uses_per_rank_wrapper(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(rocm_wrap.ARTIFACT_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_CONFIG", raising=False)
    monkeypatch.delenv("PROBING_TORCH_ROOFLINE_ROCPROF_CMD", raising=False)

    code = rocm_wrap.main(
        ["--dry-run", "--", "torchrun", "--nproc_per_node", "4", "train.py"]
    )
    assert code == 0
    captured = capsys.readouterr().err
    assert "--no_python" in captured
    assert rocm_wrap.RANK_WRAPPER_FILENAME in captured
    assert "rank$RANK" in captured
