"""Unit tests for ProfilerController finalize and status."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from unittest.mock import MagicMock

import pytest

from probing.profiling.torch_profiler.controller import ProfilerController
from probing.profiling.torch_profiler.session_store import get_session_store


@dataclass
class _FakeEvent:
    key: str
    self_cuda_time_total: int = 0
    self_cpu_time_total: int = 0
    cuda_time_total: int = 0
    cpu_time_total: int = 0
    count: int = 1


def _mock_profiler(events: list[_FakeEvent]) -> MagicMock:
    profiler = MagicMock()
    profiler.key_averages.return_value = events
    profiler.__exit__ = MagicMock(return_value=None)
    return profiler


def _install_fake_torch(monkeypatch) -> MagicMock:
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    optimizer_module = types.ModuleType("torch.optim.optimizer")
    optimizer_module.register_optimizer_step_post_hook = MagicMock(return_value=1)
    optimizer_package = types.ModuleType("torch.optim")
    optimizer_package.optimizer = optimizer_module
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.optim", optimizer_package)
    monkeypatch.setitem(sys.modules, "torch.optim.optimizer", optimizer_module)
    return fake_torch


def test_finalize_materializes_sql_rows(monkeypatch):
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.row_fields",
        lambda _snap=None: {
            "local_step": 3,
            "global_step": 3,
            "rank": 0,
            "world_size": 1,
        },
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.current_role",
        lambda: "",
    )
    ctrl = ProfilerController()
    ctrl._profiler = _mock_profiler([_FakeEvent("aten::mm", self_cpu_time_total=42)])
    ctrl._running = True
    ctrl._started_at_us = 0
    ctrl._trigger = "unit"
    ctrl._step_count = 1

    capture_id = ctrl._finalize_capture(status="completed")
    assert capture_id is not None
    store = get_session_store()
    assert len(store.captures()) == 1
    assert store.captures()[0].capture_id == capture_id
    assert len(store.hotspots()) == 1
    assert store.captures()[0].status == "completed"
    assert ctrl.is_running is False


def test_finalize_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.row_fields",
        lambda _snap=None: {
            "local_step": 1,
            "global_step": 1,
            "rank": 0,
            "world_size": 1,
        },
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.current_role",
        lambda: "",
    )
    ctrl = ProfilerController()
    ctrl._profiler = _mock_profiler([_FakeEvent("aten::mm", self_cpu_time_total=10)])
    ctrl._running = True
    ctrl._started_at_us = 0
    ctrl._trigger = "unit"
    ctrl._step_count = 1

    first = ctrl._finalize_capture(status="completed")
    second = ctrl._finalize_capture(status="completed")
    assert second == first
    assert len(get_session_store().captures()) == 1


def test_stop_when_idle_returns_latest_capture(monkeypatch):
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.row_fields",
        lambda _snap=None: {
            "local_step": 1,
            "global_step": 1,
            "rank": 0,
            "world_size": 1,
        },
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.adaptor.current_role",
        lambda: "",
    )
    ctrl = ProfilerController()
    ctrl._profiler = _mock_profiler([_FakeEvent("aten::mm", self_cpu_time_total=5)])
    ctrl._running = True
    ctrl._started_at_us = 0
    ctrl._trigger = "unit"
    ctrl._step_count = 1
    expected = ctrl._finalize_capture(status="completed")

    assert ctrl.stop() == expected
    assert ctrl.stop() == expected


def test_status_reflects_controller(monkeypatch):
    mock_profile = MagicMock()
    fake_torch = _install_fake_torch(monkeypatch)
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.HAS_TORCH",
        True,
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )
    fake_torch.profiler.profile.return_value = mock_profile
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )

    ctrl = ProfilerController()
    ctrl.start(steps=2, trigger="test", analysis="roofline")
    status = ctrl.status()
    assert status["running"] is True
    assert status["steps_target"] == 2
    assert status["trigger"] == "test"
    assert status["analysis"] == "roofline"


def test_double_start_raises(monkeypatch):
    mock_profile = MagicMock()
    fake_torch = _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.HAS_TORCH",
        True,
    )
    fake_torch.profiler.profile.return_value = mock_profile
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )

    ctrl = ProfilerController()
    ctrl.start(steps=1, trigger="a")
    with pytest.raises(RuntimeError, match="already running"):
        ctrl.start(steps=1, trigger="b")


def test_start_failure_does_not_leave_controller_running(monkeypatch):
    fake_torch = _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.HAS_TORCH",
        True,
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )
    fake_torch.profiler.profile.side_effect = RuntimeError("construction failed")

    ctrl = ProfilerController()
    with pytest.raises(RuntimeError, match="construction failed"):
        ctrl.start(steps=1, trigger="test")
    assert ctrl.is_running is False
    assert ctrl._profiler is None


def test_roofline_experimental_config_uses_pytorch_named_arguments(monkeypatch):
    fake_torch = _install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.HAS_TORCH",
        True,
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )
    monkeypatch.setenv(
        "PROBING_TORCH_ROOFLINE_METRICS",
        "dram__bytes_read.sum,dram__bytes_write.sum",
    )
    config = fake_torch.profiler._ExperimentalConfig
    config.side_effect = TypeError("unsupported arguments")
    fake_torch.profiler.profile.return_value = MagicMock()

    ctrl = ProfilerController()
    with pytest.raises(RuntimeError, match="compatible PyTorch/CUPTI runtime"):
        ctrl.start(steps=1, trigger="test", analysis="roofline")
    config.assert_called_once_with(
        profiler_metrics=["dram__bytes_read.sum", "dram__bytes_write.sum"],
        profiler_measure_per_kernel=True,
    )
    assert ctrl.is_running is False
    assert ctrl._profiler is None


def test_roofline_capability_probe_failure_is_diagnostic(monkeypatch):
    fake_torch = _install_fake_torch(monkeypatch)
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.HAS_TORCH",
        True,
    )
    monkeypatch.setattr(
        "probing.profiling.torch_profiler.controller.torch",
        fake_torch,
    )
    probe = MagicMock()
    probe.__enter__.side_effect = RuntimeError("CUPTI range unsupported")
    fake_torch.profiler.profile.return_value = probe

    ctrl = ProfilerController()
    with pytest.raises(RuntimeError, match="roofline capability check failed"):
        ctrl.start(steps=1, trigger="test", analysis="roofline")
    assert ctrl.is_running is False
    assert ctrl._profiler is None
