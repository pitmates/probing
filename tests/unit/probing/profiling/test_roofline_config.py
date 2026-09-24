"""Unit tests for the consolidated roofline config loader."""

from __future__ import annotations

import json

from probing.profiling.torch_profiler.config import (
    CONFIG_ENV,
    load_roofline_config,
)


def test_inline_json_config(monkeypatch):
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps(
            {
                "backend": "rocm",
                "rocm_enabled": True,
                "rocprof_cmd": "rocprofv2 --output {output}",
                "metrics": ["A", "B", "A"],
                "peaks": {"backend": "rocm", "peaks": {"fp16_tensor_dense": {}}},
                "flop_weights": {"A": 2, "B": 1},
            }
        ),
    )
    config = load_roofline_config()
    assert config.rocm_enabled is True
    assert config.backend == "rocm"
    assert config.rocprof_cmd == "rocprofv2 --output {output}"
    assert config.metrics == ("A", "B")
    assert config.peaks == {"backend": "rocm", "peaks": {"fp16_tensor_dense": {}}}
    assert config.flop_weights == {"A": 2, "B": 1}


def test_json_file_config(monkeypatch, tmp_path):
    path = tmp_path / "roofline.json"
    path.write_text(
        json.dumps({"backend": "cuda", "metrics": ["m0"], "rocm_enabled": False}),
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV, str(path))
    config = load_roofline_config()
    assert config.backend == "cuda"
    assert config.rocm_enabled is False
    assert config.metrics == ("m0",)


def test_missing_or_invalid_config_falls_back_to_defaults(monkeypatch):
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert load_roofline_config().backend == "auto"

    monkeypatch.setenv(CONFIG_ENV, "not json")
    config = load_roofline_config()
    assert config.backend == "auto"
    assert config.rocm_enabled is True


def test_nan_and_infinity_weights_are_dropped(monkeypatch):
    monkeypatch.setenv(
        CONFIG_ENV,
        '{"flop_weights": {"A": 2, "B": NaN, "C": Infinity}}',
    )
    config = load_roofline_config()
    assert config.flop_weights == {"A": 2}
