"""On-demand ``torch.profiler`` / Kineto capture and virtual SQL tables.

This package is independent from :mod:`probing.profiling.torch_probe`: it owns
short capture sessions and bounded in-process results, not long-running module
hooks or memtables. Starting either path does not configure the other.

The heavy names below are re-exported lazily so lightweight helpers such as
``rocm_wrap`` / ``rocm_runner`` / ``rocm_metrics`` (which only need stdlib and
the roofline config) can be imported without pulling ``torch``, Kineto, or the
``probing.tracing`` / ``register_table_docs`` runtime.
"""

import importlib
from typing import Any

__all__ = [
    "BackendInfo",
    "CapabilityResult",
    "CudaRooflineBackend",
    "RocmRooflineBackend",
    "RooflineBackend",
    "ProfilerController",
    "SessionStore",
    "create_roofline_backend",
    "get_controller",
    "get_session_store",
    "profile_counter_rows",
    "profile_roofline_rows",
    "profiler_status",
]

_LAZY_ATTRS = {
    "BackendInfo": ("backends", "BackendInfo"),
    "CapabilityResult": ("backends", "CapabilityResult"),
    "CudaRooflineBackend": ("backends", "CudaRooflineBackend"),
    "RocmRooflineBackend": ("backends", "RocmRooflineBackend"),
    "RooflineBackend": ("backends", "RooflineBackend"),
    "create_roofline_backend": ("backends", "create_roofline_backend"),
    "ProfilerController": ("controller", "ProfilerController"),
    "get_controller": ("controller", "get_controller"),
    "profiler_status": ("controller", "profiler_status"),
    "profile_counter_rows": ("sql", "profile_counter_rows"),
    "profile_roofline_rows": ("sql", "profile_roofline_rows"),
    "SessionStore": ("session_store", "SessionStore"),
    "get_session_store": ("session_store", "get_session_store"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value
    return value
